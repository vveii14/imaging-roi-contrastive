"""
Unified Optuna hyperparameter tuning entry for k-fold experiments.

Design:
- Single entry point for brain_structure / ADHD with image_only, roi_only, and fusion modes.
- Search space: common (lr, weight_decay, dropout, batch_size, epochs); ROI/NeuroGraph
  (hidden_channels, hidden_mlp, num_layers, gnn_name, edge_top_p); image encoder;
  fusion (concat / contrastive / cross_attention with temperature and weight for contrastive).
- Reuses from train_kfold: load_config, run_one_fold, compute_test_metrics, collate.
- This module defines the Optuna search space (conditional on dataset/mode/roi_backend) and
  runs n-fold validation via run_one_fold, returning a single objective (e.g. mean AUC).

Example:
    CUDA_VISIBLE_DEVICES=1 python train_kfold_optuna_unified_v2.py \\
        --dataset adhd --mode roi_only --config configs/adhd_roi_only.yaml --n_trials 50
    CUDA_VISIBLE_DEVICES=0 python train_kfold_optuna_unified_v2.py \\
        --dataset brain_structure --mode image_only \\
        --config configs/brain_structure_image_only.yaml --n_trials 30
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from train_kfold import load_config, run_one_fold  # type: ignore
from data.adhd import get_adhd_meta  # type: ignore

# Store per-trial metrics (mean ± std) so we can save to JSON after each trial
_trial_metrics_store: Dict[int, Dict[str, Dict[str, float]]] = {}

# Global variables for V2 enhancements
_LOG_DIR: Optional[Path] = None
_SUMMARY_CSV_PATH: Optional[Path] = None
_TRIAL_START_TIME: Dict[int, float] = {}

# NeuroGraph native path: roi_only ADHD uses NeuroGraph-main/train.py
_PROJECT_ROOT = Path(__file__).resolve().parent
_NEUROGRAPH_ROOT = _PROJECT_ROOT / "NeuroGraph-main"


def _save_trial_json(trial_number: int, params: Dict, means: Dict, stds: Dict, args) -> None:
    """Save individual trial results to JSON file immediately after completion."""
    if _LOG_DIR is None:
        return

    trial_json_dir = _LOG_DIR / "trial_jsons"
    trial_json_dir.mkdir(parents=True, exist_ok=True)

    trial_json_path = trial_json_dir / f"trial_{trial_number:03d}.json"
    data = {
        "trial_number": trial_number,
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "mode": args.mode,
        "fusion": getattr(args, "fusion", None),
        "params": params,
        "metrics": {
            "means": means,
            "stds": stds,
        },
        "duration_seconds": time.time() - _TRIAL_START_TIME.get(trial_number, time.time()),
    }

    with open(trial_json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Trial {trial_number}] Saved results to {trial_json_path}", flush=True)


def _update_summary_csv(trial_number: int, params: Dict, means: Dict, stds: Dict, objective_value: float) -> None:
    """Append trial results to summary CSV file for easy tracking."""
    if _SUMMARY_CSV_PATH is None:
        return

    # Prepare row data
    row = {
        "trial": trial_number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "objective_auc": f"{objective_value:.2f}",
        "acc_mean": f"{means.get('accuracy', 0):.2f}",
        "acc_std": f"{stds.get('accuracy', 0):.2f}",
        "auc_mean": f"{means.get('auc', 0):.2f}",
        "auc_std": f"{stds.get('auc', 0):.2f}",
        "sens_mean": f"{means.get('sensitivity', 0):.2f}",
        "sens_std": f"{stds.get('sensitivity', 0):.2f}",
        "spec_mean": f"{means.get('specificity', 0):.2f}",
        "spec_std": f"{stds.get('specificity', 0):.2f}",
        "f1_mean": f"{means.get('f1', 0):.2f}",
        "f1_std": f"{stds.get('f1', 0):.2f}",
        "duration_min": f"{(time.time() - _TRIAL_START_TIME.get(trial_number, time.time())) / 60:.1f}",
    }

    # Add key hyperparameters
    for key in ["lr", "weight_decay", "dropout", "batch_size", "epochs"]:
        if key in params:
            row[key] = params[key]

    # Write header if file doesn't exist
    file_exists = _SUMMARY_CSV_PATH.exists()
    with open(_SUMMARY_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[Trial {trial_number}] Updated summary CSV: {_SUMMARY_CSV_PATH}", flush=True)


def _create_trial_logger(trial_number: int) -> callable:
    """Create a logger function that writes to both console and trial-specific log file."""
    if _LOG_DIR is None:
        return lambda s: print(s, flush=True)

    trial_log_dir = _LOG_DIR / f"trial_{trial_number:03d}"
    trial_log_dir.mkdir(parents=True, exist_ok=True)
    trial_log_file = trial_log_dir / "detailed.log"

    def log_fn(s: str) -> None:
        print(s, flush=True)
        with open(trial_log_file, "a") as f:
            f.write(s + "\n")

    return log_fn


def _prepare_base_cfg_and_data(args) -> tuple[Dict[str, Any], np.ndarray, np.ndarray, str, np.ndarray]:
    """
    Load base config and data indices/labels, aligned with train_kfold.
    Returns: (cfg_base, indices, labels, data_path, image_shape).
    """
    cfg = load_config(args.config)

    if not cfg.get("data_root") or cfg.get("data_root") == ".":
        cfg["data_root"] = "."

    cfg["seed"] = args.seed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    image_shape = np.array(cfg.get("image_shape", [96, 96, 96]), dtype=int)

    # mode -> which branches
    if args.mode == "image_only":
        cfg["use_image_branch"] = True
        cfg["use_roi_branch"] = False
    elif args.mode == "roi_only":
        cfg["use_image_branch"] = False
        cfg["use_roi_branch"] = True
        cfg.setdefault("roi_encoder", "neurograph")  # config may set chen2019
    else:  # fusion
        cfg["use_image_branch"] = True
        cfg["use_roi_branch"] = True
        cfg["roi_encoder"] = cfg.get("roi_encoder", "neurograph")
        if getattr(args, "fusion", None) is not None:
            cfg["fusion"] = args.fusion

    # dataset -> data_path + labels
    if args.dataset == "adhd":
        data_path = cfg.get("adhd_data_dir", str(Path(cfg["data_root"]) / "adhd"))
        use_test_set_only = cfg.get("adhd_use_test_set_only", False) or getattr(args, "adhd_use_test_set_only", False)
        cfg["adhd_use_test_set_only"] = use_test_set_only
        indices, labels = get_adhd_meta(data_path, use_test_set_only=use_test_set_only)
        indices = np.array(indices)
        labels = np.asarray(labels)
    else:
        # brain_structure: pool train+validation+test, then (if balance_classes) all AD + same number CN
        data_path = cfg.get(
            "brain_structure_cache_dir",
            str(Path(cfg["data_root"]) / "brain_structure_cache"),
        )
        data_path = Path(data_path)
        full_meta = []
        for split in ("train", "validation", "test"):
            meta_path = data_path / split / "meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                split_meta = json.load(f)
            for i, m in enumerate(split_meta):
                full_meta.append({
                    "split": split,
                    "roi_idx": i,
                    "nii_filepath": m["nii_filepath"],
                    "label": m["label"],
                })
        if not full_meta:
            raise FileNotFoundError(f"No meta under {data_path}. Run preprocess first.")
        labels = np.array([m["label"] for m in full_meta])
        num_classes = int(cfg.get("num_classes", 2))
        if getattr(args, "balance_classes", False):
            idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
            n_min = int(min(len(idx_by_class[c]) for c in range(num_classes)))
            rng = np.random.default_rng(cfg["seed"])
            balanced_pos = np.concatenate([
                rng.choice(idx_by_class[c], size=n_min, replace=False) for c in range(num_classes)
            ]).astype(int)
            rng.shuffle(balanced_pos)
            balanced_meta = [full_meta[i] for i in balanced_pos]
            cfg["_brain_structure_meta_list"] = balanced_meta
            indices = np.arange(len(balanced_meta))
            labels = labels[balanced_pos]
        else:
            cfg["_brain_structure_meta_list"] = full_meta
            indices = np.arange(len(full_meta))

    return cfg, indices, labels, data_path, image_shape


def _apply_common_hparams(trial: optuna.Trial, cfg: Dict[str, Any], args) -> None:
    """Define shared hyperparameter search space for all models."""
    cfg["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg["dropout"] = trial.suggest_float("dropout", 0.1, 0.6)
    # Config may set optuna_batch_size_range or optuna_batch_size_choices.
    # batch_size must be >= 2 (BatchNorm requires more than one sample in training).
    bs_range = cfg.get("optuna_batch_size_range")
    bs_choices = cfg.get("optuna_batch_size_choices")
    if isinstance(bs_range, (list, tuple)) and len(bs_range) >= 2:
        bs_min = max(2, int(bs_range[0]))
        bs_max = int(bs_range[1])
        bs_step = int(bs_range[2]) if len(bs_range) >= 3 else 1
        cfg["batch_size"] = trial.suggest_int("batch_size", bs_min, bs_max, step=bs_step)
        cfg["epochs"] = trial.suggest_int("epochs", 30, 100, step=10)
        return
    if isinstance(bs_choices, (list, tuple)) and len(bs_choices) > 0:
        choices = [x for x in [int(c) for c in bs_choices] if x >= 2]
        if not choices:
            choices = [2]
        cfg["batch_size"] = trial.suggest_categorical("batch_size", choices)
        cfg["epochs"] = trial.suggest_int("epochs", 30, 100, step=10)
        return
    img_enc = cfg.get("image_encoder", "")
    if args.mode in ("image_only", "fusion") and img_enc == "3dsctf":
        cfg["batch_size"] = trial.suggest_categorical("batch_size", [2, 4])
        # 3D-SCTF with 128^3 images may need more epochs to converge
        cfg["epochs"] = trial.suggest_int("epochs", 50, 200, step=25)
    else:
        cfg["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])
        cfg["epochs"] = trial.suggest_int("epochs", 50, 150, step=25)


def _apply_roi_chen2019_hparams(trial: optuna.Trial, cfg: Dict[str, Any]) -> None:
    """Hyperparameter search for ROI branch when using Chen2019 DNN (hidden, dropout, etc.)."""
    roi_kw = dict(cfg.get("roi_encoder_kwargs") or {})
    roi_kw["hidden"] = trial.suggest_categorical("roi_hidden", [256, 512, 1024])
    cfg["roi_encoder_kwargs"] = roi_kw


def _apply_roi_neurograph_hparams(trial: optuna.Trial, cfg: Dict[str, Any]) -> None:
    """Hyperparameter search space for ROI branch with NeuroGraph ResidualGNNs."""
    roi_kw = dict(cfg.get("roi_encoder_kwargs") or {})
    roi_kw["hidden_channels"] = trial.suggest_categorical("roi_hidden_channels", [16, 32, 64])
    roi_kw["hidden_mlp"] = trial.suggest_categorical("roi_hidden_mlp", [32, 64, 128])
    roi_kw["num_layers"] = trial.suggest_int("roi_num_layers", 2, 5)
    roi_kw["gnn_name"] = trial.suggest_categorical(
        "roi_gnn_name", ["GCNConv", "GINConv", "SAGEConv", "GATConv", "GraphConv"]
    )
    roi_kw["edge_top_p"] = trial.suggest_float("roi_edge_top_p", 0.02, 0.2)
    cfg["roi_encoder_kwargs"] = roi_kw


def _apply_image_hparams(trial: optuna.Trial, cfg: Dict[str, Any]) -> None:
    """Image branch hyperparameters: feature dim and encoder-specific structure/regularization."""
    cfg["image_feat_dim"] = trial.suggest_categorical("image_feat_dim", [128, 256])
    enc = cfg.get("image_encoder", "")
    kw = dict(cfg.get("image_encoder_kwargs") or {})

    if enc == "vit3d":
        # ViT3D: with pretrained only tune regularization; without pretrained tune full structure
        has_pretrained = bool(kw.get("pretrained_checkpoint", ""))
        kw["drop_path_rate"] = trial.suggest_float("img_drop_path_rate", 0.0, 0.3)
        kw["qkv_bias"] = trial.suggest_categorical("img_qkv_bias", [True, False])
        if not has_pretrained:
            kw["patch_size"] = trial.suggest_categorical("img_patch_size", [8, 16, 32])
            kw["embed_dim"] = trial.suggest_categorical("img_embed_dim", [384, 768])
            kw["depth"] = trial.suggest_int("img_depth", 6, 12, step=3)
            kw["n_heads"] = trial.suggest_categorical("img_n_heads", [4, 6, 8, 12])
            kw["mlp_ratio"] = trial.suggest_float("img_mlp_ratio", 2.0, 4.0)
        cfg["image_encoder_kwargs"] = kw
    elif enc == "rae_vit_ad":
        # RAE-ViT: embed_dim / num_heads (4,8 divide 384/512/768) / input resolution
        kw["embed_dim"] = trial.suggest_categorical("img_embed_dim", [384, 512, 768])
        kw["num_heads"] = trial.suggest_categorical("img_num_heads", [4, 8])
        kw["img_size"] = trial.suggest_categorical("img_size", [96, 128])
        cfg["image_encoder_kwargs"] = kw
    elif enc == "3dsctf":
        # 3DSCTF: patch / dim / depth / num_heads (4,8 divide 96/128/192) / MLP ratio / regularization
        kw["patch_size"] = trial.suggest_categorical("img_patch_size", [16, 32])
        kw["embed_dim"] = trial.suggest_categorical("img_embed_dim", [96, 128, 192])
        kw["depth"] = trial.suggest_int("img_depth", 4, 8, step=2)
        kw["num_heads"] = trial.suggest_categorical("img_num_heads", [4, 8])
        kw["mlp_ratio"] = trial.suggest_float("img_mlp_ratio", 2.0, 4.0)
        kw["qkv_bias"] = trial.suggest_categorical("img_qkv_bias", [True, False])
        kw["drop_path_rate"] = trial.suggest_float("img_drop_path_rate", 0.0, 0.3)
        cfg["image_encoder_kwargs"] = kw
    else:
        pass


def _apply_fusion_hparams(trial: optuna.Trial, cfg: Dict[str, Any]) -> None:
    """Fusion-related hyperparameter search.
    If fusion is already fixed in cfg (from CLI or config), do not resample type in trial.
    """
    fusion = cfg.get("fusion")
    if fusion is None:
        fusion = trial.suggest_categorical("fusion", ["concat", "contrastive", "cross_attention"])
        cfg["fusion"] = fusion
    if fusion == "contrastive":
        cfg["contrastive_temperature"] = trial.suggest_float("temp", 0.03, 0.2)
        cfg["contrastive_weight"] = trial.suggest_float("cont_weight", 0.05, 0.5)


def _run_neurograph_roi_only_native(trial: optuna.Trial, args) -> float:
    """
    ADHD ROI-only via NeuroGraph (NeuroGraph-main/train.py).
    Uses predefined 5-fold (fold_assignments.csv), same .pkl graph loading and ResidualGNNs.
    Objective: AUC (aligned with standalone script).
    """
    if not _NEUROGRAPH_ROOT.is_dir():
        raise FileNotFoundError(f"NeuroGraph (NeuroGraph-main) not found at {_NEUROGRAPH_ROOT}")

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    # batch_size 4 or 8 to avoid OOM with ~465 samples per fold
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    epochs = trial.suggest_int("epochs", 30, 300, step=10)
    hidden = trial.suggest_categorical("hidden", [16, 32, 64])
    hidden_mlp = trial.suggest_categorical("hidden_mlp", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    model_name = trial.suggest_categorical(
        "model", ["GCNConv", "GINConv", "SAGEConv", "GATConv", "GraphConv"]
    )
    edge_percent_to_keep = trial.suggest_float("edge_percent_to_keep", 0.02, 0.2)

    print(
        f"[Trial {trial.number}] dataset=adhd mode=roi_only (NeuroGraph native) "
        f"lr={lr:.2e} wd={weight_decay:.2e} bs={batch_size} epochs={epochs} "
        f"hidden={hidden} hidden_mlp={hidden_mlp} num_layers={num_layers} "
        f"model={model_name} edge_keep={edge_percent_to_keep:.3f}",
        flush=True,
    )

    # Build NeuroGraph Args (aligned with train.py argsDictTune_a)
    dataset_dir = str(_NEUROGRAPH_ROOT / "data")
    folds_dir = str(_PROJECT_ROOT)
    param_dict = {
        "dataset": "ADHD",
        "dataset_dir": dataset_dir,
        "timeseries_dir": str(_NEUROGRAPH_ROOT / "data" / "fMRIROItimeseries"),
        "folds_dir": folds_dir,
        "edge_dir_prefix": "pearson_correlation",
        "atlas": "AAL116",
        "model": model_name,
        "num_classes": 2,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "hidden_mlp": hidden_mlp,
        "hidden": hidden,
        "num_layers": num_layers,
        "runs": 1,
        "lr": lr,
        "epochs": epochs,
        "edge_percent_to_keep": edge_percent_to_keep,
        "n_splits": 5,
        "seed": args.seed,
        "graph_type": "static",
        "early_stop_patience": 60,
        "early_stop_min_delta": 1e-4,
        "tune_name": f"unified_trial{trial.number}",
        "rank": 0,
    }

    # Import NeuroGraph-compatible module and run bench_from_args as in train.py
    prev_path = list(sys.path)
    try:
        sys.path.insert(0, str(_NEUROGRAPH_ROOT))
        from utils import Args, fix_seed  # type: ignore
        from train import bench_from_args  # type: ignore
    finally:
        sys.path[:] = prev_path

    bn_args = Args(param_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bn_args.device = str(device)
    bn_args.gpu_id = device.index if device.type == "cuda" else None
    fix_seed(bn_args.seed)

    avg_metrics, std_metrics = bench_from_args(bn_args, verbose=False, use_predefined_folds=True)

    # Metrics in [0,1]; report as percentage with std for reproducibility
    acc = avg_metrics["accuracy"] * 100
    auc = avg_metrics["auroc"] * 100
    sens = avg_metrics["sensitivity"] * 100
    spec = avg_metrics["specificity"] * 100
    f1 = avg_metrics["f1_score"] * 100
    s_acc = std_metrics["accuracy"] * 100
    s_auc = std_metrics["auroc"] * 100
    s_sens = std_metrics["sensitivity"] * 100
    s_spec = std_metrics["specificity"] * 100
    s_f1 = std_metrics["f1_score"] * 100
    print(
        f"[Trial {trial.number}] Acc {acc:.2f}±{s_acc:.2f}%  AUC {auc:.2f}±{s_auc:.2f}%  Sens {sens:.2f}±{s_sens:.2f}%  Spec {spec:.2f}±{s_spec:.2f}%  F1 {f1:.2f}±{s_f1:.2f}%",
        flush=True,
    )
    _trial_metrics_store[trial.number] = {
        "means": {"accuracy": acc, "auc": auc, "sensitivity": sens, "specificity": spec, "f1": f1},
        "stds": {"accuracy": s_acc, "auc": s_auc, "sensitivity": s_sens, "specificity": s_spec, "f1": s_f1},
    }
    objective_value = float(avg_metrics["auroc"])
    print(f"[Trial {trial.number}] -> objective (AUC) = {objective_value:.4f}", flush=True)
    return objective_value


def objective(trial: optuna.Trial, args) -> float:
    """
    Unified Optuna objective: build trial-level config from dataset/mode/roi_backend,
    run n-fold cross-validation via train_kfold.run_one_fold, return single objective (e.g. mean AUC).
    On each trial completion: save JSON, update CSV, write detailed logs.
    """
    _TRIAL_START_TIME[trial.number] = time.time()

    cfg_for_encoder = load_config(args.config)
    roi_enc = cfg_for_encoder.get("roi_encoder", "neurograph")
    # Use NeuroGraph native path only when 5_folds/ and NeuroGraph data exist; else use unified (e.g. smoke with dummy adhd_fold*)
    if args.dataset == "adhd" and args.mode == "roi_only" and roi_enc == "neurograph":
        if (_PROJECT_ROOT / "5_folds").exists() and _NEUROGRAPH_ROOT.is_dir():
            return _run_neurograph_roi_only_native(trial, args)
        # Fall through to unified path (adhd_fold*_775.npy + roi_matrices_775.npy)
    cfg_base, indices, labels, data_path, image_shape = _prepare_base_cfg_and_data(args)

    cfg = copy.deepcopy(cfg_base)
    _apply_common_hparams(trial, cfg, args)
    if cfg.get("use_roi_branch", False) and cfg.get("roi_encoder") == "neurograph":
        _apply_roi_neurograph_hparams(trial, cfg)
    elif cfg.get("use_roi_branch", False) and cfg.get("roi_encoder") == "chen2019":
        _apply_roi_chen2019_hparams(trial, cfg)
    if cfg.get("use_image_branch", False):
        _apply_image_hparams(trial, cfg)
    if args.mode == "fusion":
        _apply_fusion_hparams(trial, cfg)

    if getattr(args, "epochs", None) is not None:
        cfg["epochs"] = args.epochs

    # BatchNorm requires batch_size >= 2; enforce for fusion/ROI
    if args.mode == "fusion" or cfg.get("use_roi_branch"):
        cfg["batch_size"] = max(2, int(cfg.get("batch_size", 2)))
    if cfg.get("adhd_use_test_set_only"):
        cfg["batch_size"] = 2
    img_kw = cfg.get("image_encoder_kwargs") or {}
    img_enc = cfg.get("image_encoder", "")
    if img_enc == "vit3d":
        img_summary = (
            f" patch={img_kw.get('patch_size')} dim={img_kw.get('embed_dim')} depth={img_kw.get('depth')} "
            f"heads={img_kw.get('n_heads')} mlp_r={img_kw.get('mlp_ratio', 4):.1f} "
            f"drop_path={img_kw.get('drop_path_rate', 0):.2f} qkv_bias={img_kw.get('qkv_bias', True)}"
        )
    elif img_enc == "rae_vit_ad":
        img_summary = (
            f" img_size={img_kw.get('img_size')} embed_dim={img_kw.get('embed_dim')} "
            f"n_heads={img_kw.get('num_heads')}"
        )
    elif img_enc == "3dsctf":
        img_summary = (
            f" patch={img_kw.get('patch_size')} dim={img_kw.get('embed_dim')} depth={img_kw.get('depth')} "
            f"heads={img_kw.get('num_heads')} mlp_r={img_kw.get('mlp_ratio', 2):.1f} "
            f"drop_path={img_kw.get('drop_path_rate', 0):.2f} qkv_bias={img_kw.get('qkv_bias', True)}"
        )
    else:
        img_summary = ""
    print(
        f"[Trial {trial.number}] dataset={args.dataset} mode={args.mode} "
        f"lr={cfg['lr']:.2e} wd={cfg['weight_decay']:.2e} bs={cfg['batch_size']} "
        f"epochs={cfg['epochs']} dropout={cfg['dropout']:.2f} img_feat={cfg['image_feat_dim']}{img_summary}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_fold = _create_trial_logger(trial.number)
    fold_metrics_list = []
    n_total = len(indices)
    use_fixed_test_split = args.dataset == "adhd" and cfg.get("adhd_use_test_set_only", False) and n_total == 10

    if use_fixed_test_split:
        train_idx = list(range(6))
        val_idx = list(range(6, 8))
        test_idx = list(range(8, 10))
        fold_splits = [(train_idx, val_idx, test_idx)]
        log_fold(f"[Trial {trial.number}] Using fixed 6/2/2 split (6 train, 2 val, 2 test)")
    else:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=cfg["seed"])
        fold_splits = []
        for fold, (train_pos, val_pos) in enumerate(skf.split(np.arange(n_total), labels)):
            if args.max_folds is not None and (fold + 1) > args.max_folds:
                break
            t_idx = indices[train_pos].tolist() if hasattr(indices, "__getitem__") else train_pos.tolist()
            v_idx = indices[val_pos].tolist() if hasattr(indices, "__getitem__") else val_pos.tolist()
            fold_splits.append((t_idx, v_idx, None))

    for fold, one_split in enumerate(fold_splits):
        train_idx, val_idx = one_split[0], one_split[1]
        test_idx = one_split[2] if len(one_split) > 2 else None
        if args.max_folds is not None and not use_fixed_test_split and (fold + 1) > args.max_folds:
            log_fold(f"[Trial {trial.number}] max_folds={args.max_folds} reached, skipping remaining folds (smoke).")
            break
        log_fold(
            f"[Trial {trial.number}] Fold {fold+1}/{len(fold_splits) if not use_fixed_test_split else 1}: "
            f"train_n={len(train_idx)} val_n={len(val_idx)}"
            + (f" test_n={len(test_idx)}" if test_idx else "")
        )

        ckpt_path = None
        if getattr(args, "save_ckpt_dir", None):
            ckpt_path = str(Path(args.save_ckpt_dir) / f"trial{trial.number:03d}" / f"fold{fold+1}_best.pth")

        fold_start = time.time()
        metrics = run_one_fold(
            cfg,
            data_path,
            train_idx,
            val_idx,
            fold + 1,
            device,
            log_fold,
            image_shape,
            dataset_name=args.dataset,
            ckpt_path=ckpt_path,
            test_idx=test_idx,
        )
        fold_duration = time.time() - fold_start

        fold_metrics_list.append(metrics)
        log_fold(
            f"[Trial {trial.number}] Fold {fold+1} completed in {fold_duration/60:.1f}min: "
            f"Acc={metrics['accuracy']:.2f}%  AUC={metrics['auc']:.2f}%  "
            f"Sens={metrics['sensitivity']:.2f}%  Spec={metrics['specificity']:.2f}%  F1={metrics['f1']:.2f}%"
        )
        if use_fixed_test_split:
            break

    keys = ["accuracy", "auc", "sensitivity", "specificity", "f1"]
    means = {k: float(np.mean([m[k] for m in fold_metrics_list])) for k in keys}
    # ddof=1 for unbiased std estimate across k folds
    stds = {k: float(np.std([m[k] for m in fold_metrics_list], ddof=1)) for k in keys}

    log_fold(
        f"[Trial {trial.number}] "
        f"Acc {means['accuracy']:.2f}±{stds['accuracy']:.2f}%  "
        f"AUC {means['auc']:.2f}±{stds['auc']:.2f}%  "
        f"Sens {means['sensitivity']:.2f}±{stds['sensitivity']:.2f}%  "
        f"Spec {means['specificity']:.2f}±{stds['specificity']:.2f}%  "
        f"F1 {means['f1']:.2f}±{stds['f1']:.2f}%"
    )
    _trial_metrics_store[trial.number] = {"means": means, "stds": stds}

    # Objective: AUC (macro, consistent with train_kfold best-model selection)
    objective_value = means["auc"]
    log_fold(f"[Trial {trial.number}] -> objective (AUC) = {objective_value:.2f}%")

    # Save trial results immediately after each trial
    _save_trial_json(trial.number, trial.params, means, stds, args)
    _update_summary_csv(trial.number, trial.params, means, stds, objective_value)

    return objective_value


def main():
    global _LOG_DIR, _SUMMARY_CSV_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/brain_structure.yaml")
    parser.add_argument("--dataset", type=str, default="adhd", choices=["brain_structure", "adhd"])
    parser.add_argument("--mode", type=str, choices=["image_only", "roi_only", "fusion"], default="image_only")
    parser.add_argument("--fusion", type=str, choices=["concat", "contrastive", "cross_attention"], default=None)
    parser.add_argument("--roi_backend", type=str, default="neurograph", help="ROI backend (config key: neurograph for NeuroGraph ResidualGNNs)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_folds", type=int, default=None, help="If set, run only first max_folds folds (smoke test).")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="kfold_unified")
    parser.add_argument("--balance_classes", action="store_true", help="For brain_structure: undersample to equal AD/CN before k-fold.")
    # V2: New parameter for log directory
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for detailed logs, trial JSONs, and CSV summary. If not set, uses logs_v2_<dataset>_<mode>.")
    parser.add_argument("--save_ckpt_dir", type=str, default=None, help="If set, save best-per-fold checkpoints under this directory for each trial.")
    parser.add_argument("--save_results_dir", type=str, default=None, help="(DEPRECATED: use --log_dir) If set, save best-trial metrics (mean±std) to this dir as <study_name>_best_metrics.json.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs (e.g. 2 for smoke).")
    parser.add_argument("--adhd_use_test_set_only", action="store_true", help="Use only adhd_test_775.npy + test_roi_matrices_775.npy (10 samples, 6/2/2 split); batch_size forced to 2.")
    # V2: Auto-create storage if not specified
    parser.add_argument("--storage", type=str, default=None, help="Optuna DB storage URL (e.g. sqlite:///optuna.db). If not set, auto-creates sqlite DB in log_dir.")
    args = parser.parse_args()

    # V2: Setup log directory
    if args.log_dir:
        _LOG_DIR = Path(args.log_dir)
    else:
        fusion_suffix = f"_{args.fusion}" if args.fusion else ""
        _LOG_DIR = Path(f"logs_v2_{args.dataset}_{args.mode}{fusion_suffix}")
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # V2: Setup summary CSV
    _SUMMARY_CSV_PATH = _LOG_DIR / "summary.csv"

    # V2: Auto-create SQLite storage if not specified
    storage = args.storage
    if storage is None:
        storage = f"sqlite:///{_LOG_DIR / 'optuna_study.db'}"

    # V2: Setup checkpoint directory if not specified
    if args.save_ckpt_dir is None:
        args.save_ckpt_dir = str(_LOG_DIR / "checkpoints")

    print("=" * 80, flush=True)
    print(f"Optuna unified k-fold tuning V2: study={args.study_name}", flush=True)
    print(f"  dataset={args.dataset} mode={args.mode} fusion={args.fusion}", flush=True)
    print(f"  n_folds={args.n_folds} n_trials={args.n_trials}", flush=True)
    print(f"  storage={storage}", flush=True)
    print(f"  log_dir={_LOG_DIR}", flush=True)
    print(f"  checkpoint_dir={args.save_ckpt_dir}", flush=True)
    print(f"  summary_csv={_SUMMARY_CSV_PATH}", flush=True)
    print("=" * 80, flush=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,  # resume if DB already has this study
    )
    already_done = len(study.trials)
    if already_done > 0:
        print(f"Resuming study '{args.study_name}': {already_done} trials already completed.", flush=True)

    study.optimize(
        lambda t: objective(t, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
        catch=(RuntimeError, torch.cuda.OutOfMemoryError),
    )

    print("=" * 80, flush=True)
    completed_with_value = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed_with_value:
        print("No trial completed successfully; skipping best-trial summary (avoid 'Record does not exist').", flush=True)
    else:
        print("Best trial:", flush=True)
        b = study.best_trial
        print(f"  Value (AUC): {b.value:.4f}" if args.mode == "roi_only" else f"  Value (AUC %): {b.value:.2f}%", flush=True)
        print("  Params:", b.params, flush=True)

        # Save best-trial metrics to log_dir
        if b.number in _trial_metrics_store:
            best_metrics_path = _LOG_DIR / "best_trial_metrics.json"
            rec = {
                "study_name": args.study_name,
                "dataset": args.dataset,
                "mode": args.mode,
                "fusion": args.fusion,
                "best_trial_number": b.number,
                "objective_value": float(b.value),
                "params": b.params,
                "metrics": _trial_metrics_store[b.number],
            }
            with open(best_metrics_path, "w") as f:
                json.dump(rec, f, indent=2)
            print(f"  Saved best-trial metrics to {best_metrics_path}", flush=True)

    print(f"  All trial JSONs saved to {_LOG_DIR / 'trial_jsons'}/", flush=True)
    print(f"  Summary CSV: {_SUMMARY_CSV_PATH}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

