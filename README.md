# Imaging–ROI Contribution Analysis: A Pluggable Contrastive Framework for Brain Imaging Classification

**MICCAI 2026**

This repository provides the **full classification and tuning pipeline** for joint imaging and ROI analysis with pluggable contrastive fusion. It is set up as an **ADHD demonstration**: 5-fold cross-validation, Optuna hyperparameter search, and support for **image_only**, **roi_only**, and **fusion** (concat / contrastive / cross_attention) on ADHD data.

## Framework Overview

- **Dual-branch**: 3D imaging encoder + ROI encoder; fusion combines both branches.
- **Encoders** (all supported in this release):
  - **ROI**: **Chen2019** ([DNN](https://pubs.rsna.org/doi/10.1148/ryai.2019190012) on connectivity), [**NeuroGraph**](https://github.com/Anwar-Said/NeuroGraph) (GNN on correlation graph).
  - **Image**: [**3D-SCTF**](https://github.com/NWPU-903PR/3DSC-TF) (3D Swin / conv), [**ViT3D**](https://github.com/qasymjomart/ViT_recipe_for_AD) (Vision Transformer 3D; weights also available there), [**RAE-ViT**](https://github.com/jomeiri/RAE-ViT-AD) (RAE-ViT-AD).
- **Fusion**: **concat**, **contrastive**, or **cross_attention** (any image encoder + any ROI encoder).
- **Training**: Best-validation model selection per fold; metrics (Accuracy, AUC, Sensitivity, Specificity, F1) on held-out fold.
- **Tuning**: Optuna study over lr, weight_decay, dropout, batch_size, epochs, and (for fusion) image/ROI/fusion hyperparameters.

### Encoder references (links)

| Encoder | Description | Link |
|---------|-------------|------|
| **NeuroGraph** | ROI GNN (correlation graph) | [NeuroGraph](https://github.com/Anwar-Said/NeuroGraph) |
| **3D-SCTF** | 3D Swin / conv image encoder | [3DSC-TF](https://github.com/NWPU-903PR/3DSC-TF) |
| **ViT3D** | Vision Transformer 3D; pretrained weights available | [ViT_recipe_for_AD](https://github.com/qasymjomart/ViT_recipe_for_AD) |
| **RAE-ViT** | RAE-ViT-AD image encoder | [RAE-ViT-AD](https://github.com/jomeiri/RAE-ViT-AD) |
| **Chen2019** | DNN on connectivity (ROI) | [Radiology: Artificial Intelligence (RSNA)](https://pubs.rsna.org/doi/10.1148/ryai.2019190012) |

### ViT3D pretrained weights

ViT3D can load checkpoints in ViT_recipe format for transfer or fine-tuning. Weights are available at [ViT_recipe_for_AD](https://github.com/qasymjomart/ViT_recipe_for_AD).

- **Config** (under `image_encoder_kwargs`):
  - `pretrained_checkpoint`: path to local `.pth` or `.pth.tar`; set to `null` to train from scratch.
  - `checkpoint_key`: key to extract backbone state_dict when the checkpoint is a dict; typically **`net`** for ViT_recipe.
- **Supported formats**:
  - **ViT_recipe / MAE**: `torch.load(path)` returns a dict with `state_dict = ckpt["net"]` (e.g. MAE `vit_b_mae.pth`). Set `checkpoint_key: net` or `state_dict` as appropriate.
  - **Raw state_dict**: if the file is a state_dict only, `checkpoint_key` is ignored; keys must match ViT3D (`patch_embed`, `blocks`, `cls_token`, `pos_embed`, etc.).
- **Example** (in config):
  ```yaml
  image_encoder_kwargs:
    pretrained_checkpoint: /path/to/vit_b_mae.pth   # or null
    checkpoint_key: net
  ```

## Requirements

- **Python** 3.8+ (tested with 3.10, 3.12)
- **CUDA**-capable GPU recommended (smoke runs in ~15–30 min on one GPU)
- **Install:**
  ```bash
  pip install -r requirements.txt
  ```
  For strict reproducibility, use a virtual env and consider pinning versions (e.g. `torch`, `torch-geometric`, `optuna`) to match your test environment.

## Main Entry: train_kfold_optuna_unified_v2.py (full functionality)

This script is the **complete pipeline** used for experiments: Optuna search, SQLite study storage, per-trial JSON/CSV, checkpoints, and (optionally) NeuroGraph-native roi_only for ADHD.

### Supported modes (ADHD)

| Mode | Description |
|------|-------------|
| **image_only** | 3D-SCTF, ViT3D, or RAE-ViT on 3D volumes |
| **roi_only** | NeuroGraph (unified or native path) or Chen2019 on ROI matrices |
| **fusion** | concat / contrastive / cross_attention (any image + any ROI encoder) |

### Example: ADHD fusion contrastive (Optuna, 30 trials)

Edit `configs/adhd_fusion_3dsctf_contrastive.yaml` so `data_root` and `adhd_data_dir` point to your data, then:

```bash
python train_kfold_optuna_unified_v2.py --dataset adhd --mode fusion --fusion contrastive \
  --config configs/adhd_fusion_3dsctf_contrastive.yaml --n_trials 30 --n_folds 5 \
  --log_dir logs_adhd_contrastive --study_name adhd_contrastive
```

### Outputs under `--log_dir`

- `summary.csv` — one row per completed trial.
- `trial_jsons/trial_XXX.json` — params and metrics per trial.
- `checkpoints/` — best per-fold checkpoints per trial (if not disabled).
- `optuna_study.db` — SQLite study (resume with `load_if_exists=True`).
- `best_trial_metrics.json` — written when the study finishes.

Optional: `--storage sqlite:///path/to.db` to use a different DB; `--save_ckpt_dir` to override checkpoint directory.

## Smoke run

A short run with minimal ADHD-style data to verify environment and scripts. **The repository includes a 10-sample ADHD test set** (`data/adhd/adhd_test_775.npy`, `data/adhd/test_roi_matrices_775.npy`, ~68 MB, under GitHub’s 100 MB limit) so reviewers can reproduce the smoke without generating data.

### Reproducing the smoke run (for reviewers)

To reproduce the smoke run on your machine:

1. **Clone and install**
   ```bash
   git clone https://github.com/vveii14/imaging-roi-contrastive.git
   cd imaging-roi-contrastive
   pip install -r requirements.txt
   ```

2. **Run the full smoke (recommended: one GPU)**  
   Default uses **GPU 1**. Edit `CUDA_VISIBLE_DEVICES` in the script to use another GPU (e.g. `0`).
   ```bash
   bash run_smoke_gpu1.sh
   ```
   This runs 6 experiments in sequence (fusion contrastive, image-only 3D-SCTF, ViT3D, RAE-ViT, roi_only NeuroGraph, roi_only Chen2019), each with 1 trial, 1 fold, 2 epochs, on the included 10-sample set (fixed 6/2/2 train/val/test). **No need to run `prepare_dummy_adhd.py`** — the test set is in the repo. If missing (e.g. partial clone), the script runs it once.

3. **Check outputs**  
   Logs and checkpoints are written under `logs_smoke_gpu1/` (see [Log locations](#log-locations) below). Approximate runtime: ~2–5 minutes per experiment on a single GPU (total ~15–30 min).

**Alternative (any GPU):** `bash run_smoke.sh` — same 6 experiments, uses `CUDA_VISIBLE_DEVICES` from your environment; logs go to `logs_smoke/`.

**Seed:** The pipeline uses a fixed seed (default `42`) so runs are reproducible.

### Log locations

| What | Where |
|------|--------|
| Full console output per experiment | `logs_smoke_gpu1/<name>.log` (e.g. `fusion_contrastive.log`, `roi_neurograph.log`) |
| Summary CSV per experiment | `logs_smoke_gpu1/<name>/summary.csv` |
| Best-trial metrics (JSON) | `logs_smoke_gpu1/<name>/best_trial_metrics.json` |
| Per-trial details | `logs_smoke_gpu1/<name>/trial_XXX/detailed.log` |
| Checkpoints | `logs_smoke_gpu1/<name>/checkpoints/trialXXX/fold1_best.pth` |

### Single-command smoke (fusion only)

```bash
python train_kfold_optuna_unified_v2.py --dataset adhd --mode fusion --fusion contrastive \
  --config configs/adhd_fusion_3dsctf_contrastive.yaml --adhd_use_test_set_only \
  --n_trials 1 --max_folds 1 --epochs 2 --log_dir logs_smoke --study_name smoke
```

**Note:** The 10-sample setup is for **verifying that the code runs** only. For real experiments, use the full ADHD dataset and normal 5-fold (or your own) splits without `--adhd_use_test_set_only`. RAE-ViT is memory-heavy; `batch_size=2` is used for the smoke to avoid OOM.

## Config and encoder mapping (ADHD)

| Mode | Config |
|------|--------|
| **image_only 3D-SCTF** | `adhd_image_only_3dsctf.yaml` |
| **image_only ViT3D** | `adhd_image_only_vit3d.yaml` |
| **image_only RAE-ViT** | `adhd_image_only_rae_vit.yaml` |
| **roi_only NeuroGraph** | `adhd_roi_only.yaml` |
| **roi_only Chen2019** | `adhd_roi_only_chen2019.yaml` |
| **fusion 3D-SCTF** (concat/contrastive/cross_attention) | `adhd_fusion_3dsctf.yaml`, `adhd_fusion_3dsctf_contrastive.yaml` |
| **fusion RAE-ViT** | `adhd_fusion_rae_vit.yaml` |

In all configs, `data_root` / `adhd_data_dir` default to `./data`; ViT3D `pretrained_checkpoint` defaults to `null`—set the path in the relevant YAML for pretrained weights (see ViT3D pretrained weights above).

## Optional: Single-config 5-fold (train_kfold.py)

The module `train_kfold.py` is **required** by the main entry (it provides `load_config` and `run_one_fold`). It can also be run standalone for a single fixed config (no Optuna):

```bash
python train_kfold.py --dataset adhd --mode fusion --fusion contrastive \
  --config configs/adhd_fusion_contrastive.yaml --n_folds 5
```

Set `data_root` and `adhd_data_dir` inside the YAML.

## Data and config (ADHD)

### ADHD (fusion / image_only / roi_only)

- **Config**: e.g. `configs/adhd_fusion_3dsctf_contrastive.yaml` or `configs/adhd_fusion_contrastive.yaml`.
- **Data**: ROI matrices (e.g. `roi_matrices_775.npy`, 116×116) and labels under `adhd_data_dir`; optional 3D images. See `data/adhd.py` for expected layout.
- **Test set (in repo, for smoke only)**: We provide 10 samples in `data/adhd/adhd_test_775.npy` and `data/adhd/test_roi_matrices_775.npy` (~68 MB total, under GitHub 100 MB limit). With `--adhd_use_test_set_only`, the split is **fixed 6/2/2** (6 train, 2 val, 2 test) and `batch_size` is forced to 2. **Only for verifying that the code runs; for real experiments use the full ADHD dataset and normal 5-fold without `--adhd_use_test_set_only`.**

### ROI-only (NeuroGraph native path)

For **adhd roi_only** using the NeuroGraph-native pipeline (predefined folds and [NeuroGraph](https://github.com/Anwar-Said/NeuroGraph) data layout):

- Project root must contain `5_folds/` with fold assignments; `NeuroGraph-main/data/` (and timeseries/graph layout as in [NeuroGraph](https://github.com/Anwar-Said/NeuroGraph)) must be present. The script uses `NeuroGraph-main/train.py` and `utils` for this path.

If `5_folds/` is missing, the pipeline falls back to the unified path (e.g. `adhd_fold*_775.npy` + `roi_matrices_775.npy` or test-set files when using `--adhd_use_test_set_only`).

The repo includes **NeuroGraph-main** (NeuroGraph model, train, utils) so that the native roi_only path runs when the corresponding data and fold files are in place.

## Project structure

```
imaging-roi-contrastive/
├── README.md
├── requirements.txt
├── run_smoke.sh                        # Smoke run: 1 trial × 1 fold × 2 epochs per encoder/fusion
├── train_kfold_optuna_unified_v2.py    # Main: full Optuna pipeline (ADHD modes)
├── train_kfold.py                      # Required module; also runnable for single-config 5-fold
├── configs/
│   ├── adhd_image_only_3dsctf.yaml, adhd_image_only_vit3d.yaml, adhd_image_only_rae_vit.yaml
│   ├── adhd_roi_only.yaml (NeuroGraph), adhd_roi_only_chen2019.yaml
│   ├── adhd_fusion_3dsctf.yaml, adhd_fusion_3dsctf_contrastive.yaml, adhd_fusion_rae_vit.yaml
│   └── adhd_fusion_contrastive.yaml (single-config)
├── scripts/
│   └── prepare_dummy_adhd.py           # Minimal ADHD data for smoke run
├── data/
│   ├── adhd.py
│   └── ...
├── models/
│   ├── fusion_model.py, fusion.py
│   ├── image_encoder_3dsctf.py, roi_encoder_brainnet.py  # NeuroGraph ROI encoder
│   └── ...
└── NeuroGraph-main/                    # NeuroGraph; for roi_only native (ADHD)
    ├── model.py, train.py, utils.py
    └── data/ (layout as per [NeuroGraph](https://github.com/Anwar-Said/NeuroGraph); put your data here or link)
```


## License

See repository license file.
