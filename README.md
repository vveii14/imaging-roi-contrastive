# Imaging–ROI Contribution Analysis: A Pluggable Contrastive Framework for Brain Imaging Classification

**MICCAI 2026**

This repository provides the **full classification and tuning pipeline** for joint imaging and ROI analysis with pluggable contrastive fusion. It includes 5-fold cross-validation, Optuna hyperparameter search, and supports **image_only**, **roi_only**, and **fusion** (concat / contrastive / cross_attention) on ADHD and ADNI.

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

- Python 3.8+
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

## Main Entry: train_kfold_optuna_unified_v2.py (full functionality)

This script is the **complete pipeline** used for experiments: Optuna search, SQLite study storage, per-trial JSON/CSV, checkpoints, and (optionally) NeuroGraph-native roi_only for ADHD/ADNI.

### Supported modes and datasets

| Dataset | image_only | roi_only | fusion (concat / contrastive / cross_attention) |
|---------|------------|----------|--------------------------------------------------|
| ADHD    | ✓          | ✓ (NeuroGraph native or unified path) | ✓ |
| ADNI    | ✓          | ✓ (NeuroGraph native or unified path) | ✓ |

### Example: ADHD fusion contrastive (Optuna, 30 trials)

Edit `configs/adhd_fusion_3dsctf_contrastive.yaml` so `data_root` and `adhd_data_dir` point to your data, then:

```bash
python train_kfold_optuna_unified_v2.py --dataset adhd --mode fusion --fusion contrastive \
  --config configs/adhd_fusion_3dsctf_contrastive.yaml --n_trials 30 --n_folds 5 \
  --log_dir logs_adhd_contrastive --study_name adhd_contrastive
```

### Example: ADNI fusion contrastive (Optuna)

Edit `configs/adni_fusion_3dsctf_contrastive.yaml` for `adni_data_dir` and `adni_image_dir`, then:

```bash
python train_kfold_optuna_unified_v2.py --dataset adni --mode fusion --fusion contrastive \
  --config configs/adni_fusion_3dsctf_contrastive.yaml --n_trials 30 --n_folds 5 \
  --log_dir logs_adni_contrastive --study_name adni_contrastive
```

### Outputs under `--log_dir`

- `summary.csv` — one row per completed trial.
- `trial_jsons/trial_XXX.json` — params and metrics per trial.
- `checkpoints/` — best per-fold checkpoints per trial (if not disabled).
- `optuna_study.db` — SQLite study (resume with `load_if_exists=True`).
- `best_trial_metrics.json` — written when the study finishes.

Optional: `--storage sqlite:///path/to.db` to use a different DB; `--save_ckpt_dir` to override checkpoint directory.

## Smoke run

A short run with minimal ADHD-style data to verify environment and scripts.

```bash
cd /path/to/imaging-roi-contrastive
pip install -r requirements.txt
python scripts/prepare_dummy_adhd.py
bash run_smoke.sh
```

Single run (fusion contrastive, 1 trial, 1 fold, 2 epochs):

```bash
python scripts/prepare_dummy_adhd.py
python train_kfold_optuna_unified_v2.py --dataset adhd --mode fusion --fusion contrastive \
  --config configs/adhd_fusion_3dsctf_contrastive.yaml --n_trials 1 --max_folds 1 --epochs 2 \
  --log_dir logs_smoke --study_name smoke
```

`run_smoke.sh` runs in sequence: fusion contrastive (3D-SCTF+NeuroGraph), image-only 3D-SCTF, ViT3D, RAE-ViT, roi_only NeuroGraph, roi_only Chen2019 (each 1 trial, 1 fold, 2 epochs); logs go to `logs_smoke/`. Without `5_folds/`, roi_only NeuroGraph uses the unified data path (adhd_fold*_775.npy). RAE-ViT is memory-heavy; use a dedicated GPU or smaller batch if sharing a device.

**Running all experiments (baselines + fusion) on a single GPU**: use `run_smoke_gpu1.sh` (default: GPU 1). This script uses `--adhd_use_test_set_only` (25 samples, fixed 15/5/5 split, `batch_size=2`) **only to verify that the code runs**. For real experiments, obtain the full dataset, run your own preprocessing, and do not use `--adhd_use_test_set_only`. The script runs: fusion contrastive, image-only 3D-SCTF, ViT3D, RAE-ViT, roi_only NeuroGraph, roi_only Chen2019; logs in `logs_smoke_gpu1/`. Change `CUDA_VISIBLE_DEVICES` in the script to use another GPU.

## Config and encoder mapping

| 类型 | ADHD config | ADNI config |
|------|-------------|-------------|
| **image_only 3D-SCTF** | `adhd_image_only_3dsctf.yaml` | `adni_image_only_3dsctf.yaml` |
| **image_only ViT3D** | `adhd_image_only_vit3d.yaml` | `adni_image_only_vit3d.yaml` |
| **image_only RAE-ViT** | `adhd_image_only_rae_vit.yaml` | `adni_image_only_rae_vit.yaml` |
| **roi_only NeuroGraph** | `adhd_roi_only.yaml` | (NeuroGraph native requires 5_folds_adni + .pkl) |
| **roi_only Chen2019** | `adhd_roi_only_chen2019.yaml` | `adni_roi_only_chen2019.yaml` |
| **fusion 3D-SCTF** (concat/contrastive/cross_attention) | `adhd_fusion_3dsctf.yaml`, `adhd_fusion_3dsctf_contrastive.yaml` | `adni_fusion_3dsctf.yaml`, `adni_fusion_3dsctf_contrastive.yaml` |
| **fusion ViT3D** | — | `adni_fusion_vit3d.yaml` |
| **fusion RAE-ViT** | `adhd_fusion_rae_vit.yaml` | `adni_fusion_rae_vit.yaml` |

In all configs, `data_root` / `adhd_data_dir` / `adni_data_dir` default to `./data`; ViT3D `pretrained_checkpoint` defaults to `null`—set the path in the relevant YAML for pretrained weights (see ViT3D pretrained weights above).

## Optional: Single-config 5-fold (train_kfold.py)

The module `train_kfold.py` is **required** by the main entry (it provides `load_config` and `run_one_fold`). It can also be run standalone for a single fixed config (no Optuna):

```bash
python train_kfold.py --dataset adhd --mode fusion --fusion contrastive \
  --config configs/adhd_fusion_contrastive.yaml --n_folds 5
```

Use `configs/adhd_fusion_contrastive.yaml` or `configs/adni_fusion_contrastive.yaml`; set `data_root` and dataset paths inside the YAML.

## Data and config

### ADHD (fusion / image_only)

- **Config**: `configs/adhd_fusion_3dsctf_contrastive.yaml` or `configs/adhd_fusion_contrastive.yaml`.
- **Data**: ROI matrices (e.g. `roi_matrices_775.npy`, 116×116) and labels under `adhd_data_dir`; optional 3D images. See `data/adhd.py` for expected layout.
- **Test set (in repo, for quick testing only)**: We provide 25 samples in `data/adhd/adhd_test_775.npy` and `data/adhd/test_roi_matrices_775.npy` so you can run the pipeline without the full dataset. With `--adhd_use_test_set_only`, the split is **fixed 15/5/5** (15 train, 5 val, 5 test) and `batch_size` is forced to 2. **This is only for verifying that the code runs; for real experiments you must obtain the full ADHD dataset, run your own preprocessing, and use the normal data layout (e.g. 5-fold or your own splits) without `--adhd_use_test_set_only`.**

### ADNI (fusion / image_only)

- **Config**: `configs/adni_fusion_3dsctf_contrastive.yaml` or `configs/adni_fusion_contrastive.yaml`.
- **Data**: `FDG.csv`, `disease label.xlsx`, and preprocessed 3D volumes (e.g. 128³ `.npy`) under `adni_data_dir`; set `adni_image_dir` to the image subfolder. See `data/adni_fdg_pet.py` and `data/adni.py`.

### ROI-only (NeuroGraph native path)

For **adhd roi_only** or **adni roi_only** using the NeuroGraph-native pipeline (predefined folds and [NeuroGraph](https://github.com/Anwar-Said/NeuroGraph) data layout):

- **ADHD**: Project root must contain `5_folds/` with fold assignments; `BrainNet_EndtoEnd-main/data/` (and timeseries/graph layout as in NeuroGraph) must be present. The script uses `BrainNet_EndtoEnd-main/train.py` and `utils` for this path.
- **ADNI**: Project root must contain `5_folds_adni/5_folds/fold_assignments.csv`; NeuroGraph-style graph `.pkl` files under `BrainNet_EndtoEnd-main/data/.../ADNI/...`. Prepare via e.g. `scripts/prepare_brainnet_from_adni.py` (if provided in your data pipeline).

The repo includes **BrainNet_EndtoEnd-main** (NeuroGraph-compatible model, train, utils) so that these native roi_only paths run when the corresponding data and fold files are in place.

## Project structure

```
imaging-roi-contrastive/
├── README.md
├── requirements.txt
├── run_smoke.sh                        # Smoke run: 1 trial × 1 fold × 2 epochs per encoder/fusion
├── train_kfold_optuna_unified_v2.py    # Main: full Optuna pipeline (all modes)
├── train_kfold.py                      # Required module; also runnable for single-config 5-fold
├── configs/
│   ├── adhd_image_only_3dsctf.yaml, adhd_image_only_vit3d.yaml, adhd_image_only_rae_vit.yaml
│   ├── adhd_roi_only.yaml (NeuroGraph), adhd_roi_only_chen2019.yaml
│   ├── adhd_fusion_3dsctf.yaml, adhd_fusion_3dsctf_contrastive.yaml, adhd_fusion_rae_vit.yaml
│   ├── adni_image_only_*.yaml, adni_roi_only_chen2019.yaml, adni_fusion_*.yaml
│   └── adhd_fusion_contrastive.yaml, adni_fusion_contrastive.yaml (single-config)
├── scripts/
│   └── prepare_dummy_adhd.py           # Minimal ADHD data for smoke run
├── data/
│   ├── adhd.py, adni.py, adni_fdg_pet.py
│   └── ...
├── models/
│   ├── fusion_model.py, fusion.py
│   ├── image_encoder_3dsctf.py, roi_encoder_brainnet.py
│   └── ...
└── BrainNet_EndtoEnd-main/             # NeuroGraph-compatible; for roi_only native (ADHD/ADNI)
    ├── model.py, train.py, utils.py
    └── data/ (layout as per [NeuroGraph](https://github.com/Anwar-Said/NeuroGraph); put your data here or link)
```

## Citation

If you use this code, please cite our MICCAI 2026 paper:

```bibtex
@inproceedings{imaging-roi-contrastive-miccai26,
  title     = {Imaging–ROI Contribution Analysis: A Pluggable Contrastive Framework for Brain Imaging Classification},
  booktitle = {MICCAI},
  year      = {2026},
}
```

## License

See repository license file.
