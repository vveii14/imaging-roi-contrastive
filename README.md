# Imaging–ROI Contribution Analysis: A Pluggable Contrastive Framework for Brain Imaging Classification

**MICCAI 2026**

This repository provides the classification framework for joint imaging and ROI (region-of-interest) analysis with a pluggable contrastive fusion module. It supports 5-fold cross-validation, multiple fusion strategies (concat, contrastive, cross-attention), and is runnable on ADHD and ADNI datasets.

## Framework Overview

- **Dual-branch**: 3D imaging encoder (e.g. 3D-SCTF) + ROI encoder (e.g. BrainNet-style GNN).
- **Fusion**: Concat, contrastive, or cross-attention; contrastive learning aligns image and ROI representations.
- **Training**: Best-validation model selection per fold; metrics (Accuracy, AUC, Sensitivity, Specificity, F1) on held-out fold.

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

## Data

### ADHD

- Place ROI matrices and optional 3D images under a directory (e.g. `data/adhd/`).
- Config key: `adhd_data_dir`. Expects e.g. `roi_matrices_775.npy` (116×116 per subject) and labels; see `data/adhd.py` for expected layout.

### ADNI

- Place `FDG.csv`, `disease label.xlsx`, and preprocessed 3D volumes (e.g. 128³ `.npy`) under a directory (e.g. `data/adni_fdg_pet/`).
- Config keys: `adni_data_dir`, `adni_image_dir`. See `data/adni_fdg_pet.py` and `data/adni.py` for formats.

Set `data_root` and dataset-specific paths in the YAML config or via environment.

## Quick Run

**ADHD, fusion with contrastive** (edit `configs/adhd_fusion_contrastive.yaml` to point to your `adhd_data_dir`):

```bash
python train_kfold.py --dataset adhd --mode fusion --fusion contrastive \
  --config configs/adhd_fusion_contrastive.yaml --n_folds 5
```

**ADNI, fusion with contrastive** (edit `configs/adni_fusion_contrastive.yaml` for `adni_data_dir` and `adni_image_dir`):

```bash
python train_kfold.py --dataset adni --mode fusion --fusion contrastive \
  --config configs/adni_fusion_contrastive.yaml --n_folds 5
```

Optional: `--epochs`, `--batch_size`, `--output_dir` to override config or write logs/checkpoints to a given directory.

## Project Structure

```
imaging-roi-contrastive/
├── README.md
├── requirements.txt
├── train_kfold.py          # 5-fold training entry
├── configs/
│   ├── adhd_fusion_contrastive.yaml
│   └── adni_fusion_contrastive.yaml
├── data/
│   ├── adhd.py
│   ├── adni.py
│   ├── adni_fdg_pet.py
│   └── ...
└── models/
    ├── fusion_model.py     # Dual-branch + fusion
    ├── fusion.py           # Concat / Contrastive / CrossAttention
    ├── image_encoder_3dsctf.py
    ├── roi_encoder_brainnet.py
    └── ...
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
