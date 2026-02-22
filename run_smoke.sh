#!/bin/bash
# Smoke run in release folder: 1 trial, 1 fold, 2 epochs per encoder/fusion.
# Run from repo root after: pip install -r requirements.txt && python scripts/prepare_dummy_adhd.py

set -e
cd "$(dirname "$0")"
LOG=logs_smoke
mkdir -p "$LOG"

echo "=== Smoke: ADHD fusion contrastive (3D-SCTF + NeuroGraph) ==="
python train_kfold_optuna_unified_v2.py --dataset adhd --mode fusion --fusion contrastive \
  --config configs/adhd_fusion_3dsctf_contrastive.yaml --n_trials 1 --max_folds 1 --epochs 2 \
  --log_dir "$LOG/fusion_contrastive" --study_name smoke_fusion 2>&1 | tee "$LOG/smoke_fusion_contrastive.log" | tail -5

echo "=== Smoke: ADHD image-only 3D-SCTF ==="
python train_kfold_optuna_unified_v2.py --dataset adhd --mode image_only \
  --config configs/adhd_image_only_3dsctf.yaml --n_trials 1 --max_folds 1 --epochs 2 \
  --log_dir "$LOG/image_3dsctf" --study_name smoke_3dsctf 2>&1 | tee "$LOG/smoke_3dsctf.log" | tail -5

echo "=== Smoke: ADHD image-only ViT3D (from scratch, no .pth) ==="
python train_kfold_optuna_unified_v2.py --dataset adhd --mode image_only \
  --config configs/adhd_image_only_vit3d.yaml --n_trials 1 --max_folds 1 --epochs 2 \
  --log_dir "$LOG/image_vit3d" --study_name smoke_vit3d 2>&1 | tee "$LOG/smoke_vit3d.log" | tail -5

echo "=== Smoke: ADHD image-only RAE-ViT ==="
python train_kfold_optuna_unified_v2.py --dataset adhd --mode image_only \
  --config configs/adhd_image_only_rae_vit.yaml --n_trials 1 --max_folds 1 --epochs 2 \
  --log_dir "$LOG/image_rae_vit" --study_name smoke_rae_vit 2>&1 | tee "$LOG/smoke_rae_vit.log" | tail -5

echo "=== Smoke: ADHD ROI-only NeuroGraph ==="
rm -f "$LOG/roi_neurograph/optuna_study.db"
python train_kfold_optuna_unified_v2.py --dataset adhd --mode roi_only \
  --config configs/adhd_roi_only.yaml --n_trials 1 --max_folds 1 --epochs 2 \
  --log_dir "$LOG/roi_neurograph" --study_name smoke_neurograph 2>&1 | tee "$LOG/smoke_neurograph.log" | tail -5

echo "=== Smoke: ADHD ROI-only Chen2019 ==="
python train_kfold_optuna_unified_v2.py --dataset adhd --mode roi_only \
  --config configs/adhd_roi_only_chen2019.yaml --n_trials 1 --max_folds 1 --epochs 2 \
  --log_dir "$LOG/roi_chen2019" --study_name smoke_chen2019 2>&1 | tee "$LOG/smoke_chen2019.log" | tail -5

echo "=== All smoke runs finished. Logs under $LOG/ ==="
