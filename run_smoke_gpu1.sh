#!/bin/bash
# All baseline + fusion on GPU 1, 1 fold, 1 trial, 2 epochs.
# Data: adhd_test_775.npy + test_roi_matrices_775.npy (25 samples) only; batch_size=2 to avoid OOM.
# Run from repo root. Ensures test set exists: python scripts/prepare_dummy_adhd.py (once).

set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1
LOG=logs_smoke_gpu1
mkdir -p "$LOG"

# Ensure 25-sample test set exists (adhd_test_775.npy, test_roi_matrices_775.npy)
if [ ! -f "data/adhd/adhd_test_775.npy" ]; then
  echo "Generating test set (25 samples)..."
  python scripts/prepare_dummy_adhd.py
fi

EXTRA="--adhd_use_test_set_only --n_trials 1 --max_folds 1 --epochs 2"

run_one() {
  local name="$1"
  shift
  echo ""
  echo "=============================================="
  echo "=== $name ==="
  echo "=============================================="
  "$@"
  echo ""
}

# --- Fusion ---
run_one "Fusion contrastive (3D-SCTF + NeuroGraph)" \
  python train_kfold_optuna_unified_v2.py --dataset adhd --mode fusion --fusion contrastive \
  --config configs/adhd_fusion_3dsctf_contrastive.yaml $EXTRA \
  --log_dir "$LOG/fusion_contrastive" --study_name smoke_fusion 2>&1 | tee "$LOG/fusion_contrastive.log"

# --- Baselines: image_only ---
run_one "Baseline: image-only 3D-SCTF" \
  python train_kfold_optuna_unified_v2.py --dataset adhd --mode image_only \
  --config configs/adhd_image_only_3dsctf.yaml $EXTRA \
  --log_dir "$LOG/image_3dsctf" --study_name smoke_3dsctf 2>&1 | tee "$LOG/image_3dsctf.log"

run_one "Baseline: image-only ViT3D" \
  python train_kfold_optuna_unified_v2.py --dataset adhd --mode image_only \
  --config configs/adhd_image_only_vit3d.yaml $EXTRA \
  --log_dir "$LOG/image_vit3d" --study_name smoke_vit3d 2>&1 | tee "$LOG/image_vit3d.log"

run_one "Baseline: image-only RAE-ViT" \
  python train_kfold_optuna_unified_v2.py --dataset adhd --mode image_only \
  --config configs/adhd_image_only_rae_vit.yaml $EXTRA \
  --log_dir "$LOG/image_rae_vit" --study_name smoke_rae_vit 2>&1 | tee "$LOG/image_rae_vit.log"

# --- Baselines: roi_only ---
run_one "Baseline: ROI-only NeuroGraph" \
  bash -c "rm -f $LOG/roi_neurograph/optuna_study.db; python train_kfold_optuna_unified_v2.py --dataset adhd --mode roi_only \
  --config configs/adhd_roi_only.yaml $EXTRA \
  --log_dir $LOG/roi_neurograph --study_name smoke_neurograph" 2>&1 | tee "$LOG/roi_neurograph.log"

run_one "Baseline: ROI-only Chen2019" \
  python train_kfold_optuna_unified_v2.py --dataset adhd --mode roi_only \
  --config configs/adhd_roi_only_chen2019.yaml $EXTRA \
  --log_dir "$LOG/roi_chen2019" --study_name smoke_chen2019 2>&1 | tee "$LOG/roi_chen2019.log"

echo ""
echo "=============================================="
echo "=== All experiments finished. Logs under $LOG/ ==="
echo "=============================================="
