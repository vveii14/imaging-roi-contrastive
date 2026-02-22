"""
Create minimal ADHD-format data under data/adhd/ for smoke runs.
- 5 folds: adhd_fold1_775.npy ... adhd_fold5_775.npy (each SAMPLES_PER_FOLD for train/val).
- roi_matrices_775.npy: (N_train, 116, 116) for ROI-based encoders.
- Test set (5 samples per fold, 25 total): adhd_test_775.npy, test_roi_matrices_775.npy.
  Full ADHD has 155 samples per fold; we provide 5 per fold in repo for evaluation.
Run from repo root: python scripts/prepare_dummy_adhd.py
"""

import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADHD_DIR = REPO_ROOT / "data" / "adhd"
SHAPE = (96, 96, 96)
N_ROIS = 116
SAMPLES_PER_FOLD = 4  # train/val per fold for smoke (2 per class)
TEST_SAMPLES_PER_FOLD = 5  # 5 per fold in repo for test (full ADHD has 155/fold)
N_FOLDS = 5


def _make_sample(rng, label: int):
    raw_img = rng.standard_normal(SHAPE).astype(np.float32) * 0.5
    aal_mask = np.zeros(SHAPE, dtype=np.float32)
    for roi in range(1, N_ROIS + 1):
        aal_mask.ravel()[roi - 1 : roi] = roi
    return {"raw_img_np": raw_img, "raw_img_aal_mask_np": aal_mask, "label": label}


def main():
    ADHD_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Train/val folds (for smoke)
    for fold in range(1, N_FOLDS + 1):
        rows = []
        for i in range(SAMPLES_PER_FOLD):
            label = i % 2
            rows.append(_make_sample(rng, label))
        path = ADHD_DIR / f"adhd_fold{fold}_775.npy"
        np.save(path, np.array(rows, dtype=object), allow_pickle=True)
        print(f"Wrote {path} ({SAMPLES_PER_FOLD} samples)")

    n_train = N_FOLDS * SAMPLES_PER_FOLD
    roi_mat = np.zeros((n_train, N_ROIS, N_ROIS), dtype=np.float32)
    for i in range(n_train):
        v = rng.standard_normal(N_ROIS).astype(np.float32)
        m = np.outer(v, v)
        m = (m + m.T) * 0.5
        np.fill_diagonal(m, 1.0)
        roi_mat[i] = np.clip(m, -1, 1)
    path = ADHD_DIR / "roi_matrices_775.npy"
    np.save(path, roi_mat)
    print(f"Wrote {path} shape {roi_mat.shape}")

    # Test set: 5 samples per fold (25 total), committed to repo for evaluation
    test_rows = []
    for fold in range(1, N_FOLDS + 1):
        for i in range(TEST_SAMPLES_PER_FOLD):
            label = i % 2
            test_rows.append(_make_sample(rng, label))
    test_arr = np.array(test_rows, dtype=object)
    path_test = ADHD_DIR / "adhd_test_775.npy"
    np.save(path_test, test_arr, allow_pickle=True)
    print(f"Wrote {path_test} ({len(test_rows)} test samples, 5 per fold)")

    test_roi = np.zeros((len(test_rows), N_ROIS, N_ROIS), dtype=np.float32)
    for i in range(len(test_rows)):
        v = rng.standard_normal(N_ROIS).astype(np.float32)
        m = np.outer(v, v)
        m = (m + m.T) * 0.5
        np.fill_diagonal(m, 1.0)
        test_roi[i] = np.clip(m, -1, 1)
    path_test_roi = ADHD_DIR / "test_roi_matrices_775.npy"
    np.save(path_test_roi, test_roi)
    print(f"Wrote {path_test_roi} shape {test_roi.shape}")

    print("Done. Test set: 5 samples per fold (25 total) in adhd_test_775.npy, test_roi_matrices_775.npy.")
    print("Smoke run: python train_kfold_optuna_unified_v2.py --dataset adhd --mode fusion --fusion contrastive \\")
    print("  --config configs/adhd_fusion_3dsctf_contrastive.yaml --n_trials 1 --max_folds 1 --epochs 2 --log_dir logs_smoke")


if __name__ == "__main__":
    main()
