"""
ROI encoder: Chen et al. 2019-style DNN on single-atlas connectivity.
DOI: 10.1148/ryai.2019190012 — multichannel DNN; here we use single atlas:
  input (B, n_rois, n_rois) -> flatten upper triangle -> FC+BN+ReLU+Dropout -> feat_dim.
"""

import torch
import torch.nn as nn
from typing import Optional


class Chen2019ROIEncoder(nn.Module):
    """
    Single-atlas version of Chen et al. 2019: connectivity matrix -> flatten upper triangle
    -> FC + BN + ReLU + Dropout (×2) -> feat_dim. No multi-atlas; same DNN style.
    """

    def __init__(
        self,
        n_rois: int,
        feat_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5,
        hidden: int = 512,
        mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.feat_dim = feat_dim
        # Upper triangle (no diagonal): n_rois*(n_rois-1)//2
        self.input_dim = n_rois * (n_rois - 1) // 2

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_rois, n_rois)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.size(0)
        device = x.device
        # Upper triangle indices (offset=1, no diagonal)
        i, j = torch.triu_indices(self.n_rois, self.n_rois, offset=1, device=device)
        flat = x[:, i, j]  # (B, input_dim)
        return self.net(flat)
