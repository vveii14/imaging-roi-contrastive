"""
ROI encoder using BrainNet-EndtoEnd ResidualGNNs (GCN-based).

设计目标：
- 完全删掉 NeuroGraph 依赖，只保留 BrainNet 这一条 ROI GNN 分支；
- 和原来的 NeuroGraphROIEncoder 一样，对外暴露一个 (B, n_rois, n_rois) -> (B, feat_dim) 的编码器，
  方便在 roi_only 和 fusion 模式下统一使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import types

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


# ---------------------------------------------------------------------------
# ResidualGNNs: inlined in this repo (models/brainnet_residual_gnn.py)
# ---------------------------------------------------------------------------
from .brainnet_residual_gnn import ResidualGNNs

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    SAGEConv,
    GraphConv,
    ChebConv,
)

_GNN_MAP = {
    "GCNConv": GCNConv,
    "GATConv": GATConv,
    "GINConv": GINConv,
    "SAGEConv": SAGEConv,
    "GraphConv": GraphConv,
    "ChebConv": ChebConv,
}


def _correlation_matrix_to_pyg_data(
    C: torch.Tensor,
    n_rois: int,
    edge_top_p: float = 0.05,
    device: Optional[torch.device] = None,
) -> Data:
    """(n_rois, n_rois) correlation matrix -> PyG Data (node features = rows of C, edges from top triu)."""
    device = device or C.device
    C = C.to(device).float()
    if C.dim() == 3:
        C = C.squeeze(0)
    x = C  # BrainNet 风格：节点特征使用整行相关向量
    triu_idx = torch.triu_indices(n_rois, n_rois, offset=1, device=device)
    vals = C[triu_idx[0], triu_idx[1]]
    k = max(1, int(vals.numel() * edge_top_p))
    _, top_idx = torch.topk(vals, k)
    edge_index = triu_idx[:, top_idx]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(x=x, edge_index=edge_index)


def _batch_roi_to_pyg(
    roi_batch: torch.Tensor,
    n_rois: int,
    edge_top_p: float = 0.05,
    device: Optional[torch.device] = None,
) -> Batch:
    """(B, n_rois, n_rois) 或 (B, n_rois) -> PyG Batch。3D 视为预计算好的相关矩阵。"""
    device = device or roi_batch.device
    B = roi_batch.size(0)
    graphs = []
    if roi_batch.dim() == 3:
        for i in range(B):
            d = _correlation_matrix_to_pyg_data(roi_batch[i], n_rois, edge_top_p, device)
            graphs.append(d)
    else:
        # 若输入是 (B, n_rois) 向量，则构造 C = v v^T
        for i in range(B):
            roi_vec = roi_batch[i].to(device).float().unsqueeze(0)
            C = roi_vec.T @ roi_vec
            C = (C + C.t()).clamp(-1, 1) * 0.5
            d = _correlation_matrix_to_pyg_data(C, n_rois, edge_top_p, device)
            graphs.append(d)
    return Batch.from_data_list(graphs)


class BrainNetROIEncoder(nn.Module):
    """
    Wrapper around BrainNet-EndtoEnd ResidualGNNs.
    - 输入: (B, n_rois, n_rois) Pearson 相关矩阵（或 (B, n_rois) 向量，会内部转为 C）。
    - 输出: (B, feat_dim) 图级表示，用于 roi_only 或 fusion。
    """

    def __init__(
        self,
        n_rois: int,
        feat_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5,
        hidden_channels: int = 32,
        hidden_mlp: int = 64,
        num_layers: int = 2,
        gnn_name: str = "GCNConv",
        edge_top_p: float = 0.05,
        mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.feat_dim = feat_dim
        self.edge_top_p = edge_top_p

        gnn_cls = _GNN_MAP.get(gnn_name)
        if gnn_cls is None:
            raise ValueError(f"Unknown gnn_name: {gnn_name}. Available: {list(_GNN_MAP.keys())}")

        _num_classes = num_classes  # avoid NameError in class Args body (RHS lookup)

        class Args:
            model = gnn_name
            num_classes = _num_classes

        # ResidualGNNs expects train_dataset[0].num_features
        class DummyDataset:
            def __init__(self, nf: int):
                self._nf = nf
            def __getitem__(self, i: int):
                if i != 0:
                    raise IndexError(i)
                return types.SimpleNamespace(num_features=self._nf)
            def __len__(self):
                return 1

        args = Args()
        dummy_dataset = DummyDataset(n_rois)

        # 直接复用 BrainNet-EndtoEnd 的 ResidualGNNs 结构
        self.gnn = ResidualGNNs(
            args,
            dummy_dataset,
            hidden_channels,
            hidden_mlp,
            num_layers,
            gnn_cls,
        )

        # 与旧 NeuroGraph 包装器类似：hook 住 MLP 最后一层的输入，作为 pre-logit 表示
        triu_len = (n_rois * (n_rois + 1)) // 2
        self.gnn.bn = nn.BatchNorm1d(triu_len)
        # 调整 mlp 的第一层输入维度：上三角特征 + 每层 hidden 的聚合
        self.gnn.mlp[0] = nn.Linear(triu_len + hidden_channels * num_layers, hidden_mlp)
        self._pre_logit_dim = hidden_mlp // 2
        self.proj = nn.Sequential(
            nn.Linear(self._pre_logit_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self._captured = None

    def _hook_capture(self, module, input):
        self._captured = input[0].detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        device = x.device
        # BatchNorm1d in BrainNet GNN requires batch_size > 1 in TRAINING mode.
        # In eval mode, running stats are used so batch_size=1 is fine.
        # Duplicate single-graph batches for training; also handle eval with bn workaround.
        single_graph = x.size(0) == 1
        if single_graph:
            x = x.repeat(2, *([1] * (x.dim() - 1)))
        batch = _batch_roi_to_pyg(x, self.n_rois, self.edge_top_p, device)
        batch = batch.to(device)
        self._captured = None
        handle = self.gnn.mlp[-1].register_forward_pre_hook(self._hook_capture)
        _ = self.gnn(batch)
        handle.remove()
        if self._captured is None:
            raise RuntimeError("BrainNet ROI encoder: pre-logit tensor not captured")
        out = self.proj(self._captured)
        if single_graph:
            out = out[:1]
        return out

