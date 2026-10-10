"""Jumping-knowledge aggregators over layer outputs (ported from 2024 ``nn/jumping_know.py``).

Changes: ``WeightedSum`` uses ``torch.nn.Linear`` (the 2024 version used PyG's lazy
Linear); the 'lstm' mode still needs torch_geometric and is imported lazily.
"""
from __future__ import annotations

from torch import Tensor
from torch.nn import Linear, Module, MultiheadAttention


class SelfAttention(MultiheadAttention):
    def forward(self, xs: Tensor) -> Tensor:  # xs: (batch, hidden, phases)
        x = xs.transpose(2, int(self.batch_first))
        out: Tensor = super().forward(x, x, x, need_weights=True)[0]
        return out.mean(dim=int(self.batch_first))

    def reset_parameters(self):
        super()._reset_parameters()


class WeightedSum(Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim, self.num_heads = hidden_dim, num_heads
        self.Q = Linear(hidden_dim, num_heads, bias=False)
        if num_heads > 1:
            self.fc = Linear(num_heads, 1, bias=False)

    def forward(self, xs: Tensor) -> Tensor:  # xs: (batch, hidden, phases)
        H = xs.transpose(1, 2)  # (node, hop, dim)
        W = self.Q(H).softmax(dim=1)  # (node, hop, head)
        Z = H.transpose(1, 2).matmul(W)  # (node, dim, head)
        if self.num_heads > 1:
            Z = self.fc(Z)
        return Z.squeeze(-1)

    def reset_parameters(self):
        self.Q.reset_parameters()
        if self.num_heads > 1:
            self.fc.reset_parameters()


class JumpingKnowledge(Module):
    supported_modes = ["cat", "max", "lstm", "sum", "mean", "attn", "wsum"]

    def __init__(self, mode: str, **kwargs):
        super().__init__()
        if mode not in self.supported_modes:
            raise NotImplementedError(f"Unsupported JK mode: {mode}")
        self.mode = mode
        if mode == "attn":
            self.attn = SelfAttention(kwargs["hidden_dim"], num_heads=kwargs["num_heads"], batch_first=True)
        elif mode == "wsum":
            self.wsum = WeightedSum(kwargs["hidden_dim"], num_heads=kwargs["num_heads"])
        elif mode == "lstm":
            from torch_geometric.nn import JumpingKnowledge as JK

            self.lstm = JK(mode="lstm", **kwargs)

    def forward(self, xs: Tensor) -> Tensor:
        if self.mode == "cat":
            return xs.transpose(1, 2).reshape(xs.size(0), -1)
        if self.mode == "sum":
            return xs.sum(dim=-1)
        if self.mode == "mean":
            return xs.mean(dim=-1)
        if self.mode == "max":
            return xs.max(dim=-1)[0]
        if self.mode == "attn":
            return self.attn(xs)
        if self.mode == "wsum":
            return self.wsum(xs)
        return self.lstm(xs.unbind(dim=-1))

    def reset_parameters(self):
        for module in self.children():
            module.reset_parameters()
