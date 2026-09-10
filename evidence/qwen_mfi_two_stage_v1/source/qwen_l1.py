"""Pinned model and unchanged adapter construction; no writer."""
import torch

MODEL = 'Qwen/Qwen3-0.6B'

REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'

ANCHORS = ['The capital of France is', 'Water freezes at', 'Two plus two equals']

class LoRAB(torch.nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.base = base
        self.register_buffer('A', torch.randn(rank, base.in_features) / base.in_features**0.5)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = 2.0

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
