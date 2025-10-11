import torch
import torch.nn as nn


class ReLU2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).clamp(max=255.0).square()