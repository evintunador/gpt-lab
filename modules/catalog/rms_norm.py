import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return F.rms_norm(x, (x.size(-1),))