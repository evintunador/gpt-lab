from typing import List, Union, Tuple, Any
import math

import torch
import torch.nn as nn

from modules.base_test_bench_utils import (
    ModuleTestConfig, 
    BenchmarkConfig, 
    Competitor,
    TensorParallelConfig
)


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################


class ReLU2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).clamp(max=255.0).square()