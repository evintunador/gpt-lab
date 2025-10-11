import torch
from torch import nn, Tensor
import torch.nn.functional as F

from gpt_lab.nn_modules.backbone import ModdedNanoGPTBackbone


class ModdedNanoGPTTrainingModel(nn.Module):
    """
    A "training model" that wraps a backbone nn.Module and handles the loss calculation.
    This allows the training loop to be agnostic to the loss function.
    """
    def __init__(self, backbone: ModdedNanoGPTBackbone):
        super().__init__()
        self.backbone = backbone
    def forward(
        self, 
        batch: Tensor, # (1, B*N)
    ):
        batch = batch.squeeze(0) # Remove batch dim that was put there by the default collate function
        inputs = batch[:-1]
        targets = batch[1:]
        logits = self.backbone(inputs)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets, 
            reduction = 'sum' if self.training else 'mean'
        )