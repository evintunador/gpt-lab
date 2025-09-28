import torch.nn as nn

from gpt_lab.device import get_available_devices

class SimpleTestTrainingModel(nn.Module):
    """A simple nn.Module wrapper for testing that acts as a TrainingModel."""
    def __init__(self, backbone, loss_fn):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def forward(self, batch):
        xb, yb = batch
        logits = self.backbone(xb)
        return self.loss_fn(logits, yb)

# Discover available devices once and export for all tests to use
AVAILABLE_DEVICES, _ = get_available_devices()