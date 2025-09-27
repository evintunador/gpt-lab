import torch.nn as nn

from gpt_lab.nn_modules.catalog.backbones.nano_gpt import NanoGPTBackbone


class NanoGPTTrainingModel(nn.Module):
    """
    A "training model" that wraps a backbone nn.Module and handles the loss calculation.
    This allows the training loop to be agnostic to the loss function.
    """
    def __init__(self, backbone: NanoGPTBackbone):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, batch):
        """
        Calculates the loss for the given batch.
        The training loop will call this and then backpropagate the returned loss.
        """
        idx, targets = batch
        logits = self.backbone(idx)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return loss
