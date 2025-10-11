import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, call

from gpt_lab.train_loops.checkpoint_over_epochs import run_training
from gpt_lab.train_loops.tests.test_utils import SimpleTestTrainingModel, AVAILABLE_DEVICES


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
@patch('gpt_lab.checkpointer.save_checkpoint')
def test_checkpointing_over_epochs(mock_save_checkpoint, device, tmp_path):
    """Test that checkpointing is triggered correctly every N epochs."""
    torch.manual_seed(0)
    X = torch.randn(20, 4).to(device)
    y = torch.randint(0, 2, (20,)).to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=10)

    backbone = nn.Linear(4, 2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    model = SimpleTestTrainingModel(backbone, loss_fn).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    output_dir = tmp_path
    save_interval = 3
    num_epochs = 10

    run_training(
        model=model,
        optimizer=optimizer,
        train_loader=dl,
        save_every_epochs=save_interval,
        num_epochs=num_epochs,
        output_dir=output_dir,
    )

    # The logic saves before training, on epoch 0, every `save_interval`, and the last epoch.
    # For 10 epochs and interval 3, it should save on epochs: -1, 0, 3, 6, 9
    assert mock_save_checkpoint.call_count == 5
    
    expected_epochs = [-1, 0, 3, 6, 9]
    called_epochs = {c.kwargs['metadata']['epoch'] for c in mock_save_checkpoint.call_args_list}
    assert called_epochs == set(expected_epochs)

# Registry for discovery
__specific_tests__ = [
    test_checkpointing_over_epochs,
]
