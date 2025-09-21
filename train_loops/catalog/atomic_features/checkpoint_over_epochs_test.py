import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, call

from device import get_available_devices
from .checkpoint_over_epochs import run_training

AVAILABLE_DEVICES, _ = get_available_devices()

@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
@patch('train_loops.catalog.atomic_features.checkpoint_over_epochs.checkpointer.save_checkpoint')
def test_checkpointing_over_epochs(mock_save_checkpoint, device, tmp_path):
    """Test that checkpointing is triggered correctly every N epochs."""
    torch.manual_seed(0)
    X = torch.randn(20, 4).to(device)
    y = torch.randint(0, 2, (20,)).to(device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=10)

    model = nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    output_dir = tmp_path
    save_interval = 3
    num_epochs = 10

    run_training(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dl,
        save_every_epochs=save_interval,
        num_epochs=num_epochs,
        output_dir=output_dir,
    )

    # The logic saves on epoch 0, every `save_interval`, and the last epoch.
    # For 10 epochs and interval 3, it should save on epochs: 0, 3, 6, 9
    assert mock_save_checkpoint.call_count == 4
    
    expected_epochs = [0, 3, 6, 9]
    called_epochs = {c.kwargs['metadata']['epoch'] for c in mock_save_checkpoint.call_args_list}
    assert called_epochs == set(expected_epochs)

# Registry for discovery
__specific_tests__ = [
    test_checkpointing_over_epochs,
]
