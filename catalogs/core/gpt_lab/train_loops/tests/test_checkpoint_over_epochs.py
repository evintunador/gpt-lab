import pytest
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, call, MagicMock
from importlib import reload

from gpt_lab.train_loops.compiler_validation import SimpleTestTrainingModel, AVAILABLE_DEVICES


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_checkpointing_over_epochs(device, tmp_path, monkeypatch):
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

    # Mock save_checkpoint to prevent actual file saving
    mock_save_checkpoint = MagicMock()
    monkeypatch.setattr("gpt_lab.checkpointer.save_checkpoint", mock_save_checkpoint)
    
    # Reload the module to pick up the patched function
    if 'gpt_lab.train_loops.checkpoint_over_epochs' in sys.modules:
        import gpt_lab.train_loops.checkpoint_over_epochs as checkpoint_module
        reload(checkpoint_module)
    
    from gpt_lab.train_loops.checkpoint_over_epochs import run_training
    
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
    expected_calls = [
        call(
            filepath=str(output_dir / "checkpoints" / f"epoch_{epoch}.pt"),
            metadata={"epoch": epoch, "config": {}},
            model=model,
            optimizer=optimizer
        ) for epoch in expected_epochs
    ]
    mock_save_checkpoint.assert_has_calls(expected_calls, any_order=True)

# Registry for discovery
__specific_tests__ = [
    test_checkpointing_over_epochs,
]
