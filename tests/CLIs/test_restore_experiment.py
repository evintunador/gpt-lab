import pytest

from CLIs.restore_experiment import create_storage_backend, restore_experiment
from gpt_lab.reproducibility import LocalFileSystemBackend


@pytest.mark.parametrize(
    "backend,expected_type",
    [
        ("local", LocalFileSystemBackend),
    ],
)
def test_create_storage_backend_local(tmp_path, backend, expected_type):
    """Tests creating a local storage backend."""
    storage_root = str(tmp_path / "storage")
    storage = create_storage_backend(backend, storage_root)
    assert isinstance(storage, expected_type)
    # The backend may convert to absolute path, so just check it ends with storage
    assert storage.root_dir.endswith("storage")


def test_create_storage_backend_unknown():
    """Tests that unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend: unknown"):
        create_storage_backend("unknown", "/path")


@pytest.mark.parametrize(
    "backend,expected_exception",
    [
        ("s3", ValueError),
        ("azure", ValueError),
        ("gcs", ValueError),
    ],
)
def test_create_storage_backend_unsupported(backend, expected_exception):
    """Tests that unsupported backends raise appropriate errors."""
    with pytest.raises(expected_exception, match=f"Unknown backend: {backend}"):
        create_storage_backend(backend, "/path")


def test_restore_experiment_calls_restore_state(monkeypatch):
    """Tests that restore_experiment calls restore_experiment_state with correct arguments."""
    called_with = []
    
    def mock_restore_state(experiment_id, storage_backend, restore_path):
        called_with.append({
            "experiment_id": experiment_id,
            "storage_backend": storage_backend,
            "restore_path": restore_path,
        })
    
    monkeypatch.setattr(
        "CLIs.restore_experiment.restore_experiment_state",
        mock_restore_state
    )
    
    restore_experiment(
        experiment_id="test-exp/2025-01-01_12-00-00_abc123",
        backend="local",
        restore_path="/tmp/restored",
        storage_root="/tmp/storage",
    )
    
    assert len(called_with) == 1
    assert called_with[0]["experiment_id"] == "test-exp/2025-01-01_12-00-00_abc123"
    assert called_with[0]["restore_path"] == "/tmp/restored"
    assert isinstance(called_with[0]["storage_backend"], LocalFileSystemBackend)


@pytest.mark.parametrize(
    "experiment_id,backend,restore_path",
    [
        ("exp1/2025-01-01_12-00-00_abc", "local", "restored"),
        ("exp2/2025-02-02_13-30-00_def", "local", "restored"),
        ("exp3/2025-03-03_14-45-00_ghi", "local", "restored_experiments"),
    ],
)
def test_restore_experiment_with_various_parameters(
    tmp_path, monkeypatch, experiment_id, backend, restore_path
):
    """Tests restore_experiment with various parameter combinations."""
    called = []
    storage_root = str(tmp_path / "storage")
    
    def mock_restore_state(experiment_id, storage_backend, restore_path):
        called.append(True)
    
    monkeypatch.setattr(
        "CLIs.restore_experiment.restore_experiment_state",
        mock_restore_state
    )
    
    restore_experiment(
        experiment_id=experiment_id,
        backend=backend,
        restore_path=restore_path,
        storage_root=storage_root,
    )
    
    assert len(called) == 1


def test_restore_experiment_default_parameters(monkeypatch):
    """Tests restore_experiment uses correct default parameters."""
    called_with = []
    
    def mock_restore_state(experiment_id, storage_backend, restore_path):
        called_with.append({
            "experiment_id": experiment_id,
            "storage_backend": storage_backend,
            "restore_path": restore_path,
        })
    
    monkeypatch.setattr(
        "CLIs.restore_experiment.restore_experiment_state",
        mock_restore_state
    )
    
    # Call with only required parameter
    restore_experiment(experiment_id="test-exp/2025-01-01_12-00-00_abc123")
    
    assert len(called_with) == 1
    assert called_with[0]["experiment_id"] == "test-exp/2025-01-01_12-00-00_abc123"
    assert called_with[0]["restore_path"] == "restored_experiments"
    assert isinstance(called_with[0]["storage_backend"], LocalFileSystemBackend)
    # The backend may convert to absolute path, so just check it contains the expected directory
    assert "experiment_artifacts" in called_with[0]["storage_backend"].root_dir


def test_restore_experiment_with_invalid_backend(monkeypatch):
    """Tests that invalid backend type raises error."""
    def mock_restore_state(experiment_id, storage_backend, restore_path):
        pass
    
    monkeypatch.setattr(
        "CLIs.restore_experiment.restore_experiment_state",
        mock_restore_state
    )
    
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        restore_experiment(
            experiment_id="test-exp/2025-01-01_12-00-00_abc123",
            backend="invalid",
        )

