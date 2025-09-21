import argparse
import pytest
from pathlib import Path
from unittest import mock

from gpt_lab.configuration import get_config


@pytest.fixture
def create_test_config(tmp_path: Path):
    """A pytest fixture to create a temporary YAML config file for tests."""
    config_content = """
    learning_rate: 0.001
    optimizer: adam
    model:
        name: transformer
        dim: 512
        layers: 6
    use_fp16: false
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return config_file


def test_basic_yaml_loading(create_test_config):
    """Tests that the config is loaded correctly from a YAML file."""
    parser = argparse.ArgumentParser()
    
    with mock.patch('sys.argv', ['test_script', '--config', str(create_test_config)]):
        config = get_config(parser)

    assert config.learning_rate == 0.001
    assert config.model.name == "transformer"
    assert config.model.dim == 512
    assert not config.use_fp16


def test_cli_override(create_test_config):
    """Tests that a flat CLI argument correctly overrides a YAML value."""
    parser = argparse.ArgumentParser()
    argv = ['script', '--config', str(create_test_config), '--learning_rate', '0.05']
    
    with mock.patch('sys.argv', argv):
        config = get_config(parser)

    assert config.learning_rate == 0.05
    assert config.optimizer == "adam"  # Ensure other values are untouched


def test_nested_cli_override(create_test_config):
    """Tests that a nested CLI argument with dot notation overrides a YAML value."""
    parser = argparse.ArgumentParser()
    argv = ['script', '--config', str(create_test_config), '--model.dim', '1024']

    with mock.patch('sys.argv', argv):
        config = get_config(parser)

    assert config.model.dim == 1024
    assert config.model.name == "transformer"  # Ensure other nested values are untouched


@pytest.mark.parametrize(
    "key, cli_value, expected_value",
    [
        ("epochs", "10", 10),
        ("dropout", "0.5", 0.5),
        ("use_amp", "true", True),
        ("use_amp", "YES", True),
        ("use_amp", "1", True),
        ("use_amp", "false", False),
        ("use_amp", "NO", False),
        ("use_amp", "0", False),
        ("optimizer", "sgd", "sgd"),
    ],
)
def test_cli_type_conversion(tmp_path: Path, key, cli_value, expected_value):
    """
    Uses parameterization to test automatic type conversion for int, float,
    and bool-like command-line arguments.
    """
    parser = argparse.ArgumentParser()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    
    argv = ['script', '--config', str(config_file), f'--{key}', cli_value]
    
    with mock.patch('sys.argv', argv):
        config = get_config(parser)

    assert getattr(config, key) == expected_value


def test_user_defined_args(create_test_config):
    """
    Tests that arguments added to the parser by the user are correctly
    integrated and can be overridden by the config file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float)

    # Case 1: Use defaults from the parser and the YAML file
    with mock.patch('sys.argv', ['script', '--config', str(create_test_config)]):
        config = get_config(parser)
    
    assert config.batch_size == 32       # From parser default
    assert config.learning_rate == 0.001  # From YAML

    # Case 2: Override all with CLI arguments
    argv = [
        'script', '--config', str(create_test_config), 
        '--batch_size', '128', 
        '--learning_rate', '0.1'
    ]
    with mock.patch('sys.argv', argv):
        config = get_config(parser)

    assert config.batch_size == 128      # From CLI
    assert config.learning_rate == 0.1    # From CLI


def test_user_defined_nested_dest(create_test_config):
    """
    Tests that user-defined arguments with nested `dest` attributes
    correctly create nested structures and override YAML values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dim",
        dest="model.dim",
        type=int,
        help="Override model dimension."
    )
    parser.add_argument(
        "--dropout",
        dest="model.dropout",
        type=float,
        default=0.1,
        help="Set model dropout."
    )

    # Case 1: Override existing key (dim) and add new key (dropout) from its default
    argv = ['script', '--config', str(create_test_config), '--dim', '1024']
    with mock.patch('sys.argv', argv):
        config = get_config(parser)

    assert config.model.dim == 1024      # Overridden by CLI
    assert config.model.dropout == 0.1   # Added from parser default
    assert config.model.layers == 6      # Preserved from YAML

    # Case 2: Override both from the command line
    argv = [
        'script', '--config', str(create_test_config),
        '--dim', '2048',
        '--dropout', '0.5'
    ]
    with mock.patch('sys.argv', argv):
        config = get_config(parser)

    assert config.model.dim == 2048      # Overridden by CLI
    assert config.model.dropout == 0.5   # Overridden by CLI
