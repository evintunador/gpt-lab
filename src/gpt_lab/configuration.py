import argparse
import yaml
import os
import sys
import logging
from typing import Dict, Any, Optional, Union, MutableMapping, List

logger = logging.getLogger(__name__)


class Config(dict):
    """
    A dictionary subclass that supports attribute-style access and automatic nested
    Config conversion.

    Key Features:
    - Attribute access: cfg.model.embed_dim equivalent to cfg['model']['embed_dim']
    - Nested instantiation: Assigning a dict to a key converts it to a Config
    - Dictionary compatibility: serialization, printing, and iteration work as expected
    """

    def __init__(
        self,
        mapping: Optional[Union[Dict, MutableMapping]] = None,
        **kwargs
    ):
        """
        Initialize the Config object.

        Args:
            mapping: Initial dictionary or mapping to populate the config.
            **kwargs: Additional key-value pairs to populate the config.
        """
        super().__init__()

        # Merge mapping and kwargs
        initial_data = {}
        if mapping is not None:
            initial_data.update(mapping)
        initial_data.update(kwargs)

        # Populate dict via __setitem__ to ensure recursive conversion
        for key, value in initial_data.items():
            self[key] = value

    def __setitem__(self, key: Any, value: Any):
        """Set item with recursive conversion of dicts to Configs."""
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    def __getattr__(self, name: str) -> Any:
        """
        Handle attribute access for keys.
        Prioritizes Dict keys (via self[name])
        """
        try:
            return self[name]
        except KeyError:
            pass

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """Enable setting dict keys via attributes."""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str):
        try:
            del self[name]
        except KeyError:
            super().__delattr__(name)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert Config to a standard dictionary."""
        result = {}
        for k, v in self.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def __str__(self) -> str:
        """Pretty print the config as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({super().__repr__()})"


def _str_to_bool(value):
    """Converts common boolean string representations to a boolean."""
    if isinstance(value, str):
        s_lower = value.lower()
        if s_lower in ('yes', 'true', 'on', '1', 't'):
            return True
        elif s_lower in ('no', 'false', 'off', '0', 'f'):
            return False
    return value


def load_config(path: str) -> Config:
    """
    Loads a YAML or JSON configuration file into a Config object.

    Args:
        path: Path to the YAML or JSON file.

    Returns:
        A Config object populated with data from the file.
    """
    import json

    def detect_format(file_path: str) -> str:
        """Detect file format by extension (.yaml/.yml/.json)"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in {'.yaml', '.yml'}:
            return 'yaml'
        elif ext == '.json':
            return 'json'
        else:
            # Attempt format detection by file contents if not clear from extension
            with open(file_path, 'r') as f:
                first_chars = f.read(128).lstrip()
                if first_chars.startswith('{') or first_chars.startswith('['):
                    return 'json'
                else:
                    return 'yaml'

    if path and os.path.exists(path):
        logger.info(f"Loading configuration from: {path}")
        file_format = detect_format(path)
        with open(path, 'r') as f:
            if file_format == 'json':
                data = json.load(f)
            else:
                data = yaml.safe_load(f) or {}
        return Config(data)
    
    if path:
        logger.warning(f"Config file specified but not found: {path}")
    
    return Config()


def _apply_override(config: Config, key: str, value: Any):
    """
    Applies a single key-value override to the config.
    Supports dot-notation for nested keys.
    """
    if '.' in key:
        # Dotted access - explicit path
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                # Create intermediate Configs if missing
                current[k] = Config()
            current = current[k]
            if not isinstance(current, MutableMapping):
                # We are traversing through a non-dict value, which is problematic.
                # We will treat it as a structure change and overwrite.
                # Note: This might lose data if the user didn't intend to overwrite.
                logger.warning(f"Overwriting non-dict value at path segment '{k}' with new Config structure.")
                current = Config()
        current[keys[-1]] = value
    else:
        # Standard behavior: update/create top-level key
        config[key] = value


def update_config_from_args(config: Config, args: List[str], argparse_namespace: Optional[argparse.Namespace] = None) -> Config:
    """
    Updates a Config object with CLI arguments.

    Args:
        config: The Config object to update.
        args: A list of strings (e.g. sys.argv[1:] or unknown args).
        argparse_namespace: Optional argparse Namespace containing known args.

    Returns:
        The updated Config object.
    """
    overrides = {}

    # 1. Process argparse Namespace (known args)
    if argparse_namespace:
        for key, value in vars(argparse_namespace).items():
            if key == 'config' or value is None:
                continue
            overrides[key] = value

    # 2. Process unknown args list (e.g. ["--model.dim=512", "--flag"])
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith(("-", "--")):
            key_val = arg.lstrip('-').split('=', 1)
            key = key_val[0]
            
            if len(key_val) > 1:
                value = key_val[1]
            else:
                # No '=' found. Check next arg if it looks like a value.
                if i + 1 < len(args) and not args[i+1].startswith('-'):
                    value = args[i+1]
                    i += 1
                else:
                    value = "true" # Treat as flag -> true

            # Type conversion
            try:
                value = int(value)
            except (ValueError, TypeError):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = _str_to_bool(value)
            
            overrides[key] = value
        i += 1

    # 3. Apply all overrides
    for key, value in overrides.items():
        _apply_override(config, key, value)
    
    if overrides:
        logger.info(f"Applied {len(overrides)} CLI overrides to configuration")
        logger.debug(f"CLI overrides: {overrides}")

    return config


def compose_config(
    parser: argparse.ArgumentParser
) -> Config:
    """
    Main entry point for configuration.
    1. Sets up --config argument.
    2. Loads YAML or JSON config.
    3. Applies CLI overrides.

    Args:
        parser: ArgumentParser to attach --config to.

    Returns:
        Final Config object.
    """
    # Ensure --config exists
    if not any(action.dest == 'config' for action in parser._actions):
        default_config_path = None
        try:
            main_script_path = os.path.abspath(sys.argv[0])
            caller_dir = os.path.dirname(main_script_path)
            # Prefer config.yaml, fallback to config.json if yaml does not exist
            potential_config_yaml = os.path.join(caller_dir, 'config.yaml')
            potential_config_json = os.path.join(caller_dir, 'config.json')

            if os.path.exists(potential_config_yaml):
                default_config_path = potential_config_yaml
            elif os.path.exists(potential_config_json):
                default_config_path = potential_config_json
        except Exception:
            pass

        parser.add_argument(
            "--config",
            type=str,
            default=default_config_path,
            required=default_config_path is None,
            help="Path to the YAML or JSON configuration file. Defaults to 'config.yaml' or 'config.json' in the script's directory."
        )

    # Parse known args (defined in parser) and unknown args (overrides)
    args, unknown = parser.parse_known_args()
    
    # Load base config (auto-detects yaml or json)
    config = load_config(args.config)

    # Update with CLI args
    # Pass 'args' (known) and 'unknown' (dynamic)
    update_config_from_args(config, unknown, argparse_namespace=args)

    return config
