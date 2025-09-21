import argparse
import yaml
import os
from collections.abc import Mapping


class ConfigObject(object):
    """
    Wraps a dictionary to provide attribute-style access, recursively
    turning nested dictionaries into ConfigObjects.
    """
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, Mapping):
                self.__dict__[key] = ConfigObject(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(f"'ConfigObject' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return f"ConfigObject({self.__dict__})"


def _str_to_bool(value):
    """Converts common boolean string representations to a boolean."""
    if isinstance(value, str):
        s_lower = value.lower()
        if s_lower in ('yes', 'true', 'on', '1', 't'):
            return True
        elif s_lower in ('no', 'false', 'off', '0', 'f'):
            return False
    return value


def _merge_dicts(base_dict, override_dict):
    """Recursively merges the override_dict into the base_dict."""
    for key, value in override_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _merge_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def get_config(parser: argparse.ArgumentParser) -> ConfigObject:
    """
    Builds a configuration object by merging a YAML file with CLI arguments.

    This function integrates with argparse to load a base configuration from
    a YAML file and then overrides it with any arguments specified on the
    command line. It supports dot-notation for overriding nested keys
    (e.g., --model.dim 512).

    Args:
        parser: An argparse.ArgumentParser instance.

    Returns:
        A ConfigObject containing the final, merged configuration.
    """
    if not any(action.dest == 'config' for action in parser._actions):
        parser.add_argument(
            "--config",
            type=str,
            help="Path to the YAML configuration file."
        )

    args, unknown = parser.parse_known_args()
    
    config_dict = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}

    cli_args = {}
    for arg in unknown:
        if arg.startswith(("-", "--")):
            key_val = arg.lstrip('-').split('=', 1)
            key = key_val[0]
            if len(key_val) > 1:
                value = key_val[1]
            else:
                # This simple parser assumes the next item in `unknown` is the value.
                # It is not robust to flags without values.
                idx = unknown.index(arg)
                if idx + 1 < len(unknown) and not unknown[idx+1].startswith('-'):
                    value = unknown[idx+1]
                else:
                    value = True # Treat as a flag
            
            try:
                value = int(value)
            except (ValueError, TypeError):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = _str_to_bool(value)
            
            keys = key.split('.')
            d = cli_args
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    for key, value in vars(args).items():
        if value is not None and key != 'config':
            keys = key.split('.')
            d = cli_args
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    final_config_dict = _merge_dicts(config_dict, cli_args)
    
    return ConfigObject(final_config_dict)
