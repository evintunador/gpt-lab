import argparse
import yaml
import os
import sys
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


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


def get_config(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    """
    Builds a configuration object by merging a YAML file with CLI arguments.

    This function integrates with argparse to load a base configuration from
    a YAML file and then overrides it with any arguments specified on the
    command line. It supports dot-notation for overriding nested keys
    (e.g., --model.dim 512).

    Args:
        parser: An argparse.ArgumentParser instance.

    Returns:
        A dictionary containing the final, merged configuration.
    """
    if not any(action.dest == 'config' for action in parser._actions):
        default_config_path = None
        try:
            # sys.argv[0] is the path to the script being executed.
            main_script_path = os.path.abspath(sys.argv[0])
            caller_dir = os.path.dirname(main_script_path)
            potential_config_path = os.path.join(caller_dir, 'config.yaml')
            if os.path.exists(potential_config_path):
                default_config_path = potential_config_path
        except Exception:
            pass  # Fails gracefully if path cannot be determined.

        parser.add_argument(
            "--config",
            type=str,
            default=default_config_path,
            required=default_config_path is None,
            help="Path to the YAML configuration file. Defaults to 'config.yaml' in the script's directory if available."
        )

    args, unknown = parser.parse_known_args()
    
    config_dict = {}
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        logger.debug(f"Loaded base config with {len(config_dict)} top-level keys")
    elif args.config:
        logger.warning(f"Config file specified but not found: {args.config}")

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
    
    if cli_args:
        logger.info(f"Applied {len(cli_args)} CLI overrides to configuration")
        logger.debug(f"CLI overrides: {cli_args}")
    
    return final_config_dict
