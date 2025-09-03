import importlib
import os
import sys
import os
from pathlib import Path
from typing import List, Union, Tuple, Any, Dict

def list_all_files_in_folder_and_subdirs(folder_path: str) -> List[str]:
    """
    Recursively list all .py files in the given folder and its subdirectories.
    Returns a list of file paths relative to the folder_path.
    """
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith('.py'):
                continue
            rel_dir = os.path.relpath(root, folder_path)
            if rel_dir == ".":
                rel_file = file
            else:
                rel_file = os.path.join(rel_dir, file)
            all_files.append(rel_file)
    return all_files


def import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class SkipModuleException(Exception):
    """A special exception used to signal that a module should be skipped during test discovery."""
    pass


def discover_dunder_objects(
        dunder: str, 
        object: Any,
        excluded_files: List[str] = [],
        search_folders: Union[None, str, List[str]] = None
    ) -> Tuple[List[Any], Dict[str, Exception]]:
    """
    Discover objects with a given dunder name in Python files within specified folders.

    Args:
        dunder: The dunder attribute name to look for (e.g., '__test_config__').
        object: The type or class to check isinstance(obj, object).
        excluded_files: List of filenames to exclude from search.
        search_folders: A folder path, or list of folder paths, to search. If None, uses the current directory.

    Returns:
        A tuple containing:
        - A list of discovered objects.
        - A dictionary mapping filenames to the exceptions that occurred while processing them.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Determine which folders to search
    if search_folders is None:
        folders_to_search = [current_dir]
    elif isinstance(search_folders, str):
        folders_to_search = [os.path.abspath(search_folders)]
    else:
        folders_to_search = [os.path.abspath(f) for f in search_folders]

    all_files = []
    for folder_path in folders_to_search:
        all_files.extend(list_all_files_in_folder_and_subdirs(folder_path))

    all_files = [f for f in all_files if os.path.basename(f) not in excluded_files]

    objects = []
    errors = {}
    for file in all_files:
        try:
            # Figure out the correct module name for importlib
            # file may be absolute or relative to project_root
            abs_file_path = os.path.abspath(os.path.join(folder_path if not os.path.isabs(file) else '', file))
            relative_path = os.path.relpath(abs_file_path, project_root)
            module_name = relative_path.replace('.py', '').replace(os.sep, '.')
            
            module = importlib.import_module(module_name)

            obj = getattr(module, dunder, None)
            if isinstance(obj, object):
                objects.append(obj)
        except SkipModuleException:
            # This is a graceful skip, not an error.
            continue
        except Exception as e:
            errors[file] = e

    return objects, errors