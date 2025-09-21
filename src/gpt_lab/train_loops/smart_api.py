"""
Smart training loop API that automatically selects and compiles atomic features
based on user-provided kwargs.

Usage:
    from train_loops.smart_api import smart_train
    
    result = smart_train(
        model=model,
        optimizer=optimizer, 
        loss_fn=loss_fn,
        train_loader=train_loader,
        # Any atomic feature kwargs
        accum_steps=4,
        val_loader=val_loader,
        patience=5,
        use_amp=True,
        num_epochs=3
    )
"""

import ast
import inspect
from pathlib import Path
from typing import Dict, Set, List, Tuple, Any, Optional
from collections import defaultdict

from gpt_lab.catalog_utils import import_module_from_path
from gpt_lab.llm_code_compiler import create_llm
from .llm_train_loop_compiler import compile_loop


def _get_atomic_features_dir() -> Path:
    """Get the path to the atomic features directory."""
    return Path(__file__).resolve().parent / "catalog" / "atomic_features"


def _parse_function_kwargs(func_node: ast.FunctionDef) -> Set[str]:
    """
    Parse an AST FunctionDef node to extract keyword-only arguments.
    Returns the set of kwarg names (excluding **kwargs).
    """
    kwargs = set()
    
    # Get keyword-only arguments (after *)
    for arg in func_node.args.kwonlyargs:
        kwargs.add(arg.arg)
    
    return kwargs


def _parse_file_for_kwargs(file_path: Path) -> Set[str]:
    """
    Parse a Python file to extract kwargs from its run_training function.
    Returns empty set if no run_training function found or parsing fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef) and 
                node.name == "run_training"):
                return _parse_function_kwargs(node)
        
        return set()
    
    except Exception as e:
        print(f"Warning: Failed to parse {file_path}: {e}")
        return set()


def _load_feature_metadata(feature_name: str) -> Dict[str, Any]:
    """
    Load optional metadata from an atomic feature file.
    
    Args:
        feature_name: Name of the atomic feature (without .py extension)
        
    Returns:
        Dictionary containing metadata, or empty dict if none found
    """
    try:
        atomic_features_dir = _get_atomic_features_dir()
        feature_path = atomic_features_dir / f"{feature_name}.py"
        
        if not feature_path.exists():
            return {}
        
        # Import the module to access its metadata
        feature_module = import_module_from_path(f"metadata_{feature_name}", feature_path)
        
        # Look for metadata
        metadata = getattr(feature_module, '__smart_train_metadata__', {})
        return metadata if isinstance(metadata, dict) else {}
        
    except Exception as e:
        print(f"Warning: Failed to load metadata for {feature_name}: {e}")
        return {}


def _check_feature_conflicts(selected_features: List[str]) -> None:
    """
    Check for conflicts between selected features and raise error if found.
    
    Args:
        selected_features: List of feature names to check
        
    Raises:
        ValueError: If conflicting features are detected
    """
    if len(selected_features) <= 1:
        return  # No conflicts possible with 0 or 1 feature
    
    # Load metadata for all features
    feature_metadata = {}
    for feature in selected_features:
        metadata = _load_feature_metadata(feature)
        if metadata:
            feature_metadata[feature] = metadata
    
    # Check for conflicts
    conflicts_found = []
    
    for feature, metadata in feature_metadata.items():
        conflicts_with = metadata.get('conflicts_with', [])
        if not conflicts_with:
            continue
            
        # Check if any conflicting features are in the selected list
        for conflict_feature in conflicts_with:
            if conflict_feature in selected_features:
                conflicts_found.append((feature, conflict_feature))
    
    if conflicts_found:
        # Format error message
        conflict_pairs = [f"'{feat1}' conflicts with '{feat2}'" for feat1, feat2 in conflicts_found]
        raise ValueError(
            f"Feature conflicts detected: {', '.join(conflict_pairs)}. "
            f"These features cannot be used together in the same training loop."
        )


def visual_test_conflict_detection():
    """Test the conflict detection system."""
    print("\n" + "=" * 80)
    print("TESTING CONFLICT DETECTION")
    print("=" * 80)
    
    # Test case that should trigger conflict
    conflict_kwargs = {"norm_clip_value": 1.0, "elem_grad_clip": 0.5}
    
    print(f"\nTesting conflict case: {conflict_kwargs}")
    print("-" * 60)
    try:
        selected = select_features_from_kwargs(conflict_kwargs)
        print(f"Selected features: {selected}")
        
        # Check conflicts manually
        _check_feature_conflicts(selected)
        print("❌ ERROR: Conflict should have been detected!")
        
    except ValueError as e:
        print(f"✅ Conflict correctly detected: {e}")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")
    
    print("\n" + "=" * 80)
    print("Conflict detection test complete!")


def visual_test_feature_selection():
    """Test the feature selection logic with various scenarios."""
    test_cases = [
        # Basic single feature cases
        {"accum_steps": 4},
        
        # Overlapping kwargs - validation only (most specific subset)
        {"val_loader": "some_loader"},
        {"val_loader": "some_loader", "val_interval": 5},
        
        # Overlapping kwargs - early stopping only (has unique kwargs)
        {"patience": 3},
        {"patience": 5, "min_delta": 0.01},
        
        # Overlapping kwargs - early stopping (user provides early_stopping specific kwargs)
        {"val_loader": "some_loader", "patience": 3},
        {"val_loader": "some_loader", "val_interval": 5, "patience": 3, "min_delta": 0.01},
        
        # Multiple non-overlapping features
        {"accum_steps": 4, "norm_clip_value": 1.0, "track_loss": True},
        
        # Complex mixed case
        {"accum_steps": 4, "val_loader": "loader", "patience": 5, "use_amp": True, "num_epochs": 3},
        
        # Test conflict detection
        {"norm_clip_value": 1.0, "elem_grad_clip": 0.5},
    ]
    
    print("=" * 80)
    print("TESTING FEATURE SELECTION ALGORITHM")
    print("=" * 80)
    
    for i, kwargs in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {kwargs}")
        print("-" * 60)
        try:
            selected = select_features_from_kwargs(kwargs)
            print(f"Selected features: {selected}")
            
            # Quick validation of the logic
            feature_to_kwargs, _ = discover_atomic_feature_mappings()
            for feature in selected:
                feature_kwargs = feature_to_kwargs[feature]
                matched = feature_kwargs & set(kwargs.keys())
                print(f"  {feature}: matched {sorted(matched)}")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("Test complete!")


def discover_atomic_feature_mappings() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Discover all atomic features and create bidirectional mappings.
    
    Returns:
        tuple: (feature_to_kwargs, kwarg_to_features)
            - feature_to_kwargs: Maps feature name to set of its kwargs
            - kwarg_to_features: Maps kwarg name to set of features that use it
    """
    atomic_features_dir = _get_atomic_features_dir()
    
    feature_to_kwargs: Dict[str, Set[str]] = {}
    kwarg_to_features: Dict[str, Set[str]] = defaultdict(set)
    
    # Discover all atomic feature files
    for feature_file in atomic_features_dir.glob("*.py"):
        # Skip test files, __init__.py, and base_loop.py
        if (feature_file.name.endswith("_test.py") or 
            feature_file.name == "__init__.py" or
            feature_file.name == "base_loop.py"):
            continue
        
        feature_name = feature_file.stem
        
        # Parse kwargs from the file
        kwargs = _parse_file_for_kwargs(feature_file)
        
        # Only include features that have a run_training function with kwargs
        if kwargs:
            feature_to_kwargs[feature_name] = kwargs
            
            # Build reverse mapping
            for kwarg in kwargs:
                kwarg_to_features[kwarg].add(feature_name)
    
    # Convert defaultdict to regular dict for cleaner output
    kwarg_to_features = dict(kwarg_to_features)
    
    return feature_to_kwargs, kwarg_to_features


def visual_test_discovered_mappings():
    """
    Print all discovered features and kwargs for visual validation.
    """
    feature_to_kwargs, kwarg_to_features = discover_atomic_feature_mappings()
    
    print("=" * 60)
    print("DISCOVERED ATOMIC FEATURES AND KWARGS")
    print("=" * 60)
    
    print(f"\nFound {len(feature_to_kwargs)} atomic features:\n")
    
    for feature_name in sorted(feature_to_kwargs.keys()):
        kwargs = feature_to_kwargs[feature_name]
        print(f"📦 {feature_name}:")
        if kwargs:
            for kwarg in sorted(kwargs):
                print(f"    - {kwarg}")
        else:
            print("    (no kwargs)")
        print()
    
    print("=" * 60)
    print("KWARG TO FEATURES MAPPING")
    print("=" * 60)
    
    print(f"\nFound {len(kwarg_to_features)} unique kwargs:\n")
    
    for kwarg in sorted(kwarg_to_features.keys()):
        features = kwarg_to_features[kwarg]
        print(f"🔑 {kwarg}:")
        for feature in sorted(features):
            print(f"    - {feature}")
        print()
    
    # Find overlapping kwargs (used by multiple features)
    overlapping_kwargs = {k: v for k, v in kwarg_to_features.items() if len(v) > 1}
    
    if overlapping_kwargs:
        print("=" * 60)
        print("OVERLAPPING KWARGS (Multiple Features)")
        print("=" * 60)
        print()
        
        for kwarg in sorted(overlapping_kwargs.keys()):
            features = overlapping_kwargs[kwarg]
            print(f"⚠️  {kwarg} used by {len(features)} features: {', '.join(sorted(features))}")
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total atomic features: {len(feature_to_kwargs)}")
    print(f"Total unique kwargs: {len(kwarg_to_features)}")
    print(f"Overlapping kwargs: {len(overlapping_kwargs)}")
    print("=" * 60)


def _find_overlapping_feature_groups(candidate_features: Set[str], feature_to_kwargs: Dict[str, Set[str]]) -> List[Set[str]]:
    """
    Group features that share kwargs into overlapping groups.
    Features in the same group compete with each other for selection.
    """
    # Build a graph of which features share kwargs
    feature_kwargs_map = {f: feature_to_kwargs[f] for f in candidate_features}
    
    groups = []
    remaining_features = set(candidate_features)
    
    while remaining_features:
        # Start a new group with an arbitrary remaining feature
        current_feature = remaining_features.pop()
        current_group = {current_feature}
        current_kwargs = feature_kwargs_map[current_feature]
        
        # Find all features that share any kwargs with this group
        changed = True
        while changed:
            changed = False
            to_remove = set()
            
            for feature in remaining_features:
                if feature_kwargs_map[feature] & current_kwargs:
                    # This feature shares kwargs with the current group
                    current_group.add(feature)
                    current_kwargs.update(feature_kwargs_map[feature])
                    to_remove.add(feature)
                    changed = True
            
            remaining_features -= to_remove
        
        groups.append(current_group)
    
    return groups


def _select_most_specific_from_group(group: Set[str], feature_to_kwargs: Dict[str, Set[str]], user_kwarg_set: Set[str]) -> List[str]:
    """
    From a group of overlapping features, select the appropriate features.
    
    Simple Algorithm:
    1. Find all satisfiable features (user provided at least one kwarg)
    2. Check if user provided any kwargs that are unique to specific features
    3. If yes: include features with unique kwargs + most specific shared feature
    4. If no: include only the most specific feature (smallest kwarg set)
    """
    # Find features where user provided at least one kwarg
    satisfiable_features = []
    for feature in group:
        feature_kwargs = feature_to_kwargs[feature]
        matched_kwargs = feature_kwargs & user_kwarg_set
        
        if matched_kwargs:
            satisfiable_features.append((feature, feature_kwargs, matched_kwargs))
    
    if not satisfiable_features:
        return []
    
    # Count how many satisfiable features use each user kwarg
    kwarg_counts = {}
    for user_kwarg in user_kwarg_set:
        count = sum(1 for _, feature_kwargs, _ in satisfiable_features if user_kwarg in feature_kwargs)
        if count > 0:  # Only count kwargs that are actually used by features in this group
            kwarg_counts[user_kwarg] = count
    
    # Check if user provided any unique kwargs (used by only one feature)
    has_unique_kwargs = any(count == 1 for count in kwarg_counts.values())
    
    if has_unique_kwargs:
        # User provided unique kwargs - include features with unique kwargs + shared features
        selected = set()
        
        # Add features that have unique kwargs
        for feature, feature_kwargs, matched_kwargs in satisfiable_features:
            has_unique = any(kwarg_counts.get(kwarg, 0) == 1 for kwarg in matched_kwargs)
            if has_unique:
                selected.add(feature)
        
        # Also add the most specific feature that only uses shared kwargs
        shared_only_features = []
        for feature, feature_kwargs, matched_kwargs in satisfiable_features:
            has_unique = any(kwarg_counts.get(kwarg, 0) == 1 for kwarg in matched_kwargs)
            if not has_unique:
                shared_only_features.append((feature, feature_kwargs, matched_kwargs))
        
        if shared_only_features:
            # Find most specific (smallest kwarg set)
            min_size = min(len(fk) for _, fk, _ in shared_only_features)
            most_specific = [f for f, fk, _ in shared_only_features if len(fk) == min_size]
            selected.update(most_specific)
        
        return sorted(list(selected))
    
    else:
        # No unique kwargs - select only the most specific feature (smallest total kwarg set)
        min_kwarg_count = min(len(feature_kwargs) for _, feature_kwargs, _ in satisfiable_features)
        most_specific = [f for f, fk, _ in satisfiable_features if len(fk) == min_kwarg_count]
        return sorted(most_specific)


def select_features_from_kwargs(user_kwargs: Dict[str, Any]) -> List[str]:
    """
    Select atomic features based on user-provided kwargs using subset-based specificity logic.
    
    For overlapping features (features that share kwargs), prefers the most specific feature:
    - The feature with the smallest kwarg set that the user has satisfied
    - This ensures we pick 'validation' over 'early_stopping' when user only provides val_loader
    
    Args:
        user_kwargs: Dictionary of kwargs provided by the user
        
    Returns:
        List of feature names to include in the compiled loop
        
    Raises:
        ValueError: If user provides kwargs that don't match any known features
    """
    feature_to_kwargs, kwarg_to_features = discover_atomic_feature_mappings()
    
    # Find all user kwargs that match known feature kwargs
    user_kwarg_set = set(user_kwargs.keys())
    known_kwargs = set(kwarg_to_features.keys())
    
    # Check for unknown kwargs
    unknown_kwargs = user_kwarg_set - known_kwargs
    if unknown_kwargs:
        available_kwargs = sorted(known_kwargs)
        raise ValueError(
            f"Unknown kwargs provided: {sorted(unknown_kwargs)}. "
            f"Available kwargs: {available_kwargs}"
        )
    
    # Find features that have any overlap with user kwargs
    candidate_features = set()
    for user_kwarg in user_kwarg_set:
        if user_kwarg in kwarg_to_features:
            candidate_features.update(kwarg_to_features[user_kwarg])
    
    if not candidate_features:
        return []
    
    # Group overlapping features
    overlapping_groups = _find_overlapping_feature_groups(candidate_features, feature_to_kwargs)
    
    # Select most specific feature from each group
    selected_features = []
    for group in overlapping_groups:
        group_selection = _select_most_specific_from_group(group, feature_to_kwargs, user_kwarg_set)
        selected_features.extend(group_selection)
    
    return sorted(selected_features)


def smart_train(
    model,
    optimizer, 
    loss_fn,
    train_loader,
    *,
    llm_compiler_model="anthropic/claude-3-5-sonnet-20240620",
    api_key=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Smart training function that automatically selects and compiles atomic features
    based on the provided kwargs, then executes the compiled training loop.
    
    Args:
        model: PyTorch model to train
        optimizer: PyTorch optimizer
        loss_fn: Loss function  
        train_loader: Training data loader
        llm_compiler_model: "provider/model_name" for LLM code compiler (Default "anthropic/claude-3-5-sonnet-20240620")
        api_key: API key for LLM provider (Default: None - automatically checks .env file)
        **kwargs: Any atomic feature arguments (e.g., accum_steps, val_loader, patience, etc.)
        
    Returns:
        Dict containing training results (at minimum {'model': nn.Module})
        
    Raises:
        ValueError: If unknown kwargs are provided or compilation fails
        
    Examples:
        # Simple training with gradient accumulation
        result = smart_train(model, optimizer, loss_fn, train_loader, accum_steps=4)
        
        # Training with validation and early stopping  
        result = smart_train(
            model, optimizer, loss_fn, train_loader,
            val_loader=val_loader, patience=5, min_delta=0.01
        )
        
        # Complex training with multiple features
        result = smart_train(
            model, optimizer, loss_fn, train_loader,
            accum_steps=4, val_loader=val_loader, patience=3, 
            use_amp=True, num_epochs=5, lr_scheduler_type="cosine"
        )
    """
    # Filter out None values from kwargs (common pattern in ML)
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    if not filtered_kwargs:
        # No additional features requested - use base training loop
        from train_loops.catalog.atomic_features.base_loop import run_training
        return run_training(model, optimizer, loss_fn, train_loader)
    
    # Select appropriate atomic features based on kwargs
    selected_features = select_features_from_kwargs(filtered_kwargs)
    
    if not selected_features:
        # No features selected (shouldn't happen if kwargs are valid, but safety check)
        from train_loops.catalog.atomic_features.base_loop import run_training
        return run_training(model, optimizer, loss_fn, train_loader)
    
    # Check for feature conflicts
    _check_feature_conflicts(selected_features)
    
    print(f"[smart_train] Selected features: {selected_features}")
    
    # Optimization: For single atomic features, use them directly
    if len(selected_features) == 1:
        feature_name = selected_features[0]
        print(f"[smart_train] Single feature optimization: using {feature_name}.py directly")
        
        try:
            # Load the atomic feature directly
            atomic_features_dir = _get_atomic_features_dir()
            feature_path = atomic_features_dir / f"{feature_name}.py"
            
            if not feature_path.exists():
                raise FileNotFoundError(f"Atomic feature file not found: {feature_path}")
            
            atomic_module = import_module_from_path(f"direct_{feature_name}", feature_path)
            atomic_run_training = atomic_module.run_training
            
        except Exception as e:
            raise RuntimeError(f"Failed to load atomic feature {feature_name} directly: {e}")
            
        print(f"[smart_train] Using atomic feature directly: {feature_path}")
        
        # Execute the atomic feature directly with user's kwargs
        return atomic_run_training(model, optimizer, loss_fn, train_loader, **filtered_kwargs)
    
    # instantiate llm code compiler
    llm = create_llm(model=llm_compiler_model, api_key=api_key)

    # create the training loop code
    compilation_result = compile_loop(selected_features, llm=llm)
    compiled_module_path = compilation_result["code_path"]
    
    # Load the compiled training function
    compiled_module = import_module_from_path("smart_compiled_loop", compiled_module_path)
    compiled_run_training = compiled_module.run_training
    
    print(f"[smart_train] Using compiled loop: {compiled_module_path}")
    
    # Execute the compiled training loop with user's kwargs
    return compiled_run_training(model, optimizer, loss_fn, train_loader, **filtered_kwargs)


if __name__ == "__main__":
    print_discovered_mappings()
    test_feature_selection()
    test_conflict_detection()