from typing import List, Dict, Any, Union, Sequence, Callable
import copy
import os
import shutil

import pytest
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from modules.base_test_bench_utils import (
    ModuleTestConfig, 
    get_available_devices,
    discover_dunder_objects,
    get_total_loss,
)


@pytest.fixture(scope="session", autouse=True)
def clear_test_error_maps():
    """
    Clears the test error maps directory before the test session starts.
    """
    folder = "modules/test_error_maps"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def _save_heatmaps(
    torch_tensor: torch.Tensor,
    competitor_tensor: torch.Tensor,
    test_name: str,
    folder: str,
    atol: float,
    rtol: float,
    group_name: str
):
    """
    Saves multiple sets of heatmaps comparing torch_tensor and competitor_tensor:
      1) Raw absolute differences
      2) Absolute tolerance failure mask (where abs diff > atol)
      3) Relative tolerance failure mask (where abs diff > rtol * abs(expected))
    """
    # Convert to numpy arrays
    actual = competitor_tensor.detach().to(torch.float32).cpu().numpy()
    expected = torch_tensor.detach().to(torch.float32).cpu().numpy()

    # Compute differences and masks
    abs_diff = np.abs(expected - actual)
    abs_threshold = atol
    rel_threshold = rtol * np.abs(expected)

    abs_fail_mask = (abs_diff > abs_threshold).astype(np.int32)
    rel_fail_mask = (abs_diff > rel_threshold).astype(np.int32)

    def save_figure(matrix, original_ndim: int, title: str, filename: str, cmap: str = "hot"):
        plt.figure(figsize=(8, 6))

        matrix_to_show = np.atleast_2d(matrix)

        plt.imshow(matrix_to_show, cmap=cmap, aspect="auto")
        plt.title(title)
        
        if original_ndim > 1:
            plt.xlabel(f"Dimension {original_ndim - 1}")
            if matrix_to_show.shape[0] > 1:
                 plt.ylabel(f"Dimension {original_ndim - 2}")
            else: # It's a 1D array that has been reshaped to (1, N)
                 plt.ylabel("")
                 plt.yticks([])
        else: # 1D tensor originally
            plt.xlabel("Dimension 0")
            plt.ylabel("")
            plt.yticks([])

        plt.colorbar()
        plt.savefig(os.path.join(folder, filename))
        plt.close()

    def save_all_figures(diff: np.ndarray, abs_mask: np.ndarray, rel_mask: np.ndarray, 
                        suffix: str, filename_suffix: str, original_ndim: int):
        # Raw difference
        save_figure(diff, original_ndim, f"{test_name} {suffix} - raw diff ({group_name})",
                   f"{test_name}_{filename_suffix}_raw_diff_{group_name}.png")
        # Absolute tolerance failures
        save_figure(abs_mask, original_ndim, f"{test_name} {suffix} - abs failure mask ({group_name})",
                   f"{test_name}_{filename_suffix}_abs_fail_{group_name}.png", cmap="Reds")
        # Relative tolerance failures
        save_figure(rel_mask, original_ndim, f"{test_name} {suffix} - rel failure mask ({group_name})",
                   f"{test_name}_{filename_suffix}_rel_fail_{group_name}.png", cmap="Reds")

    # Generic handling of tensor dimensions
    original_ndim = expected.ndim
    if original_ndim <= 2:
        save_all_figures(abs_diff, abs_fail_mask, rel_fail_mask, "diff", "diff", original_ndim)
    else:
        # Iterate over all dimensions except the last two
        slice_indices = expected.shape[:-2]
        for index in np.ndindex(slice_indices):
            # Make the slice representation cleaner for titles
            slice_repr = str(index)
            if len(index) == 1:
                slice_repr = f"[{index[0]}]"

            suffix = f"diff for slice at index {slice_repr}"
            filename_suffix = f"diff_slice_{'_'.join(map(str, index))}"
            save_all_figures(
                abs_diff[index], abs_fail_mask[index], rel_fail_mask[index],
                suffix, filename_suffix, original_ndim
            )


def _assert_tensors_close_and_generate_heatmaps(
    tensor_groups: Dict[str, Dict[str, Union[List[torch.Tensor], List[str]]]],
    test_id: str,
    tolerances: Dict[str, float],
    base_folder: str = "modules/test_error_maps",
):
    failures = []
    has_failed = False

    for group_name, group_data in tensor_groups.items():
        ref_tensors = group_data["ref"]
        comp_tensors = group_data["comp"]
        tensor_names = group_data["names"]

        if len(ref_tensors) != len(comp_tensors):
            failures.append(f"Mismatch in number of tensors for group '{group_name}'. Ref: {len(ref_tensors)}, Comp: {len(comp_tensors)}")
            has_failed = True
            continue

        for i in range(len(ref_tensors)):
            ref_tensor = ref_tensors[i]
            comp_tensor = comp_tensors[i]
            tensor_name = tensor_names[i]

            if not torch.allclose(ref_tensor, comp_tensor, **tolerances):
                has_failed = True
                failures.append(
                    f"Mismatch in group '{group_name}', tensor '{tensor_name}'.\n"
                    f"  Max abs diff: {(ref_tensor - comp_tensor).abs().max().item():.6f}\n"
                    f"  Max rel diff: {((ref_tensor - comp_tensor).abs() / (ref_tensor.abs() + 1e-8)).max().item():.6f}"
                )

    if has_failed:
        heatmap_folder = os.path.join(base_folder, test_id)
        if os.path.exists(heatmap_folder):
            shutil.rmtree(heatmap_folder)
        os.makedirs(heatmap_folder)

        print(f"\n[FAILURE] Correctness test failed for '{test_id}'. Generating heatmaps in '{heatmap_folder}'...")

        for group_name, group_data in tensor_groups.items():
            for i in range(len(group_data["ref"])):
                _save_heatmaps(
                    torch_tensor=group_data["ref"][i],
                    competitor_tensor=group_data["comp"][i],
                    test_name=group_data["names"][i],
                    folder=heatmap_folder,
                    atol=tolerances.get('atol', 1e-8),
                    rtol=tolerances.get('rtol', 1e-5),
                    group_name=group_name,
                )

        pytest.fail(
            f"Test '{test_id}' failed with {len(failures)} tensor mismatch(es):\n" + "\n".join(failures),
            pytrace=False
        )


def build_test_suite(test_configs: List[ModuleTestConfig], available_devices: List[str]) -> List[Any]:
    test_suite = []
    
    for config in test_configs:
        
        # Get the reference competitor class
        ref_competitor_config = config.competitors.get(config.reference_competitor)
        if not ref_competitor_config or not ref_competitor_config.module_class:
            print(f"[WARNING] Reference competitor '{config.reference_competitor}' not found or has no module_class. Skipping.")
            continue

        ReferenceModuleCls = ref_competitor_config.module_class

        # Compare every other competitor to the reference
        for competitor_name, competitor_config in config.competitors.items():

            # For now, we only test non-TP modules in this suite.
            if competitor_config.tp_config:
                continue
                
            CompetitorModuleCls = competitor_config.module_class
            if CompetitorModuleCls is None:
                continue

            for test_case in config.test_cases:
                run_filter = competitor_config.run_filter
                
                for device in available_devices:
                    test_id = f"{config.reference_competitor}_vs_{competitor_name}_{test_case['case_descriptor']}_{device}"
                    test_suite.append(
                        pytest.param(
                            ReferenceModuleCls,
                            CompetitorModuleCls,
                            test_case,
                            device,
                            run_filter,
                            id=test_id
                        )
                    )
    
    return test_suite


ALL_TEST_CONFIGS, DISCOVERY_ERRORS = discover_dunder_objects(
    dunder='__test_config__', 
    object=ModuleTestConfig,
    search_folders="modules/catalog/"
)
AVAILABLE_DEVICES, _ = get_available_devices()
TEST_SUITE = build_test_suite(ALL_TEST_CONFIGS, AVAILABLE_DEVICES)

# Add this to make pytest show more info about parameterized tests
if len(TEST_SUITE) == 0 and not DISCOVERY_ERRORS:
    print("[ERROR] No tests generated!")
    exit()


def test_module_discovery_errors():
    if not DISCOVERY_ERRORS:
        return

    error_messages = [f"  - {file}: {err}" for file, err in DISCOVERY_ERRORS.items()]
    report = (
        "The following files failed to import during test discovery and were skipped:\n"
        + "\n".join(error_messages)
    )
    pytest.fail(report, pytrace=False)


@pytest.mark.parametrize(
    "ReferenceModuleCls, CompetitorModuleCls, test_case, device, run_filter", 
    TEST_SUITE
)
def test_bulk_module_correctness(
    ReferenceModuleCls: nn.Module, 
    CompetitorModuleCls: nn.Module, 
    test_case: Dict[str, Any], 
    device: str,
    run_filter: Union[Callable[Union[torch.Tensor, Sequence[Any]], bool], "NoneType"],
    request: pytest.FixtureRequest,
):
    """
    This function tests that a 'competitor' module implementation is numerically equivalent
    to a 'reference' implementation (e.g., a kernel vs. pure PyTorch).
    Pytest calls this function repeatedly for each parameter set in TEST_SUITE.
    """
    
    # Handle the dummy test case
    if ReferenceModuleCls is None:
        pytest.fail("No tests were generated. Check the debug output above.")
    
    # Instantiate and validate the reference module
    ref_module = ReferenceModuleCls(**test_case['init_args']).to(device)
    ref_inputs = test_case['input_args'](device)

    # Check if the competitor module should be run
    if run_filter is not None and not run_filter(ref_inputs):
        pytest.skip(f"Skipping {CompetitorModuleCls.__name__} on {device} due to run_filter()->False.")
        return
    
    # Run a validator on the reference implementation output to catch baseline bugs
    ref_outputs = ref_module(*ref_inputs)
    if 'output_validator' in test_case:
        outputs_for_validator = ref_outputs if isinstance(ref_outputs, tuple) else (ref_outputs,)
        test_case['output_validator'](ref_module, ref_inputs, outputs_for_validator)

    # Instantiate the competitor module and copy weights
    competitor_module = CompetitorModuleCls(**test_case['init_args']).to(device)
    competitor_module.load_state_dict(ref_module.state_dict())
    
    competitor_inputs = [
        t.clone().detach().requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else copy.deepcopy(t)
        for t in ref_inputs
    ]
    competitor_outputs = competitor_module(*competitor_inputs)

    # Test numerical equivalence
    get_total_loss(ref_outputs).backward()
    get_total_loss(competitor_outputs).backward()
    tolerances = test_case.get('tolerances', {})

    # Use the test's unique ID for the heatmap folder name
    test_id = request.node.callspec.id

    # Prepare tensor groups for comparison
    ref_outputs_tuple = ref_outputs if isinstance(ref_outputs, tuple) else (ref_outputs,)
    competitor_outputs_tuple = competitor_outputs if isinstance(competitor_outputs, tuple) else (competitor_outputs,)

    tensor_groups_to_check = {
        "forward_output": {
            "ref": [o for o in ref_outputs_tuple if isinstance(o, torch.Tensor)],
            "comp": [o for o in competitor_outputs_tuple if isinstance(o, torch.Tensor)],
            "names": [f'output_{i}' for i, o in enumerate(ref_outputs_tuple) if isinstance(o, torch.Tensor)]
        },
        "param_grads": {
            "ref": [p.grad for _, p in ref_module.named_parameters() if p.grad is not None],
            "comp": [p.grad for _, p in competitor_module.named_parameters() if p.grad is not None],
            "names": [f'{p_name}_grad' for p_name, p in ref_module.named_parameters() if p.grad is not None]
        },
        "input_grads": {
            "ref": [i.grad for i in ref_inputs if isinstance(i, torch.Tensor) and i.requires_grad and i.grad is not None],
            "comp": [i.grad for i in competitor_inputs if isinstance(i, torch.Tensor) and i.requires_grad and i.grad is not None],
            "names": [f'input_{i}_grad' for i, t in enumerate(ref_inputs) if isinstance(t, torch.Tensor) and t.requires_grad]
        }
    }
    
    # Check all tensor groups for correctness and generate heatmaps on failure
    _assert_tensors_close_and_generate_heatmaps(
        tensor_groups=tensor_groups_to_check,
        test_id=test_id,
        tolerances=tolerances
    )

    for i, (ref_in, comp_in) in enumerate(zip(ref_inputs, competitor_inputs)):
        # For non-tensor objects, check for equality only if the class has overridden python's default __eq__ method.
        if not isinstance(ref_in, torch.Tensor) and ref_in.__class__.__eq__ is not object.__eq__:
            if ref_in != comp_in:
                ref_str = str(ref_in) if hasattr(ref_in, '__str__') else None
                comp_str = str(comp_in) if hasattr(comp_in, '__str__') else None
                ref_repr = repr(ref_in) if hasattr(ref_in, '__repr__') else None
                comp_repr = repr(comp_in) if hasattr(comp_in, '__repr__') else None
                msg = (
                    f"Non-tensor input {i} mismatch between reference and competitor.\n"
                    f"ref_in == comp_in: {ref_in == comp_in}\n"
                    f"ref_in type: {type(ref_in)}, comp_in type: {type(comp_in)}\n"
                )
                if ref_str is not None and comp_str is not None:
                    msg += f"ref_in str: {ref_str}\ncomp_in str: {comp_str}\n"
                if ref_repr is not None and comp_repr is not None:
                    msg += f"ref_in repr: {ref_repr}\ncomp_in repr: {comp_repr}\n"
                raise AssertionError(msg)