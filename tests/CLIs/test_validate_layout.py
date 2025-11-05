from pathlib import Path

import pytest

from CLIs.validate_layout import validate_layout


@pytest.mark.parametrize(
    "missing_items,expected_problems",
    [
        ([], []),
        ([".gpt_lab_root"], ["Missing .gpt_lab_root marker at repo root"]),
        (
            ["catalogs/core/gpt_lab/artifacts"],
            ["Missing artifacts directory:"],
        ),
        (
            ["catalogs/packs/nlp/gpt_lab/artifacts"],
            ["Missing artifacts directory:"],
        ),
        (
            ["catalogs/packs/cv/gpt_lab/artifacts"],
            ["Missing artifacts directory:"],
        ),
        (
            [".gpt_lab_root", "catalogs/core/gpt_lab/artifacts"],
            ["Missing .gpt_lab_root marker at repo root", "Missing artifacts directory:"],
        ),
        (
            [
                ".gpt_lab_root",
                "catalogs/core/gpt_lab/artifacts",
                "catalogs/packs/nlp/gpt_lab/artifacts",
                "catalogs/packs/cv/gpt_lab/artifacts",
            ],
            [
                "Missing .gpt_lab_root marker at repo root",
                "Missing artifacts directory:",
                "Missing artifacts directory:",
                "Missing artifacts directory:",
            ],
        ),
    ],
)
def test_validate_layout(tmp_path, missing_items, expected_problems):
    """Tests validation with various missing items."""
    # Create the full expected structure
    root_marker = tmp_path / ".gpt_lab_root"
    artifacts_dirs = [
        tmp_path / "catalogs" / "core" / "gpt_lab" / "artifacts",
        tmp_path / "catalogs" / "packs" / "nlp" / "gpt_lab" / "artifacts",
        tmp_path / "catalogs" / "packs" / "cv" / "gpt_lab" / "artifacts",
    ]

    # Create everything first
    root_marker.touch()
    for artifacts_dir in artifacts_dirs:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Remove the items that should be missing
    for item in missing_items:
        item_path = tmp_path / item
        if item_path.is_dir():
            item_path.rmdir()
        elif item_path.is_file():
            item_path.unlink()

    # Run validation
    problems = validate_layout(tmp_path)

    # Check results
    assert len(problems) == len(expected_problems)
    for problem, expected_substring in zip(problems, expected_problems):
        assert expected_substring in problem


def test_validate_layout_valid_structure(tmp_path):
    """Tests validation passes with a complete valid structure."""
    # Create root marker
    (tmp_path / ".gpt_lab_root").touch()

    # Create all artifacts directories
    artifacts_dirs = [
        tmp_path / "catalogs" / "core" / "gpt_lab" / "artifacts",
        tmp_path / "catalogs" / "packs" / "nlp" / "gpt_lab" / "artifacts",
        tmp_path / "catalogs" / "packs" / "cv" / "gpt_lab" / "artifacts",
    ]
    for artifacts_dir in artifacts_dirs:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    problems = validate_layout(tmp_path)
    assert problems == []


def test_validate_layout_returns_correct_path_in_error(tmp_path):
    """Tests that error messages contain the actual missing path."""
    # Create root marker but no artifacts directories
    (tmp_path / ".gpt_lab_root").touch()

    problems = validate_layout(tmp_path)

    # Should have 3 problems for the 3 missing artifacts directories
    assert len(problems) == 3
    
    # Check that each problem contains the correct path
    expected_paths = [
        str(tmp_path / "catalogs" / "core" / "gpt_lab" / "artifacts"),
        str(tmp_path / "catalogs" / "packs" / "nlp" / "gpt_lab" / "artifacts"),
        str(tmp_path / "catalogs" / "packs" / "cv" / "gpt_lab" / "artifacts"),
    ]
    
    for expected_path in expected_paths:
        assert any(expected_path in problem for problem in problems)

