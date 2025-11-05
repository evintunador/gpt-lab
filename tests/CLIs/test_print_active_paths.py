import pytest

from CLIs.print_active_paths import format_context_output


@pytest.mark.parametrize(
    "context,verbose,expected_substrings",
    [
        (
            {
                "repo_root": "/path/to/repo",
                "current_experiment": "test_exp",
                "active_experiments": ["test_exp"],
                "active_packs": ["nlp"],
                "ordered_roots": ["/root1", "/root2"],
            },
            False,
            [
                "repo_root: /path/to/repo",
                "current_experiment: test_exp",
                "active_experiments: ['test_exp']",
                "active_packs: ['nlp']",
                "ordered_roots:",
                " - /root1",
                " - /root2",
            ],
        ),
        (
            {
                "repo_root": None,
                "current_experiment": None,
                "active_experiments": [],
                "active_packs": [],
                "ordered_roots": [],
            },
            False,
            [
                "repo_root: None",
                "current_experiment: None",
                "active_experiments: []",
                "active_packs: []",
                "ordered_roots:",
            ],
        ),
        (
            {
                "repo_root": "/path/to/repo",
                "current_experiment": "exp1",
                "active_experiments": ["exp1", "exp2"],
                "active_packs": ["nlp", "cv"],
                "ordered_roots": ["/root1"],
            },
            False,
            [
                "repo_root: /path/to/repo",
                "current_experiment: exp1",
                "active_experiments: ['exp1', 'exp2']",
                "active_packs: ['nlp', 'cv']",
                " - /root1",
            ],
        ),
    ],
)
def test_format_context_output(context, verbose, expected_substrings):
    """Tests context output formatting with various inputs."""
    output = format_context_output(context, verbose=verbose)
    
    for expected in expected_substrings:
        assert expected in output


def test_format_context_output_verbose():
    """Tests verbose output includes package __path__s section."""
    context = {
        "repo_root": "/path/to/repo",
        "current_experiment": "test",
        "active_experiments": ["test"],
        "active_packs": [],
        "ordered_roots": [],
    }
    
    output = format_context_output(context, verbose=True)
    
    # Should include basic info
    assert "repo_root: /path/to/repo" in output
    
    # Should include verbose section
    assert "package __path__s:" in output


def test_format_context_output_ordering():
    """Tests that output sections appear in the expected order."""
    context = {
        "repo_root": "/repo",
        "current_experiment": "exp",
        "active_experiments": ["exp"],
        "active_packs": ["pack"],
        "ordered_roots": ["/root"],
    }
    
    output = format_context_output(context, verbose=False)
    lines = output.split("\n")
    
    # Check order of sections
    assert lines[0].startswith("repo_root:")
    assert lines[1].startswith("current_experiment:")
    assert lines[2].startswith("active_experiments:")
    assert lines[3].startswith("active_packs:")
    assert lines[4] == "ordered_roots:"
    assert lines[5].startswith(" - ")


def test_format_context_output_with_missing_keys():
    """Tests handling of missing keys in context dict."""
    context = {}  # Empty context
    
    output = format_context_output(context, verbose=False)
    
    # Should handle missing keys gracefully with .get()
    assert "repo_root: None" in output
    assert "current_experiment: None" in output
    assert "active_experiments: None" in output
    assert "active_packs: None" in output
    assert "ordered_roots:" in output


@pytest.mark.parametrize(
    "ordered_roots,expected_count",
    [
        ([], 0),
        (["/root1"], 1),
        (["/root1", "/root2", "/root3"], 3),
        (["/a", "/b", "/c", "/d", "/e"], 5),
    ],
)
def test_format_context_output_ordered_roots_count(ordered_roots, expected_count):
    """Tests correct number of ordered roots are displayed."""
    context = {
        "repo_root": "/repo",
        "current_experiment": None,
        "active_experiments": [],
        "active_packs": [],
        "ordered_roots": ordered_roots,
    }
    
    output = format_context_output(context, verbose=False)
    
    # Count lines that start with " - "
    root_lines = [line for line in output.split("\n") if line.startswith(" - ")]
    assert len(root_lines) == expected_count
    
    # Verify each root appears
    for root in ordered_roots:
        assert f" - {root}" in output
