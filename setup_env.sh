#!/bin/bash

# Get the root directory of the Git repository.
# The `2>/dev/null` silences errors if not in a git repo.
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

# Exit if not in a Git repository. Using 'return' so it can be sourced.
if [ $? -ne 0 ]; then
    echo "Error: This does not appear to be a git repository."
    return 1
fi

# Check if PYTHONPATH is already set.
if [ -n "$PYTHONPATH" ]; then
    # Check if the git root is already in PYTHONPATH to avoid duplicates.
    if [[ ":$PYTHONPATH:" != *":$GIT_ROOT:"* ]]; then
        export PYTHONPATH="$GIT_ROOT:$PYTHONPATH"
        echo "Project root has been added to PYTHONPATH."
    else
        echo "Project root is already in PYTHONPATH."
    fi
else
    # If PYTHONPATH is not set, initialize it with the project root.
    export PYTHONPATH="$GIT_ROOT"
    echo "PYTHONPATH has been set to the project root."
fi