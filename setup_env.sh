#!/bin/bash

# Get the root directory of the Git repository.
# The `2>/dev/null` silences errors if not in a git repo.
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

# Exit if not in a Git repository. Using 'return' so it can be sourced.
if [ $? -ne 0 ]; then
    echo "Error: This does not appear to be a git repository."
    return 1
fi

# Define the src directory inside the git root.
SRC_DIR="$GIT_ROOT/src"

# Check if PYTHONPATH is already set.
if [ -n "$PYTHONPATH" ]; then
    # Check if the src directory is already in PYTHONPATH to avoid duplicates.
    if [[ ":$PYTHONPATH:" != *":$SRC_DIR:"* ]]; then
        export PYTHONPATH="$SRC_DIR:$PYTHONPATH"
        echo "src directory has been added to PYTHONPATH."
    else
        echo "src directory is already in PYTHONPATH."
    fi
else
    # If PYTHONPATH is not set, initialize it with the src directory.
    export PYTHONPATH="$SRC_DIR"
    echo "PYTHONPATH has been set to the src directory."
fi