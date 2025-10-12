# Validate Layout CLI

The `validate_layout.py` script verifies that the GPT-Lab repository has the correct directory structure and required markers.

## Overview

This CLI tool:
- Checks for `.gpt_lab_root` marker at repo root
- Verifies existence of required `artifacts/` directories
- Validates catalog pack structure
- Exits with error code if validation fails
- Useful for CI/CD and setup verification

## Usage

```bash
# Validate repository layout
python CLIs/validate_layout.py
```

## What It Checks

### 1. Root Marker

Checks for `.gpt_lab_root` file at repository root:

```
/path/to/gpt-lab/.gpt_lab_root
```

This marker helps the bootstrap system locate the repository root.

### 2. Core Artifacts Directory

```
catalogs/core/gpt_lab/artifacts/
```

This directory stores compiled train loops and other generated artifacts for the core catalog.

### 3. Pack Artifacts Directories

```
catalogs/packs/nlp/gpt_lab/artifacts/
catalogs/packs/cv/gpt_lab/artifacts/
```

Each pack should have an `artifacts/` directory for its generated content.

## Output

### Success

```bash
$ python CLIs/validate_layout.py
Layout OK
```

Exit code: 0

### Failure

```bash
$ python CLIs/validate_layout.py
[ERROR] Missing .gpt_lab_root marker at repo root
[ERROR] Missing artifacts directory: /path/to/catalogs/core/gpt_lab/artifacts
```

Exit code: 1

## When to Run

### 1. After Cloning

Always validate after cloning:

```bash
git clone <repo>
cd <repo>
python CLIs/validate_layout.py
```

### 2. After Pulling

After pulling changes that might affect structure:

```bash
git pull
python CLIs/validate_layout.py
```

### 3. Before Releasing

Validate before creating a release:

```bash
python CLIs/validate_layout.py
git tag v1.0.0
git push --tags
```

### 4. In CI/CD

Always include in continuous integration:

```yaml
- name: Validate layout
  run: python CLIs/validate_layout.py
```

### 5. After Manual Changes

After manually creating/moving directories:

```bash
# Make changes
mkdir -p new_directory

# Validate
python CLIs/validate_layout.py
```