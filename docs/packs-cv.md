# CV Pack

The CV (Computer Vision) pack (`catalogs/packs/cv/gpt_lab/`) provides components for computer vision experiments.

## Overview

The CV pack is currently a placeholder for future computer vision components. It provides the structure for adding:
- **models**: Vision models (CNNs, ViTs, etc.)
- **data_sources**: Image datasets (ImageNet, CIFAR, etc.)
- **benchmarks**: Vision benchmarks and evaluation metrics
- **nn_modules**: CV-specific layers (convolutions, pooling, etc.)

## Activation

To use the CV pack (when populated), activate it in your experiment:

```yaml
# experiments/my_cv_exp/gpt_lab.yaml
include_experiments: []
include_packs: ['cv']
```

Or via environment variable:

```bash
export GPT_LAB_ACTIVE_PACKS=cv
```

## Current Status

The CV pack currently contains only the artifacts directory structure. This pack is intended for future expansion with computer vision components.