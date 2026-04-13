# HyUOD Adaptive Variant

This folder contains `hyuod_adapt.yaml`, an ablation-ready model config that replaces one `frequent_block` with `AdaptiveEdgeWindowBlock`.

## What was added

- Background suppression gate (global + local)
- Fixed Sobel edge enhancement branch
- Local window self-attention branch
- Residual weighted fusion with learnable `alpha`, `beta`, `gamma`

## Quick start

Use your existing training script and point model yaml to `train_yaml/hyuod_adapt.yaml`.

```bash
python train.py --model_yaml train_yaml/hyuod_adapt.yaml
```

## Suggested ablation path

1. Baseline: `train_yaml/hyuod.yaml`
2. Adaptive block (P3 only): `train_yaml/hyuod_adapt.yaml`
3. If metrics improve, consider adding another block at P4 in a new yaml.

