# Checkpoint Format Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 19:50:31
> Generated from: `scripts/generate_api_docs.py`

This document describes the checkpoint file format used by Thalia. Checkpoints are created using `brain.save_checkpoint()` and restored with `brain.load_checkpoint()`.

## Overview

Thalia checkpoints use a hierarchical structure:

```
checkpoint.thalia
├── metadata (timestamp, versions, sizes)
├── regions (component states)
├── pathways (connection states)
├── oscillators (rhythm generator states)
├── neuromodulators (dopamine, acetylcholine, etc.)
└── config (brain configuration)
```

## Top-Level Structure

The checkpoint is a dictionary with these top-level keys:

### Keys from `DynamicBrain.get_full_state()`

| Key | Type | Description |
|-----|------|-------------|
| `global_config` | `Any` | Component state |
| `current_time` | `Any` | Component state |
| `topology` | `Any` | Component state |
| `regions` | `Dict[str, Any]` | Component state |
| `pathways` | `Dict[str, Any]` | Component state |
| `oscillators` | `Dict[str, Any]` | Component state |
| `neuromodulators` | `Dict[str, Any]` | Component state |
| `config` | `Dict[str, Any]` | Component state |
| `growth_history` | `List[Any]` | Component state |

**Source**: `core\dynamic_brain.py`

## Component State Structure

Each component (region or pathway) stores its state in the checkpoint.

### NeuralRegion State (Base Class)

All regions include these fields:

| Key | Type | Description |
|-----|------|-------------|
| `type` | `Any` | Base region data |
| `n_neurons` | `Any` | Base region data |
| `n_input` | `Any` | Base region data |
| `n_output` | `Any` | Base region data |
| `device` | `Any` | Base region data |
| `dt_ms` | `Any` | Base region data |
| `default_learning_rule` | `Any` | Base region data |
| `input_sources` | `Any` | Base region data |
| `synaptic_weights` | `Dict[str, Any]` | Base region data |
| `plasticity_enabled` | `Any` | Base region data |

**Source**: `core\neural_region.py`

## File Format Details

Checkpoints can be saved in multiple formats:

1. **PyTorch Format** (`.pt`, `.pth`, `.ckpt`) - Standard PyTorch `torch.save()` format
2. **Binary Format** (`.thalia`, `.thalia.zst`) - Custom binary format with compression

### Compression Support

- `.zst` extension → Zstandard compression
- `.lz4` extension → LZ4 compression
- No extension → Uncompressed

### Precision Policies

Checkpoints support mixed precision:

- `fp32` - Full precision (default)
- `fp16` - Half precision (smaller files, some accuracy loss)
- `mixed` - fp16 for weights, fp32 for critical state

## Usage Examples

```python
# Save checkpoint
brain.save_checkpoint(
    "checkpoints/epoch_100.ckpt",
    metadata={"epoch": 100, "loss": 0.42}
)

# Load checkpoint
brain.load_checkpoint("checkpoints/epoch_100.ckpt")

# Save with compression
brain.save_checkpoint("checkpoints/epoch_100.thalia.zst", compression="zstd")

# Save with mixed precision
brain.save_checkpoint("checkpoints/epoch_100.ckpt", precision_policy="fp16")
```

## Validation

Use `CheckpointManager.validate()` to check checkpoint integrity:

```python
from thalia.io import CheckpointManager

manager = CheckpointManager(brain)
is_valid, error = manager.validate("checkpoints/epoch_100.ckpt")
if not is_valid:
    print(f"Checkpoint invalid: {error}")
```

## Version Compatibility

Checkpoints include version metadata for compatibility checking:

- `checkpoint_format_version` - Format version (2.0+)
- `thalia_version` - Thalia library version
- `pytorch_version` - PyTorch version used

The checkpoint manager can migrate old formats when loading.

