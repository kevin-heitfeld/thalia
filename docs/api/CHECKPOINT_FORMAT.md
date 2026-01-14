# Checkpoint Format Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-14 17:31:39
> Generated from: `scripts/generate_api_docs.py`

> **📚 For complete binary format specification, version compatibility, and implementation details, see [checkpoint_format.md](../design/checkpoint_format.md)**

This document provides a quick reference for checkpoint usage and the state dictionary structure returned by `brain.get_full_state()`.

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

## Top-Level State Structure

The checkpoint is a dictionary with these top-level keys returned by `DynamicBrain.get_full_state()`:

| Key | Type | Description |
|-----|------|-------------|
| `global_config` | `Any` | Global configuration |
| `current_time` | `Any` | Current simulation time |
| `topology` | `Any` | Brain topology graph |
| `regions` | `Dict[str, Any]` | All region states |
| `pathways` | `Dict[str, Any]` | All pathway states |
| `oscillators` | `Dict[str, Any]` | Oscillator states |
| `neuromodulators` | `Dict[str, Any]` | Neuromodulator levels |
| `config` | `Dict[str, Any]` | Brain configuration |
| `growth_history` | `List[Any]` | Growth event log |

**Source**: [`thalia/core/dynamic_brain.py`](../../src/thalia/core/dynamic_brain.py)

## Component State Structure

Each component (region or pathway) stores its state in the checkpoint.

### NeuralRegion State (Base Class)

All regions include these base fields:

| Key | Type | Description |
|-----|------|-------------|
| `type` | `Any` | Region type identifier |
| `n_neurons` | `Any` | Number of neurons |
| `n_input` | `Any` | Input dimension |
| `n_output` | `Any` | Output dimension |
| `device` | `Any` | Device (CPU/GPU) |
| `dt_ms` | `Any` | Timestep in milliseconds |
| `default_learning_rule` | `Any` | Default learning strategy |
| `input_sources` | `Any` | Input source names |
| `synaptic_weights` | `Dict[str, Any]` | Weight matrices per source |

**Source**: [`thalia/core/neural_region.py`](../../src/thalia/core/neural_region.py)

## File Formats

Checkpoints can be saved in two formats:

1. **PyTorch Format** (`.pt`, `.pth`, `.ckpt`) - Standard PyTorch `torch.save()` format (default)
2. **Binary Format** (`.thalia`, `.thalia.zst`) - Custom binary format with compression (advanced)

## Usage Examples

```python
# Save checkpoint (PyTorch format - default)
brain.save_checkpoint(
    "checkpoints/epoch_100.ckpt",
    metadata={"epoch": 100, "loss": 0.42}
)

# Load checkpoint
brain.load_checkpoint("checkpoints/epoch_100.ckpt")

# Save with compression (binary format)
brain.save_checkpoint("checkpoints/epoch_100.thalia.zst", compression="zstd")

# Save with mixed precision
brain.save_checkpoint("checkpoints/epoch_100.ckpt", precision_policy="fp16")
```

## Validation

```python
from thalia.io import CheckpointManager

manager = CheckpointManager(brain)
is_valid, error = manager.validate("checkpoints/epoch_100.ckpt")
if not is_valid:
    print(f"Checkpoint invalid: {error}")
```

## See Also

- **[Checkpoint Format Specification](../design/checkpoint_format.md)** - Complete binary format details, byte layouts, compression algorithms
- **[Curriculum Strategy](../design/curriculum_strategy.md)** - Training stages and checkpoint usage in curriculum training
- **[GETTING_STARTED_CURRICULUM](../GETTING_STARTED_CURRICULUM.md)** - Tutorial including checkpoint management

