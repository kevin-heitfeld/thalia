# Type Aliases Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 19:47:07
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all type aliases used in Thalia for clearer type hints.

Total: 17 type aliases

## Type Aliases by Category

### Component Organization

#### `ComponentGraph`

**Definition**: `Dict[str, NeuralRegion]`

**Description**: name -> component instance

**Source**: `.github/copilot-instructions.md`

---

#### `ConnectionGraph`

**Definition**: `Dict[Tuple[str, str], NeuralRegion]`

**Description**: (src, tgt) -> pathway

**Source**: `.github/copilot-instructions.md`

---

#### `TopologyGraph`

**Definition**: `Dict[str, List[str]]`

**Description**: src -> [tgt1, tgt2, ...]

**Source**: `.github/copilot-instructions.md`

---

### Configuration

#### `ComponentSpec`

**Definition**: `dataclass`

**Description**: Pre-instantiation component definition

**Source**: `.github/copilot-instructions.md`

---

#### `ConnectionSpec`

**Definition**: `dataclass`

**Description**: Pre-instantiation connection definition

**Source**: `.github/copilot-instructions.md`

---

### Diagnostics

#### `DiagnosticsDict`

**Definition**: `Dict[str, Any]`

**Description**: Component health/performance metrics

**Source**: `.github/copilot-instructions.md`

---

#### `HealthReport`

**Definition**: `dataclass`

**Description**: Structured health check results

**Source**: `.github/copilot-instructions.md`

---

### Multi-Source Pathways

#### `InputSizes`

**Definition**: `Dict[str, int]`

**Description**: {region_name: size}

**Source**: `.github/copilot-instructions.md`

---

#### `SourceOutputs`

**Definition**: `Dict[str, torch.Tensor]`

**Description**: {region_name: output_spikes}

**Source**: `.github/copilot-instructions.md`

---

#### `SourceSpec`

**Definition**: `Tuple[str, Optional[str]]`

**Description**: (region_name, port)

**Source**: `.github/copilot-instructions.md`

---

### Other

#### `CompressionType`

**Definition**: `Literal['zstd', 'lz4', None]`

**Source**: `io\compression.py`

---

#### `PresetBuilderFn`

**Definition**: `Callable[['BrainBuilder', Any], None]`

**Source**: `core\brain_builder.py`

---

#### `SpilloverMode`

**Definition**: `Literal['connectivity', 'similarity', 'lateral']`

**Source**: `synapses\spillover.py`

---

### Port-Based Routing

#### `SourcePort`

**Definition**: `Optional[str]`

**Description**: 'l23', 'l5', 'l4', None

**Source**: `.github/copilot-instructions.md`

---

#### `TargetPort`

**Definition**: `Optional[str]`

**Description**: 'feedforward', 'top_down', None

**Source**: `.github/copilot-instructions.md`

---

### State Management

#### `CheckpointMetadata`

**Definition**: `Dict[str, Any]`

**Description**: Training progress, stage info

**Source**: `.github/copilot-instructions.md`

---

#### `StateDict`

**Definition**: `Dict[str, torch.Tensor]`

**Description**: Component state for checkpointing

**Source**: `.github/copilot-instructions.md`

---

