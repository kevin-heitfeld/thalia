# Type Aliases Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-26 14:17:33
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all type aliases used in Thalia for clearer type hints.

Total: 14 type aliases

## Type Aliases by Category

### 

#### `BatchData`

**Definition**: `Dict[str, torch.Tensor]`

**Description**: Batch of training/inference data.  Example: batch: BatchData = { "input": input_spikes,    # (batch_size, timesteps, input_dim) "target": target_labels,   # (batch_size,) "mask": attention_mask,    # (batch_size, timesteps) }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `CheckpointMetadata`

**Definition**: `Dict[str, Any]`

**Description**: Training progress and stage information.  Example: metadata: CheckpointMetadata = { "stage": 2, "epoch": 150, "global_step": 45000, "timestamp": "2025-12-21T08:00:00Z", }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `ComponentGraph`

**Definition**: `Dict[str, 'NeuralRegion']`

**Description**: Maps component names to component instances.  Example: components: ComponentGraph = { "cortex": cortex_region, "hippocampus": hippocampus_region, }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `ConnectionGraph`

**Definition**: `Dict[Tuple[str, str], 'NeuralRegion']`

**Description**: Maps (source, target) pairs to pathway instances.  Example: connections: ConnectionGraph = { ("thalamus", "cortex"): thalamic_pathway, ("cortex", "striatum"): corticostriatal_pathway, }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `InputSizes`

**Definition**: `Dict[str, int]`

**Description**: Maps source names to their input sizes.  Used by multi-source pathways to track input dimensions per source.  Example: input_sizes: InputSizes = { "cortex": 100, "hippocampus": 64, }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `NeuromodulatorLevels`

**Definition**: `Dict[str, float]`

**Description**: Maps neuromodulator names to their current levels (0.0-1.0).  Example: levels: NeuromodulatorLevels = { "dopamine": 0.8, "acetylcholine": 0.6, "norepinephrine": 0.5, }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `SourceOutputs`

**Definition**: `Dict[str, torch.Tensor]`

**Description**: Maps source names to their output spike tensors.  Used for multi-source pathways where multiple regions project to one target.  Example: source_outputs: SourceOutputs = { "cortex": cortex_spikes, "hippocampus": hippocampus_spikes, }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `SourcePort`

**Definition**: `Optional[str]`

**Description**: Optional source port identifier for layer-specific outputs.  Examples: 'l23', 'l5', 'l4', 'ca1', 'ca3', None

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `SourceSpec`

**Definition**: `Tuple[str, Optional[str]]`

**Description**: Specification for a source component with optional port.  Tuple of (region_name, port) where port identifies layer-specific outputs.  Example: source_spec: SourceSpec = ("cortex", "l23")  # Layer 2/3 output source_spec: SourceSpec = ("hippocampus", None)  # Default output

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `StateDict`

**Definition**: `Dict[str, torch.Tensor]`

**Description**: Component state for checkpointing.  Contains all tensors needed to restore component state.  Example: state: StateDict = { "membrane_voltage": v_mem, "synaptic_traces": traces, "eligibility": eligibility, }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `TargetPort`

**Definition**: `Optional[str]`

**Description**: Optional target port identifier for input types.  Examples: 'feedforward', 'top_down', 'ec_l3', 'pfc_modulation', None

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

### Other

#### `LearningStrategies`

**Definition**: `Dict[str, 'LearningStrategy']`

**Description**: Maps source names to their learning strategies.  Each source can have its own learning rule (STDP, BCM, Hebbian, etc.)  Example: strategies: LearningStrategies = { "cortex": stdp_strategy, "hippocampus": hebbian_strategy, }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `SynapticWeights`

**Definition**: `Dict[str, torch.Tensor]`

**Description**: Maps source names to their synaptic weight matrices.  Weights are stored at target dendrites, organized by source region.  Example: synaptic_weights: SynapticWeights = { "cortex": torch.randn(n_output, 100), "hippocampus": torch.randn(n_output, 64), }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

#### `TopologyGraph`

**Definition**: `Dict[str, List[str]]`

**Description**: Maps source region names to lists of target region names.  Example: topology: TopologyGraph = { "thalamus": ["cortex", "striatum"], "cortex": ["striatum", "hippocampus"], }

**Source**: [`thalia/typing.py`](../../src/thalia/typing.py)

---

