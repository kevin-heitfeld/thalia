# State Classes Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-09 00:11:42
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all state classes used for serialization in Thalia's checkpoint system. State classes inherit from `RegionState`, `BaseRegionState`, or `PathwayState`.

Total: 7 state classes

## Overview

State classes provide serialization support for checkpoints. Each state class:

- Inherits from a base state class (`RegionState`, `BaseRegionState`, or `PathwayState`)
- Implements `to_dict()` and `from_dict()` for serialization
- Includes `STATE_VERSION` for migration support
- Contains only mutable state (not configuration or learned parameters)

## State Class Hierarchy

```
RegionState (Protocol)
├── BaseRegionState (dataclass)
│   ├── PrefrontalState
│   ├── ThalamicRelayState
│   └── ... (other region states)
│
└── PathwayState (Protocol)
    └── AxonalProjectionState
```

## Region State Classes

Total region states: 7

### [``AxonalProjectionState``](../../src/thalia/core/pathway_state.py#L130)

**Base Class**: `PathwayState`  
**Version**: 1  
**Source**: [`thalia/core/pathway_state.py`](../../src/thalia/core/pathway_state.py)

**Description**: State for AxonalProjection with delay buffers.

**Fields**:

| Field | Type | Default |
|-------|------|----------|
| `delay_buffers` | `Dict[str, Tuple[torch.Tensor, int, int, int]]` | `{}` |

---

### [``CerebellumState``](../../src/thalia/regions/cerebellum_region.py#L214)

**Base Class**: `BaseRegionState`  
**Version**: 1  
**Source**: [`thalia/regions/cerebellum_region.py`](../../src/thalia/regions/cerebellum_region.py)

**Description**: Complete state for Cerebellum region.

**Fields**:

| Field | Type | Default |
|-------|------|----------|
| `input_trace` | `Optional[torch.Tensor]` | `None` |
| `output_trace` | `Optional[torch.Tensor]` | `None` |
| `stdp_eligibility` | `Optional[torch.Tensor]` | `None` |
| `climbing_fiber_error` | `Optional[torch.Tensor]` | `None` |
| `io_membrane` | `Optional[torch.Tensor]` | `None` |
| `v_mem` | `Optional[torch.Tensor]` | `None` |
| `g_exc` | `Optional[torch.Tensor]` | `None` |
| `g_inh` | `Optional[torch.Tensor]` | `None` |
| `stp_pf_purkinje_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |
| `stp_mf_granule_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |
| `granule_layer_state` | `Optional[Dict[str, Any]]` | `None` |
| `purkinje_cells_state` | `Optional[list]` | `None` |
| `deep_nuclei_state` | `Optional[Dict[str, Any]]` | `None` |

---

### [``HippocampusState``](../../src/thalia/regions/hippocampus/config.py#L282)

**Base Class**: `BaseRegionState`  
**Version**: 1  
**Source**: [`thalia/regions/hippocampus/config.py`](../../src/thalia/regions/hippocampus/config.py)

**Description**: State for hippocampus (DG→CA3→CA2→CA1 circuit) with RegionState protocol compliance.

**Fields**:

| Field | Type | Default |
|-------|------|----------|
| `dg_spikes` | `Optional[torch.Tensor]` | `None` |
| `ca3_spikes` | `Optional[torch.Tensor]` | `None` |
| `ca2_spikes` | `Optional[torch.Tensor]` | `None` |
| `ca1_spikes` | `Optional[torch.Tensor]` | `None` |
| `ca3_membrane` | `Optional[torch.Tensor]` | `None` |
| `ca1_membrane` | `Optional[torch.Tensor]` | `None` |
| `ca3_persistent` | `Optional[torch.Tensor]` | `None` |
| `sample_trace` | `Optional[torch.Tensor]` | `None` |
| `dg_trace` | `Optional[torch.Tensor]` | `None` |
| `ca3_trace` | `Optional[torch.Tensor]` | `None` |
| `ca2_trace` | `Optional[torch.Tensor]` | `None` |
| `nmda_trace` | `Optional[torch.Tensor]` | `None` |
| `stored_dg_pattern` | `Optional[torch.Tensor]` | `None` |
| `ffi_strength` | `float` | `0.0` |
| `stp_mossy_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |
| `stp_ca3_ca2_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |
| `stp_ca2_ca1_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |
| `stp_ec_ca2_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |
| `stp_schaffer_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |
| `stp_ec_ca1_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |
| `stp_ca3_recurrent_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |

---

### [``LayeredCortexState``](../../src/thalia/regions/cortex/config.py#L208)

**Base Class**: `BaseRegionState`  
**Version**: 1  
**Source**: [`thalia/regions/cortex/config.py`](../../src/thalia/regions/cortex/config.py)

**Description**: State for layered cortex with RegionState protocol compliance.

**Fields**:

| Field | Type | Default |
|-------|------|----------|
| `input_spikes` | `Optional[torch.Tensor]` | `None` |
| `l4_spikes` | `Optional[torch.Tensor]` | `None` |
| `l23_spikes` | `Optional[torch.Tensor]` | `None` |
| `l5_spikes` | `Optional[torch.Tensor]` | `None` |
| `l6a_spikes` | `Optional[torch.Tensor]` | `None` |
| `l6b_spikes` | `Optional[torch.Tensor]` | `None` |
| `l23_membrane` | `Optional[torch.Tensor]` | `None` |
| `l23_recurrent_activity` | `Optional[torch.Tensor]` | `None` |
| `l4_trace` | `Optional[torch.Tensor]` | `None` |
| `l23_trace` | `Optional[torch.Tensor]` | `None` |
| `l5_trace` | `Optional[torch.Tensor]` | `None` |
| `l6a_trace` | `Optional[torch.Tensor]` | `None` |
| `l6b_trace` | `Optional[torch.Tensor]` | `None` |
| `top_down_modulation` | `Optional[torch.Tensor]` | `None` |
| `ffi_strength` | `float` | `0.0` |
| `alpha_suppression` | `float` | `1.0` |
| `gamma_attention_phase` | `Optional[float]` | `None` |
| `gamma_attention_gate` | `Optional[torch.Tensor]` | `None` |
| `last_plasticity_delta` | `float` | `0.0` |
| `stp_l23_recurrent_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |

---

### [``PrefrontalState``](../../src/thalia/regions/prefrontal.py#L184)

**Base Class**: `BaseRegionState`  
**Version**: 1  
**Source**: [`thalia/regions/prefrontal.py`](../../src/thalia/regions/prefrontal.py)

**Description**: State for prefrontal cortex region.

**Fields**:

| Field | Type | Default |
|-------|------|----------|
| `working_memory` | `Optional[torch.Tensor]` | `None` |
| `update_gate` | `Optional[torch.Tensor]` | `None` |
| `active_rule` | `Optional[torch.Tensor]` | `None` |
| `stp_recurrent_state` | `Optional[Dict[str, Any]]` | `None` |

---

### [``StriatumState``](../../src/thalia/regions/striatum/config.py#L225)

**Base Class**: `BaseRegionState`  
**Version**: 1  
**Source**: [`thalia/regions/striatum/config.py`](../../src/thalia/regions/striatum/config.py)

**Description**: Complete state for Striatum region.

**Fields**:

| Field | Type | Default |
|-------|------|----------|
| `fsi_membrane` | `Optional[torch.Tensor]` | `None` |
| `d1_pathway_state` | `Optional[Dict[str, Any]]` | `None` |
| `d2_pathway_state` | `Optional[Dict[str, Any]]` | `None` |
| `d1_votes_accumulated` | `Optional[torch.Tensor]` | `None` |
| `d2_votes_accumulated` | `Optional[torch.Tensor]` | `None` |
| `last_action` | `Optional[int]` | `None` |
| `recent_spikes` | `Optional[torch.Tensor]` | `None` |
| `exploring` | `bool` | `False` |
| `last_uncertainty` | `Optional[float]` | `None` |
| `last_exploration_prob` | `Optional[float]` | `None` |
| `exploration_manager_state` | `Optional[Dict[str, Any]]` | `None` |
| `value_estimates` | `Optional[torch.Tensor]` | `None` |
| `last_rpe` | `Optional[float]` | `None` |
| `last_expected` | `Optional[float]` | `None` |
| `pfc_modulation_d1` | `Optional[torch.Tensor]` | `None` |
| `pfc_modulation_d2` | `Optional[torch.Tensor]` | `None` |
| `d1_delay_buffer` | `Optional[torch.Tensor]` | `None` |
| `d2_delay_buffer` | `Optional[torch.Tensor]` | `None` |
| `d1_delay_ptr` | `int` | `0` |
| `d2_delay_ptr` | `int` | `0` |
| `activity_ema` | `float` | `0.0` |
| `trial_spike_count` | `int` | `0` |
| `trial_timesteps` | `int` | `0` |
| `homeostatic_scaling_applied` | `bool` | `False` |
| `homeostasis_manager_state` | `Optional[Dict[str, Any]]` | `None` |
| `stp_corticostriatal_u` | `Optional[torch.Tensor]` | `None` |
| `stp_corticostriatal_x` | `Optional[torch.Tensor]` | `None` |
| `stp_thalamostriatal_u` | `Optional[torch.Tensor]` | `None` |
| `stp_thalamostriatal_x` | `Optional[torch.Tensor]` | `None` |

---

### [``ThalamicRelayState``](../../src/thalia/regions/thalamus.py#L237)

**Base Class**: `BaseRegionState`  
**Version**: 1  
**Source**: [`thalia/regions/thalamus.py`](../../src/thalia/regions/thalamus.py)

**Description**: State for thalamic relay nucleus with RegionState protocol compliance.

**Fields**:

| Field | Type | Default |
|-------|------|----------|
| `relay_spikes` | `Optional[torch.Tensor]` | `None` |
| `relay_membrane` | `Optional[torch.Tensor]` | `None` |
| `trn_spikes` | `Optional[torch.Tensor]` | `None` |
| `trn_membrane` | `Optional[torch.Tensor]` | `None` |
| `current_mode` | `Optional[torch.Tensor]` | `None` |
| `burst_counter` | `Optional[torch.Tensor]` | `None` |
| `alpha_gate` | `Optional[torch.Tensor]` | `None` |
| `stp_sensory_relay_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |
| `stp_l6_feedback_state` | `Optional[Dict[str, torch.Tensor]]` | `None` |

---

## Usage Guide

### Creating a New State Class

```python
from dataclasses import dataclass
from typing import Optional
import torch
from thalia.core.region_state import BaseRegionState

@dataclass
class MyRegionState(BaseRegionState):
    """State for MyRegion."""
    STATE_VERSION: int = 1

    # Add your state fields
    custom_spikes: Optional[torch.Tensor] = None
    custom_membrane: Optional[torch.Tensor] = None
```

### Serialization

State classes automatically inherit `to_dict()` and `from_dict()` methods:

```python
# Save state
state_dict = region.get_state().to_dict()

# Load state
loaded_state = MyRegionState.from_dict(state_dict, device='cpu')
region.load_state(loaded_state)
```

### Version Migration

When adding new fields, increment `STATE_VERSION` and add migration logic:

```python
@dataclass
class MyRegionState(BaseRegionState):
    STATE_VERSION: int = 2  # Incremented from 1

    # New field in v2
    new_field: Optional[torch.Tensor] = None

    @classmethod
    def _migrate_from_v1(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1 state to v2."""
        data['new_field'] = None  # Initialize with default
        return data
```

**See Also**:
- `docs/patterns/state-management.md` - State management patterns
- `docs/api/CHECKPOINT_FORMAT.md` - Checkpoint file format

