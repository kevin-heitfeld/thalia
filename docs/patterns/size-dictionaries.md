# Size Dictionary Pattern

**Status**: Active  
**Last Updated**: January 25, 2026  
**Applies To**: All brain regions (Striatum, LayeredCortex, Hippocampus, etc.)

## Overview

Brain regions follow a standardized pattern for size specification using the **Config + Sizes + Device** separation introduced in January 2026. This document defines the standard keys used in the `sizes` dictionary parameter.

## Pattern

```python
class RegionName(NeuralRegion):
    def __init__(self, config: RegionConfig, sizes: Dict[str, int], device: str):
        """Initialize region.
        
        Args:
            config: Behavioral configuration (learning rates, sparsity, etc.)
            sizes: Size specification from LayerSizeCalculator
            device: Device for tensors ("cpu" or "cuda")
        """
        # Extract sizes
        self.input_size = sizes["input_size"]
        self.layer_size = sizes["layer_size"]
        # ...
```

## Standard Keys

### Required Keys

All regions **must** include these keys in their `sizes` dictionary:

#### `input_size` (int)
**Purpose**: Total external input dimension  
**Usage**: Sum of all incoming connections from other regions  
**Calculation**: Typically computed by BrainBuilder based on connectivity graph

```python
# Example: Cortex receives from thalamus (64) + hippocampus (128)
sizes = {"input_size": 192, ...}
```

#### `output_size` (int) 
**Purpose**: Total output dimension (for simple regions)  
**Usage**: For regions without internal layer structure  
**Alternatives**: Region-specific keys (see below)

```python
# Simple region (Prefrontal):
sizes = {"input_size": 256, "n_neurons": 128, ...}

# Layered region (LayeredCortex):
sizes = {"input_size": 192, "output_size": 864, ...}  # output_size = l23 + l5
```

### Region-Specific Keys

Regions with internal structure use **biologically-named** layer keys:

#### Cortex (Laminar Structure)
```python
sizes = {
    "input_size": 192,      # External inputs
    "l4_size": 288,         # Layer 4 (input layer)
    "l23_size": 576,        # Layer 2/3 (processing/output)
    "l5_size": 288,         # Layer 5 (subcortical output)
    "l6a_size": 115,        # Layer 6a (CT type I → TRN)
    "l6b_size": 75,         # Layer 6b (CT type II → relay)
    "output_size": 864,     # l23_size + l5_size
    "total_neurons": 1342,  # Sum of all layers
}
```

**Naming Convention**: Lowercase layer names with underscores (`l4_size`, `l23_size`, not `L4Size` or `layer4_size`)

#### Hippocampus (Trisynaptic Circuit)
```python
sizes = {
    "input_size": 128,      # Entorhinal cortex input
    "dg_size": 512,         # Dentate Gyrus (pattern separation)
    "ca3_size": 256,        # CA3 (pattern completion)
    "ca2_size": 128,        # CA2 (social memory)
    "ca1_size": 256,        # CA1 (output/comparison)
    "output_size": 256,     # CA1 output
}
```

**Naming Convention**: Lowercase biological region names (`dg_size`, `ca3_size`, not `DG_size` or `dentate_gyrus_size`)

#### Striatum (Opponent Pathways)
```python
sizes = {
    "input_size": 256,      # Cortical/thalamic input
    "n_actions": 4,         # Number of discrete actions
    "d1_size": 40,          # D1 pathway (Go signal)
    "d2_size": 40,          # D2 pathway (NoGo signal)
    "neurons_per_action": 10,  # Population coding (optional)
}
```

**Naming Convention**: 
- `n_actions` for semantic output dimension (not `output_size`)
- `d1_size`, `d2_size` for pathway-specific populations
- Optional parameters use `.get()` with defaults

#### Cerebellum (Granule-Purkinje)
```python
sizes = {
    "input_size": 128,      # Mossy fiber input
    "granule_size": 512,    # Granule cell layer (expansion)
    "purkinje_size": 128,   # Purkinje cell layer
    "output_size": 128,     # Deep nuclei output
}
```

#### Thalamus (Relay + TRN)
```python
sizes = {
    "input_size": 64,       # Sensory input (or default to relay_size)
    "relay_size": 64,       # Relay neurons (sensory pathway)
    "trn_size": 19,         # Thalamic Reticular Nucleus (gating)
    "output_size": 64,      # Relay output to cortex
}
```

### Optional Keys with Defaults

Use `.get()` with sensible defaults for optional parameters:

```python
# Striatum example:
self.input_size = sizes.get("input_size", 0)  # May be added later via add_input_source()
self.neurons_per_action = sizes.get("neurons_per_action", 10)  # Population coding parameter

# Thalamus example:
self.input_size = sizes.get("input_size", self.relay_size)  # Default to relay_size
```

## Size Calculation

Use `LayerSizeCalculator` for consistent, biologically-motivated size computations:

```python
from thalia.config.size_calculator import LayerSizeCalculator, BiologicalRatios

# Initialize calculator
calc = LayerSizeCalculator()  # Uses default biological ratios

# Calculate sizes from input
cortex_sizes = calc.cortex_from_input(input_size=192)
hipp_sizes = calc.hippocampus_from_input(input_size=128)
striatum_sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=10)

# Or from desired output
cortex_sizes = calc.cortex_from_output(target_output_size=864)

# Custom ratios
custom_ratios = BiologicalRatios(l23_to_l4=3.0)  # Larger L2/3
calc = LayerSizeCalculator(ratios=custom_ratios)
```

## Naming Guidelines

### DO:
- ✅ Use lowercase with underscores: `l4_size`, `ca3_size`, `d1_size`
- ✅ Use biological names: `dg_size` (Dentate Gyrus), `trn_size` (Thalamic Reticular Nucleus)
- ✅ Use semantic names for action-based regions: `n_actions`, `n_neurons`
- ✅ Always include `input_size` (external inputs)
- ✅ Use `output_size` for simple regions OR layer-specific outputs for complex regions
- ✅ Use `.get()` for optional parameters with sensible defaults

### DON'T:
- ❌ CamelCase: `L4Size`, `DGSize`
- ❌ Verbose names: `dentate_gyrus_size`, `layer_4_size`
- ❌ Abbreviations without context: `l4` (should be `l4_size`)
- ❌ Inconsistent input naming: `n_input`, `input_neurons` (always use `input_size`)
- ❌ Missing defaults for optional parameters

## Examples

### Creating a Region (Region Implementation)

```python
from thalia.core.neural_region import NeuralRegion
from thalia.config.region_configs import MyRegionConfig

class MyRegion(NeuralRegion):
    def __init__(self, config: MyRegionConfig, sizes: Dict[str, int], device: str):
        # Extract sizes
        self.input_size = sizes["input_size"]
        self.layer1_size = sizes["layer1_size"]
        self.layer2_size = sizes["layer2_size"]
        
        # Optional with default
        self.optional_param = sizes.get("optional_param", 10)
        
        # Initialize base class
        total_neurons = self.layer1_size + self.layer2_size
        super().__init__(n_neurons=total_neurons, device=device)
```

### Using in BrainBuilder (User Code)

```python
from thalia.core.dynamic_brain import BrainBuilder
from thalia.config.size_calculator import LayerSizeCalculator

# Create builder
builder = BrainBuilder(config=brain_config)

# Calculate sizes
calc = LayerSizeCalculator()
cortex_sizes = calc.cortex_from_input(input_size=192)
striatum_sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=10)

# Add regions (BrainBuilder passes sizes automatically)
builder.add_region("cortex", cortex_config)
builder.add_region("striatum", striatum_config)

# Or with explicit sizes
builder.add_region("cortex", cortex_config, sizes=cortex_sizes)
```

## Migration Notes

### Pre-January 2026 Pattern (Deprecated)

**Old pattern** (sizes embedded in config):
```python
class MyRegionConfig:
    n_input: int = 256
    n_output: int = 128
    layer_size: int = 64

class MyRegion(NeuralRegion):
    def __init__(self, config: MyRegionConfig):
        self.input_size = config.n_input  # Size from config
        # ...
```

**New pattern** (config + sizes + device separation):
```python
class MyRegionConfig:
    learning_rate: float = 0.01
    # NO size parameters in config!

class MyRegion(NeuralRegion):
    def __init__(self, config: MyRegionConfig, sizes: Dict[str, int], device: str):
        self.input_size = sizes["input_size"]  # Size from sizes dict
        # ...
```

**Rationale**: 
- Separates behavioral config (learning, sparsity) from structural config (sizes)
- Enables dynamic size calculation via LayerSizeCalculator
- Makes checkpoint compatibility easier (sizes can change without config changes)

## Validation

Regions should validate sizes in `__init__`:

```python
def __init__(self, config: MyRegionConfig, sizes: Dict[str, int], device: str):
    # Validate required keys
    required_keys = ["input_size", "layer1_size", "layer2_size"]
    missing = [k for k in required_keys if k not in sizes]
    if missing:
        raise ValueError(f"Missing required size keys: {missing}")
    
    # Validate size constraints
    if sizes["layer1_size"] <= 0:
        raise ValueError(f"layer1_size must be positive, got {sizes['layer1_size']}")
```

## See Also

- [LayerSizeCalculator API](../api/SIZE_CALCULATOR.md) - Size computation utilities
- [Config Pattern](./config-separation.md) - Config + Sizes + Device separation
- [BiologicalRatios](../api/BIOLOGICAL_RATIOS.md) - Neuroscience-based size ratios
- [BrainBuilder](../api/BRAIN_BUILDER.md) - Brain construction and connectivity

## References

This pattern is based on:
- ADR-012: Config + Sizes + Device Separation (January 2026)
- Architectural Review 2026-01-25: Tier 1.2
- LayerSizeCalculator implementation (January 2026)
