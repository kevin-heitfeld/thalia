# AI Assistant Navigation Guide

This guide helps AI assistants (and developers) navigate the Thalia codebase efficiently using search-based navigation instead of fragile line numbers.

## Quick Reference Card

**Architecture**:
- **NeuralRegion**: Base class (nn.Module + 4 mixins) with synaptic_weights dict at dendrites
- **AxonalProjection**: Pure spike routing (NO weights, NO learning, ONLY delays)
- **CircularDelayBuffer**: Axonal conduction delays (1-20ms)
- **Learning**: Pluggable strategies at target synapses (region.synaptic_weights), NOT in pathways
- **Input Pattern**: Regions receive `Dict[str, torch.Tensor]` from multiple sources
- **Mixins**: Neuromodulator, Growth, Resettable, Diagnostics

**Finding Code (use built-in tools):**
- Regions: `grep_search(query="@register_region", isRegexp=false)`
- Pathways: `grep_search(query="@register_pathway", isRegexp=false)`
- Growth methods: `grep_search(query="def grow_output", isRegexp=false)`
- Axonal routing: `grep_search(query="AxonalProjection", isRegexp=false)`
- Delay buffers: `grep_search(query="CircularDelayBuffer", isRegexp=false)`
- Symbol usages: `list_code_usages(symbolName="grow_output")`
- Conceptual search: `semantic_search(query="where does learning happen in striatum")`

**Section Markers (use as search targets):**
- `=== GROWTH METHODS ===` - Growth implementations
- `=== SYNAPTIC WEIGHTS ===` - Weight storage at dendrites
- `=== AXONAL ROUTING ===` - Spike transmission logic

**Key Files:**
- `docs/architecture/ARCHITECTURE_OVERVIEW.md` - System overview (START HERE)
- `src/thalia/core/neural_region.py` - **NeuralRegion base class** (synapses at dendrites)
- `src/thalia/pathways/axonal_projection.py` - **AxonalProjection** (pure routing, NO weights)
- `src/thalia/utils/delay_buffer.py` - **CircularDelayBuffer** (axonal delays)
- `src/thalia/core/dynamic_brain.py` - Main brain implementation
- `src/thalia/core/brain_builder.py` - Fluent construction API
- `src/thalia/learning/strategies/` - Learning strategy implementations

## Quick Start

**Built-in Search Tools (Preferred):**

```python
# Exact text/regex search
grep_search(query="@register_region", isRegexp=False)
grep_search(query="def grow_output", isRegexp=False)
grep_search(query="AxonalProjection", isRegexp=False)

# Conceptual/semantic search
semantic_search(query="where does learning happen in striatum")
semantic_search(query="how do axonal delays work")
semantic_search(query="synaptic weights at dendrites")

# Find all usages of a symbol
list_code_usages(symbolName="NeuralRegion")
list_code_usages(symbolName="AxonalProjection")
list_code_usages(symbolName="grow_output")

# Find files by pattern
file_search(query="**/*pathway*.py")
file_search(query="**/*region*.py")
```

**Why use built-in tools?**
- `grep_search`: Fast, indexed, returns structured results
- `semantic_search`: Understands concepts, not just text matching
- `list_code_usages`: Finds ALL usages including indirect references
- `file_search`: Quick file pattern matching without content overhead

**PowerShell alternative (if needed):**
```powershell
# User can run these manually
Select-String -Path src\thalia\* -Pattern "@register_region" -Recurse
Select-String -Path src\thalia\* -Pattern "def grow_output" -Recurse
```

## Type Alias Glossary

> **📚 For complete type alias documentation, see [TYPE_ALIASES.md](../api/TYPE_ALIASES.md)**

Key type patterns for understanding the codebase:

**Component Organization**:
- `ComponentGraph` - Maps region names to component instances
- `ConnectionGraph` - Maps (source, target) pairs to pathways
- `TopologyGraph` - Maps source regions to lists of target regions

**Multi-Source Routing**:
- `SourceSpec` - Defines a source with region name, optional port, size, and delay
- `SourceOutputs` - Maps source names to their spike outputs
- `SynapticWeights` / `LearningStrategies` - Per-source organization at dendrites

**State & Checkpointing**:
- `StateDict` - Component state tensors for checkpointing
- `CheckpointMetadata` - Training progress and stage information

**Port-Based Routing**:
- `SourcePort` / `TargetPort` - Optional layer/pathway identifiers (e.g., 'l23', 'feedforward')

See [TYPE_ALIASES.md](../api/TYPE_ALIASES.md) for complete definitions with usage contexts.

## Standard Growth API

**All `NeuralRegion` subclasses** implement these standardized growth methods:

### Growth Methods

```python
def grow_output(self, n_new: int) -> None:
    """Grow output dimension by adding neurons.

    Called when this component needs to produce more outputs.
    This adds neurons to the component's output population.

    Args:
        n_new: Number of output neurons/dimensions to add

    Effects:
        - Expands output-related weight matrices (adds rows)
        - Adds new neurons to neuron population
        - Expands output-side state tensors (membrane, traces, etc.)
        - Updates config.n_output

    Example:
        >>> region.n_output  # 100
        >>> region.grow_output(20)
        >>> region.n_output  # 120
    """

def grow_input(self, n_new: int) -> None:
    """Grow input dimension to accept more inputs.

    Called when upstream components grow their outputs.
    This expands the component's receptive field.

    Args:
        n_new: Number of input dimensions to add

    Effects:
        - Expands input-related weight matrices (adds columns)
        - Does NOT add neurons (neuron count unchanged)
        - Expands input-side state tensors (traces, buffers)
        - Updates config.n_input

    Example:
        >>> region.n_input  # 256
        >>> region.grow_input(32)
        >>> region.n_input  # 288
    """
```

### Multi-Source Pathway Growth

```python
def grow_source(self, source_name: str, new_size: int) -> None:
    """Grow input dimension for a specific source (MultiSourcePathway only).

    Called when one source region in a multi-source connection grows.
    Only that source's contribution to the total input expands.

    Args:
        source_name: Name of the source region that grew
        new_size: New total size for that source (not delta!)

    Effects:
        - Updates self.input_sizes[source_name]
        - Resizes weight matrix to accommodate new total input size
        - Preserves weights from other sources
        - Updates config.n_input to sum of all source sizes

    Example:
        >>> pathway.input_sizes  # {'cortex': 100, 'hippocampus': 64, 'pfc': 32}
        >>> pathway.grow_source('cortex', 120)  # Cortex grew by 20
        >>> pathway.input_sizes  # {'cortex': 120, 'hippocampus': 64, 'pfc': 32}
        >>> pathway.config.n_input  # 216 (was 196)
    """
```

## Common Tasks

### 1. Adding a New Region

**Search Strategy:**
```python
# 1. Find existing region examples
grep_search(query="@register_region", isRegexp=False, includePattern="src/thalia/regions/**")

# 2. Look at similar region for template (use semantic search)
semantic_search(query="prefrontal cortex region implementation with working memory")

# 3. Find base class requirements
list_code_usages(symbolName="NeuralComponent")
```

**Required Methods:**
- `__init__(config)` - Initialize with config
- `forward(input, **kwargs)` - Process one timestep
- `reset_state()` - Clear state between episodes
- `get_diagnostics()` - Return health metrics
- `grow_output(n_new)` - Add output neurons
- `grow_input(n_new)` - Expand receptive field

**Steps:**
1. Create `src/thalia/regions/<name>/<name>.py`
2. Create `<Name>Config` dataclass
3. Inherit from `NeuralComponent`
4. Add `@register_region("name", config_class=<Name>Config)`
5. Implement required methods
6. Update `docs/architecture/ARCHITECTURE_OVERVIEW.md`

### 2. Debugging Growth Issues

**Search Strategy:**
```python
# Find growth trigger points
grep_search(query="check_growth", isRegexp=False, includePattern="src/thalia/training/**")

# Find growth implementations
list_code_usages(symbolName="grow_output")
list_code_usages(symbolName="grow_input")

# Find pathway growth coordination
semantic_search(query="pathway growth coordination when components change size")
```

**Common Issues:**
- **Size mismatch**: Check that pathway dimensions match region sizes
  - Pathway input should match source output
  - Pathway output should match target input
- **Duplicate growth**: Check that component IDs don't get added twice
- **Wrong dimension**: Check if growing output vs input
- **Multi-source**: Check if using `grow_source()` for multi-source pathways

**Debug Checklist:**
- [ ] Component actually grew? (check `component.n_output`)
- [ ] Pathway exists in brain.connections?
- [ ] MultiSourcePathway or single-source?
- [ ] For MultiSourcePathway: source_name in pathway.input_sizes?
- [ ] Pathway weights shape matches new dimensions?
- [ ] Target region can accept new input size?

### 3. Finding Learning Rules

**Search Strategy:**
```python
# Find all learning strategies
grep_search(query="class.*Strategy", isRegexp=True, includePattern="src/thalia/learning/**")

# Find STDP implementations
semantic_search(query="STDP spike timing dependent plasticity implementation")

# Find where strategies are applied
list_code_usages(symbolName="compute_update")
```

**Learning Rule Locations:**
- `src/thalia/learning/rules/stdp.py` - STDP implementation
- `src/thalia/learning/rules/bcm.py` - BCM plasticity
- `src/thalia/learning/rules/hebbian.py` - Hebbian learning
- `src/thalia/learning/strategy_registry.py` - Strategy factory
- `src/thalia/learning/` - Strategy pattern implementations

**Integration Points:**
- Regions: Learning happens in `forward()` method
- Pathways: STDP applied automatically during spike transmission
- Strategy pattern: `learning_strategy.compute_update(...)`

### 4. Tracing Event Flow

**Search Strategy:**
```python
# Find event scheduler
list_code_usages(symbolName="EventScheduler")

# Find event-driven execution
semantic_search(query="event-driven brain execution and spike scheduling")

# Find event scheduling for specific pathway types
grep_search(query="_schedule_downstream_events", isRegexp=False, includePattern="src/thalia/core/**")
```

**Event Flow:**
1. **Start**: `dynamic_brain.py` → search: `"def _forward_event_driven"`
2. **Schedule**: Search: `"def _schedule_downstream_events"`
3. **Transform**: Pathways process spikes in their `forward()`
4. **Deliver**: Target region receives via `forward()`

**Multi-Source Event Flow:**
1. Source fires → output scheduled
2. `_schedule_downstream_events` checks if `MultiSourcePathway`
3. If multi-source: buffer in `_multi_source_buffers[target][source]`
4. When all sources ready: forward through pathway
5. Clear buffer after processing

### 5. Understanding Port-Based Routing

**Search Strategy:**
```python
# Find port extraction logic
list_code_usages(symbolName="_extract_port")

# Find layered cortex outputs
semantic_search(query="cortex layer specific outputs L23 L5 port routing")

# Find port specifications
grep_search(query="source_port|target_port", isRegexp=True, includePattern="src/thalia/core/**")
```

**Port Types:**
- **Source Ports**: Layer-specific outputs (cortex: 'l23', 'l5', 'l4')
- **Target Ports**: Input types ('feedforward', 'top_down', 'ec_l3', 'pfc_modulation')

**How It Works:**
1. Connection specifies: `source_port='l5'`
2. BrainBuilder calculates L5 size during construction
3. DynamicBrain extracts L5 slice from cortex output
4. Pathway receives only L5 portion

### 6. Working with Multi-Source Pathways

**Search Strategy:**
```python
# Find multi-source pathway implementation
list_code_usages(symbolName="MultiSourcePathway")

# Understand how they're created and used
semantic_search(query="multi-source pathway construction and event buffering")

# Find specific implementation details
grep_search(query="_multi_source_buffers", isRegexp=False, includePattern="src/thalia/core/**")
```

**Key Concepts:**
- Created automatically when multiple connections target same region
- Each source tracked individually: `pathway.input_sizes = {'cortex': 100, 'hippo': 64}`
- Forward pass requires dict: `pathway.forward({'cortex': spikes1, 'hippo': spikes2})`
- Growth per source: `pathway.grow_source('cortex', new_size=120)`

**Search Patterns:**
- Creation logic: `brain_builder.py` → search: `"connections_by_target"`
- Event buffering: `dynamic_brain.py` → search: `"isinstance(pathway, MultiSourcePathway)"`
- Growth coordination: `dynamic_pathway_manager.py` → search: `"grow_source"`

## File Organization

### Core Architecture
- `src/thalia/core/brain_builder.py` - Fluent API for brain construction
- `src/thalia/core/dynamic_brain.py` - Main brain implementation
- `src/thalia/regions/base.py` - NeuralComponent base class
- `src/thalia/regions/` - All brain regions (cortex, hippocampus, etc.)
- `src/thalia/pathways/` - Inter-region connections

### Learning & Plasticity
- `src/thalia/learning/rules/` - Learning rule implementations
- `src/thalia/learning/strategy_registry.py` - Strategy pattern factory
- `src/thalia/components/synapses/` - Synaptic models (STDP, STP)

### Growth & Coordination
- `src/thalia/coordination/growth.py` - Growth triggering logic
- `src/thalia/pathways/dynamic_pathway_manager.py` - Pathway growth coordination

### Training
- `src/thalia/training/curriculum/stage_manager.py` - Main training orchestrator
- `src/thalia/training/curriculum/` - Curriculum training subsystems
- `src/thalia/datasets/` - Dataset loaders and generators

### Documentation
- `docs/architecture/ARCHITECTURE_OVERVIEW.md` - **Start here!**
- `docs/patterns/` - Design patterns used in codebase
- `.github/copilot-instructions.md` - AI assistant context

## Search Patterns by Topic

### Component Registration
```python
grep_search(query="@register_region", isRegexp=False)
grep_search(query="@register_pathway", isRegexp=False)
grep_search(query="@register_module", isRegexp=False)
```

### Configuration
```python
grep_search(query="Config.*:", isRegexp=True)  # Find all config classes
grep_search(query="from thalia.config import", isRegexp=False)  # Config usage
```

### State Management
```python
list_code_usages(symbolName="get_state")
list_code_usages(symbolName="load_state")
list_code_usages(symbolName="reset_state")
```

### Diagnostics
```python
list_code_usages(symbolName="get_diagnostics")
list_code_usages(symbolName="check_health")
grep_search(query="HealthReport", isRegexp=False)
```

### Checkpointing
```python
list_code_usages(symbolName="CheckpointManager")
list_code_usages(symbolName="save_checkpoint")
list_code_usages(symbolName="load_checkpoint")
```

## Architecture Concepts

### Component Graph
The brain is a directed graph:
- **Nodes**: Regions, pathways, modules (all `NeuralComponent`)
- **Edges**: Data flow (spike transmission)
- **Execution**: Topological order or event-driven

### Growth System
Components grow when capacity is saturated:
1. **Detection**: Growth manager monitors utilization
2. **Trigger**: Component grows output dimension
3. **Propagation**: Pathways grow to maintain compatibility
4. **Types**:
   - Output growth: Adds neurons (affects downstream pathways)
   - Input growth: Expands receptive field (affects weight matrices)

### Event-Driven Execution
Spikes are events with timestamps:
1. Region produces output → schedule downstream events
2. Events include axonal delay
3. Pathway processes spikes when event fires
4. Multi-source pathways buffer until all sources ready

### Port-Based Routing
Supports layer-specific connectivity:
- **Source ports**: Which layer sends (e.g., cortex L5)
- **Target ports**: What input type (e.g., feedforward vs top-down)
- Enables biological accuracy (L5 → striatum, L2/3 → other cortex)

## Common Search Commands

```python
# Find where a class is used
list_code_usages(symbolName="ClassName")

# Find method implementations
list_code_usages(symbolName="method_name")

# Find imports of a module
grep_search(query="from thalia.module import", isRegexp=False)

# Find all TODOs
grep_search(query="TODO|FIXME", isRegexp=True)

# Find error handling
grep_search(query="raise.*Error", isRegexp=True)

# Find assertions
grep_search(query="assert ", isRegexp=False)
```

## When Things Break

### Growth Failures
1. Search: `"def grow_output"` in component
2. Check: Weight matrix dimensions
3. Verify: Pathway manager called
4. Test: Component sizes before/after

### Training Failures
1. Search: `"def train_stage"` in stage_manager.py
2. Check: Task loader providing correct data
3. Verify: Brain forward pass succeeds
4. Test: Individual components with simple inputs

### Size Mismatches
1. Search: `"n_input\|n_output"` in error trace
2. Check: BrainBuilder size inference
3. Verify: Pathway dimensions match components
4. Test: Print shapes at each connection

### Import Errors
1. Check: `src/thalia/<module>/__init__.py` exports
2. Verify: Circular import detection (use `TYPE_CHECKING`)
3. Test: Import in isolation

## Best Practices

1. **Use search tools strategically**:
   - `semantic_search` for concepts and understanding
   - `list_code_usages` for finding all symbol references
   - `grep_search` for exact text/regex patterns
2. **Follow conventions** - Match existing patterns
3. **Check base classes** - Understand protocols first
4. **Read docstrings** - They contain architecture decisions
5. **Test incrementally** - Validate each change
6. **Update docs** - Keep architecture docs current

## Key Files to Understand

Priority order for understanding the system:

1. `docs/architecture/ARCHITECTURE_OVERVIEW.md` - System overview
2. `.github/copilot-instructions.md` - Coding patterns and constraints
3. `src/thalia/regions/base.py` - Component protocol
4. `src/thalia/core/dynamic_brain.py` - Brain implementation
5. `src/thalia/core/brain_builder.py` - Construction API
6. `src/thalia/pathways/spiking_pathway.py` - Pathway implementation
7. `src/thalia/training/curriculum/stage_manager.py` - Training orchestration
