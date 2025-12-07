# Thalia Checkpoint Format Specification

**Version**: 0.1.0  
**Status**: Design Phase  
**Last Updated**: December 7, 2025

> **Related Document**: See [`curriculum_strategy.md`](curriculum_strategy.md) for training stages and curriculum design.

## Overview

Custom binary format for persisting and restoring Thalia brain states across training sessions. Designed for:
- **Efficiency**: Compact representation, fast loading
- **Versioning**: Forward/backward compatibility
- **Incremental Growth**: Support adding neurons without full rewrite
- **Portability**: Language-agnostic (can read from C++, Rust, etc.)

## Why Custom Format?

**vs PyTorch (.pt)**:
- ❌ Python-specific, breaks between PyTorch versions
- ❌ Opaque format, hard to inspect or migrate
- ❌ No incremental updates
- ✅ But: Simple to implement initially

**vs HDF5**:
- ❌ Large dependency (h5py)
- ❌ Complex API
- ✅ But: Mature, battle-tested

**Custom Binary**:
- ✅ Full control over format evolution
- ✅ Optimized for our use case (sparse tensors, growth)
- ✅ Lightweight, no external dependencies
- ✅ Can embed domain-specific metadata
- ❌ But: Maintenance burden

## File Format Structure

### High-Level Layout

```
[HEADER]           256 bytes fixed
[METADATA]         Variable length JSON
[REGION_INDEX]     Variable length
[REGION_DATA...]   Multiple regions
[CONNECTIVITY]     Inter-region connections
[CHECKSUM]         32 bytes (SHA-256)
```

### Detailed Specification

#### 1. Header (256 bytes)

```
Offset | Size | Type    | Field              | Description
-------|------|---------|--------------------|--------------------------
0x0000 | 4    | char[4] | magic              | "THAL" (0x5448414C)
0x0004 | 2    | uint16  | major_version      | Format major version
0x0006 | 2    | uint16  | minor_version      | Format minor version
0x0008 | 2    | uint16  | patch_version      | Format patch version
0x000A | 2    | uint16  | flags              | Feature flags (see below)
0x000C | 8    | uint64  | timestamp          | Unix timestamp (seconds)
0x0014 | 8    | uint64  | metadata_offset    | Byte offset to metadata
0x001C | 8    | uint64  | metadata_length    | Metadata size in bytes
0x0024 | 8    | uint64  | region_index_offset| Byte offset to region index
0x002C | 8    | uint64  | region_index_length| Index size in bytes
0x0034 | 8    | uint64  | connectivity_offset| Byte offset to connectivity
0x003C | 8    | uint64  | connectivity_length| Connectivity size in bytes
0x0044 | 8    | uint64  | total_neurons      | Total neuron count
0x004C | 8    | uint64  | total_synapses     | Total synapse count
0x0054 | 8    | uint64  | training_steps     | Cumulative training steps
0x005C | 4    | uint32  | num_regions        | Number of brain regions
0x0060 | 4    | uint32  | checksum_type      | 0=none, 1=SHA256, 2=CRC32
0x0064 | 156  | byte[]  | reserved           | Reserved for future use
```

**Flags Field** (16 bits):
```
Bit  | Meaning
-----|----------------------------------------------------------
0    | Compressed (1=yes, 0=no)
1    | Compression type (0=zstd, 1=lz4) - only if bit 0 set
2    | Includes homeostatic state
3    | Includes neuromodulator baselines
4    | Sparse encoding enabled
5    | Mixed precision (FP16/FP32)
6    | Delta checkpoint (stores differences from base)
7-15 | Reserved
```

#### 2. Metadata Section (Variable)

JSON-encoded dictionary with human-readable information:

```json
{
  "version": "0.1.0",
  "thalia_version": "0.2.0",
  "created": "2025-12-07T14:30:00Z",
  "description": "Language learning experiment, stage 3",
  "tags": ["language", "curriculum", "stage3"],
  "growth_history": [
    {
      "step": 10000,
      "region": "cortex_l4",
      "neurons_added": 100,
      "reason": "high_utilization",
      "type": "growth"
    },
    {
      "step": 45000,
      "region": "striatum",
      "synapses_pruned": 234,
      "reason": "consolidation",
      "type": "pruning",
      "safety_level": "conservative",
      "importance_threshold": 0.001
    }
  ],
  "training_info": {
    "total_epochs": 150,
    "best_loss": 0.042,
    "curriculum_stage": 3,
    "dataset": "simple_grammar_v2"
  },
  "hardware": {
    "device": "cuda:0",
    "gpu_model": "RTX 4090",
    "precision": "float32"
  },
  "custom": {
    // User-defined metadata
  }
}
```

#### 3. Region Index (Variable)

Array of region entries for quick lookup:

```
Entry structure (40 bytes per region):
Offset | Size | Type     | Field           | Description
-------|------|----------|-----------------|---------------------------
0x00   | 32   | char[32] | region_name     | Null-terminated region name
0x20   | 8    | uint64   | data_offset     | Byte offset to region data
0x28   | 8    | uint64   | data_length     | Region data size in bytes
```

#### 4. Region Data (Multiple Sections)

Each region contains:

```
[REGION_HEADER]    64 bytes
[CONFIG]           Variable (JSON)
[WEIGHTS]          Variable (tensor data)
[HOMEOSTATIC]      Variable (optional)
[REGION_METADATA]  Variable (JSON)
```

**Region Header** (64 bytes):
```
Offset | Size | Type    | Field              | Description
-------|------|---------|--------------------|--------------------------
0x00   | 4    | uint32  | n_neurons          | Number of neurons
0x04   | 4    | uint32  | n_inputs           | Input dimension
0x08   | 4    | uint32  | neuron_type        | 0=LIF, 1=ConductanceLIF
0x0C   | 4    | uint32  | encoding_type      | 0=dense, 1=sparse_csr
0x10   | 8    | uint64  | config_offset      | Offset to config (relative)
0x18   | 8    | uint64  | config_length      | Config size
0x20   | 8    | uint64  | weights_offset     | Offset to weights
0x28   | 8    | uint64  | weights_length     | Weights size
0x30   | 8    | uint64  | homeostatic_offset | Offset to homeostatic
0x38   | 8    | uint64  | homeostatic_length | Homeostatic size
```

**Weight Encoding**:

*Dense Format*:
```
[TENSOR_HEADER]    32 bytes
[DATA]             Variable (raw float32/float16)

Tensor Header:
Offset | Size | Type    | Field      | Description
-------|------|---------|------------|---------------------------
0x00   | 1    | uint8   | dtype      | 0=float32, 1=float16, 2=int8
0x01   | 1    | uint8   | ndim       | Number of dimensions
0x02   | 2    | uint16  | reserved   |
0x04   | 8    | uint64  | shape[0]   | First dimension
0x0C   | 8    | uint64  | shape[1]   | Second dimension (if ndim>=2)
0x14   | 8    | uint64  | shape[2]   | Third dimension (if ndim>=3)
0x1C   | 4    | uint32  | checksum   | CRC32 of data
```

*Sparse CSR Format* (for sparse connectivity):
```
[CSR_HEADER]       32 bytes
[ROW_POINTERS]     (n_rows + 1) * 4 bytes
[COLUMN_INDICES]   nnz * 4 bytes
[VALUES]           nnz * 4 bytes (float32)

CSR Header:
Offset | Size | Type    | Field      | Description
-------|------|---------|------------|---------------------------
0x00   | 1    | uint8   | dtype      | Value dtype
0x01   | 3    | byte[]  | reserved   |
0x04   | 8    | uint64  | n_rows     | Number of rows
0x0C   | 8    | uint64  | n_cols     | Number of columns
0x14   | 8    | uint64  | nnz        | Non-zero count
0x1C   | 4    | uint32  | checksum   | CRC32
```

**Homeostatic State** (optional):
```
[HOMEOSTATIC_HEADER] 16 bytes
[THRESHOLDS]         n_neurons * 4 bytes (float32)
[SCALING_FACTORS]    n_neurons * 4 bytes (float32)

Header:
Offset | Size | Type    | Field              
-------|------|---------|-----------------------
0x00   | 4    | uint32  | n_neurons          
0x04   | 4    | uint32  | flags (reserved)
0x08   | 8    | uint64  | timestamp_updated
```

**Dynamic State** (CRITICAL for resuming training):
```
[DYNAMIC_STATE_HEADER] 32 bytes
[REGION_STATE]         Variable (current spikes, membrane potentials)
[LEARNING_STATE]       Variable (BCM thresholds, eligibility traces, STP efficacy)
[OSCILLATOR_STATE]     Variable (theta/gamma phases for hippocampus/cortex)
[NEUROMODULATOR_STATE] Variable (current dopamine, acetylcholine, norepinephrine)

Header:
Offset | Size | Type    | Field              
-------|------|---------|-----------------------
0x00   | 4    | uint32  | has_region_state   | 1 if RegionState present
0x04   | 4    | uint32  | has_learning_state | 1 if learning rule state present
0x08   | 4    | uint32  | has_oscillators    | 1 if oscillator state present
0x0C   | 4    | uint32  | has_neuromodulators| 1 if neuromodulator levels present
0x10   | 8    | uint64  | region_state_offset
0x18   | 8    | uint64  | learning_state_offset
```

**Critical State Components**:
- **RegionState**: Current spikes, membrane potentials, conductances, refractory state
- **Learning State**: BCM thresholds, eligibility traces, STP u/x values
- **Oscillator State**: Theta/gamma current phase, frequency, time_ms
- **Neuromodulator Levels**: Current dopamine, ACh, norepinephrine baselines

#### 5. Connectivity Section

Inter-region connections:

```
[CONNECTIVITY_HEADER]  32 bytes
[CONNECTION_ENTRIES]   Multiple entries

Header:
Offset | Size | Type    | Field              
-------|------|---------|-----------------------
0x00   | 4    | uint32  | num_connections
0x04   | 4    | uint32  | reserved
0x08   | 24   | byte[]  | reserved

Connection Entry (variable):
[ENTRY_HEADER]     64 bytes
[WEIGHT_MATRIX]    Variable (sparse/dense)
[DELAY_MATRIX]     Variable (optional)

Entry Header:
Offset | Size | Type     | Field              
-------|------|----------|-----------------------
0x00   | 32   | char[32] | source_region
0x20   | 32   | char[32] | target_region
0x40   | 8    | uint64   | weight_offset
0x48   | 8    | uint64   | weight_length
0x50   | 8    | uint64   | delay_offset (0 if none)
0x58   | 8    | uint64   | delay_length
```

#### 6. Checksum (32 bytes)

SHA-256 hash of all data from offset 0 to checksum_offset.

---

## Implementation Priority Summary

**Phase 1 (NOW - Week 1)**: Core I/O
- Custom binary format (header, metadata, region index)
- Basic save/load for weights and config
- Tensor serialization (dense and sparse CSR)
- Checksum validation

**Phase 2 (CRITICAL - Week 2)**: State Management
- RegionState serialization (spikes, membrane, conductances)
- Learning rule state (BCM thresholds, eligibility traces, STP)
- Oscillator state (theta/gamma phases)
- Neuromodulator levels
- Growth mechanisms (add neurons without disruption)

**Phase 3 (Week 3)**: Consolidation
- Synaptic scaling (not structural pruning)
- Long-window synapse importance tracking
- Task transition detection

**Phase 4 (Week 4)**: Curriculum Integration
- Stage-based checkpointing
- Growth history tracking
- Resume from any stage

**Phase 5 (Future)**: Optimization
- Compression (zstd/lz4)
- Mixed precision (FP16)
- Delta checkpoints (v2.0)
- Streaming/lazy loading (v1.1)

---

## Implementation Plan

### Phase 1: Core I/O (Week 1)

**Files to Create**:
- `src/thalia/io/__init__.py`
- `src/thalia/io/checkpoint.py` - Main checkpoint API
- `src/thalia/io/binary_format.py` - Low-level binary encoding/decoding
- `src/thalia/io/tensor_encoding.py` - Tensor serialization
- `tests/unit/test_checkpoint.py` - Unit tests

**API Design**:
```python
from thalia.io import BrainCheckpoint

# Save
BrainCheckpoint.save(
    brain,
    path="checkpoints/my_brain.thalia",
    metadata={
        "experiment": "language_learning",
        "stage": 3
    },
    compression=False,
    include_homeostatic=True
)

# Load
brain = BrainCheckpoint.load(
    path="checkpoints/my_brain.thalia",
    device="cuda:0"
)

# Inspect without loading
info = BrainCheckpoint.info("checkpoints/my_brain.thalia")
print(info['total_neurons'])
print(info['training_steps'])
print(info['metadata']['experiment'])

# Validate
is_valid, issues = BrainCheckpoint.validate("checkpoints/my_brain.thalia")
```

**Core Classes**:
```python
class BinaryWriter:
    """Low-level binary writing with checksums"""
    def write_header(self, header: Header) -> None
    def write_tensor(self, tensor: torch.Tensor, sparse: bool = False) -> None
    def write_json(self, data: dict) -> None
    def finalize(self) -> None  # Write checksum

class BinaryReader:
    """Low-level binary reading with validation"""
    def read_header(self) -> Header
    def read_tensor(self, offset: int, shape: tuple) -> torch.Tensor
    def read_json(self, offset: int, length: int) -> dict
    def validate_checksum(self) -> bool

class BrainCheckpoint:
    """High-level brain save/load API"""
    @staticmethod
    def save(brain: Brain, path: str, **kwargs) -> None
    
    @staticmethod
    def load(path: str, device: str = 'cpu') -> Brain
    
    @staticmethod
    def info(path: str) -> dict
    
    @staticmethod
    def validate(path: str) -> tuple[bool, list]
```

### Phase 2: State Management & Growth Support (Week 2)

**Priority: State management is CRITICAL for resuming training**

**Files to Create/Modify**:
- `src/thalia/io/state_serialization.py` - Save/load RegionState objects
- `src/thalia/core/growth.py` - Growth mechanisms
- `src/thalia/regions/base.py` - Add growth methods and state getters
- `tests/unit/test_state_serialization.py` - State save/load tests
- `tests/unit/test_growth.py` - Growth tests

**State Management API** (CRITICAL):
```python
class BrainRegion:
    def get_full_state(self) -> Dict[str, Any]:
        """
        Get complete state for checkpointing.
        
        Returns:
            dict containing:
            - 'weights': Learnable parameters
            - 'region_state': Current RegionState (spikes, membrane, etc.)
            - 'learning_state': BCM thresholds, eligibility, STP, etc.
            - 'oscillator_state': Phases if applicable
            - 'neuromodulator_state': Current levels
        """
        
    def load_full_state(self, state: Dict[str, Any]) -> None:
        """
        Restore complete state from checkpoint.
        
        Must restore:
        - Weight matrices
        - RegionState objects
        - Learning rule internal state
        - Oscillator phases
        - Neuromodulator baselines
        """
```

**Growth API**:
```python
class BrainRegion:
    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1
    ) -> None:
        """Add neurons without disrupting existing weights"""
        
    def prune_synapses(
        self,
        threshold: float = 0.01,
        min_age: int = 10000,  # Only prune synapses older than this (steps)
        importance_history_window: int = 50000  # Track usage over long window
    ) -> dict:
        """
        CONSERVATIVE pruning - only remove truly dead synapses.
        
        Safety mechanisms:
        - Only prune synapses that have been consistently weak AND unused
        - Never prune recently created synapses (let them stabilize)
        - Track importance over long time windows (not just recent activity)
        - Return metadata about what was pruned for logging
        
        Returns:
            dict with 'synapses_removed', 'neurons_removed', 'regions_affected'
        """
        
    def remove_neurons(
        self,
        n_remove: int,
        strategy: str = 'least_active'
    ) -> dict:
        """
        Remove truly unused neurons (RARELY needed, use with caution).
        
        Only triggers when:
        - Neurons have been silent for extended periods (>100k steps)
        - No strong incoming or outgoing connections
        - Not part of critical circuits (checked via connectivity analysis)
        
        Returns:
            dict with removed neuron indices and affected connections
        """
        
    def get_capacity_metrics(self) -> dict:
        """Return utilization metrics for growth decisions"""
        return {
            'firing_rate': float,
            'weight_saturation': float,
            'synapse_usage': float,
            'growth_recommended': bool,
            'pruning_safe': bool  # False if recent task changes detected
        }

class Brain:
    def check_growth_needs(self) -> dict:
        """Determine which regions need capacity expansion"""
        
    def auto_grow(self, threshold: float = 0.8) -> None:
        """Automatically grow regions based on need"""
        
    def auto_prune(
        self,
        enable: bool = False,  # DISABLED by default
        safety_level: str = 'conservative'  # 'conservative' or 'aggressive'
    ) -> None:
        """
        Automatically prune unused capacity (USE WITH EXTREME CAUTION).
        
        Disabled by default because:
        - Risk of catastrophic forgetting
        - Temporary inactivity != permanent irrelevance
        - Curriculum learning causes natural activity fluctuations
        
        If enabled, uses safety mechanisms:
        - Long observation windows (50k+ steps)
        - Never prune during task transitions
        - Preserve synapses with historical importance
        - Log all pruning decisions for rollback
        """
```

**Checkpoint Integration**:
- Growth history stored in metadata
- Weight matrices handle variable sizes
- Backward compatibility with older checkpoints

### Phase 3: Consolidation (Week 3)

**Note**: Pruning is DEFERRED to Stage 4+ when evidence shows it's needed.
Focus on synaptic scaling, not structural removal.

**Files to Create/Modify**:
- `src/thalia/training/consolidation.py` - Consolidation mechanisms
- `tests/unit/test_consolidation.py`

**Consolidation API**:
```python
class ConsolidationManager:
    """Manages synaptic consolidation through scaling, NOT pruning."""
    
    def consolidate_region(
        self,
        region: BrainRegion,
        importance_threshold: float = 0.1,
        strengthen_factor: float = 1.05,  # Modest strengthening
        weaken_factor: float = 0.98  # Very gentle weakening
    ) -> dict:
        """
        Strengthen important synapses, GENTLY weaken unused ones.
        
        Does NOT remove synapses - only adjusts strengths.
        Think "synaptic scaling" not "pruning".
        
        Biological inspiration:
        - Synaptic scaling during sleep
        - Maintains overall excitability
        - Preserves relative importance
        
        Returns metrics about changes made.
        """
        
    def track_synapse_usage(
        self,
        region: BrainRegion,
        window: int = 50000  # LONG window (not just recent 1000 steps)
    ) -> torch.Tensor:
        """
        Track which synapses are actively used over LONG time periods.
        
        Importance = weighted average over time:
        - Recent activity: 30% weight
        - Medium-term (last 10k steps): 40% weight  
        - Long-term (last 50k steps): 30% weight
        
        This prevents over-weighting temporarily inactive knowledge
        due to curriculum stage changes.
        """
        
    def detect_task_transition(self, brain: Brain) -> bool:
        """
        Detect if task distribution has recently changed.
        
        Returns True if:
        - Activity patterns shifted significantly
        - New regions becoming active
        - Different neuron populations firing
        
        When True, use more conservative consolidation to avoid
        weakening temporarily inactive but still important knowledge.
        """
        
# Pruning deferred to Phase 5+ (Stage 4 in curriculum)
# Only implement if evidence shows capacity saturation is a real problem
```

### Phase 4: Curriculum Training Integration (Week 4)

**Note**: For detailed curriculum training strategy, stages, and implementation,
see **[`curriculum_strategy.md`](curriculum_strategy.md)**.

**Checkpoint Integration Tasks**:
- Integrate checkpointing into curriculum trainer
- Save/load at stage boundaries
- Track growth history across curriculum stages
- Support resuming from any stage
- Implement backward compatibility for format evolution

**Files to Create**:
- `src/thalia/training/curriculum.py` - Curriculum training framework (see curriculum_strategy.md)
- `tests/integration/test_curriculum.py`
- `tests/integration/test_curriculum_checkpoints.py` - Checkpoint integration tests

**Key Requirements**:
```python
# Checkpoint manager should support curriculum workflow:
# 1. Auto-save at stage transitions
# 2. Track which stage each checkpoint represents
# 3. Allow resuming from any stage
# 4. Preserve growth history across stages
# 5. Support delta checkpoints for efficiency (v2.0)
```

### Phase 5: Optimization & Tools (Week 5+)

**Additional Features**:
- Compression support (zstd + lz4)
- Mixed precision (FP16/FP32)
- Checkpoint diffing (what changed between saves?)
- Checkpoint merging (combine multiple training runs)
- Export to ONNX/other formats
- Visualization tools (plot growth over time)

**Files to Create**:
- `src/thalia/io/compression.py` - Compression utilities (zstd/lz4)
- `src/thalia/io/precision.py` - Mixed precision conversion
- `src/thalia/io/diff.py` - Checkpoint diffing
- `src/thalia/io/export.py` - Export to other formats
- `tools/checkpoint_inspector.py` - CLI tool
- `tools/checkpoint_diff.py` - Compare checkpoints
- `tools/visualize_growth.py` - Plot growth history
- `tools/checkpoint_compress.py` - Compress existing checkpoints

**Compression API**:
```python
# Automatic compression based on extension
BrainCheckpoint.save(brain, "checkpoint.thalia.zst")  # zstd
BrainCheckpoint.save(brain, "checkpoint.thalia.lz4")  # lz4

# Or explicit
BrainCheckpoint.save(
    brain,
    "checkpoint.thalia",
    compression='zstd',  # or 'lz4' or None
    compression_level=3   # 1-22 for zstd, 1-12 for lz4
)

# Compress existing checkpoint
from thalia.io import compress_checkpoint
compress_checkpoint(
    "checkpoint.thalia",
    "checkpoint.thalia.zst",
    algorithm='zstd',
    level=9
)
```

**Mixed Precision API**:
```python
# Save with FP16 for large weights
BrainCheckpoint.save(
    brain,
    "checkpoint.thalia",
    precision_policy={
        'weights': 'fp16',      # Large weight matrices
        'biases': 'fp32',       # Keep biases in FP32
        'homeostatic': 'fp32'   # Critical parameters stay FP32
    }
)

# Or simple
BrainCheckpoint.save(
    brain,
    "checkpoint.thalia",
    mixed_precision=True  # Auto-convert large tensors to FP16
)
```

## Version Compatibility

### Version Numbering

Format version: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (can't read old files)
- **MINOR**: Backward compatible additions (can read old files)
- **PATCH**: Bug fixes, no format changes

### Migration Strategy

```python
class FormatMigrator:
    """Migrate checkpoints between format versions"""
    
    @staticmethod
    def migrate(
        source_path: str,
        target_path: str,
        target_version: str
    ) -> None:
        """Convert checkpoint to different format version"""
        
    @staticmethod
    def can_migrate(
        from_version: str,
        to_version: str
    ) -> bool:
        """Check if migration is possible"""
```

**Supported Migrations**:
- 0.1.x → 0.2.x: Automatic (add new fields with defaults)
- 0.2.x → 0.1.x: Lossy (drop new fields, warn user)
- 1.x → 2.x: Manual migration required

## Testing Strategy

### Unit Tests
- Binary encoding/decoding (tensors, JSON, headers)
- Checksum validation
- Sparse/dense tensor conversion
- Version compatibility

### Integration Tests
- Save/load full brain (all regions)
- Growth: add neurons, save, load, verify
- Curriculum: multi-stage training with checkpoints
- Error handling: corrupted files, missing data

### Performance Benchmarks
- Save time vs PyTorch .pt
- Load time vs PyTorch .pt
- File size vs PyTorch .pt
- Compression ratios

**Target Metrics**:
- Save time: <5 seconds for 1M neurons
- Load time: <10 seconds for 1M neurons
- File size: <50% of PyTorch .pt (uncompressed)
- Checksum validation: <1 second

## Security Considerations

### Potential Threats
1. **Malicious Checkpoints**: Crafted files that exploit deserialization
2. **Data Corruption**: Bit flips, partial writes
3. **Version Confusion**: Loading incompatible formats

### Mitigations
1. **Checksum Validation**: SHA-256 on all data
2. **Size Limits**: Reject files >10GB by default (configurable)
3. **Magic Number Check**: Verify "THAL" header
4. **Version Validation**: Refuse unsupported versions
5. **Bounded Reads**: Never read more than header specifies
6. **No Code Execution**: Pure data format, no pickle/eval

## File Extension

Use `.thalia` extension with optional suffixes:
- `my_brain.thalia` - Full checkpoint (uncompressed)
- `my_brain.thalia.zst` - zstd compressed (best ratio)
- `my_brain.thalia.lz4` - lz4 compressed (fastest)
- `my_brain.delta.thalia` - Delta checkpoint (v2.0)
- `my_brain.thalia.meta` - Metadata only (for indexing)

**File Size Examples** (estimated for 1M neuron brain):
```
Uncompressed:           2.0 GB
zstd (level 3):         0.7 GB  (3x compression)
lz4 (level 1):          1.0 GB  (2x compression, 5x faster)
Mixed precision (FP16): 1.0 GB  (50% savings on weights)
FP16 + zstd:            0.35 GB (6x total compression)
Delta checkpoint:       0.04 GB (20x savings, 98% unchanged weights)
```

## Example Usage

### Basic Save/Load
```python
from thalia import Brain
from thalia.io import BrainCheckpoint

# Create and train brain
brain = Brain(config)
train(brain, epochs=100)

# Save
BrainCheckpoint.save(
    brain,
    "checkpoints/stage1.thalia",
    metadata={"stage": 1, "accuracy": 0.85}
)

# Load later
brain = BrainCheckpoint.load("checkpoints/stage1.thalia")
```

### Progressive Growth
```python
from thalia.training import CurriculumTrainer, StageConfig

trainer = CurriculumTrainer(
    brain,
    checkpoint_dir="checkpoints/curriculum",
    growth_policy='auto'
)

# Stage 1: Simple patterns
trainer.train_stage(StageConfig(
    epochs=50,
    difficulty=0.3,
    data_config={'vocab_size': 100}
))

# Stage 2: More complex (brain auto-grows if needed)
trainer.train_stage(StageConfig(
    epochs=100,
    difficulty=0.6,
    data_config={'vocab_size': 500}
))

# Resume from stage 1 if needed
trainer.load_checkpoint(stage=1)
```

### Inspect Checkpoint
```python
from thalia.io import BrainCheckpoint

info = BrainCheckpoint.info("checkpoints/stage2.thalia")
print(f"Neurons: {info['total_neurons']}")
print(f"Training steps: {info['training_steps']}")
print(f"Growth events: {len(info['growth_history'])}")

for event in info['growth_history']:
    print(f"  Step {event['step']}: {event['region']} +{event['neurons_added']}")
```

### Lazy Loading (v1.1)
```python
# Load only specific regions
brain = BrainCheckpoint.load(
    "checkpoints/large_brain.thalia",
    regions=['cortex_l4', 'striatum'],  # Only load these
    lazy=True  # Other regions loaded on first access
)

# cortex_l4 and striatum are in memory
# Other regions load automatically when accessed
prediction = brain.regions['hippocampus'].forward(input)  # Loads hippocampus now
```

### Compression Examples
```python
# Save with zstd (long-term storage)
BrainCheckpoint.save(
    brain,
    "archive.thalia.zst",
    compression='zstd',
    compression_level=9  # Max compression
)

# Save with lz4 (frequent checkpointing during training)
BrainCheckpoint.save(
    brain,
    "training_checkpoint.thalia.lz4",
    compression='lz4',
    compression_level=1  # Fast compression
)

# Mixed precision for large brains
BrainCheckpoint.save(
    brain,
    "efficient.thalia.zst",
    mixed_precision=True,  # FP16 for weights
    compression='zstd'      # + compression = 6x smaller
)
```

### Delta Checkpoints (v2.0)
```python
# Curriculum learning with delta checkpoints
base_path = "checkpoints/stage0.thalia"
BrainCheckpoint.save(brain, base_path)  # Full checkpoint

# Train stage 1
train_stage(brain, stage=1)
BrainCheckpoint.save(
    brain,
    "checkpoints/stage1.delta.thalia",
    base_checkpoint=base_path,
    delta=True  # Only save differences
)

# Train stage 2
train_stage(brain, stage=2)
BrainCheckpoint.save(
    brain,
    "checkpoints/stage2.delta.thalia",
    base_checkpoint="checkpoints/stage1.delta.thalia",
    delta=True
)

# Load any stage (automatically resolves delta chain)
brain = BrainCheckpoint.load("checkpoints/stage2.delta.thalia")
```
```

## Design Decisions

1. **Compression**: ✅ **Support both zstd and lz4**
   - Implementation: Minimal work - add compression type field in header
   - zstd: Use for long-term storage (best compression ratio ~2-3x)
   - lz4: Use for frequent save/load cycles (fastest, ~1.5-2x compression)
   - Detection: Automatic based on file extension or header flag
   - Dependencies: `python-zstandard` and `lz4` packages (both lightweight)
   
2. **Mixed Precision**: ✅ **FP16 support**
   - Already planned in format (dtype field in tensor header)
   - Can mix FP32 and FP16 within same checkpoint
   - Strategy: FP16 for large weight matrices, FP32 for critical parameters
   - Expected savings: ~50% file size for weight-dominated checkpoints
   
3. **Distributed**: ⏸️ **Deferred**
   - Current scope: Single-device checkpoints
   - Future: Add multi-GPU sharding when needed
   - Format already supports extension via reserved fields
   
4. **Streaming/Lazy Loading**: ✅ **Planned for v1.1**
   - Implementation: Region index enables loading specific regions only
   - Use case: Load cortex layers on-demand as needed
   - Memory savings: Only load active regions (important for large brains)
   - API: `brain = BrainCheckpoint.load(path, regions=['cortex_l4', 'striatum'])`
   
5. **Delta Checkpoints**: ✅ **Planned for v2.0**
   - Store only weight differences from previous checkpoint
   - Huge savings during curriculum learning (most weights unchanged)
   - Implementation: Base checkpoint + delta chain
   - Format: `stage1.thalia` (base) + `stage2.delta.thalia` (diff) + ...
   - Reconstruction: Apply deltas in sequence to base checkpoint

## Future Extensions

### Potential Additions
- **Replay Buffers**: Store recent experiences for replay
- **Attention Maps**: Save attention/saliency for visualization
- **Gradient Stats**: Track gradient magnitudes for analysis
- **Provenance**: Full training history (configs, hyperparameters)
- **Multi-Brain**: Save ensembles of brains
- **Incremental**: Append-only format for online learning

### Versioning Plan
- **v0.1**: Basic save/load, dense tensors
- **v0.2**: Sparse tensors, compression (zstd/lz4), mixed precision (FP16)
- **v0.3**: Growth support, homeostatic state
- **v1.0**: Production-ready, stable format
- **v1.1**: Streaming/lazy loading (load regions on-demand)
- **v2.0**: Delta checkpoints (incremental saves), distributed support

### Delta Checkpoint Implementation (v2.0)

**Concept**: Store only changed weights since last checkpoint.

**File Structure**:
```
base_checkpoint.thalia           # Full checkpoint (stage 0)
stage1.delta.thalia              # Only differences from base
stage2.delta.thalia              # Only differences from stage1
stage3.delta.thalia              # Only differences from stage2
```

**Delta Format**:
```
[DELTA_HEADER]      64 bytes
[BASE_REFERENCE]    Variable (path/hash of base checkpoint)
[REGION_DELTAS]     Multiple regions

Delta Header:
Offset | Size | Type     | Field              
-------|------|----------|-----------------------
0x00   | 4    | char[4]  | magic ("ΔTHL")
0x04   | 4    | uint32   | delta_version
0x08   | 32   | byte[32] | base_checkpoint_hash (SHA-256)
0x28   | 8    | uint64   | base_step
0x30   | 8    | uint64   | current_step
0x38   | 18   | byte[]   | reserved

Region Delta (per region):
- Only regions with changed weights
- Sparse encoding of weight differences
- New neurons (if growth occurred)
- Threshold: Only store deltas > 1e-5
```

**Usage**:
```python
# Initial save (base)
BrainCheckpoint.save(brain, "stage0.thalia")

# Subsequent saves (delta)
BrainCheckpoint.save(
    brain,
    "stage1.delta.thalia",
    base_checkpoint="stage0.thalia",
    delta=True
)

# Load (automatically reconstructs from base + deltas)
brain = BrainCheckpoint.load("stage3.delta.thalia")
# Internally: stage0.thalia + stage1.delta + stage2.delta + stage3.delta

# Or load specific stage
brain = BrainCheckpoint.load("stage3.delta.thalia", resolve_deltas=True)
```

**Expected Savings**: 80-95% file size during curriculum learning (most weights stable)

## References

- PyTorch serialization: https://pytorch.org/docs/stable/notes/serialization.html
- HDF5 format: https://www.hdfgroup.org/solutions/hdf5/
- Safetensors (Hugging Face): https://github.com/huggingface/safetensors
- ONNX format: https://onnx.ai/

---

## Implementation Priority Summary

**Phase 1 (NOW - Week 1)**: Core I/O
- Custom binary format (header, metadata, region index)
- Basic save/load for weights and config
- Tensor serialization (dense and sparse CSR)
- Checksum validation

**Phase 2 (CRITICAL - Week 2)**: State Management
- RegionState serialization (spikes, membrane, conductances)
- Learning rule state (BCM thresholds, eligibility traces, STP)
- Oscillator state (theta/gamma phases)
- Neuromodulator levels
- Growth mechanisms (add neurons without disruption)

**Phase 3 (Week 3)**: Consolidation
- Synaptic scaling (not structural pruning)
- Long-window synapse importance tracking
- Task transition detection

**Phase 4 (Week 4)**: Curriculum Integration
- Stage-based checkpointing
- Growth history tracking
- Resume from any stage

**Phase 5 (Future)**: Optimization
- Compression (zstd/lz4)
- Mixed precision (FP16)
- Delta checkpoints (v2.0)
- Streaming/lazy loading (v1.1)

---

**Status**: Ready for implementation  
**Next Steps**: Begin Phase 1 (Core I/O)  
**Estimated Timeline**: 5 weeks to full implementation
