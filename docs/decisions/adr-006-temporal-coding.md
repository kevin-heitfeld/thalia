# ADR-006: Temporal/Latency Coding for Sensory Pathways

**Status**: Implemented
**Date**: 2025-12-07
**Context**: Part of 1D bool tensor architecture migration (ADR-005)

## Decision

Sensory pathways use **temporal/latency coding** where information is encoded in the **timing** of spikes, not just their presence.

### Encoding Scheme

```python
# High activity (1.0) → early spike (t=0)
# Medium activity (0.5) → middle spike (t=10)
# Low activity (0.1) → late spike (t=19)
# Very low activity → no spike

latency = (1.0 - normalized_activity) * (n_timesteps - 1)
```

### Output Format

- **Shape**: `[n_timesteps, output_size]` (2D bool tensor)
- **Processing**: Brain consumes sequentially: `for t: brain.forward(spikes[t])`
- **Each timestep**: 1D tensor `[output_size]` of bool spikes
- **Timing**: Each neuron spikes at most once; timing encodes stimulus strength

## Rationale

### Biological Accuracy
- Real neurons encode information in spike timing with millisecond precision
- Latency coding observed in sensory systems (vision, audition, somatosensation)
- More powerful than rate coding: timing + presence conveys information

### Information Richness
- **Temporal resolution**: 20 timesteps provides 20 levels of precision
- **Sparsity**: Only neurons above threshold fire
- **Efficiency**: Single spike per neuron encodes strength via latency

### Architecture Fit
- Compatible with 1D bool single-brain architecture (ADR-005)
- Sensory pathways produce temporal sequences
- Brain processes sequentially (no batch dimension)

## Implementation

### VisualPathway

**RetinalEncoder** (`pathways.py` lines 288-433):
```python
def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """Input: [C, H, W] or [H, W] (single image)
       Output: [n_timesteps, output_size] (2D bool)"""

    # 1. Photoreceptor + temporal contrast
    # 2. Center-surround + ON/OFF channels
    # 3. Ganglion cell activity [output_size]

    # 4. Temporal/latency coding
    spikes = self._generate_temporal_spikes(ganglion_activity)
    return spikes, metadata
```

**VisualPathway** (`pathways.py` lines 448-477):
- Simplified: Just passes through retinal spikes
- Future: Add V1-like processing (edges, orientation) preserving temporal structure

### AuditoryPathway

**CochlearEncoder** (`pathways.py` lines 520-670):
```python
def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """Input: [samples] (single audio clip)
       Output: [n_timesteps, output_size] (2D bool)"""

    # 1. FFT spectrogram
    # 2. Mel filterbank (cochlear frequency channels)
    # 3. Hair cell processing + adaptation
    # 4. Project to output size

    # 5. Temporal/latency coding
    spikes = self._generate_temporal_spikes(output_activity)
    return spikes, metadata
```

**AuditoryPathway** (`pathways.py` lines 672-714):
- Simplified: Just passes through cochlear spikes
- Future: Add A1-like processing (spectrotemporal patterns) preserving temporal structure

### Core Algorithm

**_generate_temporal_spikes()** (in both RetinalEncoder and CochlearEncoder):
```python
def _generate_temporal_spikes(self, activity: torch.Tensor) -> torch.Tensor:
    """Convert activity [output_size] to spikes [n_timesteps, output_size]."""
    n_neurons = activity.shape[0]
    n_timesteps = self.config.n_timesteps

    spikes = torch.zeros(n_timesteps, n_neurons, dtype=torch.bool)

    # Normalize activity to [0, 1]
    activity_norm = (activity - activity.min()) / (activity.max() - activity.min() + 1e-6)

    # Latency coding: high activity → low latency (early spike)
    latencies = ((1.0 - activity_norm) * (n_timesteps - 1)).long()

    # Generate spikes (threshold-based)
    threshold = self.config.sparsity
    for n in range(n_neurons):
        if activity_norm[n].item() > threshold:
            t = int(latencies[n].item())
            spikes[t, n] = True

    return spikes  # [n_timesteps, output_size]
```

## Consequences

### Positive
✅ **Biological realism**: Matches how real sensory neurons work
✅ **Information density**: Timing + presence > presence alone
✅ **Temporal dynamics**: Brain can learn temporal patterns
✅ **Clean architecture**: 2D output (time × neurons) fits sequential processing

### Negative
❌ **Complexity**: More complex than single-timestep rate coding
❌ **Processing overhead**: Brain must process T timesteps per input
❌ **Parameter sensitivity**: Latency mapping depends on normalization

### Migration Path
- ✅ RetinalEncoder: Fully implemented with latency coding
- ✅ VisualPathway: Simplified (retinal passthrough)
- ✅ CochlearEncoder: Fully implemented with latency coding
- ✅ AuditoryPathway: Simplified (cochlear passthrough)
- ⏳ LanguagePathway: Pending update
- ⏳ All sensory pathway tests: Need updates for new format

## Alternatives Considered

### 1. **Rate Coding** (rejected)
- Encode as single timestep: `[output_size]`
- Information in spike probability/count
- ❌ Less biologically accurate
- ❌ Less information per spike
- ✅ Simpler implementation

### 2. **Population Coding** (rejected)
- Multiple neurons encode same value
- Information in population activity pattern
- ❌ Requires more neurons
- ❌ Not clearly better than latency coding

### 3. **Rank Order Coding** (considered for future)
- Only relative timing matters
- First spike wins
- ⏳ Could be added later as alternative encoding

## References

- Gerstner & Kistler (2002): Spiking Neuron Models, Chapter 11: Temporal Coding
- Thorpe et al. (2001): Spike-based strategies for rapid processing
- VanRullen & Thorpe (2001): Rate coding versus temporal order coding

## See Also

- **ADR-004**: Bool Spike Types (foundation for temporal coding)
- **ADR-005**: 1D Tensor Architecture (no batch dimension)
- **Component Parity**: `docs/patterns/component-parity.md` (regions and pathways)

---

**Implementation Files**:
- `src/thalia/sensory/pathways.py`: RetinalEncoder, CochlearEncoder, Visual/AuditoryPathway
- `temp/test_visual_pathway.py`: VisualPathway validation
- `temp/test_auditory_pathway.py`: AuditoryPathway validation
