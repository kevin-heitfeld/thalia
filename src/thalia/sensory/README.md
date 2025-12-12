# Sensory Processing Module (PLANNED)

**Status:** Not yet implemented

## Current State

The `sensory/` module is currently empty. Sensory encoding functionality is currently handled by:

### Existing Sensory Components

1. **Spike Encoding (Language)**
   - Location: `language/encoder.py`
   - Class: `SpikeEncoder`
   - Handles: Text → spike patterns conversion
   - Encoding types: Rate coding, temporal coding, SDR (Sparse Distributed Representation)

2. **Visual Pathway**
   - Location: `pathways/sensory_pathways.py`
   - Class: `VisualPathway`
   - Handles: Retina-like preprocessing, spike generation from visual input
   - Features: Contrast detection, temporal dynamics

3. **Auditory Pathway**
   - Location: `pathways/sensory_pathways.py`
   - Class: `AuditoryPathway`
   - Handles: Cochlea-like frequency decomposition, spike generation
   - Features: Frequency selectivity, temporal precision

4. **Language Pathway**
   - Location: `pathways/sensory_pathways.py`
   - Class: `LanguagePathway`
   - Handles: Text encoding via embedding → spike conversion

5. **Dataset Loaders**
   - Location: `training/datasets/loaders.py`
   - Functions: MNIST → spike encoding, temporal sequence generation
   - Handles: Dataset-specific preprocessing

## Why This Module is Empty

The current architecture implements sensory processing as **pathways** rather than standalone regions, following the biological principle that sensory systems are:

1. **Thalamic relays**: Sensory input goes through thalamus (implemented in `regions/thalamus.py`)
2. **Pathway transformations**: Sensory-to-cortical connections have their own neurons and learning (implemented as `SensoryPathway` subclasses)
3. **Distributed**: Sensory processing is not a single "region" but a distributed transformation pipeline

This design choice aligns with biological sensory systems where:
- Visual: Retina → LGN (thalamus) → V1 (cortex)
- Auditory: Cochlea → MGN (thalamus) → A1 (cortex)
- Each stage is a neural population with its own dynamics

## Future Plans

If we decide to implement dedicated sensory regions, they would include:

### Planned Components

1. **Retina Model** (`sensory/retina.py`)
   - ON/OFF center-surround cells
   - Magnocellular (motion) and parvocellular (detail) pathways
   - Temporal contrast adaptation

2. **Cochlea Model** (`sensory/cochlea.py`)
   - Basilar membrane frequency decomposition
   - Inner hair cell transduction
   - Auditory nerve spike generation

3. **Somatosensory Cortex** (`sensory/somatosensory.py`)
   - S1/S2 processing
   - Tactile feature extraction
   - Body map organization

### Design Considerations

**Pros of dedicated sensory regions:**
- More biological detail (e.g., center-surround receptive fields)
- Explicit sensory preprocessing stages
- Easier to model sensory-specific phenomena (adaptation, gain control)

**Cons (why current design may be better):**
- Adds complexity for minimal gain in current tasks
- Sensory processing already working via pathways + thalamus
- Would duplicate functionality from `pathways/sensory_pathways.py`

## How to Work with Current Sensory System

### Example: Add a New Sensory Modality

```python
# Option 1: Add to pathways/sensory_pathways.py
class TactilePathway(SensoryPathway):
    """Somatosensory input → spike encoding."""
    def forward(self, tactile_input: torch.Tensor) -> torch.Tensor:
        # Preprocess tactile signals
        processed = self._apply_receptive_fields(tactile_input)
        # Generate spikes
        spikes = self._encode_to_spikes(processed)
        return spikes

# Option 2: If more complex, create dedicated sensory/tactile.py
# Then connect via pathway: Tactile → Thalamus → Somatosensory Cortex
```

### Example: Modify Visual Encoding

```python
# Modify pathways/sensory_pathways.py:
class VisualPathway(SensoryPathway):
    def _apply_gabor_filters(self, image):
        """Add orientation-selective preprocessing."""
        # Implement Gabor filter bank
        ...
```

## References

- **Current sensory implementations**: See `pathways/sensory_pathways.py` docstrings
- **Thalamic relay**: See `regions/thalamus.py` for gating and modulation
- **Spike encoding**: See `language/encoder.py` for encoding strategies
- **Architecture diagram**: See `docs/architecture/README.md` (if exists)

## Questions?

If you're unsure where to add sensory functionality:
1. **Simple preprocessing** → Add to existing `SensoryPathway` subclass
2. **New modality (simple)** → Add to `pathways/sensory_pathways.py`
3. **Complex sensory model** → Create dedicated `sensory/<modality>.py` and connect via pathway

For questions about sensory architecture, see:
- `docs/design/architecture.md`
- `.github/copilot-instructions.md`
