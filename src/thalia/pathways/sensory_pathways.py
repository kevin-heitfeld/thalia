"""
Sensory Pathways - Multimodal Input Encoding for Spiking Neural Networks.

This module provides the architecture for encoding different sensory modalities
(vision, audition, language, touch) into spike patterns that feed into the
unified brain architecture.

**ARCHITECTURE DECISION: Why Pathways, Not Separate Sensory Regions?**
======================================================================

Thalia implements sensory processing as **pathways** rather than standalone
"sensory regions" because:

1. **Biological Reality**: Sensory systems are **transformation pipelines**, not
   isolated cortical regions. Information flows: Receptors → Thalamus → Primary
   Cortex → Higher Areas. Each stage is a neural population with learning.

2. **Pathway = Neural Population**: In Thalia, pathways ARE neural populations
   with weights, plasticity, and dynamics - not just connection matrices. This
   matches biology where LGN (thalamic relay) and V1 (primary visual cortex)
   are both active, learning populations.

3. **Import Clarity**: Sensory implementations live here in `pathways/`, not in
   a separate `sensory/` module. Import directly: `from thalia.pathways.sensory_pathways import VisualPathway`

**For Cortical Sensory Areas (V1, A1, S1):**
Use `regions/cortex/` with sensory-specific configurations when modeling
cortical sensory areas with full cortical microcircuits (L4→L2/3→L5).

**How the Real Brain Does It**:
===============================

1. **SPECIALIZED RECEPTORS**:
   Each modality has dedicated receptor cells:
   - **Vision**: Photoreceptors (rods/cones) in retina
   - **Audition**: Hair cells in cochlea (frequency-tuned)
   - **Touch**: Mechanoreceptors in skin (pressure, temperature)
   - **Proprioception**: Muscle spindles, joint receptors

2. **PRIMARY SENSORY CORTICES**:
   Each modality has dedicated cortical areas:
   - **V1**: Visual cortex (retinotopic maps, orientation columns)
   - **A1**: Auditory cortex (tonotopic maps, frequency analysis)
   - **S1**: Somatosensory cortex (body maps/homunculus)
   - **Language areas**: Wernicke's (comprehension), Broca's (production)

3. **HIERARCHICAL PROCESSING**:
   Information flows through processing hierarchy:
   - Vision: V1 → V2 → V4 → IT (simple → complex → abstract features)
   - Audition: A1 → belt → parabelt → STS (frequency → patterns → speech)

4. **MULTIMODAL CONVERGENCE**:
   All modalities eventually integrate:
   - Association areas (parietal, temporal lobes)
   - Prefrontal cortex (executive integration)
   - Hippocampus (episodic binding across modalities)

**Key Insight**:
================
Once converted to SPIKES, the brain processes all modalities using the SAME
circuits. The magic is in the encoding - after that, it's all just spikes
flowing through unified neural architecture.

Our Design:
===========

    Raw Input (any modality)
           │
           ▼
    ┌──────────────┐
    │   Sensory    │  Modality-specific preprocessing
    │   Encoder    │  (retina, cochlea, tokenizer, etc.)
    └──────┬───────┘
           │ Spike Patterns
           ▼
    ┌──────────────┐
    │   Primary    │  Modality-specific primary cortex
    │   Cortex     │  (V1, A1, language area)
    └──────┬───────┘
           │ Processed Spikes
           ▼
    ┌──────────────┐
    │  Unified     │  Shared processing (our existing Brain)
    │   Brain      │  Cortex → Hippocampus → PFC → Striatum
    └──────────────┘

FILE ORGANIZATION (1058 lines):
===============================
Lines 1-85:    Module docstring, imports
Lines 86-215:  SensoryPathwayConfig, VisualConfig classes
Lines 216-505: VisualPathway implementation (retina-like)
Lines 506-750: AuditoryPathway implementation (cochlea-like)
Lines 751-950: LanguagePathway implementation (embedding-based)
Lines 951-1058: Utility functions and sensory encoding helpers

NAVIGATION TIP: Use VSCode's "Go to Symbol" (Ctrl+Shift+O) to navigate between modalities.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.components.coding import compute_firing_rate
from thalia.core.protocols.component import LearnableComponent
from thalia.managers.component_registry import register_pathway


class Modality(Enum):
    """Sensory modalities."""
    VISION = "vision"
    AUDITION = "audition"
    LANGUAGE = "language"
    TOUCH = "touch"
    PROPRIOCEPTION = "proprioception"


@dataclass
class SensoryPathwayConfig:
    """Base configuration for sensory pathways.

    All sensory pathways share these parameters to ensure
    compatible output formats for brain integration.
    """
    # Output format (must match brain's input expectations)
    output_size: int = 256  # Number of output neurons
    n_timesteps: int = 20   # Timesteps per input

    # Sparse coding
    sparsity: float = 0.05  # Target sparsity

    # Timing
    dt_ms: float = 1.0

    device: str = "cpu"


class SensoryPathway(LearnableComponent):
    """
    Abstract base class for sensory pathways.

    Inherits from LearnableComponent, implementing the NeuralPathway protocol
    to provide a standardized way to encode raw sensory input into spike patterns.

    All modalities must implement:
    1. forward(): Convert raw input to spike patterns (standard PyTorch, ADR-007)
    2. get_modality(): Return modality type
    3. reset_state(): Clear temporal state (inherited from Protocol)
    4. get_diagnostics(): Report pathway metrics (inherited from Protocol)

    The output format is standardized so the brain can
    process any modality uniformly.
    """

    def __init__(self, config: SensoryPathwayConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    @abstractmethod
    def forward(
        self,
        raw_input: Any,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode raw input to spike patterns using temporal/latency coding.

        Temporal coding: Information encoded in WHEN neurons spike, not just IF.
        - Strong stimuli → early spikes (t=0, t=1)
        - Weak stimuli → late spikes (t=15, t=19)
        - Very weak → no spike in temporal window

        Args:
            raw_input: Modality-specific raw input (single input, not batch)
            **kwargs: Additional encoding parameters

        Returns:
            spikes: Spike train [n_timesteps, output_size] (2D bool, temporal coding)
                    Brain processes sequentially: spikes[t] is 1D [output_size]
            metadata: Dictionary with encoding metadata
        """
        ...

    @abstractmethod
    def get_modality(self) -> Modality:
        """Return the modality type."""
        ...

    def reset_state(self) -> None:
        """
        Reset pathway temporal state.

        Default implementation does nothing. Override if pathway
        has temporal state (adaptation, traces, etc.).
        """
        pass  # Default does nothing - override in subclasses with temporal state

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get pathway diagnostics.

        Default implementation returns basic info. Override to
        add pathway-specific metrics.
        """
        return {
            "modality": self.get_modality().value,
            "config": {
                "n_timesteps": getattr(self.config, 'n_timesteps', None),
                "output_size": getattr(self.config, 'output_size', None),
            }
        }

    def to_brain_format(
        self,
        spikes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ensure spikes are in the format expected by Brain.

        Brain expects: [n_timesteps, output_size] temporal spike train
        Brain consumes sequentially: for t: brain.forward(spikes[t])  # 1D
        """
        assert spikes.dim() == 2, (
            f"Sensory pathway output must be 2D [n_timesteps, output_size], "
            f"got shape {spikes.shape}. Use temporal/latency coding."
        )
        assert spikes.dtype == torch.bool, (
            f"Sensory pathway output must be bool (ADR-004), got {spikes.dtype}"
        )
        return spikes


# =============================================================================
# VISUAL PATHWAY - Retinal Processing
# =============================================================================

@dataclass
class VisualConfig(SensoryPathwayConfig):
    """Configuration for visual pathway.

    Models retinal processing:
    - Photoreceptors: Light intensity encoding
    - Bipolar cells: Center-surround processing
    - Ganglion cells: ON/OFF channels, motion detection
    """
    # Input image properties
    input_height: int = 28
    input_width: int = 28
    input_channels: int = 1  # Grayscale

    # Retinal processing
    use_center_surround: bool = True
    use_temporal_contrast: bool = True  # Respond to changes

    # Ganglion cell types
    n_on_cells: int = 128   # ON-center cells
    n_off_cells: int = 128  # OFF-center cells

    # DVS-style event encoding
    event_threshold: float = 0.1  # Change threshold for spike


class RetinalEncoder(nn.Module):
    """
    Retina-inspired visual encoder.

    Mimics the computational steps of the retina:
    1. Photoreceptors: Log-transform light intensity
    2. Horizontal cells: Spatial smoothing
    3. Bipolar cells: Center-surround (DoG filter)
    4. Ganglion cells: ON/OFF channels, temporal contrast

    This creates sparse, event-driven spike patterns similar
    to biological retinal output and DVS cameras.
    """

    def __init__(self, config: VisualConfig):
        super().__init__()
        self.config = config

        # Photoreceptor adaptation state
        self.register_buffer(
            "adaptation_state",
            torch.zeros(1, config.input_channels, config.input_height, config.input_width),
        )

        # Center-surround filters (Difference of Gaussians)
        self._create_dog_filters()

        # Spatial pooling to output size
        # ON and OFF cells each get half the output
        n_ganglion = config.n_on_cells + config.n_off_cells
        pool_size = config.input_height * config.input_width
        self.spatial_pool = nn.Linear(pool_size * 2, config.output_size)  # *2 for ON/OFF

    def _create_dog_filters(self) -> None:
        """Create Difference of Gaussians filters for center-surround."""
        size = DOG_FILTER_SIZE
        sigma_center = DOG_SIGMA_CENTER
        sigma_surround = DOG_SIGMA_SURROUND

        # Create coordinate grids
        x = torch.arange(size) - size // 2
        y = torch.arange(size) - size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # Gaussian functions
        center = torch.exp(-(xx**2 + yy**2) / (2 * sigma_center**2))
        surround = torch.exp(-(xx**2 + yy**2) / (2 * sigma_surround**2))

        # Normalize
        center = center / center.sum()
        surround = surround / surround.sum()

        # DoG = Center - Surround
        dog = center - surround
        # PyTorch conv2d requires [out_channels, in_channels, H, W]
        # This is a legitimate exception to ADR-005 (PyTorch API requirement)
        dog = dog.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        self.register_buffer("dog_filter", dog)

    def forward(
        self,
        image: torch.Tensor,
        reset_adaptation: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode image to retinal spike train using temporal/latency coding.

        Strong visual features spike early, weak features spike late.
        This mimics how retinal ganglion cells respond with lower latency
        to stronger stimuli.

        Args:
            image: Input image [channels, height, width] or [height, width] (single image)
            reset_adaptation: Reset temporal adaptation state

        Returns:
            spikes: Spike train [n_timesteps, output_size] (2D bool, temporal coding)
            metadata: Encoding statistics
        """
        # Ensure 3D input [channels, height, width]
        if image.dim() == 2:
            # Single channel image: [H, W] → [1, H, W]
            image = image.unsqueeze(0)  # Add channel dim

        assert image.dim() == 3, (
            f"RetinalEncoder expects [C, H, W] or [H, W], got shape {image.shape}"
        )

        # PyTorch conv2d requires [batch, channels, height, width]
        # This is a legitimate exception to ADR-005 (PyTorch API requirement)
        image = image.unsqueeze(0)  # [C, H, W] → [1, C, H, W]

        # Initialize adaptation state on first call
        if not hasattr(self, 'adaptation_state') or reset_adaptation:
            self.adaptation_state = torch.zeros(
                1, image.shape[1], image.shape[2], image.shape[3],
                device=image.device
            )

        # 1. Photoreceptor response (log transform for light adaptation)
        photo_response = torch.log1p(image.clamp(min=0))

        # 2. Temporal contrast (respond to changes)
        if self.config.use_temporal_contrast:
            temporal_diff = photo_response - self.adaptation_state
            self.adaptation_state = self.adaptation_state * RETINA_ADAPTATION_DECAY + photo_response * RETINA_ADAPTATION_RATE
        else:
            temporal_diff = photo_response

        # 3. Center-surround processing (DoG filter)
        if self.config.use_center_surround:
            cs_response = F.conv2d(
                temporal_diff,
                self.dog_filter.expand(temporal_diff.shape[1], -1, -1, -1),
                padding=3,
                groups=temporal_diff.shape[1],
            )
        else:
            cs_response = temporal_diff

        # 4. ON and OFF channels
        on_response = F.relu(cs_response)   # Responds to brightness increase
        off_response = F.relu(-cs_response)  # Responds to brightness decrease

        # 5. Flatten and combine ON/OFF
        # Remove batch dimension (was added for conv2d) and flatten spatial dims
        on_flat = on_response.squeeze(0).flatten()  # [1, C, H, W] → [features]
        off_flat = off_response.squeeze(0).flatten()  # [1, C, H, W] → [features]
        combined = torch.cat([on_flat, off_flat], dim=-1)  # [2*features]

        # 6. Project to output size: [2*features] → [output_size]
        # nn.Linear requires [batch, features], so temporarily add batch dim
        combined = combined.unsqueeze(0)  # [2*features] → [1, 2*features]
        ganglion_activity = self.spatial_pool(combined).squeeze(0)  # [1, output_size] → [output_size]

        # ADR-005: ganglion_activity is now 1D [output_size]
        assert ganglion_activity.dim() == 1, f"Expected 1D, got {ganglion_activity.shape}"

        # 7. Generate temporal spike train using latency coding
        # Higher activity → earlier spike (lower latency)
        # Weak activity → later spike (higher latency)
        spikes = self._generate_temporal_spikes(ganglion_activity)  # [n_timesteps, output_size]

        metadata = {
            "modality": "vision",
            "on_activity": on_flat.mean().item(),
            "off_activity": off_flat.mean().item(),
            "sparsity": compute_firing_rate(spikes),
            "mean_latency": self._compute_mean_latency(spikes),
        }

        return spikes, metadata

    def _generate_temporal_spikes(
        self,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert ganglion activity to temporal spike train using latency coding.

        Latency coding: Information encoded in spike timing
        - High activity (1.0) → spike at t=0 (immediate)
        - Medium activity (0.5) → spike at t=10 (middle)
        - Low activity (0.1) → spike at t=18 (late)
        - Very low activity → no spike in temporal window

        Each neuron spikes at most once; timing encodes strength.
        """
        n_neurons = activity.shape[0]  # [output_size]
        n_timesteps = self.config.n_timesteps

        spikes = torch.zeros(
            n_timesteps, n_neurons,
            dtype=torch.bool,
            device=activity.device,
        )

        # Normalize activity to [0, 1]
        activity_norm = (activity - activity.min()) / (activity.max() - activity.min() + 1e-6)

        # Latency coding: map activity to spike time
        # High activity (1.0) → t=0 (early spike)
        # Low activity (0.0) → t=n_timesteps-1 (late spike)
        latencies = ((1.0 - activity_norm) * (n_timesteps - 1)).long()  # [n_neurons]

        # Generate spikes at computed latencies
        # Only neurons above threshold spike
        threshold = self.config.sparsity
        for n in range(n_neurons):
            if activity_norm[n] > threshold:
                t = int(latencies[n].item())
                spikes[t, n] = True

        return spikes  # [n_timesteps, output_size]

    def _compute_mean_latency(self, spikes: torch.Tensor) -> float:
        """Compute mean spike latency (for diagnostics)."""
        n_timesteps = spikes.shape[0]
        spike_times: List[int] = []
        for t in range(n_timesteps):
            if spikes[t].any():
                spike_times.extend([t] * int(spikes[t].sum().item()))
        return float(sum(spike_times)) / len(spike_times) if spike_times else n_timesteps / 2.0


@register_pathway(
    "visual",
    aliases=["visual_pathway", "retinal_pathway"],
    description="Visual pathway from retinal encoding to cortical spikes",
    version="1.0",
    author="Thalia Project"
)
class VisualPathway(SensoryPathway):
    """Complete visual pathway from image to cortical input."""

    def __init__(self, config: VisualConfig):
        super().__init__(config)
        self.visual_config = config

        # Retinal encoder
        self.retina = RetinalEncoder(config)

        # Simple V1-like processing (edge detection, orientation)
        # Could be expanded to full V1 model
        self.v1_process = nn.Sequential(
            nn.Linear(config.output_size, config.output_size),
            nn.ReLU(),
        )

    def forward(
        self,
        raw_input: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode image to temporal spike train (standard PyTorch, ADR-007).

        Args:
            raw_input: Image tensor [C, H, W] or [H, W]

        Returns:
            spikes: Temporal spike train [n_timesteps, output_size] (2D bool)
            metadata: Encoding metadata
        """
        # Retinal processing already produces temporal spikes [n_timesteps, output_size]
        retinal_spikes, metadata = self.retina(raw_input, **kwargs)

        # Optional: V1 processing could go here (e.g., edge detection, orientation)
        # For now, just pass through retinal spikes
        # Future: Add simple V1-like processing that preserves temporal structure

        metadata["pathway"] = "visual"
        return retinal_spikes, metadata

    def get_modality(self) -> Modality:
        """Return visual modality."""
        return Modality.VISION


# =============================================================================
# AUDITORY PATHWAY - Cochlear Processing
# =============================================================================

@dataclass
class AuditoryConfig(SensoryPathwayConfig):
    """Configuration for auditory pathway.

    Models cochlear processing:
    - Basilar membrane: Frequency decomposition
    - Hair cells: Mechanical to electrical transduction
    - Auditory nerve: Spike encoding
    """
    # Audio properties
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160  # 10ms at 16kHz

    # Cochlear filterbank
    n_filters: int = 64  # Number of frequency channels
    f_min: float = 80.0   # Minimum frequency
    f_max: float = 7600.0  # Maximum frequency

    # Temporal integration
    integration_window_ms: float = 25.0


class CochlearEncoder(nn.Module):
    """
    Cochlea-inspired auditory encoder.

    Mimics cochlear processing:
    1. Basilar membrane: Frequency decomposition (like FFT)
    2. Hair cells: Half-wave rectification, compression
    3. Auditory nerve: Adaptation, spike generation

    Uses gammatone-like filterbank for biologically plausible
    frequency decomposition.
    """

    def __init__(self, config: AuditoryConfig):
        super().__init__()
        self.config = config

        # Create filterbank (simplified mel-scale, could use gammatone)
        self._create_filterbank()

        # Adaptation state for each frequency channel
        self.register_buffer(
            "adaptation_state",
            torch.zeros(config.n_filters),
        )

        # Project to output size
        self.output_projection = nn.Linear(config.n_filters, config.output_size)

    def _create_filterbank(self) -> None:
        """Create mel-scale filterbank weights."""
        # Simplified mel filterbank
        # In practice, could use torchaudio or implement gammatone
        n_fft = self.config.n_fft
        n_filters = self.config.n_filters

        # Create triangular filters on mel scale
        filterbank = torch.zeros(n_filters, n_fft // 2 + 1)

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * math.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(self.config.f_min)
        mel_max = hz_to_mel(self.config.f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_filters + 2)
        hz_points = torch.tensor([mel_to_hz(m) for m in mel_points])

        # Convert to FFT bin indices
        bin_points = (hz_points * n_fft / self.config.sample_rate).long()

        for i in range(n_filters):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising edge
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling edge
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        self.register_buffer("filterbank", filterbank)

    def forward(
        self,
        audio: torch.Tensor,
        reset_adaptation: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode audio waveform to auditory nerve spikes using temporal/latency coding.

        Args:
            audio: Audio waveform [samples] or [1, samples] (single audio clip)
            reset_adaptation: Reset adaptation state

        Returns:
            spikes: Spike train [n_timesteps, output_size] (2D bool, temporal coding)
            metadata: Encoding statistics
        """
        # Enforce 1D audio input (ADR-005)
        assert audio.dim() == 1, (
            f"AuditoryPathway.forward: Expected 1D audio [samples] (ADR-005), "
            f"got shape {audio.shape}"
        )

        # Initialize adaptation state on first call
        if not hasattr(self, 'adaptation_state') or reset_adaptation:
            self.adaptation_state = torch.zeros(self.config.n_filters, device=audio.device)

        # 1. Compute spectrogram (like basilar membrane frequency decomposition)
        # Using simple FFT; could use STFT for time-frequency
        spec = torch.fft.rfft(audio, n=self.config.n_fft, dim=-1)
        magnitude = torch.abs(spec)

        # Take first n_fft//2+1 bins
        magnitude = magnitude[:self.config.n_fft // 2 + 1]  # [n_fft//2+1]

        # 2. Apply filterbank (cochlear frequency channels)
        cochlear_response = torch.matmul(self.filterbank, magnitude)  # [n_filters]

        # 3. Hair cell processing
        # Half-wave rectification (already positive from magnitude)
        # Compressive nonlinearity (like hair cell response)
        hair_cell_response = torch.pow(cochlear_response + LATENCY_EPSILON, HAIR_CELL_COMPRESSION_EXPONENT)

        # 4. Adaptation (auditory nerve adapts to sustained sounds)
        adapted = hair_cell_response - self.adaptation_state * HAIR_CELL_ADAPTATION_SUPPRESSION
        adapted = F.relu(adapted)
        self.adaptation_state = self.adaptation_state * AUDITORY_NERVE_ADAPTATION_DECAY + hair_cell_response * AUDITORY_NERVE_ADAPTATION_RATE

        # 5. Project to output size
        output_activity = self.output_projection(adapted)

        # 6. Generate spikes over time
        spikes = self._generate_temporal_spikes(output_activity)

        metadata = {
            "modality": "audition",
            "cochlear_energy": cochlear_response.mean().item(),
            "sparsity": (spikes > 0).float().mean().item(),
        }

        return spikes, metadata

    def _generate_temporal_spikes(
        self,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert auditory activity to temporal spike train using latency coding.

        Latency coding: Information encoded in spike timing
        - High activity → early spike (t=0)
        - Low activity → late spike (t=n_timesteps-1)
        """
        n_neurons = activity.shape[0] if activity.dim() == 1 else activity.shape[-1]
        n_timesteps = self.config.n_timesteps

        # Flatten if needed
        if activity.dim() > 1:
            activity = activity.flatten()

        spikes = torch.zeros(n_timesteps, n_neurons, dtype=torch.bool, device=activity.device)

        # Normalize activity to [0, 1]
        activity_norm = (activity - activity.min()) / (activity.max() - activity.min() + 1e-6)

        # Latency coding: map activity to spike time
        latencies = ((1.0 - activity_norm) * (n_timesteps - 1)).long()

        # Generate spikes at computed latencies
        threshold = self.config.sparsity
        for n in range(n_neurons):
            if activity_norm[n].item() > threshold:
                t = int(latencies[n].item())
                spikes[t, n] = True

        return spikes  # [n_timesteps, output_size]


@register_pathway(
    "auditory",
    aliases=["auditory_pathway", "cochlear_pathway"],
    description="Auditory pathway from cochlear encoding to cortical spikes",
    version="1.0",
    author="Thalia Project"
)
class AuditoryPathway(SensoryPathway):
    """Complete auditory pathway from audio to cortical input."""

    def __init__(self, config: AuditoryConfig):
        super().__init__(config)
        self.auditory_config = config

        # Cochlear encoder
        self.cochlea = CochlearEncoder(config)

        # Simple A1-like processing
        self.a1_process = nn.Sequential(
            nn.Linear(config.output_size, config.output_size),
            nn.ReLU(),
        )

    def forward(
        self,
        raw_input: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode audio to temporal spike train.

        Args:
            raw_input: Audio waveform [samples] or [1, samples]

        Returns:
            spikes: Temporal spike train [n_timesteps, output_size] (2D bool)
            metadata: Encoding metadata
        """
        # Cochlear processing already produces temporal spikes [n_timesteps, output_size]
        cochlear_spikes, metadata = self.cochlea(raw_input, **kwargs)

        # Optional: A1 processing could go here (e.g., spectrotemporal patterns)
        # For now, just pass through cochlear spikes
        # Future: Add simple A1-like processing that preserves temporal structure

        metadata["pathway"] = "auditory"
        return cochlear_spikes, metadata

    def get_modality(self) -> Modality:
        """Return auditory modality."""
        return Modality.AUDITION



# =============================================================================
# LANGUAGE PATHWAY - Text/Token Processing
# =============================================================================

@dataclass
class LanguageConfig(SensoryPathwayConfig):
    """Configuration for language pathway."""
    vocab_size: int = 50257
    embedding_dim: int = 256
    use_position_encoding: bool = True
    max_seq_len: int = 1024


@register_pathway(
    "language",
    aliases=["language_pathway", "linguistic_pathway"],
    description="Language pathway from token encoding to cortical spikes",
    version="1.0",
    author="Thalia Project"
)
class LanguagePathway(SensoryPathway):
    """
    Language pathway for text/token input using temporal/latency coding.

    Processes discrete tokens (unlike continuous vision/audio signals):
    1. Token embeddings encode semantic information
    2. Temporal spikes encode embedding dimensions via latency
    3. Optional position encoding can be added

    Input: Single token ID (scalar or [1])
    Output: Temporal spike train [n_timesteps, output_size]
    """

    def __init__(self, config: LanguageConfig):
        super().__init__(config)
        self.language_config = config

        # Simple token embedding
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.output_size,
        )

        # Optional position encoding
        if config.use_position_encoding:
            # Store position embeddings
            self.position_embedding = nn.Embedding(
                config.max_seq_len,
                config.output_size,
            )
        else:
            self.position_embedding = None

    def forward(
        self,
        raw_input: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode token ID to temporal spike train using latency coding.

        Args:
            raw_input: Token ID (scalar or [1])
            position_ids: Optional position index (scalar or [1])

        Returns:
            spikes: Temporal spike train [n_timesteps, output_size] (2D bool)
            metadata: Encoding statistics
        """
        # Convert scalar token ID to 1D for nn.Embedding (PyTorch API requirement)
        # ADR-005: We operate on single tokens, but nn.Embedding requires [1] not []
        if raw_input.dim() == 0:
            token_id = raw_input.unsqueeze(0)  # scalar → [1]
        elif raw_input.dim() == 1:
            token_id = raw_input[:1]  # Take first token if multiple provided
        else:
            token_id = raw_input.flatten()[:1]  # Flatten and take first

        # Get token embedding [1, output_size] (nn.Embedding output)
        token_emb = self.embedding(token_id)  # [1, output_size]

        # Add position encoding if enabled
        if self.position_embedding is not None:
            if position_ids is None:
                position_ids = torch.tensor([0], device=token_id.device)
            elif position_ids.dim() == 0:
                position_ids = position_ids.unsqueeze(0)  # scalar → [1]
            elif position_ids.dim() == 1:
                position_ids = position_ids[:1]  # Take first position
            else:
                position_ids = position_ids.flatten()[:1]

            pos_emb = self.position_embedding(position_ids)  # [1, output_size]
            combined = token_emb + pos_emb
        else:
            combined = token_emb

        # Extract activity [output_size] (ADR-005: 1D output)
        activity = combined.squeeze(0)  # [1, output_size] → [output_size]
        assert activity.dim() == 1, f"Expected 1D activity, got {activity.shape}"

        # Generate temporal spikes via latency coding
        spikes = self._generate_temporal_spikes(activity)

        metadata = {
            "modality": "language",
            "token_id": token_id.item(),
            "sparsity": (spikes.sum().item() / spikes.numel()),
        }

        return spikes, metadata

    def _generate_temporal_spikes(
        self,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert token embedding to temporal spike train using latency coding.

        Latency coding: Information encoded in spike timing
        - High embedding value → early spike (t=0)
        - Low embedding value → late spike (t=n_timesteps-1)
        """
        n_neurons = activity.shape[0] if activity.dim() == 1 else activity.shape[-1]
        n_timesteps = self.config.n_timesteps

        # Flatten if needed
        if activity.dim() > 1:
            activity = activity.flatten()

        spikes = torch.zeros(n_timesteps, n_neurons, dtype=torch.bool, device=activity.device)

        # Normalize activity to [0, 1]
        activity_norm = (activity - activity.min()) / (activity.max() - activity.min() + 1e-6)

        # Latency coding: map activity to spike time
        latencies = ((1.0 - activity_norm) * (n_timesteps - 1)).long()

        # Generate spikes at computed latencies
        threshold = self.config.sparsity
        for n in range(n_neurons):
            if activity_norm[n].item() > threshold:
                t = int(latencies[n].item())
                spikes[t, n] = True

        return spikes  # [n_timesteps, output_size]

    def get_modality(self) -> Modality:
        return Modality.LANGUAGE



# =============================================================================
# MULTIMODAL INTEGRATION
# =============================================================================

class MultimodalPathway(nn.Module):
    """
    Integrates multiple sensory pathways into unified brain input.

    Handles:
    1. Temporal alignment across modalities
    2. Cross-modal attention for binding
    3. Unified spike output for brain
    """

    def __init__(
        self,
        pathways: Dict[str, SensoryPathway],
        output_size: int = 256,
        n_timesteps: int = 20,
        device: str = "cpu",
    ):
        super().__init__()
        self.pathways = nn.ModuleDict(pathways)
        self.output_size = output_size
        self.n_timesteps = n_timesteps
        self.device = torch.device(device)

        # Cross-modal integration layer
        # Sum of all pathway outputs should project to unified size
        total_input = sum(p.config.output_size for p in pathways.values())
        self.integration = nn.Linear(total_input, output_size)

        # Temporal alignment buffer
        self.register_buffer(
            "alignment_buffer",
            torch.zeros(1, n_timesteps, output_size),
        )

    def forward(
        self,
        inputs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process multimodal inputs.

        Args:
            inputs: Dict mapping modality name to raw input

        Returns:
            spikes: Unified spike output [batch, n_timesteps, output_size]
            metadata: Per-modality metadata
        """
        all_spikes = []
        all_metadata = {}

        # Encode each modality (all pathways use forward() per ADR-007)
        for name, pathway in self.pathways.items():
            if name in inputs:
                spikes, metadata = pathway(inputs[name])  # Callable syntax

                # Ensure consistent shape: [batch, n_timesteps, neurons]
                if spikes.dim() == 4:  # [batch, seq, time, neurons]
                    # For now, use first position (could do attention over sequence)
                    spikes = spikes[:, 0, :, :]

                all_spikes.append(spikes)
                all_metadata[name] = metadata

        if len(all_spikes) == 0:
            raise ValueError("No valid inputs provided")

        # Concatenate along neuron dimension
        combined = torch.cat(all_spikes, dim=-1)

        # Integrate to unified representation
        batch, n_timesteps, _ = combined.shape
        combined_flat = combined.view(batch * n_timesteps, -1)
        unified = self.integration(combined_flat)
        unified = unified.view(batch, n_timesteps, -1)

        # Convert to spikes
        spikes = (torch.sigmoid(unified) > 0.5).float()

        all_metadata["unified_sparsity"] = spikes.mean().item()

        return spikes, all_metadata

    def add_pathway(self, name: str, pathway: SensoryPathway) -> None:
        """Add a new sensory pathway."""
        self.pathways[name] = pathway
        # Would need to reinitialize integration layer


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_visual_pathway(
    output_size: int = 256,
    input_size: Tuple[int, int] = (28, 28),
    device: str = "cpu",
) -> VisualPathway:
    """Create a visual pathway for image input."""
    config = VisualConfig(
        output_size=output_size,
        input_height=input_size[0],
        input_width=input_size[1],
        device=device,
    )
    return VisualPathway(config)


def create_auditory_pathway(
    output_size: int = 256,
    sample_rate: int = 16000,
    device: str = "cpu",
) -> AuditoryPathway:
    """Create an auditory pathway for audio input."""
    config = AuditoryConfig(
        output_size=output_size,
        sample_rate=sample_rate,
        device=device,
    )
    return AuditoryPathway(config)


def create_language_pathway(
    output_size: int = 256,
    vocab_size: int = 50257,
    device: str = "cpu",
) -> LanguagePathway:
    """Create a language pathway for text input."""
    config = LanguageConfig(
        output_size=output_size,
        vocab_size=vocab_size,
        device=device,
    )
    return LanguagePathway(config)


def create_multimodal_pathway(
    modalities: Optional[List[str]] = None,
    output_size: int = 256,
    device: str = "cpu",
) -> MultimodalPathway:
    """Create a multimodal pathway combining specified modalities."""
    if modalities is None:
        modalities = ["vision", "audition", "language"]

    pathways = {}

    if "vision" in modalities:
        pathways["vision"] = create_visual_pathway(output_size, device=device)

    if "audition" in modalities:
        pathways["audition"] = create_auditory_pathway(output_size, device=device)

    if "language" in modalities:
        pathways["language"] = create_language_pathway(output_size, device=device)

    return MultimodalPathway(pathways, output_size=output_size, device=device)
