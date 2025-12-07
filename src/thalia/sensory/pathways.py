"""
Sensory Pathways - Multimodal input encoding for the spiking brain.

This module provides the architecture for encoding different sensory modalities
(vision, audition, language, etc.) into spike patterns that feed into the
unified brain architecture.

How the Real Brain Does It:
===========================

1. SPECIALIZED RECEPTORS
   Each modality has specialized receptor cells:
   - Vision: Photoreceptors (rods/cones) in retina
   - Audition: Hair cells in cochlea (frequency-tuned)
   - Touch: Mechanoreceptors in skin
   - Proprioception: Muscle spindles, joint receptors

2. PRIMARY SENSORY CORTICES
   Each modality has dedicated cortical areas:
   - V1: Visual cortex (retinotopic, orientation columns)
   - A1: Auditory cortex (tonotopic, frequency maps)
   - S1: Somatosensory cortex (body maps/homunculus)

3. HIERARCHICAL PROCESSING
   Information flows through hierarchy:
   - Vision: V1 → V2 → V4 → IT (increasingly abstract)
   - Audition: A1 → belt → parabelt → STS

4. MULTIMODAL CONVERGENCE
   All modalities eventually converge:
   - Association areas (parietal, temporal)
   - Prefrontal cortex (executive integration)
   - Hippocampus (episodic binding)

Key Insight:
============
Once information is converted to SPIKES, the brain processes all modalities
the same way. The magic is in the encoding - after that, it's all spikes
flowing through the same neural circuits.

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

from thalia.core.pathway_protocol import BaseNeuralPathway


class Modality(Enum):
    """Sensory modalities."""
    VISION = "vision"
    AUDITION = "audition"
    LANGUAGE = "language"
    TOUCH = "touch"
    PROPRIOCEPTION = "proprioception"


@dataclass
class SensoryConfig:
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


class SensoryPathway(BaseNeuralPathway):
    """
    Abstract base class for sensory pathways.

    Inherits from BaseNeuralPathway, implementing the SensoryPathwayProtocol
    interface to provide a standardized way to encode raw sensory input
    into spike patterns.

    All modalities must implement:
    1. encode(): Convert raw input to spike patterns
    2. get_modality(): Return modality type
    3. reset_state(): Clear temporal state (inherited from Protocol)
    4. get_diagnostics(): Report pathway metrics (inherited from Protocol)

    The output format is standardized so the brain can
    process any modality uniformly.
    """

    def __init__(self, config: SensoryConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    @abstractmethod
    def encode(
        self,
        raw_input: Any,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode raw input to spike patterns.

        Args:
            raw_input: Modality-specific raw input
            **kwargs: Additional encoding parameters

        Returns:
            spikes: Spike patterns [batch, n_timesteps, output_size]
            metadata: Dictionary with encoding metadata
        """
        pass

    @abstractmethod
    def get_modality(self) -> Modality:
        """Return the modality type."""
        pass

    def reset_state(self) -> None:
        """
        Reset pathway temporal state.

        Default implementation does nothing. Override if pathway
        has temporal state (adaptation, traces, etc.).
        """
        pass

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

        Brain expects: [batch, output_size] per timestep
        This method handles any necessary reshaping.
        """
        # Standard format: [batch, n_timesteps, output_size]
        if spikes.dim() == 2:
            # [batch, output_size] -> [batch, 1, output_size]
            spikes = spikes.unsqueeze(1)
        return spikes


# =============================================================================
# VISUAL PATHWAY - Retinal Processing
# =============================================================================

@dataclass
class VisualConfig(SensoryConfig):
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
        size = 7
        sigma_center = 1.0
        sigma_surround = 2.0

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
        dog = dog.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        self.register_buffer("dog_filter", dog)

    def forward(
        self,
        image: torch.Tensor,
        reset_adaptation: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode image to retinal spike output.

        Args:
            image: Input image [batch, channels, height, width] or [batch, height, width]
            reset_adaptation: Reset temporal adaptation state

        Returns:
            spikes: Retinal output [batch, n_timesteps, output_size]
            metadata: Encoding statistics
        """
        # Ensure 4D input
        if image.dim() == 3:
            image = image.unsqueeze(1)  # Add channel dim

        batch = image.shape[0]

        if reset_adaptation:
            self.adaptation_state = torch.zeros_like(self.adaptation_state)
            self.adaptation_state = self.adaptation_state.expand(batch, -1, -1, -1)

        # 1. Photoreceptor response (log transform for light adaptation)
        photo_response = torch.log1p(image.clamp(min=0))

        # 2. Temporal contrast (respond to changes)
        if self.config.use_temporal_contrast:
            temporal_diff = photo_response - self.adaptation_state[:batch]
            self.adaptation_state = self.adaptation_state[:batch] * 0.9 + photo_response * 0.1
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
        on_flat = on_response.view(batch, -1)
        off_flat = off_response.view(batch, -1)
        combined = torch.cat([on_flat, off_flat], dim=-1)

        # 6. Project to output size
        ganglion_activity = self.spatial_pool(combined)

        # 7. Generate spikes over time
        spikes = self._generate_temporal_spikes(ganglion_activity)

        metadata = {
            "modality": "vision",
            "on_activity": on_flat.mean().item(),
            "off_activity": off_flat.mean().item(),
            "sparsity": (spikes > 0).float().mean().item(),
        }

        return spikes, metadata

    def _generate_temporal_spikes(
        self,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert ganglion activity to temporal spike train.

        Uses rate coding: higher activity = higher spike probability.
        Also includes latency coding: higher activity = earlier spikes.
        """
        batch, n_neurons = activity.shape
        n_timesteps = self.config.n_timesteps

        spikes = torch.zeros(
            batch, n_timesteps, n_neurons,
            device=activity.device,
        )

        # Normalize activity to [0, 1]
        activity_norm = torch.sigmoid(activity)

        for t in range(n_timesteps):
            # Spike probability based on activity
            spike_prob = activity_norm * self.config.sparsity * 2

            # Earlier timesteps have higher threshold (latency coding)
            latency_factor = 1.0 - (t / n_timesteps) * 0.5
            adjusted_prob = spike_prob * latency_factor

            # Generate spikes
            spikes[:, t, :] = (torch.rand_like(adjusted_prob) < adjusted_prob).float()

        return spikes


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

    def encode(
        self,
        raw_input: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode image to spikes."""
        # Retinal processing
        retinal_spikes, metadata = self.retina(raw_input, **kwargs)

        # V1 processing (on integrated spikes)
        integrated = retinal_spikes.mean(dim=1)  # [batch, output_size]
        v1_activity = self.v1_process(integrated)

        # Generate V1 output spikes
        v1_spikes = self._activity_to_spikes(v1_activity)

        metadata["v1_activity"] = v1_activity.mean().item()

        return v1_spikes, metadata

    def _activity_to_spikes(self, activity: torch.Tensor) -> torch.Tensor:
        """Convert V1 activity to spike trains."""
        batch, n_neurons = activity.shape
        n_timesteps = self.config.n_timesteps

        spikes = torch.zeros(batch, n_timesteps, n_neurons, device=activity.device)
        spike_prob = torch.sigmoid(activity) * self.config.sparsity * 2

        for t in range(n_timesteps):
            spikes[:, t, :] = (torch.rand_like(spike_prob) < spike_prob).float()

        return spikes

    def get_modality(self) -> Modality:
        return Modality.VISION


# =============================================================================
# AUDITORY PATHWAY - Cochlear Processing
# =============================================================================

@dataclass
class AuditoryConfig(SensoryConfig):
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
            torch.zeros(1, config.n_filters),
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
        Encode audio waveform to auditory nerve spikes.

        Args:
            audio: Audio waveform [batch, samples] or [batch, 1, samples]
            reset_adaptation: Reset adaptation state

        Returns:
            spikes: Auditory nerve output [batch, n_timesteps, output_size]
            metadata: Encoding statistics
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        batch = audio.shape[0]

        if reset_adaptation:
            self.adaptation_state = torch.zeros(batch, self.config.n_filters, device=audio.device)

        # 1. Compute spectrogram (like basilar membrane frequency decomposition)
        # Using simple FFT; could use STFT for time-frequency
        spec = torch.fft.rfft(audio, n=self.config.n_fft, dim=-1)
        magnitude = torch.abs(spec)

        # Take first n_fft//2+1 bins
        magnitude = magnitude[:, :self.config.n_fft // 2 + 1]

        # 2. Apply filterbank (cochlear frequency channels)
        cochlear_response = torch.matmul(magnitude, self.filterbank.T)

        # 3. Hair cell processing
        # Half-wave rectification (already positive from magnitude)
        # Compressive nonlinearity (like hair cell response)
        hair_cell_response = torch.pow(cochlear_response + 1e-6, 0.3)

        # 4. Adaptation (auditory nerve adapts to sustained sounds)
        adapted = hair_cell_response - self.adaptation_state[:batch] * 0.5
        adapted = F.relu(adapted)
        self.adaptation_state = self.adaptation_state[:batch] * 0.95 + hair_cell_response * 0.05

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
        """Convert auditory activity to spike trains."""
        batch, n_neurons = activity.shape
        n_timesteps = self.config.n_timesteps

        spikes = torch.zeros(batch, n_timesteps, n_neurons, device=activity.device)
        activity_norm = torch.sigmoid(activity)

        for t in range(n_timesteps):
            spike_prob = activity_norm * self.config.sparsity * 2
            spikes[:, t, :] = (torch.rand_like(spike_prob) < spike_prob).float()

        return spikes


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

    def encode(
        self,
        raw_input: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode audio to spikes."""
        # Cochlear processing
        cochlear_spikes, metadata = self.cochlea(raw_input, **kwargs)

        # A1 processing
        integrated = cochlear_spikes.mean(dim=1)
        a1_activity = self.a1_process(integrated)

        # Generate A1 output spikes
        a1_spikes = self._activity_to_spikes(a1_activity)

        metadata["a1_activity"] = a1_activity.mean().item()

        return a1_spikes, metadata

    def _activity_to_spikes(self, activity: torch.Tensor) -> torch.Tensor:
        """Convert A1 activity to spike trains."""
        batch, n_neurons = activity.shape
        n_timesteps = self.config.n_timesteps

        spikes = torch.zeros(batch, n_timesteps, n_neurons, device=activity.device)
        spike_prob = torch.sigmoid(activity) * self.config.sparsity * 2

        for t in range(n_timesteps):
            spikes[:, t, :] = (torch.rand_like(spike_prob) < spike_prob).float()

        return spikes

    def get_modality(self) -> Modality:
        return Modality.AUDITION


# =============================================================================
# LANGUAGE PATHWAY - Text/Token Processing
# =============================================================================

@dataclass
class LanguageConfig(SensoryConfig):
    """Configuration for language pathway."""
    vocab_size: int = 50257
    embedding_dim: int = 256
    use_position_encoding: bool = True
    max_seq_len: int = 1024


class LanguagePathway(SensoryPathway):
    """
    Language pathway for text/token input.

    Unlike vision/audition which process continuous signals,
    language processes discrete tokens. We use:
    1. Sparse Distributed Representations (SDR) for tokens
    2. Oscillatory position encoding
    3. Sequential spike generation
    """

    def __init__(self, config: LanguageConfig):
        super().__init__(config)
        self.language_config = config

        # Import from our existing encoder
        from thalia.language.encoder import (
            SpikeEncoder,
            SpikeEncoderConfig,
            EncodingType,
        )
        from thalia.language.position import (
            OscillatoryPositionEncoder,
            PositionEncoderConfig,
            PositionEncodingType,
        )

        # Token encoder
        encoder_config = SpikeEncoderConfig(
            vocab_size=config.vocab_size,
            n_neurons=config.output_size,
            n_timesteps=config.n_timesteps,
            sparsity=config.sparsity,
            device=config.device,
        )
        self.encoder = SpikeEncoder(encoder_config)

        # Position encoder (optional)
        if config.use_position_encoding:
            pos_config = PositionEncoderConfig(
                n_neurons=config.output_size // 4,
                max_positions=config.max_seq_len,
                n_timesteps=config.n_timesteps,
                device=config.device,
            )
            self.position_encoder = OscillatoryPositionEncoder(pos_config)

            # Mixer for content + position
            self.mixer = nn.Linear(
                config.output_size + config.output_size // 4,
                config.output_size,
                bias=False,
            )
        else:
            self.position_encoder = None
            self.mixer = None

    def encode(
        self,
        raw_input: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode token IDs to spikes.

        Args:
            raw_input: Token IDs [batch, seq_len]
            position_ids: Optional position indices

        Returns:
            spikes: Spike patterns [batch, seq_len, n_timesteps, output_size]
            metadata: Encoding statistics
        """
        batch, seq_len = raw_input.shape

        # Generate position IDs if needed
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=raw_input.device)
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)

        # Encode tokens
        content_spikes, sdr = self.encoder(raw_input, position_ids)
        # content_spikes: [batch, seq_len, n_timesteps, output_size]

        # Add position encoding if enabled
        if self.position_encoder is not None:
            pos_spikes = self.position_encoder(position_ids, as_spikes=True)
            # pos_spikes: [batch, seq_len, n_timesteps, pos_size]

            # Combine content and position
            combined = torch.cat([content_spikes, pos_spikes], dim=-1)

            # Mix to output size
            shape = combined.shape
            combined_flat = combined.view(-1, shape[-1])
            mixed = self.mixer(combined_flat)
            spikes = mixed.view(shape[0], shape[1], shape[2], -1)

            # Re-binarize
            spikes = (spikes > 0).float()
        else:
            spikes = content_spikes

        # Reshape for brain: [batch, seq_len, n_timesteps, output_size]
        # Brain processes one token at a time, so we'll flatten seq into batch dimension
        # when feeding to brain

        metadata = {
            "modality": "language",
            "seq_len": seq_len,
            "sdr_sparsity": sdr.mean().item(),
            "spike_sparsity": spikes.mean().item(),
        }

        return spikes, metadata

    def encode_for_brain(
        self,
        raw_input: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode tokens in format suitable for brain's process_sample.

        Returns spikes for one token at a time (for sequential processing),
        or all tokens flattened (for parallel processing).
        """
        spikes, metadata = self.encode(raw_input, **kwargs)

        # For brain: [batch * seq_len, n_timesteps, output_size]
        batch, seq_len, n_timesteps, output_size = spikes.shape
        brain_format = spikes.view(batch * seq_len, n_timesteps, output_size)

        return brain_format, metadata

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

        # Encode each modality
        for name, pathway in self.pathways.items():
            if name in inputs:
                spikes, metadata = pathway.encode(inputs[name])

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
    modalities: List[str] = ["vision", "audition", "language"],
    output_size: int = 256,
    device: str = "cpu",
) -> MultimodalPathway:
    """Create a multimodal pathway combining specified modalities."""
    pathways = {}

    if "vision" in modalities:
        pathways["vision"] = create_visual_pathway(output_size, device=device)

    if "audition" in modalities:
        pathways["audition"] = create_auditory_pathway(output_size, device=device)

    if "language" in modalities:
        pathways["language"] = create_language_pathway(output_size, device=device)

    return MultimodalPathway(pathways, output_size=output_size, device=device)
