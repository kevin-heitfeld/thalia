"""ThalamusConfig"""

from __future__ import annotations

from dataclasses import dataclass

from thalia.errors import ConfigurationError

from .neural_region import NeuralRegionConfig


@dataclass
class ThalamusConfig(NeuralRegionConfig):
    """Configuration for thalamic relay nucleus.

    Thalamus sits between sensory input and cortex, providing:
    - Sensory gating (alpha-based suppression)
    - Mode switching (burst vs tonic)
    - Gain modulation (norepinephrine)
    - Spatial filtering
    """

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
    target_firing_rate: float = 0.08  # Target firing rate for homeostatic plasticity
    gain_learning_rate: float = 0.0001  # Let conductance scales work first
    gain_tau_ms: float = 10000.0  # Very slow adaptation

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    # =========================================================================
    threshold_learning_rate: float = 0.03
    threshold_min: float = 0.05
    threshold_max: float = 1.5

    # =========================================================================
    # BURST vs TONIC MODE SWITCHING
    # =========================================================================
    burst_threshold: float = -0.2
    """Membrane potential threshold for burst mode (hyperpolarized)."""

    tonic_threshold: float = 0.3
    """Membrane potential threshold for tonic mode (depolarized)."""

    burst_gain: float = 2.0
    """Amplification factor for burst mode (alerting signal)."""

    # =========================================================================
    # GAP JUNCTIONS: TRN INTERNEURONS
    # =========================================================================
    gap_junction_strength: float = 0.10  # Moderate coupling for alpha synchronization (8-13 Hz)
    gap_junction_threshold: float = 0.3  # Neighborhood connectivity threshold
    gap_junction_max_neighbors: int = 8  # Max neighbors per interneuron (biological: 4-12)

    # =========================================================================
    # SPATIAL FILTERING: CENTER-SURROUND RECEPTIVE FIELDS
    # =========================================================================
    spatial_filter_width: float = 0.15
    """Gaussian filter width for center-surround (as fraction of input)."""

    center_excitation: float = 3.0
    """Center enhancement in receptive field."""

    surround_inhibition: float = 0.5
    """Surround suppression in receptive field."""

    # =========================================================================
    # TEMPORAL DYNAMICS & DELAYS
    # =========================================================================
    trn_recurrent_delay_ms: float = 8.0
    """TRN→TRN recurrent inhibition delay (prevents instant feedback).

    Biological: 2-3ms local inhibition
    Set to 8ms to stabilize dynamics (must be > refractory period of 5ms)

    Note: For alpha oscillations (8-12 Hz), the relay→TRN→relay loop needs ~80-120ms total.
    This delay contributes to the overall loop timing. Additional delays come from:
    - Relay → TRN: ~2-5ms (synaptic)
    - TRN → Relay: ~3-5ms (inhibitory synaptic)
    - Gap junction synchronization: <1ms (electrical)
    - Total loop: Gap junctions + recurrent inhibition create alpha rhythm
    """

    # =========================================================================
    # BURST vs TONIC MODE THRESHOLD
    # =========================================================================
    mode_threshold: float = 0.5
    """Threshold separating burst mode (< threshold) from tonic mode (>= threshold).

    ``current_mode`` is a continuous value in [0, 1] computed by ``_determine_mode()``.
    Values below this threshold trigger burst amplification; values at or above use
    normal tonic relay. Different thalamic nuclei have different switching thresholds.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.burst_threshold >= self.tonic_threshold:
            raise ConfigurationError(
                f"burst_threshold ({self.burst_threshold}) must be < tonic_threshold ({self.tonic_threshold})"
            )
        if self.burst_gain <= 0:
            raise ConfigurationError(f"burst_gain must be > 0, got {self.burst_gain}")
        if self.spatial_filter_width <= 0:
            raise ConfigurationError(f"spatial_filter_width must be > 0, got {self.spatial_filter_width}")
        if self.trn_recurrent_delay_ms <= 0:
            raise ConfigurationError(f"trn_recurrent_delay_ms must be > 0, got {self.trn_recurrent_delay_ms}")
        if not (0.0 < self.mode_threshold < 1.0):
            raise ConfigurationError(f"mode_threshold must be in (0, 1), got {self.mode_threshold}")
