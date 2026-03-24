"""ThalamusConfig"""

from __future__ import annotations

from dataclasses import dataclass, field

from thalia.brain.gap_junctions import GapJunctionConfig
from thalia.brain.regions.population_names import ThalamusPopulation
from thalia.brain.synapses import NMReceptorType
from thalia.errors import ConfigurationError
from thalia.typing import NeuromodulatorChannel

from .neural_region import (
    HomeostaticGainConfig,
    HomeostaticThresholdConfig,
    NMReceptorConfig,
    NeuralRegionConfig,
)


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
    homeostatic_gain: HomeostaticGainConfig = field(default_factory=lambda: HomeostaticGainConfig(
        lr_per_ms=0.001,  # Reduced 0.005→0.001: effective tau must be ≥ 1000 ms (biological minimum for homeostasis)
        tau_ms=5000.0,  # Reduced 10000→5000: faster rate averaging tracks relay firing changes
    ))

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    # =========================================================================
    homeostatic_threshold: HomeostaticThresholdConfig = field(default_factory=lambda: HomeostaticThresholdConfig(
        lr_per_ms=0.001,  # Reduced 0.005→0.001: effective tau must be ≥ 1000 ms (biological minimum for homeostasis).
        # Previously 0.03 caused threshold crash from 0.80→0.05 in ~25ms during warmup.
        # Slower rate prevents catastrophic overshoot while still allowing adaptation.
        threshold_min=0.55,  # Raised 0.40→0.55: at 0.40 relay settled at 40 Hz (1.6× target).
        # Higher floor constrains excitability closer to the 25 Hz homeostatic target.
        threshold_max=1.5,
    ))
    homeostatic_target_rates: dict[str, float] = field(default_factory=lambda: {
        ThalamusPopulation.RELAY: 0.025,
    })

    neuromodulator_receptors: list[NMReceptorConfig] = field(default_factory=lambda: [
        NMReceptorConfig(NMReceptorType.ACH_MUSCARINIC_M1, NeuromodulatorChannel.ACH, "_ach_concentration_trn", (ThalamusPopulation.TRN,)),
        NMReceptorConfig(NMReceptorType.DA_D1, NeuromodulatorChannel.DA_MESOCORTICAL, "_da_concentration_relay", (ThalamusPopulation.RELAY,)),
        NMReceptorConfig(NMReceptorType.NE_ALPHA1, NeuromodulatorChannel.NE, "_ne_concentration_relay", (ThalamusPopulation.RELAY,)),
    ])

    # =========================================================================
    # GAP JUNCTIONS: TRN INTERNEURONS
    # =========================================================================
    gap_junctions: GapJunctionConfig = field(default_factory=lambda: GapJunctionConfig(
        coupling_strength=0.06,  # Increased: 0.04→0.02 worsened ρ 0.46→0.53. Stronger coupling smooths subthreshold dynamics
        connectivity_threshold=0.3,  # Neighborhood connectivity threshold
        max_neighbors=4,  # Reduced from 8: prevents pathological synchrony in small TRN population
    ))

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
    # BRAINSTEM ASCENDING AROUSAL DRIVE
    # =========================================================================
    relay_baseline_drive: float = 0.001  # Reduced 0.002→0.001: with external sensory (0.50 fraction) + L6B feedback
    # (0.50 fraction) + GPi inhibition, 0.002 caused E/I=5.5 and relay burst mode.
    # 0.001 provides sub-threshold arousal without dominating; cooperative drive from
    # other sources brings relay to tonic 20-30 Hz.
    """Per-step AMPA conductance added to relay neurons each timestep.

    Represents ascending arousal from brainstem nuclei (LC norepinephrine,
    cholinergic pedunculopontine nucleus, raphe serotonin) that tonically
    depolarize thalamic relay cells during wakefulness.

    Biological range: 0.001–0.005.
    """

    trn_baseline_drive: float = 0.001  # Reduced 0.003→0.001: 0.003 caused 6% sensory TRN epileptiform (T100408)
    # (TRN assoc went 3.61→2.96 Hz, MD 4.96 Hz).  TRN g_L=0.10, so needs ~3% of g_L
    # tonic drive to meaningfully depolarise toward threshold.
    # to provide adequate inhibition for physiological E/I ratios.
    """Per-step AMPA conductance added to TRN neurons each timestep.

    Biological range: 0.0005–0.002.
    """

    trn_relay_gaba_a_mean: float = 0.015
    """Mean weight for TRN→relay GABA_A synapses (sparse Gaussian initialization).

    Controls per-spike inhibitory impact of TRN on relay neurons.
    Higher values strengthen gating but risk excessive hyperpolarization
    and T-channel rebound bursting.  Per-instance overrides allow
    different thalamic nuclei to have different inhibitory strengths.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.spatial_filter_width <= 0:
            raise ConfigurationError(f"spatial_filter_width must be > 0, got {self.spatial_filter_width}")
        if self.trn_recurrent_delay_ms <= 0:
            raise ConfigurationError(f"trn_recurrent_delay_ms must be > 0, got {self.trn_recurrent_delay_ms}")
