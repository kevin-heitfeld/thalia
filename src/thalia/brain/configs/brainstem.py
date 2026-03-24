"""Configurations for brainstem regions: Cerebellum, Locus Coeruleus, Nucleus Basalis."""

from __future__ import annotations

from dataclasses import dataclass, field

from thalia.brain.gap_junctions import GapJunctionConfig
from thalia.brain.regions.population_names import CerebellumPopulation
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
class CerebellumConfig(NeuralRegionConfig):
    """Configuration specific to cerebellar regions.

    The cerebellum implements ERROR-CORRECTIVE learning through:
    1. Parallel fiber → Purkinje cell connections (learned)
    2. Climbing fiber error signals from inferior olive
    3. LTD when climbing fiber active with parallel fiber

    Key biological features:
    - Error signal triggers immediate learning (not delayed like RL)
    - Can learn arbitrary input-output mappings quickly
    - Uses eligibility traces for temporal credit assignment
    """

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
    homeostatic_gain: HomeostaticGainConfig = field(default_factory=lambda: HomeostaticGainConfig(
        lr_per_ms=0.001,  # Reduced 0.005→0.001: effective tau must be ≥ 1000 ms (biological minimum for homeostasis)
        tau_ms=2000.0,  # 2s averaging window (slow for motor stability)
    ))

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    # =========================================================================
    homeostatic_threshold: HomeostaticThresholdConfig = field(default_factory=lambda: HomeostaticThresholdConfig(
        lr_per_ms=0.001,  # Reduced 0.005→0.001: effective tau must be ≥ 1000 ms (biological minimum for homeostasis)
        threshold_min=0.5,   # Moderate min (ensures some baseline excitability)
        threshold_max=1.2,  # Moderate max (Purkinje cells naturally active)
    ))
    homeostatic_target_rates: dict[str, float] = field(default_factory=lambda: {
        CerebellumPopulation.PURKINJE: 0.065,
    })

    neuromodulator_receptors: list[NMReceptorConfig] = field(default_factory=lambda: [
        NMReceptorConfig(NMReceptorType.NE_BETA, NeuromodulatorChannel.NE, "_ne_concentration", (CerebellumPopulation.PURKINJE,), amplitude_scale=1.5),
        NMReceptorConfig(NMReceptorType.DA_D1, NeuromodulatorChannel.DA_NIGROSTRIATAL, "_da_concentration", (CerebellumPopulation.GRANULE,), amplitude_scale=2.5),
        NMReceptorConfig(NMReceptorType.ACH_MUSCARINIC_M2, NeuromodulatorChannel.ACH, "_ach_concentration", (), amplitude_scale=2.0),
        NMReceptorConfig(NMReceptorType.SHT_2A, NeuromodulatorChannel.SHT, "_sht_concentration", (CerebellumPopulation.PURKINJE,)),
    ])

    # =========================================================================
    # ARCHITECTURE: GRANULE→PURKINJE→DCN CIRCUIT
    # =========================================================================
    # Uses granule→Purkinje→DCN circuit instead of direct
    # parallel fiber→Purkinje mapping. Provides:
    # - 4× sparse expansion in granule layer (pattern separation)
    # - Dendritic computation in Purkinje cells (complex/simple spikes)
    # - DCN integration (Purkinje sculpts tonic output)

    granule_connectivity: float = 0.03  # Fraction of granule cells active (3%)
    golgi_ratio: float = 0.05
    """Golgi-cell count as a fraction of the granule population.

    Biology: roughly 1 Golgi cell per 5–10 granule cells (human cerebellum).
    A value of 0.05 gives 500 Golgi cells for 10 000 granule cells.  Golgi
    cells receive feedforward excitation from both mossy fibers and granule
    cells, then provide broad GABA-A inhibition back to granule dendrites,
    enforcing the <5% population sparsity required for pattern separation.
    """
    purkinje_n_dendrites: int = 100  # Simplified dendritic compartments

    # =========================================================================
    # MARR-ALBUS-ITO LEARNING RATES
    # =========================================================================
    # Cerebellar plasticity at parallel fiber → Purkinje synapses is governed by
    # the conjunctive climbing fiber + parallel fiber rule.
    # LTD is 100× faster than LTP (strong asymmetry is characteristic of cerebellar learning).
    mai_ltd_rate: float = 0.01
    """LTD rate when climbing fiber fires coincident with parallel fiber activity."""

    mai_ltp_rate: float = 0.0001
    """Normalizing LTP rate when parallel fiber fires but climbing fiber is silent."""

    # =========================================================================
    # DCN TONIC BASELINE DRIVE
    # =========================================================================
    dcn_baseline_drive: float = 0.012
    """Per-step AMPA conductance added to DCN neurons each timestep for intrinsic pacemaking.

    Biology: DCN neurons are spontaneously active (40-60 Hz at rest) due to intrinsic
    persistent sodium and HCN currents. This is approximated as a constant excitatory
    conductance baseline that Purkinje inhibition then sculpts.
    """

    # =========================================================================
    # GAP JUNCTIONS (Inferior Olive Synchronization)
    # =========================================================================
    # Inferior olive (IO) neurons are electrically coupled via gap junctions,
    # creating synchronized complex spikes across multiple Purkinje cells.
    # This coordination is critical for motor learning and timing precision.
    #
    # Biology:
    # - IO neurons form one of the densest gap junction networks in the brain
    # - Synchronization time: <1ms (ultra-fast electrical coupling)
    # - Functional role: Coordinates learning across multiple cerebellar modules
    # - Complex spikes arrive synchronously at related Purkinje cells
    gap_junctions: GapJunctionConfig = field(default_factory=lambda: GapJunctionConfig(
        coupling_strength=0.18,  # Strong coupling for IO synchronization
        connectivity_threshold=0.20,  # Low threshold for dense connectivity
        max_neighbors=12,  # Many neighbors for global sync
    ))

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.purkinje_n_dendrites <= 0:
            raise ConfigurationError(f"purkinje_n_dendrites must be > 0, got {self.purkinje_n_dendrites}")
        if not (0.0 < self.granule_connectivity <= 1.0):
            raise ConfigurationError(f"granule_connectivity must be in (0, 1], got {self.granule_connectivity}")
        if self.mai_ltd_rate < 0:
            raise ConfigurationError(f"mai_ltd_rate must be >= 0, got {self.mai_ltd_rate}")
        if self.mai_ltp_rate < 0:
            raise ConfigurationError(f"mai_ltp_rate must be >= 0, got {self.mai_ltp_rate}")


@dataclass
class DorsalRapheNucleusConfig(NeuralRegionConfig):
    """Configuration for the Dorsal Raphe Nucleus (DRN) — serotonin system.

    The DRN is the brain's primary source of serotonin (5-HT), broadcasting
    patience, mood, and behavioural inhibition signals.  DRN neurons fire
    tonically at 2-4 Hz and are suppressed by punishment via LHb input.

    Key features:
    - 5-HT neurons: Tonic 2-4 Hz + 5-HT1A autoreceptor self-inhibition
    - LHb → DRN inhibition: punishment suppresses 5-HT (disinhibits aversive circuits)
    - DRN output channel ``'5ht'``: broadcast to striatum, PFC, hippocampus, BLA, thalamus
    """

    tonic_drive_gain: float = 0.78
    """Overall gain on tonic excitatory drive to 5-HT neurons.

    Lowered 0.92→0.78: DRN still at 4.24 Hz, target 0.5-3 Hz. Also LHb fix should
    provide stronger inhibition; combined reduction should bring DRN into range.
    biological target. Downstream serotonin receptor levels were low (0.0039)
    but this was a receptor kinetics issue not source firing. Keep DRN within
    2-3 Hz tonic range and address receptor accumulation separately.
    """

    lhb_inhibition_gain: float = 3.0
    """Gain converting normalised LHb spiking rate to inhibitory conductance.

    High LHb activity (punishment) → strong DRN suppression → 5-HT pause.
    Biology: LHb→DRN operates via GABA interneurons and direct habenulo-raphe fibres.
    Typical range: 2.0-5.0.
    """

    pfc_excitation_gain: float = 1.5
    """Gain converting normalised PFC spiking rate to excitatory DRN drive.

    PFC glutamatergic L5 neurons project to DRN serotonin neurons, providing
    top-down excitatory control.  When PFC is active (executive control engaged),
    it supports serotonin tone → patience and impulse control.
    Biology: Celada et al. 2001; Hajós et al. 2007.
    Typical range: 1.0-3.0.
    """


@dataclass
class LocusCoeruleusConfig(NeuralRegionConfig):
    """Configuration for LC (locus coeruleus) region.

    The LC is the brain's primary norepinephrine source, broadcasting arousal
    and uncertainty signals that modulate attention, gain, and exploration.
    LC neurons exhibit synchronized bursting due to dense gap junction coupling.

    Key features:
    - Norepinephrine neurons: Tonic (1-3 Hz) + phasic bursts (10-15 Hz for 500ms)
    - Gap junction coupling → synchronized population bursts
    - Uncertainty signal → exploratory behavior, network reset
    - Global projections → all brain regions
    """

    gap_junctions: GapJunctionConfig = field(default_factory=lambda: GapJunctionConfig(
        coupling_strength=0.05,
    ))
    """Gap junction coupling config (voltage-gated to prevent pacemaker quenching)."""

    gap_junction_radius: int = 50
    """Radius for gap junction connectivity (neurons within this range are coupled)."""
