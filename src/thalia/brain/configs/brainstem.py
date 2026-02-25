"""Configurations for brainstem regions: Cerebellum, Locus Coeruleus, Nucleus Basalis."""

from __future__ import annotations

from dataclasses import dataclass

from thalia.errors import ConfigurationError

from .neural_region import NeuralRegionConfig


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
    gain_learning_rate: float = 0.005  # Slow learning rate for gain adaptation (prevents instability)
    gain_tau_ms: float = 2000.0  # 2s averaging window (slow for motor stability)

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    # =========================================================================
    threshold_learning_rate: float = 0.005  # Slow learning rate for threshold adaptation (prevents instability)
    threshold_min: float = 0.5   # Moderate min (ensures some baseline excitability)
    threshold_max: float = 1.2  # Moderate max (Purkinje cells naturally active)

    # =========================================================================
    # ARCHITECTURE: GRANULE→PURKINJE→DCN CIRCUIT
    # =========================================================================
    # Uses granule→Purkinje→DCN circuit instead of direct
    # parallel fiber→Purkinje mapping. Provides:
    # - 4× sparse expansion in granule layer (pattern separation)
    # - Dendritic computation in Purkinje cells (complex/simple spikes)
    # - DCN integration (Purkinje sculpts tonic output)

    granule_connectivity: float = 0.03  # Fraction of granule cells active (3%)
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
    gap_junction_strength: float = 0.18  # Strong coupling for IO synchronization
    gap_junction_threshold: float = 0.20  # Low threshold for dense connectivity
    gap_junction_max_neighbors: int = 12  # Many neighbors for global sync

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

    tonic_drive_gain: float = 1.0
    """Overall gain on tonic excitatory drive to 5-HT neurons.

    Values > 1 increase baseline firing; values < 1 decrease it.
    Normal range: 0.5-2.0.
    """

    lhb_inhibition_gain: float = 3.0
    """Gain converting normalised LHb spiking rate to inhibitory conductance.

    High LHb activity (punishment) → strong DRN suppression → 5-HT pause.
    Biology: LHb→DRN operates via GABA interneurons and direct habenulo-raphe fibres.
    Typical range: 2.0-5.0.
    """

    drive_normalization: bool = True
    """Enable adaptive normalization of the tonic drive to prevent firing rate drift."""


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

    uncertainty_gain: float = 20.0
    """Gain for converting uncertainty to membrane current (mV per uncertainty unit).

    High uncertainty → depolarization → synchronized burst (10-15 Hz)
    Low uncertainty → hyperpolarization → pause or low tonic
    """

    gap_junction_strength: float = 0.05
    """Gap junction coupling strength (voltage-gated to prevent pacemaker quenching)."""

    gap_junction_radius: int = 50
    """Radius for gap junction connectivity (neurons within this range are coupled)."""

    uncertainty_normalization: bool = True
    """Enable adaptive uncertainty normalization to prevent saturation."""


@dataclass
class NucleusBasalisConfig(NeuralRegionConfig):
    """Configuration for NB (nucleus basalis) region.

    The NB is the brain's primary source of cortical acetylcholine, broadcasting
    attention and encoding/retrieval mode signals. NB neurons exhibit fast,
    brief bursts in response to prediction errors and attention shifts.

    Key features:
    - Acetylcholine neurons: Tonic (2-5 Hz) + fast bursts (10-20 Hz for 50-100ms)
    - Brief bursts (shorter than DA/NE) due to fast SK adaptation
    - Prediction error magnitude → encoding mode, attention enhancement
    - Selective projections → cortex and hippocampus (not striatum)
    """

    pe_gain: float = 25.0
    """Gain for converting prediction error to membrane current (mV per PE unit).

    High |PE| → depolarization → fast brief burst (10-20 Hz for 50-100ms)
    Low |PE| → baseline tonic firing
    """

    pe_normalization: bool = True
    """Enable adaptive prediction error normalization to prevent saturation."""
