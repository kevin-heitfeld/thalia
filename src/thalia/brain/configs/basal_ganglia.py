"""Configurations for basal ganglia regions: GPe, LHb, RMTg, Striatum, SNr, STN, VTA."""

from __future__ import annotations

from dataclasses import dataclass

from .neural_region import NeuralRegionConfig


@dataclass
class GlobusPallidusExternaConfig(NeuralRegionConfig):
    """Configuration for GPe (globus pallidus externa) region.

    The GPe is a key node in the basal ganglia indirect pathway, containing
    two distinct cell types:
    - Prototypic neurons: GABAergic, ~50 Hz tonic, project to STN and SNr
    - Arkypallidal neurons: GABAergic, project back to striatum (global suppression)

    Key features:
    - High tonic baseline (~50 Hz for prototypic neurons)
    - Receives inhibition from D2-MSNs (indirect pathway)
    - Provides inhibition to STN (closing the GPe-STN loop)
    - Arkypallidal neurons provide global feedback suppression to striatum
    """

    baseline_drive: float = 0.007
    """Tonic drive conductance for ~48 Hz prototypic baseline firing.

    Calibrated with ConductanceLIF (tau_mem=15ms, threshold=1.0, g_L=0.10) using
    the AMPA/NMDA split (nmda_ratio=0.05) as done in region.forward():
    g=0.007 → 47.5 Hz, g=0.0075 → 55 Hz. The NMDA component (tau_nmda=100ms)
    accumulates to ss_g_NMDA ≈ 0.05 × g × 100.5, contributing as much as AMPA.
    Previous calibration (g=0.011) was done with pure AMPA, underestimating
    the NMDA contribution and causing ~80 Hz overshoot in the connected brain.
    """

    tau_mem: float = 15.0
    """Membrane time constant (10-20ms typical for GPe neurons)."""

    v_threshold: float = 1.0
    """Firing threshold."""

    tau_ref: float = 2.0
    """Refractory period."""


@dataclass
class LateralHabenulaConfig(NeuralRegionConfig):
    """Configuration for LHb (lateral habenula) region.

    The lateral habenula computes negative reward prediction errors by
    encoding aversive outcomes. High SNr activity (indicating suppressed
    basal ganglia output = bad outcome) excites LHb, which then drives
    RMTg to pause VTA dopamine neurons.

    Key features:
    - Low tonic baseline (mostly silent until driven)
    - Excited by SNr output (high SNr → bad outcome signal)
    - Projects to RMTg to mediate dopamine pauses
    - Glutamatergic principal neurons
    """

    baseline_drive: float = 0.001
    """Low tonic drive conductance (LHb is mostly silent at baseline)."""

    tau_mem: float = 20.0
    """Membrane time constant."""

    v_threshold: float = 1.0
    """Firing threshold."""

    tau_ref: float = 2.0
    """Refractory period."""


@dataclass
class RostromedialTegmentumConfig(NeuralRegionConfig):
    """Configuration for RMTg (rostromedial tegmental nucleus) region.

    The RMTg (also called the tail of the VTA) is a GABAergic nucleus that
    receives excitation from the lateral habenula and inhibits VTA dopamine
    neurons, mediating the dopamine pause response to aversive prediction errors.

    Key features:
    - Receives strong excitation from LHb
    - Projects GABAergic inhibition to VTA DA neurons
    - Drives dopamine pauses (negative RPE signal)
    - Moderate tonic baseline activity
    """

    baseline_drive: float = 0.003
    """Tonic drive conductance for moderate baseline activity."""

    tau_mem: float = 15.0
    """Membrane time constant."""

    v_threshold: float = 1.0
    """Firing threshold."""

    tau_ref: float = 2.0
    """Refractory period."""


@dataclass
class StriatumConfig(NeuralRegionConfig):
    """Configuration specific to striatal regions.

    Behavioral parameters:
    - Learning rates
    - Exploration
    - Neuromodulation (tonic_dopamine, d1/d2 sensitivity)
    - Others as needed

    Key Features:
    =============
    1. THREE-FACTOR LEARNING: Δw = eligibility × dopamine
    2. D1/D2 OPPONENT PATHWAYS: Go/No-Go balance
    3. POPULATION CODING: Multiple neurons per action
    4. ADAPTIVE EXPLORATION: UCB + uncertainty-driven exploration
    5. HOMEOSTATIC PLASTICITY: Maintain stable activity and weight norms
    """

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
    # For striatum: Maintain COMBINED D1+D2 rate, not independent rates
    # Biology: D1 and D2 naturally balance via competition and FSI inhibition
    # BIOLOGY: Homeostatic plasticity operates on minutes-to-hours timescale
    # Increased tau from 1.5s to 30s (should be hours, but 30s for testing)
    # Reduced learning rate 10x to prevent rapid changes
    target_firing_rate: float = 0.08  # 8% target per pathway (16% combined for D1+D2)
    gain_learning_rate: float = 0.0004  # REDUCED 10x: slow homeostatic adaptation
    gain_tau_ms: float = 30000.0  # INCREASED 20x: 30s averaging window (biological: minutes to hours)

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (for MSN neurons)
    # =========================================================================
    threshold_learning_rate: float = 0.02  # Moderate threshold adaptation
    threshold_min: float = 0.05  # Lower floor to allow more aggressive adaptation for under-firing
    threshold_max: float = 1.5  # Allow some increase above default

    # =========================================================================
    # ELIGIBILITY TRACES: MULTISCALE
    # =========================================================================
    # Biological: Synaptic tags (eligibility traces) have multiple timescales:
    # - Fast traces (~500ms): Immediate pre-post spike coincidence tagging
    # - Slow traces (~60s): Consolidated tags from fast traces, enables credit
    #   assignment over multiple seconds (e.g., delayed rewards in RL tasks)
    # Combined eligibility: fast_trace + α × slow_trace enables both rapid and
    # delayed credit assignment.
    fast_eligibility_tau_ms: float = 500.0  # Fast trace decay (~500ms)
    slow_eligibility_tau_ms: float = 60000.0  # Slow trace decay (~60s)
    eligibility_consolidation_rate: float = 0.01  # Transfer rate from fast to slow (1% per timestep)
    slow_trace_weight: float = 0.3  # Weight of slow trace in combined eligibility

    # =========================================================================
    # EXPLORATION STRATEGIES
    # =========================================================================
    # Adaptive exploration based on recent performance
    # Increases exploration when performance drops
    performance_window: int = 10

    # =========================================================================
    # GAP JUNCTIONS
    # =========================================================================
    # Gap junction configuration for FSI networks
    gap_junction_strength: float = 0.15  # Biological: 0.05-0.3
    gap_junction_threshold: float = 0.25  # Neighborhood inference threshold
    gap_junction_max_neighbors: int = 10  # Biological: 4-12 neighbors

    # =========================================================================
    # LEARNING RATES
    # =========================================================================
    # Striatum three-factor learning rate (dopamine-gated plasticity)
    # NOTE: With eligibility trace fix (no double-scaling), this is the actual learning rate
    # Biological range: 0.0001-0.001 for stable opponent pathway dynamics
    learning_rate: float = 0.01  # 10× increase from 0.001 for faster weight changes

    # =========================================================================
    # NEUROMODULATION: TONIC DOPAMINE
    # =========================================================================
    tonic_dopamine: float = 0.3
    min_tonic_dopamine: float = 0.1
    max_tonic_dopamine: float = 0.5

    # =========================================================================
    # TEMPORAL DYNAMICS & DELAYS
    # =========================================================================
    # Biological timing for opponent pathways creates temporal competition:
    # - D1 "Go" pathway: Striatum → GPi/SNr → Thalamus (~15-20ms total)
    #   Direct inhibition of GPi/SNr → disinhibits thalamus → facilitates action
    # - D2 "No-Go" pathway: Striatum → GPe → STN → GPi/SNr (~23-28ms total)
    #   Indirect route via GPe and STN → inhibits thalamus → suppresses action
    # - Key insight: D1 pathway is ~8ms FASTER than D2 pathway
    #   Creates temporal competition window where D1 "vote" arrives first,
    #   D2 "veto" arrives later. Explains action selection timing and impulsivity.
    d1_to_output_delay_ms: float = 15.0  # D1 direct pathway delay
    d2_to_output_delay_ms: float = 25.0  # D2 indirect pathway delay (slower!)


@dataclass
class SubstantiaNigraConfig(NeuralRegionConfig):
    """Configuration for SNr (substantia nigra pars reticulata) region.

    The SNr is the primary output nucleus of the basal ganglia, consisting of
    tonically-active GABAergic neurons that gate thalamic output and provide
    value feedback to VTA for dopamine-based reinforcement learning.

    Key features:
    - Tonic firing at 50-70 Hz baseline
    - Disinhibition mechanism: Striatum D1 reduces SNr → releases thalamus
    - Value encoding: Firing rate inversely proportional to state value
    - Closed-loop TD learning: Striatum → SNr → VTA → Striatum
    """

    baseline_drive: float = 0.015
    """Tonic drive conductance for ~62 Hz baseline tonic firing.

    Biological SNr neurons fire tonically at 50-70 Hz due to intrinsic pacemaker
    currents (persistent Na+, T-type Ca2+). Calibrated with AMPA/NMDA split
    (nmda_ratio=0.01) matching region.forward():
    ConductanceLIF (tau_mem=15ms, v_threshold=1.25, g_L=0.10): g=0.015 → 62.5 Hz.
    The higher v_threshold (1.25 vs 1.0) requires higher baseline to achieve
    tonic firing (~60-80 Hz target range). In connected network, STN excitation
    will push SNr toward 70-80 Hz. D1 inhibition suppresses to ~30-50 Hz for Go.
    """

    tau_mem: float = 15.0
    """Membrane time constant for realistic integration (10-20ms typical for SNr)."""

    v_threshold: float = 1.25
    """Firing threshold."""

    tau_ref: float = 2.0
    """Refractory period for realistic max frequency (~500 Hz ceiling, actual 50-70 Hz)."""


@dataclass
class SubstantiaNigraCompactaConfig(NeuralRegionConfig):
    """Configuration for SNc (substantia nigra pars compacta) region.

    The SNc contains tonically-active dopaminergic neurons that project via the
    nigrostriatal pathway to the dorsal striatum (caudate + putamen), providing
    dopamine critical for motor control and habit learning.

    Distinct from VTA:
    - VTA projects mesolimbically (to ventral striatum / NAc) and mesocortically
      (to PFC) for reward and executive control.
    - SNc projects nigrostriatally (to dorsal striatum) for action sequencing and
      motor learning. Loss of SNc DA neurons causes Parkinson's disease.

    Key features:
    - Tonic pacemaking at 4-6 Hz via intrinsic conductances (Ca2+, HCN)
    - Spike-frequency adaptation (slow AHP) suppresses runaway activity
    - Sends `da_nigrostriatal` neuromodulator to dorsal striatum
    - Receives inhibition from striatal D1/D2 neurons (short-loop feedback)
    """

    rpe_normalization: bool = False
    """SNc does not compute RPE; motor dopamine is largely tonic.

    Unlike VTA, SNc does not receive direct reward feedback.
    This flag is kept for API symmetry with VTAConfig.
    """

    baseline_drive: float = 0.0016
    """Tonic excitatory baseline conductance for DA neurons.

    Same calibration as VTA: ~4-6 Hz tonic pacemaking.
    """

    tau_mem: float = 20.0
    """DA neuron membrane time constant in ms."""

    g_L: float = 0.08
    """DA neuron leak conductance (normalized units)."""

    tau_ref: float = 3.0
    """DA neuron refractory period in ms."""

    noise_std: float = 0.002
    """DA neuron membrane voltage noise standard deviation."""

    adapt_increment: float = 0.013
    """Spike-triggered adaptation conductance increment (slow AHP).

    Each spike adds this to the slow K+ adaptation conductance.
    Decays with tau_adapt. Shared calibration with VTA DA neurons.
    """

    tau_adapt: float = 300.0
    """Adaptation conductance time constant in ms (~300ms for DA pacemakers)."""


@dataclass
class SubthalamicNucleusConfig(NeuralRegionConfig):
    """Configuration for STN (subthalamic nucleus) region.

    The STN is a key glutamatergic nucleus in the basal ganglia that forms
    the hyperdirect pathway from cortex and the GPe-STN oscillatory loop.
    STN neurons are autonomous pacemakers (~20 Hz) driven by HCN channels.

    Key features:
    - Autonomous pacemaking via I_h (HCN) currents (~20 Hz)
    - Receives hyperdirect cortical input (fast, strong excitation)
    - Receives GPe inhibition (indirect pathway modulation)
    - Projects excitatory output to SNr and back to GPe
    """

    baseline_drive: float = 0.001
    """Tonic drive conductance for ~20 Hz autonomous pacemaker baseline."""

    i_h_conductance: float = 0.0006
    """HCN channel conductance supporting autonomous pacemaking."""

    tau_mem: float = 18.0
    """Membrane time constant (15-25ms typical for STN neurons)."""

    v_threshold: float = 1.0
    """Firing threshold."""

    tau_ref: float = 2.0
    """Refractory period."""


@dataclass
class VTAConfig(NeuralRegionConfig):
    """Configuration for VTA (ventral tegmental area) region.

    The VTA is the brain's primary dopamine source, computing reward prediction
    errors (RPE) through burst/pause dynamics. It forms the core of the
    reinforcement learning system by broadcasting teaching signals to all regions.

    Key features:
    - Dopamine neurons: Tonic (4-5 Hz) + phasic (burst/pause)
    - RPE computation: δ = r - V(s)
    - Closed-loop TD learning with SNr feedback
    - Adaptive normalization to prevent saturation
    - Strong baseline inhibition to maintain biological firing rates
    """

    rpe_gain: float = 0.0015
    """Gain for converting RPE to excitatory conductance.

    Positive RPE → extra g_exc on DA neurons → burst (15-20 Hz)
    Negative RPE handled by RMTg→DA GABA pathway (dopamine pause).
    """

    gamma: float = 0.99
    """TD learning discount factor for future reward weighting."""

    rpe_normalization: bool = True
    """Enable adaptive RPE normalization to prevent saturation."""

    # -------------------------------------------------------------------------
    # ConductanceLIF DA neuron parameters (calibrated for 4-6 Hz tonic firing)
    # -------------------------------------------------------------------------
    baseline_drive: float = 0.0016
    """Tonic excitatory baseline conductance for DA neurons.

    Above-threshold drive balanced by spike-frequency adaptation (adapt_increment)
    to produce biologically accurate 4-6 Hz tonic pacemaking.
    Calibrated value: g=0.0016, adapt_inc=0.013 → 5 Hz.
    """

    tau_mem: float = 20.0
    """DA neuron membrane time constant in ms."""

    g_L: float = 0.08
    """DA neuron leak conductance (normalized units)."""

    tau_ref: float = 3.0
    """DA neuron refractory period in ms."""

    noise_std: float = 0.002
    """DA neuron membrane voltage noise standard deviation."""

    adapt_increment: float = 0.013
    """Spike-triggered adaptation conductance increment (mimics I_KCA slow AHP).

    Each spike adds this much to the slow K+ adaptation conductance.
    Decays with tau_adapt. Tuned to reduce DA firing from ~18 Hz → 5 Hz.
    """

    tau_adapt: float = 300.0
    """Adaptation conductance time constant in ms (slow AHP, ~300ms for DA neurons)."""

    d2_autoreceptor_gain: float = 0.3
    """Somatodendritic D2 autoreceptor gain for mesolimbic DA neurons.

    Previous-step DA spike rate suppresses tonic baseline drive next step.
    Biological basis: D2Rs activate Gi/o → open GIRK channels (K+ outward) + inhibit
    adenylyl cyclase → reduce VGCCs → net hyperpolarisation and reduced excitability.
    Range 0.2–0.5: at peak tonic rate (~0.01 mean/step at 1ms dt), suppression <0.5%;
    at burst rate (~0.05 mean/step), suppression ~1.5% — gentle, self-limiting.
    Set to 0.0 to disable.
    """

    # -------------------------------------------------------------------------
    # Mesocortical sub-population parameters (Lammel et al. 2008)
    # Mesocortical DA neurons differ from mesolimbic in three key ways:
    #   1. Lack somatodendritic D2 autoreceptors → higher baseline firing (~7-9 Hz)
    #   2. Faster spike-frequency adaptation (broader dynamic range)
    #   3. Respond more to stress/aversive stimuli, less to reward per se
    # -------------------------------------------------------------------------
    mesocortical_baseline_drive: float = 0.0022
    """Tonic baseline conductance for mesocortical DA neurons.

    Higher than mesolimbic (0.0016) because absence of D2 autoreceptor feedback
    allows higher tonic firing (~7-9 Hz vs 4-6 Hz for mesolimbic).
    Calibrated: g=0.0022, adapt_inc=0.018 → ~8 Hz.
    """

    mesocortical_adapt_increment: float = 0.018
    """Adaptation increment for mesocortical DA neurons.

    Slightly larger than mesolimbic (0.013) reflecting broader dynamic range
    and faster adaptation to tonic bursting seen in mesocortical neurons.
    """
