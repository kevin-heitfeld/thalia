"""Configurations for basal ganglia regions: GPe, GPi, LHb, RMTg, Striatum, SNr, STN, VTA."""

from __future__ import annotations

from dataclasses import dataclass

from .neural_region import NeuralRegionConfig


@dataclass
class TonicPacemakerConfig(NeuralRegionConfig):
    """Shared configuration for tonically active nuclei driven by a baseline conductance input.

    Used by basal ganglia output nuclei (GPe, GPi, SNr), the habenulo-tegmental
    aversive pathway (LHb, RMTg), and the STN glutamatergic pacemaker.  All share
    the same conductance-LIF biophysical footprint:
    - Baseline per-step AMPA conductance for autonomous tonic firing
    - Short membrane time constant and refractory period
    - Optional I_h (HCN) conductance for voltage-sag pacemaking (STN; 0.0 = disabled)

    Nucleus-specific parameter values are set via factory functions (``get_default_*``).
    """

    baseline_drive: float = 0.012
    """Per-step AMPA conductance added each timestep for tonic pacemaking.

    This is NOT steady-state conductance. Actual g_E_ss = baseline_drive / (1 - exp(-dt/tau_E)).
    See the nucleus-specific factory docstring for tuning details.
    """

    tau_mem: float = 15.0
    """Membrane time constant in ms (10-20ms typical for BG output nuclei)."""

    v_threshold: float = 1.0
    """Firing threshold (SNr overrides to 1.25 to match its higher tonic drive)."""

    tau_ref: float = 2.0
    """Absolute refractory period in ms."""

    i_h_conductance: float = 0.0
    """Peak HCN (I_h) conductance for voltage-sag pacemaking (0.0 = disabled).

    Non-zero only for nuclei whose autonomous firing is driven by HCN rebound
    (e.g. STN: 0.0006).  All other nuclei leave this at the default 0.0.
    """


def get_default_gpe_config() -> TonicPacemakerConfig:
    """Default config for GPe (globus pallidus externa).

    The GPe is a key node in the basal ganglia indirect pathway, containing
    two distinct cell types:
    - Prototypic neurons: GABAergic, ~50 Hz tonic, project to STN and SNr
    - Arkypallidal neurons: GABAergic, project back to striatum (global suppression)

    baseline_drive=0.011: Per-step AMPA conductance for tonic pacemaking.
    IMPORTANT: This is NOT the steady-state conductance. Actual g_E_ss is:

        g_E_ss = baseline_drive / (1 - exp(-dt/tau_E))
               = 0.011 / (1 - exp(-1/5))  [dt=1ms, tau_E=5ms]
               ≈ 0.011 / 0.181 ≈ 0.061

    With g_L=0.10 and g_E_ss≈0.061, V_inf ≈ 1.13 (just above threshold), giving
    ~40-50 Hz intrinsic rate for prototypic neurons (target 30–80 Hz). Arkypallidal
    neurons receive 85.7% → g_E_ss≈0.052, V_inf≈1.03 → ~25-30 Hz (target 5–20 Hz).

    Previous value 0.060 was 5.5× too large (treated as steady-state rather than
    per-step), causing g_E_ss≈0.331, V_inf≈2.3 → 300-440 Hz hyperactivity.
    """
    return TonicPacemakerConfig(baseline_drive=0.011)


def get_default_gpi_config() -> TonicPacemakerConfig:
    """Default config for GPi (globus pallidus interna / entopeduncular nucleus).

    The GPi is the primary output nucleus of the basal ganglia for motor and
    cognitive loops, complementing the SNr (which gates saccades and VTA output).
    Two distinct cell types:
    - Principal neurons (~75%): GABAergic, ~60-80 Hz tonic, project to thalamus VA/VL
      (motor loop) and MD (cognitive/limbic loop).
    - Border cells (~25%): Pause on unexpected reward; proposed to encode a value signal
      analogous to SNr's value-coding subset.

    baseline_drive=0.013: Per-step AMPA conductance for tonic pacemaking.
    IMPORTANT: This is NOT the steady-state conductance. Actual g_E_ss is:

        g_E_ss = baseline_drive / (1 - exp(-dt/tau_E))
               = 0.013 / (1 - exp(-1/5))  [dt=1ms, tau_E=5ms]
               ≈ 0.013 / 0.181 ≈ 0.072

    With g_L=0.10 and g_E_ss≈0.072, V_inf ≈ 1.19 (above threshold), giving
    ~60-80 Hz intrinsic rate for principal neurons (target 60-80 Hz; slightly
    higher than GPe ~50 Hz). Border cells receive 0.4× → g_E_ss≈0.029,
    V_inf≈0.85 (sub-threshold at rest; fire only when driven by excitatory input).
    """
    return TonicPacemakerConfig(baseline_drive=0.013)


def get_default_lhb_config() -> TonicPacemakerConfig:
    """Default config for LHb (lateral habenula).

    The lateral habenula encodes negative reward prediction errors by
    exciting RMTg to pause VTA dopamine neurons on aversive outcomes.

    Key features:
    - Low tonic baseline (mostly silent until driven by SNr)
    - Excited by SNr output (high SNr → bad outcome signal)
    - Projects to RMTg to mediate dopamine pauses
    - Glutamatergic principal neurons

    baseline_drive=0.007: Per-step AMPA conductance to LHb principal neurons.
    IMPORTANT: This is NOT the steady-state conductance; actual g_E_ss is:

        g_E_ss = baseline_drive / (1 - exp(-dt/tau_E))
               = 0.007 / 0.181 ≈ 0.039

    Restored to 0.007 (run-04: 0.004 gave V_inf=0.648, far below threshold → 0.016 Hz).
    At 0.007, g_E_ss≈0.039 gives V_inf≈0.977 (just below threshold, g_L=0.08).
    LHb fires via noise + SNr excitation when SNr > ~40 Hz (aversive events).
    At SNr=45 Hz (tonic), combined V_inf≈1.004 → ~14 Hz tonic (target 5–20 Hz).
    At SNr < 40 Hz (reward suppresses SNr), LHb is quiet → correct biological gating.

    In run-03: 0.007 + fraction_of_drive=0.60 gave 24.8 Hz (too high).
    In run-04: 0.004 + fraction_of_drive=0.05 + DEPRESSING STP gave 0.016 Hz (too low).
    Fix: restore 0.007 baseline + keep fraction_of_drive=0.05 but remove STP depletion.

    Previous value 0.040 was 5.7× too large (treated as steady-state rather than
    per-step), causing g_E_ss≈0.221, V_inf≈2.2 → ~250 Hz hyperactivity.

    tau_mem=20.0: Longer than the default (15ms) — LHb principal cells are
    glutamatergic with slower membrane integration than fast GABAergic pacemakers.
    """
    return TonicPacemakerConfig(baseline_drive=0.007, tau_mem=20.0)


def get_default_rmtg_config() -> TonicPacemakerConfig:
    """Default config for RMTg (rostromedial tegmental nucleus).

    The RMTg (tail of VTA) mediates dopamine pauses: LHb excites RMTg,
    which provides fast GABAergic inhibition to VTA DA neurons.

    Key features:
    - Receives strong excitation from LHb
    - Projects GABAergic inhibition to VTA DA neurons
    - Drives dopamine pauses (negative RPE signal)
    - Moderate tonic baseline activity

    baseline_drive=0.004: Per-step AMPA conductance to RMTg GABA neurons.
    IMPORTANT: This is NOT the steady-state conductance; actual g_E_ss is:

        g_E_ss = baseline_drive / (1 - exp(-dt/tau_E))
               = 0.004 / (1 - exp(-1/4))  [tau_E=4ms for RMTg]
               ≈ 0.004 / 0.221 ≈ 0.018

    Reduced from 0.012 (run-03: 65.9 Hz → target 5–30 Hz). LHb excitatory input
    is the primary driver; baseline just sets a sub-threshold floor so LHb can
    push RMTg into 15-25 Hz territory during aversive signalling.

    tau_mem=15.0 (default): RMTg GABAergic neurons share the fast membrane
    dynamics of BG output nuclei, enabling precise DA pause timing.
    """
    return TonicPacemakerConfig(baseline_drive=0.004)


def get_default_snr_config() -> TonicPacemakerConfig:
    """Default config for SNr (substantia nigra pars reticulata).

    The SNr is the primary output nucleus of the basal ganglia, consisting of
    tonically-active GABAergic neurons that gate thalamic output and provide
    value feedback to VTA for dopamine-based reinforcement learning.

    Key features:
    - Tonic firing at 50-70 Hz baseline
    - Disinhibition mechanism: Striatum D1 reduces SNr → releases thalamus
    - Value encoding: Firing rate inversely proportional to state value
    - Closed-loop TD learning: Striatum → SNr → VTA → Striatum

    baseline_drive=0.015: Per-step AMPA conductance for SNr (vta_feedback) neurons.
    IMPORTANT: This is NOT the steady-state conductance; actual g_E_ss is:

        g_E_ss = baseline_drive / (1 - exp(-dt/tau_E))
               = 0.015 / 0.181 ≈ 0.083

    Reduced from 0.022 (run-03: 98.7 Hz → target 50–70 Hz). After STN fix (63.7→20 Hz)
    STN→SNr excitation drops ~3×. With baseline g_ss≈0.083 + STN g_ss≈0.015 at 20 Hz,
    total g_E≈0.098, V_inf≈1.48 (v_threshold=1.25) → ~60-70 Hz.

    With g_L=0.10, g_E_ss≈0.122, and v_threshold=1.25: V_inf_no_inh ≈ 1.51
    (above threshold). With biological GPe→SNr inhibition (g_I≈0.015 at corrected
    GPe weight), V_inf ≈ 1.37 → ~60-70 Hz tonic rate (target 50-70 Hz).

    v_threshold=1.25: Higher than GPe/GPi (1.0) to match SNr's stronger tonic drive.
    """
    return TonicPacemakerConfig(baseline_drive=0.015, v_threshold=1.25)


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
    gain_learning_rate: float = 0.0004
    gain_tau_ms: float = 30000.0  # (should be hours, but 30s for testing

    # =========================================================================
    # LEARNING RATES
    # =========================================================================
    learning_rate: float = 0.0001  # Striatum three-factor learning rate (dopamine-gated plasticity)

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

    # =========================================================================
    # TAN (TONICALLY ACTIVE NEURONS) PAUSE DYNAMICS
    # =========================================================================
    # Biology: TANs pause for ~300 ms on coincident cortical + thalamic bursts.
    # The pause is mediated by mAChR autoreceptors (M2/M4) and triggers the
    # corticostriatal plasticity window.
    tan_baseline_drive:  float = 0.003  # Tonic excitatory conductance for TAN intrinsic pacemaking.
    tan_pause_threshold: float = 0.050  # Mean g_ampa per TAN neuron that signals a burst
    tan_pause_strength:  float = 0.200  # Inhibitory g_gaba_a injected per TAN during pause
    # D2 autoreceptor-mediated pause: phasic DA burst activates D2Rs on TANs, coupling to
    # GIRK channels (slow K⁺ outward current). Approximated as a GABA_B-like conductance
    # proportional to the excess DA level above tan_da2_threshold.
    # References: Straub et al. 2014 (Nat Neurosci); Aosaki et al. 1994 (Science).
    tan_da2_threshold:      float = 0.30   # DA concentration above which D2Rs suppress TAN firing
    tan_da2_pause_strength: float = 0.15   # GABA_B-equivalent conductance per unit excess DA

    # =========================================================================
    # FSI (FAST-SPIKING INTERNEURONS) PARAMETERS
    # =========================================================================
    fsi_min_neurons: int = 30
    """Minimum absolute FSI count regardless of MSN population size.
    Biology: FSI are ~2% of MSNs; at small simulation scales (200-400 MSNs),
    2% = 4-8 neurons — too few for a reliable gap-junction small-world topology
    and prone to all-or-nothing collapse (one silent FSI silences entire feedforward
    inhibition). 30 neurons provides a minimum viable network. The 2% formula takes
    over at >1500 MSNs."""

    fsi_baseline_drive: float = 0.003
    """Tonic excitatory conductance seed for FSI (PV+ fast-spiking interneurons)."""


@dataclass
class DopaminePacemakerConfig(NeuralRegionConfig):
    """Shared configuration base for dopaminergic pacemaker regions (VTA, SNc).

    Captures the biophysical parameters common to all DA pacemaker populations:
    membrane dynamics, spike-frequency adaptation, and leak conductance.

    Both VTA and SNc DA neurons:
    - Share identical membrane time constants (~20 ms)
    - Share the same I_h pacemaker kinetics (g_h_max=0.03, tau_h=150 ms)
    - Use spike-frequency adaptation (slow AHP, tau ~300 ms) for tonic pacemaking
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
    """Spike-triggered adaptation conductance increment (slow AHP)."""

    tau_adapt: float = 300.0
    """Adaptation conductance time constant in ms (~300ms for DA pacemakers)."""


@dataclass
class SubstantiaNigraCompactaConfig(DopaminePacemakerConfig):
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

    baseline_drive: float = 0.008
    """Per-step AMPA conductance added to SNc DA neurons each timestep.

    IMPORTANT: This is NOT the steady-state conductance; actual g_E_ss is:

        g_E_ss = baseline_drive / (1 - exp(-dt/tau_E))
               = 0.008 / 0.181 ≈ 0.044

    Reduced from 0.010 (run-03: 10.0 Hz → target 2–8 Hz). With g_L=0.08 and
    g_E_ss≈0.044, V_inf≈1.06 (barely above threshold). Spike-frequency adaptation
    (adapt_increment=0.013, tau=300ms) yields ~5–7 Hz autonomous pacemaking.

    With g_L=0.08 and g_E_ss≈0.055, V_inf ≈ 1.22 (above threshold). Spike-frequency
    adaptation (adapt_increment=0.013, tau_adapt=300ms) then suppresses re-firing
    until g_adapt decays, yielding ~5-6 Hz tonic ISI via adaptation equilibrium.
    Previous value 0.050 was 5.5× too large, causing g_E_ss≈0.276, V_inf≈2.5
    → ~119 Hz hyperactivity even with adaptation.
    """


def get_default_stn_config() -> TonicPacemakerConfig:
    """Default config for STN (subthalamic nucleus).

    The STN is the sole glutamatergic nucleus within the basal ganglia, forming
    the hyperdirect pathway from cortex and the reciprocal GPe-STN oscillatory loop.
    STN neurons are autonomous pacemakers (~20 Hz) driven by HCN (I_h) channels.

    Key features:
    - Autonomous pacemaking via I_h (HCN) currents (~20 Hz)
    - Receives hyperdirect cortical input (fast, strong excitation)
    - Receives GPe inhibition (indirect pathway modulation)
    - Projects excitatory output to SNr and back to GPe

    baseline_drive=0.007: Tonic drive for autonomous pacemaker baseline.
    Reduced from 0.011 (run-03: 63.7 Hz) → 0.007 targeting ~20 Hz.
    g_E_ss = 0.007/0.181 ≈ 0.039 + cortical hyperdirect input (~0.017) → total ~0.056.
    V_inf ≈ 1.08 at target 20 Hz. (dt=1ms, tau_E=5ms)

    i_h_conductance=0.0006: Peak HCN conductance supporting voltage-sag pacemaking.

    tau_mem=18.0: Slightly longer than GPe/GPi/SNr (15ms); STN glutamatergic
    neurons have slower membrane integration (~15-25ms biological range).
    """
    return TonicPacemakerConfig(
        baseline_drive=0.007,
        tau_mem=18.0,
        i_h_conductance=0.0006,
    )


@dataclass
class VTAConfig(DopaminePacemakerConfig):
    """Configuration for VTA (ventral tegmental area) region.

    The VTA is the brain's primary dopamine source, computing reward prediction
    errors (RPE) through burst/pause dynamics. It forms the core of the
    reinforcement learning system by broadcasting teaching signals to all regions.

    Key features:
    - Dopamine neurons: Tonic (4-6 Hz mesolimbic / 7-9 Hz mesocortical) + phasic (burst/pause)
    - Full TD RPE: δ = r + γ·V(s’) − V(s)  with SNr value feedback
    - Anticipatory DA ramp (Howe et al. 2013) for temporal credit assignment
    - Closed-loop TD learning with SNr feedback via CircularDelayBuffer
    - Adaptive RPE normalization to prevent saturation
    - D2 somatodendritic autoreceptors on mesolimbic sub-population
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
    # ConductanceLIF DA neuron parameters
    #
    # Tonic pacemaking is achieved via baseline_drive in combination with
    # spike-frequency adaptation (slow AHP).  The mechanism:
    #   1. baseline_drive sets V_inf above threshold (V_inf > 1.0)
    #   2. Each spike increments g_adapt (adaptation conductance, tau=300ms)
    #   3. Adaptation hyperpolarises the cell, suppressing re-firing
    #   4. As g_adapt decays, V_inf rises back above threshold → next spike
    # This yields autonomous pacing at ~4-8 Hz without requiring I_h alone.
    #
    # NOTE: E_h in the DA neuron ConductanceLIFConfig was previously set to
    # -0.3 (normalised), which places E_h *below* E_L=0.0 and makes I_h
    # hyperpolarising at rest — the opposite of biology.  The correct normalised
    # E_h for HCN channels (-45 mV biological) is ≈ +0.9 (between rest and
    # threshold in the E_L=0, E_E=3 scale).  This is fixed in vta.py.
    # -------------------------------------------------------------------------
    mesolimbic_baseline_drive: float = 0.007
    """Per-step AMPA conductance added to mesolimbic DA neurons each timestep.

    IMPORTANT: This is NOT the steady-state conductance; actual g_E_ss is:

        g_E_ss = baseline_drive / (1 - exp(-dt/tau_E))
               = 0.007 / 0.181 ≈ 0.039

    Reduced from 0.010 (run-03: 12.0 Hz → target 2–8 Hz). After fixing RMTeg
    (65.9→15 Hz) the inhibitory input to VTA decreases, so baseline must come
    down to compensate. With g_L=0.08 and g_E_ss≈0.039, V_inf≈1.08.

    With g_L=0.08 and g_E_ss≈0.055, V_inf ≈ 1.22 (above threshold). Spike-frequency
    adaptation (adapt_increment=0.013, tau_adapt=300ms) then hyperpolarises after
    each spike until g_adapt decays, yielding ~5-6 Hz autonomous pacemaking.
    The adaptation threshold-crossing ISI solves to ~167ms (≈6 Hz) under the
    spike-frequency adaptation equilibrium.

    Previous value 0.055 was 5.5× too large (treated as steady-state rather than
    per-step), causing g_E_ss≈0.304, V_inf≈2.7 → ~147 Hz even with adaptation.
    """

    mesocortical_baseline_drive: float = 0.009
    """Per-step AMPA conductance added to mesocortical DA neurons each timestep.

    IMPORTANT: This is NOT the steady-state conductance; actual g_E_ss is:

        g_E_ss = 0.009 / 0.181 ≈ 0.050

    Reduced from 0.012 (run-03: 14.0 Hz → target 2–8 Hz). Slightly higher than
    mesolimbic (0.007) to achieve 7–9 Hz vs 5–7 Hz, compensating for faster
    adaptation in mesocortical sub-population.

    Slightly higher than mesolimbic (0.010) to achieve 7-9 Hz tonic vs 5-6 Hz.
    Mesocortical neurons have no D2 autoreceptors and higher g_adapt increment
    (0.018 vs 0.013), so a larger per-step drive compensates, yielding ~7-8 Hz
    via the spike-frequency adaptation equilibrium.

    Previous value 0.062 was 5.2× too large, causing ~128 Hz hyperactivity.
    """

    d2_autoreceptor_gain: float = 0.3
    """Somatodendritic D2 autoreceptor gain for mesolimbic DA neurons.

    Previous-step DA spike rate suppresses phasic RPE/ramp drive next step.
    Biological basis: D2Rs activate Gi/o → open GIRK channels (K+ outward) + inhibit
    adenylyl cyclase → reduce VGCCs → net hyperpolarisation and reduced excitability.
    Range 0.2–0.5: at peak tonic rate (~0.01 mean/step at 1ms dt), suppression <0.5%;
    at burst rate (~0.05 mean/step), suppression ~1.5% — gentle, self-limiting.
    Set to 0.0 to disable.
    """

    # -------------------------------------------------------------------------
    # Mesocortical sub-population parameters (Lammel et al. 2008)
    # Mesocortical DA neurons differ from mesolimbic in three key ways:
    #   1. Lack somatodendritic D2 autoreceptors → I_h-only pacing (~7-9 Hz)
    #   2. Faster spike-frequency adaptation (broader dynamic range)
    #   3. Respond more to stress/aversive stimuli, less to reward per se
    # -------------------------------------------------------------------------
    mesocortical_adapt_increment: float = 0.018
    """Adaptation increment for mesocortical DA neurons."""

    # -------------------------------------------------------------------------
    # Full TD-learning: V(s') — next-state value estimate
    # -------------------------------------------------------------------------
    value_lag_ms: float = 50.0
    """Lag (ms) between V(s) and V(s') for the TD next-state estimate.

    V(s') is approximated as the SNr-decoded value from ``value_lag_ms`` ago.
    Biologically, this corresponds to the dopamine ramp: DA neurons predict a
    reward ~50-100 ms in the future via an internal timing signal.
    Range: 20–200 ms.  Default 50 ms balances temporal credit assignment
    with stability under fast reward schedules.
    """

    # -------------------------------------------------------------------------
    # DA ramp — anticipatory reward signal
    # -------------------------------------------------------------------------
    da_ramp_enabled: bool = True
    """Enable slowly-building anticipatory DA ramp before expected rewards.

    The ramp models the "dopamine ramp" observed in primate recording:
    tonic DA rises monotonically during an approach
    to reward even before the reward is received.  This provides additional
    temporal credit assignment beyond the eligibility trace.

    Mechanics: each timestep without a reward increments a bounded ramp signal;
    each reward delivery resets it.  The ramp is added to the baseline DA drive.
    """

    da_ramp_tau_ms: float = 500.0
    """Time constant (ms) for DA ramp build-up.  The ramp asymptotes to
    ``da_ramp_gain / (dt_ms / da_ramp_tau_ms)`` under no-reward conditions.
    Biological range: 200–2000 ms depending on trial duration.
    """

    da_ramp_gain: float = 0.00001
    """Per-timestep increment added to the DA ramp signal each step without reward.

    Scales with dt_ms in practice; keep small to avoid overwhelming tonic drive.
    The mesocortical ramp is weighted at 50% of the mesolimbic value.

    The actual increment each step is ``da_ramp_gain * dt_ms * V(s')`` — the ramp
    is gated by the learned next-state value estimate from the SNr/striatum, so
    it only builds when the brain has learned to predict upcoming reward.  If no
    SNr connection is present (V_s_prime ≈ 0) the ramp stays flat regardless of
    this setting.  This makes the ramp emerge from corticostriatal learning rather
    than being a fixed-rate timer (Howe et al. 2013; Hamid et al. 2016).
    """
