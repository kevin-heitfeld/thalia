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

    tau_mem_ms: float = 15.0
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
    gain_tau_ms: float = 30000.0  # (should be hours, but 30s for testing)

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
    # CALIBRATION NOTE: g_gap_total = strength × max_neighbors must be << g_L (0.10)
    # to avoid shunting excitation. With strength=0.15 × 10 neighbors = 1.5 >> g_L,
    # the gap junction term dominated the V_inf denominator and drove FSI V_inf ≈ 0.13
    # (below threshold=1.0), completely silencing all FSI.
    # Fix: strength=0.005 × neighbors=4 → g_gap_total=0.02 << g_L=0.10 ✓
    # Biology: striatal FSI gap junctions have g_gap ≈ 0.3 nS, at g_L ≈ 10–15 nS
    # → g_gap/g_L ≈ 0.02-0.03. Our unitless ratio matches this.
    gap_junction_strength: float = 0.005  # Reduced 0.15→0.005: g_gap_total=1.5 shunted FSI to V_inf≈0.13 (threshold=1.0)
    gap_junction_threshold: float = 0.25  # Neighborhood inference threshold
    gap_junction_max_neighbors: int = 4   # Reduced 10→4: g_gap_total=0.15×10=1.5 >> g_L=0.10; now 0.005×4=0.02 << g_L

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

    tau_mem_ms: float = 20.0
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

    baseline_drive: float = 0.0
    """Per-step AMPA conductance added to DA neurons each timestep for tonic pacemaking."""


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
    mesolimbic_baseline_drive: float = 0.005
    """Per-step AMPA conductance added to mesolimbic DA neurons each timestep."""

    mesocortical_baseline_drive: float = 0.007
    """Per-step AMPA conductance added to mesocortical DA neurons each timestep."""

    d2_autoreceptor_gain: float = 0.3
    """Somatodendritic D2 autoreceptor gain for mesolimbic DA neurons."""

    # -------------------------------------------------------------------------
    # Mesocortical sub-population parameters (Lammel et al. 2008)
    # Mesocortical DA neurons differ from mesolimbic in three key ways:
    #   1. Lack somatodendritic D2 autoreceptors → I_h-only pacing (~7-9 Hz)
    #   2. Faster spike-frequency adaptation (broader dynamic range)
    #   3. Respond more to stress/aversive stimuli, less to reward per se
    # -------------------------------------------------------------------------
    noise_std: float = 0.015
    """DA neuron membrane noise standard deviation."""

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
