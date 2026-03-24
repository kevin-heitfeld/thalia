"""Configurations for basal ganglia regions: GPe, GPi, LHb, RMTg, Striatum, SNr, STN, VTA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from thalia.brain.gap_junctions import GapJunctionConfig
from thalia.brain.regions.population_names import StriatumPopulation, VTAPopulation
from thalia.brain.synapses import NMReceptorType
from thalia.typing import NeuromodulatorChannel, PopulationPolarity

from .neural_region import (
    HomeostaticGainConfig,
    HomeostaticThresholdConfig,
    NMReceptorConfig,
    NeuralPopulationConfig,
    NeuralRegionConfig,
)


# ---------------------------------------------------------------------------
# Per-population config for BG output nuclei
# ---------------------------------------------------------------------------

@dataclass
class BGPopulationConfig(NeuralPopulationConfig):
    """Per-population biophysical parameters for a BG output nucleus.

    Extends :class:`NeuralPopulationConfig` with BG-specific fields for
    baseline drive scaling, NMDA routing, and Dale's law polarity.
    Replaces the former ``SubpopulationSpec`` frozen dataclass: all base
    biophysical parameters (tau_mem_ms, v_threshold, etc.) are now explicit
    per-population rather than shared at the region config level.
    """

    polarity: PopulationPolarity = PopulationPolarity.INHIBITORY
    """Dale's law polarity for this population."""

    baseline_multiplier: float = 1.0
    """Fraction of ``config.baseline_drive`` used for this population's tonic drive."""

    nmda_ratio: float = 0.05
    """Fraction of excitatory conductance routed to NMDA receptors."""


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

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
class BGOutputConfig(TonicPacemakerConfig):
    """Configuration for basal ganglia output nuclei (GPe, GPi, SNr).

    Extends :class:`TonicPacemakerConfig` with a dict of
    :class:`BGPopulationConfig` entries defining each population's biophysical
    parameters.  One registered class (``BasalGangliaOutputNucleus``) is
    instantiated three times with different configs — no per-nucleus subclass needed.
    """

    population_overrides: Dict[str, BGPopulationConfig] = field(default_factory=lambda: {})
    """Dict mapping population name → per-population biophysical config."""


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
    homeostatic_gain: HomeostaticGainConfig = field(default_factory=lambda: HomeostaticGainConfig(
        lr_per_ms=0.0004,
        tau_ms=30000.0,  # (should be hours, but 30s for testing)
    ))

    # =========================================================================
    # LEARNING RATES
    # =========================================================================
    learning_rate: float = 0.0001  # Striatum three-factor learning rate (dopamine-gated plasticity)

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (for MSN neurons)
    # =========================================================================
    homeostatic_threshold: HomeostaticThresholdConfig = field(default_factory=lambda: HomeostaticThresholdConfig(
        lr_per_ms=0.001,  # Reduced 0.02→0.001: effective tau must be ≥ 1000 ms (biological minimum for homeostasis)
        threshold_min=0.05,  # Lower floor to allow more aggressive adaptation for under-firing
        threshold_max=1.5,  # Allow some increase above default
    ))
    homeostatic_target_rates: dict[str, float] = field(default_factory=lambda: {
        StriatumPopulation.D1: 0.002,
        StriatumPopulation.D2: 0.002,
    })

    neuromodulator_receptors: list[NMReceptorConfig] = field(default_factory=lambda: [
        NMReceptorConfig(NMReceptorType.DA_D1, NeuromodulatorChannel.DA_MESOLIMBIC, "_da_mesolimbic_d1", (StriatumPopulation.D1,)),
        NMReceptorConfig(NMReceptorType.DA_D2, NeuromodulatorChannel.DA_MESOLIMBIC, "_da_mesolimbic_d2", (StriatumPopulation.D2,)),
        NMReceptorConfig(NMReceptorType.DA_D1, NeuromodulatorChannel.DA_NIGROSTRIATAL, "_da_nigrostriatal_d1", (StriatumPopulation.D1,)),
        NMReceptorConfig(NMReceptorType.DA_D2, NeuromodulatorChannel.DA_NIGROSTRIATAL, "_da_nigrostriatal_d2", (StriatumPopulation.D2,)),
        NMReceptorConfig(NMReceptorType.NE_ALPHA1, NeuromodulatorChannel.NE, "_ne_concentration_d1", (StriatumPopulation.D1,)),
        NMReceptorConfig(NMReceptorType.NE_ALPHA1, NeuromodulatorChannel.NE, "_ne_concentration_d2", (StriatumPopulation.D2,)),
        NMReceptorConfig(NMReceptorType.SHT_2A, NeuromodulatorChannel.SHT, "_sht_concentration_d1", (StriatumPopulation.D1,)),
        NMReceptorConfig(NMReceptorType.ACH_MUSCARINIC_M1, NeuromodulatorChannel.ACH, "_nb_ach_concentration_d1", (StriatumPopulation.D1,)),
        NMReceptorConfig(NMReceptorType.ACH_MUSCARINIC_M1, NeuromodulatorChannel.ACH, "_nb_ach_concentration_d2", (StriatumPopulation.D2,)),
    ])

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
    gap_junctions: GapJunctionConfig = field(default_factory=lambda: GapJunctionConfig(
        coupling_strength=0.005,  # Reduced 0.15→0.005: g_gap_total=1.5 shunted FSI to V_inf≈0.13 (threshold=1.0)
        connectivity_threshold=0.25,  # Neighborhood inference threshold
        max_neighbors=4,   # Reduced 10→4: g_gap_total=0.15×10=1.5 >> g_L=0.10; now 0.005×4=0.02 << g_L
    ))

    # =========================================================================
    # NEUROMODULATION: TONIC DOPAMINE
    # =========================================================================
    tonic_dopamine: float = 0.3
    min_tonic_dopamine: float = 0.1
    max_tonic_dopamine: float = 0.5

    # =========================================================================
    # TAN (TONICALLY ACTIVE NEURONS) PAUSE DYNAMICS
    # =========================================================================
    # Biology: TANs pause for ~300 ms on coincident cortical + thalamic bursts.
    # The pause is mediated by mAChR autoreceptors (M2/M4) and triggers the
    # corticostriatal plasticity window.
    tan_baseline_drive:  float = 0.005  # Swept via auto_calibrate: v_threshold=1.70 + baseline=0.005
                                        # gives 7.4 Hz in isolated region test (score=0.011).
    tan_pause_threshold: float = 0.050  # Mean g_ampa per TAN neuron that signals a burst
    tan_pause_strength:  float = 0.200  # Inhibitory g_gaba_a injected per TAN during pause
    # D2 autoreceptor-mediated pause: phasic DA burst activates D2Rs on TANs, coupling to
    # GIRK channels (slow K⁺ outward current). Approximated as a GABA_B-like conductance
    # proportional to the excess DA level above tan_da2_threshold.
    # References: Straub et al. 2014 (Nat Neurosci); Aosaki et al. 1994 (Science).
    tan_da2_threshold:      float = 0.30  # DA concentration above which D2Rs suppress TAN firing
    tan_da2_pause_strength: float = 0.15  # GABA_B-equivalent conductance per unit excess DA

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
class GPeConfig(BGOutputConfig):
    """Configuration for globus pallidus externa.

    Extends :class:`BGOutputConfig` with electrical gap junction parameters for the
    PROTOTYPIC population.  PROTOTYPIC neurons are coupled via Cx36 connexin gap
    junctions (Connelly et al. 2010, J Neurosci) that promote synchrony within the
    GPe-STN oscillatory loop.  ARKYPALLIDAL neurons do not show the same coupling
    and use the standard :class:`BGOutputConfig` path.

    Gap junction parameters
    -----------------------
    The coupling current at each timestep is:

        I_gap[i] = Σ_j  g_ij · (V_j[t-1] - V_i[t-1])

    implemented as a matrix multiply minus the weighted self-term, then scaled by
    ``gap_junction_scale`` before being added to g_exc.
    """

    # Biology: GPe Cx36 coupling is weaker than cortical PV interneurons; normalised
    # g_gap / g_L ≈ 0.02–0.04 matches published estimates (Connelly et al. 2010).
    # At g_L = 0.10, coupling_strength = 0.004 gives g_gap / g_L = 0.04.
    gap_junctions: GapJunctionConfig = field(default_factory=lambda: GapJunctionConfig(
        coupling_strength=0.004,
    ))
    """Gap junction config for PROTOTYPIC neurons."""

    gap_junction_connectivity: float = 0.40
    """Fraction of PROTOTYPIC neuron pairs connected by gap junctions."""

    gap_junction_scale: float = 0.30
    """Multiplicative scale applied to the gap junction current before adding to g_exc.

    Keeps coupling sub-threshold at rest; coupling is strongest during synchronised
    bursting when neighbouring neurons depolarise together.
    """


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

    # Reverted 0.036→0.025: higher noise caused 100% epileptiform in full brain
    # (synchronous threshold crossings). Using sparse recurrent GABA instead to
    # break pacemaker synchrony (proven pattern from VTA mesocortical fix).
    noise_std: float = 0.025
    """DA neuron membrane voltage noise standard deviation."""

    # Raised 0.013→0.06→0.10: at tonic 5 Hz, g_adapt_ss = 0.10×5×0.20 = 0.10 (1.25× g_L);
    # at 10 Hz, g_adapt_ss = 0.10×10×0.20 = 0.20 (2.5× g_L).  Previous 0.06 left SNc
    # 80% epileptiform — I_h rebound still overcame adaptation at moderate rates.
    adapt_increment: float = 0.10
    """Spike-triggered adaptation conductance increment (slow AHP)."""

    # Reduced 300→200: faster adaptation response prevents I_h rebound from
    # re-exciting DA neurons before adaptation engages.
    tau_adapt_ms: float = 200.0
    """Adaptation conductance time constant in ms."""

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
    noise_std: float = 0.025
    """DA neuron membrane noise standard deviation."""

    mesocortical_noise_std: float = 0.07
    """Mesocortical DA neuron noise (2.8× base).  Without D2 autoreceptors,
    high membrane noise is essential to desynchronise; combined with increased
    v_threshold CV (0.35) and adapt_increment (0.18) in the region."""

    # Raised 0.018→0.08→0.18: mesocortical neurons LACK D2 autoreceptors (Lammel
    # et al. 2008), so adaptation is their primary negative-feedback mechanism.
    # At 0.08 the population was still 100% epileptiform — adaptation+noise alone
    # couldn't break phase-lock.  At 0.18, g_adapt_ss at 7 Hz = 0.18×7×0.20 = 0.252
    # (3.15× g_L), strong enough to desynchronise ISIs without D2 feedback.
    mesocortical_adapt_increment: float = 0.18
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

    neuromodulator_receptors: list[NMReceptorConfig] = field(default_factory=lambda: [
        NMReceptorConfig(NMReceptorType.SHT_1A, NeuromodulatorChannel.SHT, "_sht_concentration", (VTAPopulation.DA_MESOLIMBIC,), amplitude_scale=1.33),
    ])
