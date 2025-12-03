"""
Cortex - Unsupervised Hebbian Learning with Biologically Realistic Mechanisms

The cerebral cortex learns through unsupervised Hebbian plasticity, discovering
statistical structure in sensory inputs without explicit teaching signals.

Core Mechanisms:
================
1. HEBBIAN STDP: Spike-timing dependent plasticity
   - Pre before post → LTP (strengthen connection)
   - Post before pre → LTD (weaken connection)
   - No external teaching signal required

2. BCM HOMEOSTASIS: Sliding threshold prevents runaway learning
   - High activity → raise threshold → favor LTD
   - Low activity → lower threshold → favor LTP

3. LATERAL INHIBITION: Competition between neurons
   - Winner-take-all dynamics via k-WTA
   - Creates sparse, selective representations

4. NEUROMODULATION: ACh gates plasticity magnitude (not direction)

Advanced Biological Mechanisms (Optional):
==========================================
INHIBITION & COMPETITION:
- SFA (Spike-Frequency Adaptation): Ca2+-activated K+ channels reduce
  excitability after spiking, preventing monopolization
- SOM+ Inhibition: Slow surround suppression from somatostatin interneurons
- Tonic GABA: Baseline inhibitory conductance for E/I balance
- kWTA: k-Winners-Take-All for controlled sparsity

TEMPORAL DYNAMICS:
- Theta-Gamma Coupling: Oscillatory modulation for temporal organization
- Axonal Delays: Per-synapse transmission delays (0.5-5ms)
- STP (Short-Term Plasticity): Vesicle depression/facilitation dynamics

COINCIDENCE DETECTION:
- NMDA Gating: Voltage-dependent Mg2+ block implements biological AND gate
- Dendritic Neurons: Multi-compartment neurons with local NMDA spikes

SYNAPTIC PLASTICITY:
- Triplet STDP: Frequency-dependent plasticity (Pfister & Gerstner, 2006)
- Synaptic Scaling: Slow weight normalization prevents explosion/collapse
- Recurrent Learning: Lateral connections for pattern completion

HOMEOSTASIS:
- Homeostatic Conductance: Dynamic tonic excitation adjustment
- Intrinsic Plasticity: Per-neuron excitability adjustment

Biological Basis:
=================
- Visual cortex V1 learns oriented edge detectors
- Auditory cortex learns frequency tuning
- Association cortex learns correlational structure

When to Use:
============
- Feature extraction from sensory data
- Unsupervised clustering/categorization
- When you DON'T have explicit target labels

Example:
========
    from thalia.regions.cortex import Cortex, CortexConfig

    # Basic cortex with recommended settings
    config = CortexConfig(
        n_input=784,
        n_output=100,
        diagonal_bias=0.2,  # Symmetry breaking
        sfa_enabled=True,   # Prevent monopolization
    )
    cortex = Cortex(config)

    # Process input
    output = cortex.forward(input_spikes)
    cortex.learn(input_spikes, output)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch

from thalia.regions.base import (
    BrainRegion,
    RegionConfig,
    LearningRule,
    NeuromodulatorSystem,
)
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.core.dendritic import DendriticNeuron, DendriticNeuronConfig, DendriticBranchConfig
from thalia.learning import (
    hebbian_update,
    update_bcm_threshold,
    TripletSTDP,
    TripletSTDPConfig,
    synaptic_scaling,
    update_homeostatic_conductance,
    update_homeostatic_excitability,
)


@dataclass
class CortexConfig(RegionConfig):
    """Configuration for cortical regions with biologically realistic mechanisms.

    The defaults are chosen to work reasonably for most unsupervised learning tasks.
    For best results, enable diagonal_bias (0.2) and sfa_enabled (True).

    Key Parameters:
        hebbian_lr: Learning rate for Hebbian/STDP plasticity (0.001-0.1)
        diagonal_bias: Developmental pre-wiring strength (0.0=off, 0.2=recommended)
        sfa_enabled: Spike-frequency adaptation prevents winner monopolization
        kwta_k: k-Winners-Take-All sparsity (0=disabled, or ~10-15% of n_output)

    See module docstring for descriptions of all biological mechanisms.
    """

    # ==========================================================================
    # CORE LEARNING PARAMETERS
    # ==========================================================================

    # Hebbian learning rate: controls speed of weight updates
    # Higher = faster learning but potentially unstable
    # Range: 0.001 (slow/stable) to 0.1 (fast/aggressive)
    hebbian_lr: float = 0.01

    # STDP time constants: window for spike-timing dependent plasticity
    # ~20ms matches biological STDP windows in cortex
    stdp_tau_plus_ms: float = 20.0  # LTP window (pre→post)
    stdp_tau_minus_ms: float = 20.0  # LTD window (post→pre)

    # BCM homeostasis: sliding threshold prevents runaway learning
    # tau_ms: how quickly threshold adapts (slower = more stable)
    # target_rate_hz: desired average firing rate
    bcm_tau_ms: float = 1000.0  # Slow adaptation (1 second)
    bcm_target_rate_hz: float = 10.0  # Target 10 Hz firing rate

    # ==========================================================================
    # COMPETITION & INHIBITION
    # ==========================================================================

    # Lateral inhibition: winner-take-all competition
    lateral_inhibition: bool = True
    inhibition_strength: float = 0.3  # How strongly winners suppress losers

    # Heterosynaptic competition: LTD on non-active synapses
    # Higher values = more competition, sparser representations
    heterosynaptic_ratio: float = 0.3

    # Soft weight bounds: gradual saturation vs hard clipping
    soft_bounds: bool = True

    # k-Winners-Take-All: only top k neurons can spike per timestep
    # Set to ~10-15% of n_output for sparse coding, or 0 to disable
    kwta_k: int = 0  # 0 = disabled, >0 = keep only top k winners

    # ==========================================================================
    # SYMMETRY BREAKING (Critical for learning diverse features)
    # ==========================================================================

    # Diagonal Bias: developmental pre-wiring for symmetry breaking
    # Without this, all neurons start identical and converge to same features
    # Represents biological topographic maps from molecular gradients
    # Values: 0.0 = disabled, 0.2-0.3 = recommended
    diagonal_bias: float = 0.2  # Enabled by default for better learning

    # ==========================================================================
    # SPIKE-FREQUENCY ADAPTATION (Prevents monopolization)
    # ==========================================================================

    # SFA: Ca2+-activated K+ channels reduce excitability after spiking
    # Critical for preventing "winner-take-all-forever" dynamics
    sfa_enabled: bool = True  # Enabled by default
    sfa_tau_ms: float = 200.0  # Decay time constant (100-500ms biological)
    sfa_increment: float = 0.15  # Adaptation increase per spike
    sfa_strength: float = 1.5  # Excitability reduction factor

    # ==========================================================================
    # SLOW INHIBITORY MECHANISMS (Optional, for complex dynamics)
    # ==========================================================================

    # SOM+ Inhibition: Slow surround suppression from somatostatin interneurons
    # Provides history-dependent competition over hundreds of milliseconds
    som_enabled: bool = False
    som_strength: float = 0.5  # Inhibition strength (0.3-1.0)
    som_tau_ms: float = 200.0  # Decay time constant (100-300ms)
    som_sigma: float = 4.0  # Spatial spread in neuron indices

    # Tonic GABA: Extrasynaptic baseline inhibitory conductance
    # Maintains E/I balance even without phasic inhibition
    tonic_gaba_enabled: bool = False
    tonic_gaba_conductance: float = 0.1  # Baseline g_I (0.05-0.2)

    # ==========================================================================
    # TEMPORAL MECHANISMS (Optional, for sequence/timing tasks)
    # ==========================================================================

    # Theta-Gamma Coupling: Oscillatory modulation for temporal organization
    # Useful for sequence learning and temporal binding
    theta_gamma_enabled: bool = False
    theta_freq_hz: float = 6.0  # Theta frequency (4-8 Hz typical)
    gamma_freq_hz: float = 40.0  # Gamma frequency (30-100 Hz typical)
    theta_strength: float = 0.3  # Modulation amplitude (0.1-0.5)

    # Axonal Delays: Per-synapse transmission delays
    # Creates temporal structure for spike-timing computations
    axonal_delays_enabled: bool = False
    axonal_delay_min_ms: float = 0.5  # Minimum delay
    axonal_delay_max_ms: float = 5.0  # Maximum delay

    # Short-Term Plasticity (STP): Vesicle dynamics
    # Depression: repeated spikes → weaker transmission (adaptation)
    # Facilitation: repeated spikes → stronger transmission (priming)
    stp_enabled: bool = False
    stp_mode: str = "depression"  # "depression", "facilitation", or "both"
    stp_depression_rate: float = 0.2  # Vesicle depletion fraction per spike
    stp_recovery_tau_ms: float = 200.0  # Vesicle recovery time constant
    stp_facilitation_rate: float = 0.1  # Ca2+ accumulation per spike
    stp_facilitation_tau_ms: float = 50.0  # Ca2+ decay time constant

    # ==========================================================================
    # COINCIDENCE DETECTION (Optional, for associative learning)
    # ==========================================================================

    # NMDA Gating: Voltage-dependent Mg2+ block
    # Implements biological AND gate: requires both pre and post activity
    nmda_enabled: bool = False
    nmda_fraction: float = 0.3  # Fraction through NMDA vs AMPA (0.2-0.5)
    nmda_mg_concentration: float = 1.0  # mM, affects voltage sensitivity
    nmda_v_half: float = -0.2  # Voltage at 50% unblock (normalized)

    # Dendritic Neurons: Multi-compartment with local NMDA spikes
    # Enables complex input integration and coincidence detection
    dendritic_enabled: bool = False
    dendritic_n_branches: int = 4  # Branches per neuron (2-8 typical)
    dendritic_nmda_threshold: float = 0.3  # Local spike threshold
    dendritic_branch_attenuation: float = 0.5  # Branch → soma attenuation

    # ==========================================================================
    # RECURRENT CONNECTIONS (Optional, for attractor/sequence dynamics)
    # ==========================================================================

    # Recurrent: Lateral excitatory connections between output neurons
    # Enables pattern completion and sequence learning
    recurrent_enabled: bool = False
    recurrent_strength: float = 0.3  # Initial weight strength (0.1-0.5)
    recurrent_learning: bool = True  # Whether to learn recurrent weights

    # ==========================================================================
    # ADVANCED PLASTICITY RULES (Optional, for enhanced learning)
    # ==========================================================================

    # Triplet STDP: Frequency-dependent plasticity (Pfister & Gerstner, 2006)
    # More biologically accurate than pair-based STDP
    # High-frequency bursts → enhanced LTP
    triplet_stdp_enabled: bool = False
    triplet_a2_plus: float = 0.005  # Pair-based potentiation amplitude
    triplet_a2_minus: float = 0.007  # Pair-based depression amplitude
    triplet_a3_plus: float = 0.006  # Triplet LTP (post-post-pre)
    triplet_a3_minus: float = 0.002  # Triplet LTD (pre-pre-post)

    # Synaptic Scaling: Slow weight normalization
    # Prevents weight explosion/collapse across all synapses
    synaptic_scaling_enabled: bool = False
    synaptic_scaling_target: float = 0.3  # Target L2 norm fraction (0.2-0.5)
    synaptic_scaling_tau: float = 100.0  # Adaptation time constant

    # ==========================================================================
    # HOMEOSTATIC MECHANISMS (Optional, for stable learning)
    # ==========================================================================

    # Homeostatic Conductance: Dynamic tonic excitation adjustment
    # Underactive neurons get more tonic excitation
    homeostatic_conductance_enabled: bool = False
    homeostatic_g_tau: float = 10.0  # Rate averaging time constant
    homeostatic_g_strength: float = 0.001  # Conductance change per Hz error
    homeostatic_g_bounds: tuple[float, float] = (0.0, 0.5)  # Conductance bounds

    # Intrinsic Plasticity: Per-neuron excitability adjustment
    # Adjusts additive current to maintain target firing rate
    # Complementary to homeostatic conductance (multiplicative vs additive)
    intrinsic_plasticity_enabled: bool = False
    intrinsic_plasticity_tau: float = 5.0  # Rate averaging time constant
    intrinsic_plasticity_strength: float = 0.01  # Excitability change per Hz
    intrinsic_plasticity_bounds: tuple[float, float] = (-0.5, 0.5)  # Bounds

    # ==========================================================================
    # LEARNING RULE SELECTION
    # ==========================================================================
    # The cortex supports multiple learning paradigms:
    # - HEBBIAN (default): Spike-timing dependent plasticity (STDP)
    # - PREDICTIVE: Predictive coding - minimize prediction error (rate-coded)
    # - PREDICTIVE_STDP: True spiking with prediction error modulation
    #
    # PREDICTIVE_STDP combines:
    # 1. Real spikes for forward pass (LIF neurons)
    # 2. STDP eligibility traces from spike timing
    # 3. Prediction error as the "third factor" that gates learning
    #
    # Weight update: Δw ∝ STDP(pre, post) × |prediction_error|
    # This is biologically plausible: only surprising inputs cause learning.
    #
    # Use LearningRule enum to select the learning mode.
    learning_rule: LearningRule = LearningRule.HEBBIAN

    # ==========================================================================
    # PREDICTIVE CODING PARAMETERS 
    # (used when learning_rule=PREDICTIVE or PREDICTIVE_STDP)
    # ==========================================================================
    # Based on Rao & Ballard (1999) and Friston's Free Energy Principle
    # Each layer predicts its input; learning minimizes prediction error
    # 
    # Key insight: Instead of Hebbian "fire together, wire together",
    # predictive coding learns representations that reconstruct inputs
    # while being constrained by higher-level predictions.
    
    # Learning rate for prediction weights
    predictive_lr: float = 0.05
    
    # Number of inference steps to settle dynamics per input
    predictive_inference_steps: int = 20
    
    # Time constant for representation dynamics (how fast repr changes)
    predictive_tau: float = 10.0
    
    # Integration timestep for inference dynamics
    predictive_dt: float = 0.1
    
    # Precision (inverse variance) of predictions - higher = trust predictions more
    predictive_precision: float = 1.0
    
    # ==========================================================================
    # PREDICTIVE STDP PARAMETERS (used when learning_rule=PREDICTIVE_STDP)
    # ==========================================================================
    # Combines spiking dynamics with prediction error modulation
    
    # How strongly prediction error modulates STDP (0 = pure STDP, 1 = full gating)
    pred_error_modulation: float = 0.8
    
    # Time constant for error signal (smooths bursty prediction errors)
    pred_error_tau_ms: float = 50.0
    
    # Minimum modulation (prevents complete learning shutdown)
    pred_error_min_mod: float = 0.1


class AcetylcholineSystem(NeuromodulatorSystem):
    """Acetylcholine neuromodulatory system for attention gating."""

    def __init__(self, tau_ms: float = 100.0, device: str = "cpu"):
        super().__init__(tau_ms, device)
        self.baseline = 0.5
        self.level = self.baseline

    def compute(
        self,
        novelty: float = 0.0,
        attention: float = 0.0,
        uncertainty: float = 0.0,
        **kwargs: Any,
    ) -> float:
        boost = 0.3 * novelty + 0.4 * attention + 0.3 * uncertainty
        self.level = min(1.0, self.baseline + boost)
        return self.level

    def get_learning_modulation(self) -> float:
        return 0.5 + self.level * 1.5


class Cortex(BrainRegion):
    """Cerebral cortex with unsupervised Hebbian learning."""

    def __init__(self, config: RegionConfig):
        if not isinstance(config, CortexConfig):
            config = CortexConfig(
                n_input=config.n_input,
                n_output=config.n_output,
                neuron_type=config.neuron_type,
                learning_rate=config.learning_rate,
                w_max=config.w_max,
                w_min=config.w_min,
                target_firing_rate_hz=config.target_firing_rate_hz,
                dt_ms=config.dt_ms,
                device=config.device,
            )

        self.cortex_config: CortexConfig = config  # type: ignore
        super().__init__(config)

        self.bcm_threshold = torch.ones(1, config.n_output, device=self.device) * 0.5
        self.ach = AcetylcholineSystem(device=config.device)
        self.input_trace = torch.zeros(config.n_input, device=self.device)
        self.output_trace = torch.zeros(config.n_output, device=self.device)
        self.recent_spikes = torch.zeros(config.n_output, device=self.device)

        # Homeostatic boost: tracks which neurons haven't been active
        # Inactive neurons get boosted to prevent "dead neurons"
        self.activity_count = torch.zeros(config.n_output, device=self.device)

        # Spike-Frequency Adaptation state (Ca2+-activated K+ channels)
        # This accumulates with each spike and decays over time
        # High adaptation = reduced excitability (hyperpolarizing current)
        self.sfa_adaptation = torch.zeros(config.n_output, device=self.device)

        # ======================================================================
        # NEW BIOLOGICAL MECHANISM STATES
        # ======================================================================

        # SOM+ inhibition state: slow surround suppression
        if self.cortex_config.som_enabled:
            self.som_activation = torch.zeros(config.n_output, device=self.device)
            # Precompute Gaussian inhibition kernel based on neuron distance
            positions = torch.arange(config.n_output, device=self.device, dtype=torch.float32)
            distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
            self.som_kernel = torch.exp(-distance_matrix**2 / (2 * self.cortex_config.som_sigma**2))
            self.som_kernel = self.som_kernel * (1 - torch.eye(config.n_output, device=self.device))
        else:
            self.som_activation = None
            self.som_kernel = None

        # Theta-gamma oscillation state
        if self.cortex_config.theta_gamma_enabled:
            self.oscillation_phase = 0.0  # Current phase in radians
        else:
            self.oscillation_phase = None

        # STP state: vesicle availability and facilitation
        if self.cortex_config.stp_enabled:
            # Vesicle availability (1.0 = full, depletes with use)
            self.stp_vesicles = torch.ones(config.n_input, device=self.device)
            # Facilitation factor (accumulates with pre activity)
            self.stp_facilitation = torch.zeros(config.n_input, device=self.device)
        else:
            self.stp_vesicles = None
            self.stp_facilitation = None

        # Axonal delays: per-synapse delay buffer
        if self.cortex_config.axonal_delays_enabled:
            delay_range = self.cortex_config.axonal_delay_max_ms - self.cortex_config.axonal_delay_min_ms
            # Random delays per input, scaled to timesteps
            self.axonal_delays = (
                torch.rand(config.n_input, device=self.device) * delay_range
                + self.cortex_config.axonal_delay_min_ms
            ) / config.dt_ms
            self.axonal_delays = self.axonal_delays.int()
            max_delay = int(self.cortex_config.axonal_delay_max_ms / config.dt_ms) + 1
            # Circular buffer for delayed spikes
            self.delay_buffer = torch.zeros(max_delay, config.n_input, device=self.device)
            self.delay_buffer_idx = 0
        else:
            self.axonal_delays = None
            self.delay_buffer = None

        # Recurrent connections
        if self.cortex_config.recurrent_enabled:
            self.recurrent_weights = self._initialize_recurrent_weights()
            # Previous output for recurrent input
            self.prev_output = torch.zeros(config.n_output, device=self.device)
        else:
            self.recurrent_weights = None
            self.prev_output = None

        # ======================================================================
        # PHASE 2 MECHANISM STATES
        # ======================================================================

        # Triplet STDP state
        if self.cortex_config.triplet_stdp_enabled:
            triplet_config = TripletSTDPConfig(
                a2_plus=self.cortex_config.triplet_a2_plus,
                a2_minus=self.cortex_config.triplet_a2_minus,
                a3_plus=self.cortex_config.triplet_a3_plus,
                a3_minus=self.cortex_config.triplet_a3_minus,
                w_min=config.w_min,
                w_max=config.w_max,
            )
            self.triplet_stdp = TripletSTDP(
                n_pre=config.n_input,
                n_post=config.n_output,
                config=triplet_config,
            ).to(self.device)
        else:
            self.triplet_stdp = None

        # Tonic GABA state: persistent baseline inhibition
        if self.cortex_config.tonic_gaba_enabled:
            self.tonic_gaba = torch.ones(config.n_output, device=self.device) * self.cortex_config.tonic_gaba_conductance
        else:
            self.tonic_gaba = None

        # Homeostatic conductance state
        if self.cortex_config.homeostatic_conductance_enabled:
            self.homeostatic_g_tonic = torch.zeros(config.n_output, device=self.device)
            self.homeostatic_avg_rate = torch.zeros(config.n_output, device=self.device)
        else:
            self.homeostatic_g_tonic = None
            self.homeostatic_avg_rate = None

        # Dendritic neuron state
        if self.cortex_config.dendritic_enabled:
            # Use dendritic neurons instead of simple ConductanceLIF
            inputs_per_branch = max(1, config.n_input // self.cortex_config.dendritic_n_branches)
            branch_config = DendriticBranchConfig(
                nmda_threshold=self.cortex_config.dendritic_nmda_threshold,
                subthreshold_attenuation=self.cortex_config.dendritic_branch_attenuation,
            )
            dendritic_config = DendriticNeuronConfig(
                n_branches=self.cortex_config.dendritic_n_branches,
                inputs_per_branch=inputs_per_branch,
                branch_config=branch_config,
            )
            self.dendritic_neurons = DendriticNeuron(
                n_neurons=config.n_output,
                config=dendritic_config,
            ).to(self.device)
        else:
            self.dendritic_neurons = None

        # Intrinsic Plasticity state: per-neuron excitability adjustment
        if self.cortex_config.intrinsic_plasticity_enabled:
            self.intrinsic_excitability = torch.zeros(config.n_output, device=self.device)
            self.intrinsic_avg_rate = torch.zeros(config.n_output, device=self.device)
        else:
            self.intrinsic_excitability = None
            self.intrinsic_avg_rate = None

        # ======================================================================
        # PREDICTIVE CODING STATE
        # ======================================================================
        if self.cortex_config.learning_rule == LearningRule.PREDICTIVE:
            # Prediction weights: representation -> input reconstruction
            # These are the generative model weights
            self.prediction_weights = torch.randn(
                config.n_output, config.n_input, device=self.device
            ) * 0.1 / (config.n_output ** 0.5)
            
            # Bottom-up weights for error -> representation mapping
            # Initialized as transpose of prediction weights (approx inverse)
            self.error_weights = torch.randn(
                config.n_input, config.n_output, device=self.device
            ) * 0.1 / (config.n_input ** 0.5)
            
            # Representation state (rate-coded, not spikes)
            self.pc_representation = torch.zeros(config.n_output, device=self.device)
            
            # Prediction error (input - prediction)
            self.pc_error = torch.zeros(config.n_input, device=self.device)
        elif self.cortex_config.learning_rule == LearningRule.PREDICTIVE_STDP:
            # PREDICTIVE_STDP: Spiking + prediction error modulation
            # Need prediction weights to generate predictions from output spike rates
            self.prediction_weights = torch.randn(
                config.n_output, config.n_input, device=self.device
            ) * 0.1 / (config.n_output ** 0.5)
            
            # Smoothed prediction error for modulating STDP
            self.pred_error_smooth = torch.zeros(1, device=self.device)
            
            # Running estimate of output firing rate (for prediction)
            self.output_rate_estimate = torch.zeros(config.n_output, device=self.device)
            
            # STDP eligibility traces (per synapse)
            self.stdp_eligibility = torch.zeros(config.n_output, config.n_input, device=self.device)
            
            # Not used in PREDICTIVE_STDP mode
            self.error_weights = None
            self.pc_representation = None
            self.pc_error = None
        else:
            self.prediction_weights = None
            self.error_weights = None
            self.pc_representation = None
            self.pc_error = None

    def _get_learning_rule(self) -> LearningRule:
        return self.cortex_config.learning_rule

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize weights with optional diagonal bias for symmetry breaking.

        Diagonal bias represents developmental pre-wiring (topographic maps from
        molecular gradients, spontaneous waves before eye opening). Without it,
        all neurons receive identical input and learn to respond to the same
        features, causing representation collapse.
        """
        n_out = self.config.n_output
        n_in = self.config.n_input

        # Base weights: lognormal distribution (many weak, few strong - Song et al. 2005)
        initial_fraction = 0.1
        target_median = initial_fraction * self.config.w_max
        lognormal_mu = torch.tensor(target_median).log().item()
        lognormal_sigma = 0.3  # tight spread

        weights = torch.exp(
            torch.randn(n_out, n_in) * lognormal_sigma + lognormal_mu
        )
        max_initial = min(0.3, initial_fraction * 3) * self.config.w_max
        weights = weights.clamp(0.01, max_initial)

        # Apply diagonal bias for topographic initialization
        if self.cortex_config.diagonal_bias > 0:
            ratio = max(1, n_in // n_out)  # inputs per output neuron
            diagonal_bias_tensor = torch.zeros_like(weights)

            for i in range(n_out):
                # Primary inputs for neuron i
                for j in range(ratio):
                    inp_idx = (ratio * i + j) % n_in
                    diagonal_bias_tensor[i, inp_idx] = self.cortex_config.diagonal_bias * self.config.w_max

                # Smaller gradient to neighbors (smooth, not cliff)
                left_idx = (ratio * i - 1) % n_in
                right_idx = (ratio * (i + 1)) % n_in
                diagonal_bias_tensor[i, left_idx] = 0.3 * self.cortex_config.diagonal_bias * self.config.w_max
                diagonal_bias_tensor[i, right_idx] = 0.3 * self.cortex_config.diagonal_bias * self.config.w_max

            weights = weights + diagonal_bias_tensor
            weights = weights.clamp(0.01, self.config.w_max * 0.9)  # leave headroom

        return weights.clamp(self.config.w_min, self.config.w_max).to(self.device)

    def _initialize_recurrent_weights(self) -> torch.Tensor:
        """Initialize recurrent weights with optional topographic structure.

        Recurrent connections support attractor dynamics and pattern completion.
        Key features:
        - No self-connections (diagonal = 0)
        - Distance-dependent connectivity (nearby neurons more connected)
        - Excitatory only (inhibition handled separately)
        """
        n_out = self.config.n_output

        # Base weights: lognormal like feedforward (sparse connectivity)
        initial_fraction = 0.05  # Recurrent typically sparser than feedforward
        target_median = initial_fraction * self.config.w_max
        lognormal_mu = torch.tensor(target_median).log().item()
        lognormal_sigma = 0.4

        weights = torch.exp(
            torch.randn(n_out, n_out) * lognormal_sigma + lognormal_mu
        )
        weights = weights * self.cortex_config.recurrent_strength  # Scale by config
        weights = weights.clamp(0.0, self.config.w_max * 0.5)

        # Optional: Distance-dependent connectivity (Mexican hat / local excitation)
        # Nearby neurons more strongly connected for local pattern completion
        if self.cortex_config.recurrent_strength > 0:
            distance_bias = torch.zeros(n_out, n_out)
            for i in range(n_out):
                for j in range(n_out):
                    dist = min(abs(i - j), n_out - abs(i - j))  # Circular distance
                    # Gaussian falloff: nearby neurons get stronger connections
                    distance_bias[i, j] = torch.exp(torch.tensor(-dist**2 / (n_out / 4)**2))
            weights = weights * (0.5 + 0.5 * distance_bias)

        # Zero diagonal (no self-connections)
        weights = weights * (1 - torch.eye(n_out))

        return weights.clamp(0.0, self.config.w_max).to(self.device)

    def _create_neurons(self) -> ConductanceLIF:
        neuron_config = ConductanceLIFConfig(
            v_threshold=1.0, v_reset=0.0, E_L=0.0, E_E=3.0, E_I=-0.5,
            tau_E=5.0, tau_I=10.0,
            dt=self.config.dt_ms,  # Use region's dt for consistency
            tau_ref=2.0,  # 2ms refractory period
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
        neurons.to(self.device)
        return neurons

    def forward(
        self,
        input_spikes: torch.Tensor,
        recurrent_input: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        if self.neurons.membrane is None:
            self.neurons.reset_state(input_spikes.shape[0])

        decay = 1.0 - self.config.dt_ms / self.cortex_config.stdp_tau_plus_ms

        # ======================================================================
        # AXONAL DELAYS: Buffer input spikes and retrieve delayed versions
        # ======================================================================
        if self.cortex_config.axonal_delays_enabled and self.delay_buffer is not None:
            # Store current input in buffer
            self.delay_buffer[self.delay_buffer_idx] = input_spikes.squeeze()

            # Retrieve delayed spikes for each input
            delayed_spikes = torch.zeros_like(input_spikes.squeeze())
            for inp_idx in range(self.config.n_input):
                delay = self.axonal_delays[inp_idx].item()
                buffer_idx = (self.delay_buffer_idx - int(delay)) % self.delay_buffer.shape[0]
                delayed_spikes[inp_idx] = self.delay_buffer[buffer_idx, inp_idx]

            self.delay_buffer_idx = (self.delay_buffer_idx + 1) % self.delay_buffer.shape[0]
            effective_input = delayed_spikes.unsqueeze(0)
        else:
            effective_input = input_spikes

        self.input_trace = self.input_trace * decay + effective_input.squeeze()

        # ======================================================================
        # SHORT-TERM PLASTICITY (STP): Modify synaptic transmission
        # ======================================================================
        if self.cortex_config.stp_enabled and self.stp_vesicles is not None:
            # Decay facilitation
            fac_decay = torch.exp(torch.tensor(-self.config.dt_ms / self.cortex_config.stp_facilitation_tau_ms))
            self.stp_facilitation = self.stp_facilitation * fac_decay

            # Recover vesicles
            rec_decay = torch.exp(torch.tensor(-self.config.dt_ms / self.cortex_config.stp_recovery_tau_ms))
            self.stp_vesicles = self.stp_vesicles + (1.0 - self.stp_vesicles) * (1.0 - rec_decay)

            # Compute STP factor
            if self.cortex_config.stp_mode in ("depression", "both"):
                depression_factor = self.stp_vesicles
            else:
                depression_factor = torch.ones_like(self.stp_vesicles)

            if self.cortex_config.stp_mode in ("facilitation", "both"):
                facilitation_factor = 1.0 + self.stp_facilitation
            else:
                facilitation_factor = torch.ones_like(self.stp_facilitation)

            stp_factor = depression_factor * facilitation_factor

            # Apply STP to input (modulate effective synaptic strength)
            modulated_input = effective_input * stp_factor.unsqueeze(0)

            # Update STP state based on presynaptic spikes
            pre_spikes = effective_input.squeeze()
            self.stp_vesicles = self.stp_vesicles - pre_spikes * self.stp_vesicles * self.cortex_config.stp_depression_rate
            self.stp_facilitation = self.stp_facilitation + pre_spikes * self.cortex_config.stp_facilitation_rate
        else:
            modulated_input = effective_input

        # ======================================================================
        # COMPUTE EXCITATORY CONDUCTANCE
        # ======================================================================
        g_exc = torch.matmul(modulated_input, self.weights.T)

        # Add recurrent input if provided externally
        if recurrent_input is not None:
            g_exc = g_exc + recurrent_input

        # Add internal recurrent connections
        if self.cortex_config.recurrent_enabled and self.recurrent_weights is not None:
            recurrent_exc = torch.matmul(self.prev_output.unsqueeze(0), self.recurrent_weights.T)
            g_exc = g_exc + recurrent_exc

        # ======================================================================
        # NMDA GATING: Voltage-dependent Mg2+ block
        # ======================================================================
        if self.cortex_config.nmda_enabled:
            # Get current membrane potential (normalized)
            if self.neurons.membrane is not None:
                v_post = self.neurons.membrane.squeeze()
            else:
                v_post = torch.zeros(self.config.n_output, device=self.device)

            # Compute NMDA unblock factor (sigmoid based on voltage)
            # At rest (v=0), mostly blocked; depolarized (v>0.5), mostly unblocked
            nmda_unblock = torch.sigmoid(
                (v_post - self.cortex_config.nmda_v_half) * 5.0 / self.cortex_config.nmda_mg_concentration
            )

            # Split excitation into AMPA (fast) and NMDA (voltage-gated) components
            ampa_fraction = 1.0 - self.cortex_config.nmda_fraction
            nmda_contribution = g_exc * self.cortex_config.nmda_fraction * nmda_unblock.unsqueeze(0)
            ampa_contribution = g_exc * ampa_fraction
            g_exc = ampa_contribution + nmda_contribution

        # ======================================================================
        # THETA-GAMMA COUPLING: Oscillatory modulation
        # ======================================================================
        if self.cortex_config.theta_gamma_enabled and self.oscillation_phase is not None:
            # Update oscillation phase
            theta_period_ms = 1000.0 / self.cortex_config.theta_freq_hz
            phase_increment = (2 * 3.14159 * self.config.dt_ms) / theta_period_ms
            self.oscillation_phase = (self.oscillation_phase + phase_increment) % (2 * 3.14159)

            # Theta modulation: sinusoidal excitability modulation
            import math
            theta_mod = 1.0 + self.cortex_config.theta_strength * math.sin(self.oscillation_phase)
            g_exc = g_exc * theta_mod

        # ======================================================================
        # SPIKE-FREQUENCY ADAPTATION (SFA)
        # ======================================================================
        if self.cortex_config.sfa_enabled:
            sfa_decay = torch.exp(torch.tensor(-self.config.dt_ms / self.cortex_config.sfa_tau_ms))
            self.sfa_adaptation = self.sfa_adaptation * sfa_decay
            sfa_inhibition = self.sfa_adaptation * self.cortex_config.sfa_strength
            g_exc = g_exc - sfa_inhibition.unsqueeze(0)

        # ======================================================================
        # LATERAL INHIBITION (fast, blanket)
        # ======================================================================
        g_inh = self.recent_spikes.unsqueeze(0) * self.cortex_config.inhibition_strength if self.cortex_config.lateral_inhibition else None

        # ======================================================================
        # TONIC GABA INHIBITION: Baseline inhibitory conductance
        # ======================================================================
        if self.cortex_config.tonic_gaba_enabled and self.tonic_gaba is not None:
            if g_inh is not None:
                g_inh = g_inh + self.tonic_gaba.unsqueeze(0)
            else:
                g_inh = self.tonic_gaba.unsqueeze(0)

        # ======================================================================
        # HOMEOSTATIC CONDUCTANCE: Add tonic excitation for underactive neurons
        # ======================================================================
        if self.cortex_config.homeostatic_conductance_enabled and self.homeostatic_g_tonic is not None:
            g_exc = g_exc + self.homeostatic_g_tonic.unsqueeze(0)

        # ======================================================================
        # INTRINSIC PLASTICITY: Per-neuron excitability adjustment
        # ======================================================================
        if self.cortex_config.intrinsic_plasticity_enabled and self.intrinsic_excitability is not None:
            # Excitability acts as additive current (can be positive or negative)
            g_exc = g_exc + self.intrinsic_excitability.unsqueeze(0)

        # ======================================================================
        # SOM+ INHIBITION: Slow surround suppression
        # ======================================================================
        if self.cortex_config.som_enabled and self.som_activation is not None:
            # Decay SOM+ activation
            som_decay = torch.exp(torch.tensor(-self.config.dt_ms / self.cortex_config.som_tau_ms))
            self.som_activation = self.som_activation * som_decay

            # Compute SOM+ inhibition (distance-weighted from recently active neurons)
            som_inhibition = torch.matmul(self.som_activation, self.som_kernel) * self.cortex_config.som_strength

            # Add to inhibitory conductance
            if g_inh is not None:
                g_inh = g_inh + som_inhibition.unsqueeze(0)
            else:
                g_inh = som_inhibition.unsqueeze(0)

        # ======================================================================
        # NEURON DYNAMICS
        # ======================================================================
        output_spikes, _ = self.neurons(g_exc, g_inh)

        # ======================================================================
        # k-WINNERS-TAKE-ALL
        # ======================================================================
        if self.cortex_config.kwta_k > 0:
            k = self.cortex_config.kwta_k
            selection_score = g_exc.squeeze().clone()

            # Mild boosting for inactive neurons
            avg_activity = self.activity_count.mean() + 1e-8
            relative_inactivity = avg_activity / (self.activity_count + 1e-8)
            boost = torch.clamp(relative_inactivity - 1.0, 0, 1.0) * 0.15
            selection_score = selection_score * (1.0 + boost)

            _, top_k_indices = torch.topk(selection_score, k)
            mask = torch.zeros_like(output_spikes)
            mask[:, top_k_indices] = 1.0
            output_spikes = output_spikes * mask

        # ======================================================================
        # UPDATE STATE VARIABLES
        # ======================================================================

        # Update SFA
        if self.cortex_config.sfa_enabled:
            self.sfa_adaptation = self.sfa_adaptation + output_spikes.squeeze() * self.cortex_config.sfa_increment

        # Update SOM+ activation
        if self.cortex_config.som_enabled and self.som_activation is not None:
            self.som_activation = self.som_activation + output_spikes.squeeze()

        # Update recurrent state
        if self.cortex_config.recurrent_enabled and self.prev_output is not None:
            self.prev_output = output_spikes.squeeze().detach()

        # Update activity count for homeostasis
        self.activity_count = self.activity_count * 0.999 + output_spikes.squeeze()

        self.output_trace = self.output_trace * decay + output_spikes.squeeze()
        self.recent_spikes = self.recent_spikes * 0.9 + output_spikes.squeeze()
        self.state.spikes = output_spikes
        self.state.t += 1

        return output_spikes

    def predictive_forward(
        self,
        sensory_input: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run predictive coding inference.
        
        Instead of spike-based feedforward processing, this runs iterative
        inference to find representations that minimize prediction error.
        
        Args:
            sensory_input: Rate-coded input (0-1 values, not spikes)
            n_steps: Number of inference steps (uses config default if None)
            
        Returns:
            representation: Rate-coded output representation
        """
        if self.cortex_config.learning_rule != LearningRule.PREDICTIVE:
            raise RuntimeError("Predictive coding is not enabled. Set learning_rule=LearningRule.PREDICTIVE")
        
        if sensory_input.dim() == 1:
            sensory_input = sensory_input.unsqueeze(0)
        
        n_steps = n_steps or self.cortex_config.predictive_inference_steps
        dt = self.cortex_config.predictive_dt
        tau = self.cortex_config.predictive_tau
        
        # Reset representation if needed
        if self.pc_representation is None:
            self.pc_representation = torch.zeros(self.config.n_output, device=self.device)
        
        # Run inference dynamics
        for _ in range(n_steps):
            # Generate prediction of input from current representation
            prediction = torch.sigmoid(self.pc_representation @ self.prediction_weights)
            
            # Compute prediction error
            self.pc_error = sensory_input.squeeze() - prediction
            
            # Bottom-up drive: error → representation
            bu_drive = self.pc_error @ self.error_weights
            
            # Update representation (leaky integration)
            d_repr = (-self.pc_representation + bu_drive) / tau
            self.pc_representation = self.pc_representation + dt * d_repr
            
            # Clamp to valid range
            self.pc_representation = torch.clamp(self.pc_representation, 0.0, 1.0)
        
        # Store for learning
        self.state.t += 1
        
        return self.pc_representation.unsqueeze(0)
    
    def get_prediction(self) -> torch.Tensor:
        """Get the current prediction of the input.
        
        Call this after predictive_forward() to get the reconstruction.
        """
        if self.cortex_config.learning_rule != LearningRule.PREDICTIVE:
            raise RuntimeError("Predictive coding is not enabled")
        if self.pc_representation is None:
            raise RuntimeError("No representation computed. Call predictive_forward() first")
        
        return torch.sigmoid(self.pc_representation @ self.prediction_weights)
    
    def get_prediction_error(self) -> float:
        """Get the current prediction error magnitude."""
        if self.pc_error is None:
            return float('inf')
        return self.pc_error.pow(2).mean().item()

    def _predictive_learn(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        """Update prediction weights to minimize prediction error.
        
        Uses the current representation (from predictive_forward) to update
        weights so that predictions better match the input.
        
        Args:
            sensory_input: The input that was presented
            
        Returns:
            Dict with learning metrics
        """
        if self.pc_representation is None or self.pc_error is None:
            return {"pred_error": float('inf'), "dw_pred": 0.0, "dw_error": 0.0}
        
        if sensory_input.dim() == 1:
            sensory_input = sensory_input.unsqueeze(0)
        
        lr = self.cortex_config.predictive_lr
        
        # Recompute prediction and error with current representation
        prediction = torch.sigmoid(self.pc_representation @ self.prediction_weights)
        error = sensory_input.squeeze() - prediction
        
        # Update prediction weights to reduce error
        # Gradient descent on MSE: d(error^2)/dW = -2 * repr^T @ error
        # repr: (n_output,), error: (n_input,)
        # dW: (n_output, n_input)
        dW_pred = torch.outer(self.pc_representation, error)
        self.prediction_weights = self.prediction_weights + lr * dW_pred
        self.prediction_weights = self.prediction_weights.clamp(-2.0, 2.0)
        
        # Update error weights (inverse mapping)
        # These map errors to representation updates
        dW_error = torch.outer(error, self.pc_representation) * 0.1  # Slower update
        self.error_weights = self.error_weights + lr * dW_error
        self.error_weights = self.error_weights.clamp(-2.0, 2.0)
        
        # Update BCM threshold based on representation activity
        # This helps maintain diverse representations
        # Convert representation to Hz-like scale (0-1 → 0-100 Hz)
        activity_hz = self.pc_representation * 100.0
        self.bcm_threshold = update_bcm_threshold(
            self.bcm_threshold,
            avg_activity_hz=activity_hz.unsqueeze(0),
            target_rate_hz=self.cortex_config.bcm_target_rate_hz,
            tau=self.cortex_config.bcm_tau_ms,
            min_threshold=0.1,
            max_threshold=0.9,
        )
        
        return {
            "pred_error": error.pow(2).mean().item(),
            "dw_pred": dW_pred.abs().mean().item(),
            "dw_error": dW_error.abs().mean().item(),
        }

    def _predictive_stdp_learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
    ) -> Dict[str, Any]:
        """STDP learning modulated by prediction error.
        
        This combines:
        1. Real spiking dynamics (already happened in forward())
        2. STDP eligibility from spike timing correlations
        3. Prediction error as the "third factor" that gates learning
        
        Key insight: Only surprising inputs should cause learning.
        If the network correctly predicts its input, no weight change.
        If prediction error is high, STDP is allowed to update weights.
        
        Weight update: Δw = lr × STDP(pre, post) × |prediction_error|
        
        Args:
            input_spikes: Pre-synaptic spikes from this timestep
            output_spikes: Post-synaptic spikes from this timestep
            
        Returns:
            Dict with learning metrics
        """
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)
        if output_spikes.dim() == 1:
            output_spikes = output_spikes.unsqueeze(0)
        
        dt = self.config.dt_ms
        cfg = self.cortex_config
        
        # ======================================================================
        # Step 1: Update firing rate estimate (exponential moving average)
        # ======================================================================
        rate_decay = 1.0 - dt / 100.0  # ~100ms time constant
        spike_rate = output_spikes.squeeze()  # Instantaneous (0 or 1)
        self.output_rate_estimate = self.output_rate_estimate * rate_decay + spike_rate * (1 - rate_decay)
        
        # ======================================================================
        # Step 2: Compute prediction error
        # ======================================================================
        # Generate prediction of input from current output firing rate
        # prediction_weights: (n_output, n_input) - maps output rate to predicted input
        predicted_input = torch.sigmoid(self.output_rate_estimate @ self.prediction_weights)
        
        # Compare to actual input (convert spikes to rate-like signal)
        input_rate = input_spikes.squeeze()
        
        # Prediction error: MSE between predicted and actual input
        pred_error = (predicted_input - input_rate).pow(2).mean()
        
        # Smooth the prediction error (prevents noisy modulation)
        error_decay = 1.0 - dt / cfg.pred_error_tau_ms
        self.pred_error_smooth = self.pred_error_smooth * error_decay + pred_error * (1 - error_decay)
        
        # ======================================================================
        # Step 3: Compute STDP eligibility update
        # ======================================================================
        # Update pre and post traces (exponential decay + spike)
        trace_decay = 1.0 - dt / cfg.stdp_tau_plus_ms
        self.input_trace = self.input_trace * trace_decay + input_spikes.squeeze()
        self.output_trace = self.output_trace * trace_decay + output_spikes.squeeze()
        
        # STDP rule: 
        # - LTP: post spike with pre trace → strengthen (outer product)
        # - LTD: pre spike with post trace → weaken (outer product)
        # dw shape: (n_output, n_input)
        ltp = torch.outer(output_spikes.squeeze(), self.input_trace)  # post spike × pre trace
        ltd = torch.outer(self.output_trace, input_spikes.squeeze())  # post trace × pre spike
        
        # COMPETITIVE LEARNING: Anti-Hebbian for non-winners
        # Neurons that DIDN'T spike when inputs were present should get weaker
        # This forces specialization - only winners get to keep their inputs
        non_spiking = 1.0 - output_spikes.squeeze()  # (n_output,)
        anti_hebbian = torch.outer(non_spiking, input_spikes.squeeze())  # loser × active inputs
        
        # Soft bounds: reduce learning as weights approach limits
        # This prevents saturation and allows stable learning
        w_normalized = (self.weights - self.config.w_min) / (self.config.w_max - self.config.w_min)
        ltp_factor = 1.0 - w_normalized  # More room to grow → more LTP
        ltd_factor = w_normalized         # Higher weight → more LTD
        
        # Apply soft bounds to STDP
        soft_ltp = ltp * ltp_factor
        soft_ltd = ltd * ltd_factor
        soft_anti_hebbian = anti_hebbian * w_normalized  # Only subtract if weights are high
        
        # Net STDP (before modulation) with competitive component
        # LTP for winners, LTD for normal STDP, anti-Hebbian for losers
        stdp_dw = cfg.hebbian_lr * (soft_ltp - cfg.heterosynaptic_ratio * soft_ltd - 0.1 * soft_anti_hebbian)
        
        # Accumulate into eligibility trace (decays over time)
        eligibility_decay = 1.0 - dt / 200.0  # ~200ms eligibility window
        self.stdp_eligibility = self.stdp_eligibility * eligibility_decay + stdp_dw
        
        # ======================================================================
        # Step 4: Modulate eligibility by prediction error
        # ======================================================================
        # High error → more learning; low error → less learning
        # This is the "third factor" that gates STDP
        error_modulation = cfg.pred_error_min_mod + (1.0 - cfg.pred_error_min_mod) * torch.sigmoid(
            cfg.pred_error_modulation * 10.0 * (self.pred_error_smooth - 0.1)
        )
        
        # Apply only CURRENT timestep's STDP modulated by error (not accumulated!)
        # The eligibility trace is for tracking timing relationships, but we apply
        # the instantaneous stdp_dw modulated by error, not the accumulated trace
        dw = stdp_dw * error_modulation
        self.weights = (self.weights + dw).clamp(self.config.w_min, self.config.w_max)
        
        # WEIGHT NORMALIZATION: Normalize each neuron's weights to prevent homogenization
        # This forces neurons to specialize - they can't just increase all weights
        row_sums = self.weights.sum(dim=1, keepdim=True)
        target_sum = self.config.n_input * 0.3  # Target: average weight of 0.3
        self.weights = self.weights * (target_sum / (row_sums + 1e-6))
        self.weights = self.weights.clamp(self.config.w_min, self.config.w_max)
        
        # ======================================================================
        # Step 5: Update prediction weights (learn to predict input)
        # ======================================================================
        # Gradient descent on prediction error
        pred_error_grad = predicted_input - input_rate  # (n_input,)
        dW_pred = -cfg.predictive_lr * torch.outer(self.output_rate_estimate, pred_error_grad)
        self.prediction_weights = (self.prediction_weights + dW_pred).clamp(-2.0, 2.0)
        
        # ======================================================================
        # Step 6: BCM homeostasis (prevent runaway)
        # ======================================================================
        output_rate_hz = output_spikes.mean(dim=0) * (1000.0 / dt)
        self.bcm_threshold = update_bcm_threshold(
            self.bcm_threshold,
            avg_activity_hz=output_rate_hz.unsqueeze(0),
            target_rate_hz=cfg.bcm_target_rate_hz,
            tau=cfg.bcm_tau_ms,
            min_threshold=0.1,
            max_threshold=0.9,
        )
        
        return {
            "pred_error": pred_error.item(),
            "pred_error_smooth": self.pred_error_smooth.item(),
            "error_modulation": error_modulation.item(),
            "dw_mean": dw.abs().mean().item(),
            "ltp": soft_ltp.sum().item(),
            "ltd": soft_ltd.sum().item(),
        }

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        novelty: float = 0.0,
        attention: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Learn from input-output pairs.
        
        For Hebbian mode: updates weights based on spike correlations.
        For Predictive Coding mode: updates prediction weights to minimize error.
        For Predictive STDP mode: STDP modulated by prediction error.
        """
        # ======================================================================
        # PREDICTIVE CODING LEARNING (rate-coded)
        # ======================================================================
        if self.cortex_config.learning_rule == LearningRule.PREDICTIVE:
            return self._predictive_learn(input_spikes)
        
        # ======================================================================
        # PREDICTIVE STDP LEARNING (spiking + prediction error modulation)
        # ======================================================================
        if self.cortex_config.learning_rule == LearningRule.PREDICTIVE_STDP:
            return self._predictive_stdp_learn(input_spikes, output_spikes)
        
        # ======================================================================
        # HEBBIAN LEARNING (default)
        # ======================================================================
        self.ach.compute(novelty=novelty, attention=attention)
        effective_lr = self.cortex_config.hebbian_lr * self.ach.get_learning_modulation()

        # ======================================================================
        # LEARNING: Triplet STDP or Simple Hebbian (both return dw)
        # ======================================================================
        if self.cortex_config.triplet_stdp_enabled and self.triplet_stdp is not None:
            # Use Triplet STDP for more biologically accurate learning
            dw = self.triplet_stdp(input_spikes, output_spikes).T * effective_lr
        else:
            # Use simple Hebbian (default) - returns dw
            dw = hebbian_update(
                self.weights, self.input_trace, output_spikes,
                learning_rate=effective_lr, w_max=self.config.w_max,
                heterosynaptic_ratio=self.cortex_config.heterosynaptic_ratio,
            )

        # ======================================================================
        # BCM HOMEOSTASIS: Modulate weight changes based on activity vs threshold
        # High-activity neurons get their potentiation reduced toward baseline
        # Low-activity neurons get their depression reduced toward baseline
        # This encourages diverse representations across the population
        # ======================================================================
        if dw.abs().sum() > 0:
            # BCM modulation: scale weight changes per output neuron
            bcm_thresh = self.bcm_threshold.squeeze()  # (n_output,)
            post_active = output_spikes.squeeze()  # (n_output,)
            bcm_mod = (post_active - bcm_thresh)  # (n_output,)
            
            # bcm_scale: neurons above threshold get reduced LTP (scale < 1)
            #            neurons below threshold get reduced LTD (scale > 1 for positive dw)
            bcm_strength = 2.0
            bcm_scale = 1.0 - 0.5 * torch.tanh(bcm_strength * bcm_mod)  # 0.5 to 1.5
            
            # Apply per-neuron scaling to weight changes
            # dw is (n_output, n_input), bcm_scale is (n_output,)
            dw_scaled = dw * bcm_scale.unsqueeze(1)  # broadcast across input dimension
        else:
            dw_scaled = dw
            
        # Apply weight updates with clamping
        self.weights = (self.weights + dw_scaled).clamp(self.config.w_min, self.config.w_max)

        # ======================================================================
        # SYNAPTIC SCALING: Global weight normalization
        # ======================================================================
        if self.cortex_config.synaptic_scaling_enabled:
            self.weights = synaptic_scaling(
                self.weights,
                target_norm_fraction=self.cortex_config.synaptic_scaling_target,
                tau=self.cortex_config.synaptic_scaling_tau,
                w_max=self.config.w_max,
            )

        # ======================================================================
        # HOMEOSTATIC CONDUCTANCE: Update tonic excitation based on activity
        # ======================================================================
        if self.cortex_config.homeostatic_conductance_enabled and self.homeostatic_g_tonic is not None:
            current_rate_hz = output_spikes.mean(dim=0) * (1000.0 / self.config.dt_ms)
            new_avg_rate, new_g_tonic = update_homeostatic_conductance(
                current_rate=current_rate_hz,
                avg_firing_rate=self.homeostatic_avg_rate,
                g_tonic=self.homeostatic_g_tonic,
                target_rate=self.cortex_config.bcm_target_rate_hz,
                tau=self.cortex_config.homeostatic_g_tau,
                strength=self.cortex_config.homeostatic_g_strength,
                bounds=self.cortex_config.homeostatic_g_bounds,
            )
            self.homeostatic_avg_rate = new_avg_rate
            self.homeostatic_g_tonic = new_g_tonic

        # ======================================================================
        # INTRINSIC PLASTICITY: Update per-neuron excitability based on activity
        # ======================================================================
        if self.cortex_config.intrinsic_plasticity_enabled and self.intrinsic_excitability is not None:
            current_rate_hz = output_spikes.mean(dim=0) * (1000.0 / self.config.dt_ms)
            new_avg_rate, new_excitability = update_homeostatic_excitability(
                current_rate=current_rate_hz,
                avg_firing_rate=self.intrinsic_avg_rate,
                excitability=self.intrinsic_excitability,
                target_rate=self.cortex_config.bcm_target_rate_hz,
                tau=self.cortex_config.intrinsic_plasticity_tau,
                strength=self.cortex_config.intrinsic_plasticity_strength,
                v_threshold=1.0,  # Normalized threshold
                bounds=self.cortex_config.intrinsic_plasticity_bounds,
            )
            self.intrinsic_avg_rate = new_avg_rate
            self.intrinsic_excitability = new_excitability

        # Compute LTP/LTD metrics from scaled weight delta
        ltp = dw_scaled[dw_scaled > 0].sum().item() if (dw_scaled > 0).any() else 0.0
        ltd = dw_scaled[dw_scaled < 0].sum().item() if (dw_scaled < 0).any() else 0.0

        # Update per-neuron BCM thresholds based on each neuron's firing rate
        if output_spikes.sum() > 0:
            # Per-neuron firing rate (Hz)
            per_neuron_rate_hz = output_spikes.squeeze() * (1000.0 / self.config.dt_ms)
            self.bcm_threshold = update_bcm_threshold(
                self.bcm_threshold.squeeze(), per_neuron_rate_hz, self.cortex_config.bcm_target_rate_hz,
                tau=self.cortex_config.bcm_tau_ms, min_threshold=0.01, max_threshold=2.0,
            ).unsqueeze(0)

        self.ach.decay(self.config.dt_ms)
        return {"ltp": ltp, "ltd": ltd, "net_change": ltp + ltd, "ach_level": self.ach.level}

    def reset(self) -> None:
        super().reset()
        self.input_trace.zero_()
        self.output_trace.zero_()
        self.recent_spikes.zero_()
        self.sfa_adaptation.zero_()
        self.activity_count.zero_()  # Reset activity boosting counter
        self.ach.reset()
        if self.neurons is not None:
            self.neurons.reset_state(1)
        if self.som_activation is not None:
            self.som_activation.zero_()
        if self.oscillation_phase is not None:
            self.oscillation_phase = 0.0
        if self.stp_vesicles is not None:
            self.stp_vesicles.fill_(1.0)
        if self.stp_facilitation is not None:
            self.stp_facilitation.zero_()
        if self.delay_buffer is not None:
            self.delay_buffer.zero_()
            self.delay_buffer_idx = 0
        if self.prev_output is not None:
            self.prev_output.zero_()
        if self.triplet_stdp is not None:
            self.triplet_stdp.reset_traces()
        if self.tonic_gaba is not None:
            self.tonic_gaba.fill_(self.cortex_config.tonic_gaba_conductance)
        if self.homeostatic_g_tonic is not None:
            self.homeostatic_g_tonic.zero_()
        if self.homeostatic_avg_rate is not None:
            self.homeostatic_avg_rate.zero_()
        if self.dendritic_neurons is not None:
            self.dendritic_neurons.reset_state(1)
        if self.intrinsic_excitability is not None:
            self.intrinsic_excitability.zero_()
        if self.intrinsic_avg_rate is not None:
            self.intrinsic_avg_rate.zero_()

    # =========================================================================
    # RATE-CODED API
    # These methods provide simpler interfaces for rate-coded experiments
    # =========================================================================
    
    def encode_rate(
        self,
        input_pattern: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode a rate-coded input pattern.
        
        Simple linear encoding through weights with optional normalization.
        Useful for experiments that don't need full spiking dynamics.
        
        Args:
            input_pattern: Rate-coded input [n_input] or [batch, n_input]
            normalize: Whether to apply tanh normalization
            
        Returns:
            Encoded pattern [batch, n_output] or [n_output]
        """
        if input_pattern.dim() == 1:
            input_pattern = input_pattern.unsqueeze(0)
        
        # Linear encoding through weights
        encoded = torch.matmul(input_pattern, self.weights.t())
        
        if normalize:
            encoded = torch.tanh(encoded)
        
        return encoded.squeeze(0) if encoded.shape[0] == 1 else encoded
    
    def decode_rate(
        self,
        encoded: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Decode an encoded pattern back to input space.
        
        Uses transposed weights for reconstruction (like an autoencoder).
        
        Args:
            encoded: Encoded pattern [n_output] or [batch, n_output]
            normalize: Whether to apply tanh normalization
            
        Returns:
            Reconstructed pattern [batch, n_input] or [n_input]
        """
        if encoded.dim() == 1:
            encoded = encoded.unsqueeze(0)
        
        # Linear decoding through transposed weights
        decoded = torch.matmul(encoded, self.weights)
        
        if normalize:
            decoded = torch.tanh(decoded)
        
        return decoded.squeeze(0) if decoded.shape[0] == 1 else decoded
    
    def learn_hebbian_rate(
        self,
        input_pattern: torch.Tensor,
        output_pattern: torch.Tensor,
        learning_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Apply Hebbian learning with rate-coded patterns.
        
        Args:
            input_pattern: Input pattern [n_input] or [batch, n_input]
            output_pattern: Output pattern [n_output] or [batch, n_output]
            learning_rate: Optional override for learning rate
            
        Returns:
            Dictionary with learning metrics
        """
        if input_pattern.dim() == 1:
            input_pattern = input_pattern.unsqueeze(0)
        if output_pattern.dim() == 1:
            output_pattern = output_pattern.unsqueeze(0)
        
        lr = learning_rate if learning_rate is not None else self.config.learning_rate
        
        # Hebbian: dW = lr * output^T @ input
        input_norm = input_pattern / (input_pattern.norm(dim=1, keepdim=True) + 1e-6)
        output_norm = output_pattern / (output_pattern.norm(dim=1, keepdim=True) + 1e-6)
        
        dW = lr * torch.matmul(output_norm.t(), input_norm)
        
        with torch.no_grad():
            self.weights.data += dW
            self.weights.data.clamp_(self.config.w_min, self.config.w_max)
        
        return {
            "weight_change_norm": float(dW.norm().item()),
            "weight_mean": float(self.weights.data.mean().item()),
            "learning_rate": lr
        }
