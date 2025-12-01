#!/usr/bin/env python3
"""
Shared utilities for THALIA experiments.

This module provides common infrastructure used across multiple experiments:
- Network configuration and setup
- Weight initialization (lognormal distribution, diagonal bias)
- CLI argument parsing helpers
- Evaluation and success criteria checking
- Output formatting

By centralizing these utilities, we:
1. Reduce code duplication across experiments
2. Ensure consistent hyperparameters and defaults
3. Make it easier to run comparative experiments
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from thalia.core import (
    LIFConfig,
    LIFNeuron,
    ConductanceLIFConfig,
    ConductanceLIF,
    DendriticNeuronConfig,
    DendriticNeuron,
    DendriticBranchConfig,
)
from thalia.diagnostics import DiagnosticConfig, MechanismConfig
from thalia.dynamics import (
    NetworkState,
    NetworkConfig,
    STPConfig,
    NMDAConfig,
    DendriticConfig,
    NeuromodulationConfig,
    create_synaptic_mechanisms,
)
from thalia.learning import TripletSTDPConfig, TripletSTDP


# =============================================================================
# BIOLOGICAL CONSTANTS (derived from literature, do not change)
# =============================================================================

# Temporal resolution: sub-millisecond for proper spike timing
DEFAULT_DT = 0.1  # ms (100μs resolution)

# LIF neuron parameters (Destexhe & Bhalla)
DEFAULT_TAU_MEM = 20.0  # ms (range: 10-30ms)
DEFAULT_V_THRESHOLD = 1.0  # normalized units
DEFAULT_NOISE_STD = 0.1  # membrane noise

# Synaptic weight bounds: derived from EPSP amplitudes
# Single synapse = 0.1-2mV, threshold = 10-20mV
# So ~5-10 coincident inputs needed to fire
N_COINCIDENT_FOR_FIRING = 5
K_FACTOR = 2.5  # safety factor

# Refractory periods (Na+/K+ channel dynamics)
DEFAULT_ABSOLUTE_REFRACTORY_MS = 2.0  # Na+ inactivation
DEFAULT_RELATIVE_REFRACTORY_MS = 3.0  # K+ recovery
DEFAULT_RELATIVE_REFRACTORY_FACTOR = 0.3

# Spike-frequency adaptation (Ca2+-activated K+ channels)
DEFAULT_SFA_TAU_MS = 200.0
DEFAULT_SFA_INCREMENT = 0.15
DEFAULT_SFA_STRENGTH = 1.5

# Oscillatory timescales
DEFAULT_THETA_PERIOD_MS = 160.0  # 6.25 Hz (theta band)
DEFAULT_GAMMA_PERIOD_MS = 10.0   # 100 Hz (gamma band)

# Triplet STDP time constants (Pfister & Gerstner 2006)
DEFAULT_TAU_PLUS = 16.8   # LTP window (ms)
DEFAULT_TAU_MINUS = 33.7  # LTD window (ms)
DEFAULT_TAU_X = 101.0     # Triplet pre trace (ms)
DEFAULT_TAU_Y = 125.0     # Triplet post trace (ms)

# BCM sliding threshold
DEFAULT_BCM_TAU = 200.0  # ms
DEFAULT_BCM_BOUNDS = (0.01, 2.0)

# Recurrent connection constraints (excitatory only - biological)
DEFAULT_RECURRENT_W_MIN = 0.0
DEFAULT_RECURRENT_W_MAX = 1.5


# =============================================================================
# DATACLASSES FOR CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a temporal pattern learning experiment."""

    # Network dimensions
    n_input: int = 20
    n_output: int = 10

    # Timing
    dt: float = DEFAULT_DT
    n_cycles: int = 120
    warmup_cycles: int = 2
    pattern_type: Literal["gapped", "circular"] = "gapped"
    gap_duration_ms: float = 50.0

    # Neuron model
    neuron_model: Literal["current", "conductance", "dendritic"] = "dendritic"

    # Learning
    w_max_scale: float = 1.0
    n_coincidences_to_learn: int = 100
    target_firing_rate_hz: float = 20.0
    diagonal_bias: float = 0.3
    heterosynaptic_ratio: float = 0.5

    # Oscillations
    theta_modulation_strength: float = 2.5
    sigma_inhibition: float = 4.0

    # Recurrent learning
    recurrent_start_cycle: int = 60
    recurrent_base_lr: float = 0.05

    # Device
    device: torch.device | None = None

    def __post_init__(self):
        if self.device is None:
            self.device = select_device(self.n_input + self.n_output)


@dataclass
class NetworkSetup:
    """Contains initialized network components ready for training."""

    # Neurons
    output_neurons: LIFNeuron | ConductanceLIF | DendriticNeuron

    # Weights
    weights: torch.Tensor
    initial_weights: torch.Tensor
    recurrent_weights: torch.Tensor

    # Configs
    net_config: NetworkConfig
    recurrent_stdp: TripletSTDP
    synaptic_mechanisms: dict

    # State
    train_state: NetworkState

    # Parameters for evaluation
    w_max: float
    n_input: int
    n_output: int
    device: torch.device


# =============================================================================
# DEVICE SELECTION
# =============================================================================

def select_device(n_neurons: int, verbose: bool = False) -> torch.device:
    """Select optimal device based on network size.

    For small networks (<2000 neurons), CPU is faster due to reduced
    GPU→CPU sync overhead. For larger networks, GPU wins.

    Args:
        n_neurons: Total number of neurons
        verbose: Print device selection info

    Returns:
        torch.device for computation
    """
    if torch.cuda.is_available() and n_neurons > 2000:
        device = torch.device("cuda")
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print(f"Using CPU (network size {n_neurons} < 2000)")
    return device


# =============================================================================
# WEIGHT INITIALIZATION
# =============================================================================

def create_weight_matrix(
    n_output: int,
    n_input: int,
    w_max: float,
    device: torch.device,
    initial_fraction: float = 0.1,
    diagonal_bias_strength: float = 0.3,
    distribution: Literal["lognormal", "uniform"] = "lognormal",
) -> torch.Tensor:
    """Create initial feedforward weight matrix with optional diagonal bias.

    The diagonal bias represents developmental pre-wiring (topographic maps
    from molecular gradients, spontaneous waves before eye opening).

    This is biologically necessary for symmetry breaking - without it,
    all neurons receive identical input and learn to respond to the same
    (late) input, causing representation collapse.

    Args:
        n_output: Number of output neurons
        n_input: Number of input neurons
        w_max: Maximum weight value
        device: Torch device
        initial_fraction: Initial weight as fraction of w_max (default 0.1)
        diagonal_bias_strength: Bias strength for topographic init (0=none, 0.3=moderate)
        distribution: Weight distribution type

    Returns:
        Weight matrix (n_output, n_input)

    Notes:
        - Minimum diagonal_bias for reliable learning is ~0.2 (from exp2 experiments)
        - Higher values (0.3) provide safety margin
        - With proper bias, 10/10 feedforward learning in ~120 cycles
    """
    if distribution == "lognormal":
        # Lognormal distribution (Song et al. 2005): many weak, few strong
        target_median = initial_fraction * w_max
        lognormal_mu = np.log(target_median)
        lognormal_sigma = 0.3  # tight spread
        weights = torch.exp(
            torch.randn(n_output, n_input, device=device) * lognormal_sigma + lognormal_mu
        )
        max_initial = min(0.3, initial_fraction * 3) * w_max
        weights = weights.clamp(0.01, max_initial)
    else:
        # Uniform distribution
        weights = torch.rand(n_output, n_input, device=device) * initial_fraction * w_max
        weights = weights.clamp(0.01, initial_fraction * w_max)

    # Apply diagonal bias for topographic initialization
    if diagonal_bias_strength > 0:
        ratio = n_input // n_output  # e.g., 20/10 = 2
        diagonal_bias = torch.zeros_like(weights)

        for i in range(n_output):
            # Primary inputs for neuron i: [ratio*i, ratio*i + ratio-1]
            for j in range(ratio):
                inp_idx = ratio * i + j
                if inp_idx < n_input:
                    diagonal_bias[i, inp_idx] = diagonal_bias_strength * w_max

            # Smaller gradient to neighbors (smooth, not cliff)
            if ratio * i > 0:
                diagonal_bias[i, ratio * i - 1] = 0.3 * diagonal_bias_strength * w_max
            if ratio * (i + 1) < n_input:
                diagonal_bias[i, ratio * (i + 1)] = 0.3 * diagonal_bias_strength * w_max

        weights = weights + diagonal_bias
        weights = weights.clamp(0.01, w_max * 0.9)  # leave headroom

    return weights


def create_recurrent_weights(
    n_output: int,
    device: torch.device,
    initial_mean: float = 0.15,
    initial_std: float = 0.05,
    w_min: float = DEFAULT_RECURRENT_W_MIN,
    w_max: float = DEFAULT_RECURRENT_W_MAX,
) -> torch.Tensor:
    """Create initial recurrent weight matrix.

    Recurrent weights are EXCITATORY ONLY (biological constraint).
    Inhibition comes from interneurons, not recurrent connections.

    Args:
        n_output: Number of neurons
        device: Torch device
        initial_mean: Mean initial weight
        initial_std: Std of initial weights
        w_min: Minimum weight (must be >= 0)
        w_max: Maximum weight

    Returns:
        Recurrent weight matrix (n_output, n_output) with zero diagonal
    """
    weights = torch.randn(n_output, n_output, device=device) * initial_std + initial_mean
    weights = weights * (1 - torch.eye(n_output, device=device))  # zero diagonal
    return weights.clamp(w_min, w_max)


# =============================================================================
# NETWORK SETUP
# =============================================================================

def create_output_neurons(
    n_output: int,
    neuron_model: Literal["current", "conductance", "dendritic"],
    device: torch.device,
    n_input: int = 20,
    dt: float = DEFAULT_DT,
    tau_mem: float = DEFAULT_TAU_MEM,
    v_threshold: float = DEFAULT_V_THRESHOLD,
    noise_std: float = DEFAULT_NOISE_STD,
    sfa_tau_ms: float = DEFAULT_SFA_TAU_MS,
    sfa_increment: float = DEFAULT_SFA_INCREMENT,
    acceleration_factor: float = 1.0,
    n_branches: int = 4,
    nmda_threshold: float = 0.3,
    nmda_gain: float = 1.5,
) -> LIFNeuron | ConductanceLIF | DendriticNeuron:
    """Create output neurons based on selected model type.

    Args:
        n_output: Number of output neurons
        neuron_model: Type of neuron model
        device: Torch device
        n_input: Number of inputs (for dendritic routing)
        dt: Timestep (ms)
        tau_mem: Membrane time constant (ms)
        v_threshold: Firing threshold
        noise_std: Membrane noise
        sfa_tau_ms: SFA time constant (ms)
        sfa_increment: SFA increment per spike
        acceleration_factor: Learning acceleration
        n_branches: Branches per neuron (dendritic only)
        nmda_threshold: NMDA spike threshold (dendritic only)
        nmda_gain: NMDA amplification (dendritic only)

    Returns:
        Initialized neuron layer
    """
    # Common conductance config for both conductance and dendritic
    cond_config = ConductanceLIFConfig(
        C_m=1.0,
        g_L=1.0 / tau_mem,
        E_L=0.0,
        E_E=3.0,
        E_I=-0.5,
        tau_E=5.0,
        tau_I=10.0,
        v_threshold=v_threshold,
        v_reset=-0.1,
        tau_ref=2.0,
        dt=dt,
        tau_adapt=sfa_tau_ms / acceleration_factor,
        adapt_increment=sfa_increment,
        E_adapt=-0.5,
        noise_std=noise_std,
    )

    if neuron_model == "dendritic":
        inputs_per_branch = n_input // n_branches
        branch_config = DendriticBranchConfig(
            nmda_threshold=nmda_threshold,
            nmda_gain=nmda_gain,
            plateau_tau_ms=50.0,
            tau_syn_ms=15.0,
            saturation_level=2.0,
            subthreshold_attenuation=0.8,
            branch_coupling=1.0,
            dt=dt,
        )
        neuron_config = DendriticNeuronConfig(
            n_branches=n_branches,
            inputs_per_branch=inputs_per_branch,
            branch_config=branch_config,
            soma_config=cond_config,
            input_routing="fixed",
        )
        return DendriticNeuron(n_neurons=n_output, config=neuron_config).to(device)

    elif neuron_model == "conductance":
        return ConductanceLIF(n_neurons=n_output, config=cond_config).to(device)

    else:  # current
        config = LIFConfig(
            tau_mem=tau_mem,
            v_threshold=v_threshold,
            v_reset=-0.1,
            noise_std=noise_std,
            dt=dt,
            v_min=-0.5,
        )
        return LIFNeuron(n_neurons=n_output, config=config).to(device)


def create_network_config(
    n_input: int,
    n_output: int,
    device: torch.device,
    dt: float = DEFAULT_DT,
    neuron_model: str = "dendritic",
    theta_modulation_strength: float = 2.5,
    sigma_inhibition: float = 4.0,
    shunting_relative_strength: float = 0.6,
    blanket_inhibition_strength: float = 0.5,
    som_strength: float = 0.5,
    acceleration_factor: float = 1.0,
    theta_phase_offset: float = 0.0,
    theta_mode: str = "uniform",
) -> tuple[NetworkConfig, dict]:
    """Create NetworkConfig and compute derived parameters.

    Args:
        n_input: Number of input neurons
        n_output: Number of output neurons
        device: Torch device
        dt: Timestep (ms)
        neuron_model: Type of neuron model
        theta_modulation_strength: Phase-based firing bias
        sigma_inhibition: Lateral inhibition spread
        shunting_relative_strength: Shunting inhibition fraction
        blanket_inhibition_strength: Global inhibition on spike
        som_strength: SOM+ inhibition strength
        acceleration_factor: Learning acceleration factor
        theta_phase_offset: Phase offset in input phases to compensate for
            membrane integration time (~14 phases for 8ms/phase inputs).
            Only used when theta_mode="per_neuron".
        theta_mode: How theta affects different neurons:
            - "uniform": All neurons get same theta modulation (biologically realistic)
            - "per_neuron": Each neuron has preferred theta phase (for sequence learning)

    Returns:
        Tuple of (NetworkConfig, derived_params dict)
    """
    # Timing parameters
    cycle_duration_ms = n_input * 8.0  # 160ms for n_input=20
    cycle_duration = int(cycle_duration_ms / dt)
    theta_period = int(DEFAULT_THETA_PERIOD_MS / dt)
    gamma_period = int(DEFAULT_GAMMA_PERIOD_MS / dt)

    interneuron_delay = int(2.0 / dt)  # 2ms disynaptic delay
    recurrent_delay = int(10.0 / dt)
    absolute_refractory = int(DEFAULT_ABSOLUTE_REFRACTORY_MS / dt)
    relative_refractory = int(DEFAULT_RELATIVE_REFRACTORY_MS / dt)

    # Shunting dynamics
    shunting_tau_ms = 5.0  # GABA_A decay
    shunting_decay = np.exp(-dt / shunting_tau_ms)
    shunting_strength = shunting_relative_strength / (1 - shunting_relative_strength)

    # SOM+ dynamics (accelerated)
    som_tau_biological_ms = 200.0
    som_tau_effective_ms = som_tau_biological_ms / acceleration_factor
    som_decay = np.exp(-dt / som_tau_effective_ms)

    # SFA dynamics (accelerated)
    sfa_tau_effective_ms = DEFAULT_SFA_TAU_MS / acceleration_factor
    sfa_decay = np.exp(-dt / sfa_tau_effective_ms)

    # Spatial structure (lateral inhibition kernel)
    output_positions = torch.arange(n_output, device=device, dtype=torch.float32)
    distance_matrix = torch.abs(output_positions.unsqueeze(0) - output_positions.unsqueeze(1))
    inhibition_kernel = torch.exp(-distance_matrix**2 / (2 * sigma_inhibition**2))
    inhibition_kernel = inhibition_kernel * (1 - torch.eye(n_output, device=device))

    # Theta phase preference (only used when theta_mode="per_neuron")
    # For "uniform" mode, all neurons have same phase (0.0)
    theta_phase_preference = torch.zeros(n_output, device=device)
    if theta_mode == "per_neuron":
        # Apply phase offset to compensate for membrane integration time
        phase_offset_radians = theta_phase_offset * 2 * np.pi / n_input
        for i in range(n_output):
            base_phase = ((2 * i + 1) / n_input) * 2 * np.pi
            theta_phase_preference[i] = (base_phase - phase_offset_radians) % (2 * np.pi)

    # Feedforward delays (axonal, 0.5-5ms)
    ff_delay_min_ms = 0.5
    ff_delay_max_ms = 5.0
    ff_delays_ms = torch.linspace(ff_delay_min_ms, ff_delay_max_ms, n_input, device=device)
    ff_delays = (ff_delays_ms / dt).int()
    max_ff_delay = int(ff_delay_max_ms / dt) + 1

    # Theta modulation scaling for conductance neurons
    theta_mod = theta_modulation_strength
    if neuron_model in ("conductance", "dendritic"):
        theta_mod = theta_modulation_strength / 3.0

    # Target firing rate
    target_rate = 20.0 / 1000.0 * dt  # 20 Hz → spikes/timestep

    net_config = NetworkConfig(
        n_input=n_input,
        n_output=n_output,
        device=device,
        dt=dt,
        recurrent_delay=recurrent_delay,
        interneuron_delay=interneuron_delay,
        theta_period=theta_period,
        gamma_period=gamma_period,
        cycle_duration=cycle_duration,
        effective_cycle=cycle_duration,  # Will be updated if gapped
        shunting_strength=shunting_strength,
        shunting_decay=shunting_decay,
        blanket_inhibition_strength=blanket_inhibition_strength,
        gamma_reset_factor=0.3,
        som_strength=som_strength,
        som_activation_rate=0.02,
        som_decay=som_decay,
        use_som_inhibition=True,
        sfa_strength=DEFAULT_SFA_STRENGTH,
        sfa_increment=DEFAULT_SFA_INCREMENT,
        sfa_decay=sfa_decay,
        absolute_refractory=absolute_refractory,
        relative_refractory=relative_refractory,
        relative_refractory_factor=DEFAULT_RELATIVE_REFRACTORY_FACTOR,
        theta_mode=theta_mode,  # "uniform" (default) or "per_neuron"
        theta_phase_preference=theta_phase_preference,
        theta_modulation_strength=theta_mod,
        theta_reversal=-0.5,
        v_threshold=DEFAULT_V_THRESHOLD,
        target_rate=target_rate,
        intrinsic_strength_fraction=0.002,
        inhibition_kernel=inhibition_kernel,
        ff_delays=ff_delays,
        max_ff_delay=max_ff_delay,
    )

    derived = {
        "cycle_duration": cycle_duration,
        "cycle_duration_ms": cycle_duration_ms,
        "theta_period": theta_period,
        "gamma_period": gamma_period,
        "max_ff_delay": max_ff_delay,
        "shunting_strength": shunting_strength,
    }

    return net_config, derived


# =============================================================================
# SYNAPTIC PLASTICITY SETUP
# =============================================================================

def create_default_synaptic_mechanisms(
    n_input: int,
    n_output: int,
    device: torch.device,
    enable_stp: bool = True,
    enable_nmda: bool = True,
) -> dict:
    """Create default short-term synaptic plasticity mechanisms.

    Args:
        n_input: Number of input neurons
        n_output: Number of output neurons
        device: Torch device
        enable_stp: Enable short-term plasticity (vesicle dynamics)
        enable_nmda: Enable NMDA receptor gating

    Returns:
        Dictionary of synaptic mechanism objects
    """
    stp_config = STPConfig(
        mode="depression",
        depression_rate=0.2,
        recovery_tau_ms=200.0,
        facilitation_rate=0.1,
        facilitation_tau_ms=50.0,
    ) if enable_stp else None

    nmda_config = NMDAConfig(nmda_fraction=0.3) if enable_nmda else None

    dendritic_config = DendriticConfig(saturation_threshold=2.0)
    neuromod_config = NeuromodulationConfig(
        dopamine_baseline=0.5,
        learning_rate_modulation=2.0,
    )

    return create_synaptic_mechanisms(
        n_pre=n_input,
        n_post=n_output,
        device=device,
        stp_config=stp_config,
        nmda_config=nmda_config,
        dendritic_config=dendritic_config,
        neuromod_config=neuromod_config,
    )


def create_recurrent_stdp(
    n_output: int,
    device: torch.device,
    w_min: float = DEFAULT_RECURRENT_W_MIN,
    w_max: float = DEFAULT_RECURRENT_W_MAX,
) -> TripletSTDP:
    """Create recurrent STDP module for sequence learning.

    Uses LTD > LTP for excitatory-only weights (biological constraint).

    Args:
        n_output: Number of neurons
        device: Torch device
        w_min: Minimum weight
        w_max: Maximum weight

    Returns:
        TripletSTDP module
    """
    config = TripletSTDPConfig(
        tau_plus=DEFAULT_TAU_PLUS,
        tau_minus=DEFAULT_TAU_MINUS,
        tau_x=DEFAULT_TAU_X,
        tau_y=DEFAULT_TAU_Y,
        a2_plus=0.005,
        a2_minus=0.010,  # LTD stronger
        a3_plus=0.003,
        a3_minus=0.006,
        w_max=w_max,
        w_min=w_min,
    )
    return TripletSTDP(n_pre=n_output, n_post=n_output, config=config).to(device)


# =============================================================================
# CLI ARGUMENT HELPERS
# =============================================================================

def add_common_experiment_args(parser: argparse.ArgumentParser) -> None:
    """Add common experiment arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    # Experiment configuration
    parser.add_argument("--n_input", type=int, default=20,
                        help="Number of input neurons")
    parser.add_argument("--n_output", type=int, default=10,
                        help="Number of output neurons")
    parser.add_argument("--max_cycles", type=int, default=20,
                        help="Training duration in cycles")
    parser.add_argument("--warmup_cycles", type=int, default=2,
                        help="Warmup cycles without learning")
    parser.add_argument("--pattern_type", type=str, default="gapped",
                        choices=["gapped", "circular"],
                        help="Pattern type")
    parser.add_argument("--gap_duration_ms", type=float, default=50.0,
                        help="Silence between sequences (ms)")

    # Neuron model
    parser.add_argument("--neuron_model", type=str, default="dendritic",
                        choices=["current", "conductance", "dendritic"],
                        help="Neuron model type")

    # Learning parameters
    parser.add_argument("--diagonal_bias", type=float, default=0.3,
                        help="Initial diagonal weight bias (0=none, 0.3=moderate)")
    parser.add_argument("--target_firing_rate_hz", type=float, default=20.0,
                        help="Target firing rate in Hz")
    parser.add_argument("--heterosynaptic_ratio", type=float, default=0.5,
                        help="Heterosynaptic LTD as fraction of LTP")

    # Oscillations
    parser.add_argument("--theta_modulation_strength", type=float, default=2.5,
                        help="Phase-based firing bias strength")
    parser.add_argument("--theta_mode", type=str, default="uniform",
                        choices=["uniform", "per_neuron"],
                        help="Theta modulation mode: 'uniform' (all neurons same, biological) or "
                             "'per_neuron' (each neuron has preferred phase, for sequence learning)")
    parser.add_argument("--sigma_inhibition", type=float, default=4.0,
                        help="Lateral inhibition spread")

    # Debug
    parser.add_argument("--verbose", action="store_true",
                        help="Print additional debug info")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip saving the figure")
    parser.add_argument("--diagnostics", type=str, default="summary",
                        choices=["none", "summary", "verbose", "debug"],
                        help="Diagnostic output level")


def add_dendritic_args(parser: argparse.ArgumentParser) -> None:
    """Add dendritic neuron arguments to a parser.

    These are used when --neuron_model=dendritic.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument("--n_branches", type=int, default=4,
                        help="Number of dendritic branches per neuron")
    parser.add_argument("--nmda_threshold", type=float, default=0.3,
                        help="NMDA spike threshold per branch")
    parser.add_argument("--nmda_gain", type=float, default=1.5,
                        help="NMDA spike amplification factor")
    parser.add_argument("--subthreshold_attenuation", type=float, default=0.8,
                        help="Attenuation for inputs below NMDA threshold")


def add_inhibition_args(parser: argparse.ArgumentParser) -> None:
    """Add inhibition-related arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument("--shunting_relative_strength", type=float, default=0.6,
                        help="Shunting inhibition as fraction of total conductance")
    parser.add_argument("--blanket_inhibition", type=float, default=0.5,
                        help="Global inhibition strength on any spike")
    parser.add_argument("--som_strength", type=float, default=0.5,
                        help="SOM+ inhibition strength")
    parser.add_argument("--g_inh_tonic_max", type=float, default=5.0,
                        help="Maximum tonic inhibitory conductance for homeostasis")


def add_learning_args(parser: argparse.ArgumentParser) -> None:
    """Add learning-related arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument("--initial_weight_fraction", type=float, default=0.1,
                        help="Initial weight as fraction of w_max")
    parser.add_argument("--eligibility_tau_factor", type=float, default=1.5,
                        help="Eligibility tau as multiple of phase_duration")
    parser.add_argument("--homeostatic_strength_hz", type=float, default=0.01,
                        help="Threshold shift per Hz of rate error")
    parser.add_argument("--theta_phase_offset", type=float, default=0.0,
                        help="Theta phase offset in input phases")
    parser.add_argument("--recurrent_start_cycle", type=int, default=1,
                        help="Cycle to start recurrent learning")
    parser.add_argument("--acceleration_factor", type=float, default=1.0,
                        help="Learning acceleration factor (1.0=real-time)")


def add_mechanism_ablation_args(parser: argparse.ArgumentParser) -> None:
    """Add mechanism ablation arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    # Core mechanisms
    parser.add_argument("--disable_stp", action="store_true",
                        help="Disable short-term plasticity")
    parser.add_argument("--disable_nmda", action="store_true",
                        help="Disable NMDA receptor dynamics")
    parser.add_argument("--disable_som", action="store_true",
                        help="Disable SOM+ interneuron inhibition")
    parser.add_argument("--disable_lateral", action="store_true",
                        help="Disable lateral inhibition")
    parser.add_argument("--disable_shunting", action="store_true",
                        help="Disable shunting inhibition")
    parser.add_argument("--disable_sfa", action="store_true",
                        help="Disable spike-frequency adaptation")
    parser.add_argument("--disable_homeostasis", action="store_true",
                        help="Disable firing rate homeostasis")
    parser.add_argument("--disable_bcm", action="store_true",
                        help="Disable BCM threshold dynamics")
    parser.add_argument("--disable_theta", action="store_true",
                        help="Disable theta rhythm modulation")

    # Learning mechanisms
    parser.add_argument("--enable_synaptic_scaling", action="store_true",
                        help="Enable synaptic scaling (disabled by default - too slow)")
    parser.add_argument("--disable_feedforward_learning", action="store_true",
                        help="Disable feedforward Hebbian learning")
    parser.add_argument("--disable_recurrent_learning", action="store_true",
                        help="Disable recurrent Hebbian learning")
    parser.add_argument("--disable_heterosynaptic", action="store_true",
                        help="Disable heterosynaptic LTD")


def create_mechanism_config(args) -> MechanismConfig:
    """Create MechanismConfig from parsed CLI arguments.

    Args:
        args: Parsed argparse namespace

    Returns:
        MechanismConfig with ablation settings
    """
    return MechanismConfig(
        enable_stp=not getattr(args, "disable_stp", False),
        enable_nmda=not getattr(args, "disable_nmda", False),
        enable_som_inhibition=not getattr(args, "disable_som", False),
        enable_lateral_inhibition=not getattr(args, "disable_lateral", False),
        enable_shunting_inhibition=not getattr(args, "disable_shunting", False),
        enable_sfa=not getattr(args, "disable_sfa", False),
        enable_homeostasis=not getattr(args, "disable_homeostasis", False),
        enable_bcm=not getattr(args, "disable_bcm", False),
        enable_theta_modulation=not getattr(args, "disable_theta", False),
        enable_synaptic_scaling=getattr(args, "enable_synaptic_scaling", False),
        enable_feedforward_learning=not getattr(args, "disable_feedforward_learning", False),
        enable_recurrent_learning=not getattr(args, "disable_recurrent_learning", False),
        enable_heterosynaptic_ltd=not getattr(args, "disable_heterosynaptic", False),
    )


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_experiment_header(
    title: str,
    config: ExperimentConfig | None = None,
    extra_info: dict | None = None,
) -> None:
    """Print formatted experiment header.

    Args:
        title: Experiment title
        config: Optional experiment configuration
        extra_info: Additional key-value pairs to display
    """
    print("=" * 60)
    print(title)
    print("=" * 60)

    if config:
        print(f"\nNetwork: {config.n_input} inputs → {config.n_output} outputs")
        print(f"Neuron model: {config.neuron_model}")
        print(f"Pattern type: {config.pattern_type}")
        print(f"Training: {config.n_cycles} cycles")
        if config.diagonal_bias > 0:
            print(f"Diagonal bias: {config.diagonal_bias:.2f}")

    if extra_info:
        print("\nAdditional settings:")
        for key, value in extra_info.items():
            print(f"  {key}: {value}")


def print_success_criteria(
    criteria: list[tuple[str, bool]],
    title: str = "Success Criteria Check",
) -> bool:
    """Print success criteria and return overall pass/fail.

    Args:
        criteria: List of (name, passed) tuples
        title: Section title

    Returns:
        True if all criteria passed
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    all_passed = True
    for name, passed in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed

    print("\n" + ("🎉 All criteria passed!" if all_passed else "⚠️ Some criteria failed"))
    return all_passed


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def compute_w_max(w_max_scale: float = 1.0) -> float:
    """Compute biologically-derived maximum weight.

    Args:
        w_max_scale: Scaling factor for learning difficulty

    Returns:
        Maximum weight value
    """
    w_max_biological = K_FACTOR * DEFAULT_V_THRESHOLD / N_COINCIDENT_FOR_FIRING
    return w_max_biological * w_max_scale


def compute_hebbian_learning_rate(n_coincidences: int = 100) -> float:
    """Compute Hebbian learning rate from desired coincidences to learn.

    With soft-bounded Hebbian (Δw = lr * (w_max - w) * pre * post),
    after N coincidences starting from 0: w ≈ w_max * (1 - (1-lr)^N)

    Args:
        n_coincidences: Number of coincidences to reach 90% of w_max

    Returns:
        Learning rate
    """
    return 1.0 - 0.1 ** (1.0 / n_coincidences)


def get_results_dir() -> Path:
    """Get the experiments results directory, creating if needed.

    Returns:
        Path to results directory
    """
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir
