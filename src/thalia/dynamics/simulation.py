"""
Network simulation infrastructure for spiking neural networks.

This module provides the core simulation loop with biologically realistic
mechanisms including:
- Winner-take-all (WTA) dynamics
- Shunting (divisive) inhibition - conductance-based GABAergic
- Fast (PV+) inhibition: rapid onset/offset, somatic targeting, output-driven
- Slow (SOM+) inhibition: surround suppression, dendritic targeting, input-driven
- Spike-frequency adaptation (Ca2+-activated K+ channels)
- Refractory periods (absolute and relative)
- Theta-gamma oscillatory coupling
- Intrinsic plasticity (dynamic thresholds for stability)

Note: Homeostatic excitability updates are in thalia.learning.phase_homeostasis.

Key functions (in order of abstraction):
- forward_timestep: Core single-timestep forward pass (no STP)
- forward_timestep_with_stp: Single timestep with STP modulation
- forward_pattern: Full pattern over T timesteps

Usage:
    from thalia.dynamics.simulation import NetworkState, NetworkConfig, forward_timestep

    # Create state and config
    state = NetworkState.create(n_output=10, recurrent_delay=100, ...)
    config = NetworkConfig(n_input=20, n_output=10, ...)

    # Run single timestep
    for t in range(duration):
        output_spikes = forward_timestep(t, input_spikes[t], state, config, ...)
    
    # Or run entire pattern at once
    output_spikes, winners = forward_pattern(input_pattern, state, config, ...)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Any, TYPE_CHECKING
import torch
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from thalia.core import LIFNeuron


def select_device(
    n_neurons: int,
    threshold: int = 2000,
    prefer_gpu: bool = True,
    verbose: bool = False,
) -> torch.device:
    """Select optimal device (CPU or GPU) based on network size.

    For small networks, CPU is typically faster due to GPU kernel launch overhead.
    This function automatically selects the best device based on network size.

    Args:
        n_neurons: Total number of neurons in the network (input + output).
        threshold: Networks with fewer neurons than this use CPU. Default 2000.
            Based on benchmarks (Nov 2025), CPU is faster for:
            - LIF neurons: all tested sizes (up to 2000)
            - ConductanceLIF: all tested sizes
            - DendriticNeuron: all tested sizes
            GPU becomes faster only for large weight updates (hebbian_update)
            at >1000 neurons, and even then only ~1.8× faster.
        prefer_gpu: If True and GPU available and n_neurons >= threshold, use GPU.
        verbose: If True, print device selection info.

    Returns:
        torch.device for CPU or CUDA.

    Example:
        >>> device = select_device(n_input + n_output, verbose=True)
        >>> # Selecting device: cpu (500 neurons < 2000 threshold)

    Rationale:
        GPU kernel launch overhead (~100-500 μs per kernel) dominates for small
        networks and batch sizes. Benchmarks show:
        - CPU is 3× faster for LIF/ConductanceLIF at 1000 neurons
        - CPU is 2× faster for DendriticNeuron at 1000 neurons
        - GPU only wins for heavy operations like hebbian_update at >1000 neurons
        The threshold of 2000 is conservative; for batch_size=1, CPU may be
        faster even for larger networks.
    """
    cuda_available = torch.cuda.is_available()

    if n_neurons >= threshold and prefer_gpu and cuda_available:
        device = torch.device("cuda")
        reason = f"{n_neurons} neurons >= {threshold} threshold, GPU available"
    elif cuda_available and n_neurons >= threshold:
        device = torch.device("cuda")
        reason = f"{n_neurons} neurons >= {threshold} threshold"
    else:
        device = torch.device("cpu")
        if n_neurons < threshold:
            reason = f"{n_neurons} neurons < {threshold} threshold (CPU faster)"
        else:
            reason = "GPU not available"

    if verbose:
        print(f"Device selection: {device} ({reason})")

    return device


@dataclass
class NetworkState:
    """All state variables needed for network simulation.

    This dataclass holds all the dynamic state that changes during simulation,
    including spike history, inhibition levels, adaptation currents, and
    plasticity-related variables.

    Use NetworkState.create() to initialize with proper tensor shapes.
    All tensor fields are initialized to None but will be populated by create().
    """
    # Spike history for delayed recurrent connections
    spike_history: torch.Tensor = field(default=None)  # type: ignore[assignment]
    history_index: int = 0

    # Feedforward input history for axonal delays
    # Shape: (max_ff_delay, n_input) - circular buffer of past inputs
    ff_input_history: torch.Tensor = field(default=None)  # type: ignore[assignment]
    ff_history_index: int = 0

    # Inhibition state (shunting only - divisive inhibition is more biologically realistic)
    shunting_conductance: torch.Tensor = field(default=None)  # type: ignore[assignment]
    inhibition_buffer: torch.Tensor = field(default=None)  # type: ignore[assignment]
    inhibition_buffer_index: int = 0

    # SOM+ (somatostatin) interneuron state - slow, surround suppression
    som_inhibition: torch.Tensor = field(default=None)  # type: ignore[assignment]  # Current SOM+ inhibition level
    som_activation: torch.Tensor = field(default=None)  # type: ignore[assignment]  # SOM+ interneuron activation (builds up slowly)

    # Adaptation state
    sfa_current: torch.Tensor = field(default=None)  # type: ignore[assignment]
    time_since_spike: torch.Tensor = field(default=None)  # type: ignore[assignment]

    # Plasticity state (only used during training)
    # For current-based neurons: use dynamic_threshold and excitability as currents
    # For conductance-based neurons: use g_adapt_external and g_tonic as conductances
    dynamic_threshold: torch.Tensor = field(default=None)  # type: ignore[assignment]
    excitability: torch.Tensor = field(default=None)  # type: ignore[assignment]
    avg_firing_rate: torch.Tensor = field(default=None)  # type: ignore[assignment]

    # Native conductance-based plasticity state (for ConductanceLIF)
    # g_tonic: Tonic excitatory conductance for homeostatic excitability
    #   - Increased g_tonic = more excitable (like persistent Na+ or reduced leak)
    #   - Updated by homeostatic plasticity to maintain target firing rate
    g_tonic: Optional[torch.Tensor] = None
    # g_inh_tonic: Tonic inhibitory conductance (extrasynaptic GABA_A)
    #   - Models α5/δ-subunit GABA_A receptors activated by ambient GABA
    #   - Increased g_inh_tonic = less excitable (more tonic inhibition)
    #   - Updated by homeostatic plasticity when neurons fire too much
    #   - Biologically: ~5-10% of total inhibitory conductance at rest
    g_inh_tonic: Optional[torch.Tensor] = None
    # g_adapt_external: External adaptation conductance for intrinsic plasticity
    #   - Added to neuron's internal g_adapt
    #   - Increased after spiking to raise effective threshold
    #   - Provides activity-dependent threshold modulation
    g_adapt_external: Optional[torch.Tensor] = None

    # Tracking
    winners: List[int] = field(default_factory=list)
    output_spikes_list: List[NDArray[np.floating[Any]]] = field(default_factory=list)
    # GPU-based spike history (optional, for efficient tracking)
    output_spikes_tensor: Optional[torch.Tensor] = None
    spike_tensor_index: int = 0
    # Skip spike tracking entirely (for training where we don't need history)
    skip_spike_tracking: bool = False

    # Diagnostic fields (populated by simulate_step for debugging)
    last_feedforward_current: Optional[torch.Tensor] = None
    last_recurrent_current: Optional[torch.Tensor] = None
    last_effective_current: Optional[torch.Tensor] = None
    last_theta_modulation: Optional[torch.Tensor] = None
    # Delayed input used for feedforward computation (for accurate Hebbian learning)
    last_delayed_input: Optional[torch.Tensor] = None

    # Input eligibility trace for Hebbian learning
    # Tracks recent input activity with exponential decay, allowing LTP even when
    # output fires slightly after the input spike (membrane integration delay)
    input_eligibility: Optional[torch.Tensor] = None
    # Decay time constant in timesteps. Must be >= phase duration for proper learning!
    # With 8ms phases at dt=0.1ms (80 timesteps), tau should be >= 80 to avoid recency bias.
    # Biological NMDA receptor decay is ~50-100ms, supporting longer traces.
    input_eligibility_tau: float = 100.0  # 10ms at dt=0.1ms

    @classmethod
    def create(cls, n_output: int, recurrent_delay: int, interneuron_delay: int,
               v_threshold: float, target_rate: float,
               device: Union[torch.device, str],
               n_input: int = 0, max_ff_delay: int = 1) -> 'NetworkState':
        """Create a fresh network state with all tensors initialized.

        Args:
            n_output: Number of output neurons
            recurrent_delay: Delay for recurrent connections (in timesteps)
            interneuron_delay: Delay for interneuron inhibition (in timesteps)
            v_threshold: Voltage threshold for intrinsic plasticity initialization
            target_rate: Target firing rate for homeostasis
            device: Torch device (cpu or cuda) or string like "cuda:0"
            n_input: Number of input neurons (for feedforward delay buffer)
            max_ff_delay: Maximum feedforward delay in timesteps

        Returns:
            Initialized NetworkState with all tensors on the specified device
        """
        if isinstance(device, str):
            device = torch.device(device)
        return cls(
            spike_history=torch.zeros(recurrent_delay, n_output, device=device),
            history_index=0,
            ff_input_history=torch.zeros(max(1, max_ff_delay), n_input, device=device) if n_input > 0 else None,
            ff_history_index=0,
            shunting_conductance=torch.zeros(1, n_output, device=device),
            inhibition_buffer=torch.zeros(interneuron_delay, n_output, device=device),
            inhibition_buffer_index=0,
            som_inhibition=torch.zeros(1, n_output, device=device),
            som_activation=torch.zeros(1, n_output, device=device),
            sfa_current=torch.zeros(1, n_output, device=device),
            time_since_spike=torch.ones(1, n_output, device=device) * 100,  # Start as "long ago"
            dynamic_threshold=torch.ones(1, n_output, device=device) * v_threshold,
            excitability=torch.zeros(1, n_output, device=device),
            avg_firing_rate=torch.ones(1, n_output, device=device) * target_rate,
            # Native conductance fields (for ConductanceLIF)
            g_tonic=torch.zeros(1, n_output, device=device),
            g_inh_tonic=torch.zeros(1, n_output, device=device),
            g_adapt_external=torch.zeros(1, n_output, device=device),
            winners=[],
            output_spikes_list=[],
        )

    def reset_tracking(self):
        """Reset tracking lists for a new simulation run."""
        self.winners = []
        self.output_spikes_list = []
        self.output_spikes_tensor = None
        self.spike_tensor_index = 0

    def reset_all(self, v_threshold: float, target_rate: float):
        """Reset all state tensors to initial values.

        Args:
            v_threshold: Voltage threshold for intrinsic plasticity
            target_rate: Target firing rate for homeostasis
        """
        self.spike_history.zero_()
        self.history_index = 0
        self.shunting_conductance.zero_()
        self.inhibition_buffer.zero_()
        self.inhibition_buffer_index = 0
        self.som_inhibition.zero_()
        self.som_activation.zero_()
        self.sfa_current.zero_()
        self.time_since_spike.fill_(100)
        self.dynamic_threshold.fill_(v_threshold)
        self.excitability.zero_()
        self.avg_firing_rate.fill_(target_rate)
        # Reset native conductance fields
        if self.g_tonic is not None:
            self.g_tonic.zero_()
        if self.g_inh_tonic is not None:
            self.g_inh_tonic.zero_()
        if self.g_adapt_external is not None:
            self.g_adapt_external.zero_()
        self.reset_tracking()


@dataclass
class SynapticInputs:
    """Container for computed synaptic inputs.

    Holds both current-based and conductance-based representations
    of synaptic inputs, along with metadata about phase/timing.
    """
    # Raw currents (computed from weights × spikes)
    feedforward_current: torch.Tensor  # (1, n_output)
    recurrent_current: torch.Tensor    # (1, n_output)

    # Conductances (for conductance-based neurons)
    g_exc: Optional[torch.Tensor] = None  # Total excitatory conductance
    g_inh: Optional[torch.Tensor] = None  # Total inhibitory conductance
    g_nmda: Optional[torch.Tensor] = None  # NMDA-specific conductance

    # Timing context
    phase_number: int = 0
    cycle_position: float = 0.0
    is_gamma_reset: bool = False

    # Delayed input (for learning)
    delayed_input: Optional[torch.Tensor] = None


@dataclass
class NetworkConfig:
    """All configuration parameters for network simulation.

    This dataclass holds all the hyperparameters that control network behavior.
    All timing parameters are in TIMESTEPS (not milliseconds) for efficiency.
    """
    # Network dimensions
    n_input: int
    n_output: int
    device: Union[torch.device, str]

    # Timing parameters (in TIMESTEPS)
    dt: float  # Simulation timestep in ms
    recurrent_delay: int  # In timesteps
    interneuron_delay: int  # In timesteps
    theta_period: int  # In timesteps
    gamma_period: int  # In timesteps
    cycle_duration: int  # In timesteps
    effective_cycle: int  # In timesteps (includes gaps)

    # Inhibition parameters (shunting only - divisive inhibition is biologically realistic)
    # PV+ (parvalbumin): fast, somatic, rapid onset/offset
    shunting_strength: float
    shunting_decay: float
    blanket_inhibition_strength: float
    gamma_reset_factor: float

    # SOM+ (somatostatin) interneuron parameters: slow, dendritic, surround suppression
    # SOM+ neurons are activated by sustained local network activity and provide
    # slower, longer-lasting inhibition that suppresses neurons with similar inputs
    som_strength: float              # Strength of SOM+ inhibition
    som_activation_rate: float       # How fast SOM+ interneurons activate (slow buildup)
    som_decay: float                 # Slow decay (τ ≈ 200ms at dt=0.1ms)
    use_som_inhibition: bool         # Enable/disable SOM+ inhibition

    # Spike-frequency adaptation parameters
    sfa_strength: float
    sfa_increment: float
    sfa_decay: float

    # Refractory parameters (in TIMESTEPS)
    absolute_refractory: int
    relative_refractory: int
    relative_refractory_factor: float

    # Theta modulation (conductance-based for biological realism)
    # Theta rhythm modulates excitability via GABAergic interneurons
    # Out-of-phase neurons receive inhibition that pulls toward E_theta (BELOW rest)
    # In-phase neurons receive less inhibition (disinhibition = excitation)
    #
    # CRITICAL: E_theta must be BELOW v_rest for theta to have any effect!
    # In biology: E_GABA ≈ -70 to -80mV, v_rest ≈ -65mV
    # In normalized units: E_theta ≈ -0.3 to -0.5, v_rest = 0.0
    #
    # theta_mode controls how theta affects different neurons:
    #   - "uniform": All neurons receive the same theta modulation (biologically realistic)
    #   - "per_neuron": Each neuron has a preferred theta phase (useful for sequence learning)
    theta_mode: str  # "uniform" (default) or "per_neuron"
    theta_phase_preference: torch.Tensor  # Only used when theta_mode="per_neuron"
    theta_modulation_strength: float  # Max conductance (g_theta)
    theta_reversal: float             # Reversal potential (E_theta < v_rest for GABA!)

    # Intrinsic Plasticity parameters
    v_threshold: float
    target_rate: float
    intrinsic_strength_fraction: float

    # Spatial inhibition kernel
    inhibition_kernel: torch.Tensor

    # Feedforward axonal delay parameters (optional, with defaults)
    # Real axons have 0.5-5ms conduction delays depending on myelination and distance
    # ff_delays: per-input delay in timesteps (shape: n_input,), or None for no delays
    ff_delays: Optional[torch.Tensor] = None  # Per-input axonal delays
    max_ff_delay: int = 1  # Maximum delay for buffer sizing

    @classmethod
    def create_default(cls, n_input: int, n_output: int, dt: float = 0.1,
                       device: Optional[torch.device] = None) -> 'NetworkConfig':
        """Create a NetworkConfig with sensible default values.

        Args:
            n_input: Number of input neurons
            n_output: Number of output neurons
            dt: Timestep in ms (default 0.1)
            device: Torch device (default: cuda if available, else cpu)

        Returns:
            NetworkConfig with biologically-inspired default parameters
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Timing parameters (in ms, converted to timesteps)
        theta_period_ms = 160.0
        gamma_period_ms = 10.0
        interneuron_delay_ms = 2.0
        recurrent_delay_ms = 10.0
        absolute_refractory_ms = 2.0
        relative_refractory_ms = 3.0

        # Compute timesteps
        theta_period = int(theta_period_ms / dt)
        gamma_period = int(gamma_period_ms / dt)
        interneuron_delay = int(interneuron_delay_ms / dt)
        recurrent_delay = int(recurrent_delay_ms / dt)
        absolute_refractory = int(absolute_refractory_ms / dt)
        relative_refractory = int(relative_refractory_ms / dt)

        # Pattern timing
        phase_duration_ms = 8.0
        cycle_duration_ms = n_input * phase_duration_ms
        gap_duration_ms = 50.0
        effective_cycle_ms = cycle_duration_ms + gap_duration_ms
        cycle_duration = int(cycle_duration_ms / dt)
        effective_cycle = int(effective_cycle_ms / dt)

        # Theta phase preferences (only used when theta_mode="per_neuron")
        # For default "uniform" mode, all neurons have same phase (0.0)
        theta_phase_preference = torch.zeros(n_output, device=device)

        # Spatial inhibition kernel (Gaussian falloff)
        output_positions = torch.arange(n_output, device=device, dtype=torch.float32)
        distance_matrix = torch.abs(output_positions.unsqueeze(0) - output_positions.unsqueeze(1))
        sigma_inhibition = 2.0
        inhibition_kernel = torch.exp(-distance_matrix**2 / (2 * sigma_inhibition**2))
        inhibition_kernel = inhibition_kernel * (1 - torch.eye(n_output, device=device))

        # Target firing rate
        target_firing_rate_hz = 20.0
        target_rate = target_firing_rate_hz / 1000.0 * dt

        return cls(
            n_input=n_input,
            n_output=n_output,
            device=device,
            dt=dt,
            recurrent_delay=recurrent_delay,
            interneuron_delay=interneuron_delay,
            theta_period=theta_period,
            gamma_period=gamma_period,
            cycle_duration=cycle_duration,
            effective_cycle=effective_cycle,
            shunting_strength=3.0,  # relative_factor / (1 - relative_factor) with factor=0.75
            shunting_decay=0.5,
            blanket_inhibition_strength=0.5,
            gamma_reset_factor=0.3,
            som_strength=0.5,
            som_activation_rate=0.02,
            som_decay=0.995,
            use_som_inhibition=True,
            sfa_strength=1.5,
            sfa_increment=0.15,
            sfa_decay=0.995 ** dt,
            absolute_refractory=absolute_refractory,
            relative_refractory=relative_refractory,
            relative_refractory_factor=0.3,
            theta_mode="uniform",  # Default: all neurons get same theta modulation
            theta_phase_preference=theta_phase_preference,
            theta_modulation_strength=2.5,
            theta_reversal=-0.5,  # E_theta < v_rest (GABAergic, ~E_GABA)
            v_threshold=1.0,
            target_rate=target_rate,
            intrinsic_strength_fraction=0.002,
            inhibition_kernel=inhibition_kernel,
        )


# =============================================================================
# HELPER FUNCTIONS FOR SIMULATE_STEP
# =============================================================================


def _compute_phase_context(t: int, config: NetworkConfig) -> Tuple[int, float, bool]:
    """Compute phase timing context for the current timestep.

    Args:
        t: Current timestep
        config: Network configuration

    Returns:
        Tuple of (phase_number, cycle_position, is_gamma_reset)
    """
    cycle_position = t % config.effective_cycle
    phase_number = int(cycle_position // (config.cycle_duration / config.n_output))
    is_gamma_reset = (t % config.gamma_period) == 0
    return phase_number, cycle_position, is_gamma_reset


def _get_delayed_input(
    input_spikes: torch.Tensor,
    state: NetworkState,
    config: NetworkConfig,
) -> torch.Tensor:
    """Get input spikes with axonal delays applied.

    Args:
        input_spikes: Current input spikes (1, n_input)
        state: Network state with input history
        config: Network configuration with delay info

    Returns:
        Delayed input spikes (1, n_input)
    """
    if config.ff_delays is None or state.ff_input_history is None:
        return input_spikes

    n_input = input_spikes.shape[1]
    delayed_input = torch.zeros_like(input_spikes.squeeze())

    for i in range(n_input):
        delay = int(config.ff_delays[i].item())
        if delay == 0:
            delayed_input[i] = input_spikes[0, i]
        else:
            hist_idx = int((state.ff_history_index - delay) % config.max_ff_delay)
            delayed_input[i] = state.ff_input_history[hist_idx, i]

    # Update history buffer
    state.ff_input_history[state.ff_history_index] = input_spikes.squeeze()
    state.ff_history_index = (state.ff_history_index + 1) % config.max_ff_delay

    return delayed_input.unsqueeze(0)


def _compute_theta_modulation(
    t: int,
    config: NetworkConfig,
    is_conductance: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute theta rhythm modulation as conductance.

    Theta rhythm modulates excitability via GABAergic interneurons.
    
    Two modes are supported:
    - "uniform": All neurons receive the same theta modulation (biologically realistic).
      Theta oscillation provides global rhythmic gating but doesn't bias any specific neuron.
    - "per_neuron": Each neuron has a preferred theta phase. Neurons fire more easily
      at their preferred phase. Useful for sequence learning where you want neurons
      to specialize for different temporal positions.

    Args:
        t: Current timestep
        config: Network configuration
        is_conductance: Whether using conductance-based neurons

    Returns:
        Tuple of (theta_conductance, theta_current_equivalent)
        - theta_conductance: For conductance neurons (add to g_inh)
        - theta_current_equivalent: For current neurons (add to effective_current)
    """
    # Compute phase of theta oscillation (0 to 2π)
    theta_phase = (2 * np.pi * t / config.theta_period) % (2 * np.pi)

    if config.theta_mode == "uniform":
        # Uniform theta: all neurons receive same modulation based on global theta phase
        # At theta_phase=0: modulation=0 (most excitable)
        # At theta_phase=π: modulation=1 (most inhibited)
        modulation = (1 - np.cos(theta_phase)) / 2
        # Broadcast to all neurons
        modulation = torch.full(
            (1, config.n_output), modulation, 
            device=config.device, dtype=torch.float32
        )
    else:
        # Per-neuron phase preference: each neuron has preferred theta phase
        # Compute phase difference for each neuron
        phase_diff = theta_phase - config.theta_phase_preference
        # Modulation: 0 at preferred phase, max at anti-phase
        # cos(0) = 1, cos(π) = -1, so (1 - cos(diff))/2 gives 0 to 1
        modulation = (1 - torch.cos(phase_diff)) / 2
        modulation = modulation.unsqueeze(0)

    # Theta conductance (inhibitory, add to g_inh)
    theta_g = config.theta_modulation_strength * modulation

    # For current-based neurons, convert to current effect
    # I_theta = g_theta * (E_theta - V_rest) where V_rest = 0
    # This is negative (hyperpolarizing) because E_theta < 0
    theta_current = theta_g * config.theta_reversal

    return theta_g, theta_current


def _apply_refractory_and_adaptation(
    current: torch.Tensor,
    state: NetworkState,
    config: NetworkConfig,
    is_conductance: bool,
) -> torch.Tensor:
    """Apply refractory period and spike-frequency adaptation.

    Args:
        current: Input current before refractory/adaptation
        state: Network state
        config: Network configuration
        is_conductance: Whether using conductance-based neurons

    Returns:
        Current after refractory/adaptation effects
    """
    # Refractory period masking
    refractory_mask = torch.ones_like(current)

    # Absolute refractory: completely block input
    abs_refrac = state.time_since_spike < config.absolute_refractory
    refractory_mask[abs_refrac] = 0.0

    # Relative refractory: reduced excitability
    rel_refrac = (state.time_since_spike >= config.absolute_refractory) & \
                 (state.time_since_spike < config.absolute_refractory + config.relative_refractory)
    refractory_mask[rel_refrac] = config.relative_refractory_factor

    current = current * refractory_mask

    # Apply SFA (spike-frequency adaptation)
    # For conductance neurons, SFA is handled internally via g_adapt
    if not is_conductance:
        adaptation_factor = 1.0 / (1.0 + config.sfa_strength * state.sfa_current)
        current = current * adaptation_factor

    return current


def _convert_to_conductances(
    feedforward_current: torch.Tensor,
    recurrent_current: torch.Tensor,
    output_neurons: "LIFNeuron",
    state: NetworkState,
    config: NetworkConfig,
    theta_g: torch.Tensor,
    synaptic_mechanisms: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert currents to conductances for conductance-based neurons.

    This is the core conversion from current-based representation to
    native conductance-based representation.

    Args:
        feedforward_current: Feedforward current (1, n_output)
        recurrent_current: Recurrent current (1, n_output)
        output_neurons: LIF neuron layer (must have E_E, E_L attributes)
        state: Network state
        config: Network configuration
        theta_g: Theta modulation conductance
        synaptic_mechanisms: Optional synaptic mechanisms (for NMDA)

    Returns:
        Tuple of (g_exc, g_inh) conductances
    """
    E_E = output_neurons.E_E
    E_L = output_neurons.E_L
    E_I = output_neurons.E_I

    # Total excitatory current
    total_exc_current = feedforward_current + torch.clamp(recurrent_current, min=0.0)

    # Convert to excitatory conductance
    # I = g * (E - V) → g = I / (E - V)
    # At rest (V ≈ E_L), g_exc = I / (E_E - E_L)
    g_exc = torch.clamp(total_exc_current, min=0.0) / (E_E - E_L)

    # Add tonic excitatory conductance (for homeostatic excitability)
    if state.g_tonic is not None:
        g_exc = g_exc + state.g_tonic

    # Apply NMDA gating if available (voltage-dependent modulation)
    if synaptic_mechanisms is not None and "nmda" in synaptic_mechanisms:
        nmda = synaptic_mechanisms["nmda"]
        membrane_potential = output_neurons.membrane.squeeze() if output_neurons.membrane is not None else torch.zeros(config.n_output, device=config.device)
        
        # If NMDA has conductance method, use it; otherwise fall back to current method
        if hasattr(nmda, 'apply_to_conductance'):
            g_exc = nmda.apply_to_conductance(g_exc, membrane_potential)
        else:
            # Legacy: apply to the pre-conversion current equivalent
            gate = nmda.compute_gate(membrane_potential)
            nmda_fraction = nmda.config.nmda_fraction
            g_exc = g_exc * (1 - nmda_fraction + nmda_fraction * gate)    # Apply dendritic saturation to g_exc if available
    if synaptic_mechanisms is not None and "dendritic" in synaptic_mechanisms:
        dendritic = synaptic_mechanisms["dendritic"]
        if hasattr(dendritic, 'apply_to_conductance'):
            # Pass reversal potentials for proper threshold scaling
            # E_E and E_L may be tensors, so extract scalar values
            E_E_val = float(E_E.item()) if hasattr(E_E, 'item') else float(E_E)
            E_L_val = float(E_L.item()) if hasattr(E_L, 'item') else float(E_L)
            g_exc = dendritic.apply_to_conductance(g_exc, E_E=E_E_val, E_L=E_L_val)
        else:
            # Legacy: apply saturation to conductance directly
            threshold = dendritic.config.saturation_threshold / (E_E - E_L)  # Scale for conductance
            steepness = dendritic.config.saturation_steepness
            saturation_factor = 1.0 / (1.0 + torch.pow(g_exc / threshold, steepness))
            g_exc = g_exc * saturation_factor

    # Compute inhibitory conductance
    # Shunting inhibition + theta modulation + adaptation + tonic GABA
    g_inh = state.shunting_conductance.clone()
    g_inh = g_inh + theta_g  # Theta as inhibitory conductance

    # Add tonic inhibitory conductance (extrasynaptic GABA_A)
    # This provides a constant "brake" that can be upregulated when neurons fire too much
    if state.g_inh_tonic is not None:
        g_inh = g_inh + state.g_inh_tonic

    # Add external adaptation conductance (intrinsic plasticity)
    if state.g_adapt_external is not None:
        g_inh = g_inh + state.g_adapt_external

    return g_exc, g_inh


def _is_dendritic_neuron(output_neurons) -> bool:
    """Check if the neuron layer is a DendriticNeuron."""
    return hasattr(output_neurons, 'branch_weights') and hasattr(output_neurons, 'soma')


def forward_timestep(
    t: int,
    input_spikes: torch.Tensor,
    state: NetworkState,
    config: NetworkConfig,
    weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    output_neurons: "LIFNeuron",
    synaptic_mechanisms: Optional[dict] = None,
) -> torch.Tensor:
    """Forward pass for one timestep of network dynamics.

    This function implements all biological mechanisms for a single timestep:
    - Feedforward + recurrent current computation
    - NMDA voltage-dependent gating (if provided via synaptic_mechanisms)
    - Dendritic saturation (if provided via synaptic_mechanisms)
    - Refractory period (absolute and relative)
    - Spike-frequency adaptation (Ca2+-activated K+ channels)
    - Theta phase modulation
    - Shunting (divisive) inhibition - conductance-based GABAergic
    - Fast (PV+) inhibition: rapid, somatic, output-driven
    - Slow (SOM+) inhibition: surround suppression, dendritic, input-driven
    - Blanket inhibition (global inhibition on any spike)
    - Gamma rhythm reset (periodic inhibition reset)
    - Intrinsic plasticity (dynamic thresholds for stability)

    Supports two neuron modes:
    - Current-based (LIFNeuron): All mechanisms converted to effective current
    - Conductance-based (ConductanceLIF): Native g_exc/g_inh representation

    Args:
        t: Current timestep
        input_spikes: Input spike tensor, shape (1, n_input)
        state: NetworkState object (will be modified in place)
        config: NetworkConfig with all parameters
        weights: Feedforward weight matrix, shape (n_output, n_input)
        recurrent_weights: Recurrent weight matrix, shape (n_output, n_output)
        output_neurons: LIF neuron layer
        synaptic_mechanisms: Optional dict with 'nmda' and/or 'dendritic' mechanisms

    Returns:
        Output spikes tensor, shape (1, n_output)
    """
    # Detect neuron type once at the start
    is_conductance_neuron = hasattr(output_neurons, 'E_E')
    is_dendritic_neuron = _is_dendritic_neuron(output_neurons)

    # =================================================================
    # PHASE CONTEXT AND GAMMA RESET
    # =================================================================
    phase_number, cycle_position, is_gamma_reset = _compute_phase_context(t, config)

    if is_gamma_reset and t > 0:
        state.shunting_conductance = state.shunting_conductance * config.gamma_reset_factor

    # =================================================================
    # FEEDFORWARD AXONAL DELAYS
    # =================================================================
    input_for_ff = _get_delayed_input(input_spikes, state, config)
    state.last_delayed_input = input_for_ff.clone()

    # =================================================================
    # UPDATE INPUT ELIGIBILITY TRACE
    # =================================================================
    if state.input_eligibility is not None:
        decay = torch.exp(torch.tensor(-1.0 / state.input_eligibility_tau))
        state.input_eligibility = state.input_eligibility * decay + input_for_ff.squeeze()
    else:
        state.input_eligibility = input_for_ff.squeeze().clone()

    # =================================================================
    # COMPUTE SYNAPTIC CURRENTS
    # =================================================================
    feedforward_current = torch.mm(input_for_ff, weights.t())

    # Dendritic saturation (limits total input integration)
    if synaptic_mechanisms is not None and "dendritic" in synaptic_mechanisms:
        if is_conductance_neuron:
            # For conductance neurons, saturation is applied in _convert_to_conductances
            pass
        else:
            feedforward_current = synaptic_mechanisms["dendritic"].apply(feedforward_current)

    # Recurrent current with soft saturation
    delayed_spikes = state.spike_history[state.history_index].unsqueeze(0)
    recurrent_current = torch.mm(delayed_spikes, recurrent_weights.t())
    recurrent_saturation_max = 1.0
    recurrent_current = recurrent_saturation_max * torch.tanh(
        recurrent_current / recurrent_saturation_max
    )

    current = feedforward_current + recurrent_current

    # NMDA gating for current-based neurons (conductance neurons handle in _convert_to_conductances)
    if synaptic_mechanisms is not None and "nmda" in synaptic_mechanisms and not is_conductance_neuron:
        membrane_potential = output_neurons.membrane if output_neurons.membrane is not None else torch.zeros(1, config.n_output, device=config.device)
        current = synaptic_mechanisms["nmda"].apply_to_current(current, membrane_potential)

    # Store for diagnostics
    state.last_feedforward_current = feedforward_current
    state.last_recurrent_current = recurrent_current

    # =================================================================
    # TIMING: UPDATE STATE TIMERS
    # =================================================================
    state.time_since_spike = state.time_since_spike + 1.0

    # =================================================================
    # REFRACTORY PERIOD & SPIKE-FREQUENCY ADAPTATION
    # =================================================================
    # Conductance neurons handle refractory/adaptation internally
    if not is_conductance_neuron:
        current = _apply_refractory_and_adaptation(current, state, config, is_conductance_neuron)

    # =================================================================
    # THETA PHASE MODULATION
    # =================================================================
    time_in_cycle = t % config.effective_cycle
    if time_in_cycle < config.cycle_duration:
        theta_g, theta_current = _compute_theta_modulation(t, config, is_conductance_neuron)
    else:
        theta_g = torch.zeros(1, config.n_output, device=config.device)
        theta_current = torch.zeros(config.n_output, device=config.device)

    # =================================================================
    # FORWARD PASS (MODE-DEPENDENT)
    # =================================================================
    if is_dendritic_neuron:
        # =========================================================
        # DENDRITIC NEURON MODE (DendriticNeuron wrapping ConductanceLIF)
        # =========================================================
        # DendriticNeuron handles its own branch weights internally.
        # We pass the raw delayed input and optional inhibitory conductance.
        
        # Compute inhibitory conductance from theta and shunting
        # theta_g: (1, n_output), shunting_conductance: (1, n_output)
        g_inh_dendritic = theta_g + state.shunting_conductance  # Keep batch dim
        
        # Forward pass - DendriticNeuron expects shape (batch, total_inputs)
        # input_for_ff has shape (1, n_input)
        output_spikes, _ = output_neurons(input_for_ff, g_inh=g_inh_dendritic)
        
        # Ensure proper output shape (1, n_output)
        if output_spikes.dim() == 1:
            output_spikes = output_spikes.unsqueeze(0)
        
        # Diagnostics
        state.last_effective_current = torch.zeros(1, config.n_output, device=config.device)
        state.last_theta_modulation = theta_current
        
    elif is_conductance_neuron:
        # =========================================================
        # NATIVE CONDUCTANCE MODE (ConductanceLIF)
        # =========================================================
        g_exc, g_inh = _convert_to_conductances(
            feedforward_current, recurrent_current,
            output_neurons, state, config, theta_g,
            synaptic_mechanisms
        )

        # Forward pass with native conductances
        output_spikes, _ = output_neurons(g_exc, g_inh)

        # Diagnostics: compute effective current from conductances
        E_E = output_neurons.E_E
        E_L = output_neurons.E_L
        E_I = output_neurons.E_I
        membrane_potential = output_neurons.membrane if output_neurons.membrane is not None else torch.zeros(1, config.n_output, device=config.device)
        V = membrane_potential.squeeze()
        I_from_g = g_exc * (E_E - V) + g_inh * (E_I - V) + output_neurons.g_L * (E_L - V)
        state.last_effective_current = I_from_g.unsqueeze(0).clone()
        state.last_theta_modulation = theta_current
    else:
        # =========================================================
        # CURRENT-BASED MODE (LIFNeuron)
        # =========================================================
        # Shunting (divisive) inhibition
        effective_current = current / (1.0 + state.shunting_conductance)

        # SOM+ inhibition
        if config.use_som_inhibition:
            effective_current = effective_current / (1.0 + state.som_inhibition)

        # Add homeostatic excitability and theta modulation
        effective_current = effective_current + state.excitability + theta_current

        # Intrinsic plasticity (dynamic threshold offset)
        threshold_offset = state.dynamic_threshold - config.v_threshold
        effective_current = effective_current - threshold_offset

        # Store for diagnostics
        state.last_effective_current = effective_current.clone()
        state.last_theta_modulation = theta_current

        # Forward pass
        if hasattr(output_neurons, 'forward_current'):
            output_spikes, _ = output_neurons.forward_current(effective_current)
        else:
            output_spikes, _ = output_neurons(effective_current)

    # =================================================================
    # UPDATE SPIKE HISTORY
    # =================================================================
    state.spike_history[state.history_index] = output_spikes.squeeze()
    state.history_index = (state.history_index + 1) % config.recurrent_delay

    # Track spikes
    if state.output_spikes_tensor is not None:
        state.output_spikes_tensor[state.spike_tensor_index] = output_spikes.squeeze()
        state.spike_tensor_index += 1
    elif not state.skip_spike_tracking:
        if output_spikes.sum() > 0:
            state.winners.append(output_spikes.squeeze().argmax().item())
        state.output_spikes_list.append(output_spikes.squeeze().cpu().numpy())

    # =================================================================
    # UPDATE SHUNTING INHIBITION (with interneuron delay)
    # =================================================================
    delayed_for_inh = state.inhibition_buffer[state.inhibition_buffer_index].unsqueeze(0)
    state.inhibition_buffer[state.inhibition_buffer_index] = output_spikes.squeeze()
    state.inhibition_buffer_index = (state.inhibition_buffer_index + 1) % config.interneuron_delay

    if delayed_for_inh.sum() > 0:
        spatial_inh = torch.mm(delayed_for_inh, config.inhibition_kernel)
        state.shunting_conductance = state.shunting_conductance + config.shunting_strength * spatial_inh
        state.shunting_conductance = state.shunting_conductance + config.blanket_inhibition_strength * delayed_for_inh.sum()

    state.shunting_conductance = state.shunting_conductance * config.shunting_decay

    # =================================================================
    # UPDATE SOM+ INHIBITION (slow, input-driven surround suppression)
    # =================================================================
    if config.use_som_inhibition:
        input_activity = feedforward_current.abs()
        state.som_activation = (state.som_activation * config.som_decay +
                                input_activity * config.som_activation_rate)
        som_spatial = torch.mm(state.som_activation, config.inhibition_kernel)
        state.som_inhibition = config.som_strength * som_spatial

    # =================================================================
    # UPDATE ADAPTATION
    # =================================================================
    state.time_since_spike = state.time_since_spike * (1.0 - output_spikes)
    state.sfa_current = state.sfa_current * config.sfa_decay + output_spikes * config.sfa_increment

    # =================================================================
    # INTRINSIC PLASTICITY
    # =================================================================
    if is_conductance_neuron and state.g_adapt_external is not None:
        adapt_update = config.intrinsic_strength_fraction * (output_spikes - config.target_rate)
        state.g_adapt_external = state.g_adapt_external + adapt_update
        state.g_adapt_external = state.g_adapt_external.clamp(min=0.0, max=2.0)
    else:
        threshold_update = config.intrinsic_strength_fraction * state.dynamic_threshold * (output_spikes - config.target_rate)
        state.dynamic_threshold = state.dynamic_threshold + threshold_update
        state.dynamic_threshold = state.dynamic_threshold.clamp(0.5, 2.0)

    return output_spikes


def forward_timestep_with_stp(
    t: int,
    input_spikes: torch.Tensor,
    state: NetworkState,
    config: NetworkConfig,
    weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    output_neurons: "LIFNeuron",
    synaptic_mechanisms: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for one timestep with short-term plasticity (STP).

    This is the main entry point for single-timestep network forward pass.
    It handles:
    1. STP weight modulation (STD/STF) - presynaptic vesicle dynamics
    2. Core neural dynamics (LIF, inhibition, adaptation, etc.)
    3. NMDA voltage-dependent gating - coincidence detection
    4. Dendritic saturation - prevents runaway excitation
    5. STP state update - vesicle depletion/recovery

    This function wraps forward_timestep() and handles all STP bookkeeping,
    so callers don't need to manually manage STP modulate/update calls.

    Args:
        t: Current timestep
        input_spikes: Input spike tensor, shape (1, n_input)
        state: NetworkState object (will be modified in place)
        config: NetworkConfig with all parameters
        weights: Feedforward weight matrix, shape (n_output, n_input)
        recurrent_weights: Recurrent weight matrix, shape (n_output, n_output)
        output_neurons: LIF neuron layer
        synaptic_mechanisms: Optional dict with synaptic plasticity mechanisms:
            - 'stp': ShortTermPlasticity (STD/STF)
            - 'nmda': NMDAGating
            - 'dendritic': DendriticSaturation
            - 'neuromod': Neuromodulation

    Returns:
        Tuple of:
            - output_spikes: Output spike tensor, shape (1, n_output)
            - effective_weights: STP-modulated weights used for this timestep
    """
    # Step 1: Apply STP modulation to get effective synaptic weights
    # STD reduces weights for recently-active synapses (vesicle depletion)
    # STF increases weights for recently-active synapses (calcium buildup)
    effective_weights = weights
    if synaptic_mechanisms is not None and "stp" in synaptic_mechanisms:
        effective_weights = synaptic_mechanisms["stp"].modulate_weights(weights, input_spikes)

    # Step 2: Core forward pass (includes NMDA gating and dendritic saturation)
    output_spikes = forward_timestep(
        t, input_spikes, state, config,
        effective_weights, recurrent_weights, output_neurons,
        synaptic_mechanisms=synaptic_mechanisms
    )

    # Step 3: Update STP state based on presynaptic activity
    # This must happen AFTER forward pass so the modulation used current state
    if synaptic_mechanisms is not None and "stp" in synaptic_mechanisms:
        synaptic_mechanisms["stp"].update(input_spikes, dt=config.dt)

    return output_spikes, effective_weights


def forward_pattern(
    input_pattern: torch.Tensor,
    state: NetworkState,
    config: NetworkConfig,
    weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    output_neurons: "LIFNeuron",
    synaptic_mechanisms: Optional[dict] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Forward pass over an entire input pattern (T timesteps).

    This runs the network forward pass without any learning updates.
    Uses forward_timestep_with_stp() internally so STP modulation is applied
    consistently with training.

    Args:
        input_pattern: (T, n_input) tensor of input spikes
        state: NetworkState to use (will be modified in place)
        config: NetworkConfig with all parameters
        weights: Feedforward weight matrix
        recurrent_weights: Recurrent weight matrix
        output_neurons: LIF neuron layer
        synaptic_mechanisms: Optional dict with synaptic mechanisms (stp, nmda, dendritic, etc.)

    Returns:
        output_spikes: (T, n_output) array of output spikes
        winners: List of winner indices for each spike
    """
    output_neurons.reset_state(batch_size=1)
    state.reset_tracking()

    # Pre-allocate GPU tensor for efficient spike tracking
    T = len(input_pattern)
    state.output_spikes_tensor = torch.zeros(T, config.n_output, device=config.device)
    state.spike_tensor_index = 0

    for t in range(T):
        input_spikes = input_pattern[t].unsqueeze(0)
        # Use forward_timestep_with_stp for consistent STP handling
        output_spikes, _ = forward_timestep_with_stp(
            t, input_spikes, state, config, weights, recurrent_weights,
            output_neurons, synaptic_mechanisms=synaptic_mechanisms
        )
        # Store spikes (forward_timestep_with_stp doesn't update output_spikes_tensor)
        state.output_spikes_tensor[t] = output_spikes.squeeze()

    # Convert GPU tensor to numpy (single transfer at end)
    output_spikes_np = state.output_spikes_tensor.cpu().numpy()

    # Compute winners from the GPU tensor (single transfer)
    spike_times = state.output_spikes_tensor.sum(dim=1) > 0  # Which timesteps had spikes
    winners = []
    for t in range(T):
        if spike_times[t]:
            winners.append(int(state.output_spikes_tensor[t].argmax().item()))

    # Clean up GPU tensor
    state.output_spikes_tensor = None
    state.spike_tensor_index = 0

    return output_spikes_np, winners
