"""
Neuromodulatory systems for supervised and reinforcement learning.

This module provides biologically-inspired components for three-factor
learning rules (pre × post × neuromodulator):
- Eligibility traces: Bridge timing between activity and reward
- Dopamine system: Compute reward/error signals based on performance

Biological basis:
- Dopamine encodes "reward prediction error" (Schultz et al., 1997)
- Burst (>baseline) → LTP, Dip (<baseline) → LTD
- Eligibility traces = calcium/signaling cascades (~50-100ms)
"""

import torch
import numpy as np
from typing import Dict, Optional, Union


class EligibilityTraces:
    """Eligibility traces for three-factor learning.

    Biological basis: Calcium transients and signaling cascades persist
    for ~50-100ms after synaptic activity, creating a window for
    neuromodulatory signals (dopamine) to gate plasticity.

    The trace at synapse (i,j) represents recent pre-post coincidences
    that are eligible for modification if a reward signal arrives.

    Args:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons
        tau_ms: Time constant for trace decay in milliseconds
        max_value: Maximum trace value (for saturation)
        device: Torch device

    Example:
        >>> traces = EligibilityTraces(n_pre=20, n_post=10, tau_ms=50.0)
        >>> traces.update(pre_spikes, post_spikes, dt=0.1)
        >>> eligibility = traces.get()  # Shape: (n_post, n_pre)
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_ms: float = 50.0,
        max_value: float = 1.0,
        device: Optional[Union[torch.device, str]] = None
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.tau_ms = tau_ms
        self.max_value = max_value

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.traces = torch.zeros(n_post, n_pre, device=self.device)

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, dt: float) -> None:
        """Update traces based on current activity.

        Args:
            pre_spikes: Presynaptic spikes, shape (1, n_pre) or (n_pre,)
            post_spikes: Postsynaptic spikes, shape (1, n_post) or (n_post,)
            dt: Timestep in milliseconds
        """
        # Exponential decay
        decay = np.exp(-dt / self.tau_ms)
        self.traces = self.traces * decay

        # Add new coincidences
        if pre_spikes.sum() > 0 and post_spikes.sum() > 0:
            # Outer product: traces[post, pre] += post_spike * pre_spike
            post_vec = post_spikes.squeeze()
            pre_vec = pre_spikes.squeeze()

            if post_vec.dim() == 0:
                post_vec = post_vec.unsqueeze(0)
            if pre_vec.dim() == 0:
                pre_vec = pre_vec.unsqueeze(0)

            coincidence = post_vec.unsqueeze(1) * pre_vec.unsqueeze(0)
            self.traces = self.traces + coincidence

        # Saturate at max value
        self.traces = self.traces.clamp(0, self.max_value)

    def update_subthreshold(
        self,
        pre_spikes: torch.Tensor,
        membrane_potential: torch.Tensor,
        v_threshold: float,
        dt: float,
        subthreshold_scale: float = 0.3,
    ) -> None:
        """Update traces based on membrane potential (subthreshold eligibility).
        
        This enables credit assignment even when neurons don't spike.
        Neurons driven close to threshold accumulate partial eligibility,
        proportional to how close they are to firing.
        
        Biological basis: Dendritic calcium signals can occur even without
        somatic spikes, providing a substrate for subthreshold plasticity.
        Back-propagating action potentials aren't the only source of
        postsynaptic calcium.
        
        Args:
            pre_spikes: Presynaptic spikes, shape (1, n_pre) or (n_pre,)
            membrane_potential: Postsynaptic membrane potentials, shape (1, n_post) or (n_post,)
            v_threshold: Firing threshold
            dt: Timestep in milliseconds
            subthreshold_scale: Scaling factor for subthreshold contribution (0-1)
        """
        # Exponential decay
        decay = np.exp(-dt / self.tau_ms)
        self.traces = self.traces * decay
        
        # Only update if there's presynaptic input
        if pre_spikes.sum() > 0:
            pre_vec = pre_spikes.squeeze()
            if pre_vec.dim() == 0:
                pre_vec = pre_vec.unsqueeze(0)
            
            # Compute subthreshold activation: how close to threshold?
            # sigmoid centered at threshold, scaled to [0, 1]
            v = membrane_potential.squeeze()
            if v.dim() == 0:
                v = v.unsqueeze(0)
            
            # Smooth activation: high when v approaches threshold
            # Uses sigmoid: σ((v - threshold + margin) / temperature)
            margin = 0.3 * v_threshold  # start building eligibility at 70% of threshold
            temperature = 0.1 * v_threshold  # sharpness of transition
            activation = torch.sigmoid((v - v_threshold + margin) / temperature)
            
            # Scale by subthreshold_scale (lower than spike-based eligibility)
            activation = activation * subthreshold_scale
            
            # Build eligibility: traces[post, pre] += activation * pre_spike
            eligibility_update = activation.unsqueeze(1) * pre_vec.unsqueeze(0)
            self.traces = self.traces + eligibility_update
        
        # Saturate at max value
        self.traces = self.traces.clamp(0, self.max_value)

    def reset(self) -> None:
        """Reset all traces to zero."""
        self.traces.zero_()

    def get(self) -> torch.Tensor:
        """Get current trace values.

        Returns:
            Tensor of shape (n_post, n_pre) with current eligibility values
        """
        return self.traces


class DopamineSystem:
    """Computes dopamine signal based on response correctness.

    Implements the teaching signal for supervised learning:
    - BURST (positive): Correct response → reinforce active synapses
    - DIP (negative): Wrong response → weaken active synapses
    - BASELINE (zero): No response or during gaps

    The dopamine signal is computed based on whether the network's
    response matches the target mapping for the current input phase.

    Args:
        target_mapping: Mapping defining correct input→output relationships
        burst_magnitude: Dopamine level for correct responses
        dip_magnitude: Dopamine level for incorrect responses (typically negative)
        tau_ms: Time constant for dopamine decay
        device: Torch device

    Example:
        >>> from thalia.learning import TargetMapping
        >>> mapping = TargetMapping(name="default", input_to_output={0: 0, 1: 0, 2: 1, 3: 1})
        >>> dopamine = DopamineSystem(mapping, burst_magnitude=1.0, dip_magnitude=-0.5)
        >>> da_level = dopamine.compute(output_spikes, current_phase=0, dt=0.1)
    """

    def __init__(
        self,
        target_mapping: "TargetMapping",
        burst_magnitude: float = 1.0,
        dip_magnitude: float = -0.5,
        tau_ms: float = 20.0,
        device: Optional[Union[torch.device, str]] = None
    ):
        self.target_mapping = target_mapping
        self.burst_magnitude = burst_magnitude
        self.dip_magnitude = dip_magnitude
        self.tau_ms = tau_ms

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.current_level: float = 0.0

    def compute(
        self,
        output_spikes: torch.Tensor,
        current_phase: int,
        dt: float,
        in_gap: bool = False
    ) -> float:
        """Compute dopamine signal for current timestep.

        Args:
            output_spikes: Output spike tensor, shape (1, n_output)
            current_phase: Current input phase (0 to n_phases-1)
            dt: Timestep in milliseconds
            in_gap: Whether we're in a gap between sequences

        Returns:
            Current dopamine level (positive for correct, negative for incorrect)
        """
        # Decay existing dopamine
        decay = np.exp(-dt / self.tau_ms)
        self.current_level = self.current_level * decay

        # No updates during gaps or invalid phases
        if in_gap or current_phase < 0:
            return self.current_level

        # Get target output for this phase
        target_output = self.target_mapping.get_target_for_phase(current_phase)

        # Check if response matches target
        if output_spikes.sum() > 0:
            winner = output_spikes.squeeze().argmax().item()
            if winner == target_output:
                self.current_level = self.burst_magnitude  # Reward!
            else:
                self.current_level = self.dip_magnitude  # Punishment

        return self.current_level

    def reset(self) -> None:
        """Reset dopamine level to baseline (zero)."""
        self.current_level = 0.0

    def get_level(self) -> float:
        """Get current dopamine level."""
        return self.current_level


class TargetMapping:
    """Defines the desired input→output mapping to learn.

    This class encapsulates the "ground truth" that the network should
    learn during supervised training. It maps input neuron indices to
    the output neuron that should respond.

    Args:
        name: Descriptive name for this mapping
        input_to_output: Dictionary mapping input indices to output indices

    Example:
        >>> # Default 2:1 mapping (inputs 0,1→0, inputs 2,3→1, etc.)
        >>> mapping = TargetMapping(
        ...     name="default",
        ...     input_to_output={0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
        ... )
        >>> mapping.get_target_output(3)  # Returns 1
        >>> mapping.get_target_for_phase(1)  # Returns 1 (phase 1 = inputs 2,3)
    """

    def __init__(self, name: str, input_to_output: Dict[int, int]):
        self.name = name
        self.input_to_output = input_to_output

    def get_target_output(self, input_idx: int) -> int:
        """Return the correct output neuron for a given input.

        Args:
            input_idx: Index of the input neuron

        Returns:
            Index of the output neuron that should respond, or -1 if unknown
        """
        return self.input_to_output.get(input_idx, -1)

    def get_target_for_phase(self, phase: int, n_inputs_per_phase: int = 2) -> int:
        """Return the correct output for a given temporal phase.

        Args:
            phase: Phase index (0, 1, 2, ...)
            n_inputs_per_phase: How many inputs fire per phase (default: 2)

        Returns:
            Index of the output neuron that should respond during this phase
        """
        first_input = phase * n_inputs_per_phase
        return self.get_target_output(first_input)

    def __repr__(self) -> str:
        return f"TargetMapping(name='{self.name}', n_mappings={len(self.input_to_output)})"


def create_default_mapping(n_input: int, n_output: int) -> TargetMapping:
    """Create default diagonal mapping: input 2i,2i+1 → output i.

    Args:
        n_input: Number of input neurons
        n_output: Number of output neurons

    Returns:
        TargetMapping with default 2:1 mapping
    """
    mapping = {}
    for i in range(n_input):
        output_idx = min(i // 2, n_output - 1)
        mapping[i] = output_idx
    return TargetMapping(name="default", input_to_output=mapping)


def create_shuffled_mapping(n_input: int, n_output: int, seed: int = 42) -> TargetMapping:
    """Create random shuffled mapping - arbitrary permutation.

    Args:
        n_input: Number of input neurons
        n_output: Number of output neurons
        seed: Random seed for reproducibility

    Returns:
        TargetMapping with randomly shuffled output assignments
    """
    rng = np.random.RandomState(seed)
    shuffled_outputs = rng.permutation(n_output)

    mapping = {}
    for i in range(n_input):
        phase = i // 2
        output_idx = int(shuffled_outputs[phase % n_output])
        mapping[i] = output_idx

    return TargetMapping(name="shuffled", input_to_output=mapping)


def create_reversed_mapping(n_input: int, n_output: int) -> TargetMapping:
    """Create reversed mapping: input 2i,2i+1 → output (n-1-i).

    Args:
        n_input: Number of input neurons
        n_output: Number of output neurons

    Returns:
        TargetMapping with reversed output assignments
    """
    mapping = {}
    for i in range(n_input):
        phase = i // 2
        output_idx = n_output - 1 - (phase % n_output)
        mapping[i] = output_idx

    return TargetMapping(name="reversed", input_to_output=mapping)


def apply_dopamine_modulated_update(
    weights: torch.Tensor,
    eligibility: torch.Tensor,
    dopamine: float,
    learning_rate: float,
    w_min: float,
    w_max: float,
    soft_bounds: bool = True,
    stp_resources: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply three-factor learning rule: Δw = lr × eligibility × dopamine.

    This implements the core supervised learning update where weight changes
    are gated by both eligibility (recent pre-post activity) and dopamine
    (reward/error signal).

    Optional STP gating (biological basis):
    - Short-term plasticity (STP) models vesicle depletion at synapses
    - When vesicles are depleted (low resources), the synapse transmitted less
    - Therefore, learning at that synapse should also be reduced
    - This prevents early-firing inputs from dominating learning
    - Key for learning arbitrary temporal patterns, not just first-phase

    Args:
        weights: Current weight matrix, shape (n_post, n_pre)
        eligibility: Eligibility trace matrix, shape (n_post, n_pre)
        dopamine: Current dopamine level (positive=reward, negative=punishment)
        learning_rate: Base learning rate
        w_min: Minimum weight value
        w_max: Maximum weight value
        soft_bounds: If True, use soft bounds (harder to change near limits)
        stp_resources: Optional STP resource levels, shape (n_post, n_pre)
            Values 0-1, where 1 = full resources, 0 = depleted.
            When provided, learning rate is scaled per-synapse by resources.
            This implements the biological constraint that heavily-used
            synapses (depleted vesicles) have reduced plasticity.

    Returns:
        Updated weight matrix
    """
    if abs(dopamine) < 1e-6:
        return weights

    # Base update: three-factor rule
    dw = learning_rate * eligibility * dopamine

    # STP gating: scale learning by vesicle resources
    # This prevents early-phase inputs from dominating learning
    if stp_resources is not None:
        dw = dw * stp_resources

    if soft_bounds:
        if dopamine > 0:
            # Potentiation: harder to increase near ceiling
            headroom = (w_max - weights) / w_max
            dw = dw * headroom.clamp(0, 1)
        else:
            # Depression: harder to decrease near floor
            footroom = (weights - w_min) / w_max
            dw = dw * footroom.clamp(0, 1)

    weights = weights + dw
    weights = weights.clamp(w_min, w_max)

    return weights
