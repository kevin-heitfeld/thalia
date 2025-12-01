"""Hebbian learning rules for unsupervised synaptic plasticity.

This module provides biologically-grounded learning rules that update
synaptic weights based on pre- and post-synaptic activity patterns.

Key learning rules:
- Pure Hebbian coincidence detection (feedforward)
- Predictive coding (recurrent sequence learning)
- Slow synaptic scaling (weight normalization)
"""

import torch
from typing import Tuple


def hebbian_update(
    weights: torch.Tensor,
    input_spikes: torch.Tensor,
    output_spikes: torch.Tensor,
    learning_rate: float,
    w_max: float,
    heterosynaptic_ratio: float,
    stp_resources: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply competitive Hebbian learning with heterosynaptic LTD.

    This implements pure coincidence detection WITH competition:
    - LTP: strengthen connections where input AND output are both active
    - Heterosynaptic LTD: weaken connections where output fires but input is INACTIVE

    CRITICAL DISTINCTION from temporal STDP:
    - Temporal STDP LTD: post-before-pre timing → depression
      → Causes systematic drift toward later inputs in sequences
    - Heterosynaptic LTD: post fires, pre is INACTIVE → depression
      → Creates competition without temporal asymmetry

    Biological basis:
    - NMDA-mediated coincidence detection for LTP (simultaneous pre + post)
    - Heterosynaptic LTD from calcium spread and limited receptor resources
    - When synapse A is potentiated, nearby inactive synapses get weakened
    - This is RESOURCE COMPETITION, not timing-dependent

    Uses soft-bounded learning to prevent runaway potentiation:
    - LTP: Δw ∝ (w_max - w) for potentiation (harder near ceiling)
    - LTD: Δw ∝ w for depression (harder near floor)

    Optional STP gating:
    - If stp_resources is provided, learning rate is modulated per-synapse
    - Low vesicle resources → reduced learning (synapse was heavily used)
    - Prevents early-firing inputs from dominating learning

    Args:
        weights: Feedforward weight matrix, shape (n_output, n_input)
        input_spikes: Input spike tensor, shape (batch, n_input) or (n_input,)
        output_spikes: Output spike tensor, shape (batch, n_output) or (n_output,)
        learning_rate: Base learning rate for weight updates
        w_max: Maximum weight value (biologically-derived)
        heterosynaptic_ratio: LTD rate as fraction of LTP rate
            Higher values = stronger competition = faster unlearning
        stp_resources: Optional STP resource levels, shape (n_output, n_input)
            Values 0-1, where 1 = full resources, 0 = depleted.
            When provided, learning rate is scaled per-synapse by resources.

    Returns:
        Updated weight matrix
    """
    # Early exit if no post-synaptic activity
    if output_spikes.sum() == 0:
        return weights

    # Early exit if no pre-synaptic activity
    if input_spikes.sum() == 0:
        return weights

    # Ensure proper shapes for outer product
    if input_spikes.dim() == 1:
        input_spikes = input_spikes.unsqueeze(0)
    if output_spikes.dim() == 1:
        output_spikes = output_spikes.unsqueeze(0)

    # Identify which neurons fired (for heterosynaptic LTD)
    # With eligibility traces, input_spikes contains values 0-1 representing
    # recent input activity. We use a threshold to distinguish "recently active"
    # from "inactive" for LTD purposes.
    # Use squeeze(0) to remove batch dim but keep at least 1D
    post_active = output_spikes.squeeze(0)  # (n_output,)
    pre_trace = input_spikes.squeeze(0)     # (n_input,) - eligibility trace values
    
    # Handle edge case: if still 0-dim (single element), make 1D
    if post_active.dim() == 0:
        post_active = post_active.unsqueeze(0)
    if pre_trace.dim() == 0:
        pre_trace = pre_trace.unsqueeze(0)
    
    # For LTP: use the full trace value (graded learning based on recency)
    # For LTD: only depress synapses with very low recent activity
    activity_threshold = 0.1  # Below this = "inactive" for heterosynaptic LTD
    pre_inactive_mask = (pre_trace < activity_threshold).float()
    n_inactive = pre_inactive_mask.sum().item()

    # === Compute per-synapse learning rate ===
    # If STP resources provided, gate learning by vesicle availability
    # Low resources = synapse was heavily used = reduce learning
    if stp_resources is not None:
        lr_matrix = learning_rate * stp_resources  # (n_output, n_input)
    else:
        lr_matrix = learning_rate  # Scalar broadcast

    # === LTP: Strengthen active input→active output connections ===
    # Hebbian outer product: (n_output, 1) @ (1, n_input) → (n_output, n_input)
    hebbian_dw = post_active.unsqueeze(1) @ pre_trace.unsqueeze(0)

    # Soft-bounded LTP (biological saturation)
    headroom = (w_max - weights) / w_max
    headroom = headroom.clamp(0.0, 1.0)

    ltp_update = lr_matrix * hebbian_dw * headroom

    # === Heterosynaptic LTD: Weaken inactive input→active output connections ===
    # When post fires, synapses from INACTIVE inputs get weakened
    # This creates competition: "you didn't help me fire, so you get weaker"
    # 
    # NORMALIZATION: The total LTD should be proportional to total LTP,
    # not to the number of inactive synapses. Otherwise with 19 inactive
    # inputs vs 1 active, LTD would dominate 19x.
    #
    # We divide by n_inactive to distribute the competition pressure evenly.
    # heterosynaptic_ratio=0.5 means: total LTD = 0.5 × total LTP
    if n_inactive > 0:
        ltd_mask = post_active.unsqueeze(1) @ pre_inactive_mask.unsqueeze(0)  # (n_output, n_input)
        
        # Soft-bounded LTD (can't go below 0)
        # Depression is proportional to current weight (harder to depress weak synapses)
        # Normalize by n_inactive so total LTD is independent of input count
        # Use lr_matrix for STP gating of LTD as well
        if stp_resources is not None:
            per_synapse_ltd = heterosynaptic_ratio * lr_matrix / n_inactive
        else:
            per_synapse_ltd = heterosynaptic_ratio * learning_rate / n_inactive
        ltd_update = per_synapse_ltd * ltd_mask * weights
    else:
        ltd_update = 0.0

    # Apply updates
    weights = weights + ltp_update - ltd_update
    weights = weights.clamp(0.0, w_max)

    return weights


def synaptic_scaling(
    weights: torch.Tensor,
    target_norm_fraction: float,
    tau: float,
    w_max: float,
) -> torch.Tensor:
    """Apply slow synaptic scaling for weight normalization.

    Real synaptic scaling happens over hours/days. We simulate this
    by making small adjustments each cycle, moving slowly toward target.

    Args:
        weights: Weight matrix, shape (n_output, n_input)
        target_norm_fraction: Target L2 norm as fraction of max possible
        tau: Time constant (cycles to reach target)
        w_max: Maximum weight value

    Returns:
        Scaled weight matrix
    """
    import numpy as np

    n_input = weights.shape[1]

    # Compute current norms
    weight_norms = torch.norm(weights, dim=1, keepdim=True)

    # Target L2 norm: if all weights were at w_max, norm = sqrt(n_input) * w_max
    target_norm = np.sqrt(n_input) * w_max * target_norm_fraction

    # Move 1/tau of the way toward target
    scale_factor = 1.0 + (target_norm / (weight_norms + 1e-8) - 1.0) / tau

    weights = weights * scale_factor
    weights = weights.clamp(0.0, w_max)

    return weights


class PredictiveCoding:
    """Gamma-locked predictive coding for recurrent sequence learning.

    This implements phase-locked predictive coding where learning is
    gated by gamma oscillation phase. Only the integrated evidence over
    each gamma cycle determines the winner and drives learning.

    Biological basis:
    - In hippocampus, LTP/LTD is gated by gamma phase
    - Spikes during gamma troughs → LTP, peaks → LTD
    - Creates precise temporal windows for sequence learning
    """

    def __init__(
        self,
        n_output: int,
        gamma_period: int = 100,
        learning_phase: int = 90,
        start_cycle: int = 80,
        base_lr: float = 0.05,
        confidence_threshold: float = 0.3,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize predictive coding state.

        Args:
            n_output: Number of output neurons
            gamma_period: Gamma cycle period in timesteps
            learning_phase: Phase within gamma cycle when learning occurs
            start_cycle: Cycle number after which to start learning
            base_lr: Base learning rate
            confidence_threshold: Minimum confidence to apply learning
            device: Torch device
        """
        self.n_output = n_output
        self.gamma_period = gamma_period
        self.learning_phase = learning_phase
        self.start_cycle = start_cycle
        self.base_lr = base_lr
        self.confidence_threshold = confidence_threshold
        self.device = device

        # State
        self.gamma_spike_counts = torch.zeros(n_output, device=device)
        self.last_gamma_winner = -1

        # Tracking (optional)
        self.winner_matches = 0
        self.total_transitions = 0

    def reset(self):
        """Reset state for new sequence/cycle."""
        self.gamma_spike_counts.zero_()
        self.last_gamma_winner = -1

    def accumulate_spikes(self, output_spikes: torch.Tensor):
        """Accumulate spikes during gamma cycle.

        Args:
            output_spikes: Output spikes, shape (batch, n_output) or (n_output,)
        """
        if output_spikes.dim() == 2:
            output_spikes = output_spikes.squeeze(0)
        self.gamma_spike_counts += output_spikes.float()

    def update_recurrent(
        self,
        t: int,
        recurrent_weights: torch.Tensor,
        current_cycle: int,
        w_min: float = 0.0,
        w_max: float = 0.5,
    ) -> Tuple[torch.Tensor, bool]:
        """Apply predictive coding update at gamma learning phase.

        Args:
            t: Current timestep
            recurrent_weights: Recurrent weight matrix (n_output, n_output)
            current_cycle: Current training cycle number
            w_min: Minimum recurrent weight
            w_max: Maximum recurrent weight

        Returns:
            Tuple of (updated weights, whether learning occurred)
        """
        gamma_phase = t % self.gamma_period

        if gamma_phase != self.learning_phase:
            return recurrent_weights, False

        # At learning phase - determine winner
        if self.gamma_spike_counts.sum() == 0:
            self.last_gamma_winner = -1
            self.gamma_spike_counts.zero_()
            return recurrent_weights, False

        current_winner = int(self.gamma_spike_counts.argmax().item())
        winner_count = self.gamma_spike_counts[current_winner].item()

        # Compute confidence from runner-up
        sorted_counts, _ = self.gamma_spike_counts.sort(descending=True)
        runner_up_count = sorted_counts[1].item() if len(sorted_counts) > 1 else 0
        confidence = (winner_count - runner_up_count) / max(winner_count, 1)

        learning_occurred = False

        # Learn if: confident, have previous winner, past start cycle
        if (confidence > self.confidence_threshold and
            self.last_gamma_winner >= 0 and
            current_cycle >= self.start_cycle):

            prev_winner = self.last_gamma_winner

            # Get current prediction
            prediction_weights = recurrent_weights[prev_winner, :]
            predicted_winner = int(prediction_weights.argmax().item())

            if predicted_winner == current_winner:
                # Correct prediction - reinforce
                recurrent_weights[prev_winner, current_winner] += self.base_lr * 1.0
            else:
                # Wrong prediction - correct it
                recurrent_weights[prev_winner, predicted_winner] -= self.base_lr * 0.8
                recurrent_weights[prev_winner, current_winner] += self.base_lr * 1.0

                # Soft decay on others
                for other in range(self.n_output):
                    if other not in [current_winner, predicted_winner, prev_winner]:
                        recurrent_weights[prev_winner, other] *= 0.995

            # Clamp and remove self-connections
            recurrent_weights = recurrent_weights.clamp(w_min, w_max)
            eye = torch.eye(self.n_output, device=self.device)
            recurrent_weights = recurrent_weights * (1 - eye)

            learning_occurred = True

        # Update state for next gamma cycle
        self.last_gamma_winner = current_winner
        self.gamma_spike_counts.zero_()

        return recurrent_weights, learning_occurred
