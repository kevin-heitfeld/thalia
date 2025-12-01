"""Per-phase homeostatic plasticity and winner tracking.

This module provides mechanisms for tracking phase-wise winners and applying
homeostatic pressure to ensure diverse neural representations across phases.

Biological basis:
- Spike-frequency adaptation causes neurons to become less responsive after
  repeated activation within a short time window
- Short-term synaptic depression reduces synaptic efficacy after repeated use
- Metabolic constraints favor sparse, distributed representations

Key classes:
- PhaseHomeostasis: Applies homeostatic pressure based on win distribution
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class PhaseHomeostasis:
    """Per-phase homeostatic pressure to ensure diverse representations.

    Tracks how many phases each neuron wins and applies pressure to prevent
    any single neuron from dominating. This is biologically plausible as it
    mimics the effects of metabolic constraints on redundant representations.

    Key insight: We don't specify HOW MANY phases each neuron should win.
    We just prevent any neuron from winning WAY more than average.

    Attributes:
        n_output: Number of output neurons
        device: Torch device
        tau: Time constant for averaging (cycles)
        strength: Strength of homeostatic pressure
        phase_win_counts: Wins per neuron THIS cycle
        avg_phase_wins: Running average of wins per neuron
    """
    n_output: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    tau: float = 5.0
    strength: float = 0.5
    phase_win_counts: Optional[torch.Tensor] = field(default=None)
    avg_phase_wins: Optional[torch.Tensor] = field(default=None)

    def __post_init__(self):
        """Initialize tensors after dataclass initialization."""
        if self.phase_win_counts is None:
            self.phase_win_counts = torch.zeros(self.n_output, device=self.device)
        if self.avg_phase_wins is None:
            self.avg_phase_wins = torch.zeros(self.n_output, device=self.device)

    def record_win(self, winner_idx: int):
        """Record a win for the given neuron.

        Args:
            winner_idx: Which neuron won (0 to n_output-1)
        """
        if 0 <= winner_idx < self.n_output:
            self.phase_win_counts[winner_idx] += 1

    def update_cycle(self) -> torch.Tensor:
        """Update running averages at end of cycle.

        Should be called once per cycle (not per timestep).

        Returns:
            Updated avg_phase_wins tensor
        """
        self.avg_phase_wins = self.avg_phase_wins + (
            self.phase_win_counts - self.avg_phase_wins
        ) / self.tau
        return self.avg_phase_wins

    def get_suppression(self) -> torch.Tensor:
        """Get suppression factor for each neuron based on over-winning.

        Neurons that win more than average get suppressed.

        Returns:
            Suppression tensor (positive = suppress, negative = boost)
        """
        mean_wins = self.avg_phase_wins.mean()
        excess_wins = self.avg_phase_wins - mean_wins
        return self.strength * excess_wins

    def reset_cycle(self):
        """Reset per-cycle counters (call at start of each cycle)."""
        self.phase_win_counts.zero_()

    def reset_all(self):
        """Reset all state including running averages."""
        self.phase_win_counts.zero_()
        self.avg_phase_wins.zero_()

    def get_stats(self) -> dict:
        """Get current statistics for logging/debugging.

        Returns:
            Dictionary with wins, mean, max, min
        """
        wins = self.avg_phase_wins.cpu().numpy()
        return {
            "wins": wins.tolist(),
            "mean": float(wins.mean()),
            "max": float(wins.max()),
            "min": float(wins.min()),
        }


def update_bcm_threshold(
    bcm_threshold: torch.Tensor,
    avg_activity_hz: float,
    target_rate_hz: float,
    tau: float,
    min_threshold: float,
    max_threshold: float,
) -> torch.Tensor:
    """Update BCM (Bienenstock-Cooper-Munro) threshold.

    BCM theory: The threshold for LTP vs LTD slides based on recent
    postsynaptic activity. High activity → higher threshold (harder LTP).

    Args:
        bcm_threshold: Current BCM threshold tensor
        avg_activity_hz: Average activity in Hz (spikes/second)
        target_rate_hz: Target firing rate in Hz
        tau: Time constant (slow adaptation)
        min_threshold: Minimum threshold value
        max_threshold: Maximum threshold value

    Returns:
        Updated BCM threshold tensor
    """
    # BCM target is based on squared firing rate
    bcm_target = (target_rate_hz ** 2) / 1000.0

    # Update threshold based on deviation from target
    bcm_update = (avg_activity_hz ** 2 / 1000.0 - bcm_target) / tau
    bcm_threshold = bcm_threshold + bcm_update

    # Clamp to valid range
    bcm_threshold = bcm_threshold.clamp(min_threshold, max_threshold)

    return bcm_threshold


def update_homeostatic_excitability(
    current_rate: torch.Tensor,
    avg_firing_rate: torch.Tensor,
    excitability: torch.Tensor,
    target_rate: float,
    tau: float,
    strength: float,
    v_threshold: float,
    bounds: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update firing rate average and excitability using homeostatic plasticity.

    This implements scale-invariant homeostatic regulation where neurons
    adjust their excitability based on deviation from target firing rate.

    Biological basis:
    - Neurons that fire too much become less excitable (raise threshold)
    - Neurons that fire too little become more excitable (lower threshold)
    - This is a LOCAL and UNSUPERVISED mechanism

    The adjustment is scale-invariant: expressed as a fraction of threshold,
    so the same parameters work regardless of absolute voltage scale.

    For intuitive tuning, we recommend passing rates in Hz:
    - current_rate: current firing rate in Hz (e.g., spikes * 1000 / cycle_ms)
    - target_rate: target rate in Hz (e.g., 20.0 for 20 Hz)
    - strength: threshold shift per Hz of error (e.g., 0.01)

    Args:
        current_rate: Current firing rate tensor (recommend Hz for intuitive tuning)
        avg_firing_rate: Running average of firing rate (same units as current_rate)
        excitability: Current excitability adjustment tensor
        target_rate: Target firing rate (same units as current_rate)
        tau: Time constant for averaging (in update calls, e.g., cycles)
        strength: Adjustment strength per unit of rate error
        v_threshold: Voltage threshold for scale-invariance
        bounds: (min, max) bounds for excitability

    Returns:
        Tuple of (updated_avg_firing_rate, updated_excitability)

    Example (using Hz):
        >>> # If current rate is 50 Hz and target is 20 Hz:
        >>> # rate_error = 20 - 50 = -30 Hz
        >>> # adjustment = 0.01 * 1.0 * (-30) = -0.30 (decrease excitability)
        >>> avg_rate, excitability = update_homeostatic_excitability(
        ...     current_rate=current_spikes * 1000 / cycle_duration_ms,  # Hz
        ...     avg_firing_rate=state.avg_firing_rate,
        ...     excitability=state.excitability,
        ...     target_rate=20.0,  # 20 Hz target
        ...     tau=2.0,  # 2 cycle averaging
        ...     strength=0.01,  # 0.01 threshold shift per Hz error
        ...     v_threshold=1.0,
        ... )
    """
    # Update exponential moving average of firing rate
    new_avg_rate = avg_firing_rate + (current_rate - avg_firing_rate) / tau

    # Compute rate error (positive if firing too little)
    rate_error = target_rate - new_avg_rate

    # Scale adjustment by threshold for scale-invariance
    adjustment = strength * v_threshold * rate_error

    # Update and clamp excitability
    new_excitability = (excitability + adjustment).clamp(bounds[0], bounds[1])

    return new_avg_rate, new_excitability


def update_homeostatic_excitability_step(
    output_spikes: torch.Tensor,
    avg_firing_rate: torch.Tensor,
    excitability: torch.Tensor,
    target_rate: float,
    homeostatic_tau: float,
    homeostatic_strength: float,
    v_threshold: float,
    bounds: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update homeostatic excitability based on current spike activity (per-timestep).

    This is a simpler per-timestep version of update_homeostatic_excitability().
    Use this when you want to update homeostasis every timestep rather than
    at cycle boundaries.

    Args:
        output_spikes: Output spikes tensor, shape (1, n_output) or (n_output,)
        avg_firing_rate: Running average of firing rate (modified in place conceptually)
        excitability: Current excitability adjustment tensor
        target_rate: Target firing rate (spikes per timestep)
        homeostatic_tau: Time constant for averaging
        homeostatic_strength: Adjustment strength as fraction of threshold
        v_threshold: Voltage threshold for scale-invariance
        bounds: (min, max) bounds for excitability

    Returns:
        Tuple of (updated_avg_firing_rate, updated_excitability)
    """
    new_avg_rate = avg_firing_rate * (1 - 1/homeostatic_tau) + output_spikes / homeostatic_tau
    rate_error = target_rate - new_avg_rate
    new_excitability = excitability + homeostatic_strength * v_threshold * rate_error
    new_excitability = new_excitability.clamp(bounds[0], bounds[1])
    return new_avg_rate, new_excitability


def update_homeostatic_conductance(
    current_rate: torch.Tensor,
    avg_firing_rate: torch.Tensor,
    g_tonic: torch.Tensor,
    target_rate: float,
    tau: float,
    strength: float,
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update firing rate average and tonic excitatory conductance using homeostatic plasticity.

    This is the conductance-based version of update_homeostatic_excitability(),
    designed for use with ConductanceLIF neurons.

    Instead of adjusting an additive excitability current, this adjusts a tonic
    excitatory conductance (g_tonic) that provides persistent depolarizing drive.

    Biological basis:
    - Neurons that fire too little upregulate tonic excitatory conductance
      (like persistent Na+ current, I_h, or reduced leak conductance)
    - Neurons that fire too much downregulate tonic conductance
    - This is a LOCAL and UNSUPERVISED mechanism

    The conductance approach has advantages:
    - Bounded by reversal potentials (can't cause runaway)
    - Interacts properly with other conductances (shunting effects)
    - More biologically realistic

    Args:
        current_rate: Current firing rate tensor (recommend Hz for intuitive tuning)
        avg_firing_rate: Running average of firing rate (same units as current_rate)
        g_tonic: Current tonic excitatory conductance tensor
        target_rate: Target firing rate (same units as current_rate)
        tau: Time constant for averaging (in update calls, e.g., cycles)
        strength: Conductance adjustment per unit of rate error
        bounds: (min, max) bounds for g_tonic (default: 0.0 to 1.0)

    Returns:
        Tuple of (updated_avg_firing_rate, updated_g_tonic)

    Example:
        >>> # If current rate is 10 Hz and target is 20 Hz:
        >>> # rate_error = 20 - 10 = 10 Hz (firing too little)
        >>> # adjustment = 0.001 * 10 = 0.01 (increase g_tonic)
        >>> avg_rate, g_tonic = update_homeostatic_conductance(
        ...     current_rate=current_spikes * 1000 / cycle_duration_ms,  # Hz
        ...     avg_firing_rate=state.avg_firing_rate,
        ...     g_tonic=state.g_tonic,
        ...     target_rate=20.0,  # 20 Hz target
        ...     tau=2.0,  # 2 cycle averaging
        ...     strength=0.001,  # conductance change per Hz error
        ... )
    """
    # Update exponential moving average of firing rate
    new_avg_rate = avg_firing_rate + (current_rate - avg_firing_rate) / tau

    # Compute rate error (positive if firing too little → need more excitation)
    rate_error = target_rate - new_avg_rate

    # Adjust tonic conductance (positive error → increase g_tonic)
    adjustment = strength * rate_error

    # Update and clamp g_tonic (must be non-negative)
    new_g_tonic = (g_tonic + adjustment).clamp(bounds[0], bounds[1])

    return new_avg_rate, new_g_tonic


def update_homeostatic_conductance_step(
    output_spikes: torch.Tensor,
    avg_firing_rate: torch.Tensor,
    g_tonic: torch.Tensor,
    target_rate: float,
    homeostatic_tau: float,
    homeostatic_strength: float,
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update homeostatic tonic conductance per-timestep.

    This is a simpler per-timestep version of update_homeostatic_conductance().
    Use this when you want to update homeostasis every timestep rather than
    at cycle boundaries.

    Args:
        output_spikes: Output spikes tensor, shape (1, n_output) or (n_output,)
        avg_firing_rate: Running average of firing rate
        g_tonic: Current tonic excitatory conductance tensor
        target_rate: Target firing rate (spikes per timestep)
        homeostatic_tau: Time constant for averaging
        homeostatic_strength: Conductance adjustment strength
        bounds: (min, max) bounds for g_tonic

    Returns:
        Tuple of (updated_avg_firing_rate, updated_g_tonic)
    """
    new_avg_rate = avg_firing_rate * (1 - 1/homeostatic_tau) + output_spikes / homeostatic_tau
    rate_error = target_rate - new_avg_rate
    new_g_tonic = g_tonic + homeostatic_strength * rate_error
    new_g_tonic = new_g_tonic.clamp(bounds[0], bounds[1])
    return new_avg_rate, new_g_tonic


def update_homeostatic_conductance_bidirectional(
    current_rate: torch.Tensor,
    avg_firing_rate: torch.Tensor,
    g_tonic: torch.Tensor,
    g_inh_tonic: torch.Tensor,
    target_rate: float,
    tau: float,
    strength: float,
    exc_bounds: Tuple[float, float] = (0.0, 0.5),
    inh_bounds: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update firing rate average and BOTH tonic excitatory AND inhibitory conductances.

    This is the biologically realistic bidirectional homeostasis that real brains use:

    1. When firing too LITTLE (rate < target):
       - Increase g_tonic (more excitation via persistent Na+, reduced leak, etc.)
       - Decrease g_inh_tonic (less tonic GABA_A inhibition)

    2. When firing too MUCH (rate > target):
       - Decrease g_tonic (less excitation)
       - Increase g_inh_tonic (more tonic GABA_A inhibition via extrasynaptic receptors)

    The key biological insight is that inhibition can ALWAYS be increased, even when
    excitation has already been reduced to zero. This prevents the saturation problem
    where neurons fire too much but g_tonic is already at minimum.

    Biological basis for tonic inhibition:
    - Extrasynaptic GABA_A receptors (α5/δ-subunit containing)
    - Activated by ambient GABA in extracellular space
    - Provides ~5-10% of total inhibitory conductance at rest
    - Can be upregulated by sustained high activity

    Args:
        current_rate: Current firing rate tensor (Hz recommended)
        avg_firing_rate: Running average of firing rate
        g_tonic: Current tonic excitatory conductance tensor
        g_inh_tonic: Current tonic inhibitory conductance tensor
        target_rate: Target firing rate (same units as current_rate)
        tau: Time constant for averaging (in update calls, e.g., cycles)
        strength: Conductance adjustment per unit of rate error
        exc_bounds: (min, max) bounds for g_tonic (excitatory)
        inh_bounds: (min, max) bounds for g_inh_tonic (inhibitory)

    Returns:
        Tuple of (updated_avg_firing_rate, updated_g_tonic, updated_g_inh_tonic)
    """
    # Update exponential moving average of firing rate
    new_avg_rate = avg_firing_rate + (current_rate - avg_firing_rate) / tau

    # Compute rate error (positive if firing too little)
    rate_error = target_rate - new_avg_rate

    # Compute adjustment magnitude
    adjustment = strength * rate_error

    # Apply bidirectional adjustments:
    # - Positive rate_error (firing too little): increase g_tonic, decrease g_inh_tonic
    # - Negative rate_error (firing too much): decrease g_tonic, increase g_inh_tonic
    new_g_tonic = (g_tonic + adjustment).clamp(exc_bounds[0], exc_bounds[1])
    new_g_inh_tonic = (g_inh_tonic - adjustment).clamp(inh_bounds[0], inh_bounds[1])

    return new_avg_rate, new_g_tonic, new_g_inh_tonic
