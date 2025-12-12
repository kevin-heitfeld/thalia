"""Ventral Tegmental Area (VTA) - Dopamine Reward Prediction Error System.

The VTA is the brain's primary dopamine source, computing reward prediction errors
(RPE) and broadcasting dopamine signals globally to modulate learning across all
brain regions.

**Biological Background**:
=========================
The VTA contains dopamine neurons that exhibit two distinct firing patterns:

1. **Tonic Firing** (4-5 Hz baseline):
   - Represents background motivation/mood state
   - Slow changes based on average prediction quality
   - Sets baseline learning rate across brain

2. **Phasic Bursts/Pauses**:
   - **Burst** (5-20 spikes): Unexpected rewards → DA increase → LTP
   - **Pause** (silence): Unexpected punishments → DA decrease → LTD
   - Fast (100-200ms) transient signals
   - Both components sum at target synapses

**Key Functions**:
==================
- **Reward Prediction Error**: δ = r + γV(s') - V(s)
- **Adaptive Normalization**: Tracks RPE statistics to prevent saturation
- **Global Broadcast**: Single dopamine level sent to all regions
- **Homeostatic Regulation**: Maintains dopamine within physiological bounds

Architecture Pattern:
=====================
Follows centralized broadcast pattern:
1. Brain creates VTADopamineSystem (like OscillatorManager)
2. System computes RPE and updates dopamine each timestep
3. Dopamine broadcast to regions (like set_oscillator_phases)
4. Regions use for plasticity modulation (optional)

Author: Thalia Project
Date: December 2025
"""

from dataclasses import dataclass
from typing import Optional
from thalia.core.neuromodulator_homeostasis import NeuromodulatorHomeostasis, NeuromodulatorHomeostasisConfig


@dataclass
class VTAConfig:
    """Configuration for VTA dopamine system.

    Parameters control dopamine dynamics and normalization.
    """
    # Phasic dopamine decay (reuptake by DAT transporters)
    # τ = 200ms → decay = exp(-dt/τ) ≈ 0.995 per ms
    phasic_decay_per_ms: float = 0.995

    # Tonic dopamine smoothing (slow baseline changes)
    # α = 0.05 → τ ≈ 20ms
    tonic_alpha: float = 0.05

    # Adaptive normalization for RPE
    rpe_avg_tau: float = 0.9  # EMA decay for running average
    rpe_clip: float = 2.0      # Clip normalized RPE to this range

    # Dopamine physiological limits
    min_dopamine: float = -2.0
    max_dopamine: float = 2.0

    # Homeostatic regulation (optional, can provide custom config)
    homeostatic_config: Optional[NeuromodulatorHomeostasisConfig] = None


class VTADopamineSystem:
    """
    Ventral Tegmental Area dopamine system.

    Manages tonic and phasic dopamine, computes reward prediction errors,
    and broadcasts normalized dopamine signal to all brain regions.

    Usage:
    ======
        # In EventDrivenBrain:
        self.vta = VTADopamineSystem()

        # Each timestep:
        self.vta.update(dt_ms=1.0, intrinsic_reward=0.1)
        dopamine = self.vta.get_global_dopamine()

        # Broadcast to regions:
        for region in self.regions:
            region.set_dopamine(dopamine)

        # When external reward arrives:
        self.vta.deliver_reward(external_reward=1.0, expected_value=0.5)

    Biological Accuracy:
    ====================
    - Separates tonic (intrinsic) and phasic (external) dopamine
    - Phasic decays exponentially (τ ~200ms)
    - Tonic updates slowly based on prediction quality
    - Adaptive normalization prevents saturation
    - Output clipped to physiological range [-2, +2]
    """

    def __init__(self, config: Optional[VTAConfig] = None):
        """Initialize VTA dopamine system.

        Args:
            config: VTA configuration. Uses defaults if None.
        """
        self.config = config or VTAConfig()

        # Dopamine components
        self._tonic_dopamine: float = 0.0   # Slow baseline (intrinsic)
        self._phasic_dopamine: float = 0.0  # Fast bursts (external)
        self._global_dopamine: float = 0.0  # Combined signal

        # Adaptive normalization state
        self._avg_abs_rpe: float = 0.5      # Running average of |RPE|
        self._rpe_history_count: int = 0    # Number of rewards seen

        # Homeostatic regulation
        homeostatic_cfg = self.config.homeostatic_config or NeuromodulatorHomeostasisConfig(target_level=0.0)
        self._homeostatic = NeuromodulatorHomeostasis(config=homeostatic_cfg)

    def update(self, dt_ms: float, intrinsic_reward: float) -> None:
        """Update dopamine levels for this timestep.

        Call this every timestep to:
        1. Update tonic dopamine from intrinsic reward
        2. Decay phasic dopamine toward zero
        3. Compute global dopamine (tonic + phasic)

        Args:
            dt_ms: Timestep in milliseconds
            intrinsic_reward: Reward from prediction quality [-1, 1]
        """
        # Update tonic dopamine (slow, smoothed)
        alpha = self.config.tonic_alpha
        self._tonic_dopamine = (
            (1 - alpha) * self._tonic_dopamine + alpha * intrinsic_reward
        )

        # Decay phasic dopamine (fast, exponential)
        decay = self.config.phasic_decay_per_ms ** dt_ms
        self._phasic_dopamine *= decay

        # Compute global dopamine
        self._global_dopamine = self._tonic_dopamine + self._phasic_dopamine

        # Clip to physiological range
        self._global_dopamine = max(
            self.config.min_dopamine,
            min(self.config.max_dopamine, self._global_dopamine)
        )

        # Update homeostatic regulation
        self._homeostatic.update(self._global_dopamine)

    def deliver_reward(self, external_reward: float, expected_value: float) -> float:
        """Deliver external reward and compute RPE.

        This is called when an external reward/punishment arrives.
        Computes reward prediction error (RPE) and updates phasic dopamine.

        Args:
            external_reward: Actual reward received
            expected_value: Predicted reward (from value estimate)

        Returns:
            Normalized RPE used for dopamine burst
        """
        # Compute raw RPE
        rpe = external_reward - expected_value

        # Normalize RPE adaptively
        normalized_rpe = self._compute_normalized_dopamine(rpe)

        # Update phasic dopamine (additive burst/dip)
        self._phasic_dopamine += normalized_rpe

        # Update global dopamine immediately
        self._global_dopamine = self._tonic_dopamine + self._phasic_dopamine
        self._global_dopamine = max(
            self.config.min_dopamine,
            min(self.config.max_dopamine, self._global_dopamine)
        )

        return normalized_rpe

    def _compute_normalized_dopamine(self, rpe: float) -> float:
        """Compute normalized dopamine from raw RPE.

        Uses adaptive normalization to prevent saturation:
        - Tracks running average of |RPE| to adapt to reward statistics
        - Outputs normalized RPE in range [-rpe_clip, +rpe_clip]

        This is the VTA's core computation: converting prediction error
        into a normalized dopamine signal suitable for learning.

        Args:
            rpe: Raw reward prediction error

        Returns:
            Normalized dopamine level
        """
        abs_rpe = abs(rpe)
        self._rpe_history_count += 1

        # Adaptive smoothing: slower early on for stability
        if self._rpe_history_count < 10:
            alpha = 1.0 / self._rpe_history_count
        else:
            alpha = 1.0 - self.config.rpe_avg_tau

        # Update running average of |RPE|
        self._avg_abs_rpe = (
            self.config.rpe_avg_tau * self._avg_abs_rpe + alpha * abs_rpe
        )

        # Normalize RPE by running average (with epsilon for stability)
        epsilon = 0.1
        normalized_rpe = rpe / (self._avg_abs_rpe + epsilon)

        # Clip to prevent extreme updates
        return max(
            -self.config.rpe_clip,
            min(self.config.rpe_clip, normalized_rpe)
        )

    def get_global_dopamine(self, apply_homeostasis: bool = True) -> float:
        """Get current global dopamine level for broadcast to regions.

        Args:
            apply_homeostasis: If True, apply receptor sensitivity scaling

        Returns:
            Combined dopamine (tonic + phasic) in [-2, +2]
        """
        if apply_homeostasis:
            return self._homeostatic.apply_sensitivity(self._global_dopamine)
        return self._global_dopamine

    def get_tonic_dopamine(self) -> float:
        """Get current tonic dopamine (slow baseline).

        Returns:
            Tonic dopamine level
        """
        return self._tonic_dopamine

    def get_phasic_dopamine(self) -> float:
        """Get current phasic dopamine (fast bursts/dips).

        Returns:
            Phasic dopamine level
        """
        return self._phasic_dopamine

    def get_state(self) -> dict:
        """Get VTA state for checkpointing.

        Returns:
            Dictionary with all VTA state
        """
        return {
            'tonic_dopamine': self._tonic_dopamine,
            'phasic_dopamine': self._phasic_dopamine,
            'global_dopamine': self._global_dopamine,
            'avg_abs_rpe': self._avg_abs_rpe,
            'rpe_history_count': self._rpe_history_count,
            'homeostatic': self._homeostatic.get_state(),
        }

    def set_state(self, state: dict) -> None:
        """Restore VTA state from checkpoint.

        Args:
            state: Dictionary from get_state()
        """
        self._tonic_dopamine = state['tonic_dopamine']
        self._phasic_dopamine = state['phasic_dopamine']
        self._global_dopamine = state['global_dopamine']
        self._avg_abs_rpe = state['avg_abs_rpe']
        self._rpe_history_count = state['rpe_history_count']
        if 'homeostatic' in state:
            self._homeostatic.set_state(state['homeostatic'])

    def reset_state(self) -> None:
        """Reset VTA to initial state."""
        self._tonic_dopamine = 0.0
        self._phasic_dopamine = 0.0
        self._global_dopamine = 0.0
        self._avg_abs_rpe = 0.5
        self._rpe_history_count = 0
        self._homeostatic.reset()

    def check_health(self) -> dict:
        """Check VTA health for diagnostics.

        Returns:
            Dictionary with health metrics
        """
        issues = []

        # Check for runaway dopamine
        if abs(self._global_dopamine) > 1.5:
            issues.append(f"High dopamine: {self._global_dopamine:.2f}")

        # Check for frozen learning (no RPE history)
        if self._rpe_history_count == 0:
            issues.append("No rewards seen yet")

        # Check for abnormal RPE statistics
        if self._avg_abs_rpe > 2.0:
            issues.append(f"High RPE variance: {self._avg_abs_rpe:.2f}")

        homeostatic_health = self._homeostatic.check_health()

        return {
            'is_healthy': len(issues) == 0 and homeostatic_health['is_healthy'],
            'issues': issues + homeostatic_health['issues'],
            'warnings': homeostatic_health['warnings'],
            'tonic': self._tonic_dopamine,
            'phasic': self._phasic_dopamine,
            'global': self._global_dopamine,
            'avg_abs_rpe': self._avg_abs_rpe,
            'rewards_seen': self._rpe_history_count,
            'receptor_sensitivity': homeostatic_health['sensitivity'],
        }
