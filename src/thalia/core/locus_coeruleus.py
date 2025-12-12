"""
Locus Coeruleus (LC) - Norepinephrine Arousal System.

The locus coeruleus is the brain's primary norepinephrine (NE) source, modulating
arousal, attention, and neural gain across all brain regions.

This module implements centralized NE management following the same pattern as
VTADopamineSystem and OscillatorManager.

Biological Background:
======================
The LC is a small nucleus in the brainstem that:
1. Projects to entire brain (cortex, hippocampus, cerebellum, striatum)
2. Responds to novelty, uncertainty, and unexpected events
3. Modulates neural gain (signal-to-noise ratio)
4. Facilitates behavioral flexibility and task switching
5. High NE → arousal, attention, reset dynamics
6. Low NE → baseline processing

Key Functions:
==============
- **Arousal**: Global alertness/wakefulness state
- **Uncertainty response**: High uncertainty → high NE
- **Gain modulation**: NE increases neural responsiveness
- **Network reset**: High NE burst → clear working memory, update beliefs
- **Stress response**: Task difficulty modulates arousal

Architecture Pattern:
=====================
Follows centralized broadcast pattern:
1. Brain creates LocusCoeruleusSystem (like VTA/OscillatorManager)
2. System computes NE from uncertainty/novelty each timestep
3. NE broadcast to regions (like dopamine/oscillators)
4. Regions use for gain modulation (optional)

Author: Thalia Project
Date: December 2025
"""

from dataclasses import dataclass
from typing import Optional
from .neuromodulator_homeostasis import NeuromodulatorHomeostasis, NeuromodulatorHomeostasisConfig


@dataclass
class LocusCoeruleusConfig:
    """Configuration for LC norepinephrine system.

    Parameters control NE dynamics and arousal computation.
    """
    # Baseline arousal (resting state)
    baseline_arousal: float = 0.3  # Low but non-zero

    # NE decay (reuptake by NET transporters)
    # τ = 100ms → decay = exp(-dt/τ) ≈ 0.990 per ms
    ne_decay_per_ms: float = 0.990

    # Arousal update rate (how fast arousal tracks uncertainty)
    arousal_alpha: float = 0.1  # Medium smoothing

    # Uncertainty sensitivity
    uncertainty_gain: float = 1.0  # How much uncertainty drives arousal

    # Phasic NE burst parameters
    burst_threshold: float = 0.5   # Uncertainty > this triggers burst
    burst_magnitude: float = 0.5   # Size of phasic burst

    # NE physiological limits
    min_norepinephrine: float = 0.0
    max_norepinephrine: float = 2.0

    # Homeostatic regulation (optional, can provide custom config)
    homeostatic_config: Optional[NeuromodulatorHomeostasisConfig] = None


class LocusCoeruleusSystem:
    """
    Locus Coeruleus norepinephrine arousal system.

    Manages tonic and phasic norepinephrine, computes arousal from uncertainty,
    and broadcasts NE signal to all brain regions.

    Usage:
    ======
        # In EventDrivenBrain:
        self.locus_coeruleus = LocusCoeruleusSystem()

        # Each timestep:
        self.locus_coeruleus.update(dt_ms=1.0, uncertainty=0.4)
        norepinephrine = self.locus_coeruleus.get_norepinephrine()

        # Broadcast to regions:
        for region in self.regions:
            region.set_norepinephrine(norepinephrine)

        # For unexpected events (novelty):
        self.locus_coeruleus.trigger_phasic_burst(magnitude=0.5)

    Biological Accuracy:
    ====================
    - Separates tonic (baseline arousal) and phasic (event-driven) NE
    - Tonic tracks task uncertainty/difficulty
    - Phasic bursts for unexpected events
    - NE decays exponentially (τ ~100ms)
    - Output clipped to physiological range [0, 2]
    """

    def __init__(self, config: Optional[LocusCoeruleusConfig] = None):
        """Initialize LC norepinephrine system.

        Args:
            config: LC configuration. Uses defaults if None.
        """
        self.config = config or LocusCoeruleusConfig()

        # Norepinephrine components
        self._tonic_ne: float = self.config.baseline_arousal  # Slow arousal
        self._phasic_ne: float = 0.0                          # Fast bursts
        self._global_ne: float = self.config.baseline_arousal # Combined signal

        # Arousal tracking
        self._arousal: float = self.config.baseline_arousal
        self._uncertainty: float = 0.0  # Current uncertainty estimate

        # Homeostatic regulation
        homeostatic_cfg = self.config.homeostatic_config or NeuromodulatorHomeostasisConfig(target_level=0.5)
        self._homeostatic = NeuromodulatorHomeostasis(config=homeostatic_cfg)

    def update(self, dt_ms: float, uncertainty: float) -> None:
        """Update norepinephrine levels for this timestep.

        Call this every timestep to:
        1. Update tonic NE from uncertainty/arousal
        2. Decay phasic NE toward zero
        3. Compute global NE (tonic + phasic)

        Args:
            dt_ms: Timestep in milliseconds
            uncertainty: Task uncertainty in [0, 1]
                        High uncertainty → high arousal → high NE
        """
        # Clip uncertainty to valid range
        uncertainty = max(0.0, min(1.0, uncertainty))
        self._uncertainty = uncertainty

        # Update arousal from uncertainty (smoothed)
        alpha = self.config.arousal_alpha
        target_arousal = (
            self.config.baseline_arousal +
            self.config.uncertainty_gain * uncertainty
        )
        self._arousal = (1 - alpha) * self._arousal + alpha * target_arousal

        # Tonic NE tracks arousal
        self._tonic_ne = self._arousal

        # Check for phasic burst trigger (high uncertainty)
        if uncertainty > self.config.burst_threshold:
            # Automatic burst for high uncertainty
            burst_strength = (uncertainty - self.config.burst_threshold) / (1.0 - self.config.burst_threshold)
            self._phasic_ne += self.config.burst_magnitude * burst_strength

        # Decay phasic NE (fast, exponential)
        decay = self.config.ne_decay_per_ms ** dt_ms
        self._phasic_ne *= decay

        # Compute global NE
        self._global_ne = self._tonic_ne + self._phasic_ne

        # Clip to physiological range
        self._global_ne = max(
            self.config.min_norepinephrine,
            min(self.config.max_norepinephrine, self._global_ne)
        )

        # Update homeostatic regulation
        self._homeostatic.update(self._global_ne)

    def trigger_phasic_burst(self, magnitude: Optional[float] = None) -> None:
        """Trigger explicit phasic NE burst (e.g., for unexpected events).

        Call this when detecting:
        - Unexpected sensory input
        - Novel stimuli
        - Prediction errors
        - Task switches

        Args:
            magnitude: Burst size. Uses config default if None.
        """
        mag = magnitude if magnitude is not None else self.config.burst_magnitude
        self._phasic_ne += mag

        # Update global immediately
        self._global_ne = self._tonic_ne + self._phasic_ne
        self._global_ne = max(
            self.config.min_norepinephrine,
            min(self.config.max_norepinephrine, self._global_ne)
        )

    def get_norepinephrine(self, apply_homeostasis: bool = True) -> float:
        """Get current global NE level for broadcast to regions.

        Args:
            apply_homeostasis: If True, apply receptor sensitivity scaling

        Returns:
            Combined NE (tonic + phasic) in [0, 2]
        """
        if apply_homeostasis:
            return self._homeostatic.apply_sensitivity(self._global_ne)
        return self._global_ne

    def get_tonic_ne(self) -> float:
        """Get current tonic NE (slow arousal).

        Returns:
            Tonic NE level
        """
        return self._tonic_ne

    def get_phasic_ne(self) -> float:
        """Get current phasic NE (fast bursts).

        Returns:
            Phasic NE level
        """
        return self._phasic_ne

    def get_arousal(self) -> float:
        """Get current arousal level.

        Returns:
            Arousal level (roughly equivalent to tonic NE)
        """
        return self._arousal

    def get_uncertainty(self) -> float:
        """Get current uncertainty estimate.

        Returns:
            Uncertainty level [0, 1]
        """
        return self._uncertainty

    def get_state(self) -> dict:
        """Get LC state for checkpointing.

        Returns:
            Dictionary with all LC state
        """
        return {
            'tonic_ne': self._tonic_ne,
            'phasic_ne': self._phasic_ne,
            'global_ne': self._global_ne,
            'arousal': self._arousal,
            'uncertainty': self._uncertainty,
            'homeostatic': self._homeostatic.get_state(),
        }

    def set_state(self, state: dict) -> None:
        """Restore LC state from checkpoint.

        Args:
            state: Dictionary from get_state()
        """
        self._tonic_ne = state['tonic_ne']
        self._phasic_ne = state['phasic_ne']
        self._global_ne = state['global_ne']
        self._arousal = state['arousal']
        self._uncertainty = state['uncertainty']
        if 'homeostatic' in state:
            self._homeostatic.set_state(state['homeostatic'])

    def reset_state(self) -> None:
        """Reset LC to initial state."""
        self._tonic_ne = self.config.baseline_arousal
        self._phasic_ne = 0.0
        self._global_ne = self.config.baseline_arousal
        self._arousal = self.config.baseline_arousal
        self._uncertainty = 0.0
        self._homeostatic.reset()

    def check_health(self) -> dict:
        """Check LC health for diagnostics.

        Returns:
            Dictionary with health metrics
        """
        issues = []

        # Check for runaway NE
        if self._global_ne > 1.5:
            issues.append(f"High NE (overly aroused): {self._global_ne:.2f}")

        # Check for stuck arousal
        if self._arousal < 0.1 and self._uncertainty > 0.5:
            issues.append("Arousal not tracking uncertainty")

        # Check for abnormal phasic
        if self._phasic_ne > 1.0:
            issues.append(f"Excessive phasic NE: {self._phasic_ne:.2f}")

        homeostatic_health = self._homeostatic.check_health()

        return {
            'is_healthy': len(issues) == 0 and homeostatic_health['is_healthy'],
            'issues': issues + homeostatic_health['issues'],
            'warnings': homeostatic_health['warnings'],
            'tonic_ne': self._tonic_ne,
            'phasic_ne': self._phasic_ne,
            'global_ne': self._global_ne,
            'arousal': self._arousal,
            'uncertainty': self._uncertainty,
            'receptor_sensitivity': homeostatic_health['sensitivity'],
        }
