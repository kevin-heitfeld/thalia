"""
Curriculum-Based Noise Scheduler

Implements systematic noise scheduling across developmental stages to improve
generalization, robustness, and biological realism.

Philosophy:
===========
- Early stages: Low noise for stable foundations
- Middle stages: Moderate noise for generalization
- Late stages: Higher noise for robustness and exploration

Noise Types:
============
1. Membrane Noise: Stochastic fluctuations in neuron membrane potentials
2. Weight Noise: Exploration noise during synaptic plasticity
3. Spike Noise: Temporal jitter in spike generation
4. Input Noise: Data augmentation and perturbations

Biological Rationale:
=====================
- Real neurons are intrinsically noisy (ion channel stochasticity)
- Noise aids exploration and prevents overfitting
- Critical periods have different noise tolerance
- Developmental stages show varying noise robustness
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .curriculum import CurriculumStage


class NoiseType(Enum):
    """Types of noise that can be scheduled."""

    MEMBRANE = "membrane"  # Neuron membrane potential noise
    WEIGHT = "weight"  # Synaptic weight perturbation
    SPIKE = "spike"  # Temporal jitter
    INPUT = "input"  # Data augmentation


@dataclass
class NoiseProfile:
    """Noise configuration for a specific stage."""

    # Membrane noise (added to neuron membrane potentials)
    membrane_noise_std: float = 0.01

    # Weight noise (perturbation during plasticity for exploration)
    weight_noise_std: float = 0.0
    enable_weight_noise: bool = False

    # Oscillator phase noise (drift in oscillator phases, radians)
    oscillator_phase_noise_std: float = 0.0
    enable_oscillator_phase_noise: bool = False

    # Spike timing jitter (ms) - NOT USED at 1ms resolution
    spike_jitter_ms: float = 0.0
    enable_spike_jitter: bool = False

    # Data augmentation strength (0.0 = none, 1.0 = aggressive)
    augmentation_strength: float = 0.0
    enable_augmentation: bool = False  # Working memory noise (PFC)
    wm_noise_std: float = 0.02

    # Proprioceptive noise (sensorimotor tasks)
    proprioception_noise_scale: float = 0.1

    # REM consolidation noise (schema extraction)
    rem_noise_std: float = 0.3


@dataclass
class NoiseSchedulerConfig:
    """Configuration for curriculum-based noise scheduling."""

    # Enable/disable entire noise system
    enabled: bool = True

    # Stage-specific overrides (if None, use defaults)
    stage_overrides: Optional[Dict[CurriculumStage, NoiseProfile]] = None

    # Adaptive noise based on criticality
    enable_criticality_adaptation: bool = True
    criticality_noise_boost: float = 1.5  # Multiply noise when subcritical
    criticality_noise_reduction: float = 0.5  # Multiply noise when supercritical

    # Adaptive noise based on performance
    enable_performance_adaptation: bool = True
    performance_threshold_low: float = 0.6  # Below this, reduce noise
    performance_threshold_high: float = 0.85  # Above this, increase noise

    # Verbosity
    verbose: bool = True


class NoiseScheduler:
    """
    Curriculum-based noise scheduler.

    Manages noise levels across developmental stages, adapting to brain state
    and performance to optimize learning dynamics.
    """

    def __init__(self, config: Optional[NoiseSchedulerConfig] = None):
        """Initialize noise scheduler.

        Args:
            config: Configuration for noise scheduling
        """
        self.config = config or NoiseSchedulerConfig()

        # Default noise profiles per stage (can be overridden)
        self._default_profiles = self._create_default_profiles()

        # Current active profile
        self._current_profile: NoiseProfile = self._default_profiles[CurriculumStage.BOOTSTRAP]
        self._current_stage: CurriculumStage = CurriculumStage.BOOTSTRAP

        # Adaptation state
        self._adaptation_multiplier: float = 1.0

        if self.config.verbose:
            print("\n[NoiseScheduler] Initialized with curriculum-based noise scheduling")
            print(f"  - Criticality adaptation: {self.config.enable_criticality_adaptation}")
            print(f"  - Performance adaptation: {self.config.enable_performance_adaptation}")

    def _create_default_profiles(self) -> Dict[CurriculumStage, NoiseProfile]:
        """Create default noise profiles for each curriculum stage.

        Returns:
            Dictionary mapping stages to noise profiles
        """
        profiles = {}

        # Stage 0: Bootstrap - Minimal noise for stable foundation
        profiles[CurriculumStage.BOOTSTRAP] = NoiseProfile(
            membrane_noise_std=0.0,  # no noise for initial stability
            weight_noise_std=0.0,
            enable_weight_noise=False,  # Not yet - need stable learning
            oscillator_phase_noise_std=0.0,
            enable_oscillator_phase_noise=False,  # Not yet - need stable rhythms
            spike_jitter_ms=0.0,
            enable_spike_jitter=False,
            augmentation_strength=0.0,  # No augmentation in bootstrap
            enable_augmentation=False,
            wm_noise_std=0.0,  # No WM yet
            proprioception_noise_scale=0.0,  # No sensorimotor tasks yet
            rem_noise_std=0.3,  # Standard REM noise for consolidation
        )

        # Stage 1: Sensorimotor - Low noise for stable foundation
        profiles[CurriculumStage.SENSORIMOTOR] = NoiseProfile(
            membrane_noise_std=0.01,  # enable for exploration
            weight_noise_std=0.0,
            enable_weight_noise=False,  # Not yet - learning basics
            oscillator_phase_noise_std=0.0,
            enable_oscillator_phase_noise=False,  # Not yet - need stable rhythms
            spike_jitter_ms=0.0,
            enable_spike_jitter=False,
            augmentation_strength=0.05,  # Minimal (5%)
            enable_augmentation=True,
            wm_noise_std=0.0,  # No WM yet
            proprioception_noise_scale=0.1,  # 10% - realistic sensor noise
            rem_noise_std=0.3,  # Standard REM noise
        )

        return profiles

    def get_noise_profile(
        self,
        stage: CurriculumStage,
        apply_adaptation: bool = True,
    ) -> NoiseProfile:
        """Get noise profile for a specific stage.

        Args:
            stage: Curriculum stage
            apply_adaptation: Whether to apply adaptive multipliers

        Returns:
            NoiseProfile for the stage (with adaptations if enabled)
        """
        if not self.config.enabled:
            # Return zero-noise profile
            return NoiseProfile(
                membrane_noise_std=0.0,
                weight_noise_std=0.0,
                enable_weight_noise=False,
                spike_jitter_ms=0.0,
                enable_spike_jitter=False,
                augmentation_strength=0.0,
                enable_augmentation=False,
                wm_noise_std=0.0,
                proprioception_noise_scale=0.0,
                rem_noise_std=0.0,
            )

        # Check for stage-specific override
        if self.config.stage_overrides and stage in self.config.stage_overrides:
            profile = self.config.stage_overrides[stage]
        else:
            # Fallback to default profile
            # TODO: Add warning if stage not found in defaults
            profile = self._default_profiles.get(
                stage, self._default_profiles[CurriculumStage.BOOTSTRAP]
            )

        # Apply adaptive multiplier if enabled
        if apply_adaptation and self._adaptation_multiplier != 1.0:
            profile = self._apply_adaptation_multiplier(profile)

        return profile

    def _apply_adaptation_multiplier(self, profile: NoiseProfile) -> NoiseProfile:
        """Apply adaptive multiplier to noise profile.

        Args:
            profile: Base noise profile

        Returns:
            Adapted noise profile
        """
        return NoiseProfile(
            membrane_noise_std=profile.membrane_noise_std * self._adaptation_multiplier,
            weight_noise_std=profile.weight_noise_std * self._adaptation_multiplier,
            enable_weight_noise=profile.enable_weight_noise,
            oscillator_phase_noise_std=profile.oscillator_phase_noise_std
            * self._adaptation_multiplier,
            enable_oscillator_phase_noise=profile.enable_oscillator_phase_noise,
            spike_jitter_ms=profile.spike_jitter_ms * self._adaptation_multiplier,
            enable_spike_jitter=profile.enable_spike_jitter,
            augmentation_strength=min(
                1.0, profile.augmentation_strength * self._adaptation_multiplier
            ),
            enable_augmentation=profile.enable_augmentation,
            wm_noise_std=profile.wm_noise_std * self._adaptation_multiplier,
            proprioception_noise_scale=profile.proprioception_noise_scale,  # Keep stable
            rem_noise_std=profile.rem_noise_std,  # Keep stable for consolidation
        )

    def update(
        self,
        current_stage: CurriculumStage,
        performance: Optional[float] = None,
        criticality: Optional[float] = None,
    ) -> None:
        """Update noise scheduler based on current brain state.

        Args:
            current_stage: Current curriculum stage
            performance: Current task performance (0.0-1.0)
            criticality: Current criticality metric (1.0 = critical)
        """
        self._current_stage = current_stage
        self._current_profile = self.get_noise_profile(current_stage, apply_adaptation=False)

        # Reset adaptation multiplier
        self._adaptation_multiplier = 1.0

        # Adapt based on criticality
        if self.config.enable_criticality_adaptation and criticality is not None:
            if criticality < 0.95:  # Subcritical - boost noise for exploration
                self._adaptation_multiplier *= self.config.criticality_noise_boost
            elif criticality > 1.15:  # Supercritical - reduce noise for stability
                self._adaptation_multiplier *= self.config.criticality_noise_reduction

        # Adapt based on performance
        if self.config.enable_performance_adaptation and performance is not None:
            if performance < self.config.performance_threshold_low:
                # Low performance - reduce noise to stabilize learning
                self._adaptation_multiplier *= 0.7
            elif performance > self.config.performance_threshold_high:
                # High performance - increase noise to push generalization
                self._adaptation_multiplier *= 1.2

        # Clamp adaptation multiplier to reasonable range
        self._adaptation_multiplier = max(0.5, min(2.0, self._adaptation_multiplier))

    def get_current_profile(self) -> NoiseProfile:
        """Get current active noise profile (with adaptations).

        Returns:
            Current noise profile
        """
        return self.get_noise_profile(self._current_stage, apply_adaptation=True)

    def set_stage(self, stage: CurriculumStage) -> None:
        """Set current curriculum stage.

        Args:
            stage: New curriculum stage
        """
        if stage != self._current_stage:
            self._current_stage = stage
            self._current_profile = self.get_noise_profile(stage)

            if self.config.verbose:
                print(f"\n[NoiseScheduler] Stage changed to {stage.name}")
                print(f"  New profile: {self._current_profile}")

    def get_membrane_noise_for_region(self, region_name: str) -> float:
        """Get membrane noise for a specific region.

        Different regions may have different noise characteristics.

        Args:
            region_name: Name of the region

        Returns:
            Membrane noise standard deviation for the region
        """
        profile = self.get_current_profile()
        base_noise = profile.membrane_noise_std

        # Region-specific adjustments (biological variation)
        region_factors = {
            "cortex_l4": 1.0,  # Standard
            "cortex_l23": 1.0,  # Standard
            "cortex_l5": 0.8,  # Slightly lower (more stable)
            "hippocampus": 1.2,  # Higher (more variable)
            "prefrontal": 0.9,  # Lower (needs stability for WM)
            "striatum": 1.0,  # Standard
            "cerebellum": 0.7,  # Lower (precise timing needed)
            "thalamus": 0.8,  # Lower (relay precision)
        }

        factor = region_factors.get(region_name.lower(), 1.0)
        return base_noise * factor

    def should_apply_weight_noise(self) -> bool:
        """Check if weight noise should be applied in current state.

        Returns:
            True if weight noise should be applied
        """
        profile = self.get_current_profile()
        return profile.enable_weight_noise

    def get_weight_noise_std(self) -> float:
        """Get current weight noise standard deviation.

        Returns:
            Weight noise std
        """
        profile = self.get_current_profile()
        return profile.weight_noise_std if profile.enable_weight_noise else 0.0

    def get_augmentation_strength(self) -> float:
        """Get current data augmentation strength.

        Returns:
            Augmentation strength (0.0-1.0)
        """
        profile = self.get_current_profile()
        return profile.augmentation_strength if profile.enable_augmentation else 0.0

    def should_apply_oscillator_phase_noise(self) -> bool:
        """Check if oscillator phase noise should be applied.

        Returns:
            True if oscillator phase noise enabled
        """
        profile = self.get_current_profile()
        return profile.enable_oscillator_phase_noise

    def get_oscillator_phase_noise_std(self) -> float:
        """Get current oscillator phase noise standard deviation.

        Returns:
            Phase noise std in radians (0.0 if disabled)
        """
        profile = self.get_current_profile()
        return profile.oscillator_phase_noise_std if profile.enable_oscillator_phase_noise else 0.0
