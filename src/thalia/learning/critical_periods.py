"""
Critical Period Gating - Time-windowed plasticity modulation.

This module implements critical periods for language and skill acquisition,
modulating learning rates based on developmental stage and age.

Key Concepts:
=============

1. CRITICAL PERIODS IN DEVELOPMENT
   Human language acquisition has sensitive periods:
   - Phonology: 0-6 months (native phoneme discrimination)
   - Grammar: 1-7 years (syntax acquisition)
   - Semantics: Extended window (1-18 years)

2. SIGMOIDAL DECLINE AFTER WINDOW
   Learning doesn't stop abruptly, but becomes progressively harder:
   - Before window: Immature (50% efficiency)
   - During window: Peak plasticity (120% efficiency)
   - After window: Declining (sigmoid decay to 20% floor)

3. BIOLOGICAL BASIS
   Neural plasticity is highest during critical periods due to:
   - High NMDA receptor expression
   - Low GABAergic inhibition
   - Maximal dendritic spine turnover

   After critical periods:
   - Increased inhibition stabilizes circuits
   - Reduced spine turnover
   - Learning still possible but requires more samples

Usage:
======

    from thalia.learning.critical_periods import CriticalPeriodGating

    # Create gating module
    gating = CriticalPeriodGating()

    # Modulate learning rate based on age and domain
    current_step = 25000
    base_lr = 0.001

    # Phonology window (0-50k): still open at 25k
    phonology_lr = gating.gate_learning(base_lr, 'phonology', current_step)
    # → 0.0012 (120% of base, peak plasticity)

    # Grammar window (25k-150k): just opening
    grammar_lr = gating.gate_learning(base_lr, 'grammar', current_step)
    # → 0.0012 (120% of base, entering peak)

    # After window closes
    late_phonology_lr = gating.gate_learning(base_lr, 'phonology', 100000)
    # → 0.00025 (25% of base, declining)

References:
===========
- Werker & Tees (1984): Phoneme discrimination critical period
- Newport (1990): Less is More - Critical periods in language
- Hensch (2004): Critical period plasticity in neural circuits
- Knudsen (2004): Sensitive periods in brain development

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
import math

from thalia.core.errors import ConfigurationError


@dataclass
class CriticalPeriodWindow:
    """Configuration for a critical period window.

    Attributes:
        start_step: When window opens (peak plasticity begins)
        end_step: When window closes (decline begins)
        peak_multiplier: Learning rate multiplier during peak (default 1.2)
        early_multiplier: Multiplier before window opens (default 0.5)
        late_floor: Minimum multiplier after window closes (default 0.2)
        decay_rate: How quickly plasticity declines after window (default 20000 steps)
    """
    start_step: int
    end_step: int
    peak_multiplier: float = 1.2
    early_multiplier: float = 0.5
    late_floor: float = 0.2
    decay_rate: float = 20000.0  # Sigmoid decay rate in steps


@dataclass
class CriticalPeriodConfig:
    """Configuration for all critical periods.

    Default windows based on human developmental timeline,
    compressed to training steps.
    """
    # Phonology: Native phoneme discrimination (infant)
    phonology: CriticalPeriodWindow = field(default_factory=lambda: CriticalPeriodWindow(
        start_step=0,
        end_step=50000,
        peak_multiplier=1.2,
        early_multiplier=0.5,
        late_floor=0.2,
    ))

    # Grammar: Syntax acquisition (toddler-child)
    grammar: CriticalPeriodWindow = field(default_factory=lambda: CriticalPeriodWindow(
        start_step=25000,
        end_step=150000,
        peak_multiplier=1.2,
        early_multiplier=0.5,
        late_floor=0.2,
    ))

    # Semantics: Vocabulary and meaning (extended window)
    semantics: CriticalPeriodWindow = field(default_factory=lambda: CriticalPeriodWindow(
        start_step=50000,
        end_step=300000,
        peak_multiplier=1.15,  # Slightly lower peak
        early_multiplier=0.6,
        late_floor=0.3,  # Higher floor (easier to learn late)
    ))

    # Face recognition: Early visual expertise
    face_recognition: CriticalPeriodWindow = field(default_factory=lambda: CriticalPeriodWindow(
        start_step=0,
        end_step=100000,
        peak_multiplier=1.2,
        early_multiplier=0.5,
        late_floor=0.25,
    ))

    # Motor skills: Sensorimotor coordination
    motor: CriticalPeriodWindow = field(default_factory=lambda: CriticalPeriodWindow(
        start_step=0,
        end_step=75000,
        peak_multiplier=1.25,  # Very high peak for motor
        early_multiplier=0.4,
        late_floor=0.3,
    ))


class CriticalPeriodGating:
    """Modulate learning rates based on critical period windows.

    This class implements biologically-inspired critical periods where
    learning is easier during specific developmental windows.

    The modulation follows a three-phase pattern:
    1. Early (before window): Immature, 50% efficiency
    2. Peak (during window): Optimal plasticity, 120% efficiency
    3. Late (after window): Declining, sigmoid decay to 20% floor

    Example:
        >>> gating = CriticalPeriodGating()
        >>>
        >>> # During critical period (step 25000)
        >>> lr = gating.gate_learning(0.001, 'phonology', 25000)
        >>> # lr ≈ 0.0012 (120% boost)
        >>>
        >>> # After critical period (step 100000)
        >>> lr = gating.gate_learning(0.001, 'phonology', 100000)
        >>> # lr ≈ 0.00025 (75% decline from peak)
        >>>
        >>> # Custom window
        >>> gating.add_domain('custom_skill', start=10000, end=50000)
        >>> lr = gating.gate_learning(0.001, 'custom_skill', 30000)
    """

    def __init__(self, config: Optional[CriticalPeriodConfig] = None):
        """Initialize critical period gating.

        Args:
            config: Configuration for critical period windows.
                    If None, uses default human developmental windows.
        """
        self.config = config or CriticalPeriodConfig()

        # Build domain lookup
        self._windows: Dict[str, CriticalPeriodWindow] = {
            'phonology': self.config.phonology,
            'grammar': self.config.grammar,
            'semantics': self.config.semantics,
            'face_recognition': self.config.face_recognition,
            'motor': self.config.motor,
        }

    def add_domain(
        self,
        domain: str,
        start: int,
        end: int,
        peak_multiplier: float = 1.2,
        early_multiplier: float = 0.5,
        late_floor: float = 0.2,
        decay_rate: float = 20000.0,
    ) -> None:
        """Add a custom critical period domain.

        Args:
            domain: Domain name (e.g., 'music', 'math')
            start: Step when window opens
            end: Step when window closes
            peak_multiplier: Learning rate multiplier during peak
            early_multiplier: Multiplier before window
            late_floor: Minimum multiplier after window
            decay_rate: Sigmoid decay rate after window
        """
        self._windows[domain] = CriticalPeriodWindow(
            start_step=start,
            end_step=end,
            peak_multiplier=peak_multiplier,
            early_multiplier=early_multiplier,
            late_floor=late_floor,
            decay_rate=decay_rate,
        )

    def gate_learning(
        self,
        learning_rate: float,
        domain: str,
        age: int,
    ) -> float:
        """Modulate learning rate based on critical period.

        Args:
            learning_rate: Base learning rate
            domain: Learning domain ('phonology', 'grammar', etc.)
            age: Current training step (age of system)

        Returns:
            Modulated learning rate

        Raises:
            ValueError: If domain not recognized
        """
        if domain not in self._windows:
            raise ConfigurationError(
                f"Unknown domain '{domain}'. Available domains: "
                f"{list(self._windows.keys())}"
            )

        window = self._windows[domain]
        multiplier = self._compute_multiplier(age, window)

        return learning_rate * multiplier

    def _compute_multiplier(
        self,
        age: int,
        window: CriticalPeriodWindow,
    ) -> float:
        """Compute learning rate multiplier for given age and window.

        Three phases:
        1. Before window: early_multiplier
        2. During window: peak_multiplier
        3. After window: sigmoid decay from peak to floor

        Args:
            age: Current training step
            window: Critical period window configuration

        Returns:
            Learning rate multiplier [early_multiplier, peak_multiplier]
        """
        if age < window.start_step:
            # Phase 1: Too early (immature)
            return window.early_multiplier

        elif age <= window.end_step:
            # Phase 2: Peak plasticity
            return window.peak_multiplier

        else:
            # Phase 3: Declining (sigmoid decay)
            # Decay from peak_multiplier to late_floor
            steps_past_window = age - window.end_step

            # Sigmoid decay: 1 / (1 + exp(x / decay_rate))
            # At x=0: 1.0 (peak)
            # At x=∞: 0.0 (floor)
            decay_factor = 1.0 / (1.0 + math.exp(steps_past_window / window.decay_rate))

            # Interpolate between floor and peak
            multiplier = (
                window.late_floor +
                (window.peak_multiplier - window.late_floor) * decay_factor
            )

            return multiplier

    def get_window_status(
        self,
        domain: str,
        age: int,
    ) -> Dict[str, Any]:
        """Get detailed status of critical period window.

        Useful for monitoring and logging.

        Args:
            domain: Learning domain
            age: Current training step

        Returns:
            Dictionary with window status:
            - phase: 'early', 'peak', or 'late'
            - multiplier: Current learning rate multiplier
            - progress: Progress through window [0, 1]
            - steps_remaining: Steps until window closes (or None if closed)
        """
        if domain not in self._windows:
            raise ConfigurationError(f"Unknown domain '{domain}'")

        window = self._windows[domain]
        multiplier = self._compute_multiplier(age, window)

        if age < window.start_step:
            phase = 'early'
            progress = 0.0
            steps_remaining = window.end_step - age
        elif age <= window.end_step:
            phase = 'peak'
            window_length = window.end_step - window.start_step
            progress = (age - window.start_step) / window_length if window_length > 0 else 1.0
            steps_remaining = window.end_step - age
        else:
            phase = 'late'
            progress = 1.0
            steps_remaining = None

        return {
            'domain': domain,
            'age': age,
            'phase': phase,
            'multiplier': multiplier,
            'progress': progress,
            'steps_remaining': steps_remaining,
            'window_start': window.start_step,
            'window_end': window.end_step,
        }

    def get_all_domains(self) -> list[str]:
        """Get list of all available domains.

        Returns:
            List of domain names
        """
        return list(self._windows.keys())

    def is_in_peak(self, domain: str, age: int) -> bool:
        """Check if current age is in peak plasticity window.

        Args:
            domain: Learning domain
            age: Current training step

        Returns:
            True if in peak window, False otherwise
        """
        if domain not in self._windows:
            raise ConfigurationError(f"Unknown domain '{domain}'")

        window = self._windows[domain]
        return window.start_step <= age <= window.end_step

    def get_optimal_age(self, domain: str) -> Tuple[int, int]:
        """Get optimal age range (peak window) for a domain.

        Args:
            domain: Learning domain

        Returns:
            Tuple of (start_step, end_step) for peak window
        """
        if domain not in self._windows:
            raise ConfigurationError(f"Unknown domain '{domain}'")

        window = self._windows[domain]
        return (window.start_step, window.end_step)
