"""
Brain Oscillator System - Neural Rhythm Generation and Coordination.

Neural oscillations are fundamental to brain function, coordinating activity
across spatial and temporal scales. This module provides base classes for
implementing biologically realistic oscillatory dynamics.

**Biological Background**:
=========================
Brain oscillations span multiple frequency bands, each with distinct functions:

- **Delta (0.5-4 Hz)**: Deep sleep, large-scale synchronization
- **Theta (4-10 Hz)**: Memory encoding, spatial navigation, sequence learning
- **Alpha (8-13 Hz)**: Attention gating, inhibitory control, sensory suppression
- **Beta (13-30 Hz)**: Motor control, active cognitive processing
- **Gamma (30-100 Hz)**: Feature binding, local circuit processing
- **High-Gamma (100-200 Hz)**: Rapid information processing

**Functional Roles**:
====================
1. **Temporal Coordination**: Phase-locking synchronizes distributed neurons
2. **Information Routing**: Phase determines when information can pass
3. **Working Memory**: Nested oscillations encode sequence positions
4. **Prediction**: Phase provides temporal anticipation
5. **Gating**: Oscillatory inhibition controls information flow

Key Mechanisms:
- Phase tracking: Current position in oscillation cycle
- Frequency modulation: Dynamic adjustment of oscillation speed
- Phase synchronization: Aligning oscillations across regions
- Nested oscillations: Fast rhythms within slower rhythms

References:
- Buzsáki & Draguhn (2004): Neuronal oscillations in cortical networks
- Lisman & Jensen (2013): The theta-gamma neural code
- Fries (2015): Rhythms for cognition: communication through coherence
- Colgin (2013): Mechanisms and functions of theta rhythms

Implementation Notes:
====================

This base class provides common functionality for all oscillators:
- Phase advancement with configurable timestep
- Frequency modulation
- Phase synchronization
- State management (get/set/reset)

Subclasses implement:
- Specific oscillation properties (theta, gamma, etc.)
- Signal generation (sine, square, burst patterns)
- Coupling mechanisms (phase-amplitude, phase-phase)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch.nn as nn

from thalia.constants.time import MS_PER_SECOND, TAU
from thalia.core.errors import ConfigurationError


@dataclass
class OscillatorConfig:
    """Base configuration for oscillators.

    Attributes:
        frequency_hz: Base oscillation frequency in Hz
        dt_ms: Default timestep in milliseconds
        initial_phase: Starting phase in radians [0, 2π)
        amplitude: Base amplitude (signal range multiplier)
        phase_noise_std: Standard deviation of phase noise in radians (default: 0.0)
            Biological oscillators have drift ~0.05 rad (~3 degrees)
            This makes phase coding more robust and realistic
    """

    frequency_hz: float = 10.0
    dt_ms: float = 1.0
    initial_phase: float = 0.0
    amplitude: float = 1.0
    phase_noise_std: float = 0.0  # Enable with 0.05 for biological realism


class BrainOscillator(nn.Module, ABC):
    """
    Abstract base class for neural oscillators.

    Provides common oscillation mechanics:
    - Phase tracking and advancement
    - Frequency modulation
    - Phase synchronization
    - State management

    Subclasses implement specific oscillation types (theta, gamma, etc.)
    and their characteristic properties.

    Biological Analog:
        - Phase: Current point in oscillation cycle
        - Frequency: Rate of oscillation (Hz)
        - Period: Time for one complete cycle (ms)
        - Amplitude: Strength of oscillation

    Example:
        ```python
        # Create theta oscillator (8 Hz)
        theta = SinusoidalOscillator(frequency_hz=OSCILLATOR_DEFAULTS['theta'])

        # Advance through time
        for t in range(1000):
            theta.advance(dt_ms=1.0)
            phase = theta.phase  # Current phase [0, 2π)
            signal = theta.signal  # Oscillation value
        ```
    """

    def __init__(self, config: OscillatorConfig):
        super().__init__()
        self.config = config

        # Phase state (in radians)
        self._phase = config.initial_phase

        # Frequency (can be modulated)
        self._frequency_hz = config.frequency_hz

        # Time tracking
        self.time_ms = 0.0

        # Precompute phase increment per ms
        self._update_phase_increment()

    def _update_phase_increment(self) -> None:
        """Update phase increment based on current frequency."""
        self._phase_per_ms = TAU * self._frequency_hz / MS_PER_SECOND

    def advance(self, dt_ms: Optional[float] = None) -> None:
        """
        Advance the oscillator by one timestep.

        Updates phase based on current frequency and wraps to [0, 2π).
        Optionally adds phase noise for biological realism.

        Args:
            dt_ms: Timestep in milliseconds (uses config default if None)
        """
        dt = dt_ms or self.config.dt_ms

        # Advance phase
        self._phase += self._phase_per_ms * dt

        # Add phase noise if configured (biological oscillator drift)
        if self.config.phase_noise_std > 0:
            noise = random.gauss(0, self.config.phase_noise_std)
            self._phase += noise

        # Wrap to [0, τ)
        self._phase = self._phase % TAU

        # Update time
        self.time_ms += dt

    def sync_to_phase(self, target_phase: float) -> None:
        """
        Synchronize oscillator to a specific phase.

        Use this to coordinate multiple oscillators or sync to external signal.

        Args:
            target_phase: Target phase in radians
        """
        self._phase = target_phase % TAU

    def set_frequency(self, frequency_hz: float) -> None:
        """
        Modulate the oscillation frequency.

        Changes take effect on next advance() call.

        Args:
            frequency_hz: New frequency in Hz
        """
        self._frequency_hz = frequency_hz
        self._update_phase_increment()

    @property
    def phase(self) -> float:
        """Current phase in radians [0, 2π)."""
        return self._phase

    @property
    def frequency_hz(self) -> float:
        """Current oscillation frequency in Hz."""
        return self._frequency_hz

    @property
    @abstractmethod
    def oscillation_period_ms(self) -> float:
        """
        Period of one complete oscillation cycle in milliseconds.

        Subclasses should return: MS_PER_SECOND / frequency_hz
        """
        pass

    @property
    @abstractmethod
    def signal(self) -> float:
        """
        Current oscillation signal value.

        Subclasses implement specific waveforms:
        - Sinusoidal: sin(phase)
        - Square: sign(sin(phase))
        - Burst: complex spike patterns

        Returns:
            Signal value (typically normalized to [-1, 1] × amplitude)
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """
        Get oscillator state for serialization.

        Returns:
            Dictionary with phase, frequency, time
        """
        return {
            "phase": self._phase,
            "frequency_hz": self._frequency_hz,
            "time_ms": self.time_ms,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore oscillator state from dictionary.

        Args:
            state: State dictionary from get_state()
        """
        self._phase = state["phase"]
        self._frequency_hz = state["frequency_hz"]
        self.time_ms = state["time_ms"]
        self._update_phase_increment()

    def reset_state(self) -> None:
        """Reset oscillator to initial state."""
        self._phase = self.config.initial_phase
        self._frequency_hz = self.config.frequency_hz
        self.time_ms = 0.0
        self._update_phase_increment()


class SinusoidalOscillator(BrainOscillator):
    """
    Generic sinusoidal oscillator for all brain frequency bands.

    Supports all standard brain rhythms with biological frequency ranges:
    - Delta (0.5-4 Hz): Deep sleep, memory consolidation
    - Theta (4-10 Hz): Memory encoding, spatial navigation
    - Alpha (8-13 Hz): Attention gating, inhibitory control
    - Beta (13-30 Hz): Motor control, active thinking
    - Gamma (30-100 Hz): Feature binding, local processing

    The only difference between oscillator types is their frequency.
    Use OSCILLATOR_DEFAULTS dict for standard biological frequencies.

    Example:
        ```python
        # Create oscillators with biological defaults
        theta = SinusoidalOscillator(frequency_hz=OSCILLATOR_DEFAULTS['theta'])  # 8 Hz
        gamma = SinusoidalOscillator(frequency_hz=OSCILLATOR_DEFAULTS['gamma'])  # 40 Hz

        # Or specify custom frequency
        fast_gamma = SinusoidalOscillator(frequency_hz=60.0)
        ```
    """

    def __init__(
        self,
        frequency_hz: float = 10.0,
        dt_ms: float = 1.0,
        initial_phase: float = 0.0,
        amplitude: float = 1.0,
        phase_noise_std: float = 0.0,
    ):
        config = OscillatorConfig(
            frequency_hz=frequency_hz,
            dt_ms=dt_ms,
            initial_phase=initial_phase,
            amplitude=amplitude,
            phase_noise_std=phase_noise_std,
        )
        super().__init__(config)

    @property
    def oscillation_period_ms(self) -> float:
        """Period of one complete oscillation cycle in milliseconds."""
        return MS_PER_SECOND / self._frequency_hz

    @property
    def signal(self) -> float:
        """Sinusoidal oscillation signal."""
        return math.sin(self._phase) * self.config.amplitude

    def is_up_state(self, threshold: float = 0.0) -> bool:
        """Check if currently in up-state (positive phase).

        Up-states are optimal for:
        - Cortical plasticity
        - Hippocampal replay
        - Memory reactivation

        Particularly relevant for slow oscillations (delta/theta).

        Args:
            threshold: Minimum signal value to count as up-state

        Returns:
            True if in up-state (signal > threshold)
        """
        return self.signal > threshold

    def is_down_state(self, threshold: float = 0.0) -> bool:
        """Check if currently in down-state (negative phase).

        Down-states are for:
        - Synaptic downscaling
        - Metabolic recovery
        - Reduced activity

        Particularly relevant for slow oscillations (delta/theta).

        Args:
            threshold: Maximum signal value to count as down-state

        Returns:
            True if in down-state (signal <= threshold)
        """
        return self.signal <= threshold


# Default frequencies for brain oscillation bands (Hz)
OSCILLATOR_DEFAULTS = {
    "delta": 2.0,  # 0.5-4 Hz: Deep sleep, memory consolidation
    "theta": 8.0,  # 4-10 Hz: Memory encoding, spatial navigation
    "alpha": 10.0,  # 8-13 Hz: Attention gating, inhibitory control
    "beta": 20.0,  # 13-30 Hz: Motor control, active thinking
    "gamma": 40.0,  # 30-100 Hz: Feature binding, local processing
    "ripple": 150.0,  # 100-200 Hz: Sharp-wave ripples, memory replay
}


@dataclass
class OscillatorCoupling:
    """Defines phase-amplitude coupling for a fast oscillator.

    Biological Background:
    =====================
    Cross-frequency coupling (CFC) is widespread in brain activity:
    - Gamma modulated by theta, beta, alpha (hippocampus, motor, attention)
    - Beta modulated by theta, alpha (working memory-action coordination)
    - Theta modulated by delta (sleep consolidation)

    The general pattern: ALL slower oscillators can modulate the fast oscillator.
    This is biologically accurate - neurons integrate multiple oscillatory inputs.

    Simplified Configuration:
    ========================
    Instead of specifying (slow, fast) pairs, we only specify the FAST oscillator.
    The system automatically couples it with ALL slower oscillators based on the
    frequency hierarchy: delta < theta < alpha < beta < gamma

    Example:
        # Gamma is modulated by theta, beta, alpha, delta (all slower oscillators)
        coupling = OscillatorCoupling(
            oscillator='gamma',
            coupling_strength=0.8,
            min_amplitude=0.2,
            modulation_type='cosine'
        )

    This replaces the old approach of defining each (slow, fast) pair explicitly,
    which was redundant since we already know the frequency hierarchy.

    References:
    - Lisman & Jensen (2013): The theta-gamma neural code
    - Canolty & Knight (2010): The functional role of cross-frequency coupling
    - Tort et al. (2010): Measuring phase-amplitude coupling

    Attributes:
        oscillator: Name of fast oscillator to modulate (e.g., 'gamma', 'beta')
        coupling_strength: Base coupling strength [-1, 1]. 0 = no coupling, positive = enhancement, negative = suppression
        min_amplitude: Minimum amplitude at unfavorable phases [0, 1]
        modulation_type: 'cosine' (max at trough/0) or 'sine' (max at peak/π/2)
        per_oscillator_strength: Optional dict to override strength per slow oscillator
            e.g., {'theta': 0.8, 'beta': 0.6} for gamma coupling. Negative values suppress.
    """

    oscillator: str
    coupling_strength: float = 0.8
    min_amplitude: float = 0.2
    modulation_type: str = "cosine"  # 'cosine' or 'sine'
    per_oscillator_strength: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate coupling parameters."""
        if not -1.0 <= self.coupling_strength <= 1.0:
            raise ConfigurationError(
                f"coupling_strength must be in [-1, 1], got {self.coupling_strength}"
            )
        if not 0.0 <= self.min_amplitude <= 1.0:
            raise ConfigurationError(f"min_amplitude must be in [0, 1], got {self.min_amplitude}")
        if self.modulation_type not in ("cosine", "sine"):
            raise ConfigurationError(
                f"modulation_type must be 'cosine' or 'sine', got {self.modulation_type}"
            )
        if self.per_oscillator_strength is not None:
            for strength in self.per_oscillator_strength.values():
                if not -1.0 <= strength <= 1.0:
                    raise ConfigurationError(
                        f"per_oscillator_strength values must be in [-1, 1], got {strength}"
                    )

    def get_strength_for(self, slow_oscillator: str) -> float:
        """Get coupling strength for a specific slow oscillator.

        Args:
            slow_oscillator: Name of slower modulating oscillator

        Returns:
            Coupling strength [0, 1]
        """
        if (
            self.per_oscillator_strength is not None
            and slow_oscillator in self.per_oscillator_strength
        ):
            return self.per_oscillator_strength[slow_oscillator]
        return self.coupling_strength


class OscillatorManager:
    """
    Central oscillator system for the brain.

    Manages brain-wide oscillations and broadcasts phases to regions.
    Similar to how dopamine is centrally managed and broadcast.

    Biological Rationale:
    =====================
    - EEG records brain-wide oscillations (not local to individual regions)
    - Oscillations coordinate distributed processing across brain areas
    - Phase relationships between regions are critical for communication
    - Single global oscillator per frequency ensures synchronization

    Architecture Pattern:
    ====================
    Follows the same centralized broadcast pattern as dopamine:
    1. Brain creates OscillatorManager (like dopamine system)
    2. Manager advances all oscillators once per timestep
    3. Phases/signals broadcast to regions (like set_dopamine)
    4. Regions use phases for modulation (optional)

    Usage:
    ======
        # In the brain:
        self.oscillators = OscillatorManager(dt_ms=1.0, device="cuda")

        # Each timestep:
        self.oscillators.advance(dt_ms=1.0)
        phases = self.oscillators.get_phases()
        signals = self.oscillators.get_signals()

        # Broadcast to regions:
        self.cortex.set_oscillator_phases(phases, signals)
        self.hippocampus.set_oscillator_phases(phases, signals)

    Benefits:
    =========
    - Biological accuracy (brain-wide synchronization)
    - Efficiency (single oscillator per frequency)
    - Easy phase-amplitude coupling across regions
    - Consistent with dopamine architecture
    - Central monitoring and control

    References:
    ===========
    - Buzsáki & Draguhn (2004): Neuronal oscillations in cortical networks
    - Fries (2015): Rhythms for cognition: communication through coherence
    """

    # Oscillator hierarchy (low to high frequency)
    OSCILLATOR_HIERARCHY = ["delta", "theta", "alpha", "beta", "gamma", "ripple"]

    def __init__(
        self,
        dt_ms: float = 1.0,
        device: str = "cpu",
        delta_freq: float = OSCILLATOR_DEFAULTS["delta"],
        theta_freq: float = OSCILLATOR_DEFAULTS["theta"],
        alpha_freq: float = OSCILLATOR_DEFAULTS["alpha"],
        beta_freq: float = OSCILLATOR_DEFAULTS["beta"],
        gamma_freq: float = OSCILLATOR_DEFAULTS["gamma"],
        ripple_freq: float = OSCILLATOR_DEFAULTS["ripple"],
        couplings: Optional[List[OscillatorCoupling]] = None,
    ):
        """Initialize oscillator manager with all brain rhythms.

        Gamma oscillator is DISABLED by default to allow emergence from
        circuit dynamics (L6→TRN→Thalamus feedback loop ~25ms = 40 Hz).
        Enable explicitly if needed via enable_oscillator('gamma', True).

        Args:
            dt_ms: Timestep in milliseconds
            device: Device for tensor operations (cpu/cuda)
            delta_freq: Delta frequency in Hz (default: 2.0 for SWS)
            theta_freq: Theta frequency in Hz (default: 8.0 for memory)
            alpha_freq: Alpha frequency in Hz (default: 10.0 for attention)
            beta_freq: Beta frequency in Hz (default: 20.0 for motor)
            gamma_freq: Gamma frequency in Hz (default: 40.0 for binding)
            couplings: List of oscillator couplings (default: theta, alpha, beta, gamma coupled)
        """
        self.dt_ms = dt_ms
        self.device = device

        # Store frequencies for diagnostics
        self.delta_freq = delta_freq
        self.theta_freq = theta_freq
        self.alpha_freq = alpha_freq
        self.beta_freq = beta_freq
        self.gamma_freq = gamma_freq
        self.ripple_freq = ripple_freq

        # Create all oscillators
        self.delta = SinusoidalOscillator(frequency_hz=delta_freq, dt_ms=dt_ms)
        self.theta = SinusoidalOscillator(frequency_hz=theta_freq, dt_ms=dt_ms)
        self.alpha = SinusoidalOscillator(frequency_hz=alpha_freq, dt_ms=dt_ms)
        self.beta = SinusoidalOscillator(frequency_hz=beta_freq, dt_ms=dt_ms)
        self.gamma = SinusoidalOscillator(frequency_hz=gamma_freq, dt_ms=dt_ms)
        self.ripple = SinusoidalOscillator(frequency_hz=ripple_freq, dt_ms=dt_ms)

        # Cross-frequency couplings (simplified configuration)
        # Only specify the FAST oscillator - slow oscillators inferred from hierarchy
        # Complete default configuration: all oscillators except delta (slowest) are coupled
        if couplings is None:
            self.couplings = [
                # Theta: Modulated by delta
                # Sleep consolidation (delta-theta during NREM, memory replay)
                OscillatorCoupling(
                    oscillator="theta",
                    coupling_strength=0.7,
                    min_amplitude=0.1,
                    modulation_type="cosine",  # Max theta at delta up-state
                    per_oscillator_strength={
                        "delta": 0.7,  # Strong: replay nested in slow waves
                    },
                ),
                # Alpha: Modulated by delta, theta
                # Attention gating and inhibitory control
                OscillatorCoupling(
                    oscillator="alpha",
                    coupling_strength=0.4,  # Base strength
                    min_amplitude=0.3,
                    modulation_type="sine",  # Max alpha at phase offset
                    per_oscillator_strength={
                        "delta": 0.3,  # Weak: sleep-wake transitions
                        "theta": 0.5,  # Medium: theta-alpha interplay in attention
                    },
                ),
                # Beta: Modulated by delta, theta, alpha
                # Working memory-action coordination, motor planning
                OscillatorCoupling(
                    oscillator="beta",
                    coupling_strength=0.5,  # Base strength
                    min_amplitude=0.3,
                    modulation_type="cosine",
                    per_oscillator_strength={
                        "delta": 0.2,  # Very weak: sleep effect
                        "theta": 0.6,  # Medium: working memory coordination
                        "alpha": 0.4,  # Weak: attention modulation
                    },
                ),
                # Gamma: Modulated by ALL slower oscillators (delta, theta, alpha, beta)
                # Working memory (theta-gamma), motor timing (beta-gamma), attention (alpha-gamma)
                OscillatorCoupling(
                    oscillator="gamma",
                    coupling_strength=0.7,  # Base strength
                    min_amplitude=0.2,
                    modulation_type="cosine",  # Max at trough
                    per_oscillator_strength={
                        "delta": 0.3,  # Weakest: sleep consolidation
                        "theta": 0.8,  # Strongest: working memory (7±2 slots)
                        "alpha": 0.5,  # Medium: attention gating
                        "beta": 0.6,  # Medium: motor timing
                    },
                ),
                # Ripple: Sharp-wave ripples modulated by ALL slower oscillators
                # Memory consolidation during slow-wave sleep (delta) and offline periods (theta)
                # Suppressed during active processing (alpha/beta/gamma high)
                OscillatorCoupling(
                    oscillator="ripple",
                    coupling_strength=0.8,  # Base strength
                    min_amplitude=0.1,  # Can be strongly suppressed during waking
                    modulation_type="cosine",  # Max at specific phase
                    per_oscillator_strength={
                        "delta": 0.9,  # Very strong: ripples during slow-wave sleep
                        "theta": 0.6,  # Medium: ripples at theta trough (offline)
                        "alpha": -0.4,  # Negative: suppress during attention
                        "beta": -0.5,  # Negative: suppress during active cognition
                        "gamma": -0.3,  # Negative: suppress during sensory processing
                    },
                ),
            ]
        else:
            self.couplings = couplings

        # Build coupling lookup for faster access
        self._coupling_map: Dict[str, OscillatorCoupling] = {
            c.oscillator: c for c in self.couplings
        }

        # State tracking
        self._time_ms: float = 0.0
        self._enabled: Dict[str, bool] = {
            "delta": True,
            "theta": True,
            "alpha": True,
            "beta": True,
            "gamma": False,  # Disabled: Should emerge from L6→TRN loop (~25ms)
            "ripple": True,
        }

    def advance(self, dt_ms: Optional[float] = None) -> None:
        """Advance all enabled oscillators by one timestep.

        Args:
            dt_ms: Timestep in milliseconds (uses default if None)
        """
        dt = dt_ms or self.dt_ms

        if self._enabled["delta"]:
            self.delta.advance(dt)
        if self._enabled["theta"]:
            self.theta.advance(dt)
        if self._enabled["alpha"]:
            self.alpha.advance(dt)
        if self._enabled["beta"]:
            self.beta.advance(dt)
        if self._enabled["gamma"]:
            self.gamma.advance(dt)

        self._time_ms += dt

    def get_phases(self) -> Dict[str, float]:
        """Get current phases of all oscillators.

        Returns:
            Dictionary mapping oscillator name to phase [0, 2π)
        """
        return {
            "delta": self.delta.phase,
            "theta": self.theta.phase,
            "alpha": self.alpha.phase,
            "beta": self.beta.phase,
            "gamma": self.gamma.phase,
        }

    def get_frequencies(self) -> Dict[str, float]:
        """Get current frequencies of all oscillators.

        Returns:
            Dictionary mapping oscillator name to frequency (Hz)
        """
        return {
            "delta": self.delta.frequency_hz,
            "theta": self.theta.frequency_hz,
            "alpha": self.alpha.frequency_hz,
            "beta": self.beta.frequency_hz,
            "gamma": self.gamma.frequency_hz,
        }

    def get_signals(self) -> Dict[str, float]:
        """Get current signal values of all oscillators with coupling applied.

        Applies cross-frequency coupling: fast oscillator amplitudes are
        modulated by slow oscillator phases.

        Returns:
            Dictionary mapping oscillator name to coupled signal value [-1, 1]
        """
        # Get base signals
        signals = {
            "delta": self.delta.signal,
            "theta": self.theta.signal,
            "alpha": self.alpha.signal,
            "beta": self.beta.signal,
            "gamma": self.gamma.signal,
        }

        # Get effective amplitudes (accounts for ALL slow oscillator modulation)
        effective_amps = self.get_effective_amplitudes()

        # Apply modulation to coupled oscillators
        for osc_name in signals:
            if osc_name in effective_amps:
                signals[osc_name] *= effective_amps[osc_name]

        return signals

    def get_oscillator(self, name: str) -> BrainOscillator:
        """Get oscillator by name.

        Args:
            name: Oscillator name ('delta', 'theta', 'alpha', 'beta', 'gamma')

        Returns:
            Oscillator instance

        Raises:
            ValueError: If oscillator name is invalid
        """
        oscillators = {
            "delta": self.delta,
            "theta": self.theta,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }

        if name not in oscillators:
            raise ValueError(
                f"Unknown oscillator: {name}. " f"Choose from: {list(oscillators.keys())}"
            )

        return oscillators[name]

    def set_frequency(self, name: str, frequency_hz: float) -> None:
        """Modulate oscillator frequency.

        Useful for state-dependent modulation (e.g., REM vs NREM sleep).

        Args:
            name: Oscillator name
            frequency_hz: New frequency in Hz
        """
        oscillator = self.get_oscillator(name)
        oscillator.set_frequency(frequency_hz)

    def enable_oscillator(self, name: str, enabled: bool = True) -> None:
        """Enable or disable an oscillator.

        Disabled oscillators don't advance (saves computation).

        Args:
            name: Oscillator name
            enabled: Whether to enable (True) or disable (False)
        """
        if name in self._enabled:
            self._enabled[name] = enabled

    def get_coupled_amplitude(self, fast: str, slow: str) -> float:
        """Get fast oscillator's amplitude modulated by slow oscillator's phase.

        Implements phase-amplitude coupling: the amplitude of the fast oscillator
        varies as a function of the slow oscillator's phase.

        Biological Example:
        - Theta phase modulates gamma amplitude in hippocampus
        - Gamma is strongest at theta trough (encoding, ~180°)
        - Gamma is weakest at theta peak (retrieval, ~0°)

        Args:
            fast: Name of fast oscillator to modulate (e.g., 'gamma')
            slow: Name of slow oscillator providing phase (e.g., 'theta')

        Returns:
            Amplitude modulation factor [min_amplitude, 1.0]
            Returns 1.0 if no coupling exists for this oscillator

        Example:
            >>> manager = OscillatorManager()
            >>> amp = manager.get_coupled_amplitude('gamma', 'theta')
            >>> # amp varies from 0.2 (theta peak) to 1.0 (theta trough)
        """
        # Check if fast oscillator has coupling configured
        if fast not in self._coupling_map:
            return 1.0  # No coupling = constant amplitude

        coupling = self._coupling_map[fast]

        # Check if this slow oscillator should modulate the fast one
        # (slow must be earlier in hierarchy than fast)
        try:
            fast_idx = self.OSCILLATOR_HIERARCHY.index(fast)
            slow_idx = self.OSCILLATOR_HIERARCHY.index(slow)
        except ValueError:
            return 1.0  # Unknown oscillator

        if slow_idx >= fast_idx:
            return 1.0  # Not a valid slow→fast coupling

        # Get coupling strength for this specific slow oscillator
        strength = coupling.get_strength_for(slow)

        # Get slow oscillator's phase
        slow_osc = self.get_oscillator(slow)
        slow_phase = slow_osc.phase

        # Calculate modulation based on type
        if coupling.modulation_type == "cosine":
            # Cosine: max at trough (phase=0), min at peak (phase=π)
            # (1 + cos(θ))/2 maps [0, 2π] → [0, 1] with max at θ=0
            modulation = 0.5 * (1.0 + math.cos(slow_phase))
        else:  # sine
            # Sine: max at π/2, min at 3π/2
            modulation = 0.5 * (1.0 + math.sin(slow_phase))

        # Apply coupling strength and minimum amplitude
        # amplitude = min + (max - min) × modulation × strength
        amplitude = coupling.min_amplitude + (1.0 - coupling.min_amplitude) * modulation * strength

        return amplitude

    def get_coupled_amplitudes(self) -> Dict[str, float]:
        """Get all coupled amplitude modulation factors.

        Returns dictionary with keys like 'gamma_by_theta', 'theta_by_delta', etc.
        For each fast oscillator with coupling configured, computes modulation
        from ALL slower oscillators in the hierarchy.

        Returns:
            Dictionary mapping '{fast}_by_{slow}' to amplitude factor [0, 1]

        Example:
            >>> manager = OscillatorManager()
            >>> amps = manager.get_coupled_amplitudes()
            >>> # {'gamma_by_theta': 0.8, 'gamma_by_beta': 0.6, 'gamma_by_alpha': 0.5, ...}
        """
        amplitudes = {}

        # For each coupled oscillator, compute modulation from all slower oscillators
        for fast in self._coupling_map:
            fast_idx = self.OSCILLATOR_HIERARCHY.index(fast)

            # Get all slower oscillators
            for slow_idx in range(fast_idx):
                slow = self.OSCILLATOR_HIERARCHY[slow_idx]
                key = f"{fast}_by_{slow}"
                amplitudes[key] = self.get_coupled_amplitude(fast, slow)

        return amplitudes

    def get_effective_amplitudes(self) -> Dict[str, float]:
        """Get effective amplitudes via biologically-accurate membrane integration.

        Models how neurons integrate oscillatory inputs: ALL slower oscillators
        contribute to membrane excitability, and their effects are AVERAGED.

        Biological Mechanism:
        ====================
        Real neurons integrate synaptic currents from multiple oscillatory inputs:
        - Each oscillator modulates membrane excitability
        - Membrane integrates (sums/averages) all contributions
        - Stronger couplings have more weight in the integration
        - This naturally creates competition and cooperation

        Key Differences from Multiplication:
        - Multiplication: gamma = theta(0.8) × beta(0.6) = 0.48 (very suppressed)
        - Integration: gamma = weighted_avg([theta(0.8), beta(0.6)]) ≈ 0.7 (moderate)

        Biological Advantages:
        - Competition: Strong beta can dilute weak theta's effect
        - Cooperation: Multiple moderate inputs can boost amplitude
        - Non-saturating: Doesn't collapse to near-zero with many inputs
        - Accurate: Matches how real membranes integrate currents

        Returns:
            Dictionary mapping oscillator name to effective amplitude [0, 1]
            Keys: 'delta', 'theta', 'alpha', 'beta', 'gamma'

        Example:
            >>> manager = OscillatorManager()
            >>> amps = manager.get_effective_amplitudes()
            >>> amps['gamma']  # Theta + beta + alpha + delta all modulate
            0.7  # Weighted average of individual contributions

        References:
        - Buzsáki (2006): Rhythms of the Brain - membrane integration
        - Jensen & Colgin (2007): Cross-frequency coupling mechanisms
        """
        effective_amps = {}

        # Get all pairwise couplings
        pairwise = self.get_coupled_amplitudes()

        # For each oscillator in hierarchy
        for fast_idx, fast in enumerate(self.OSCILLATOR_HIERARCHY):
            # Check if this oscillator has coupling configured
            if fast not in self._coupling_map:
                effective_amps[fast] = 1.0  # No coupling = no modulation
                continue

            coupling = self._coupling_map[fast]

            # Collect contributions from ALL slower oscillators
            contributions = []
            weights = []

            for slow_idx in range(fast_idx):
                slow = self.OSCILLATOR_HIERARCHY[slow_idx]
                key = f"{fast}_by_{slow}"

                if key in pairwise:
                    contributions.append(pairwise[key])
                    # Weight by coupling strength for this slow oscillator
                    weights.append(coupling.get_strength_for(slow))

            if contributions:
                # Weighted average: models membrane integration of multiple inputs
                total_weight = sum(weights)
                if total_weight > 0:
                    amplitude = sum(c * w for c, w in zip(contributions, weights)) / total_weight
                else:
                    amplitude = 1.0
            else:
                # No active couplings = no modulation
                amplitude = 1.0

            effective_amps[fast] = amplitude

        return effective_amps

    def get_theta_slot(self, n_slots: int = 7) -> int:
        """Get current slot index [0, n_slots-1] based on theta phase.

        Divides theta cycle into n_slots for sequence encoding.
        This implements the 7±2 working memory capacity using theta-gamma coupling.

        Biological Background:
        - ~7±2 gamma cycles fit in one theta cycle
        - Each slot represents a distinct memory item
        - Slot position encodes temporal order in sequences
        - Phase precession uses slots to represent spatial position

        Slot Positions:
        - Slot 0: Theta trough (~180°) - encoding phase, strongest gamma
        - Slot n-1: Near theta peak (~0°) - retrieval phase, weakest gamma

        Args:
            n_slots: Number of slots per theta cycle (default: 7 for working memory)

        Returns:
            Current slot index [0, n_slots-1]
            Returns 0 if theta oscillator doesn't exist

        Example:
            >>> manager = OscillatorManager()
            >>> for t in range(125):  # One theta cycle at 8 Hz
            ...     manager.advance(dt_ms=1.0)
            ...     slot = manager.get_theta_slot(n_slots=7)
            ...     # slot cycles through 0, 1, 2, 3, 4, 5, 6, 0, ...

        References:
        - Lisman & Jensen (2013): The theta-gamma neural code
        - Jensen & Lisman (2005): Hippocampal sequence encoding driven by theta-gamma coupling
        """
        try:
            theta = self.get_oscillator("theta")
        except ValueError:
            return 0  # No theta oscillator

        # Calculate progress through theta cycle [0, 1)
        theta_progress = theta.phase / TAU

        # Map to slot index
        slot = int(theta_progress * n_slots) % n_slots

        return slot

    def set_sleep_stage(self, stage: str) -> None:
        """Modulate oscillations for sleep stages.

        Args:
            stage: Sleep stage ('AWAKE', 'NREM', 'REM')
        """
        if stage == "NREM":
            # Slow-wave sleep: strong delta, weak gamma
            self.set_frequency("delta", 2.0)
            self.set_frequency("theta", 6.0)  # Slower theta
            self.set_frequency("gamma", 30.0)  # Slow gamma
            self.enable_oscillator("alpha", False)  # No alpha during sleep
            self.enable_oscillator("beta", False)  # No beta during sleep

        elif stage == "REM":
            # Paradoxical sleep: weak delta, strong theta, fast gamma
            self.set_frequency("delta", 1.0)  # Minimal delta
            self.set_frequency("theta", 7.0)  # Theta dominant
            self.set_frequency("gamma", 60.0)  # Fast gamma
            self.enable_oscillator("alpha", False)
            self.enable_oscillator("beta", False)

        else:  # AWAKE or default
            # Restore normal frequencies
            self.set_frequency("delta", 2.0)
            self.set_frequency("theta", 8.0)
            self.set_frequency("alpha", 10.0)
            self.set_frequency("beta", 20.0)
            self.set_frequency("gamma", 40.0)
            # Enable all
            for name in self._enabled:
                self.enable_oscillator(name, True)

    def reset(self) -> None:
        """Reset all oscillators to initial state."""
        self.delta.reset_state()
        self.theta.reset_state()
        self.alpha.reset_state()
        self.beta.reset_state()
        self.gamma.reset_state()
        self._time_ms = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Get state of all oscillators for serialization.

        Returns:
            Dictionary with all oscillator states
        """
        return {
            "delta": self.delta.get_state(),
            "theta": self.theta.get_state(),
            "alpha": self.alpha.get_state(),
            "beta": self.beta.get_state(),
            "gamma": self.gamma.get_state(),
            "time_ms": self._time_ms,
            "enabled": self._enabled.copy(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore oscillator states from dictionary.

        Args:
            state: State dictionary from get_state()
        """
        self.delta.set_state(state["delta"])
        self.theta.set_state(state["theta"])
        self.alpha.set_state(state["alpha"])
        self.beta.set_state(state["beta"])
        self.gamma.set_state(state["gamma"])
        self._time_ms = state.get("time_ms", 0.0)
        self._enabled = state.get(
            "enabled", {name: True for name in ["delta", "theta", "alpha", "beta", "gamma"]}
        )

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update oscillator phase increments for new timestep.

        Called by brain when dt_ms changes during adaptive timestep adjustment.
        Updates all oscillators to maintain correct phase advancement rate.

        Args:
            dt_ms: New timestep in milliseconds

        Example:
            # Speed up replay 10x
            brain.set_timestep(10.0)
            # Oscillators automatically adjust phase increments
        """
        self.dt_ms = dt_ms

        # Update all oscillators
        # Each oscillator's config.dt_ms will be updated, which affects phase increment
        self.delta.config.dt_ms = dt_ms
        self.delta._update_phase_increment()

        self.theta.config.dt_ms = dt_ms
        self.theta._update_phase_increment()

        self.alpha.config.dt_ms = dt_ms
        self.alpha._update_phase_increment()

        self.beta.config.dt_ms = dt_ms
        self.beta._update_phase_increment()

        self.gamma.config.dt_ms = dt_ms
        self.gamma._update_phase_increment()

        self.ripple.config.dt_ms = dt_ms
        self.ripple._update_phase_increment()
