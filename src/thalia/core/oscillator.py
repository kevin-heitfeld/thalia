"""
Brain Oscillator Base Classes

Neural oscillations are fundamental to brain function, coordinating activity
across spatial and temporal scales. This module provides base classes for
implementing oscillatory dynamics in spiking neural networks.

Biological Background:
=====================

Brain oscillations span multiple frequency bands:
- Delta (0.5-4 Hz): Deep sleep, attention
- Theta (4-10 Hz): Memory encoding, spatial navigation
- Alpha (8-13 Hz): Attention, inhibitory control
- Beta (13-30 Hz): Motor control, cognitive processing
- Gamma (30-100 Hz): Binding, local processing
- High-gamma (100-200 Hz): Rapid processing

Oscillations serve multiple functions:
1. TEMPORAL COORDINATION: Phase-locking synchronizes distributed neurons
2. INFORMATION ROUTING: Phase-amplitude coupling routes information
3. WORKING MEMORY: Nested oscillations represent sequence positions
4. PREDICTION: Phase provides temporal anticipation

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import math

import torch.nn as nn


@dataclass
class OscillatorConfig:
    """Base configuration for oscillators.

    Attributes:
        frequency_hz: Base oscillation frequency in Hz
        dt_ms: Default timestep in milliseconds
        initial_phase: Starting phase in radians [0, 2π)
        amplitude: Base amplitude (signal range multiplier)
    """
    frequency_hz: float = 10.0
    dt_ms: float = 1.0
    initial_phase: float = 0.0
    amplitude: float = 1.0


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
        theta = ThetaOscillator(frequency_hz=8.0)

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
        self._phase_per_ms = 2.0 * math.pi * self._frequency_hz / 1000.0

    def advance(self, dt_ms: Optional[float] = None) -> None:
        """
        Advance the oscillator by one timestep.

        Updates phase based on current frequency and wraps to [0, 2π).

        Args:
            dt_ms: Timestep in milliseconds (uses config default if None)
        """
        dt = dt_ms or self.config.dt_ms

        # Advance phase
        self._phase += self._phase_per_ms * dt

        # Wrap to [0, 2π)
        self._phase = self._phase % (2.0 * math.pi)

        # Update time
        self.time_ms += dt

    def sync_to_phase(self, target_phase: float) -> None:
        """
        Synchronize oscillator to a specific phase.

        Use this to coordinate multiple oscillators or sync to external signal.

        Args:
            target_phase: Target phase in radians
        """
        self._phase = target_phase % (2.0 * math.pi)

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

        Subclasses should return: 1000.0 / frequency_hz
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


class ThetaOscillator(BrainOscillator):
    """
    Theta oscillation (4-10 Hz, typically ~8 Hz).

    Theta is associated with:
    - Memory encoding and retrieval
    - Spatial navigation
    - Attention and arousal
    - Sequence processing

    In hippocampus, theta coordinates:
    - Place cell firing
    - Phase precession
    - Nested gamma oscillations

    Biological Range: 4-10 Hz (rodents: 6-12 Hz)
    Typical: 8 Hz
    """

    def __init__(
        self,
        frequency_hz: float = 8.0,
        dt_ms: float = 1.0,
        initial_phase: float = 0.0,
        amplitude: float = 1.0,
    ):
        config = OscillatorConfig(
            frequency_hz=frequency_hz,
            dt_ms=dt_ms,
            initial_phase=initial_phase,
            amplitude=amplitude,
        )
        super().__init__(config)

    @property
    def oscillation_period_ms(self) -> float:
        """Theta period in milliseconds (~125ms for 8Hz)."""
        return 1000.0 / self._frequency_hz

    @property
    def signal(self) -> float:
        """Sinusoidal theta oscillation."""
        return math.sin(self._phase) * self.config.amplitude


class GammaOscillatorBase(BrainOscillator):
    """
    Gamma oscillation (30-100 Hz, typically ~40 Hz).

    Gamma is associated with:
    - Feature binding
    - Local processing
    - Attention
    - Consciousness

    Different gamma bands:
    - Slow gamma (30-50 Hz): Encoding
    - Fast gamma (60-100 Hz): Retrieval

    In cortex, gamma couples with theta:
    - Nested within theta cycles
    - Phase-amplitude coupling
    - ~5-8 gamma cycles per theta

    Biological Range: 30-100 Hz
    Typical: 40 Hz
    """

    def __init__(
        self,
        frequency_hz: float = 40.0,
        dt_ms: float = 1.0,
        initial_phase: float = 0.0,
        amplitude: float = 1.0,
    ):
        config = OscillatorConfig(
            frequency_hz=frequency_hz,
            dt_ms=dt_ms,
            initial_phase=initial_phase,
            amplitude=amplitude,
        )
        super().__init__(config)

    @property
    def oscillation_period_ms(self) -> float:
        """Gamma period in milliseconds (~25ms for 40Hz)."""
        return 1000.0 / self._frequency_hz

    @property
    def signal(self) -> float:
        """Sinusoidal gamma oscillation."""
        return math.sin(self._phase) * self.config.amplitude


class AlphaOscillator(BrainOscillator):
    """
    Alpha oscillation (8-13 Hz, typically ~10 Hz).

    Alpha is associated with:
    - Inhibitory control
    - Attention gating
    - Sensory suppression
    - Idle/resting state

    High alpha typically indicates:
    - Reduced processing
    - Inhibition of task-irrelevant areas
    - Relaxed alertness

    Biological Range: 8-13 Hz
    Typical: 10 Hz
    """

    def __init__(
        self,
        frequency_hz: float = 10.0,
        dt_ms: float = 1.0,
        initial_phase: float = 0.0,
        amplitude: float = 1.0,
    ):
        config = OscillatorConfig(
            frequency_hz=frequency_hz,
            dt_ms=dt_ms,
            initial_phase=initial_phase,
            amplitude=amplitude,
        )
        super().__init__(config)

    @property
    def oscillation_period_ms(self) -> float:
        """Alpha period in milliseconds (~100ms for 10Hz)."""
        return 1000.0 / self._frequency_hz

    @property
    def signal(self) -> float:
        """Sinusoidal alpha oscillation."""
        return math.sin(self._phase) * self.config.amplitude


class BetaOscillator(BrainOscillator):
    """
    Beta oscillation (13-30 Hz, typically ~20 Hz).

    Beta is associated with:
    - Motor control and planning
    - Active thinking
    - Focus and concentration
    - Decision making

    High beta can indicate:
    - Anxiety/stress
    - Active problem solving
    - Motor preparation

    Biological Range: 13-30 Hz
    Typical: 20 Hz
    """

    def __init__(
        self,
        frequency_hz: float = 20.0,
        dt_ms: float = 1.0,
        initial_phase: float = 0.0,
        amplitude: float = 1.0,
    ):
        config = OscillatorConfig(
            frequency_hz=frequency_hz,
            dt_ms=dt_ms,
            initial_phase=initial_phase,
            amplitude=amplitude,
        )
        super().__init__(config)

    @property
    def oscillation_period_ms(self) -> float:
        """Beta period in milliseconds (~50ms for 20Hz)."""
        return 1000.0 / self._frequency_hz

    @property
    def signal(self) -> float:
        """Sinusoidal beta oscillation."""
        return math.sin(self._phase) * self.config.amplitude


# Convenience functions

def create_oscillator(
    oscillator_type: str,
    frequency_hz: Optional[float] = None,
    **kwargs,
) -> BrainOscillator:
    """
    Factory function to create oscillators by type name.

    Args:
        oscillator_type: Type of oscillator ('theta', 'gamma', 'alpha', 'beta')
        frequency_hz: Optional frequency override
        **kwargs: Additional config arguments

    Returns:
        Oscillator instance

    Example:
        ```python
        theta = create_oscillator('theta', frequency_hz=8.0)
        gamma = create_oscillator('gamma', dt_ms=0.5)
        ```
    """
    oscillators = {
        'theta': ThetaOscillator,
        'gamma': GammaOscillatorBase,
        'alpha': AlphaOscillator,
        'beta': BetaOscillator,
    }

    if oscillator_type not in oscillators:
        raise ValueError(
            f"Unknown oscillator type: {oscillator_type}. "
            f"Choose from: {list(oscillators.keys())}"
        )

    osc_class = oscillators[oscillator_type]

    # Use frequency_hz if provided, otherwise use class default
    if frequency_hz is not None:
        kwargs['frequency_hz'] = frequency_hz

    return osc_class(**kwargs)
