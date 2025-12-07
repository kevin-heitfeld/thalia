"""
Neuromodulator Mixin for Brain Regions.

This mixin provides standardized neuromodulator handling (dopamine, acetylcholine,
norepinephrine) for brain regions, eliminating duplicate decay logic and providing
consistent interfaces.

Design Pattern:
===============
Instead of implementing custom neuromodulator methods in each region:
1. Inherit from NeuromodulatorMixin
2. Access neuromodulators via self.state.dopamine, etc.
3. Call self.decay_neuromodulators(dt_ms) in forward()
4. Optionally override DEFAULT_TAU_* constants for region-specific timescales

This consolidates:
- Exponential decay logic with configurable time constants
- set_dopamine() interface for external modulation
- get_effective_learning_rate() for dopamine-modulated plasticity
- set_neuromodulator() for generic access

Hybrid Decay Architecture:
==========================
**Dopamine - Centralized in Brain**:
- Brain computes RPE and manages tonic/phasic dopamine
- Brain decays phasic dopamine (τ=200ms) in _update_tonic_dopamine()
- Brain broadcasts combined dopamine to all regions via set_dopamine()
- Regions DON'T decay dopamine locally (Brain handles it)
- Rationale: Dopamine is global signal from VTA/SNc

**Acetylcholine & Norepinephrine - Local Decay**:
- Regions call self.decay_neuromodulators(dt) in their forward() methods
- ACh/NE decay locally with their own time constants
- Dopamine is NOT decayed (already handled by Brain)
- Rationale: ACh (nucleus basalis) and NE (locus coeruleus) have regional specificity

Biological Basis:
=================
Neuromodulators gate synaptic plasticity and influence neural dynamics:

- **Dopamine (DA)**: Reward prediction error, gates learning
  - High DA → consolidate current patterns (LTP enhancement)
  - Low DA → exploratory, reduce learning
  - Tau ~200ms (reuptake by DAT transporters)
  - Primary targets: Striatum, PFC, Hippocampus

- **Acetylcholine (ACh)**: Attention, novelty detection, encoding
  - High ACh → enhance sensory processing, suppress recurrence
  - Low ACh → retrieval mode, enhance consolidation
  - Tau ~50ms (rapid degradation by AChE)
  - Primary targets: Cortex, Hippocampus

- **Norepinephrine (NE)**: Arousal, flexibility, network gain
  - High NE → increase neural gain, reset dynamics
  - Modulates signal-to-noise ratio
  - Tau ~100ms (reuptake by NET transporters)
  - Widespread targets: Cortex, Hippocampus, PFC

Usage Example:
==============
    class MyRegion(NeuromodulatorMixin, BrainRegion):
        # Override tau constants if region-specific
        DEFAULT_ACETYLCHOLINE_TAU_MS = 30.0  # Faster ACh in this region
        
        def forward(self, input, dt=1.0):
            output = self._compute_output(input)
            
            # Decay ACh/NE locally (dopamine managed by Brain)
            self.decay_neuromodulators(dt_ms=dt)
            
            # Use dopamine-modulated learning rate
            lr = self.get_effective_learning_rate(base_lr=0.01)
            self._apply_plasticity(lr=lr)
            
            return output

Author: Thalia Project
Date: December 2025
"""

import math
from typing import Optional


class NeuromodulatorMixin:
    """Mixin providing standardized neuromodulator handling for brain regions.
    
    This mixin assumes the class has a `self.state` attribute with:
    - state.dopamine: float
    - state.acetylcholine: float
    - state.norepinephrine: float
    
    These are initialized in RegionState (base.py).
    """
    
    # Default time constants (can be overridden in subclasses)
    DEFAULT_DOPAMINE_TAU_MS: float = 200.0
    DEFAULT_ACETYLCHOLINE_TAU_MS: float = 50.0
    DEFAULT_NOREPINEPHRINE_TAU_MS: float = 100.0
    
    def set_dopamine(self, level: float) -> None:
        """Set dopamine level (modulates plasticity rate).
        
        Args:
            level: Dopamine level, typically in [-1, 1].
                   Positive = reward, consolidate current patterns
                   Negative = punishment, reduce current patterns
                   Zero = baseline learning rate
        """
        self.state.dopamine = level
    
    def set_acetylcholine(self, level: float) -> None:
        """Set acetylcholine level (modulates attention/encoding).
        
        Args:
            level: ACh level, typically in [0, 1].
                   High = encoding mode, enhance sensory processing
                   Low = retrieval mode, suppress interference
        """
        self.state.acetylcholine = level
    
    def set_norepinephrine(self, level: float) -> None:
        """Set norepinephrine level (modulates arousal/gain).
        
        Args:
            level: NE level, typically in [0, 1].
                   High = arousal, increase neural gain
                   Low = baseline gain
        """
        self.state.norepinephrine = level
    
    def set_neuromodulator(self, name: str, level: float) -> None:
        """Generic setter for any neuromodulator.
        
        Args:
            name: Neuromodulator name ('dopamine', 'acetylcholine', 'norepinephrine')
            level: Level to set
            
        Raises:
            ValueError: If name is not a recognized neuromodulator
        """
        valid_names = {'dopamine', 'acetylcholine', 'norepinephrine'}
        if name not in valid_names:
            raise ValueError(
                f"Unknown neuromodulator '{name}'. "
                f"Valid names: {valid_names}"
            )
        setattr(self.state, name, level)
    
    def decay_neuromodulators(
        self,
        dt_ms: float = 1.0,
        dopamine_tau_ms: Optional[float] = None,
        acetylcholine_tau_ms: Optional[float] = None,
        norepinephrine_tau_ms: Optional[float] = None,
    ) -> None:
        """Decay neuromodulator levels toward baseline (zero).
        
        Call this at each timestep for realistic dynamics. Uses exponential
        decay: level(t+dt) = level(t) * exp(-dt/tau).
        
        Args:
            dt_ms: Time step in milliseconds
            dopamine_tau_ms: Dopamine decay time constant (default: DEFAULT_DOPAMINE_TAU_MS)
            acetylcholine_tau_ms: ACh decay time constant (default: DEFAULT_ACETYLCHOLINE_TAU_MS)
            norepinephrine_tau_ms: NE decay time constant (default: DEFAULT_NOREPINEPHRINE_TAU_MS)
        
        Note:
            Subclasses can override DEFAULT_TAU_* class attributes for region-specific
            decay rates, or pass custom tau values to this method.
        """
        # Use defaults if not specified
        dopamine_tau_ms = dopamine_tau_ms or self.DEFAULT_DOPAMINE_TAU_MS
        acetylcholine_tau_ms = acetylcholine_tau_ms or self.DEFAULT_ACETYLCHOLINE_TAU_MS
        norepinephrine_tau_ms = norepinephrine_tau_ms or self.DEFAULT_NOREPINEPHRINE_TAU_MS
        
        # Exponential decay toward zero
        self.state.dopamine *= math.exp(-dt_ms / dopamine_tau_ms)
        self.state.acetylcholine *= math.exp(-dt_ms / acetylcholine_tau_ms)
        self.state.norepinephrine *= math.exp(-dt_ms / norepinephrine_tau_ms)
    
    def get_effective_learning_rate(
        self, 
        base_lr: Optional[float] = None,
        dopamine_sensitivity: float = 1.0,
    ) -> float:
        """Compute learning rate modulated by dopamine.
        
        The effective learning rate is:
            base_lr * (1 + dopamine_sensitivity * dopamine)
        
        This means:
            - dopamine = 0: baseline learning
            - dopamine = 1: (1 + sensitivity)x learning rate (strong consolidation)
            - dopamine = -0.5: (1 - 0.5*sensitivity)x learning rate (reduced learning)
            - dopamine = -1: (1 - sensitivity)x learning rate (suppressed if sensitivity=1)
        
        Args:
            base_lr: Base learning rate (uses self.base_learning_rate if None)
            dopamine_sensitivity: How much dopamine affects learning (0-1)
                                  1.0 = full modulation, 0.0 = no modulation
        
        Returns:
            Modulated learning rate
        """
        if base_lr is None:
            base_lr = getattr(self, 'base_learning_rate', 0.01)
        
        modulation = 1.0 + dopamine_sensitivity * self.state.dopamine
        # Clamp to non-negative (can't have negative learning rate)
        modulation = max(0.0, modulation)
        
        return base_lr * modulation
    
    def get_neuromodulator_state(self) -> dict:
        """Get current neuromodulator levels for diagnostics.
        
        Returns:
            Dict with dopamine, acetylcholine, norepinephrine levels
        """
        return {
            'dopamine': self.state.dopamine,
            'acetylcholine': self.state.acetylcholine,
            'norepinephrine': self.state.norepinephrine,
        }
