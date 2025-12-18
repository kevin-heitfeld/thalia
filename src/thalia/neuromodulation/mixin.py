"""
Neuromodulator Mixin for Brain Regions.

This mixin provides standardized neuromodulator handling (dopamine, acetylcholine,
norepinephrine) for brain regions, with setter interfaces for centralized broadcast.

Design Pattern:
===============
Neuromodulators are now CENTRALLY MANAGED by Brain and BROADCAST to regions:

1. **VTA (Dopamine)**: Brain.vta computes DA from RPE, broadcasts to all regions
2. **Locus Coeruleus (NE)**: Brain.locus_coeruleus computes NE from uncertainty
3. **Nucleus Basalis (ACh)**: Brain.nucleus_basalis computes ACh from prediction error

Regions:
- Inherit from NeuromodulatorMixin
- Access neuromodulators via self.state.dopamine, etc.
- DO NOT call decay_neuromodulators() (handled by Brain)
- Just use set_dopamine(), set_norepinephrine(), set_acetylcholine() as interfaces

This consolidates:
- Setter interfaces for centralized broadcast
- get_effective_learning_rate() for dopamine-modulated plasticity
- set_neuromodulator() for generic access

**IMPORTANT**: Regions should NOT call decay_neuromodulators() anymore!
All decay is handled centrally by Brain._update_neuromodulators() which:
1. Updates VTA dopamine (tonic + phasic decay)
2. Updates LC norepinephrine (arousal tracking + phasic decay)
3. Updates NB acetylcholine (encoding mode + phasic decay)
4. Broadcasts all three to regions every timestep

**Regional Specificity via Receptor Density**:
==============================================
The architecture reflects neuroanatomical reality:

**Global Projection**:
- Nucleus Basalis projects broadly to cortex and hippocampus
- Locus Coeruleus projects globally throughout entire brain
- VTA projects widely to cortex, striatum, limbic system
- All three are centralized nuclei with diffuse projections

**Regional Variability** comes from receptor density, not signal decay:
- Striatum: High D1/D2 receptor density → strong DA sensitivity
- Cortex: Moderate D1/D2 density → moderate DA sensitivity
- Cerebellum: Low D1/D2 density → weak DA sensitivity
- Hippocampus CA1: High M1/M2 ACh receptors → strong ACh effects
- Hippocampus CA3: Moderate ACh receptors → moderate ACh effects

Components respond differently to the SAME global signal via:
1. **Receptor density parameters**: dopamine_sensitivity, ach_sensitivity
2. **Gating logic**: Hippocampus uses ACh for encode/retrieve modes
3. **Pathway-specific modulation**: Striatum D1 vs D2 pathways respond oppositely

Example - Dopamine sensitivity varies by receptor density:

.. code-block:: python

    # Striatum: High D1/D2 receptor density
    striatum_lr = self.get_effective_learning_rate(
        base_lr=0.01, dopamine_sensitivity=2.0
    )

    # Cortex: Moderate D1/D2 density
    cortex_lr = self.get_effective_learning_rate(
        base_lr=0.01, dopamine_sensitivity=1.0
    )

    # Cerebellum: Low D1/D2 density
    cerebellum_lr = self.get_effective_learning_rate(
        base_lr=0.01, dopamine_sensitivity=0.3
    )

Biological Basis:
=================
Neuromodulators gate synaptic plasticity and influence neural dynamics:

- **Dopamine (DA)**: Reward prediction error, gates learning (VTA/SNc)
  - High DA → consolidate current patterns (LTP enhancement)
  - Low DA → exploratory, reduce learning
  - Tau ~200ms (reuptake by DAT transporters)
  - Managed by: VTADopamineSystem in Brain

- **Acetylcholine (ACh)**: Attention, novelty detection, encoding (Nucleus Basalis)
  - High ACh → enhance sensory processing, encoding mode
  - Low ACh → retrieval mode, enhance consolidation
  - Tau ~50ms (rapid degradation by AChE)
  - Managed by: NucleusBasalisSystem in Brain
  - Projection: Broad to cortex and hippocampus
  - Regional specificity via: Different receptor densities and gating mechanisms

- **Norepinephrine (NE)**: Arousal, flexibility, network gain (Locus Coeruleus)
  - High NE → increase neural gain, reset dynamics
  - Modulates signal-to-noise ratio
  - Tau ~100ms (reuptake by NET transporters)
  - Managed by: LocusCoeruleusSystem in Brain
  - Projection: Global throughout entire brain (one of most widespread systems)
  - Regional specificity via: Different α/β receptor densities and local circuit properties

Usage Example:
==============
    class MyRegion(NeuromodulatorMixin, NeuralComponent):
        def forward(self, input, dt=1.0):
            output = self._compute_output(input)

            # NO LONGER NEEDED - Brain handles decay:
            # self.decay_neuromodulators(dt_ms=dt)  # ❌ Don't do this!

            # Just use dopamine-modulated learning rate:
            lr = self.get_effective_learning_rate(base_lr=0.01)
            self._apply_plasticity(lr=lr)

            # Optionally use ACh for encoding/retrieval mode:
            if self.state.acetylcholine > 0.5:
                # Encoding mode
                pass
            else:
                # Retrieval mode
                pass

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

    These are initialized in NeuralComponentState (base.py).
    """

    # Default time constants (can be overridden in subclasses)
    DEFAULT_DOPAMINE_TAU_MS: float = 200.0
    DEFAULT_ACETYLCHOLINE_TAU_MS: float = 50.0
    DEFAULT_NOREPINEPHRINE_TAU_MS: float = 100.0

    def set_neuromodulators(
        self,
        dopamine: Optional[float] = None,
        norepinephrine: Optional[float] = None,
        acetylcholine: Optional[float] = None
    ) -> None:
        """Set neuromodulator levels atomically (efficient broadcast).

        This consolidated method is more efficient than calling individual setters
        when updating multiple neuromodulators simultaneously (3x reduction in
        function calls and hasattr checks).

        Args:
            dopamine: DA level, typically in [-1, 1].
                      Positive = reward, consolidate current patterns
                      Negative = punishment, reduce current patterns
                      Zero = baseline learning rate
            norepinephrine: NE level, typically in [0, 1].
                           High = arousal, increase neural gain
                           Low = baseline gain
            acetylcholine: ACh level, typically in [0, 1].
                          High = encoding mode, enhance sensory processing
                          Low = retrieval mode, suppress interference

        Note:
            For biological plausibility, all three neuromodulator systems
            should be updated together to maintain consistent brain state.
            This method ensures atomic updates without partial state.
            Any neuromodulator not specified will remain at its current value.
        """
        if dopamine is not None:
            self.state.dopamine = dopamine
        if norepinephrine is not None:
            self.state.norepinephrine = norepinephrine
        if acetylcholine is not None:
            self.state.acetylcholine = acetylcholine

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
