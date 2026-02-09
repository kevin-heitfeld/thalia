"""
Homeostatic Regulation for Neuromodulator Systems.

**Scope**: Global neuromodulator baseline control (dopamine, acetylcholine, norepinephrine)
**Focus**: Receptor sensitivity adaptation to maintain dynamic range

Provides common utilities for receptor sensitivity adaptation and system
coordination to prevent saturation and maintain biological plausibility.

Related Homeostasis Modules:
=============================
This module is ONE of several homeostatic mechanisms in Thalia:

1. **neuromodulation/homeostasis.py** (THIS MODULE):
   - Global neuromodulator baseline regulation
   - Receptor sensitivity adaptation (downregulation/upregulation)
   - System coordination (DA-ACh, NE-ACh interactions)

2. **learning/homeostasis/homeostatic_regulation.py**:
   - DUPLICATE of this module (legacy, will be consolidated)
   - Same receptor sensitivity logic

3. **learning/homeostasis/synaptic_homeostasis.py**:
   - Synaptic weight normalization and scaling
   - Constraint-based weight budgets (sum-to-constant)
   - Per-neuron competitive normalization

4. **learning/homeostasis/intrinsic_plasticity.py**:
   - Neuron excitability adaptation (threshold modulation)
   - Activity-dependent firing rate homeostasis
   - Non-synaptic memory via threshold changes

5. **learning/homeostasis/metabolic.py**:
   - Energy-based constraints (ATP costs)
   - Sparsity pressure through metabolic penalties
   - Global/per-region energy budgets

6. **regions/*/homeostasis_component.py**:
   - Region-specific integration of above mechanisms
   - Coordinates multiple homeostasis types for that region

**When to Use This Module**:
- Need to regulate global neuromodulator baseline levels
- Prevent receptor saturation from sustained high/low levels
- Model drug tolerance or sensitization effects

**When to Use Other Modules**:
- Synaptic weight stability → learning/homeostasis/synaptic_homeostasis.py
- Firing rate stability → learning/homeostasis/intrinsic_plasticity.py
- Energy constraints → learning/homeostasis/metabolic.py
- Region integration → regions/*/homeostasis_component.py

Biological Background:
======================
Homeostatic plasticity maintains neural circuit stability by:
1. **Receptor Downregulation**: High sustained neuromodulator → fewer/less sensitive receptors
2. **Receptor Upregulation**: Low sustained neuromodulator → more/more sensitive receptors
3. **Prevents Saturation**: Maintains dynamic range despite varying input statistics
4. **Slow Timescale**: Adaptation occurs over ~1000 timesteps (seconds to minutes)

This is analogous to:
- Synaptic scaling in cortex
- Receptor trafficking in addiction
- Homeostatic control of excitability

System Coordination:
====================
Neuromodulators interact in specific ways:
1. **DA-ACh**: High dopamine + low ACh suppresses encoding (reward without novelty)
2. **NE-ACh**: Moderate arousal optimal for encoding (inverted-U function)
3. **DA-NE**: High uncertainty with reward → enhanced learning

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class NeuromodulatorHomeostasisConfig:
    """Configuration for homeostatic receptor sensitivity adaptation.

    Controls how receptor sensitivity adapts to sustained neuromodulator levels.
    """

    # Adaptation timescale (exponential smoothing)
    # tau = 0.999 → ~1000 timesteps to adapt
    tau: float = 0.999

    # Receptor sensitivity bounds
    min_sensitivity: float = 0.5  # Maximum downregulation (50% of baseline)
    max_sensitivity: float = 1.5  # Maximum upregulation (150% of baseline)

    # Target average level (what system tries to maintain)
    target_level: float = 0.5

    # Adaptation rate (how strongly to correct deviations)
    adaptation_strength: float = 0.1


class NeuromodulatorHomeostasis:
    """Manages homeostatic receptor sensitivity adaptation.

    Usage:
        # Create regulator for dopamine system
        regulator = NeuromodulatorHomeostasis(config=NeuromodulatorHomeostasisConfig(target_level=0.0))

        # Each timestep, update with current neuromodulator level
        regulator.update(current_level=0.8)

        # Get effective neuromodulator level (scaled by sensitivity)
        effective_da = regulator.apply_sensitivity(raw_da)

        # Or get sensitivity directly for manual scaling
        sensitivity = regulator.get_sensitivity()

    Biological Mechanism:
        When dopamine is consistently high:
        1. Receptors downregulate (fewer D1/D2 receptors)
        2. Sensitivity decreases (existing receptors less responsive)
        3. Same dopamine level → smaller neural response
        4. System maintains dynamic range

        When dopamine is consistently low:
        1. Receptors upregulate
        2. Sensitivity increases
        3. Small dopamine levels → larger neural response
    """

    def __init__(self, config: Optional[NeuromodulatorHomeostasisConfig] = None):
        """Initialize homeostatic regulator.

        Args:
            config: Homeostatic configuration (uses defaults if None)
        """
        self.config = config or NeuromodulatorHomeostasisConfig()  # State
        self._receptor_sensitivity: float = 1.0  # Current sensitivity [0.5, 1.5]
        self._avg_level: float = self.config.target_level  # Running average of signal
        self._update_count: int = 0  # Number of updates (for diagnostics)

    def update(self, current_level: float) -> None:
        """Update receptor sensitivity based on current neuromodulator level.

        Call this every timestep with the current neuromodulator level.
        Sensitivity adapts slowly to prevent saturation.

        Args:
            current_level: Current neuromodulator level (any scale)
        """
        self._update_count += 1

        # Update running average of neuromodulator level (slow EMA)
        tau = self.config.tau
        self._avg_level = tau * self._avg_level + (1 - tau) * abs(current_level)

        # Compute target sensitivity to maintain dynamic range
        # If avg_level > target → decrease sensitivity (downregulate)
        # If avg_level < target → increase sensitivity (upregulate)
        deviation = self._avg_level - self.config.target_level
        target_sensitivity = 1.0 - self.config.adaptation_strength * deviation

        # Clamp to physiological range
        target_sensitivity = max(
            self.config.min_sensitivity, min(self.config.max_sensitivity, target_sensitivity)
        )

        # Update sensitivity slowly (to avoid oscillations)
        sensitivity_alpha = 0.01  # Very slow sensitivity changes
        self._receptor_sensitivity = (
            1 - sensitivity_alpha
        ) * self._receptor_sensitivity + sensitivity_alpha * target_sensitivity

    def apply_sensitivity(self, signal: float) -> float:
        """Apply receptor sensitivity to neuromodulator signal.

        This is what regions actually experience after receptor adaptation.

        Args:
            signal: Raw neuromodulator level

        Returns:
            Effective signal after sensitivity scaling
        """
        return signal * self._receptor_sensitivity

    def get_sensitivity(self) -> float:
        """Get current receptor sensitivity.

        Returns:
            Sensitivity multiplier [0.5, 1.5]
        """
        return self._receptor_sensitivity

    def get_avg_level(self) -> float:
        """Get running average of neuromodulator level.

        Returns:
            Average level tracked by homeostatic system
        """
        return self._avg_level

    def check_health(self) -> dict:
        """Check homeostatic regulator health.

        Returns:
            Health status dictionary
        """
        issues = []
        warnings = []

        # Check for abnormal sensitivity
        if self._receptor_sensitivity < 0.6:
            warnings.append(
                f"Low sensitivity (receptors downregulated): {self._receptor_sensitivity:.2f}"
            )
        if self._receptor_sensitivity > 1.4:
            warnings.append(
                f"High sensitivity (receptors upregulated): {self._receptor_sensitivity:.2f}"
            )

        # Check for extreme average levels
        if self._avg_level > 2.0:
            issues.append(f"Excessive neuromodulator levels: {self._avg_level:.2f}")

        return {
            "is_healthy": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "sensitivity": self._receptor_sensitivity,
            "avg_level": self._avg_level,
            "updates": self._update_count,
        }


class NeuromodulatorCoordination:
    """Coordinates interactions between neuromodulator systems.

    Implements biologically-observed interactions:
    1. DA-ACh: Dopamine modulates acetylcholine release
    2. NE-ACh: Arousal modulates encoding strength
    3. DA-NE: Reward and uncertainty interact for learning
    """

    @staticmethod
    def coordinate_da_ach(dopamine: float, acetylcholine: float, strength: float = 0.3) -> float:
        """Coordinate dopamine and acetylcholine.

        Biological mechanism:
        - High DA (reward) + low ACh (no novelty) → suppress encoding
        - High DA + high ACh → enhance consolidation
        - VTA dopamine inhibits cholinergic interneurons in striatum

        Args:
            dopamine: Current dopamine level [-2, 2]
            acetylcholine: Current ACh level [0, 1]
            strength: Coordination strength [0, 1], default 0.3

        Returns:
            Modulated ACh level
        """
        # High positive DA suppresses ACh (reward without novelty)
        if dopamine > 0.5:
            suppression = strength * (dopamine - 0.5) / 1.5
            return acetylcholine * (1.0 - suppression)

        return acetylcholine

    @staticmethod
    def coordinate_ne_ach(
        norepinephrine: float, acetylcholine: float, optimal_arousal: float = 0.5
    ) -> float:
        """Coordinate norepinephrine and acetylcholine.

        Biological mechanism:
        - Moderate arousal optimal for encoding (inverted-U)
        - Too low NE → insufficient attention
        - Too high NE → stress disrupts encoding
        - LC and NB interact via basal forebrain circuits

        Args:
            norepinephrine: Current NE level [0, 2]
            acetylcholine: Current ACh level [0, 1]
            optimal_arousal: Optimal NE for encoding, default 0.5

        Returns:
            Modulated ACh level with arousal effects
        """
        # Inverted-U: encoding best at moderate arousal
        arousal_factor = 1.0 - abs(norepinephrine - optimal_arousal) / 2.0
        arousal_factor = max(0.5, min(1.5, arousal_factor))

        return acetylcholine * arousal_factor

    @staticmethod
    def coordinate_da_ne(
        dopamine: float, norepinephrine: float, prediction_error: float
    ) -> tuple[float, float]:
        """Coordinate dopamine and norepinephrine for learning.

        Biological mechanism:
        - High uncertainty + reward → enhanced learning (explore-exploit)
        - DA and NE both elevated → attention + motivation
        - VTA and LC weakly coupled via ventral tegmental nucleus

        Args:
            dopamine: Current DA level [-2, 2]
            norepinephrine: Current NE level [0, 2]
            prediction_error: Current PE magnitude [0, ∞)

        Returns:
            Tuple of (modulated_da, modulated_ne)
        """
        # High PE with high arousal → boost both systems
        if prediction_error > 0.5 and norepinephrine > 0.5:
            boost = 0.2 * min(prediction_error, 1.0)
            modulated_da = dopamine + boost
            modulated_ne = norepinephrine + boost
            return (modulated_da, modulated_ne)

        return (dopamine, norepinephrine)


def inverted_u_function(x: float, peak: float = 0.5, width: float = 0.5) -> float:
    """Inverted-U (Gaussian-like) function for optimal arousal effects.

    Common in neuroscience (Yerkes-Dodson law):
    - Too little arousal → poor performance
    - Optimal arousal → best performance
    - Too much arousal → stress, poor performance

    Args:
        x: Input value (e.g., arousal level)
        peak: Location of peak performance
        width: Width of the curve (smaller → steeper)

    Returns:
        Output in [0, 1] with peak at x=peak
    """
    deviation = (x - peak) / width
    return math.exp(-0.5 * deviation * deviation)
