"""Nucleus Basalis (NB) - Acetylcholine Attention, Encoding, and Novelty System.

The nucleus basalis is the brain's primary acetylcholine (ACh) source for cortex
and hippocampus, modulating attention, encoding strength, and novelty detection
in response to prediction errors.

This module implements centralized ACh management following the same architectural
pattern as VTADopamineSystem and LocusCoeruleusSystem.

**Biological Background**:
==========================
The nucleus basalis (specifically nucleus basalis of Meynert in basal forebrain):

1. **Projection pattern**: Widespread to cortex and hippocampus (selective innervation)
2. **Trigger**: Releases ACh in response to novelty/prediction error
3. **Mode switching**: Controls encoding vs retrieval dynamics
4. **Learning coordination**: Synchronizes cortex-hippocampus encoding
5. **Attention**: Critical for selective attention and learning

**Key Functions**:
==================

**ENCODING MODE** (High ACh):
  - Strong sensory processing (enhance feedforward input)
  - Suppress recurrent connections (reduce interference)
  - New memory formation (hippocampal encoding)
  - Enhanced plasticity (faster learning)

**RETRIEVAL MODE** (Low ACh):
  - Pattern completion (enhance recurrence)
  - Memory recall (hippocampal retrieval)
  - Consolidation (strengthen existing traces)
  - Reduced plasticity (protect memories)

**NOVELTY DETECTION**:
  - Prediction error → ACh burst → encoding boost
  - Familiar stimuli → low ACh → retrieval mode
  - Adaptive: Learn from surprising events

**ATTENTION GATING**:
  - High ACh increases signal-to-noise for attended features
  - Modulates cortical gain (amplifies relevant signals)
  - Suppresses irrelevant information

**Architecture Pattern**:
=========================
Follows centralized broadcast pattern:

1. Brain creates ``NucleusBasalisSystem`` (like VTA/LC/OscillatorManager)
2. System computes ACh from prediction error each timestep
3. ACh broadcast to all regions (like dopamine/NE/oscillators)
4. Regions use ACh for encoding/retrieval gating (optional receptor density)

**Biological References**:
==========================
- Hasselmo & McGaughy (2004): "High ACh sets circuit dynamics for attention and
  encoding; low ACh sets dynamics for consolidation"
- Hasselmo (2006): "The role of acetylcholine in learning and memory"
- Parikh et al. (2007): "Prefrontal acetylcholine release controls cue detection"

**Author**: Thalia Project
**Date**: December 2025
"""

from dataclasses import dataclass
from typing import Optional
import math
from thalia.neuromodulation.homeostasis import NeuromodulatorHomeostasis, NeuromodulatorHomeostasisConfig


@dataclass
class NucleusBasalisConfig:
    """Configuration for nucleus basalis acetylcholine system.

    Parameters control ACh dynamics and encoding mode computation.
    """
    # Baseline ACh level (retrieval mode)
    # Low ACh → pattern completion, memory recall
    baseline_ach: float = 0.2

    # ACh decay (cholinesterase breakdown)
    # τ = 50ms → decay = exp(-dt/τ) ≈ 0.98 per ms (faster than DA/NE)
    ach_decay_per_ms: float = 0.98

    # Encoding mode threshold
    # ACh > threshold → encoding mode
    # ACh < threshold → retrieval mode
    encoding_threshold: float = 0.5

    # Novelty response parameters
    # How strongly prediction errors trigger ACh release
    novelty_gain: float = 2.0

    # Learning rate for baseline ACh tracking
    # Slow adaptation to task demands
    baseline_lr: float = 0.001

    # ACh saturation limits
    min_ach: float = 0.0
    max_ach: float = 1.0

    # Homeostatic regulation (optional, can provide custom config)
    homeostatic_config: Optional[NeuromodulatorHomeostasisConfig] = None


class NucleusBasalisSystem:
    """Nucleus Basalis - Centralized acetylcholine management.

    Computes global ACh level from prediction errors and broadcasts to
    cortex and hippocampus for encoding/retrieval coordination.

    Usage:
        # In Brain.__init__
        self.nucleus_basalis = NucleusBasalisSystem()

        # Each timestep in Brain
        prediction_error = self._compute_prediction_error()
        self.nucleus_basalis.update(dt_ms=1.0, prediction_error=prediction_error)

        # Broadcast to regions
        ach = self.nucleus_basalis.get_acetylcholine()
        self.cortex.impl.set_neuromodulators(acetylcholine=ach)
        self.hippocampus.impl.set_neuromodulators(acetylcholine=ach)
    """

    def __init__(self, config: Optional[NucleusBasalisConfig] = None):
        """Initialize nucleus basalis system.

        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self.config = config or NucleusBasalisConfig()

        # State variables
        self._baseline_ach: float = self.config.baseline_ach
        self._phasic_ach: float = 0.0  # Transient bursts from novelty
        self._global_ach: float = self.config.baseline_ach
        self._prediction_error: float = 0.0  # Last PE for diagnostics

        # Homeostatic regulation
        homeostatic_cfg = self.config.homeostatic_config or NeuromodulatorHomeostasisConfig(target_level=0.3)
        self._homeostatic = NeuromodulatorHomeostasis(config=homeostatic_cfg)

    def update(self, dt_ms: float, prediction_error: float) -> None:
        """Update ACh levels for this timestep.

        Args:
            dt_ms: Time step in milliseconds
            prediction_error: Current prediction error magnitude [0, ∞)
                             High PE → novelty → ACh burst → encoding mode
        """
        # Store PE for diagnostics
        self._prediction_error = prediction_error

        # Update baseline ACh (slow adaptation to task demands)
        # High average PE → increase baseline (task requires encoding)
        # Low average PE → decrease baseline (task requires retrieval)
        target_baseline = min(1.0, prediction_error)
        self._baseline_ach += self.config.baseline_lr * (target_baseline - self._baseline_ach)

        # Compute phasic ACh from prediction error
        # High PE → ACh burst → encoding mode
        phasic_target = self.config.novelty_gain * prediction_error
        phasic_target = min(self.config.max_ach - self._baseline_ach, phasic_target)

        # Decay phasic toward zero (rapid cholinesterase breakdown)
        # τ = 50ms → fast decay
        decay_factor = math.exp(-dt_ms * math.log(1.0 / self.config.ach_decay_per_ms))
        self._phasic_ach *= decay_factor

        # Add new phasic component
        self._phasic_ach += (1.0 - decay_factor) * phasic_target

        # Compute global ACh (baseline + phasic)
        self._global_ach = self._baseline_ach + self._phasic_ach

        # Clamp to valid range
        self._global_ach = max(
            self.config.min_ach,
            min(self.config.max_ach, self._global_ach)
        )

        # Update homeostatic regulation
        self._homeostatic.update(self._global_ach)

    def trigger_attention(self, magnitude: float = 0.5) -> None:
        """Explicitly trigger ACh burst for attentional event.

        Use this for explicit attention cues (e.g., task-relevant stimulus).

        Args:
            magnitude: Burst magnitude [0, 1], default 0.5
        """
        self._phasic_ach = max(self._phasic_ach, magnitude)

    def get_acetylcholine(self, apply_homeostasis: bool = True) -> float:
        """Get current global ACh level for broadcast to regions.

        Args:
            apply_homeostasis: If True, apply receptor sensitivity scaling

        Returns:
            Global ACh level [0, 1]
        """
        ach = max(
            self.config.min_ach,
            min(self.config.max_ach, self._global_ach)
        )
        if apply_homeostasis:
            return self._homeostatic.apply_sensitivity(ach)
        return ach

    def get_baseline_ach(self) -> float:
        """Get baseline (tonic) ACh level.

        Returns:
            Baseline ACh [0, 1]
        """
        return self._baseline_ach

    def get_phasic_ach(self) -> float:
        """Get phasic (burst) ACh component.

        Returns:
            Phasic ACh [0, 1]
        """
        return self._phasic_ach

    def is_encoding_mode(self) -> bool:
        """Check if currently in encoding mode (high ACh).

        Returns:
            True if ACh > encoding threshold (encoding mode)
            False if ACh < encoding threshold (retrieval mode)
        """
        return self._global_ach > self.config.encoding_threshold

    def get_encoding_strength(self) -> float:
        """Get encoding strength [0, 1] based on ACh level.

        Returns:
            Encoding strength: 0 = pure retrieval, 1 = strong encoding
        """
        # Map ACh to [0, 1] with threshold as midpoint
        if self._global_ach < self.config.encoding_threshold:
            # Below threshold → retrieval mode (0 to 0.5)
            return 0.5 * (self._global_ach / self.config.encoding_threshold)
        else:
            # Above threshold → encoding mode (0.5 to 1.0)
            excess = self._global_ach - self.config.encoding_threshold
            max_excess = self.config.max_ach - self.config.encoding_threshold
            return 0.5 + 0.5 * (excess / max_excess)

    def get_prediction_error(self) -> float:
        """Get last prediction error (for diagnostics).

        Returns:
            Last PE value used for update
        """
        return self._prediction_error

    def get_state(self) -> dict:
        """Get complete state for checkpointing.

        Returns:
            State dictionary with all internal variables
        """
        return {
            "baseline_ach": self._baseline_ach,
            "phasic_ach": self._phasic_ach,
            "global_ach": self._global_ach,
            "prediction_error": self._prediction_error,
            "homeostatic": self._homeostatic.get_state(),
            "config": {
                "baseline_ach": self.config.baseline_ach,
                "ach_decay_per_ms": self.config.ach_decay_per_ms,
                "encoding_threshold": self.config.encoding_threshold,
                "novelty_gain": self.config.novelty_gain,
                "baseline_lr": self.config.baseline_lr,
                "min_ach": self.config.min_ach,
                "max_ach": self.config.max_ach,
            },
        }

    def set_state(self, state: dict) -> None:
        """Restore state from checkpoint.

        Args:
            state: State dictionary from get_state()
        """
        self._baseline_ach = state["baseline_ach"]
        self._phasic_ach = state["phasic_ach"]
        self._global_ach = state["global_ach"]
        self._prediction_error = state.get("prediction_error", 0.0)

        if 'homeostatic' in state:
            self._homeostatic.set_state(state['homeostatic'])

        # Config is immutable after initialization, but validate compatibility
        config_state = state.get("config", {})
        if config_state:
            # Warn if config doesn't match (but don't fail)
            if abs(self.config.baseline_ach - config_state["baseline_ach"]) > 1e-6:
                import warnings
                warnings.warn("Loaded NB config differs from current config")

    def reset_state(self) -> None:
        """Reset to initial state."""
        self._baseline_ach = self.config.baseline_ach
        self._phasic_ach = 0.0
        self._global_ach = self.config.baseline_ach
        self._prediction_error = 0.0
        self._homeostatic.reset()

    def check_health(self) -> dict:
        """Check system health and return diagnostic info.

        Returns:
            Health check dictionary with status and warnings
        """
        issues = []
        warnings = []

        # Check for NaN/Inf
        if not math.isfinite(self._global_ach):
            issues.append(f"Global ACh is not finite: {self._global_ach}")

        if not math.isfinite(self._baseline_ach):
            issues.append(f"Baseline ACh is not finite: {self._baseline_ach}")

        # Check for saturation
        if self._global_ach >= self.config.max_ach * 0.99:
            warnings.append("ACh saturated at maximum")

        if self._global_ach <= self.config.min_ach * 1.01:
            warnings.append("ACh at minimum (retrieval mode)")

        # Check if baseline is drifting
        if self._baseline_ach > 0.8:
            warnings.append("Baseline ACh very high (constant encoding mode?)")

        homeostatic_health = self._homeostatic.check_health()
        is_healthy = len(issues) == 0 and homeostatic_health['is_healthy']

        return {
            "is_healthy": is_healthy,
            "issues": issues + homeostatic_health['issues'],
            "warnings": warnings + homeostatic_health['warnings'],
            "global_ach": self._global_ach,
            "baseline_ach": self._baseline_ach,
            "phasic_ach": self._phasic_ach,
            "encoding_mode": self.is_encoding_mode(),
            "encoding_strength": self.get_encoding_strength(),
            "receptor_sensitivity": homeostatic_health['sensitivity'],
        }
