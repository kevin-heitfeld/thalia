"""
Curriculum-Aware Growth Configuration

Integrates GrowthManager with curriculum stages, providing stage-specific
growth triggers and expansion parameters.

Key Concepts:
============

1. STAGE-SPECIFIC GROWTH:
   - Different capacity thresholds per stage
   - Varied expansion rates matching developmental needs
   - Conservative early, aggressive mid-stage

2. COMPONENT-WISE TRIGGERS:
   - Regions can grow at different rates
   - Critical regions (PFC, hippocampus) grow more
   - Sensory regions grow conservatively

3. CONSOLIDATION COORDINATION:
   - Growth triggers consolidation
   - Consolidate before AND after growth
   - Prevents catastrophic forgetting

4. METRIC-BASED DECISIONS:
   - Weight saturation (80%+)
   - Firing rate (90%+)
   - Performance plateau detection

Usage:
======
```python
growth_config = CurriculumGrowthConfig()
trainer = CurriculumTrainer(brain, growth_config=growth_config)

# During training
if trainer.should_trigger_growth(region, stage):
    trainer.trigger_growth(region)
```

Author: Thalia Team
Date: December 8, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional

from thalia.constants.training import AttentionStage


class CurriculumStage(IntEnum):
    """Curriculum stages matching main training plan."""

    SENSORIMOTOR = -1  # Stage -0.5 (motor control)
    PHONOLOGY = 0  # Stage 0 (phonological learning)
    TODDLER = 1  # Stage 1 (first words, joint attention)
    GRAMMAR = 2  # Stage 2 (grammar, composition)
    READING = 3  # Stage 3 (reading, planning)
    ABSTRACT = 4  # Stage 4 (abstract reasoning)


def get_attention_stage_for_curriculum(curriculum_stage: CurriculumStage) -> AttentionStage:
    """Map curriculum stage to attention developmental stage.

    Controls the balance between bottom-up (reactive) and top-down (goal-directed)
    attention across development.

    Args:
        curriculum_stage: Current curriculum training stage

    Returns:
        Corresponding AttentionStage enum
    """
    mapping = {
        CurriculumStage.SENSORIMOTOR: AttentionStage.INFANT,  # Pure bottom-up
        CurriculumStage.PHONOLOGY: AttentionStage.INFANT,  # Pure bottom-up
        CurriculumStage.TODDLER: AttentionStage.TODDLER,  # 70% bottom-up
        CurriculumStage.GRAMMAR: AttentionStage.PRESCHOOL,  # 50/50 balanced
        CurriculumStage.READING: AttentionStage.SCHOOL_AGE,  # 70% top-down
        CurriculumStage.ABSTRACT: AttentionStage.SCHOOL_AGE,  # 70% top-down
    }
    return mapping[curriculum_stage]


@dataclass
class GrowthTriggerConfig:
    """Configuration for a single growth trigger.

    Attributes:
        capacity_threshold: Trigger when capacity exceeds this (0-1)
        expansion_rate: How much to grow (0-1, fraction of current size)
        min_steps_between: Minimum training steps between growth events
        consolidate_before: Whether to consolidate before growth
        consolidate_after: Whether to consolidate after growth
        enabled: Whether this trigger is active
    """

    capacity_threshold: float = 0.80
    expansion_rate: float = 0.20
    min_steps_between: int = 10000
    consolidate_before: bool = True
    consolidate_after: bool = True
    enabled: bool = True


@dataclass
class ComponentGrowthConfig:
    """Growth configuration for a specific component type.

    Different brain regions have different growth needs:
    - Sensory regions: Conservative (recognize more features)
    - PFC: Aggressive (complex rules, large WM)
    - Hippocampus: Moderate (episodic capacity)
    - Striatum: Conservative (action repertoire expansion)
    """

    # Per-stage growth triggers
    stage_triggers: Dict[int, GrowthTriggerConfig] = field(default_factory=dict)

    # Component-specific overrides
    max_total_growth: float = 3.0  # Max 3x original size
    min_neurons_per_growth: int = 10
    max_neurons_per_growth: int = 500

    def get_trigger_for_stage(self, stage: int) -> Optional[GrowthTriggerConfig]:
        """Get growth trigger config for current stage."""
        return self.stage_triggers.get(stage)


@dataclass
class CurriculumGrowthConfig:
    """Complete growth configuration for curriculum training.

    Provides stage-specific and component-specific growth parameters.

    Default Strategy:
    - Stage -1 (Sensorimotor): Moderate growth (35%) at 80% capacity
    - Stage 0 (Phonology): Small growth (15%) - conservative, specialized
    - Stage 1 (Toddler): Large growth (50%) - rapid development
    - Stage 2 (Grammar): Moderate growth (35%) - consolidation phase
    - Stage 3 (Reading): Small growth (20%) - refinement
    - Stage 4 (Abstract): Minimal growth (10%) - mature optimization
    """

    # Global growth settings
    enable_growth: bool = True
    performance_plateau_threshold: float = 0.02  # <2% improvement = plateau
    performance_window_steps: int = 5000  # Window for plateau detection

    # Component-specific configurations
    component_configs: Dict[str, ComponentGrowthConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default configurations for standard components."""
        if not self.component_configs:
            self.component_configs = self._create_default_configs()

    def _create_default_configs(self) -> Dict[str, ComponentGrowthConfig]:
        """Create default growth configurations for brain components."""

        configs = {}

        # =====================================================================
        # PREFRONTAL CORTEX: Aggressive growth (complex rules, large WM)
        # =====================================================================
        configs["prefrontal"] = ComponentGrowthConfig(
            stage_triggers={
                -1: GrowthTriggerConfig(  # Sensorimotor
                    capacity_threshold=0.80,
                    expansion_rate=0.30,
                    min_steps_between=8000,
                ),
                0: GrowthTriggerConfig(  # Phonology
                    capacity_threshold=0.80,
                    expansion_rate=0.20,
                    min_steps_between=10000,
                ),
                1: GrowthTriggerConfig(  # Toddler - BIG GROWTH
                    capacity_threshold=0.75,
                    expansion_rate=0.50,
                    min_steps_between=8000,
                ),
                2: GrowthTriggerConfig(  # Grammar
                    capacity_threshold=0.80,
                    expansion_rate=0.40,
                    min_steps_between=12000,
                ),
                3: GrowthTriggerConfig(  # Reading
                    capacity_threshold=0.85,
                    expansion_rate=0.25,
                    min_steps_between=15000,
                ),
                4: GrowthTriggerConfig(  # Abstract
                    capacity_threshold=0.90,
                    expansion_rate=0.10,
                    min_steps_between=20000,
                ),
            },
            max_total_growth=4.0,  # PFC can grow 4x
            max_neurons_per_growth=500,
        )

        # =====================================================================
        # HIPPOCAMPUS: Moderate growth (episodic memory capacity)
        # =====================================================================
        configs["hippocampus"] = ComponentGrowthConfig(
            stage_triggers={
                -1: GrowthTriggerConfig(
                    capacity_threshold=0.80,
                    expansion_rate=0.25,
                    min_steps_between=10000,
                ),
                0: GrowthTriggerConfig(
                    capacity_threshold=0.85,
                    expansion_rate=0.15,
                    min_steps_between=12000,
                ),
                1: GrowthTriggerConfig(
                    capacity_threshold=0.80,
                    expansion_rate=0.40,
                    min_steps_between=10000,
                ),
                2: GrowthTriggerConfig(
                    capacity_threshold=0.80,
                    expansion_rate=0.35,
                    min_steps_between=12000,
                ),
                3: GrowthTriggerConfig(
                    capacity_threshold=0.85,
                    expansion_rate=0.20,
                    min_steps_between=15000,
                ),
                4: GrowthTriggerConfig(
                    capacity_threshold=0.90,
                    expansion_rate=0.10,
                    min_steps_between=20000,
                ),
            },
            max_total_growth=3.5,
            max_neurons_per_growth=400,
        )

        # =====================================================================
        # CORTEX: Conservative growth (feature detectors)
        # =====================================================================
        configs["cortex"] = ComponentGrowthConfig(
            stage_triggers={
                -1: GrowthTriggerConfig(
                    capacity_threshold=0.85,
                    expansion_rate=0.20,
                    min_steps_between=12000,
                ),
                0: GrowthTriggerConfig(
                    capacity_threshold=0.90,
                    expansion_rate=0.10,
                    min_steps_between=15000,
                ),
                1: GrowthTriggerConfig(
                    capacity_threshold=0.85,
                    expansion_rate=0.30,
                    min_steps_between=12000,
                ),
                2: GrowthTriggerConfig(
                    capacity_threshold=0.85,
                    expansion_rate=0.25,
                    min_steps_between=12000,
                ),
                3: GrowthTriggerConfig(
                    capacity_threshold=0.90,
                    expansion_rate=0.15,
                    min_steps_between=15000,
                ),
                4: GrowthTriggerConfig(
                    capacity_threshold=0.95,
                    expansion_rate=0.05,
                    min_steps_between=25000,
                ),
            },
            max_total_growth=2.5,
            max_neurons_per_growth=300,
        )

        # =====================================================================
        # STRIATUM: Conservative growth (action repertoire)
        # =====================================================================
        configs["striatum"] = ComponentGrowthConfig(
            stage_triggers={
                -1: GrowthTriggerConfig(
                    capacity_threshold=0.80,
                    expansion_rate=0.25,
                    min_steps_between=10000,
                ),
                0: GrowthTriggerConfig(
                    capacity_threshold=0.85,
                    expansion_rate=0.15,
                    min_steps_between=15000,
                ),
                1: GrowthTriggerConfig(
                    capacity_threshold=0.80,
                    expansion_rate=0.35,
                    min_steps_between=10000,
                ),
                2: GrowthTriggerConfig(
                    capacity_threshold=0.85,
                    expansion_rate=0.30,
                    min_steps_between=12000,
                ),
                3: GrowthTriggerConfig(
                    capacity_threshold=0.90,
                    expansion_rate=0.15,
                    min_steps_between=15000,
                ),
                4: GrowthTriggerConfig(
                    capacity_threshold=0.95,
                    expansion_rate=0.05,
                    min_steps_between=20000,
                ),
            },
            max_total_growth=2.5,
            max_neurons_per_growth=250,
        )

        # =====================================================================
        # DEFAULT: For any unspecified component
        # =====================================================================
        configs["default"] = ComponentGrowthConfig(
            stage_triggers={
                -1: GrowthTriggerConfig(capacity_threshold=0.80, expansion_rate=0.30),
                0: GrowthTriggerConfig(capacity_threshold=0.80, expansion_rate=0.15),
                1: GrowthTriggerConfig(capacity_threshold=0.80, expansion_rate=0.50),
                2: GrowthTriggerConfig(capacity_threshold=0.85, expansion_rate=0.35),
                3: GrowthTriggerConfig(capacity_threshold=0.85, expansion_rate=0.20),
                4: GrowthTriggerConfig(capacity_threshold=0.90, expansion_rate=0.10),
            },
            max_total_growth=3.0,
        )

        return configs

    def get_config_for_component(self, component_name: str) -> ComponentGrowthConfig:
        """Get growth configuration for a specific component.

        Args:
            component_name: Name of region or pathway

        Returns:
            ComponentGrowthConfig (uses default if not found)
        """
        # Try exact match
        if component_name in self.component_configs:
            return self.component_configs[component_name]

        # Try partial match (e.g., "cortex_l4" → "cortex")
        for config_name, config in self.component_configs.items():
            if config_name in component_name.lower():
                return config

        # Fall back to default
        return self.component_configs["default"]

    def should_trigger_growth(
        self,
        component_name: str,
        stage: int,
        capacity_metrics: Dict[str, float],
        steps_since_last_growth: int,
        current_size_ratio: float = 1.0,
    ) -> tuple[bool, str]:
        """Determine if growth should be triggered.

        Args:
            component_name: Name of component to check
            stage: Current curriculum stage
            capacity_metrics: Metrics from GrowthManager
            steps_since_last_growth: Training steps since last growth
            current_size_ratio: Current size / original size

        Returns:
            (should_grow, reason): Boolean decision and explanation
        """
        if not self.enable_growth:
            return False, "Growth globally disabled"

        config = self.get_config_for_component(component_name)
        trigger = config.get_trigger_for_stage(stage)

        if trigger is None:
            return False, f"No trigger defined for stage {stage}"

        if not trigger.enabled:
            return False, f"Growth disabled for stage {stage}"

        # Check if already grown too much
        if current_size_ratio >= config.max_total_growth:
            return False, f"Max growth reached ({config.max_total_growth}x)"

        # Check minimum steps between growth
        if steps_since_last_growth < trigger.min_steps_between:
            return False, f"Too soon (need {trigger.min_steps_between} steps)"

        # Check capacity threshold
        capacity = max(
            capacity_metrics.get("weight_saturation", 0.0), capacity_metrics.get("firing_rate", 0.0)
        )

        if capacity < trigger.capacity_threshold:
            return (
                False,
                f"Capacity ({capacity:.2f}) below threshold ({trigger.capacity_threshold})",
            )

        # All checks passed - grow!
        return True, f"Capacity {capacity:.2f} > threshold {trigger.capacity_threshold}"

    def get_expansion_params(
        self,
        component_name: str,
        stage: int,
        current_n_neurons: int,
    ) -> Dict[str, Any]:
        """Get expansion parameters for growth operation.

        Args:
            component_name: Name of component
            stage: Current curriculum stage
            current_n_neurons: Current neuron count

        Returns:
            Dict with expansion parameters
        """
        config = self.get_config_for_component(component_name)
        trigger = config.get_trigger_for_stage(stage)

        if trigger is None:
            # Default conservative
            trigger = GrowthTriggerConfig()

        # Calculate number of neurons to add
        n_new = int(current_n_neurons * trigger.expansion_rate)
        n_new = max(n_new, config.min_neurons_per_growth)
        n_new = min(n_new, config.max_neurons_per_growth)

        return {
            "n_neurons": n_new,
            "expansion_rate": trigger.expansion_rate,
            "consolidate_before": trigger.consolidate_before,
            "consolidate_after": trigger.consolidate_after,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "enable_growth": self.enable_growth,
            "performance_plateau_threshold": self.performance_plateau_threshold,
            "performance_window_steps": self.performance_window_steps,
            "component_configs": {
                name: {
                    "stage_triggers": {
                        stage: {
                            "capacity_threshold": trigger.capacity_threshold,
                            "expansion_rate": trigger.expansion_rate,
                            "min_steps_between": trigger.min_steps_between,
                            "consolidate_before": trigger.consolidate_before,
                            "consolidate_after": trigger.consolidate_after,
                            "enabled": trigger.enabled,
                        }
                        for stage, trigger in config.stage_triggers.items()
                    },
                    "max_total_growth": config.max_total_growth,
                    "min_neurons_per_growth": config.min_neurons_per_growth,
                    "max_neurons_per_growth": config.max_neurons_per_growth,
                }
                for name, config in self.component_configs.items()
            },
        }


# Convenience function for getting standard config
def get_curriculum_growth_config(
    enable_growth: bool = True, conservative: bool = False
) -> CurriculumGrowthConfig:
    """Get standard curriculum growth configuration.

    Args:
        enable_growth: Whether to enable growth
        conservative: If True, use more conservative thresholds

    Returns:
        CurriculumGrowthConfig with standard settings
    """
    config = CurriculumGrowthConfig(enable_growth=enable_growth)

    if conservative:
        # Increase all thresholds by 0.05, decrease all rates by 0.1
        for comp_config in config.component_configs.values():
            for trigger in comp_config.stage_triggers.values():
                trigger.capacity_threshold = min(0.95, trigger.capacity_threshold + 0.05)
                trigger.expansion_rate = max(0.05, trigger.expansion_rate - 0.1)

    return config
