"""
Curriculum-Aware Growth Configuration

Key Concepts:
============

1. STAGE-SPECIFIC GROWTH:
   - Different capacity thresholds per stage
   - Varied expansion rates matching developmental needs
   - Conservative early, aggressive mid-stage

2. REGION-WISE TRIGGERS:
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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .curriculum import CurriculumStage


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
class RegionGrowthConfig:
    """Growth configuration for a specific region.

    Different brain regions have different growth needs:
    - Sensory regions: Conservative (recognize more features)
    - PFC: Aggressive (complex rules, large WM)
    - Hippocampus: Moderate (episodic capacity)
    - Striatum: Conservative (action repertoire expansion)
    """

    # Per-stage growth triggers
    stage_triggers: Dict[int, GrowthTriggerConfig] = field(default_factory=dict)

    # Region-specific overrides
    max_total_growth: float = 3.0  # Max 3x original size
    min_neurons_per_growth: int = 10
    max_neurons_per_growth: int = 500

    def get_trigger_for_stage(self, stage: int) -> Optional[GrowthTriggerConfig]:
        """Get growth trigger config for current stage."""
        return self.stage_triggers.get(stage)


@dataclass
class CurriculumGrowthConfig:
    """Complete growth configuration for curriculum training.

    Provides stage-specific and region-specific growth parameters.

    Default Strategy:
    - Stage 0 (Bootstrap): No growth - focus on stabilizing initial architecture
    - Stage 1 (Sensorimotor): Moderate growth (35%) at 80% capacity
    - Stage 2 (Phonology): Small growth (15%) - conservative, specialized
    """

    # Global growth settings
    enable_growth: bool = True
    performance_plateau_threshold: float = 0.02  # <2% improvement = plateau
    performance_window_steps: int = 5000  # Window for plateau detection

    # Region-specific configurations
    region_configs: Dict[str, RegionGrowthConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default configurations for standard regions."""
        if not self.region_configs:
            self.region_configs = self._create_default_configs()

    def _create_default_configs(self) -> Dict[str, RegionGrowthConfig]:
        """Create default growth configurations for brain regions."""

        configs = {}

        # =====================================================================
        # CORTEX: Conservative growth (feature detectors)
        # =====================================================================
        configs["cortex"] = RegionGrowthConfig(
            stage_triggers={
                CurriculumStage.BOOTSTRAP: GrowthTriggerConfig(
                    capacity_threshold=1.0,  # Disabled
                    expansion_rate=0.0,
                    min_steps_between=999999,
                    enabled=False,
                ),
                CurriculumStage.SENSORIMOTOR: GrowthTriggerConfig(
                    capacity_threshold=0.85,
                    expansion_rate=0.20,
                    min_steps_between=12000,
                ),
            },
            max_total_growth=2.5,
            max_neurons_per_growth=300,
        )

        # =====================================================================
        # HIPPOCAMPUS: Moderate growth (episodic memory capacity)
        # =====================================================================
        configs["hippocampus"] = RegionGrowthConfig(
            stage_triggers={
                CurriculumStage.BOOTSTRAP: GrowthTriggerConfig(
                    capacity_threshold=1.0,  # Disabled
                    expansion_rate=0.0,
                    min_steps_between=999999,
                    enabled=False,
                ),
                CurriculumStage.SENSORIMOTOR: GrowthTriggerConfig(
                    capacity_threshold=0.80,
                    expansion_rate=0.25,
                    min_steps_between=10000,
                ),
            },
            max_total_growth=3.5,
            max_neurons_per_growth=400,
        )

        # =====================================================================
        # PREFRONTAL CORTEX: Aggressive growth (complex rules, large WM)
        # =====================================================================
        configs["prefrontal"] = RegionGrowthConfig(
            stage_triggers={
                CurriculumStage.BOOTSTRAP: GrowthTriggerConfig(
                    capacity_threshold=1.0,  # Disabled
                    expansion_rate=0.0,
                    min_steps_between=999999,
                    enabled=False,
                ),
                CurriculumStage.SENSORIMOTOR: GrowthTriggerConfig(
                    capacity_threshold=0.80,
                    expansion_rate=0.30,
                    min_steps_between=8000,
                ),
            },
            max_total_growth=4.0,  # PFC can grow 4x
            max_neurons_per_growth=500,
        )

        # =====================================================================
        # STRIATUM: Conservative growth (action repertoire)
        # =====================================================================
        configs["striatum"] = RegionGrowthConfig(
            stage_triggers={
                CurriculumStage.BOOTSTRAP: GrowthTriggerConfig(
                    capacity_threshold=1.0,  # Disabled
                    expansion_rate=0.0,
                    min_steps_between=999999,
                    enabled=False,
                ),
                CurriculumStage.SENSORIMOTOR: GrowthTriggerConfig(
                    capacity_threshold=0.80,
                    expansion_rate=0.25,
                    min_steps_between=10000,
                ),
            },
            max_total_growth=2.5,
            max_neurons_per_growth=250,
        )

        # =====================================================================
        # DEFAULT: For any unspecified region
        # =====================================================================
        configs["default"] = RegionGrowthConfig(
            stage_triggers={
                CurriculumStage.BOOTSTRAP: GrowthTriggerConfig(
                    capacity_threshold=1.0, expansion_rate=0.0, enabled=False
                ),
                CurriculumStage.SENSORIMOTOR: GrowthTriggerConfig(capacity_threshold=0.80, expansion_rate=0.30),
            },
            max_total_growth=3.0,
        )

        return configs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "enable_growth": self.enable_growth,
            "performance_plateau_threshold": self.performance_plateau_threshold,
            "performance_window_steps": self.performance_window_steps,
            "region_configs": {
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
                for name, config in self.region_configs.items()
            },
        }


# Convenience function for getting standard config
def get_curriculum_growth_config(enable_growth: bool = True, conservative: bool = False) -> CurriculumGrowthConfig:
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
        for region_config in config.region_configs.values():
            for trigger in region_config.stage_triggers.values():
                trigger.capacity_threshold = min(0.95, trigger.capacity_threshold + 0.05)
                trigger.expansion_rate = max(0.05, trigger.expansion_rate - 0.1)

    return config
