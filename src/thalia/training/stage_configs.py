"""
Stage Configurations for Curriculum Training

Provides pre-configured StageConfig instances for all curriculum stages,
including critical period domain mappings based on curriculum_strategy.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .curriculum_trainer import TaskConfig


# =============================================================================
# StageConfig DataClass
# =============================================================================


@dataclass
class StageConfig:
    """Configuration for training a single curriculum stage."""

    # Duration
    duration_steps: int = 50000  # Total training steps

    # Task configuration
    task_configs: Dict[str, TaskConfig] = field(default_factory=dict)

    # Success criteria (milestone evaluation)
    success_criteria: Dict[str, float] = field(default_factory=dict)

    # Critical period configuration
    enable_critical_periods: bool = True  # Apply critical period gating
    domain_mappings: Dict[str, List[str]] = field(default_factory=dict)  # task_name -> domains

    # Curriculum parameters
    interleaved_practice: bool = True  # Mix tasks within stage
    spaced_repetition: bool = True  # Review based on Leitner algorithm
    testing_frequency: float = 0.15  # Fraction of steps for retrieval practice
    productive_failure_steps: int = 5000  # Initial struggle period

    # Review from previous stages
    review_stages: Dict[int, float] = field(default_factory=dict)  # {stage: weight}

    # Consolidation
    consolidation_interval: int = 10000  # Steps between consolidation
    consolidation_cycles: int = 5  # Replay cycles per consolidation

    # Growth
    enable_growth: bool = True  # Allow neuron addition
    growth_check_interval: int = 5000  # Steps between capacity checks

    # Checkpointing
    checkpoint_interval: int = 5000  # Steps between checkpoints

    # Stage-specific settings
    stage_specific: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Stage 0: Bootstrap & Developmental Initialization
# =============================================================================


def get_bootstrap_config(
    duration_steps: int = 100000,
    phase: str = "all"
) -> StageConfig:
    """Configure Stage 0 (Bootstrap & Developmental Initialization).

    Focus: Establish functional connectivity before task learning begins
    Critical periods: ALL at elevated levels (developmental plasticity)

    This stage simulates prenatal/early postnatal development to solve the
    bootstrap problem: getting learning started when initial connections are weak.

    Phases:
        - "0A" or "spontaneous": Spontaneous activity only (no external input)
        - "0B" or "patterns": Simple pattern exposure (2-4 patterns)
        - "0C" or "transition": Parameter transition to adult levels
        - "all": Run all three phases sequentially (default)

    Args:
        duration_steps: Training duration (default: 100k steps = 2 weeks)
                       Phase breakdown if "all":
                       - Phase 0A: 40k steps (40%)
                       - Phase 0B: 40k steps (40%)
                       - Phase 0C: 20k steps (20%)
        phase: Which phase to run ("all", "0A"/"spontaneous", "0B"/"patterns", "0C"/"transition")

    Returns:
        StageConfig for bootstrap training
    """
    if phase in ("all", "0A", "spontaneous"):
        # Phase 0A: Spontaneous Activity
        # No external input, just noise-driven spontaneous bursts
        return StageConfig(
            duration_steps=int(duration_steps * 0.4) if phase == "all" else duration_steps,
            # Task mixing (100% spontaneous activity)
            task_configs={
                "spontaneous_activity": TaskConfig(weight=1.0, difficulty=0.0),
            },
            # Success criteria
            success_criteria={
                "all_regions_firing": True,  # 0.05-0.15 range
                "no_silent_regions": True,  # >0.01 minimum
                "no_runaway": True,  # <0.5 maximum
                "weights_stable": True,  # Thalamus→L4 >0.35
            },
            # Critical periods: All elevated (developmental)
            enable_critical_periods=True,
            domain_mappings={
                "spontaneous_activity": ["motor", "phonology", "face_recognition",
                                        "grammar", "semantics"],
            },
            # Curriculum parameters
            interleaved_practice=False,  # Single task
            spaced_repetition=False,
            testing_frequency=0.0,
            productive_failure_steps=0,
            # NO growth during bootstrap
            enable_growth=False,
            growth_check_interval=10000,
            consolidation_interval=20000,
            # Checkpointing
            checkpoint_interval=5000,
            # Bootstrap-specific settings
            stage_specific={
                "phase": "0A",
                "elevated_plasticity": True,
                "elevated_homeostasis": True,
                "no_weight_decay": True,
                "background_excitation": 0.10,
                "noise_level": 0.05,  # OU noise sigma
            },
        )

    elif phase in ("0B", "patterns"):
        # Phase 0B: Simple Pattern Exposure
        return StageConfig(
            duration_steps=int(duration_steps * 0.4) if phase == "all" else duration_steps,
            # Task mixing (simple patterns)
            task_configs={
                "single_pixel": TaskConfig(weight=0.50, difficulty=0.1),  # Simplest
                "pure_tone": TaskConfig(weight=0.50, difficulty=0.1),
            },
            # Success criteria
            success_criteria={
                "cortex_response_rate": 0.90,  # Fires on >90% of presentations
                "pattern_discrimination": 0.85,  # Can tell A from B
                "weight_strengthening": True,  # Thalamus→L4 >0.42
                "spontaneous_maintained": True,  # Still fires without input
                "stable_firing_rates": True,
            },
            # Critical periods: All elevated
            enable_critical_periods=True,
            domain_mappings={
                "single_pixel": ["face_recognition", "motor"],
                "pure_tone": ["phonology"],
            },
            # Curriculum parameters
            interleaved_practice=True,
            spaced_repetition=False,  # Too early
            testing_frequency=0.0,
            productive_failure_steps=0,
            # NO growth during bootstrap
            enable_growth=False,
            growth_check_interval=10000,
            consolidation_interval=20000,
            # Checkpointing
            checkpoint_interval=5000,
            # Bootstrap-specific
            stage_specific={
                "phase": "0B",
                "elevated_plasticity": True,
                "elevated_homeostasis": True,
                "reduced_weight_decay": True,
                "background_excitation": 0.10,
                "pattern_repetitions": 100,
            },
        )

    elif phase in ("0C", "transition"):
        # Phase 0C: Parameter Transition
        return StageConfig(
            duration_steps=int(duration_steps * 0.2) if phase == "all" else duration_steps,
            # Task mixing (slightly more complex)
            task_configs={
                "single_pixel": TaskConfig(weight=0.25, difficulty=0.1),
                "pure_tone": TaskConfig(weight=0.25, difficulty=0.1),
                "two_pixels": TaskConfig(weight=0.25, difficulty=0.2),
                "tone_pair": TaskConfig(weight=0.25, difficulty=0.2),
            },
            # Success criteria
            success_criteria={
                "stable_after_transition": True,  # 0.05-0.15 maintained
                "weights_stable": True,  # Thalamus→L4 >0.35
                "pattern_discrimination": 0.80,  # 4 patterns
                "no_instabilities": True,  # No runaway or silence
            },
            # Critical periods: Gradually declining
            enable_critical_periods=True,
            domain_mappings={
                "single_pixel": ["face_recognition"],
                "pure_tone": ["phonology"],
                "two_pixels": ["face_recognition"],
                "tone_pair": ["phonology"],
            },
            # Curriculum parameters
            interleaved_practice=True,
            spaced_repetition=False,
            testing_frequency=0.0,
            productive_failure_steps=0,
            # NO growth during bootstrap
            enable_growth=False,
            growth_check_interval=10000,
            consolidation_interval=20000,
            # Checkpointing
            checkpoint_interval=5000,
            # Bootstrap-specific
            stage_specific={
                "phase": "0C",
                "parameter_transition": True,  # Gradual decay to adult params
                "transition_steps": duration_steps if phase != "all" else int(duration_steps * 0.2),
            },
        )

    else:
        raise ValueError(f"Unknown bootstrap phase: {phase}. Use 'all', '0A', '0B', '0C', "
                        "'spontaneous', 'patterns', or 'transition'")


# =============================================================================
# Stage 1: Sensorimotor Grounding
# =============================================================================


def get_sensorimotor_config(duration_steps: int = 50000) -> StageConfig:
    """Configure Stage 1 (Sensorimotor Grounding).

    Focus: Motor control, visual-motor coordination, object manipulation
    Critical periods: motor (PEAK), face_recognition (PEAK)

    Args:
        duration_steps: Training duration (default: 50k steps = 1 month)

    Returns:
        StageConfig for sensorimotor training
    """
    return StageConfig(
        duration_steps=duration_steps,
        # Task mixing ratios
        task_configs={
            "motor_control": TaskConfig(weight=0.40, difficulty=0.5),
            "reaching": TaskConfig(weight=0.35, difficulty=0.6),
            "manipulation": TaskConfig(weight=0.20, difficulty=0.7),
            "prediction": TaskConfig(weight=0.05, difficulty=0.5),
        },
        # Success criteria
        success_criteria={
            "motor_control_accuracy": 0.95,
            "reaching_accuracy": 0.90,
            "manipulation_success": 0.85,
            "prediction_error": 0.05,
            "stable_firing_rates": True,
        },
        # Critical period domain mappings
        enable_critical_periods=True,
        domain_mappings={
            "motor_control": ["motor"],
            "reaching": ["motor", "face_recognition"],  # Visual targeting
            "manipulation": ["motor"],
            "prediction": ["motor"],
        },
        # Curriculum parameters
        interleaved_practice=True,
        spaced_repetition=True,
        testing_frequency=0.15,
        productive_failure_steps=5000,
        # Growth and consolidation
        enable_growth=True,
        growth_check_interval=5000,
        consolidation_interval=10000,
        # Checkpointing
        checkpoint_interval=5000,
    )
