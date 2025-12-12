"""
Stage Configurations for Curriculum Training

Provides pre-configured StageConfig instances for all curriculum stages,
including critical period domain mappings based on curriculum_strategy.md.

Usage:
    from thalia.training.curriculum.stage_configs import get_stage_config

    config = get_stage_config(CurriculumStage.SENSORIMOTOR)
    trainer.train_stage(stage, config, task_loader)
"""

from dataclasses import replace
from thalia.training.curriculum.stage_manager import StageConfig, TaskConfig
from thalia.config.curriculum_growth import CurriculumStage


# =============================================================================
# Stage -0.5: Sensorimotor Grounding (Week 0-4)
# =============================================================================

def get_sensorimotor_config(duration_steps: int = 50000) -> StageConfig:
    """Configure Stage -0.5 (Sensorimotor Grounding).

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
            'motor_control': TaskConfig(weight=0.40, difficulty=0.5),
            'reaching': TaskConfig(weight=0.35, difficulty=0.6),
            'manipulation': TaskConfig(weight=0.20, difficulty=0.7),
            'prediction': TaskConfig(weight=0.05, difficulty=0.5),
        },

        # Success criteria
        success_criteria={
            'motor_control_accuracy': 0.95,
            'reaching_accuracy': 0.90,
            'manipulation_success': 0.85,
            'prediction_error': 0.05,
            'stable_firing_rates': True,
        },

        # Critical period domain mappings
        enable_critical_periods=True,
        domain_mappings={
            'motor_control': ['motor'],
            'reaching': ['motor', 'face_recognition'],  # Visual targeting
            'manipulation': ['motor'],
            'prediction': ['motor'],
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


# =============================================================================
# Stage 0: Sensory Foundations + Phonology (Week 4-8)
# =============================================================================

def get_phonology_config(duration_steps: int = 50000) -> StageConfig:
    """Configure Stage 0 (Sensory Foundations + Phonology).

    Focus: Multi-modal sensory integration, phoneme discrimination
    Critical periods: phonology (PEAK), face_recognition (PEAK), motor (declining)

    Args:
        duration_steps: Training duration (default: 50k steps = 1 month)

    Returns:
        StageConfig for phonology training
    """
    return StageConfig(
        duration_steps=duration_steps,

        # Task mixing ratios (interleaved multi-modal)
        task_configs={
            # Multi-modal sensory (70% total)
            'visual_mnist': TaskConfig(weight=0.20, difficulty=0.5),
            'visual_shapes': TaskConfig(weight=0.10, difficulty=0.5),
            'temporal_sequences': TaskConfig(weight=0.15, difficulty=0.6),
            'audio_phonemes': TaskConfig(weight=0.25, difficulty=0.6),

            # Social referencing (30%)
            'gaze_following': TaskConfig(weight=0.15, difficulty=0.5),
            'joint_attention': TaskConfig(weight=0.15, difficulty=0.6),
        },

        # Success criteria
        success_criteria={
            'mnist_accuracy': 0.95,
            'temporal_prediction': 0.90,
            'phoneme_discrimination': 0.90,
            'categorical_perception': True,  # Sharp boundaries
            'gaze_following_accuracy': 0.80,
            'stable_firing_rates': True,
        },

        # Critical period domain mappings
        enable_critical_periods=True,
        domain_mappings={
            # Visual tasks
            'visual_mnist': ['face_recognition'],
            'visual_shapes': ['face_recognition'],

            # Temporal/semantic groundwork
            'temporal_sequences': ['semantics'],

            # Phonological tasks (PEAK WINDOW!)
            'audio_phonemes': ['phonology'],

            # Social learning
            'gaze_following': ['face_recognition'],
            'joint_attention': ['face_recognition', 'semantics'],
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


# =============================================================================
# Stage 1: Object Permanence & Working Memory (Week 8-16)
# =============================================================================

def get_toddler_config(duration_steps: int = 100000) -> StageConfig:
    """Configure Stage 1 (Toddler Brain - Object Permanence & Working Memory).

    Focus: Working memory, language foundations, social learning
    Critical periods: grammar (starting), phonology (declining), semantics (early)

    Args:
        duration_steps: Training duration (default: 100k steps = 2 months)

    Returns:
        StageConfig for toddler training
    """
    return StageConfig(
        duration_steps=duration_steps,

        # Task mixing ratios
        task_configs={
            # Object recognition (Week 8-10, 25%)
            'cifar10_recognition': TaskConfig(weight=0.15, difficulty=0.6),
            'object_description': TaskConfig(weight=0.10, difficulty=0.7),

            # Working memory (Week 9-11, 25%)
            'n_back_1': TaskConfig(weight=0.10, difficulty=0.5),
            'n_back_2': TaskConfig(weight=0.10, difficulty=0.7),
            'delayed_match': TaskConfig(weight=0.05, difficulty=0.6),

            # Social learning (Week 10-11.5, 20%)
            'imitation_learning': TaskConfig(weight=0.10, difficulty=0.6),
            'pedagogy_detection': TaskConfig(weight=0.05, difficulty=0.7),
            'social_referencing': TaskConfig(weight=0.05, difficulty=0.6),

            # Bilingual language (Week 11.5-13, 25%)
            'word_recognition_en': TaskConfig(weight=0.08, difficulty=0.6),
            'word_recognition_de': TaskConfig(weight=0.05, difficulty=0.6),
            'simple_commands_en': TaskConfig(weight=0.07, difficulty=0.7),
            'simple_commands_de': TaskConfig(weight=0.05, difficulty=0.7),

            # Metacognition (Week 12-13, 5%)
            'uncertainty_abstention': TaskConfig(weight=0.05, difficulty=0.7),
        },

        # Success criteria
        success_criteria={
            'cifar10_accuracy': 0.70,
            'object_description': 0.60,
            'n_back_2_accuracy': 0.80,
            'imitation_accuracy': 0.85,
            'joint_attention': 0.80,
            'bilingual_commands': 0.85,
            'phonological_mapping': 0.80,
            'abstention_accuracy': 0.70,
        },

        # Critical period domain mappings
        enable_critical_periods=True,
        domain_mappings={
            # Visual tasks
            'cifar10_recognition': ['face_recognition'],
            'object_description': ['face_recognition', 'semantics'],

            # Working memory (semantic groundwork)
            'n_back_1': ['semantics'],
            'n_back_2': ['semantics'],
            'delayed_match': ['semantics'],

            # Social learning
            'imitation_learning': ['motor', 'semantics'],
            'pedagogy_detection': ['semantics'],
            'social_referencing': ['face_recognition', 'semantics'],

            # Language (phonology declining, grammar starting)
            'word_recognition_en': ['phonology', 'grammar', 'semantics'],
            'word_recognition_de': ['phonology', 'grammar', 'semantics'],
            'simple_commands_en': ['grammar', 'semantics'],
            'simple_commands_de': ['grammar', 'semantics'],

            # Metacognition
            'uncertainty_abstention': ['semantics'],
        },

        # Curriculum parameters
        interleaved_practice=True,
        spaced_repetition=True,
        testing_frequency=0.15,
        productive_failure_steps=8000,

        # Growth and consolidation
        enable_growth=True,
        growth_check_interval=5000,
        consolidation_interval=10000,

        # Checkpointing
        checkpoint_interval=5000,
    )


# =============================================================================
# Stage 2: Grammar & Composition (Week 16-30)
# =============================================================================

def get_grammar_config(duration_steps: int = 150000) -> StageConfig:
    """Configure Stage 2 (Grammar & Composition).

    Focus: Trilingual grammar, compositional semantics, executive function
    Critical periods: grammar (PEAK), semantics (PEAK)

    Args:
        duration_steps: Training duration (default: 150k steps = 3.5 months)

    Returns:
        StageConfig for grammar training
    """
    return StageConfig(
        duration_steps=duration_steps,

        # Task mixing ratios
        task_configs={
            # Trilingual grammar (40%)
            'grammar_en': TaskConfig(weight=0.15, difficulty=0.6),
            'grammar_de': TaskConfig(weight=0.10, difficulty=0.6),
            'grammar_es': TaskConfig(weight=0.08, difficulty=0.7),  # New language
            'code_switching': TaskConfig(weight=0.07, difficulty=0.8),

            # Compositional semantics (30%)
            'noun_verb_binding': TaskConfig(weight=0.10, difficulty=0.6),
            'simple_sentences': TaskConfig(weight=0.10, difficulty=0.7),
            'semantic_roles': TaskConfig(weight=0.10, difficulty=0.7),

            # Executive function: Set shifting (20%)
            'dccs_task': TaskConfig(weight=0.10, difficulty=0.7),
            'task_switching': TaskConfig(weight=0.10, difficulty=0.7),

            # Cross-modal binding (10%)
            'visual_auditory_binding': TaskConfig(weight=0.10, difficulty=0.7),
        },

        # Success criteria
        success_criteria={
            'trilingual_grammar': 0.80,
            'compositional_accuracy': 0.75,
            'code_switching_fluency': 0.70,
            'set_shifting_accuracy': 0.80,
            'cross_modal_binding': 0.75,
        },

        # Critical period domain mappings
        enable_critical_periods=True,
        domain_mappings={
            # Grammar tasks (PEAK WINDOW!)
            'grammar_en': ['grammar', 'semantics'],
            'grammar_de': ['grammar', 'semantics'],
            'grammar_es': ['grammar', 'semantics'],
            'code_switching': ['grammar', 'semantics'],

            # Compositional semantics (PEAK WINDOW!)
            'noun_verb_binding': ['grammar', 'semantics'],
            'simple_sentences': ['grammar', 'semantics'],
            'semantic_roles': ['grammar', 'semantics'],

            # Executive function
            'dccs_task': ['semantics'],
            'task_switching': ['semantics'],

            # Cross-modal
            'visual_auditory_binding': ['semantics'],
        },

        # Curriculum parameters
        interleaved_practice=True,
        spaced_repetition=True,
        testing_frequency=0.15,
        productive_failure_steps=10000,

        # Growth and consolidation
        enable_growth=True,
        growth_check_interval=5000,
        consolidation_interval=10000,

        # Checkpointing
        checkpoint_interval=5000,
    )


# =============================================================================
# Stage 3: Reading & Planning (Week 30-46)
# =============================================================================

def get_reading_config(duration_steps: int = 150000) -> StageConfig:
    """Configure Stage 3 (Reading & Planning).

    Focus: Reading comprehension, planning, theory of mind
    Critical periods: semantics (late PEAK), grammar (declining)

    Args:
        duration_steps: Training duration (default: 150k steps = 3.5 months)

    Returns:
        StageConfig for reading training
    """
    return StageConfig(
        duration_steps=duration_steps,

        # Task mixing ratios
        task_configs={
            # Reading comprehension (40%)
            'passage_reading': TaskConfig(weight=0.15, difficulty=0.7),
            'reading_generation': TaskConfig(weight=0.15, difficulty=0.8),
            'comprehension_questions': TaskConfig(weight=0.10, difficulty=0.7),

            # Executive function: Planning (30%)
            'tower_hanoi': TaskConfig(weight=0.10, difficulty=0.8),
            'subgoal_decomposition': TaskConfig(weight=0.10, difficulty=0.8),
            'sequential_planning': TaskConfig(weight=0.10, difficulty=0.7),

            # Theory of Mind (20%)
            'false_belief': TaskConfig(weight=0.10, difficulty=0.8),
            'perspective_taking': TaskConfig(weight=0.10, difficulty=0.8),

            # Continuous metacognition (10%)
            'confidence_calibration': TaskConfig(weight=0.10, difficulty=0.7),
        },

        # Success criteria
        success_criteria={
            'reading_comprehension': 0.80,
            'reading_generation': 0.70,
            'planning_accuracy': 0.75,
            'theory_of_mind': 0.70,
            'confidence_calibration_ece': 0.15,  # Expected Calibration Error < 0.15
        },

        # Critical period domain mappings
        enable_critical_periods=True,
        domain_mappings={
            # Reading (late semantic window)
            'passage_reading': ['semantics'],
            'reading_generation': ['grammar', 'semantics'],
            'comprehension_questions': ['semantics'],

            # Planning
            'tower_hanoi': ['semantics'],
            'subgoal_decomposition': ['semantics'],
            'sequential_planning': ['semantics'],

            # Theory of Mind
            'false_belief': ['semantics'],
            'perspective_taking': ['semantics'],

            # Metacognition
            'confidence_calibration': ['semantics'],
        },

        # Curriculum parameters
        interleaved_practice=True,
        spaced_repetition=True,
        testing_frequency=0.15,
        productive_failure_steps=10000,

        # Growth and consolidation
        enable_growth=True,
        growth_check_interval=5000,
        consolidation_interval=10000,

        # Checkpointing
        checkpoint_interval=5000,
    )


# =============================================================================
# Stage 4: Abstract Reasoning (Week 46-70)
# =============================================================================

def get_abstract_config(duration_steps: int = 250000) -> StageConfig:
    """Configure Stage 4 (Abstract Reasoning).

    Focus: Fluid reasoning, multi-step inference, metacognitive control
    Critical periods: ALL domains in late/declining phase

    Args:
        duration_steps: Training duration (default: 250k steps = 6 months)

    Returns:
        StageConfig for abstract reasoning training
    """
    return StageConfig(
        duration_steps=duration_steps,

        # Task mixing ratios
        task_configs={
            # Fluid reasoning (35%)
            'ravens_matrices': TaskConfig(weight=0.15, difficulty=0.9),
            'analogical_reasoning': TaskConfig(weight=0.10, difficulty=0.9),
            'pattern_completion': TaskConfig(weight=0.10, difficulty=0.8),

            # Multi-step inference (35%)
            'multi_premise_reasoning': TaskConfig(weight=0.15, difficulty=0.9),
            'hypothesis_testing': TaskConfig(weight=0.10, difficulty=0.9),
            'causal_reasoning': TaskConfig(weight=0.10, difficulty=0.8),

            # Metacognitive control (30%)
            'active_learning': TaskConfig(weight=0.10, difficulty=0.8),
            'curriculum_selection': TaskConfig(weight=0.10, difficulty=0.8),
            'calibration_refinement': TaskConfig(weight=0.10, difficulty=0.7),
        },

        # Success criteria
        success_criteria={
            'ravens_accuracy': 0.75,
            'analogical_reasoning': 0.70,
            'multi_premise_inference': 0.75,
            'hypothesis_testing': 0.70,
            'active_learning_efficiency': 0.80,
            'calibration_ece': 0.10,  # Expected Calibration Error < 0.10
        },

        # Critical period domain mappings
        # Note: All domains in late/declining phase at this stage
        enable_critical_periods=True,
        domain_mappings={
            # Fluid reasoning (all late semantics)
            'ravens_matrices': ['semantics'],
            'analogical_reasoning': ['semantics'],
            'pattern_completion': ['semantics'],

            # Multi-step inference
            'multi_premise_reasoning': ['grammar', 'semantics'],
            'hypothesis_testing': ['semantics'],
            'causal_reasoning': ['semantics'],

            # Metacognitive control
            'active_learning': ['semantics'],
            'curriculum_selection': ['semantics'],
            'calibration_refinement': ['semantics'],
        },

        # Curriculum parameters
        interleaved_practice=True,
        spaced_repetition=True,
        testing_frequency=0.15,
        productive_failure_steps=15000,

        # Growth and consolidation
        enable_growth=True,
        growth_check_interval=5000,
        consolidation_interval=10000,

        # Checkpointing
        checkpoint_interval=5000,
    )


# =============================================================================
# Factory Function
# =============================================================================

def get_stage_config(
    stage: CurriculumStage,
    duration_steps: int | None = None,
    **overrides
) -> StageConfig:
    """Get configuration for a curriculum stage.

    Args:
        stage: Which curriculum stage
        duration_steps: Override default duration (optional)
        **overrides: Override any config fields

    Returns:
        StageConfig for the specified stage

    Example:
        # Use defaults
        config = get_stage_config(CurriculumStage.PHONOLOGY)

        # Override duration
        config = get_stage_config(CurriculumStage.PHONOLOGY, duration_steps=100000)

        # Override specific fields
        config = get_stage_config(
            CurriculumStage.GRAMMAR,
            enable_critical_periods=False,  # Disable for ablation
            testing_frequency=0.20,
        )
    """
    # Map stages to config functions
    stage_configs = {
        CurriculumStage.SENSORIMOTOR: get_sensorimotor_config,
        CurriculumStage.PHONOLOGY: get_phonology_config,
        CurriculumStage.TODDLER: get_toddler_config,
        CurriculumStage.GRAMMAR: get_grammar_config,
        CurriculumStage.READING: get_reading_config,
        CurriculumStage.ABSTRACT: get_abstract_config,
    }

    if stage not in stage_configs:
        raise ValueError(f"Unknown stage: {stage}")

    # Get base config
    config_fn = stage_configs[stage]
    if duration_steps is not None:
        config = config_fn(duration_steps=duration_steps)
    else:
        config = config_fn()

    # Apply overrides
    if overrides:
        config = replace(config, **overrides)

    return config


__all__ = [
    'get_sensorimotor_config',
    'get_phonology_config',
    'get_toddler_config',
    'get_grammar_config',
    'get_reading_config',
    'get_abstract_config',
    'get_stage_config',
]
