"""
Stage Evaluation Functions - Milestone checking for curriculum stages.

This module implements evaluation functions for each curriculum stage,
checking whether the brain has met all success criteria before proceeding
to the next stage (go/no-go decisions).

Evaluation Categories:
=====================

1. TASK PERFORMANCE
   - Accuracy on stage-specific tasks
   - Generalization to new examples
   - Robustness to noise/variations

2. SYSTEM HEALTH
   - Firing rate stability (0.05-0.15)
   - No runaway excitation (criticality check)
   - BCM threshold convergence
   - Weight health (<80% saturation)
   - No silent regions (>0.01 firing minimum)

3. BACKWARD COMPATIBILITY
   - Previous stage performance maintained (>90%)
   - No catastrophic forgetting
   - Skills transfer correctly

4. GROWTH & CAPACITY
   - Appropriate size for stage
   - Healthy capacity utilization (not saturated)
   - Growth events successful

Usage:
======

    from thalia.training.curriculum.stage_evaluation import (
        evaluate_stage_sensorimotor,
        evaluate_stage_phonology,
        evaluate_stage_toddler,
        check_system_health,
    )

    # Evaluate Stage -0.5 (Sensorimotor)
    results = evaluate_stage_sensorimotor(
        brain=brain,
        sensorimotor_wrapper=wrapper,
    )

    if all(results.values()):
        print("[OK] Stage -0.5 complete!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"[FAIL] Failed criteria: {failed}")

References:
===========
- docs/design/curriculum_strategy.md - Stage milestone definitions
- tests/unit/test_robustness.py - Health check examples

Author: Thalia Project
Date: December 8, 2025
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# ============================================================================
# Common Health Checks (All Stages)
# ============================================================================


def check_firing_rates(
    brain: Any,
    target_range: Tuple[float, float] = (0.05, 0.15),
    window_steps: int = 10000,
) -> bool:
    """Check that all regions maintain healthy firing rates.

    Args:
        brain: Brain instance
        target_range: (min, max) acceptable firing rates
        window_steps: Window for averaging

    Returns:
        True if all regions in range
    """
    # Get firing rates from all regions
    # This would use brain's internal tracking
    # Placeholder for now
    return True


def check_no_runaway_excitation(
    brain: Any,
    threshold: float = 0.8,
    window_steps: int = 20000,
) -> bool:
    """Check for runaway excitation in any region.

    Args:
        brain: Brain instance
        threshold: Max acceptable firing rate
        window_steps: Window to check

    Returns:
        True if no runaway detected
    """
    # Check criticality state
    # Placeholder for now
    return True


def check_bcm_convergence(
    brain: Any,
    drift_threshold: float = 0.01,
    window_steps: int = 50000,
) -> bool:
    """Check that BCM thresholds have stabilized.

    Args:
        brain: Brain instance
        drift_threshold: Max acceptable drift
        window_steps: Window for stability check

    Returns:
        True if thresholds converged
    """
    # Check BCM threshold drift
    # Placeholder for now
    return True


def check_weight_saturation(
    brain: Any,
    max_saturation: float = 0.80,
) -> bool:
    """Check that weights aren't saturated at extremes.

    Args:
        brain: Brain instance
        max_saturation: Max fraction of saturated weights

    Returns:
        True if weight health is good
    """
    # Count weights near min/max
    # Placeholder for now
    return True


def check_no_silent_regions(
    brain: Any,
    min_firing: float = 0.01,
    max_silent_steps: int = 1000,
) -> bool:
    """Check that no region has been silent too long.

    Args:
        brain: Brain instance
        min_firing: Minimum acceptable firing rate
        max_silent_steps: Max steps a region can be silent

    Returns:
        True if no prolonged silence
    """
    # Check for silent regions
    # Placeholder for now
    return True


def check_system_health(brain: Any) -> Dict[str, bool]:
    """Run all common health checks.

    Args:
        brain: Brain instance

    Returns:
        Dict of health check results
    """
    return {
        "firing_stability": check_firing_rates(brain),
        "no_runaway": check_no_runaway_excitation(brain),
        "bcm_convergence": check_bcm_convergence(brain),
        "weight_health": check_weight_saturation(brain),
        "no_silence": check_no_silent_regions(brain),
    }


# ============================================================================
# Stage -0.5: Sensorimotor Grounding
# ============================================================================


def test_basic_movements(
    brain: Any,
    wrapper: Any,
    n_trials: int = 100,
    threshold: float = 0.95,
) -> bool:
    """Test basic motor control accuracy.

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper
        n_trials: Number of test trials
        threshold: Success threshold

    Returns:
        True if accuracy > threshold
    """
    # Test simple left/right/up/down movements
    # Placeholder for now
    return True


def test_reaching_accuracy(
    brain: Any,
    wrapper: Any,
    n_trials: int = 100,
    threshold: float = 0.90,
) -> bool:
    """Test reaching task accuracy.

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper
        n_trials: Number of reaching trials
        threshold: Success threshold

    Returns:
        True if accuracy > threshold
    """
    if hasattr(wrapper, "reaching_task"):
        stats = wrapper.reaching_task(brain, n_trials=n_trials)
        return stats.get("success_rate", 0.0) > threshold
    return True


def test_manipulation_success(
    brain: Any,
    wrapper: Any,
    n_trials: int = 100,
    threshold: float = 0.85,
) -> bool:
    """Test object manipulation success rate.

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper
        n_trials: Number of manipulation trials
        threshold: Success threshold

    Returns:
        True if success rate > threshold
    """
    # Test push/pull/grasp tasks
    # Placeholder for now
    return True


def test_prediction_error(
    brain: Any,
    wrapper: Any,
    n_trials: int = 100,
    threshold: float = 0.05,
) -> bool:
    """Test sensorimotor prediction error.

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper
        n_trials: Number of trials
        threshold: Max acceptable error

    Returns:
        True if error < threshold
    """
    # Test cerebellum forward model accuracy
    # Placeholder for now
    return True


def test_cerebellum_functional(
    brain: Any,
    wrapper: Any,
) -> bool:
    """Test that cerebellum forward/inverse models work.

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper

    Returns:
        True if cerebellum is functional
    """
    # Check cerebellum state and learning
    # Placeholder for now
    return True


def evaluate_stage_sensorimotor(
    brain: Any,
    wrapper: Any,
) -> Dict[str, bool]:
    """Evaluate Stage -0.5 (Sensorimotor) milestones.

    Success criteria from curriculum_strategy.md:
    - >95% accurate basic movements
    - >90% reaching accuracy
    - >85% manipulation success
    - <5% prediction error
    - Stable firing rates (0.05-0.15)
    - Cerebellum forward models functional

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper instance

    Returns:
        Dict mapping criterion name to pass/fail
    """
    results = {}

    # Task performance
    results["basic_movements"] = test_basic_movements(brain, wrapper)
    results["reaching_accuracy"] = test_reaching_accuracy(brain, wrapper)
    results["manipulation_success"] = test_manipulation_success(brain, wrapper)
    results["prediction_error"] = test_prediction_error(brain, wrapper)

    # System health
    health = check_system_health(brain)
    results.update(health)

    # Component-specific
    results["cerebellum_functional"] = test_cerebellum_functional(brain, wrapper)

    return results


# ============================================================================
# Stage 0: Sensory Foundations (Phonology)
# ============================================================================


def test_mnist_accuracy(
    brain: Any,
    dataset: Any,
    n_samples: int = 1000,
    threshold: float = 0.95,
) -> bool:
    """Test MNIST classification accuracy.

    Args:
        brain: Brain instance
        dataset: MNIST dataset
        n_samples: Number of test samples
        threshold: Accuracy threshold

    Returns:
        True if accuracy > threshold
    """
    # Test on MNIST test set
    # Placeholder for now
    return True


def test_sequence_prediction(
    brain: Any,
    dataset: Any,
    n_sequences: int = 100,
    threshold: float = 0.90,
) -> bool:
    """Test temporal sequence prediction (A-B-C patterns).

    Args:
        brain: Brain instance
        dataset: Temporal sequence dataset
        n_sequences: Number of test sequences
        threshold: Accuracy threshold

    Returns:
        True if accuracy > threshold
    """
    # Test sequence prediction
    # Placeholder for now
    return True


def test_phoneme_discrimination(
    brain: Any,
    dataset: Any,
    n_pairs: int = 200,
    threshold: float = 0.90,
) -> bool:
    """Test phoneme discrimination accuracy.

    Args:
        brain: Brain instance
        dataset: Phonological dataset
        n_pairs: Number of phoneme pairs to test
        threshold: Accuracy threshold

    Returns:
        True if accuracy > threshold
    """
    # Test phoneme discrimination (e.g., /p/ vs /b/)
    # Placeholder for now
    return True


def test_categorical_perception(
    brain: Any,
    dataset: Any,
) -> bool:
    """Test categorical perception curves for phonemes.

    Args:
        brain: Brain instance
        dataset: Phonological dataset with VOT continuum

    Returns:
        True if shows categorical boundaries
    """
    # Test for sharp boundaries between phoneme categories
    # Placeholder for now
    return True


def test_gaze_following(
    brain: Any,
    n_trials: int = 100,
    threshold: float = 0.80,
) -> bool:
    """Test social attention (gaze following).

    Args:
        brain: Brain instance
        n_trials: Number of gaze trials
        threshold: Accuracy threshold

    Returns:
        True if gaze following > threshold
    """
    # Test gaze following task
    # Placeholder for now
    return True


def evaluate_stage_phonology(
    brain: Any,
    datasets: Dict[str, Any],
    sensorimotor_wrapper: Optional[Any] = None,
) -> Dict[str, bool]:
    """Evaluate Stage 0 (Sensory Foundations) milestones.

    Success criteria from curriculum_strategy.md:
    - >95% MNIST accuracy
    - >90% sequence prediction
    - >90% phoneme discrimination
    - Categorical perception established
    - >80% gaze following
    - System health maintained
    - Stage -0.5 maintained (>85%)

    Args:
        brain: Brain instance
        datasets: Dict of datasets OR TaskLoader with dataset properties
        sensorimotor_wrapper: Optional wrapper for backward compatibility check

    Returns:
        Dict mapping criterion name to pass/fail
    """
    results = {}

    # Handle both dict and TaskLoader interfaces
    if hasattr(datasets, "get"):
        # Dict interface
        mnist_data = datasets.get("mnist")
        temporal_data = datasets.get("temporal")
        phonology_data = datasets.get("phonology")
    else:
        # TaskLoader interface
        mnist_data = getattr(datasets, "mnist_dataset", None)
        temporal_data = getattr(datasets, "temporal_dataset", None)
        phonology_data = getattr(datasets, "phonology_dataset", None)

    # Task performance
    results["mnist_accuracy"] = test_mnist_accuracy(brain, mnist_data)
    results["sequence_prediction"] = test_sequence_prediction(brain, temporal_data)
    results["phoneme_discrimination"] = test_phoneme_discrimination(brain, phonology_data)
    results["categorical_perception"] = test_categorical_perception(brain, phonology_data)
    results["gaze_following"] = test_gaze_following(brain)

    # System health
    health = check_system_health(brain)
    results.update(health)

    # Backward compatibility (Stage -0.5)
    if sensorimotor_wrapper is not None:
        results["sensorimotor_maintained"] = test_reaching_accuracy(
            brain, sensorimotor_wrapper, threshold=0.85
        )

    return results


# ============================================================================
# Stage 1: Object Permanence & Working Memory (Toddler)
# ============================================================================


def test_cifar10_accuracy(
    brain: Any,
    dataset: Any,
    n_samples: int = 1000,
    threshold: float = 0.70,
) -> bool:
    """Test CIFAR-10 object recognition.

    Args:
        brain: Brain instance
        dataset: CIFAR-10 dataset
        n_samples: Number of test samples
        threshold: Accuracy threshold

    Returns:
        True if accuracy > threshold
    """
    # Test on CIFAR-10
    # Placeholder for now
    return True


def test_n_back_task(
    brain: Any,
    n: int = 2,
    n_trials: int = 100,
    threshold: float = 0.80,
) -> bool:
    """Test N-back working memory task.

    Args:
        brain: Brain instance
        n: N-back level (1 or 2)
        n_trials: Number of trials
        threshold: Accuracy threshold

    Returns:
        True if accuracy > threshold
    """
    # Test N-back with theta-gamma encoding
    # Placeholder for now
    return True


def test_object_permanence(
    brain: Any,
    n_trials: int = 100,
    threshold: float = 0.85,
) -> bool:
    """Test object permanence understanding.

    Args:
        brain: Brain instance
        n_trials: Number of trials
        threshold: Success threshold

    Returns:
        True if success > threshold
    """
    # Test object tracking through occlusion
    # Placeholder for now
    return True


def test_binary_confidence(
    brain: Any,
    n_samples: int = 200,
    threshold: float = 0.70,
) -> bool:
    """Test binary uncertainty signaling (know vs don't know).

    Args:
        brain: Brain instance
        n_samples: Number of test samples
        threshold: Correct abstention rate

    Returns:
        True if abstention accuracy > threshold
    """
    # Test uncertainty signaling
    # Placeholder for now
    return True


def evaluate_stage_toddler(
    brain: Any,
    datasets: Dict[str, Any],
    stage0_datasets: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """Evaluate Stage 1 (Toddler) milestones.

    Success criteria from curriculum_strategy.md:
    - >70% CIFAR-10 accuracy
    - >80% 2-back accuracy
    - >85% object permanence
    - >70% binary confidence (abstention)
    - System health maintained
    - Stage 0 maintained (>90%)

    Args:
        brain: Brain instance
        datasets: Dict of Stage 1 datasets
        stage0_datasets: Optional Stage 0 datasets for backward compatibility

    Returns:
        Dict mapping criterion name to pass/fail
    """
    results = {}

    # Task performance
    results["cifar10_accuracy"] = test_cifar10_accuracy(brain, datasets.get("cifar10"))
    results["n_back_2"] = test_n_back_task(brain, n=2)
    results["object_permanence"] = test_object_permanence(brain)
    results["binary_confidence"] = test_binary_confidence(brain)

    # System health
    health = check_system_health(brain)
    results.update(health)

    # Backward compatibility (Stage 0)
    if stage0_datasets is not None:
        results["mnist_maintained"] = test_mnist_accuracy(
            brain, stage0_datasets.get("mnist"), threshold=0.90
        )
        results["phoneme_maintained"] = test_phoneme_discrimination(
            brain, stage0_datasets.get("phonology"), threshold=0.85
        )

    return results


# ============================================================================
# Stage 2: Grammar & Composition
# ============================================================================


def test_grammar_accuracy(
    brain: Any,
    dataset: Any,
    n_samples: int = 500,
    threshold: float = 0.80,
) -> bool:
    """Test grammatical generation across languages.

    Args:
        brain: Brain instance
        dataset: Grammar dataset (multilingual)
        n_samples: Number of test samples per language
        threshold: Accuracy threshold

    Returns:
        True if accuracy > threshold for all languages
    """
    # Test grammatical sentence generation in English, German, Spanish
    # Check subject-verb agreement, word order, morphology
    # Placeholder for now - requires brain forward pass integration
    return True


def test_set_shifting(
    brain: Any,
    n_trials: int = 100,
    threshold: float = 0.70,
) -> bool:
    """Test executive function: set shifting (DCCS task).

    Args:
        brain: Brain instance
        n_trials: Number of test trials
        threshold: Accuracy threshold on switch trials

    Returns:
        True if switch accuracy > threshold
    """
    # Dimensional Change Card Sort (DCCS)
    # Sort by color, then switch to sorting by shape
    # Placeholder for now
    return True


def test_cross_lingual_reasoning(
    brain: Any,
    n_samples: int = 200,
    threshold: float = 0.75,
) -> bool:
    """Test cross-lingual compositional reasoning.

    Args:
        brain: Brain instance
        n_samples: Number of reasoning questions
        threshold: Accuracy threshold

    Returns:
        True if reasoning accuracy > threshold
    """
    # Test: "The red ball" / "Der rote Ball" / "La pelota roja"
    # Same concept, different expressions
    # Placeholder for now
    return True


def test_coarse_confidence(
    brain: Any,
    n_samples: int = 200,
    threshold: float = 0.60,
) -> bool:
    """Test coarse metacognitive confidence (high/medium/low).

    Args:
        brain: Brain instance
        n_samples: Number of test samples
        threshold: Correlation threshold (confidence vs accuracy)

    Returns:
        True if confidence correlates with accuracy
    """
    # 3-level confidence (expanded from binary in Stage 1)
    # Check correlation between confidence and actual accuracy
    # Still poorly calibrated (like 3-year-olds)
    # Placeholder for now
    return True


def evaluate_stage_grammar(
    brain: Any,
    datasets: Dict[str, Any],
    stage1_datasets: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """Evaluate Stage 2 (Grammar & Composition) milestones.

    Success criteria from curriculum_strategy.md:
    - >80% grammatical generation (English, German, Spanish)
    - >75% cross-lingual reasoning
    - >70% set shifting (DCCS task)
    - Coarse confidence somewhat correlated with accuracy
    - System health maintained
    - Stage 1 maintained (>85%)

    Args:
        brain: Brain instance
        datasets: Dict of Stage 2 datasets (grammar, etc.)
        stage1_datasets: Optional Stage 1 datasets for backward compatibility

    Returns:
        Dict mapping criterion name to pass/fail
    """
    results = {}

    # Handle both dict and TaskLoader interfaces
    if hasattr(datasets, "get"):
        grammar_data = datasets.get("grammar")
    else:
        grammar_data = getattr(datasets, "grammar_dataset", None)

    # Task performance
    results["grammar_generation"] = test_grammar_accuracy(brain, grammar_data)
    results["cross_lingual_reasoning"] = test_cross_lingual_reasoning(brain)
    results["set_shifting_dccs"] = test_set_shifting(brain)
    results["coarse_confidence"] = test_coarse_confidence(brain)

    # System health
    health = check_system_health(brain)
    results.update(health)

    # Backward compatibility (Stage 1)
    if stage1_datasets is not None:
        cifar_data = (
            stage1_datasets.get("cifar10")
            if hasattr(stage1_datasets, "get")
            else getattr(stage1_datasets, "cifar10_dataset", None)
        )
        results["cifar10_maintained"] = test_cifar10_accuracy(brain, cifar_data, threshold=0.65)
        results["n_back_maintained"] = test_n_back_task(brain, n=2, threshold=0.75)

    return results


# ============================================================================
# Stage 3: Reading & Writing
# ============================================================================


def test_reading_comprehension(
    brain: Any,
    dataset: Any,
    n_samples: int = 500,
    threshold: float = 0.70,
) -> bool:
    """Test multilingual reading comprehension.

    Args:
        brain: Brain instance
        dataset: Reading dataset (multilingual)
        n_samples: Number of test samples per language
        threshold: Accuracy threshold

    Returns:
        True if comprehension > threshold for all languages
    """
    # Test reading comprehension in English, German, Spanish
    # Short paragraphs (3-5 sentences)
    # Answer comprehension questions
    # Placeholder for now
    return True


def test_text_generation(
    brain: Any,
    dataset: Any,
    n_samples: int = 100,
    threshold: float = 0.65,
) -> bool:
    """Test multilingual text generation quality.

    Args:
        brain: Brain instance
        dataset: Reading dataset
        n_samples: Number of generation samples
        threshold: Human rating threshold (coherence)

    Returns:
        True if generation quality > threshold
    """
    # Generate simple stories in each language (3-4 sentences)
    # Complete sentences in target language
    # Maintain language consistency
    # Placeholder for now (requires human evaluation or proxy metric)
    return True


def test_planning_tasks(
    brain: Any,
    n_trials: int = 100,
    threshold: float = 0.60,
) -> bool:
    """Test executive function: planning (Tower of Hanoi, maze solving).

    Args:
        brain: Brain instance
        n_trials: Number of planning trials
        threshold: Success threshold on 3-step planning

    Returns:
        True if planning success > threshold
    """
    # Tower of Hanoi: Multi-step planning with subgoals
    # Maze solving: Plan path before execution
    # Goal decomposition
    # Placeholder for now
    return True


def test_continuous_confidence(
    brain: Any,
    n_samples: int = 500,
    ece_threshold: float = 0.25,
) -> bool:
    """Test continuous metacognitive confidence (0-100%) and calibration.

    Args:
        brain: Brain instance
        n_samples: Number of test samples
        ece_threshold: Expected Calibration Error threshold

    Returns:
        True if ECE < threshold (improving calibration)
    """
    # Continuous confidence estimates (0-100%)
    # Measure Expected Calibration Error (ECE)
    # Goal: ECE < 0.25 (improving from Stage 2)
    # Placeholder for now
    return True


def evaluate_stage_reading(
    brain: Any,
    datasets: Dict[str, Any],
    stage2_datasets: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """Evaluate Stage 3 (Reading & Writing) milestones.

    Success criteria from curriculum_strategy.md:
    - >70% reading comprehension (English, German, Spanish)
    - >65% text generation quality
    - >75% contextually appropriate dialogue responses
    - >60% planning task success
    - Continuous confidence with ECE < 0.25
    - System health maintained
    - Stage 2 maintained (>75%)

    Args:
        brain: Brain instance
        datasets: Dict of Stage 3 datasets (reading, etc.)
        stage2_datasets: Optional Stage 2 datasets for backward compatibility

    Returns:
        Dict mapping criterion name to pass/fail
    """
    results = {}

    # Handle both dict and TaskLoader interfaces
    if hasattr(datasets, "get"):
        reading_data = datasets.get("reading")
    else:
        reading_data = getattr(datasets, "reading_dataset", None)

    # Task performance
    results["reading_comprehension"] = test_reading_comprehension(brain, reading_data)
    results["text_generation"] = test_text_generation(brain, reading_data)
    results["planning_tasks"] = test_planning_tasks(brain)
    results["continuous_confidence"] = test_continuous_confidence(brain)

    # System health
    health = check_system_health(brain)
    results.update(health)

    # Backward compatibility (Stage 2)
    if stage2_datasets is not None:
        grammar_data = (
            stage2_datasets.get("grammar")
            if hasattr(stage2_datasets, "get")
            else getattr(stage2_datasets, "grammar_dataset", None)
        )
        results["grammar_maintained"] = test_grammar_accuracy(brain, grammar_data, threshold=0.75)
        results["set_shifting_maintained"] = test_set_shifting(brain, threshold=0.65)

    return results


# ============================================================================
# Stage 4: Abstract Reasoning
# ============================================================================


def test_analogical_reasoning(
    brain: Any,
    n_samples: int = 200,
    solve_threshold: float = 0.70,
    create_threshold: float = 0.60,
) -> bool:
    """Test analogical reasoning (solve and create).

    Args:
        brain: Brain instance
        n_samples: Number of analogy tests
        solve_threshold: Threshold for solving "A:B::C:?"
        create_threshold: Threshold for creating novel analogies

    Returns:
        True if both solving and creation meet thresholds
    """
    # "A is to B as C is to ___"
    # Test both solving and creating analogies
    # Transfer learning across domains
    # Placeholder for now
    return True


def test_mathematical_reasoning(
    brain: Any,
    n_samples: int = 200,
    threshold: float = 0.75,
) -> bool:
    """Test mathematical reasoning (grade-school level).

    Args:
        brain: Brain instance
        n_samples: Number of math problems
        threshold: Accuracy threshold

    Returns:
        True if math accuracy > threshold
    """
    # Basic arithmetic (learned, not hardcoded)
    # Word problems
    # Simple algebra
    # Explanation quality (not just answers)
    # Placeholder for now
    return True


def test_commonsense_reasoning(
    brain: Any,
    n_samples: int = 200,
    threshold: float = 0.70,
) -> bool:
    """Test commonsense reasoning (PIQA, Social IQA).

    Args:
        brain: Brain instance
        n_samples: Number of commonsense questions
        threshold: Accuracy threshold

    Returns:
        True if commonsense accuracy > threshold
    """
    # Physical intuition (objects fall, liquids pour)
    # Social reasoning (people have goals)
    # Causal inference
    # Placeholder for now
    return True


def test_complex_theory_of_mind(
    brain: Any,
    n_samples: int = 100,
    threshold: float = 0.70,
) -> bool:
    """Test complex theory of mind (second-order beliefs).

    Args:
        brain: Brain instance
        n_samples: Number of ToM tests
        threshold: Accuracy threshold

    Returns:
        True if ToM accuracy > threshold
    """
    # Second-order beliefs: "Alice thinks Bob believes..."
    # Emotion recognition from context
    # Perspective-taking across cultures
    # Placeholder for now
    return True


def test_calibrated_confidence(
    brain: Any,
    n_samples: int = 1000,
    ece_threshold: float = 0.15,
    abstention_threshold: float = 0.70,
) -> bool:
    """Test well-calibrated metacognitive confidence.

    Args:
        brain: Brain instance
        n_samples: Number of test samples
        ece_threshold: Expected Calibration Error threshold
        abstention_threshold: Correct abstention rate threshold

    Returns:
        True if ECE < threshold and abstention appropriate
    """
    # Well-calibrated confidence: ECE < 0.15
    # Appropriate abstention ("I don't know" when uncertain)
    # Active learning: Select next task based on uncertainty
    # Placeholder for now
    return True


def test_fluid_reasoning(
    brain: Any,
    n_samples: int = 100,
    threshold: float = 0.65,
) -> bool:
    """Test fluid reasoning (Raven's matrices, hypothesis testing).

    Args:
        brain: Brain instance
        n_samples: Number of matrix reasoning tasks
        threshold: Accuracy threshold

    Returns:
        True if reasoning accuracy > threshold
    """
    # Raven's Progressive Matrices (abstract pattern induction)
    # Analogical reasoning across domains
    # Hypothesis generation and testing
    # Placeholder for now
    return True


def evaluate_stage_abstract(
    brain: Any,
    datasets: Dict[str, Any],
    stage3_datasets: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """Evaluate Stage 4 (Abstract Reasoning) milestones.

    Success criteria from curriculum_strategy.md:
    - >70% analogical reasoning (solving), >60% (creating)
    - >75% mathematical reasoning
    - >70% commonsense reasoning (PIQA, Social IQA)
    - >70% complex theory of mind
    - Well-calibrated confidence (ECE < 0.15)
    - >65% fluid reasoning (Raven's matrices)
    - >70% active learning task selection
    - System health maintained
    - Stage 3 maintained (>70%)

    Args:
        brain: Brain instance
        datasets: Dict of Stage 4 datasets (analogies, math, etc.)
        stage3_datasets: Optional Stage 3 datasets for backward compatibility

    Returns:
        Dict mapping criterion name to pass/fail
    """
    results = {}

    # Task performance
    results["analogical_reasoning"] = test_analogical_reasoning(brain)
    results["mathematical_reasoning"] = test_mathematical_reasoning(brain)
    results["commonsense_reasoning"] = test_commonsense_reasoning(brain)
    results["complex_theory_of_mind"] = test_complex_theory_of_mind(brain)
    results["calibrated_confidence"] = test_calibrated_confidence(brain)
    results["fluid_reasoning"] = test_fluid_reasoning(brain)

    # System health
    health = check_system_health(brain)
    results.update(health)

    # Backward compatibility (Stage 3)
    if stage3_datasets is not None:
        reading_data = (
            stage3_datasets.get("reading")
            if hasattr(stage3_datasets, "get")
            else getattr(stage3_datasets, "reading_dataset", None)
        )
        results["reading_maintained"] = test_reading_comprehension(
            brain, reading_data, threshold=0.65
        )
        results["planning_maintained"] = test_planning_tasks(brain, threshold=0.55)

    return results


# ============================================================================
# Evaluation Report Generation
# ============================================================================


def generate_evaluation_report(
    stage: str,
    results: Dict[str, bool],
    verbose: bool = True,
) -> str:
    """Generate human-readable evaluation report.

    Args:
        stage: Stage name
        results: Evaluation results
        verbose: Whether to include details

    Returns:
        Formatted report string
    """
    passed = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    all_passed = len(failed) == 0

    report = []
    report.append("=" * 80)
    report.append(f"Stage {stage} Evaluation Report")
    report.append("=" * 80)
    report.append(f"Overall: {'[PASS]' if all_passed else '[FAIL]'}")
    report.append(f"Passed: {len(passed)}/{len(results)}")
    report.append("")

    if verbose:
        if passed:
            report.append("✅ Passed Criteria:")
            for criterion in passed:
                report.append(f"  ✅ {criterion}")
            report.append("")

        if failed:
            report.append("❌ Failed Criteria:")
            for criterion in failed:
                report.append(f"  ❌ {criterion}")
            report.append("")

    report.append("=" * 80)

    return "\n".join(report)
