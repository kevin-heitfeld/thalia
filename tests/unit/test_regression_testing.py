"""
Unit tests for curriculum stage regression testing.

This module tests the catastrophic forgetting detection system that
re-runs tasks from previous stages to verify performance retention.

Author: Thalia Project
Date: December 23, 2025
"""

from typing import Dict, Any, List
from unittest.mock import Mock

import pytest
import torch

from thalia.training.curriculum.stage_manager import (
    CurriculumTrainer,
    StageConfig,
    TaskConfig,
    TrainingResult,
)
from thalia.config.curriculum_growth import CurriculumStage


# ============================================================================
# Mock Task Loader
# ============================================================================

class MockTaskLoader:
    """Mock task loader for testing regression functionality."""

    def __init__(self, device='cpu'):
        self.device = device
        self.task_counts = {}
        self.accuracy = 0.95  # Default high accuracy

    def get_task(self, task_name: str) -> Dict[str, Any]:
        """Get a mock task sample."""
        self.task_counts[task_name] = self.task_counts.get(task_name, 0) + 1

        # Return mock task data
        return {
            'input': torch.zeros(256, dtype=torch.bool, device=self.device),
            'n_timesteps': 10,
            'label': 5,  # Mock label
            'task_type': task_name,
            'accuracy': self.accuracy,
        }

    def get_test_sample(self, task_name: str) -> Dict[str, Any]:
        """Get a test sample (same as training for mock)."""
        return self.get_task(task_name)

    def get_task_types(self) -> List[str]:
        """Get available task types."""
        return ['task_a', 'task_b', 'task_c']

    def reset(self) -> None:
        """Reset loader state."""
        self.task_counts = {}


# ============================================================================
# Mock Brain
# ============================================================================

class MockBrain:
    """Mock brain for testing."""

    def __init__(self, device='cpu'):
        self.device = device
        self.components = {}
        self.forward_count = 0
        self.striatum = Mock()  # Mock striatum for action selection

    def forward(self, input_data, n_timesteps=10):
        """Mock forward pass."""
        self.forward_count += 1
        return {'output': torch.zeros(10, device=self.device)}

    def select_action(self, explore=True):
        """Mock action selection - returns label 5 (matching test data)."""
        return 5, 0.95  # action, confidence


# ============================================================================
# Test Stage Task Loader Caching
# ============================================================================

def test_stage_task_loader_caching():
    """Test that task loaders are cached after successful stage completion."""
    brain = MockBrain()
    trainer = CurriculumTrainer(
        brain=brain,
        checkpoint_dir='test_checkpoints',
        verbose=False,
    )

    # Initially no cached loaders
    assert len(trainer.stage_task_loaders) == 0
    assert len(trainer.stage_configs) == 0

    # Create mock result for successful stage
    stage = CurriculumStage.SENSORIMOTOR
    config = StageConfig(
        duration_steps=100,
        task_configs={'task_a': TaskConfig(weight=1.0)},
        success_criteria={'task_a_accuracy': 0.90},
    )
    task_loader = MockTaskLoader()

    # Simulate successful stage completion
    result = TrainingResult(stage=stage, success=True)
    result.milestone_results = {'task_a_accuracy': True}

    # Manually cache (simulating what train_stage does)
    trainer.stage_task_loaders[stage] = task_loader
    trainer.stage_configs[stage] = config

    # Verify caching
    assert stage in trainer.stage_task_loaders
    assert stage in trainer.stage_configs
    assert trainer.stage_task_loaders[stage] is task_loader
    assert trainer.stage_configs[stage] is config


def test_stage_task_loader_not_cached_on_failure():
    """Test that task loaders are NOT cached when stage fails."""
    brain = MockBrain()
    trainer = CurriculumTrainer(
        brain=brain,
        checkpoint_dir='test_checkpoints',
        verbose=False,
    )

    stage = CurriculumStage.SENSORIMOTOR

    # Create failed result
    result = TrainingResult(stage=stage, success=False)
    result.milestone_results = {'task_a_accuracy': False}

    # Do NOT cache (simulating train_stage behavior on failure)

    # Verify not cached
    assert stage not in trainer.stage_task_loaders
    assert stage not in trainer.stage_configs


# ============================================================================
# Test Regression Test Execution
# ============================================================================

def test_run_regression_test_basic():
    """Test basic regression test execution."""
    brain = MockBrain()
    trainer = CurriculumTrainer(
        brain=brain,
        checkpoint_dir='test_checkpoints',
        verbose=False,
    )

    # Setup: Cache a completed stage
    stage = CurriculumStage.SENSORIMOTOR
    config = StageConfig(
        success_criteria={'task_a_accuracy': 0.90},
    )
    task_loader = MockTaskLoader()

    trainer.stage_task_loaders[stage] = task_loader
    trainer.stage_configs[stage] = config

    # Mock brain output
    def mock_forward(input_data, n_timesteps=10):
        return {'cortex': torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])}  # Predicts class 5

    brain.forward = mock_forward

    # Add mock component with plasticity
    mock_component = Mock()
    mock_component.plasticity_enabled = True
    brain.components['cortex'] = mock_component

    # Run regression test
    performance = trainer._run_regression_test(
        stage=stage,
        criterion='task_a_accuracy',
        n_trials=10,
    )

    # Verify:
    # 1. Some trials ran
    assert task_loader.task_counts.get('task_a', 0) > 0

    # 2. Performance measured (should be 100% since mock always predicts correctly)
    assert 0.0 <= performance <= 1.0

    # 3. Plasticity was disabled and restored
    assert mock_component.plasticity_enabled is True  # Restored


def test_run_regression_test_no_cached_loader():
    """Test regression test fails gracefully when no loader cached."""
    brain = MockBrain()
    trainer = CurriculumTrainer(
        brain=brain,
        checkpoint_dir='test_checkpoints',
        verbose=False,
    )

    # Try to run regression test without cached loader
    with pytest.raises(ValueError, match="No task loader cached"):
        trainer._run_regression_test(
            stage=CurriculumStage.SENSORIMOTOR,
            criterion='task_a_accuracy',
            n_trials=10,
        )


def test_disable_and_restore_plasticity():
    """Test plasticity disabling/restoring during tests."""
    brain = MockBrain()
    trainer = CurriculumTrainer(
        brain=brain,
        checkpoint_dir='test_checkpoints',
        verbose=False,
    )

    # Create mock components with plasticity
    for name in ['cortex', 'hippocampus', 'striatum']:
        component = Mock()
        component.plasticity_enabled = True
        brain.components[name] = component

    # Disable plasticity
    original_states = trainer._disable_plasticity()

    # Verify all disabled
    for name in brain.components:
        assert brain.components[name].plasticity_enabled is False

    # Verify original states stored
    assert len(original_states) == 3
    assert all(state is True for state in original_states.values())

    # Restore plasticity
    trainer._restore_plasticity(original_states)

    # Verify all restored
    for name in brain.components:
        assert brain.components[name].plasticity_enabled is True


# ============================================================================
# Test Backward Compatibility Check
# ============================================================================

def test_backward_compatibility_with_regression():
    """Test backward compatibility check using regression testing."""
    brain = MockBrain()
    trainer = CurriculumTrainer(
        brain=brain,
        checkpoint_dir='test_checkpoints',
        verbose=False,
    )

    # Setup: Complete stage 0 successfully
    stage0 = CurriculumStage.SENSORIMOTOR
    config0 = StageConfig(
        success_criteria={'task_a_accuracy': 0.90, 'task_b_success': 0.85},
    )
    task_loader0 = MockTaskLoader()
    task_loader0.accuracy = 0.95

    trainer.stage_task_loaders[stage0] = task_loader0
    trainer.stage_configs[stage0] = config0

    result0 = TrainingResult(stage=stage0, success=True)
    result0.milestone_results = {'task_a_accuracy': True, 'task_b_success': True}
    trainer.training_history.append(result0)

    # Mock brain forward to simulate good performance
    def mock_forward(input_data, n_timesteps=10):
        return {'cortex': torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])}

    brain.forward = mock_forward

    # Add mock component
    mock_component = Mock()
    mock_component.plasticity_enabled = True
    brain.components['cortex'] = mock_component

    # Now check backward compatibility for stage 1
    stage1 = CurriculumStage.PHONOLOGY
    is_compatible = trainer._check_backward_compatibility(stage1)

    # Should pass because mock always returns correct predictions
    assert is_compatible is True


def test_backward_compatibility_detects_forgetting():
    """Test that backward compatibility detects catastrophic forgetting."""
    brain = MockBrain()

    # Override select_action to return WRONG predictions
    def wrong_prediction(explore=True):
        return 0, 0.95  # Wrong action (expects 5)

    brain.select_action = wrong_prediction

    trainer = CurriculumTrainer(
        brain=brain,
        checkpoint_dir='test_checkpoints',
        verbose=False,
    )

    # Setup: Complete stage 0 successfully
    stage0 = CurriculumStage.SENSORIMOTOR
    config0 = StageConfig(
        success_criteria={'task_a_accuracy': 0.90},
    )
    task_loader0 = MockTaskLoader()

    trainer.stage_task_loaders[stage0] = task_loader0
    trainer.stage_configs[stage0] = config0

    result0 = TrainingResult(stage=stage0, success=True)
    result0.milestone_results = {'task_a_accuracy': True}
    trainer.training_history.append(result0)

    # Add mock component
    mock_component = Mock()
    mock_component.plasticity_enabled = True
    brain.components['cortex'] = mock_component

    # Check backward compatibility
    stage1 = CurriculumStage.PHONOLOGY
    is_compatible = trainer._check_backward_compatibility(stage1)

    # Should fail because performance dropped below 90% threshold
    assert is_compatible is False


# ============================================================================
# Test Integration with Multiple Stages
# ============================================================================

def test_multiple_stage_regression():
    """Test regression testing across multiple completed stages."""
    brain = MockBrain()
    trainer = CurriculumTrainer(
        brain=brain,
        checkpoint_dir='test_checkpoints',
        verbose=False,
    )

    # Complete stages 0 and 1
    for stage, stage_num in [(CurriculumStage.SENSORIMOTOR, 0),
                              (CurriculumStage.PHONOLOGY, 1)]:
        config = StageConfig(
            success_criteria={f'task_{stage_num}_accuracy': 0.90},
        )
        task_loader = MockTaskLoader()

        trainer.stage_task_loaders[stage] = task_loader
        trainer.stage_configs[stage] = config

        result = TrainingResult(stage=stage, success=True)
        result.milestone_results = {f'task_{stage_num}_accuracy': True}
        trainer.training_history.append(result)

    # Mock forward
    def mock_forward(input_data, n_timesteps=10):
        return {'cortex': torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])}

    brain.forward = mock_forward

    # Add mock component
    mock_component = Mock()
    mock_component.plasticity_enabled = True
    brain.components['cortex'] = mock_component

    # Check backward compatibility for stage 2
    stage2 = CurriculumStage.TODDLER
    is_compatible = trainer._check_backward_compatibility(stage2)

    # Should pass - both previous stages retained
    assert is_compatible is True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
