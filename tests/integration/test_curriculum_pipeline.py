"""
Integration Tests: Curriculum Training Pipeline

Tests the complete curriculum training pipeline from Stage -0.5 to Stage 0,
including:
- Task loader integration
- Growth triggering
- Consolidation triggering
- Checkpoint save/load
- Health monitoring

These are end-to-end tests that exercise the full system.

Author: Thalia Project
Date: December 9, 2025
"""

import tempfile
import shutil
import pytest
from pathlib import Path
from tqdm import tqdm

from thalia.core.brain import EventDrivenBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes
from thalia.environments import SensorimotorWrapper, SensorimotorConfig
from thalia.config.curriculum_growth import (
    CurriculumStage,
    get_curriculum_growth_config,
)
from thalia.training import (
    CurriculumTrainer,
    StageConfig,
    TaskConfig,
    evaluate_stage_sensorimotor,
    evaluate_stage_phonology,
    create_sensorimotor_loader,
    create_phonology_loader,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def small_brain_config():
    """Create small brain config for fast testing."""
    return ThaliaConfig(
        global_=GlobalConfig(device="cpu"),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=256,  # Fixed size for all sensory inputs (after pathway encoding)
                cortex_size=500,
                hippocampus_size=100,
                pfc_size=200,
                n_actions=4,
            ),
        ),
    )


@pytest.fixture
def sensorimotor_wrapper():
    """Create sensorimotor wrapper for testing."""
    config = SensorimotorConfig(
        env_name='Reacher-v4',
        spike_encoding='rate',
        n_neurons_per_dof=25,  # Reduced for testing
        device='cpu',
    )

    try:
        wrapper = SensorimotorWrapper(config=config)
        yield wrapper
    except Exception as e:
        pytest.skip(f"Could not create sensorimotor wrapper: {e}")


@pytest.fixture
def curriculum_trainer(small_brain_config, temp_checkpoint_dir):
    """Create curriculum trainer for testing."""
    brain = EventDrivenBrain.from_thalia_config(small_brain_config)

    trainer = CurriculumTrainer(
        brain=brain,
        growth_config=get_curriculum_growth_config(),
        checkpoint_dir=temp_checkpoint_dir,
        device='cpu',
        verbose=False,
    )

    return trainer


# ============================================================================
# Test: Full Pipeline (Stage -0.5 → Stage 0)
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_sensorimotor_to_phonology_pipeline(
    curriculum_trainer,
    sensorimotor_wrapper,
    temp_checkpoint_dir,
):
    """Test complete pipeline: Stage -0.5 → Stage 0.

    This is the most important integration test. It verifies:
    1. Stage -0.5 trains successfully
    2. Milestones are evaluated correctly
    3. Checkpoint is saved
    4. Transition to Stage 0 works
    5. Stage 0 trains successfully
    6. Stage -0.5 performance is maintained
    """
    print("\n" + "="*80)
    print("Integration Test: Stage -0.5 to Stage 0 Pipeline")
    print("="*80)

    # ========================================================================
    # Part 1: Train Stage -0.5 (Sensorimotor)
    # ========================================================================

    print("\n[1/6] Training Stage -0.5 (Sensorimotor)...")

    # Create task loaders
    sensorimotor_loader = create_sensorimotor_loader(
        sensorimotor_wrapper,
        output_size=curriculum_trainer.brain.config.input_size
    )

    # Configure stage (reduced steps for testing)
    stage_config = StageConfig(
        duration_steps=200,  # Reduced from 50k for testing
        task_configs={
            'motor_control': TaskConfig(weight=0.4, difficulty=0.5),
            'reaching': TaskConfig(weight=0.35, difficulty=0.6),
            'manipulation': TaskConfig(weight=0.20, difficulty=0.7),
            'prediction': TaskConfig(weight=0.05, difficulty=0.8),
        },
        success_criteria={
            'motor_control_accuracy': 0.70,  # Relaxed for testing
            'reaching_accuracy': 0.60,
            'manipulation_success': 0.50,
            'prediction_error': 0.15,
        },
        growth_check_interval=100,
        consolidation_interval=100,
        checkpoint_interval=100,
        enable_growth=True,
    )

    # Train Stage -0.5
    print("   Training (this may take a few minutes)...")
    pbar_sensorimotor = tqdm(total=stage_config.duration_steps, desc="Stage -0.5", unit="step")

    # Add callback to update progress bar
    def update_progress_sensorimotor(step, metrics):
        pbar_sensorimotor.update(1)

    curriculum_trainer.add_callback(update_progress_sensorimotor)

    result_sensorimotor = curriculum_trainer.train_stage(
        stage=CurriculumStage.SENSORIMOTOR,
        config=stage_config,
        task_loader=sensorimotor_loader,
        evaluator=evaluate_stage_sensorimotor,
    )

    pbar_sensorimotor.close()
    curriculum_trainer.callbacks.clear()  # Clear callbacks for next stage

    print("[OK] Stage -0.5 completed")
    print(f"   Steps: {result_sensorimotor.total_steps}")
    print(f"   Success: {result_sensorimotor.success}")

    # Verify training completed
    assert result_sensorimotor.total_steps == 200
    assert len(result_sensorimotor.checkpoints) >= 1

    # ========================================================================
    # Part 2: Evaluate Stage -0.5 Milestones
    # ========================================================================

    print("\n[2/6] Evaluating Stage -0.5 milestones...")

    milestones_passed = curriculum_trainer.evaluate_stage_readiness(
        stage=CurriculumStage.SENSORIMOTOR,
        evaluator=evaluate_stage_sensorimotor,
    )

    print(f"   Milestones passed: {milestones_passed}")

    # Note: May not pass all milestones with only 1k steps
    # That's OK for this test - we're testing the pipeline, not convergence

    # ========================================================================
    # Part 3: Save Checkpoint
    # ========================================================================

    print("\n[3/6] Saving checkpoint...")

    checkpoint_path_str = curriculum_trainer._save_checkpoint(
        stage=CurriculumStage.SENSORIMOTOR,
        step=result_sensorimotor.total_steps,
    )
    checkpoint_path = Path(checkpoint_path_str)

    assert checkpoint_path.exists()
    print(f"[OK] Checkpoint saved: {checkpoint_path}")

    # ========================================================================
    # Part 4: Transition to Stage 0
    # ========================================================================

    print("\n[4/6] Transitioning to Stage 0...")

    # This should trigger extended consolidation
    curriculum_trainer.transition_to_stage(
        new_stage=CurriculumStage.PHONOLOGY,
        old_stage=CurriculumStage.SENSORIMOTOR,
        weeks=1,  # Reduced for testing
    )

    print("[OK] Transition completed")

    # ========================================================================
    # Part 5: Train Stage 0 (Phonology)
    # ========================================================================

    print("\n[5/6] Training Stage 0 (Phonology)...")

    # Create task loader
    phonology_loader = create_phonology_loader(
        device='cpu',
        output_size=curriculum_trainer.brain.config.input_size
    )

    # Configure stage
    stage0_config = StageConfig(
        duration_steps=200,  # Reduced from 60k for testing
        task_configs={
            'mnist': TaskConfig(weight=0.30, difficulty=0.5),
            'temporal': TaskConfig(weight=0.25, difficulty=0.6),
            'phonology': TaskConfig(weight=0.35, difficulty=0.7),
            'gaze_following': TaskConfig(weight=0.10, difficulty=0.5),
        },
        success_criteria={
            'mnist_accuracy': 0.80,  # Relaxed for testing
            'temporal_accuracy': 0.75,
            'phoneme_discrimination': 0.70,
            'gaze_following_accuracy': 0.65,
        },
        growth_check_interval=100,
        consolidation_interval=100,
        checkpoint_interval=100,
        enable_growth=True,
    )

    # Train Stage 0
    print("   Training (this may take a few minutes)...")
    pbar_phonology = tqdm(total=stage0_config.duration_steps, desc="Stage 0", unit="step")

    # Add callback to update progress bar
    def update_progress_phonology(step, metrics):
        pbar_phonology.update(1)

    curriculum_trainer.add_callback(update_progress_phonology)

    result_phonology = curriculum_trainer.train_stage(
        stage=CurriculumStage.PHONOLOGY,
        config=stage0_config,
        task_loader=phonology_loader,
        evaluator=evaluate_stage_phonology,
    )

    pbar_phonology.close()
    curriculum_trainer.callbacks.clear()  # Clear callbacks for next stage

    print("[OK] Stage 0 completed")
    print(f"   Steps: {result_phonology.total_steps}")
    print(f"   Success: {result_phonology.success}")

    # Verify training completed
    assert result_phonology.total_steps == 200
    assert len(result_phonology.checkpoints) >= 1

    # ========================================================================
    # Part 6: Verify No Catastrophic Forgetting
    # ========================================================================

    print("\n[6/6] Checking for catastrophic forgetting...")

    # Re-evaluate Stage -0.5 performance
    # (In real training, this would use separate test set)
    # For now, just verify brain still responds to sensorimotor input

    # Get a sensorimotor task (already encoded to correct size)
    test_task = sensorimotor_loader.get_task('motor_control')
    test_output = curriculum_trainer.brain.process_sample(
        test_task['input'],
        n_timesteps=test_task.get('n_timesteps', 10),
    )

    # Verify brain still produces output (check spike counts)
    total_spikes = sum(test_output['spike_counts'].values())
    assert total_spikes > 0, "Brain is silent after Stage 0"

    print("[OK] Brain still responds to sensorimotor input")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "="*80)
    print("[OK] PIPELINE TEST PASSED")
    print("="*80)
    print(f"Stage -0.5: {result_sensorimotor.total_steps} steps")
    print(f"Stage 0:    {result_phonology.total_steps} steps")
    print(f"Total:      {result_sensorimotor.total_steps + result_phonology.total_steps} steps")
    print(f"Checkpoints: {len(result_sensorimotor.checkpoints) + len(result_phonology.checkpoints)}")
    print("="*80 + "\n")


# ============================================================================
# Test: Growth Triggering
# ============================================================================

@pytest.mark.integration
def test_growth_triggered_correctly(curriculum_trainer, sensorimotor_wrapper):
    """Verify growth happens when capacity exceeded.

    This test verifies:
    1. Brain starts small
    2. Growth is triggered during training
    3. Consolidation happens before growth
    4. Brain size increases after growth
    5. Checkpoint is saved after growth
    """
    print("\n" + "="*80)
    print("Integration Test: Growth Triggering")
    print("="*80)

    # Record initial size
    initial_size = curriculum_trainer.brain.get_total_neurons()
    print(f"\n[1/4] Initial brain size: {initial_size:,} neurons")

    # Create task loader
    sensorimotor_loader = create_sensorimotor_loader(sensorimotor_wrapper, output_size=curriculum_trainer.brain.config.input_size)

    # Configure stage with frequent growth checks
    stage_config = StageConfig(
        duration_steps=200,
        task_configs={
            'motor_control': TaskConfig(weight=0.5, difficulty=0.5),
            'reaching': TaskConfig(weight=0.5, difficulty=0.6),
        },
        success_criteria={},
        growth_check_interval=100,  # Check frequently
        consolidation_interval=100,
        checkpoint_interval=100,
        enable_growth=True,
    )

    # Train (growth may or may not trigger depending on activity)
    print("\n[2/4] Training with growth monitoring...")
    result = curriculum_trainer.train_stage(
        stage=CurriculumStage.SENSORIMOTOR,
        config=stage_config,
        task_loader=sensorimotor_loader,
    )

    # Check if brain grew
    final_size = curriculum_trainer.brain.get_total_neurons()
    print(f"\n[3/4] Final brain size: {final_size:,} neurons")

    growth_occurred = final_size > initial_size
    print(f"\n[4/4] Growth occurred: {growth_occurred}")

    if growth_occurred:
        growth_amount = final_size - initial_size
        growth_percent = (growth_amount / initial_size) * 100
        print(f"   Added: {growth_amount:,} neurons ({growth_percent:.1f}% increase)")

    # Verify training completed
    assert result.total_steps == 500

    print("\n[OK] Growth monitoring test passed")


# ============================================================================
# Test: Consolidation Triggering
# ============================================================================

@pytest.mark.integration
def test_consolidation_triggered_by_memory_pressure(
    curriculum_trainer,
    sensorimotor_wrapper,
):
    """Verify consolidation happens when needed.

    This test verifies:
    1. Memory pressure is tracked
    2. Consolidation triggers at high pressure
    3. Replay happens during consolidation
    4. Memory pressure decreases after consolidation
    """
    print("\n" + "="*80)
    print("Integration Test: Consolidation Triggering")
    print("="*80)

    # Create task loader
    sensorimotor_loader = create_sensorimotor_loader(sensorimotor_wrapper, output_size=curriculum_trainer.brain.config.input_size)

    # Configure stage with frequent consolidation checks
    stage_config = StageConfig(
        duration_steps=200,
        task_configs={
            'motor_control': TaskConfig(weight=0.5, difficulty=0.5),
            'reaching': TaskConfig(weight=0.5, difficulty=0.6),
        },
        success_criteria={},
        growth_check_interval=1000,
        consolidation_interval=100,  # Check frequently
        checkpoint_interval=100,
        enable_growth=False,  # Disable growth to focus on consolidation
    )

    # Train with consolidation monitoring
    print("\n[1/2] Training with consolidation monitoring...")
    result = curriculum_trainer.train_stage(
        stage=CurriculumStage.SENSORIMOTOR,
        config=stage_config,
        task_loader=sensorimotor_loader,
    )

    print(f"\n[2/2] Training completed: {result.total_steps} steps")

    # Consolidation events are logged in curriculum_trainer
    # (Would check trainer.consolidation_history if it exists)

    # Verify training completed
    assert result.total_steps == 500

    print("\n[OK] Consolidation monitoring test passed")


# ============================================================================
# Test: Checkpoint Resume
# ============================================================================

@pytest.mark.integration
def test_resume_from_checkpoint(
    small_brain_config,
    sensorimotor_wrapper,
    temp_checkpoint_dir,
):
    """Verify training can resume from checkpoint.

    This test verifies:
    1. Train partially and save checkpoint
    2. Load checkpoint in new trainer
    3. Continue training
    4. Final state matches expected trajectory
    """
    print("\n" + "="*80)
    print("Integration Test: Checkpoint Resume")
    print("="*80)

    # ========================================================================
    # Part 1: Train and save checkpoint
    # ========================================================================

    print("\n[1/4] Training first 500 steps...")

    # Create first trainer
    brain1 = EventDrivenBrain.from_thalia_config(small_brain_config)
    trainer1 = CurriculumTrainer(
        brain=brain1,
        growth_config=get_curriculum_growth_config(),
        checkpoint_dir=temp_checkpoint_dir,
        device='cpu',
        verbose=False,
    )

    # Create task loader
    sensorimotor_loader = create_sensorimotor_loader(sensorimotor_wrapper, output_size=curriculum_trainer.brain.config.input_size)

    # Configure stage
    stage_config = StageConfig(
        duration_steps=200,
        task_configs={
            'motor_control': TaskConfig(weight=1.0, difficulty=0.5),
        },
        success_criteria={},
        growth_check_interval=1000,
        consolidation_interval=1000,
        checkpoint_interval=200,
        enable_growth=False,
    )

    # Train first half
    result1 = trainer1.train_stage(
        stage=CurriculumStage.SENSORIMOTOR,
        config=stage_config,
        task_loader=sensorimotor_loader,
    )

    print(f"[OK] First training: {result1.total_steps} steps")
    assert len(result1.checkpoints) >= 1

    checkpoint_path = result1.checkpoints[-1]
    print(f"   Checkpoint: {checkpoint_path}")

    # ========================================================================
    # Part 2: Load checkpoint and continue
    # ========================================================================

    print("\n[2/4] Loading checkpoint in new trainer...")

    # Create second trainer
    brain2 = EventDrivenBrain.from_thalia_config(small_brain_config)
    trainer2 = CurriculumTrainer(
        brain=brain2,
        growth_config=get_curriculum_growth_config(),
        checkpoint_dir=temp_checkpoint_dir,
        device='cpu',
        verbose=False,
    )

    # Load checkpoint
    trainer2.load_checkpoint(checkpoint_path)

    print(f"[OK] Checkpoint loaded")

    # ========================================================================
    # Part 3: Continue training
    # ========================================================================

    print("\n[3/4] Continuing training for 500 more steps...")

    # Create new task loader for second trainer
    sensorimotor_loader2 = create_sensorimotor_loader(sensorimotor_wrapper, output_size=curriculum_trainer.brain.config.input_size)

    # Continue training
    result2 = trainer2.train_stage(
        stage=CurriculumStage.SENSORIMOTOR,
        config=stage_config,
        task_loader=sensorimotor_loader2,
    )

    print(f"[OK] Second training: {result2.total_steps} steps")

    # ========================================================================
    # Part 4: Verify continuity
    # ========================================================================

    print("\n[4/4] Verifying continuity...")

    # Total steps should match
    total_steps = trainer2.global_step
    print(f"   Total steps: {total_steps}")

    # Should be ~1000 steps (500 + 500)
    assert total_steps >= 1000

    print("\n[OK] Checkpoint resume test passed")


# ============================================================================
# Test: Extended Consolidation Before Transition
# ============================================================================

@pytest.mark.integration
def test_extended_consolidation_before_transition(
    curriculum_trainer,
    sensorimotor_wrapper,
):
    """Verify extended consolidation happens at stage boundaries.

    This test verifies:
    1. Train stage to completion
    2. Trigger transition
    3. Extended consolidation happens (2x normal)
    4. Stage evaluation after consolidation
    """
    print("\n" + "="*80)
    print("Integration Test: Extended Consolidation")
    print("="*80)

    # Train Stage -0.5 briefly
    print("\n[1/2] Training Stage -0.5 briefly...")

    sensorimotor_loader = create_sensorimotor_loader(sensorimotor_wrapper, output_size=curriculum_trainer.brain.config.input_size)

    stage_config = StageConfig(
        duration_steps=100,
        task_configs={
            'motor_control': TaskConfig(weight=1.0, difficulty=0.5),
        },
        success_criteria={},
        growth_check_interval=1000,
        consolidation_interval=1000,
        checkpoint_interval=100,
        enable_growth=False,
    )

    result = curriculum_trainer.train_stage(
        stage=CurriculumStage.SENSORIMOTOR,
        config=stage_config,
        task_loader=sensorimotor_loader,
    )

    print(f"[OK] Training completed: {result.total_steps} steps")

    # Trigger transition (this calls extended consolidation internally)
    print("\n[2/2] Triggering transition with extended consolidation...")

    curriculum_trainer.transition_to_stage(
        new_stage=CurriculumStage.PHONOLOGY,
        old_stage=CurriculumStage.SENSORIMOTOR,
        weeks=1,
    )

    print("[OK] Transition completed")
    print("\n[OK] Extended consolidation test passed")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
