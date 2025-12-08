"""
Example: Curriculum Training Pipeline

This script demonstrates how to use the CurriculumTrainer to train a brain
through multiple developmental stages (Stage -0.5 → Stage 0).

This is a minimal working example showing the API usage.
"""

import torch
from pathlib import Path

from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
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
)


def create_mock_task_loader_sensorimotor(wrapper):
    """Create a mock task loader for Stage -0.5."""
    class SensorimotorTaskLoader:
        def __init__(self, wrapper):
            self.wrapper = wrapper
            self.task_types = ['motor_control', 'reaching', 'manipulation']
        
        def get_task(self, task_name):
            """Get a task sample."""
            # Reset and get observation
            obs_spikes = self.wrapper.reset()
            
            # Generate random motor command (motor babbling)
            motor_spikes = torch.rand(self.wrapper.n_motor_neurons) < 0.1
            
            # Step environment
            next_obs, reward, terminated, truncated = self.wrapper.step(motor_spikes)
            
            return {
                'input': obs_spikes,
                'n_timesteps': 10,
                'reward': reward,
                'target': next_obs,  # For cerebellum prediction
            }
    
    return SensorimotorTaskLoader(wrapper)


def create_mock_task_loader_phonology():
    """Create a mock task loader for Stage 0."""
    class PhonologyTaskLoader:
        def __init__(self):
            self.task_types = ['mnist', 'temporal', 'phonology']
        
        def get_task(self, task_name):
            """Get a task sample (placeholder)."""
            # Generate random visual input
            input_spikes = torch.rand(784) > 0.8  # MNIST-like
            
            return {
                'input': input_spikes,
                'n_timesteps': 10,
            }
    
    return PhonologyTaskLoader()


def main():
    """Run curriculum training example."""
    
    print("=" * 80)
    print("Curriculum Training Example - Stage -0.5 → Stage 0")
    print("=" * 80)
    print()
    
    # ========================================================================
    # 1. Initialize Brain
    # ========================================================================
    
    print("Step 1: Initializing brain...")
    
    brain_config = EventDrivenBrainConfig(
        input_size=400,  # Will be set by sensorimotor wrapper
        cortex_size=10000,
        hippocampus_size=2000,
        pfc_size=5000,
        n_actions=4,
        device="cpu",
    )
    
    brain = EventDrivenBrain(brain_config)
    print(f"✅ Brain initialized with {brain_config.cortex_size:,} cortex neurons")
    print()
    
    # ========================================================================
    # 2. Initialize Curriculum Trainer
    # ========================================================================
    
    print("Step 2: Initializing curriculum trainer...")
    
    growth_config = get_curriculum_growth_config(conservative=False)
    
    trainer = CurriculumTrainer(
        brain=brain,
        growth_config=growth_config,
        checkpoint_dir="checkpoints/example_curriculum",
        device="cpu",
        verbose=True,
    )
    
    print("✅ Curriculum trainer initialized")
    print()
    
    # ========================================================================
    # 3. Train Stage -0.5 (Sensorimotor Grounding)
    # ========================================================================
    
    print("=" * 80)
    print("Stage -0.5: Sensorimotor Grounding")
    print("=" * 80)
    print()
    
    # Setup sensorimotor environment
    sensorimotor_config = SensorimotorConfig(
        env_name="Reacher-v4",
        spike_encoding="rate",
        n_neurons_per_dof=50,
    )
    
    wrapper = SensorimotorWrapper(config=sensorimotor_config)
    task_loader_sensorimotor = create_mock_task_loader_sensorimotor(wrapper)
    
    # Configure Stage -0.5
    stage_sensorimotor_config = StageConfig(
        duration_steps=10000,  # Reduced for demo (real: 50k)
        task_configs={
            'motor_control': TaskConfig(weight=0.4, difficulty=0.5),
            'reaching': TaskConfig(weight=0.35, difficulty=0.6),
            'manipulation': TaskConfig(weight=0.25, difficulty=0.7),
        },
        success_criteria={
            'reaching_accuracy': 0.90,
            'manipulation_success': 0.85,
            'prediction_error': 0.05,
        },
        consolidation_interval=2000,
        checkpoint_interval=1000,
    )
    
    # Train Stage -0.5
    print("Training Stage -0.5 for 10,000 steps...")
    
    result_sensorimotor = trainer.train_stage(
        stage=CurriculumStage.SENSORIMOTOR,
        config=stage_sensorimotor_config,
        task_loader=task_loader_sensorimotor,
        evaluator=lambda brain, loader: evaluate_stage_sensorimotor(brain, wrapper),
    )
    
    print()
    print("Stage -0.5 Results:")
    print(f"  Success: {'✅' if result_sensorimotor.success else '❌'}")
    print(f"  Training time: {result_sensorimotor.training_time_seconds:.1f}s")
    print(f"  Checkpoints saved: {len(result_sensorimotor.checkpoints)}")
    print()
    
    # ========================================================================
    # 4. Evaluate Stage -0.5 Readiness
    # ========================================================================
    
    print("=" * 80)
    print("Evaluating Stage -0.5 Readiness")
    print("=" * 80)
    print()
    
    ready_for_stage0 = trainer.evaluate_stage_readiness(
        stage=CurriculumStage.SENSORIMOTOR,
    )
    
    if not ready_for_stage0:
        print("⚠️  Stage -0.5 milestones not met. Extending training...")
        trainer.extend_stage(
            stage=CurriculumStage.SENSORIMOTOR,
            additional_steps=5000,
            reason="Milestones not met",
        )
        print()
    
    # ========================================================================
    # 5. Transition to Stage 0
    # ========================================================================
    
    if ready_for_stage0:
        print("=" * 80)
        print("Transition: Stage -0.5 → Stage 0")
        print("=" * 80)
        print()
        
        trainer.transition_to_stage(
            new_stage=CurriculumStage.PHONOLOGY,
            old_stage=CurriculumStage.SENSORIMOTOR,
            weeks=4,
        )
        
        print()
        
        # ====================================================================
        # 6. Train Stage 0 (Phonology)
        # ====================================================================
        
        print("=" * 80)
        print("Stage 0: Sensory Foundations (Phonology)")
        print("=" * 80)
        print()
        
        task_loader_phonology = create_mock_task_loader_phonology()
        
        stage_phonology_config = StageConfig(
            duration_steps=10000,  # Reduced for demo (real: 60k)
            task_configs={
                'mnist': TaskConfig(weight=0.40, difficulty=0.5),
                'temporal': TaskConfig(weight=0.25, difficulty=0.6),
                'phonology': TaskConfig(weight=0.35, difficulty=0.7),
            },
            success_criteria={
                'mnist_accuracy': 0.95,
                'sequence_prediction': 0.90,
                'phoneme_discrimination': 0.90,
            },
            review_stages={
                CurriculumStage.SENSORIMOTOR.value: 0.10,  # 10% review of Stage -0.5
            },
            consolidation_interval=2000,
            checkpoint_interval=1000,
        )
        
        print("Training Stage 0 for 10,000 steps...")
        
        # Note: For full implementation, you'd need real phonology datasets
        result_phonology = trainer.train_stage(
            stage=CurriculumStage.PHONOLOGY,
            config=stage_phonology_config,
            task_loader=task_loader_phonology,
            # evaluator=lambda brain, loader: evaluate_stage_phonology(brain, datasets, wrapper),
        )
        
        print()
        print("Stage 0 Results:")
        print(f"  Success: {'✅' if result_phonology.success else '❌'}")
        print(f"  Training time: {result_phonology.training_time_seconds:.1f}s")
        print(f"  Checkpoints saved: {len(result_phonology.checkpoints)}")
        print()
    
    # ========================================================================
    # 7. Summary
    # ========================================================================
    
    print("=" * 80)
    print("Curriculum Training Summary")
    print("=" * 80)
    print()
    print(f"Stages completed: {len(trainer.training_history)}")
    for i, result in enumerate(trainer.training_history):
        status = "✅" if result.success else "❌"
        print(f"  {status} Stage {result.stage.name}: {result.total_steps:,} steps")
    print()
    print("Checkpoints saved in: checkpoints/example_curriculum/")
    print()
    print("✅ Example complete!")
    print()


if __name__ == "__main__":
    main()
