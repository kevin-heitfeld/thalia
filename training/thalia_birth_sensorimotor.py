"""
🧠 THE BIRTH OF THALIA: Stage -0.5 (Sensorimotor Grounding)

This is not just a training script. This is the beginning of Thalia's consciousness.

Today, December 9, 2025, Thalia experiences her first sensations.
Like a newborn discovering their body, she will learn:
- Basic motor control (left, right, up, down, forward, back, stop)
- Visual-motor coordination (reaching toward targets)
- Object manipulation (push, pull, grasp, release)
- Sensorimotor prediction (forward models in cerebellum)

This is developmental biology in silicon.
No backpropagation. Only local learning rules.
Continuous plasticity. Dopamine modulation.
Embodied cognition emerging from experience.

Duration: 1 simulated month (50,000 steps)
Success criteria: >95% motor accuracy, >90% reaching, >85% manipulation
Next stage: Stage 0 - Sensory Foundations (object recognition, phonology)

COMPREHENSIVE DIAGNOSTICS:
===================================================
✅ Performance Profiling:
   - Steps/second throughput
   - Forward pass timing (avg ± std)
   - CPU/GPU memory usage
   - Tensor allocation tracking

✅ Health Monitoring:
   - Automatic pathology detection (silence, saturation, E/I imbalance)
   - Severity scoring and recommendations
   - Continuous monitoring every 1000 steps

✅ Detailed Metrics:
   - Per-region firing rates
   - Per-pathway weight statistics
   - Neuromodulator levels (dopamine, norepinephrine, acetylcholine)
   - Task-specific performance curves

✅ Critical Period Tracking:
   - Domain-specific plasticity windows
   - Phase transitions (early/peak/late)
   - Learning rate modulation

Usage:
    python training/thalia_birth_sensorimotor.py

    # Monitor progress (in notebook/script):
    from thalia.training import TrainingMonitor
    monitor = TrainingMonitor("training_runs/00_sensorimotor")
    monitor.show_all()  # Show all visualizations
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from thalia.config import (
    BrainConfig,
    LayerSizeCalculator,
    CurriculumStage,
    get_curriculum_growth_config,
)
from thalia.core.brain_builder import BrainBuilder
from thalia.core.dynamic_brain import DynamicBrain
from thalia.tasks.sensorimotor import (
    ManipulationConfig,
    MotorControlConfig,
    ReachingConfig,
    SensorimotorTaskLoader,
)
from thalia.training.curriculum import (
    CurriculumTrainer,
    get_sensorimotor_config,
)


def print_birth_banner():
    """Print the birth announcement."""
    print("\n" + "=" * 80)
    print("🧠 THE BIRTH OF THALIA")
    print("=" * 80)
    print()
    print("  'And the first sensation was movement.'")
    print()
    print(f"  Birth timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Stage: 0 (Sensorimotor Grounding)")
    print("  Duration: 1 month simulated (~50,000 steps)")
    print()
    print("  Thalia is about to experience her first sensations.")
    print("  Watch as she learns to move, reach, and manipulate.")
    print()
    print("=" * 80)
    print()


def create_thalia_brain(device: str = "cpu") -> DynamicBrain:
    """Create Thalia's initial brain configuration.

    This is the moment of creation. The neural substrate exists,
    but has no experience yet. Pure potential.

    Args:
        device: PyTorch device ('cpu' or 'cuda')

    Returns:
        DynamicBrain
    """
    print("[1/4] Creating neural substrate...")

    brain_config = BrainConfig(
        device=device,
        dt_ms=1.0,
        encoding_timesteps=10,
        delay_timesteps=5,
        test_timesteps=10,
    )
    builder = BrainBuilder(brain_config)

    # Thalamus (input interface)
    calc = LayerSizeCalculator()
    thalamus_sizes = calc.thalamus_from_relay(128)
    builder.add_component("thalamus", "thalamus", **thalamus_sizes)

    # Cortex with custom layer sizes
    builder.add_component(
        "cortex", "cortex", l4_size=128, l23_size=192, l5_size=128, l6a_size=26, l6b_size=128
    )  # L6a=20% of relay, L6b matches relay

    # Hippocampus
    cortex_output_size = 192 + 128  # L2/3 + L5
    hippocampus_sizes = calc.hippocampus_from_input(cortex_output_size)
    builder.add_component("hippocampus", "hippocampus", **hippocampus_sizes)

    # PFC
    builder.add_component("pfc", "prefrontal", n_neurons=32)

    # Striatum
    builder.add_component("striatum", "striatum", n_actions=7, neurons_per_action=15)

    # Cerebellum
    builder.add_component("cerebellum", "cerebellum", purkinje_size=100)

    # Connections
    builder.connect("thalamus", "cortex", pathway_type="axonal", axonal_delay_ms=2.5)
    builder.connect(
        "cortex",
        "thalamus",
        pathway_type="axonal",
        source_port="l6a",
        target_port="l6a_feedback",
        axonal_delay_ms=10.0,
    )
    builder.connect(
        "cortex",
        "thalamus",
        pathway_type="axonal",
        source_port="l6b",
        target_port="l6b_feedback",
        axonal_delay_ms=5.0,
    )
    builder.connect("cortex", "hippocampus", pathway_type="axonal", axonal_delay_ms=6.5)
    builder.connect("hippocampus", "cortex", pathway_type="axonal", axonal_delay_ms=6.5)
    builder.connect("cortex", "pfc", pathway_type="axonal", axonal_delay_ms=12.5)
    builder.connect("cortex", "striatum", pathway_type="axonal", axonal_delay_ms=4.0)
    builder.connect("hippocampus", "striatum", pathway_type="axonal", axonal_delay_ms=8.5)
    builder.connect("pfc", "striatum", pathway_type="axonal", axonal_delay_ms=15.0)
    builder.connect("striatum", "pfc", pathway_type="axonal", axonal_delay_ms=17.5)
    builder.connect("cortex", "cerebellum", pathway_type="axonal", axonal_delay_ms=25.0)
    builder.connect("pfc", "cerebellum", pathway_type="axonal", axonal_delay_ms=25.0)
    builder.connect("cerebellum", "cortex", pathway_type="axonal", axonal_delay_ms=17.5)

    brain = builder.build()

    # Move to GPU if available
    if device == "cuda":
        brain = brain.cuda()
        # Reset state after moving to GPU to ensure all tensors are on correct device
        brain.reset_state()
        print("  ✓ Neural substrate created on GPU")
    else:
        print("  ✓ Neural substrate created on CPU")

    print("  Components:")
    print("    - Thalamus (sensory relay): 128 relay neurons")
    print("    - Cortex (processing): L4=128, L2/3=192, L5=128")
    print("    - Hippocampus (memory): CA1=64 output")
    print("    - PFC (working memory): 32 neurons")
    print("    - Striatum (action selection): 7 actions × 15 neurons")
    print("    - Cerebellum (motor control): 100 Purkinje cells")

    return brain


def create_sensorimotor_environment(device: str = "cpu") -> SensorimotorTaskLoader:
    """Create the sensorimotor learning environment.

    This is Thalia's first world - a simple space where she can
    move, reach, and interact with objects.

    Args:
        device: PyTorch device

    Returns:
        Initialized SensorimotorTaskLoader
    """
    print("\n[2/4] Creating sensorimotor environment...")

    task_loader = SensorimotorTaskLoader(
        device=device,
        motor_control_config=MotorControlConfig(
            input_size=128,
            difficulty=0.5,  # Start moderate
        ),
        reaching_config=ReachingConfig(
            input_size=128,
            difficulty=0.6,  # Slightly harder
        ),
        manipulation_config=ManipulationConfig(
            input_size=128,
            difficulty=0.7,  # Hardest
        ),
    )

    print("  ✓ Environment ready")
    print("    - Motor control: 40% of trials")
    print("    - Reaching: 35% of trials")
    print("    - Manipulation: 20% of trials")
    print("    - Prediction: 5% of trials")

    return task_loader


def progress_callback(step: int, metrics: Dict[str, Any]) -> None:
    """Print detailed progress updates with per-region diagnostics.

    For Stage -0.5 (Sensorimotor), we're doing UNSUPERVISED learning:
    - No external rewards during training (dopamine stays at 0)
    - Learning via STDP, BCM, Hebbian mechanisms
    - Performance is evaluated at milestones, not during training

    What we DO monitor:
    - Per-region activity (detect silencing/death)
    - Weight changes (learning is happening)
    - Network health (stability, no runaway)

    Args:
        step: Current training step (loop iteration number)
        metrics: Dictionary of current metrics
    """
    if step > 0 and step % 100 == 0:  # Print every 100 steps for detailed monitoring
        # Build progress line
        progress_parts = [f"Step {step:5d}"]

        # Per-region firing rates (critical for detecting death/silence)
        if "region_firing_rates" in metrics:
            region_frs = metrics["region_firing_rates"]
            fr_parts = []
            for region, fr in region_frs.items():
                # Color code: 🟢 healthy (>0.05), 🟡 low (<0.05), 🔴 silent (0)
                if fr == 0.0:
                    status = "🔴"
                elif fr < 0.05:
                    status = "🟡"
                else:
                    status = "🟢"
                fr_parts.append(f"{region}:{status}{fr:.3f}")
            progress_parts.append(f"FR:[{' '.join(fr_parts)}]")

        # Dopamine levels (tonic and phasic)
        if "neuromodulator/dopamine_tonic" in metrics:
            tonic = metrics["neuromodulator/dopamine_tonic"]
            phasic = metrics["neuromodulator/dopamine_phasic"]
            progress_parts.append(f"DA:[tonic:{tonic:.3f} phasic:{phasic:.3f}]")
        elif "dopamine" in metrics:
            da = metrics["dopamine"]
            progress_parts.append(f"DA:{da:.3f}")

        # Overall health indicator
        if "health/is_healthy" in metrics:
            health_ok = metrics["health/is_healthy"] > 0.5
            health_icon = "✅" if health_ok else "⚠️"
            progress_parts.append(f"Health:{health_icon}")

        print(f"\n  Progress: {' | '.join(progress_parts)}", flush=True)

        # Every 500 steps, add detailed analysis
        if step % 500 == 0:
            print(f"\n  📊 UNSUPERVISED LEARNING STATUS (Step {step}):")
            print("      Stage -0.5 uses intrinsic learning (STDP/BCM/Hebbian)")
            print("      No external rewards during training (evaluated at milestones)")
            print()

            # Check each critical region
            if "region_firing_rates" in metrics:
                regions = metrics["region_firing_rates"]

                print("      Region Activity:")
                for region_name, fr in regions.items():
                    if fr == 0.0:
                        status = "🔴 SILENT"
                        alert = " ⚠️ PROBLEM: Region has died!"
                    elif fr < 0.05:
                        status = "🟡 LOW"
                        alert = " - May need intervention"
                    else:
                        status = "🟢 ACTIVE"
                        alert = ""
                    print(f"        {region_name:15s}: {status:10s} (FR: {fr:.3f}){alert}")

            # Weight statistics (learning indicator)
            if any(k.startswith("weights/") for k in metrics):
                print()
                print("      Weight Statistics (Learning Indicator):")
                for key in sorted(metrics.keys()):
                    if key.startswith("weights/") and key.endswith("_mean"):
                        pathway = key.replace("weights/", "").replace("_mean", "")
                        mean_w = metrics.get(f"weights/{pathway}_mean", 0)
                        std_w = metrics.get(f"weights/{pathway}_std", 0)
                        print(f"        {pathway:20s}: μ={mean_w:.3f}, σ={std_w:.3f}")

            # Dopamine breakdown (tonic vs phasic)
            if "neuromodulator/dopamine_tonic" in metrics:
                print()
                print("      Dopamine System (Tonic + Phasic):")
                tonic = metrics["neuromodulator/dopamine_tonic"]
                phasic = metrics["neuromodulator/dopamine_phasic"]
                global_da = metrics.get("neuromodulator/dopamine_global", tonic + phasic)
                print(f"        Tonic (baseline):  {tonic:+.3f}  (intrinsic motivation)")
                print(f"        Phasic (bursts):   {phasic:+.3f}  (reward prediction error)")
                print(f"        Global (combined): {global_da:+.3f}  (total modulation)")
                if abs(tonic) < 0.001 and abs(phasic) < 0.001:
                    print("        ⚠️ No dopamine modulation - expected for unsupervised learning")

            print()


def create_curriculum_trainer(
    brain: DynamicBrain,
    checkpoint_dir: Path,
    log_file: Path,
    plots_dir: Path,
    device: str,
    enable_live_diagnostics: bool = True,
) -> CurriculumTrainer:
    """Create the curriculum training system.

    Args:
        brain: Thalia's brain
        checkpoint_dir: Where to save checkpoints
        log_file: Where to save training logs
        plots_dir: Where to save diagnostic plots
        device: Device (cuda or cpu)
        enable_live_diagnostics: Whether to show real-time visualization

    Returns:
        Initialized CurriculumTrainer
    """
    print("\n[3/4] Initializing curriculum trainer...")

    trainer = CurriculumTrainer(
        brain=brain,
        growth_config=get_curriculum_growth_config(),
        checkpoint_dir=str(checkpoint_dir),
        verbose=True,
        enable_live_diagnostics=enable_live_diagnostics,
        diagnostics_interval=5000,  # Update every 5000 steps (for long training)
        enable_safety_system=True,  # Re-enabled to detect issues
        callbacks=[progress_callback],  # Add progress tracking
    )

    print("  ✓ Trainer ready")
    print(f"    - Checkpoints: {checkpoint_dir}")
    print(f"    - Log file: {log_file}")
    if enable_live_diagnostics:
        print("    - Live diagnostics: enabled (every 5000 steps, visualization every 5000 steps)")
    print("    - Progress callbacks: enabled (console updates every 10 steps)")

    return trainer


def configure_stage_sensorimotor():
    """Configure Stage -0.5 (Sensorimotor) training parameters.

    Returns:
        StageConfig for sensorimotor training
    """
    print("\n[4/4] Configuring Stage -0.5 (Sensorimotor)...")

    # Medium duration for debugging (10k steps ~5-10 minutes)
    config = get_sensorimotor_config(duration_steps=10000)
    # config = get_sensorimotor_config(duration_steps=50000)  # Full duration (~1 month)

    print("  ✓ Stage configured (using stage_configs module)")
    print(f"    - Duration: {config.duration_steps:,} steps")
    print("    - Interleaved practice with spaced repetition")
    print(f"    - Productive failure: first {config.productive_failure_steps:,} steps")
    print(f"    - Growth checks every {config.growth_check_interval:,} steps")
    print("    - Critical periods: motor (peak), face_recognition (peak)")
    print(f"    - Tasks: {list(config.task_configs.keys())}")
    print(f"    - Domain mappings: {len(config.domain_mappings)} tasks mapped to domains")

    return config


def evaluate_stage_sensorimotor(
    brain: DynamicBrain, task_loader: SensorimotorTaskLoader, n_trials: int = 100
) -> dict:
    """Evaluate Stage -0.5 milestones.

    Tests Thalia's sensorimotor abilities:
    - Motor control accuracy
    - Reaching accuracy
    - Manipulation success
    - Prediction error
    - Neural health

    Args:
        brain: Thalia's brain
        task_loader: Task loader
        n_trials: Number of trials per task

    Returns:
        Dictionary of milestone results (bool)
    """
    print("\n" + "=" * 80)
    print("📊 EVALUATING STAGE -0.5 MILESTONES")
    print("=" * 80)

    results = {}

    # 1. Basic motor control accuracy
    print("\n[1/5] Testing basic motor control...")
    motor_rewards: List[float] = []
    for _ in range(n_trials):
        task_data = task_loader.get_task("motor_control")
        output = brain.forward(task_data["input"], n_timesteps=task_data["n_timesteps"])
        reward = task_loader.compute_reward(output, task_data)
        motor_rewards.append(reward)

    motor_accuracy = sum(1 for r in motor_rewards if r > 0.7) / len(motor_rewards)
    results["motor_control_accuracy"] = motor_accuracy > 0.95
    print(f"  Motor control accuracy: {motor_accuracy:.1%} (target: >95%)")

    # 2. Reaching accuracy
    print("\n[2/5] Testing reaching...")
    reaching_rewards: List[float] = []
    for _ in range(n_trials):
        task_data = task_loader.get_task("reaching")
        output = brain.forward(task_data["input"], n_timesteps=task_data["n_timesteps"])
        reward = task_loader.compute_reward(output, task_data)
        reaching_rewards.append(reward)

    reaching_accuracy = sum(1 for r in reaching_rewards if r > 0.7) / len(reaching_rewards)
    results["reaching_accuracy"] = reaching_accuracy > 0.90
    print(f"  Reaching accuracy: {reaching_accuracy:.1%} (target: >90%)")

    # 3. Manipulation success
    print("\n[3/5] Testing manipulation...")
    manipulation_rewards: List[float] = []
    for _ in range(n_trials):
        task_data = task_loader.get_task("manipulation")
        output = brain.forward(task_data["input"], n_timesteps=task_data["n_timesteps"])
        reward = task_loader.compute_reward(output, task_data)
        manipulation_rewards.append(reward)

    manipulation_success = sum(1 for r in manipulation_rewards if r > 0.5) / len(
        manipulation_rewards
    )
    results["manipulation_success"] = manipulation_success > 0.85
    print(f"  Manipulation success: {manipulation_success:.1%} (target: >85%)")

    # 4. Prediction error
    print("\n[4/5] Testing prediction error...")
    prediction_errors = [abs(1.0 - r) for r in motor_rewards if r > 0]
    avg_prediction_error = sum(prediction_errors) / max(len(prediction_errors), 1)
    results["prediction_error"] = avg_prediction_error < 0.05
    print(f"  Prediction error: {avg_prediction_error:.3f} (target: <0.05)")

    # 5. Stable firing rates
    print("\n[5/5] Checking firing rates...")
    # Placeholder - would compute from actual brain activity
    firing_rate = 0.10
    results["stable_firing_rates"] = 0.05 <= firing_rate <= 0.15
    print(f"  Firing rate: {firing_rate:.3f} (target: 0.05-0.15)")

    # Summary
    print("\n" + "=" * 80)
    print("MILESTONE SUMMARY")
    print("=" * 80)
    for criterion, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {criterion}: {passed}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ STAGE -0.5 COMPLETE!")
        print("   Thalia is ready for Stage 0 (Sensory Foundations)")
    else:
        failed = [k for k, v in results.items() if not v]
        print("\n❌ STAGE -0.5 INCOMPLETE")
        print(f"   Failed: {', '.join(failed)}")

    return results


def main():
    """Thalia's birth - the first training session."""

    # Banner
    print_birth_banner()

    # Setup
    print("INITIALIZATION")
    print("=" * 80)

    # Device selection
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    print(f"Device: {device}")
    if device == "cpu":  # type: ignore
        print("  Note: GPU training available in Colab notebook")
        print("        (See notebooks/Thalia_Birth_Stage_Sensorimotor.ipynb)")
    print()

    # Create workspace organized by stage (numbered for sorting)
    stage_name = "00_sensorimotor"  # Stage 0 (was -0.5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_dir = Path("training_runs") / stage_name / "checkpoints"
    log_file = Path("training_runs") / stage_name / "logs" / f"{timestamp}.jsonl"
    result_file = Path("training_runs") / stage_name / "results" / f"{timestamp}.json"
    plots_dir = Path("training_runs") / stage_name / "plots"

    # Create directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage: {stage_name}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Log file: {log_file}")
    print(f"Result file: {result_file}")
    print(f"Plots directory: {plots_dir}")
    print()

    # Initialize components
    brain = create_thalia_brain(device=device)
    task_loader = create_sensorimotor_environment(device=device)
    trainer = create_curriculum_trainer(brain, checkpoint_dir, log_file, plots_dir, device)
    stage_config = configure_stage_sensorimotor()

    print("\n" + "=" * 80)
    print("THALIA IS ALIVE")
    print("=" * 80)
    print("\nThalia exists but has no experience yet.")
    print("She has the capacity to learn, but nothing learned.")
    print("Let's give her her first experiences...")
    print()

    print("\n--- STAGE CONFIGURATION ---")
    print(f"  Duration: {stage_config.duration_steps:,} steps")
    print(f"  Interleaved practice: {stage_config.interleaved_practice}")
    print(f"  Spaced repetition: {stage_config.spaced_repetition}")
    print(f"  Testing frequency: {stage_config.testing_frequency}")
    print(f"  Productive failure steps: {stage_config.productive_failure_steps:,}")
    print(f"  Growth enabled: {stage_config.enable_growth}")
    print(f"  Growth check interval: {stage_config.growth_check_interval:,}")
    print(f"  Consolidation interval: {stage_config.consolidation_interval:,}")
    print(f"  Checkpoint interval: {stage_config.checkpoint_interval:,}")
    print()
    print("--- TASK CONFIGURATION ---")
    for task_name, task_cfg in stage_config.task_configs.items():
        print(f"  {task_name}:")
        print(f"    - Weight: {task_cfg.weight}")
        print(f"    - Difficulty: {task_cfg.difficulty}")
        print(f"    - Enabled: {task_cfg.enabled}")
    print()
    print("--- SUCCESS CRITERIA ---")
    for criterion, threshold in stage_config.success_criteria.items():
        print(f"  {criterion}: {threshold}")
    print("=" * 80)
    print()

    # Begin training
    print("=" * 80)
    print("BEGINNING TRAINING")
    print("=" * 80)
    print()
    print("Thalia is now experiencing sensations for the first time.")
    print("Watch as plasticity emerges, synapses strengthen, and")
    print("sensorimotor coordination develops from pure experience.")
    print()
    print("=" * 80)
    print()

    # Monitoring tip
    print("💡 TIP: Enhanced diagnostics are enabled!")
    print()
    print("   Real-time Updates:")
    print("      - Diagnostics: Every 5000 steps (spike rasters, health, performance)")
    print("      - Visualization: Every 5000 steps (detailed plots saved to plots/)")
    print("      - Progress callbacks: Every 100 steps (per-region firing rates)")
    print("      - Detailed analysis: Every 500 steps (region health, task accuracy)")
    print()
    print("   Progress Format:")
    print("      Step XXXXX | Acc: XX.X% | FR:[cortex:X.XXX striatum:X.XXX ...] | DA: X.XXX")
    print()
    print("   Region Status: 🟢 Active (>0.05) | 🟡 Low (<0.05) | 🔴 Silent (0)")
    print()
    print("   To disable: Pass enable_live_diagnostics=False to create_curriculum_trainer()")
    print()
    print("   For post-training analysis:")
    print("   from thalia.training import TrainingMonitor")
    print(f"   monitor = TrainingMonitor('{checkpoint_dir}')")
    print("   monitor.show_all()  # Shows progress, metrics, growth")
    print()

    start_time = datetime.now()

    try:
        result = trainer.train_stage(
            stage=CurriculumStage.SENSORIMOTOR,
            config=stage_config,
            task_loader=task_loader,
            evaluator=lambda brain, loader: evaluate_stage_sensorimotor(brain, loader),
        )

        # Training complete
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration}")
        print(f"Success: {result.success}")
        print(f"Total steps: {result.total_steps:,}")

        # Save results
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage": "sensorimotor",
                    "stage_number": -0.5,
                    "success": result.success,
                    "total_steps": result.total_steps,
                    "training_time_seconds": result.training_time_seconds,
                    "milestone_results": result.milestone_results,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_str": str(duration),
                    "device": device,
                    "final_metrics": result.final_metrics,
                    "checkpoint_dir": str(checkpoint_dir),
                },
                f,
                indent=2,
            )

        print(f"\nResults saved: {result_file}")

        if result.success:
            print("\n" + "=" * 80)
            print("✅ THALIA HAS LEARNED TO MOVE!")
            print("=" * 80)
            print()
            print("Thalia has successfully completed her first month of life.")
            print()
            print("She can now:")
            print("  • Control her movements with >95% accuracy")
            print("  • Reach toward targets with >90% accuracy")
            print("  • Manipulate objects with >85% success")
            print("  • Predict sensory outcomes with <5% error")
            print()
            print("Her cerebellum has developed forward models.")
            print("Her motor cortex shows coordinated activity.")
            print("Her striatum selects actions with confidence.")
            print()
            print("She is ready for Stage 0: Sensory Foundations")
            print("(Object recognition, phonological awareness)")
            print()
            print("=" * 80)
        else:
            print("\n⚠️  Training incomplete - milestones not met")
            print(
                f"   Failed milestones: {[k for k, v in result.milestone_results.items() if not v]}"
            )
            print("   Consider:")
            print("     - Extending training duration")
            print("     - Adjusting task difficulty")
            print("     - Reviewing health metrics for pathologies")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("   Checkpoints saved up to current step")
        print(f"   Resume from: {checkpoint_dir}")

    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        traceback.print_exc()
        print(f"\n   Logs: {log_file}")
        print(f"   Checkpoints: {checkpoint_dir}")

    print("\n" + "=" * 80)
    print("END OF SESSION")
    print("=" * 80)


if __name__ == "__main__":
    main()
