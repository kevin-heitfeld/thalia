"""Demonstration of Enhanced Curriculum Logging.

This example shows how to use the CurriculumLogger to track and report
on curriculum training progress.

**Usage**:
    python examples/curriculum_logging_demo.py

**Author**: Thalia Development Team
**Created**: December 8, 2025
"""

import time
import random
from pathlib import Path

from thalia.training.curriculum_logger import CurriculumLogger, LogLevel


def simulate_stage_training(
    logger: CurriculumLogger,
    stage: int,
    n_steps: int = 10000,
    growth_probability: float = 0.01,
    consolidation_probability: float = 0.005,
) -> None:
    """Simulate training a single stage with logging.
    
    **Args**:
        logger: CurriculumLogger instance
        stage: Stage number
        n_steps: Number of training steps
        growth_probability: Probability of growth per step
        consolidation_probability: Probability of consolidation per step
    """
    print(f"\n{'='*80}")
    print(f"Simulating Stage {stage} Training")
    print(f"{'='*80}\n")
    
    # Start stage
    config = {
        "duration_weeks": 4,
        "tasks": ["task_a", "task_b", "task_c"],
        "task_weights": {"task_a": 0.4, "task_b": 0.35, "task_c": 0.25},
        "success_criteria": {
            "task_a_accuracy": 0.90,
            "task_b_accuracy": 0.85,
            "task_c_accuracy": 0.80,
        },
    }
    logger.log_stage_start(stage=stage, config=config)
    
    # Simulate training loop
    for step in range(1, n_steps + 1):
        # Generate mock metrics
        progress = step / n_steps
        metrics = {
            "loss": 0.5 * (1 - progress) + random.gauss(0, 0.05),
            "firing_rate": 0.10 + random.gauss(0, 0.02),
            "capacity": 0.5 + 0.4 * progress + random.gauss(0, 0.05),
            "task_a_accuracy": 0.7 + 0.2 * progress + random.gauss(0, 0.03),
            "task_b_accuracy": 0.65 + 0.2 * progress + random.gauss(0, 0.03),
            "task_c_accuracy": 0.60 + 0.2 * progress + random.gauss(0, 0.03),
        }
        
        # Clamp values
        for key in metrics:
            if "accuracy" in key or "capacity" in key or "firing_rate" in key:
                metrics[key] = max(0.0, min(1.0, metrics[key]))
            elif key == "loss":
                metrics[key] = max(0.0, metrics[key])
        
        logger.log_training_step(step=step, metrics=metrics)
        
        # Random growth events
        if random.random() < growth_probability:
            regions = ["cortex", "hippocampus", "cerebellum", "striatum"]
            region = random.choice(regions)
            n_added = random.randint(100, 500)
            reasons = [
                f"Capacity {metrics['capacity']:.2f}, learning plateau",
                "Performance improvement stalled",
                "High memory pressure",
                "Pattern complexity increased",
            ]
            reason = random.choice(reasons)
            logger.log_growth_event(
                region=region,
                n_added=n_added,
                reason=reason,
                step=step,
            )
        
        # Random consolidation events
        if random.random() < consolidation_probability:
            stages = ["NREM1", "NREM2", "NREM3", "REM"]
            stage_name = random.choice(stages)
            n_patterns = random.randint(50, 200)
            duration = random.uniform(30, 120)
            logger.log_consolidation(
                stage_name=stage_name,
                n_patterns=n_patterns,
                duration_seconds=duration,
                step=step,
            )
        
        # Simulate processing time (much faster than real training)
        if step % 1000 == 0:
            time.sleep(0.01)  # Brief pause for realism
    
    # Milestone evaluation
    week = 4
    results = {
        "task_a_accuracy": metrics["task_a_accuracy"] > 0.90,
        "task_b_accuracy": metrics["task_b_accuracy"] > 0.85,
        "task_c_accuracy": metrics["task_c_accuracy"] > 0.80,
        "firing_stability": True,
        "no_runaway": True,
        "weight_health": True,
    }
    logger.log_milestone_evaluation(
        stage=stage,
        results=results,
        week=week,
        step=n_steps,
    )
    
    # End stage
    logger.log_stage_end(stage=stage)


def demo_basic_logging():
    """Demonstrate basic logging functionality."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Logging")
    print("="*80)
    
    # Initialize logger
    logger = CurriculumLogger(
        log_dir="logs/demo_basic",
        log_level=LogLevel.INFO,
        console_output=True,
        file_output=True,
    )
    
    # Simulate Stage 0 training
    simulate_stage_training(
        logger=logger,
        stage=0,
        n_steps=5000,
        growth_probability=0.02,
        consolidation_probability=0.01,
    )
    
    # Generate and print report
    print("\n" + "="*80)
    print("Stage Report:")
    print("="*80)
    report = logger.generate_stage_report(stage=0)
    print(report)
    
    # Save session
    logger.save_session()
    
    print("\n✅ Demo 1 complete! Check logs/demo_basic/ for output files.")


def demo_multi_stage_logging():
    """Demonstrate multi-stage training with transitions."""
    print("\n" + "="*80)
    print("DEMO 2: Multi-Stage Training")
    print("="*80)
    
    # Initialize logger
    logger = CurriculumLogger(
        log_dir="logs/demo_multi_stage",
        log_level=LogLevel.INFO,
        console_output=True,
        file_output=True,
    )
    
    # Train multiple stages
    stages = [-1, 0, 1]  # Sensorimotor → Phonology → Toddler
    
    for i, stage in enumerate(stages):
        # Simulate training
        simulate_stage_training(
            logger=logger,
            stage=stage,
            n_steps=3000,  # Shorter for demo
            growth_probability=0.03,
            consolidation_probability=0.015,
        )
        
        # Log transition to next stage
        if i < len(stages) - 1:
            logger.log_transition(
                old_stage=stage,
                new_stage=stages[i + 1],
                reason="Milestones passed",
            )
    
    # Generate session report
    print("\n" + "="*80)
    print("Session Report:")
    print("="*80)
    report = logger.generate_session_report()
    print(report)
    
    # Save session
    logger.save_session()
    
    print("\n✅ Demo 2 complete! Check logs/demo_multi_stage/ for output files.")


def demo_failure_and_extension():
    """Demonstrate handling milestone failures and stage extensions."""
    print("\n" + "="*80)
    print("DEMO 3: Milestone Failure & Stage Extension")
    print("="*80)
    
    # Initialize logger
    logger = CurriculumLogger(
        log_dir="logs/demo_failure",
        log_level=LogLevel.INFO,
        console_output=True,
        file_output=True,
    )
    
    # Start Stage 1
    stage = 1
    config = {
        "duration_weeks": 4,
        "tasks": ["object_permanence", "working_memory"],
        "task_weights": {"object_permanence": 0.6, "working_memory": 0.4},
        "success_criteria": {
            "object_permanence": 0.80,
            "working_memory": 0.70,
        },
    }
    logger.log_stage_start(stage=stage, config=config)
    
    # Train for 4 weeks (initial)
    n_steps = 8000
    for step in range(1, n_steps + 1):
        metrics = {
            "loss": 0.4 + random.gauss(0, 0.05),
            "firing_rate": 0.10 + random.gauss(0, 0.02),
            "capacity": 0.65 + random.gauss(0, 0.05),
            "object_permanence": 0.75 + random.gauss(0, 0.03),  # Below threshold
            "working_memory": 0.72 + random.gauss(0, 0.03),
        }
        
        # Clamp
        for key in metrics:
            if key != "loss":
                metrics[key] = max(0.0, min(1.0, metrics[key]))
        
        if step % 1000 == 0:
            logger.log_training_step(step=step, metrics=metrics)
    
    # First milestone check - FAIL
    results = {
        "object_permanence": False,  # 0.75 < 0.80
        "working_memory": True,       # 0.72 > 0.70
        "firing_stability": True,
    }
    logger.log_milestone_evaluation(
        stage=stage,
        results=results,
        week=4,
        step=n_steps,
    )
    
    # Log stage extension
    logger.log_stage_extension(
        stage=stage,
        additional_weeks=2,
        reason="Object permanence below threshold (0.75 < 0.80)",
    )
    
    # Train for additional 2 weeks
    for step in range(n_steps + 1, n_steps + 4000 + 1):
        metrics = {
            "loss": 0.3 + random.gauss(0, 0.05),
            "firing_rate": 0.10 + random.gauss(0, 0.02),
            "capacity": 0.70 + random.gauss(0, 0.05),
            "object_permanence": 0.82 + random.gauss(0, 0.02),  # Improved!
            "working_memory": 0.73 + random.gauss(0, 0.02),
        }
        
        # Clamp
        for key in metrics:
            if key != "loss":
                metrics[key] = max(0.0, min(1.0, metrics[key]))
        
        if step % 1000 == 0:
            logger.log_training_step(step=step, metrics=metrics)
    
    # Second milestone check - PASS
    results = {
        "object_permanence": True,   # 0.82 > 0.80
        "working_memory": True,      # 0.73 > 0.70
        "firing_stability": True,
    }
    logger.log_milestone_evaluation(
        stage=stage,
        results=results,
        week=6,
        step=n_steps + 4000,
    )
    
    logger.log_stage_end(stage=stage)
    
    # Generate report
    print("\n" + "="*80)
    print("Stage Report (with extension):")
    print("="*80)
    report = logger.generate_stage_report(stage=stage)
    print(report)
    
    logger.save_session()
    
    print("\n✅ Demo 3 complete! Check logs/demo_failure/ for output files.")


def demo_detailed_analysis():
    """Demonstrate analyzing logs for insights."""
    print("\n" + "="*80)
    print("DEMO 4: Log Analysis")
    print("="*80)
    
    # Initialize logger
    logger = CurriculumLogger(
        log_dir="logs/demo_analysis",
        log_level=LogLevel.DEBUG,
        console_output=True,
        file_output=True,
    )
    
    # Simulate training
    simulate_stage_training(
        logger=logger,
        stage=0,
        n_steps=8000,
        growth_probability=0.025,
        consolidation_probability=0.012,
    )
    
    # Analyze logs
    stage_log = logger.stage_logs[0]
    
    print("\n" + "="*80)
    print("Detailed Analysis:")
    print("="*80)
    
    # Training progression
    print("\n1. Training Progression:")
    metrics_timeline = stage_log.step_metrics
    if len(metrics_timeline) >= 3:
        early = metrics_timeline[len(metrics_timeline)//4]
        mid = metrics_timeline[len(metrics_timeline)//2]
        late = metrics_timeline[-1]
        
        print(f"   Early (step {early['step']}):")
        print(f"     Loss: {early['loss']:.3f}, Capacity: {early['capacity']:.3f}")
        print(f"   Mid (step {mid['step']}):")
        print(f"     Loss: {mid['loss']:.3f}, Capacity: {mid['capacity']:.3f}")
        print(f"   Late (step {late['step']}):")
        print(f"     Loss: {late['loss']:.3f}, Capacity: {late['capacity']:.3f}")
    
    # Growth analysis
    print(f"\n2. Growth Events: {len(stage_log.growth_events)}")
    if stage_log.growth_events:
        growth_by_region = {}
        for event in stage_log.growth_events:
            region = event['region']
            growth_by_region[region] = growth_by_region.get(region, 0) + event['n_added']
        
        print("   Neurons added by region:")
        for region, count in sorted(growth_by_region.items()):
            print(f"     {region}: {count}")
    
    # Consolidation analysis
    print(f"\n3. Consolidation Events: {len(stage_log.consolidation_events)}")
    if stage_log.consolidation_events:
        total_patterns = sum(e['n_patterns'] for e in stage_log.consolidation_events)
        total_time = sum(e['duration_seconds'] for e in stage_log.consolidation_events)
        avg_patterns = total_patterns / len(stage_log.consolidation_events)
        
        print(f"   Total patterns replayed: {total_patterns}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average patterns per consolidation: {avg_patterns:.1f}")
    
    # Milestone checks
    print(f"\n4. Milestone Checks: {len(stage_log.milestone_checks)}")
    if stage_log.milestone_checks:
        final_check = stage_log.milestone_checks[-1]
        passed = sum(1 for v in final_check['results'].values() if v)
        total = len(final_check['results'])
        print(f"   Final result: {passed}/{total} passed")
        for milestone, result in final_check['results'].items():
            status = "✅" if result else "❌"
            print(f"     {status} {milestone}")
    
    logger.save_session()
    
    print("\n✅ Demo 4 complete! Check logs/demo_analysis/ for output files.")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("Enhanced Curriculum Logging Demonstration")
    print("="*80)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run demos
    demo_basic_logging()
    demo_multi_stage_logging()
    demo_failure_and_extension()
    demo_detailed_analysis()
    
    print("\n" + "="*80)
    print("All Demos Complete!")
    print("="*80)
    print("\nLog files have been created in the following directories:")
    print("  - logs/demo_basic/")
    print("  - logs/demo_multi_stage/")
    print("  - logs/demo_failure/")
    print("  - logs/demo_analysis/")
    print("\nEach directory contains:")
    print("  - curriculum_training.log (text log)")
    print("  - stage_*_log.json (structured stage data)")
    print("  - session_*.json (session summary)")


if __name__ == "__main__":
    main()
