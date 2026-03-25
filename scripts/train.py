"""Training script for Thalia."""

from __future__ import annotations

from thalia.brain.brain_builder import BrainBuilder
from thalia.training.trainer import train_loop, TrainConfig
from thalia.training.tasks.pattern_association import (
    PatternAssociationTask,
    make_readout_groups,
)

if __name__ == "__main__":
    train_config = TrainConfig(
        n_trials=50,
        log_interval=1,
        diagnostics_interval=10,
        checkpoint_interval=100,
        max_criticals_for_stop=100,
        device="cpu",
        log_dir="data/training",
    )

    print("Building brain...")
    brain = BrainBuilder.preset("default", device=train_config.device)
    print(f"Brain built: {brain.biophysics.total_neurons:,} neurons on {brain.device}")

    # Enable learning
    brain.set_learning_disabled(False)

    task = PatternAssociationTask(device=train_config.device)
    readout_groups = make_readout_groups()

    print(f"Starting training: {train_config.n_trials} trials, task={task.name}")
    print("-" * 60)

    results = train_loop(brain, task, readout_groups, train_config)

    print('Results:', [(r.correct, r.reward) for r in results])
    total_correct = sum(1 for r in results if r.correct)
    print("-" * 60)
    print(
        f"Done: {len(results)} trials, "
        f"{total_correct}/{len(results)} correct "
        f"({total_correct / len(results):.0%})"
    )
