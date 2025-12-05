"""
Event-Driven Brain Demo using EventDrivenBrain from thalia.core.brain
======================================================================

This demo showcases the new event-driven multi-region brain system with
input buffering and learnable inter-region pathways.

The demo runs a Delayed Match-to-Sample task:
1. Show sample pattern (encode in Hippocampus, PFC)
2. Delay period (PFC maintains working memory)
3. Show test pattern (compare via all pathways)
4. Respond: MATCH or NO-MATCH
5. Sleep: Hippocampus replays to Cortex for consolidation

Key features:
- Uses EventDrivenBrain from thalia.core.brain
- Input buffering for proper dimension handling
- Learnable pathways: PFC→Cortex (attention), Hippo→Cortex (replay)
- D1/D2 opponent process in Striatum
- Sleep consolidation with N1/N2/SWS/REM stages
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
import numpy as np

# Import the new event-driven brain system
from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig


@dataclass
class TrialResult:
    """Result of a single trial."""
    sample_pattern: int
    test_pattern: int
    is_match: bool
    selected_action: int
    correct: bool
    explored: bool
    cortex_activity: float
    pfc_activity: float
    striatum_activity: torch.Tensor
    population_votes: Optional[torch.Tensor] = None
    confidence: float = 0.0
    epoch: int = 0


def create_distinct_patterns(n_patterns: int = 4, size: int = 256) -> List[torch.Tensor]:
    """Create highly distinct patterns for match-to-sample task.

    Each pattern activates a different quadrant with NO shared center,
    making them maximally distinguishable.
    """
    patterns = []
    grid = int(size ** 0.5)  # 16x16
    half = grid // 2

    for i in range(n_patterns):
        pattern = torch.zeros(size)
        row_start = (i // 2) * half
        col_start = (i % 2) * half

        for r in range(row_start, row_start + half):
            for c in range(col_start, col_start + half):
                pattern[r * grid + c] = 0.9

        patterns.append(pattern)

    print("\n[DEBUG] Pattern similarities (should be ~0 for non-matching):")
    for i in range(n_patterns):
        for j in range(i+1, n_patterns):
            p1_norm = patterns[i] / (patterns[i].norm() + 1e-6)
            p2_norm = patterns[j] / (patterns[j].norm() + 1e-6)
            sim = (p1_norm * p2_norm).sum()
            print(f"  Pattern {i} vs {j}: {sim:.3f}")

    return patterns


def run_trial(
    brain: EventDrivenBrain,
    sample_pattern: torch.Tensor,
    test_pattern: torch.Tensor,
    is_match: bool,
    explore: bool = True,
) -> TrialResult:
    """Run a single trial of delayed match-to-sample."""

    # Between-trial rest (optional - skipped for demo speed)
    # brain.inter_trial_interval(n_timesteps=5)

    # 1. Encode sample - process multiple timesteps
    for _ in range(15):
        sample_result = brain.process_sample(sample_pattern)

    # 2. Delay period - PFC maintains working memory
    for _ in range(10):
        delay_result = brain.delay()

    # 3. Process test pattern
    for _ in range(15):
        test_result = brain.process_test(test_pattern)

    # 4. Select action based on comparison
    action, confidence = brain.select_action()

    # 5. Compute reward and learn
    correct_action = 0 if is_match else 1  # 0=MATCH, 1=NO-MATCH
    correct = (action == correct_action)
    reward = 1.0 if correct else -1.0

    brain.deliver_reward(reward)
    
    # 6. Store experience for later replay during sleep
    brain.store_experience(
        is_match=is_match,
        selected_action=action,
        correct=correct,
        reward=reward,
        sample_pattern=sample_pattern,
        test_pattern=test_pattern,
    )

    # Extract results - handle tensor vs scalar
    cortex_activity = sample_result.get("cortex_activity", 0.0)
    pfc_activity = sample_result.get("pfc_activity", 0.0)
    striatum_activity = test_result.get("striatum_activity", torch.zeros(2))

    # Convert tensor to scalar if needed
    if isinstance(cortex_activity, torch.Tensor):
        cortex_activity = cortex_activity.mean().item()
    if isinstance(pfc_activity, torch.Tensor):
        pfc_activity = pfc_activity.mean().item()

    return TrialResult(
        sample_pattern=0,  # Placeholder
        test_pattern=0,    # Placeholder
        is_match=is_match,
        selected_action=action,
        correct=correct,
        explored=False,  # Not tracked in EventDrivenBrain
        cortex_activity=float(cortex_activity),
        pfc_activity=float(pfc_activity),
        striatum_activity=striatum_activity,
        confidence=confidence,
    )


def run_epoch(
    brain: EventDrivenBrain,
    patterns: List[torch.Tensor],
    n_trials: int = 20,
    match_prob: float = 0.5,
) -> Dict[str, float]:
    """Run one epoch of training trials."""
    results = []

    for _ in range(n_trials):
        # Select patterns
        sample_idx = np.random.randint(len(patterns))
        is_match = np.random.random() < match_prob
        test_idx = sample_idx if is_match else np.random.choice(
            [i for i in range(len(patterns)) if i != sample_idx]
        )

        sample = patterns[sample_idx]
        test = patterns[test_idx]

        result = run_trial(brain, sample, test, is_match)
        results.append(result)

    # Compute epoch statistics
    n_correct = sum(1 for r in results if r.correct)
    n_explored = sum(1 for r in results if r.explored)

    return {
        "accuracy": n_correct / n_trials,
        "exploration_rate": n_explored / n_trials,
        "n_trials": n_trials,
    }


def main():
    """Run the demo."""
    print("\n" + "="*60)
    print(" EventDrivenBrain Demo: Delayed Match-to-Sample")
    print("="*60)

    # Configuration
    n_patterns = 4
    pattern_size = 256
    n_epochs = 10
    trials_per_epoch = 20
    sleep_every = 5

    # Create brain
    print("\n[1] Creating EventDrivenBrain...")
    brain = EventDrivenBrain(EventDrivenBrainConfig(
        input_size=pattern_size,
        cortex_size=96,
        hippocampus_size=64,
        pfc_size=20,
        n_actions=2,  # MATCH or NO-MATCH
    ))

    # Create patterns
    print("\n[2] Creating distinct patterns...")
    patterns = create_distinct_patterns(n_patterns, pattern_size)

    # Training loop
    print("\n[3] Starting training...")
    epoch_results = []

    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")

        result = run_epoch(brain, patterns, trials_per_epoch)
        epoch_results.append(result)

        print(f"  Accuracy: {result['accuracy']:.1%}")
        print(f"  Exploration: {result['exploration_rate']:.1%}")

        # Sleep consolidation
        if (epoch + 1) % sleep_every == 0 and epoch > 0:
            print(f"\n  [SLEEP] Running consolidation...")
            sleep_result = brain.sleep_epoch(
                n_cycles=2,
                replays_per_cycle=20,
            )
            print(f"  Sleep complete: {sleep_result.get('total_replays', 0)} replays")

    # Final results
    print("\n" + "="*60)
    print(" Training Complete!")
    print("="*60)

    early_acc = np.mean([r["accuracy"] for r in epoch_results[:3]])
    late_acc = np.mean([r["accuracy"] for r in epoch_results[-3:]])

    print(f"\n  Early epochs (1-3) accuracy: {early_acc:.1%}")
    print(f"  Late epochs ({n_epochs-2}-{n_epochs}) accuracy: {late_acc:.1%}")
    print(f"  Improvement: {(late_acc - early_acc) * 100:+.1f} percentage points")

    if late_acc > early_acc:
        print("\n  ✓ Learning detected!")
    else:
        print("\n  ✗ No improvement detected")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
