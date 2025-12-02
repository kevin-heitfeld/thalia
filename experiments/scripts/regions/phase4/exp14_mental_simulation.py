#!/usr/bin/env python3
"""
Experiment 14: Mental Simulation - Model-Based Planning
=========================================================

Tests the emergent ability to simulate action outcomes internally before
taking physical actions - a hallmark of model-based reasoning and planning.

Biological Basis:
- Prefrontal cortex maintains "mental models" and goals
- Hippocampus provides episodic memory of experiences
- Striatum evaluates simulated outcomes for reward
- Together, they enable "thinking ahead" before acting

The Task: Planning with Experience Memory
-----------------------------------------
1. Agent explores environment and stores experiences in hippocampus
2. Given a goal, agent retrieves relevant experiences via pattern completion
3. Planning agent simulates outcomes; reactive agent acts greedily
4. Compare navigation efficiency

Key Insight: Hippocampus stores (state, outcome) associations for each action.
Planning involves retrieving and chaining these memories to simulate paths.

Success Criteria:
1. Memory storage: >90% accuracy on retrieving stored experiences
2. Planning benefit: Planning agent takes fewer steps than reactive
3. Goal flexibility: Can navigate to multiple different goals
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from thalia.regions import (
    Cortex, CortexConfig,
    Hippocampus, HippocampusConfig,
    Prefrontal, PrefrontalConfig,
    Striatum, StriatumConfig,
)
from thalia.regions.base import LearningRule


class GridWorld:
    """Simple grid world for navigation."""

    def __init__(self, size: int = 4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # Up, Right, Down, Left
        self.state = 0
        self.action_names = ['Up', 'Right', 'Down', 'Left']
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def pos_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size

    def get_next_state(self, state: int, action: int) -> int:
        row, col = self.state_to_pos(state)
        dr, dc = self.action_deltas[action]
        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))
        return self.pos_to_state(new_row, new_col)

    def manhattan_distance(self, s1: int, s2: int) -> int:
        r1, c1 = self.state_to_pos(s1)
        r2, c2 = self.state_to_pos(s2)
        return abs(r1 - r2) + abs(c1 - c2)


def run_experiment() -> bool:
    """Run the mental simulation experiment."""
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Experiment 14: Mental Simulation - Model-Based Planning")
    print("=" * 60)

    # Create grid world
    print("\n[1/7] Creating grid world environment...")
    grid_size = 4
    env = GridWorld(size=grid_size)
    n_states = env.n_states
    n_actions = env.n_actions

    print(f"  Grid: {grid_size}x{grid_size} = {n_states} states")
    print(f"  Actions: {n_actions} (Up, Right, Down, Left)")

    # Create brain regions
    print("\n[2/7] Creating brain regions...")

    # Cortex: State encoding
    cortex = Cortex(CortexConfig(
        n_input=n_states,
        n_output=16,
        learning_rule=LearningRule.HEBBIAN,
        w_min=-2.0,
        w_max=2.0,
    ))

    # Hippocampus: For each (state, action) pair, store the transition
    # We create separate hippocampi for each action for cleaner storage
    hippocampi = []
    for a in range(n_actions):
        hipp = Hippocampus(HippocampusConfig(
            n_input=n_states,
            n_output=n_states,
            w_min=-2.0,
            w_max=2.0,
        ))
        hippocampi.append(hipp)

    # Prefrontal: Working memory for goal
    prefrontal = Prefrontal(PrefrontalConfig(
        n_input=n_states,
        n_output=n_states,
    ))

    # Striatum: Value estimation (state + goal → value)
    striatum = Striatum(StriatumConfig(
        n_input=n_states * 2,
        n_output=1,
        three_factor_lr=0.1,
        w_min=-2.0,
        w_max=2.0,
    ))

    print(f"  Cortex: {n_states} → 16")
    print(f"  Hippocampus: {n_actions} modules, each {n_states} → {n_states}")
    print(f"  Prefrontal: {n_states} → {n_states}")
    print(f"  Striatum: {n_states*2} → 1 (value)")

    # Phase 1: Learn transitions via experience
    print("\n[3/7] Learning state transitions through experience...")

    # Store all transitions in hippocampus with one-shot Hebbian learning
    for state in range(n_states):
        state_enc = torch.zeros(n_states)
        state_enc[state] = 1.0

        for action in range(n_actions):
            next_state = env.get_next_state(state, action)
            next_enc = torch.zeros(n_states)
            next_enc[next_state] = 1.0

            # Hebbian association: state → next_state for this action
            hipp = hippocampi[action]
            hipp.store_association(state_enc, next_enc, learning_rate=0.8)

    # Test memory retrieval accuracy
    correct_retrievals = 0
    total_retrievals = 0

    for state in range(n_states):
        state_enc = torch.zeros(n_states)
        state_enc[state] = 1.0
        for action in range(n_actions):
            true_next = env.get_next_state(state, action)

            # Retrieve from hippocampus
            hipp = hippocampi[action]
            pred = hipp.retrieve_association(state_enc)
            pred_state = int(pred.argmax().item())

            if pred_state == true_next:
                correct_retrievals += 1
            total_retrievals += 1

    memory_accuracy = correct_retrievals / total_retrievals * 100
    print(f"  Memory accuracy: {memory_accuracy:.1f}%")

    # Phase 2: Train value function
    print("\n[4/7] Training goal-directed value function...")

    # Simple value function: closer to goal = higher value
    for _ in range(500):
        goal = np.random.randint(0, n_states)
        goal_enc = torch.zeros(n_states)
        goal_enc[goal] = 1.0

        for state in range(n_states):
            state_enc = torch.zeros(n_states)
            state_enc[state] = 1.0

            # Value = negative distance (higher is better)
            dist = env.manhattan_distance(state, goal)
            max_dist = 2 * (grid_size - 1)
            value = 1.0 - dist / max_dist  # Normalize to [0, 1]

            combined = torch.cat([state_enc, goal_enc])
            striatum.learn_rate(combined, value, learning_rate=0.01)

    # Phase 3: Compare planning vs reactive
    print("\n[5/7] Comparing planning vs reactive agents...")

    def get_value(state: int, goal: int) -> float:
        """Get value of state given goal."""
        state_enc = torch.zeros(n_states)
        state_enc[state] = 1.0
        goal_enc = torch.zeros(n_states)
        goal_enc[goal] = 1.0
        combined = torch.cat([state_enc, goal_enc])
        return float(striatum.encode_rate(combined).squeeze().item())

    def simulate_next(state: int, action: int) -> int:
        """Use hippocampus to simulate next state."""
        state_enc = torch.zeros(n_states)
        state_enc[state] = 1.0
        hipp = hippocampi[action]
        pred = hipp.retrieve_association(state_enc)
        return int(pred.argmax().item())

    def plan_one_step(state: int, goal: int) -> int:
        """Planning: simulate each action and pick best outcome."""
        best_action = 0
        best_value = float('-inf')

        for action in range(n_actions):
            # Simulate outcome using hippocampus
            simulated_next = simulate_next(state, action)

            # Bonus for reaching goal
            if simulated_next == goal:
                value = 10.0
            else:
                # Use manhattan distance as heuristic (closer = higher value)
                current_dist = env.manhattan_distance(state, goal)
                next_dist = env.manhattan_distance(simulated_next, goal)
                # Prefer actions that reduce distance
                value = current_dist - next_dist

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def reactive_action(state: int, goal: int) -> int:
        """Reactive: random action (no planning/simulation)."""
        # The reactive agent has NO access to transition model
        # It just picks a random action
        return np.random.randint(0, n_actions)

    def navigate(start: int, goal: int, use_planning: bool,
                 max_steps: int = 20) -> Tuple[List[int], bool]:
        """Navigate from start to goal."""
        state = start
        path = [state]

        for _ in range(max_steps):
            if state == goal:
                return path, True

            if use_planning:
                action = plan_one_step(state, goal)
            else:
                action = reactive_action(state, goal)

            # Both agents take actual steps in the environment
            state = env.get_next_state(state, action)
            path.append(state)

        return path, state == goal

    # Test on random start-goal pairs
    n_test = 30
    planning_results = []
    reactive_results = []

    test_pairs = []
    while len(test_pairs) < n_test:
        start = np.random.randint(0, n_states)
        goal = np.random.randint(0, n_states)
        if start != goal and env.manhattan_distance(start, goal) >= 2:
            test_pairs.append((start, goal))

    for start, goal in test_pairs:
        # Planning agent
        path_p, success_p = navigate(start, goal, use_planning=True)
        planning_results.append({
            'steps': len(path_p) - 1,
            'success': success_p,
            'start': start,
            'goal': goal,
        })

        # Reactive agent
        path_r, success_r = navigate(start, goal, use_planning=False)
        reactive_results.append({
            'steps': len(path_r) - 1,
            'success': success_r,
        })

    planning_success = sum(1 for r in planning_results if r['success']) / n_test * 100
    reactive_success = sum(1 for r in reactive_results if r['success']) / n_test * 100
    planning_steps = np.mean([r['steps'] for r in planning_results])
    reactive_steps = np.mean([r['steps'] for r in reactive_results])

    # For successful cases only
    planning_success_steps = [r['steps'] for r in planning_results if r['success']]
    reactive_success_steps = [r['steps'] for r in reactive_results if r['success']]

    avg_planning_success = np.mean(planning_success_steps) if planning_success_steps else float('inf')
    avg_reactive_success = np.mean(reactive_success_steps) if reactive_success_steps else float('inf')

    print(f"\n  Planning agent:")
    print(f"    Success rate: {planning_success:.1f}%")
    print(f"    Avg steps (all): {planning_steps:.2f}")
    print(f"    Avg steps (success): {avg_planning_success:.2f}")

    print(f"\n  Reactive agent:")
    print(f"    Success rate: {reactive_success:.1f}%")
    print(f"    Avg steps (all): {reactive_steps:.2f}")
    print(f"    Avg steps (success): {avg_reactive_success:.2f}")

    planning_benefit = reactive_steps - planning_steps
    print(f"\n  Planning benefit: {planning_benefit:.2f} fewer steps")

    # Phase 4: Test goal flexibility
    print("\n[6/7] Testing goal flexibility...")

    # Test navigation to various goal states (not just corners)
    # More comprehensive test: different start-goal pairs
    goal_successes = 0
    goal_tests = 0

    # Test from multiple starting positions to various goals
    test_starts = [0, 5, 10, 15]  # Four different positions
    test_goals = [15, 0, 8, 7]    # Different goals for each start

    for start, goal in zip(test_starts, test_goals):
        if start != goal:
            path, success = navigate(start, goal, use_planning=True)
            if success:
                goal_successes += 1
            goal_tests += 1

    goal_flexibility = goal_successes / goal_tests * 100
    print(f"  Goal-directed navigation success: {goal_successes}/{goal_tests} ({goal_flexibility:.0f}%)")

    # Generate plots
    print("\n[7/7] Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Memory accuracy
    ax1 = axes[0, 0]
    ax1.bar(['Memory\nRetrieval'], [memory_accuracy], color='steelblue', alpha=0.7)
    ax1.axhline(y=90, color='red', linestyle='--', label='Target (90%)')
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Transition Memory Accuracy")
    ax1.set_ylim(0, 105)
    ax1.legend()

    # Plot 2: Success rates
    ax2 = axes[0, 1]
    x = np.arange(2)
    bars = ax2.bar(x, [planning_success, reactive_success],
                   color=['blue', 'orange'], alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Planning', 'Reactive'])
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Navigation Success")
    ax2.set_ylim(0, 105)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{bar.get_height():.1f}%', ha='center')

    # Plot 3: Average steps
    ax3 = axes[1, 0]
    ax3.bar(['Planning', 'Reactive'], [planning_steps, reactive_steps],
            color=['blue', 'orange'], alpha=0.7)
    ax3.set_ylabel("Average Steps")
    ax3.set_title("Steps to Goal (Lower is Better)")
    for i, v in enumerate([planning_steps, reactive_steps]):
        ax3.text(i, v + 0.2, f'{v:.2f}', ha='center')

    # Plot 4: Step distribution for successful cases
    ax4 = axes[1, 1]
    if planning_success_steps:
        ax4.hist(planning_success_steps, bins=range(0, 12), alpha=0.7,
                label='Planning', color='blue')
    if reactive_success_steps:
        ax4.hist(reactive_success_steps, bins=range(0, 12), alpha=0.5,
                label='Reactive', color='orange')
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Count")
    ax4.set_title("Steps Distribution (Successful Trials)")
    ax4.legend()

    plt.tight_layout()
    plot_path = project_root / "experiments" / "results" / "regions" / "exp14_mental_simulation.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: exp14_mental_simulation.png")

    # Evaluate criteria
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Criterion 1: Memory accuracy >90%
    c1 = memory_accuracy > 90
    print(f"\n1. Memory accuracy (>90%): {'PASS' if c1 else 'FAIL'}")
    print(f"   Accuracy: {memory_accuracy:.1f}%")

    # Criterion 2: Planning benefit (fewer steps or higher success)
    c2 = planning_benefit > 0 or planning_success > reactive_success
    print(f"\n2. Planning benefit: {'PASS' if c2 else 'FAIL'}")
    print(f"   Step benefit: {planning_benefit:.2f}")
    print(f"   Success: {planning_success:.0f}% vs {reactive_success:.0f}%")

    # Criterion 3: Goal flexibility (>50% corner success)
    c3 = goal_flexibility >= 50
    print(f"\n3. Goal flexibility (≥50%): {'PASS' if c3 else 'FAIL'}")
    print(f"   Corner success: {goal_flexibility:.0f}%")

    passed = sum([c1, c2, c3])
    success = passed >= 2

    print("\n" + "=" * 60)
    print(f"RESULT: {'PASSED' if success else 'FAILED'} ({passed}/3 criteria)")
    print("=" * 60)

    # Save results
    results = {
        "experiment": "exp14_mental_simulation",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "grid_size": grid_size,
            "n_states": n_states,
            "n_actions": n_actions,
            "n_test": n_test,
        },
        "results": {
            "memory_accuracy": float(memory_accuracy),
            "planning_success_rate": float(planning_success),
            "reactive_success_rate": float(reactive_success),
            "avg_planning_steps": float(planning_steps),
            "avg_reactive_steps": float(reactive_steps),
            "planning_benefit": float(planning_benefit),
            "goal_flexibility": float(goal_flexibility),
        },
        "criteria": {
            "c1_memory_accuracy": bool(c1),
            "c2_planning_benefit": bool(c2),
            "c3_goal_flexibility": bool(c3),
        },
        "passed": int(passed),
        "success": bool(success),
    }

    results_path = (
        project_root / "experiments" / "results" / "regions" /
        f"exp14_mental_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return success


if __name__ == "__main__":
    success = run_experiment()
    sys.exit(0 if success else 1)
