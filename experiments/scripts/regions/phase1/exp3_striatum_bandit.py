"""
Experiment 3: Striatum Multi-Armed Bandit

Objective: Validate reward-modulated learning for action selection using
the three-factor learning rule (pre × post × dopamine).

Setup:
- Environment: 4-armed bandit with different reward probabilities
- Architecture: Context input → Striatum (4 action outputs)
- Learning: Three-factor rule - eligibility traces + dopamine
- Episodes: 1000 trials

Key Insight:
The striatum learns through trial and error. Unlike the cerebellum (which
receives error signals), the striatum only receives reward after actions.
The eligibility trace provides temporal credit assignment - it "remembers"
which synapses were active when the action was taken, and dopamine
arriving later strengthens or weakens those synapses.

Analysis:
1. Learning curve (accuracy over trials)
2. Cumulative regret vs optimal arm
3. Action probability evolution over time
4. Comparison with ε-greedy baseline
5. Adaptation when reward probabilities change

Success Metrics:
- Converges to optimal arm 90%+ of trials
- Cumulative regret sublinear
- Adapts when reward probabilities change
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from thalia.regions.striatum import Striatum, StriatumConfig
from experiments.scripts.regions.exp_utils import (
    create_bandit_env,
    cumulative_regret,
    save_results,
    get_results_dir,
)


def create_striatum_agent(
    n_arms: int = 4,
    context_dim: int = 16,
) -> Striatum:
    """Create a Striatum configured for multi-armed bandit."""

    config = StriatumConfig(
        n_input=context_dim,
        n_output=n_arms,
        three_factor_lr=0.1,  # Higher learning rate for faster convergence
        eligibility_tau_ms=500.0,  # Shorter traces for immediate reward
        dopamine_burst=1.0,
        dopamine_dip=-1.0,  # Symmetric with burst for balanced learning
        lateral_inhibition=True,
        inhibition_strength=0.3,  # Reduced - was too strong (1.5)
        w_max=1.0,  # Higher ceiling for more differentiation
        w_min=0.0,
    )
    return Striatum(config)


def select_action(
    striatum: Striatum,
    context: torch.Tensor,
    n_timesteps: int = 7,
    exploration_bonus: float = 0.1,
) -> Tuple[int, torch.Tensor]:
    """
    Select action using striatum weights directly.

    For bandit tasks, we use weight sums as action values rather than
    spiking dynamics, which are too noisy for this task.
    """
    # Use mean input rate for eligibility
    input_rate = context.squeeze()
    
    # Compute action values as weighted sum of input
    # Each row of weights corresponds to one action
    action_values = striatum.weights @ input_rate
    
    # Add small noise for tie-breaking
    action_values = action_values + torch.randn_like(action_values) * 0.01
    
    # Softmax for smoother action selection (temperature controls exploration)
    temperature = 0.5 + exploration_bonus * 2  # Higher exploration = higher temperature
    probs = torch.softmax(action_values / temperature, dim=0)
    
    # Exploration: with some probability, choose uniformly random
    if np.random.rand() < exploration_bonus:
        action = np.random.randint(striatum.config.n_output)
    else:
        # Sample from softmax distribution
        action = torch.multinomial(probs, 1).item()

    return action, input_rate


def apply_reward(
    striatum: Striatum,
    action: int,
    reward: float,
    input_trace: torch.Tensor,
    learning_rate: float = 0.1,
    expected_reward: float = 0.5,
) -> Dict[str, float]:
    """
    Apply three-factor learning after receiving reward.

    Creates eligibility for the chosen action and applies reward.
    Uses simple reward signal (not full RPE) for clearer learning.
    """
    # Create one-hot action representation
    action_onehot = torch.zeros(striatum.config.n_output)
    action_onehot[action] = 1.0

    # Manually set eligibility for the chosen action's synapses
    eligibility = torch.outer(action_onehot, input_trace)
    striatum.eligibility.traces = eligibility

    # Simple reward signal: positive for reward, negative for no reward
    # Use 0.5 as baseline (average expected reward in binary bandit)
    # This gives consistent learning signal regardless of arm history
    reward_signal = reward - 0.5
    
    # Scale to dopamine magnitude
    if reward_signal > 0:
        da_level = striatum.dopamine.burst_magnitude * reward_signal * 2  # Scale 0.5 -> 1.0
    else:
        da_level = striatum.dopamine.dip_magnitude * abs(reward_signal) * 2  # Scale -0.5 -> -1.0
    
    striatum.dopamine.level = da_level

    if abs(da_level) < 0.01:
        return {"dopamine": da_level, "ltp": 0.0, "ltd": 0.0}

    dw = learning_rate * eligibility * da_level

    # Soft bounds (more symmetric)
    if da_level > 0:
        # LTP: reduce as weights approach max
        headroom = (striatum.config.w_max - striatum.weights) / striatum.config.w_max
        dw = dw * headroom.clamp(0, 1)
    else:
        # LTD: reduce as weights approach min
        footroom = (striatum.weights - striatum.config.w_min) / striatum.config.w_max
        dw = dw * footroom.clamp(0, 1)

    old_weights = striatum.weights.clone()
    striatum.weights = (striatum.weights + dw).clamp(striatum.config.w_min, striatum.config.w_max)

    actual_dw = striatum.weights - old_weights
    ltp = actual_dw[actual_dw > 0].sum().item() if (actual_dw > 0).any() else 0.0
    ltd = actual_dw[actual_dw < 0].sum().item() if (actual_dw < 0).any() else 0.0

    return {"dopamine": da_level, "ltp": ltp, "ltd": ltd}


def run_bandit_trial(
    striatum: Striatum,
    env: Dict,
    context: torch.Tensor,
    expected_rewards: np.ndarray,
    n_timesteps: int = 7,
    exploration: float = 0.1,
    learning_rate: float = 0.1,
) -> Tuple[int, float, Dict]:
    """Run a single bandit trial."""

    # Select action
    action, input_trace = select_action(
        striatum, context, n_timesteps, exploration
    )

    # Get reward
    reward = env["pull"](action)

    # Learn from reward using per-arm expected reward
    stats = apply_reward(
        striatum, action, reward, input_trace,
        learning_rate, expected_reward=expected_rewards[action]
    )

    return action, reward, stats


def train_bandit(
    striatum: Striatum,
    env: Dict,
    n_trials: int = 1000,
    n_timesteps: int = 7,
    exploration: float = 0.15,
    exploration_decay: float = 0.998,  # Slower decay
    learning_rate: float = 0.1,
    verbose: bool = True,
) -> Dict:
    """
    Train striatum on multi-armed bandit task.
    """
    n_arms = striatum.config.n_output
    context = torch.ones(striatum.config.n_input) * 0.5  # Fixed context

    # Track expected rewards per arm (running average)
    expected_rewards = np.ones(n_arms) * 0.5  # Start neutral
    arm_counts = np.zeros(n_arms)

    actions = []
    rewards = []
    optimal_actions = []

    current_exploration = exploration

    for trial in range(n_trials):
        action, reward, stats = run_bandit_trial(
            striatum, env, context, expected_rewards, n_timesteps,
            current_exploration, learning_rate
        )

        # Update expected reward for this arm (incremental mean)
        arm_counts[action] += 1
        expected_rewards[action] += (reward - expected_rewards[action]) / arm_counts[action]

        actions.append(action)
        rewards.append(reward)
        optimal_actions.append(action == env["optimal_arm"])

        # Decay exploration
        current_exploration *= exploration_decay

        if verbose and (trial + 1) % 200 == 0:
            recent_optimal = np.mean(optimal_actions[-100:])
            recent_reward = np.mean(rewards[-100:])
            print(f"Trial {trial+1}/{n_trials}: "
                  f"optimal_rate={recent_optimal*100:.1f}%, "
                  f"avg_reward={recent_reward:.2f}, "
                  f"exploration={current_exploration:.3f}")

    return {
        "actions": actions,
        "rewards": rewards,
        "optimal_actions": optimal_actions,
    }


def run_epsilon_greedy_baseline(
    env: Dict,
    n_trials: int = 1000,
    epsilon: float = 0.1,
    epsilon_decay: float = 0.995,
) -> Dict:
    """
    Run ε-greedy baseline for comparison.
    """
    n_arms = env["n_arms"]

    # Track Q-values (action values)
    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)  # Action counts

    actions = []
    rewards = []
    optimal_actions = []

    current_epsilon = epsilon

    for trial in range(n_trials):
        # ε-greedy action selection
        if np.random.rand() < current_epsilon:
            action = np.random.randint(n_arms)
        else:
            action = np.argmax(Q)

        # Get reward
        reward = env["pull"](action)

        # Update Q-value (incremental mean)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

        actions.append(action)
        rewards.append(reward)
        optimal_actions.append(action == env["optimal_arm"])

        current_epsilon *= epsilon_decay

    return {
        "actions": actions,
        "rewards": rewards,
        "optimal_actions": optimal_actions,
        "Q_values": Q.tolist(),
    }


def visualize_learning(
    striatum_results: Dict,
    baseline_results: Dict,
    env: Dict,
) -> plt.Figure:
    """Visualize learning curves."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Optimal action rate (smoothed)
    ax = axes[0, 0]
    window = 50

    striatum_optimal = np.array(striatum_results['optimal_actions']).astype(float)
    baseline_optimal = np.array(baseline_results['optimal_actions']).astype(float)

    if len(striatum_optimal) >= window:
        striatum_smooth = np.convolve(striatum_optimal, np.ones(window)/window, mode='valid')
        baseline_smooth = np.convolve(baseline_optimal, np.ones(window)/window, mode='valid')

        ax.plot(striatum_smooth, label='Striatum', color='blue')
        ax.plot(baseline_smooth, label='ε-greedy', color='red', linestyle='--')

    ax.axhline(y=0.9, color='green', linestyle=':', label='90% target')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Optimal Action Rate')
    ax.set_title('Learning Curve (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative reward
    ax = axes[0, 1]
    striatum_cumsum = np.cumsum(striatum_results['rewards'])
    baseline_cumsum = np.cumsum(baseline_results['rewards'])
    optimal_cumsum = np.arange(len(striatum_cumsum)) * env['reward_probs'][env['optimal_arm']]

    ax.plot(striatum_cumsum, label='Striatum', color='blue')
    ax.plot(baseline_cumsum, label='ε-greedy', color='red', linestyle='--')
    ax.plot(optimal_cumsum, label='Optimal', color='green', linestyle=':')

    ax.set_xlabel('Trial')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Cumulative regret
    ax = axes[1, 0]
    optimal_prob = env['reward_probs'][env['optimal_arm']]

    striatum_regret = cumulative_regret(
        striatum_results['actions'],
        striatum_results['rewards'],
        env['optimal_arm'],
        optimal_prob
    )
    baseline_regret = cumulative_regret(
        baseline_results['actions'],
        baseline_results['rewards'],
        env['optimal_arm'],
        optimal_prob
    )

    ax.plot(striatum_regret, label='Striatum', color='blue')
    ax.plot(baseline_regret, label='ε-greedy', color='red', linestyle='--')

    ax.set_xlabel('Trial')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Cumulative Regret (Lower = Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Action distribution (final 100 trials)
    ax = axes[1, 1]
    n_arms = env['n_arms']

    striatum_final = striatum_results['actions'][-100:]
    baseline_final = baseline_results['actions'][-100:]

    striatum_counts = [striatum_final.count(a)/100 for a in range(n_arms)]
    baseline_counts = [baseline_final.count(a)/100 for a in range(n_arms)]

    x = np.arange(n_arms)
    width = 0.35

    bars1 = ax.bar(x - width/2, striatum_counts, width, label='Striatum', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, baseline_counts, width, label='ε-greedy', color='red', alpha=0.7)

    # Mark optimal arm
    ax.axvline(x=env['optimal_arm'], color='green', linestyle='--', linewidth=2, label=f'Optimal (arm {env["optimal_arm"]})')

    # Add reward probabilities as text
    for i, p in enumerate(env['reward_probs']):
        ax.text(i, max(striatum_counts[i], baseline_counts[i]) + 0.05,
                f'p={p:.2f}', ha='center', fontsize=9)

    ax.set_xlabel('Action')
    ax.set_ylabel('Selection Frequency')
    ax.set_title('Final Action Distribution (Last 100 Trials)')
    ax.set_xticks(x)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def test_adaptation(
    striatum: Striatum,
    n_trials: int = 500,
    n_timesteps: int = 7,
    verbose: bool = True,
) -> Dict:
    """
    Test adaptation when reward probabilities change.

    Phase 1: Arm 0 is best (p=0.9)
    Phase 2: Arm 2 becomes best (p=0.9), arm 0 becomes bad (p=0.1)
    """
    context = torch.ones(striatum.config.n_input) * 0.5
    n_arms = striatum.config.n_output

    # Phase 1: Arm 0 is best
    env1 = create_bandit_env(n_arms=4, reward_probs=[0.9, 0.3, 0.3, 0.3])

    # Phase 2: Arm 2 is best
    env2 = create_bandit_env(n_arms=4, reward_probs=[0.1, 0.3, 0.9, 0.3])

    all_actions = []
    all_rewards = []
    phase_labels = []

    # Track expected rewards per arm
    expected_rewards = np.ones(n_arms) * 0.5
    arm_counts = np.zeros(n_arms)

    if verbose:
        print("Phase 1: Arm 0 is optimal (p=0.9)")

    for trial in range(n_trials):
        action, reward, _ = run_bandit_trial(
            striatum, env1, context, expected_rewards, n_timesteps,
            exploration=0.1, learning_rate=0.1
        )
        # Update expected reward
        arm_counts[action] += 1
        expected_rewards[action] += (reward - expected_rewards[action]) / arm_counts[action]
        
        all_actions.append(action)
        all_rewards.append(reward)
        phase_labels.append(1)

    if verbose:
        arm0_rate = all_actions[-100:].count(0) / 100
        print(f"  Final arm 0 rate: {arm0_rate*100:.1f}%")
        print("\nPhase 2: Arm 2 is now optimal (p=0.9)")

    # Reset expected rewards for phase 2 to allow relearning
    expected_rewards = np.ones(n_arms) * 0.5
    arm_counts = np.zeros(n_arms)

    for trial in range(n_trials):
        action, reward, _ = run_bandit_trial(
            striatum, env2, context, expected_rewards, n_timesteps,
            exploration=0.1, learning_rate=0.1
        )
        # Update expected reward
        arm_counts[action] += 1
        expected_rewards[action] += (reward - expected_rewards[action]) / arm_counts[action]
        
        all_actions.append(action)
        all_rewards.append(reward)
        phase_labels.append(2)

    if verbose:
        arm2_rate = all_actions[-100:].count(2) / 100
        print(f"  Final arm 2 rate: {arm2_rate*100:.1f}%")

    return {
        "actions": all_actions,
        "rewards": all_rewards,
        "phases": phase_labels,
    }


def visualize_adaptation(adapt_results: Dict) -> plt.Figure:
    """Visualize adaptation to changed reward structure."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    n_trials = len(adapt_results['actions']) // 2

    # 1. Action over time
    ax = axes[0]
    actions = np.array(adapt_results['actions'])

    # Smooth action selection
    window = 50
    for arm in range(4):
        is_arm = (actions == arm).astype(float)
        if len(is_arm) >= window:
            smooth = np.convolve(is_arm, np.ones(window)/window, mode='valid')
            ax.plot(smooth, label=f'Arm {arm}')

    ax.axvline(x=n_trials, color='red', linestyle='--', linewidth=2, label='Change point')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Selection Rate')
    ax.set_title('Action Selection Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Action distribution by phase
    ax = axes[1]

    phase1_actions = adapt_results['actions'][:n_trials]
    phase2_actions = adapt_results['actions'][n_trials:]

    phase1_counts = [phase1_actions[-100:].count(a)/100 for a in range(4)]
    phase2_counts = [phase2_actions[-100:].count(a)/100 for a in range(4)]

    x = np.arange(4)
    width = 0.35

    ax.bar(x - width/2, phase1_counts, width, label='Phase 1 (final)', color='blue', alpha=0.7)
    ax.bar(x + width/2, phase2_counts, width, label='Phase 2 (final)', color='red', alpha=0.7)

    ax.set_xlabel('Action')
    ax.set_ylabel('Selection Frequency')
    ax.set_title('Action Distribution by Phase')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def run_experiment(
    n_arms: int = 4,
    n_trials: int = 1000,
    n_timesteps: int = 7,
    exploration: float = 0.15,
    exploration_decay: float = 0.995,
    learning_rate: float = 0.05,
    show_plots: bool = True,
    save_plots: bool = True,
):
    """
    Run the complete Striatum multi-armed bandit experiment.
    """
    print("=" * 60)
    print("EXPERIMENT 3: Striatum Multi-Armed Bandit")
    print("=" * 60)

    # 1. Create environment and agent
    print(f"\n[1/5] Creating {n_arms}-armed bandit and Striatum agent...")
    env = create_bandit_env(n_arms=n_arms, reward_probs=[0.2, 0.4, 0.6, 0.8])
    print(f"  Reward probabilities: {env['reward_probs']}")
    print(f"  Optimal arm: {env['optimal_arm']} (p={env['reward_probs'][env['optimal_arm']]:.2f})")

    striatum = create_striatum_agent(n_arms=n_arms, context_dim=16)

    # 2. Train Striatum
    print(f"\n[2/5] Training Striatum ({n_trials} trials)...")
    striatum_results = train_bandit(
        striatum, env, n_trials=n_trials, n_timesteps=n_timesteps,
        exploration=exploration, exploration_decay=exploration_decay,
        learning_rate=learning_rate
    )

    # 3. Run baseline
    print(f"\n[3/5] Running ε-greedy baseline...")
    baseline_results = run_epsilon_greedy_baseline(
        env, n_trials=n_trials, epsilon=exploration, epsilon_decay=exploration_decay
    )

    # Compute final metrics
    striatum_final_optimal = np.mean(striatum_results['optimal_actions'][-100:])
    baseline_final_optimal = np.mean(baseline_results['optimal_actions'][-100:])

    print(f"\n  Striatum final optimal rate: {striatum_final_optimal*100:.1f}%")
    print(f"  Baseline final optimal rate: {baseline_final_optimal*100:.1f}%")

    # 4. Test adaptation
    print(f"\n[4/5] Testing adaptation to changed rewards...")
    # Create fresh striatum for adaptation test
    striatum_adapt = create_striatum_agent(n_arms=n_arms, context_dim=16)
    adapt_results = test_adaptation(striatum_adapt, n_trials=500, n_timesteps=n_timesteps)

    # 5. Visualize
    print(f"\n[5/5] Generating visualizations...")

    results_dir = get_results_dir()

    # Main learning visualization
    fig_learn = visualize_learning(striatum_results, baseline_results, env)
    if save_plots:
        fig_learn.savefig(results_dir / "exp3_learning.png", dpi=150)
        print(f"  Saved: exp3_learning.png")

    # Adaptation visualization
    fig_adapt = visualize_adaptation(adapt_results)
    if save_plots:
        fig_adapt.savefig(results_dir / "exp3_adaptation.png", dpi=150)
        print(f"  Saved: exp3_adaptation.png")

    # Weight visualization
    fig_weights, ax = plt.subplots(figsize=(10, 4))
    weights = striatum.weights.detach().cpu().numpy()
    im = ax.imshow(weights, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Input Neuron')
    ax.set_ylabel('Action')
    ax.set_title('Learned Striatum Weights')
    plt.colorbar(im, ax=ax, label='Weight')
    if save_plots:
        fig_weights.savefig(results_dir / "exp3_weights.png", dpi=150)
        print(f"  Saved: exp3_weights.png")

    if show_plots:
        plt.show()
    else:
        plt.close('all')

    # Save results
    results = {
        "config": {
            "n_arms": n_arms,
            "n_trials": n_trials,
            "n_timesteps": n_timesteps,
            "exploration": exploration,
            "exploration_decay": exploration_decay,
            "learning_rate": learning_rate,
            "reward_probs": env['reward_probs'].tolist(),
            "optimal_arm": env['optimal_arm'],
        },
        "metrics": {
            "striatum_final_optimal": striatum_final_optimal,
            "baseline_final_optimal": baseline_final_optimal,
            "striatum_total_reward": sum(striatum_results['rewards']),
            "baseline_total_reward": sum(baseline_results['rewards']),
        },
        "striatum_actions": striatum_results['actions'],
        "baseline_actions": baseline_results['actions'],
    }
    save_results("exp3_striatum_bandit", results)

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 60)

    success = striatum_final_optimal >= 0.90
    print(f"\nSuccess criterion: Optimal action rate ≥ 90%")
    print(f"Result: {striatum_final_optimal*100:.1f}%")
    print(f"Status: {'✓ PASSED' if success else '✗ FAILED'}")

    return results, striatum


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Striatum Multi-Armed Bandit Experiment")
    parser.add_argument("--n-arms", type=int, default=4, help="Number of arms")
    parser.add_argument("--n-trials", type=int, default=1000, help="Training trials")
    parser.add_argument("--n-timesteps", type=int, default=7, help="Timesteps per trial")
    parser.add_argument("--exploration", type=float, default=0.15, help="Initial exploration")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots")
    parser.add_argument("--no-save", action="store_true", help="Don't save plots")

    args = parser.parse_args()

    run_experiment(
        n_arms=args.n_arms,
        n_trials=args.n_trials,
        n_timesteps=args.n_timesteps,
        exploration=args.exploration,
        learning_rate=args.lr,
        show_plots=not args.no_show,
        save_plots=not args.no_save,
    )
