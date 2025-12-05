#!/usr/bin/env python3
"""
Experiment 01: Sleep Importance for Learning

This experiment tests how important sleep consolidation is for learning
in the integrated BrainSystem. We compare two conditions:

1. NO SLEEP: Wake learning only (online learning during trials)
2. REALISTIC SLEEP: Full sleep cycle with N1→N2→SWS→REM stages

Hypothesis:
- Sleep should improve learning, especially for:
  - Memory retention over time
  - Generalization to new patterns
  - Reducing interference between memories

- Realistic sleep should outperform no sleep by:
  - SWS: Strong consolidation of important memories
  - REM: Generalization and avoiding overfitting
  - N2 spindles: Hippo→Cortex transfer

Task: Delayed Match-to-Sample (DMS)
- See sample pattern
- Delay period (WM maintenance)
- See test pattern
- Decide: MATCH or NO-MATCH
- Receive reward signal

This task requires:
- Cortex: Feature extraction
- Hippocampus: Store sample, compare at test
- PFC: Maintain working memory during delay
- Striatum: Action selection based on comparison
- All regions working together!

Author: Thalia Project
Date: December 2025
"""

import argparse
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any
import random

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

from thalia.integration import BrainSystem, BrainSystemConfig


def print_diagnostics_summary(diag: Dict[str, Any], title: str = "DIAGNOSTICS") -> None:
    """Print a nicely formatted summary of brain diagnostics.

    Args:
        diag: Diagnostics dict from brain.get_diagnostics()
        title: Title for the summary block
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    summary = diag.get("summary", {})

    # Action selection
    print(f"\n  ACTION SELECTION:")
    print(f"    Last action: {summary.get('last_action', '?')} "
          f"(exploring: {summary.get('exploring', False)})")

    # D1/D2 weights (per action)
    net_means = summary.get("net_weight_means", [])
    if net_means:
        print(f"    NET weight means (D1-D2): MATCH={net_means[0]:+.4f}, NOMATCH={net_means[1]:+.4f}")

    # Votes
    net_votes = summary.get("net_votes", [])
    if net_votes:
        print(f"    NET votes this trial: MATCH={net_votes[0]:+.2f}, NOMATCH={net_votes[1]:+.2f}")

    # Hippocampus comparison (critical for match/mismatch)
    print(f"\n  HIPPOCAMPUS (Match/Mismatch Detection):")
    print(f"    CA1 spikes: {summary.get('ca1_spikes', 0)}")
    print(f"    NMDA above threshold: {summary.get('nmda_above_threshold', 0)}")
    print(f"    Mg block removal: {summary.get('mg_block_removal', 0.0):.3f}")
    print(f"    DG pattern similarity: {summary.get('dg_similarity', 0.0):.3f}")

    # Dopamine
    print(f"\n  NEUROMODULATION:")
    print(f"    Dopamine: {summary.get('dopamine', 0.0):.3f}")

    # Detailed region info if verbose
    regions = diag.get("regions", {})

    # Striatum details
    striatum = regions.get("striatum", {})
    if striatum:
        exploration = striatum.get("exploration", {})
        ucb = striatum.get("ucb", {})
        print(f"\n  STRIATUM:")
        print(f"    Recent accuracy: {exploration.get('recent_accuracy', 0.5):.2%}")
        print(f"    Exploration prob: {exploration.get('last_exploration_prob', 0.0):.2%}")
        action_counts = ucb.get("action_counts", [])
        if action_counts:
            print(f"    Action counts: MATCH={action_counts[0]:.0f}, NOMATCH={action_counts[1]:.0f}")

    # Hippocampus details
    hippo = regions.get("hippocampus", {})
    if hippo:
        nmda = hippo.get("nmda", {})
        print(f"\n  HIPPOCAMPUS (detailed):")
        print(f"    NMDA trace: mean={nmda.get('trace_mean', 0.0):.3f}, "
              f"max={nmda.get('trace_max', 0.0):.3f}")
        print(f"    NMDA threshold: {nmda.get('threshold', 0.0):.3f}")
        print(f"    Gated neurons: {nmda.get('gated_neurons', 0)}")

        pattern = hippo.get("pattern_comparison", {})
        if pattern:
            print(f"    Pattern overlap: {pattern.get('overlap', 0)}")
            print(f"    Stored active: {pattern.get('stored_active', 0)}, "
                  f"Current active: {pattern.get('current_active', 0)}")

    print(f"{'='*60}\n")


@dataclass
class HippocampusParams:
    """Hippocampus-specific parameters (configurable via CLI)."""
    nmda_tau: float = 50.0           # NMDA time constant (ms)
    nmda_threshold: float = 0.1      # Gate opening threshold
    nmda_steepness: float = 12.0     # Sigmoid sharpness
    ampa_ratio: float = 0.05         # Ungated AMPA contribution
    ec_ca1_learning_rate: float = 0.5  # EC→CA1 plasticity rate
    ca3_learning_rate: float = 0.2   # CA3 recurrent learning rate
    dg_sparsity: float = 0.02        # DG sparsity (2%)
    ca3_sparsity: float = 0.10       # CA3 sparsity (10%)
    ca1_sparsity: float = 0.15       # CA1 sparsity (15%)


@dataclass
class ExperimentConfig:
    """Configuration for the sleep importance experiment."""

    # Task settings
    n_patterns: int = 10  # Number of unique patterns
    pattern_size: int = 64  # Size of input patterns
    n_epochs: int = 20  # Training epochs
    trials_per_epoch: int = 50  # Trials per epoch
    match_probability: float = 0.5  # P(match trial)

    # Sleep settings
    sleep_every_n_epochs: int = 5  # How often to sleep
    sleep_cycles: int = 3  # Sleep cycles per sleep session
    replays_per_cycle: int = 30  # Replays per cycle

    # Network settings
    cortex_size: int = 64
    hippocampus_size: int = 32
    pfc_size: int = 16
    n_actions: int = 2  # MATCH (0) or NO-MATCH (1)

    # Hippocampus parameters
    hippo_params: HippocampusParams = field(default_factory=HippocampusParams)

    # STP and BCM settings
    enable_stp: bool = False  # Enable Short-Term Plasticity in hippocampus
    enable_bcm: bool = False  # Enable BCM sliding threshold in cortex

    # Curriculum learning settings
    enable_curriculum: bool = True  # Enable curriculum learning
    curriculum_warmup_epochs: int = 5  # Initial epochs with stronger signals

    # Evaluation
    n_test_trials: int = 100  # Trials for evaluation

    # Random seed
    seed: int = 42

    device: str = "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Experiment 01: Sleep Importance for Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task settings
    parser.add_argument("--n-patterns", type=int, default=10,
                        help="Number of unique patterns")
    parser.add_argument("--n-epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--trials-per-epoch", type=int, default=10,
                        help="Trials per epoch")
    parser.add_argument("--n-test-trials", type=int, default=100,
                        help="Number of trials for final evaluation")

    # Sleep settings
    parser.add_argument("--sleep-every", type=int, default=5,
                        help="Sleep every N epochs")
    parser.add_argument("--sleep-cycles", type=int, default=3,
                        help="Sleep cycles per session")

    # Hippocampus NMDA parameters
    parser.add_argument("--nmda-tau", type=float, default=50.0,
                        help="NMDA time constant (ms)")
    parser.add_argument("--nmda-threshold", type=float, default=0.1,
                        help="NMDA gate opening threshold")
    parser.add_argument("--nmda-steepness", type=float, default=12.0,
                        help="NMDA sigmoid steepness")
    parser.add_argument("--ampa-ratio", type=float, default=0.05,
                        help="AMPA ungated contribution ratio")

    # Hippocampus learning parameters
    parser.add_argument("--ec-ca1-lr", type=float, default=0.5,
                        help="EC→CA1 plasticity learning rate")
    parser.add_argument("--ca3-lr", type=float, default=0.2,
                        help="CA3 recurrent learning rate")

    # Hippocampus sparsity parameters
    parser.add_argument("--dg-sparsity", type=float, default=0.02,
                        help="Dentate Gyrus sparsity (fraction active)")
    parser.add_argument("--ca3-sparsity", type=float, default=0.10,
                        help="CA3 sparsity (fraction active)")
    parser.add_argument("--ca1-sparsity", type=float, default=0.15,
                        help="CA1 sparsity (fraction active)")

    # Other settings
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=["NO_SLEEP", "REALISTIC_SLEEP"],
                        help="Conditions to run")

    # STP and BCM options
    parser.add_argument("--enable-stp", action="store_true",
                        help="Enable Short-Term Plasticity in hippocampus")
    parser.add_argument("--enable-bcm", action="store_true",
                        help="Enable BCM sliding threshold in cortex")

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> ExperimentConfig:
    """Convert CLI args to ExperimentConfig."""
    hippo_params = HippocampusParams(
        nmda_tau=args.nmda_tau,
        nmda_threshold=args.nmda_threshold,
        nmda_steepness=args.nmda_steepness,
        ampa_ratio=args.ampa_ratio,
        ec_ca1_learning_rate=args.ec_ca1_lr,
        ca3_learning_rate=args.ca3_lr,
        dg_sparsity=args.dg_sparsity,
        ca3_sparsity=args.ca3_sparsity,
        ca1_sparsity=args.ca1_sparsity,
    )

    return ExperimentConfig(
        n_patterns=args.n_patterns,
        n_epochs=args.n_epochs,
        trials_per_epoch=args.trials_per_epoch,
        n_test_trials=args.n_test_trials,
        sleep_every_n_epochs=args.sleep_every,
        sleep_cycles=args.sleep_cycles,
        hippo_params=hippo_params,
        enable_stp=args.enable_stp,
        enable_bcm=args.enable_bcm,
        seed=args.seed,
        device=args.device,
    )


def generate_patterns(n_patterns: int, pattern_size: int) -> torch.Tensor:
    """Generate distinct patterns for the task."""
    patterns = torch.zeros(n_patterns, pattern_size)

    # Each pattern has ~25% active inputs in different regions
    active_per_pattern = pattern_size // 4

    for i in range(n_patterns):
        # Spread activations across the pattern with some overlap
        base_start = (i * active_per_pattern // 2) % (pattern_size - active_per_pattern)
        indices = list(range(base_start, base_start + active_per_pattern))

        # Add some random variation
        for _ in range(active_per_pattern // 4):
            indices.append(random.randint(0, pattern_size - 1))

        for idx in indices:
            patterns[i, idx % pattern_size] = 1.0

    return patterns


def run_trial(
    brain: BrainSystem,
    sample: torch.Tensor,
    test: torch.Tensor,
    is_match: bool,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single delayed match-to-sample trial.

    The brain ALWAYS LEARNS from each trial. There is no separate evaluation mode.
    Use brain.striatum.freeze_plasticity() context manager if you need frozen evaluation.
    """

    # Phase 1: Sample presentation
    sample_result = brain.process_sample(sample)

    # Phase 2: Delay period
    delay_result = brain.inter_trial_interval(n_timesteps=10, dopamine=-0.3)

    # Phase 3: Test and respond
    test_result = brain.process_test_and_respond(test, explore=True)

    selected_action = test_result["selected_action"]

    # Determine correctness
    # Action 0 = MATCH, Action 1 = NO-MATCH
    correct = (selected_action == 0 and is_match) or (selected_action == 1 and not is_match)
    reward = 1.0 if correct else -1.0

    # Collect diagnostic info
    diagnostic_info = {}
    if verbose:
        striatum = brain.striatum
        n_act = striatum.n_actions
        n_per = striatum.neurons_per_action

        # Per-action D1/D2 balance
        action_d1_means = []
        action_d2_means = []
        for a in range(n_act):
            start = a * n_per
            end = start + n_per
            action_d1_means.append(striatum.d1_weights[start:end].mean().item())
            action_d2_means.append(striatum.d2_weights[start:end].mean().item())

        # Per-action eligibility
        action_d1_elig = []
        action_d2_elig = []
        for a in range(n_act):
            start = a * n_per
            end = start + n_per
            action_d1_elig.append(striatum.d1_eligibility[start:end].abs().mean().item())
            action_d2_elig.append(striatum.d2_eligibility[start:end].abs().mean().item())

        # Track input signal strength
        combined_input = test_result.get("combined_input", None)
        input_strength = combined_input.sum().item() if combined_input is not None else 0

        diagnostic_info = {
            "g_E_mean": striatum.neurons.g_E.mean().item() if striatum.neurons.g_E is not None else 0,
            "g_I_mean": striatum.neurons.g_I.mean().item() if striatum.neurons.g_I is not None else 0,
            "membrane_max": striatum.neurons.membrane.max().item() if striatum.neurons.membrane is not None else 0,
            "d1_weights_mean": striatum.d1_weights.mean().item(),
            "d2_weights_mean": striatum.d2_weights.mean().item(),
            "d1_weights_max": striatum.d1_weights.max().item(),
            "d2_weights_max": striatum.d2_weights.max().item(),
            "d1_d2_ratio": (striatum.d1_weights.mean() / max(striatum.d2_weights.mean(), 1e-6)).item(),
            "action_d1_means": action_d1_means,
            "action_d2_means": action_d2_means,
            "action_net": [d1 - d2 for d1, d2 in zip(action_d1_means, action_d2_means)],
            "action_d1_elig": action_d1_elig,
            "action_d2_elig": action_d2_elig,
            "input_strength": input_strength,
            "is_match": is_match,
            "selected_action": selected_action,
            "correct": correct,
        }

    # Learn from outcome
    # Eligibility-based learning (biologically realistic)
    # - All actions build eligibility traces from input
    # - Only the CHOSEN action receives dopamine signal
    # - Other actions' eligibility decays unused
    # - Exploration over time samples all actions for learning
    brain.deliver_reward(reward)

    # Store experience for later replay during sleep
    brain.store_experience(
        combined_input=test_result["combined_input"],
        is_match=is_match,
        selected_action=selected_action,
        correct=correct,
        reward=reward,
        sample_pattern=sample,
        test_pattern=test,
    )

    return {
        "correct": correct,
        "selected_action": selected_action,
        "is_match": is_match,
        "reward": reward,
        "explored": test_result.get("explored", False),
        "diagnostics": diagnostic_info,
        # Hippocampus comparison diagnostics
        "ca1_similarity": test_result.get("ca1_similarity", 0.0),
        "comparison_decision": test_result.get("comparison_decision", "UNKNOWN"),
    }


def run_epoch(
    brain: BrainSystem,
    patterns: torch.Tensor,
    n_trials: int,
    match_prob: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one epoch of trials.

    The brain ALWAYS LEARNS - there is no separate evaluation mode.
    Use freeze_plasticity() context manager if you need frozen evaluation.
    """

    results = []
    n_patterns = patterns.shape[0]

    # Track input strength by trial type
    match_input_sum = 0.0
    nomatch_input_sum = 0.0
    n_match_trials = 0
    n_nomatch_trials = 0

    for trial_idx in range(n_trials):
        # Select sample pattern
        sample_idx = random.randint(0, n_patterns - 1)
        sample = patterns[sample_idx].unsqueeze(0)

        # Decide match or no-match trial
        is_match = random.random() < match_prob

        if is_match:
            test_idx = sample_idx
        else:
            # Pick different pattern
            test_idx = random.randint(0, n_patterns - 1)
            while test_idx == sample_idx:
                test_idx = random.randint(0, n_patterns - 1)

        test = patterns[test_idx].unsqueeze(0)

        # Run trial (verbose on first and last trials)
        trial_verbose = verbose and (trial_idx == 0 or trial_idx == n_trials - 1)
        result = run_trial(brain, sample, test, is_match, verbose=trial_verbose)
        results.append(result)

        # DEBUG: Print per-trial outcome for first epoch
        if verbose:
            gt = "M" if is_match else "NM"
            act = "M" if result["selected_action"] == 0 else "NM"
            rwd = "+" if result["correct"] else "-"
            # Include CA1 similarity to diagnose hippocampus discrimination
            ca1_sim = result.get("ca1_similarity", 0.0)
            comp_dec = result.get("comparison_decision", "?")
            print(f"  Trial {trial_idx+1:2d}: GT={gt}, Act={act}, {rwd} (CA1={ca1_sim:.2f}, Comp={comp_dec})")

        # Track input strength by trial type (only in verbose mode)
        if verbose and result.get("diagnostics", {}):
            input_str = result["diagnostics"].get("input_strength", 0)
            if is_match:
                match_input_sum += input_str
                n_match_trials += 1
            else:
                nomatch_input_sum += input_str
                n_nomatch_trials += 1

        # Inter-trial interval (reset dynamics, NOT weights)
        brain.inter_trial_interval(n_timesteps=30)

    # Compute epoch statistics
    n_correct = sum(r["correct"] for r in results)
    n_match_correct = sum(r["correct"] for r in results if r["is_match"])
    n_nomatch_correct = sum(r["correct"] for r in results if not r["is_match"])
    n_match = sum(r["is_match"] for r in results)
    n_nomatch = len(results) - n_match

    # Track action selection distribution
    n_action_0 = sum(r["selected_action"] == 0 for r in results)  # MATCH selected
    n_action_1 = sum(r["selected_action"] == 1 for r in results)  # NOMATCH selected

    # Collect diagnostics from first and last trial
    first_diag = results[0].get("diagnostics", {})
    last_diag = results[-1].get("diagnostics", {}) if len(results) > 1 else {}

    return {
        "accuracy": n_correct / len(results),
        "match_accuracy": n_match_correct / max(1, n_match),
        "nomatch_accuracy": n_nomatch_correct / max(1, n_nomatch),
        "n_trials": len(results),
        "n_match": n_match,
        "n_nomatch": n_nomatch,
        "n_explored": sum(r["explored"] for r in results),
        "n_action_0_selected": n_action_0,
        "n_action_1_selected": n_action_1,
        "first_trial_diagnostics": first_diag,
        "last_trial_diagnostics": last_diag,
    }


def evaluate(
    brain: BrainSystem,
    patterns: torch.Tensor,
    n_trials: int,
    match_prob: float,
) -> Dict[str, Any]:
    """Evaluate using freeze_plasticity context manager.

    During evaluation, we freeze plasticity so we can measure "what did the
    system learn" without the measurement changing the policy.

    Note: This is a debugging escape hatch. In a real biological system,
    every experience shapes learning.
    """
    # Use freeze_plasticity context manager
    with brain.striatum.freeze_plasticity():
        result = run_epoch(brain, patterns, n_trials, match_prob, verbose=True)

    # Print evaluation diagnostics
    diag = result.get("first_trial_diagnostics", {})
    if diag:
        trial_type = "MATCH" if diag.get('is_match', False) else "NOMATCH"
        action_name = "MATCH" if diag.get('selected_action', 0) == 0 else "NOMATCH"
        correct_str = "✓" if diag.get('correct', False) else "✗"
        print(f"    EVAL DIAG: g_E={diag.get('g_E_mean', 0):.2f}, "
              f"g_I={diag.get('g_I_mean', 0):.2f}, "
              f"membrane_max={diag.get('membrane_max', 0):.3f}, "
              f"input={diag.get('input_strength', 0):.1f}")
        print(f"              Trial: {trial_type}, Action: {action_name} {correct_str}")
        print(f"              D1 weights: mean={diag.get('d1_weights_mean', 0):.4f}, "
              f"max={diag.get('d1_weights_max', 0):.4f}")
        print(f"              D2 weights: mean={diag.get('d2_weights_mean', 0):.4f}, "
              f"max={diag.get('d2_weights_max', 0):.4f}")
        action_net = diag.get('action_net', [])
        action_d1 = diag.get('action_d1_means', [])
        action_d2 = diag.get('action_d2_means', [])
        if action_net:
            print(f"              Per-action NET (D1-D2): MATCH={action_net[0]:.4f}, NOMATCH={action_net[1]:.4f}")
            print(f"              Per-action D1: MATCH={action_d1[0]:.4f}, NOMATCH={action_d1[1]:.4f}")
            print(f"              Per-action D2: MATCH={action_d2[0]:.4f}, NOMATCH={action_d2[1]:.4f}")

    return result


def run_condition(
    condition: str,
    config: ExperimentConfig,
    patterns: torch.Tensor,
) -> Dict[str, Any]:
    """Run experiment for one condition (NO_SLEEP, REALISTIC_SLEEP)."""

    print(f"\n{'='*60}")
    print(f"Running condition: {condition}")
    print(f"{'='*60}")

    # Get hippocampus parameters
    hp = config.hippo_params

    # Create brain system with configured hippocampus parameters
    brain = BrainSystem(BrainSystemConfig(
        input_size=config.pattern_size,
        cortex_size=config.cortex_size,
        hippocampus_size=config.hippocampus_size,
        pfc_size=config.pfc_size,
        n_actions=config.n_actions,
        # Hippocampus parameters from CLI
        hippo_nmda_tau=hp.nmda_tau,
        hippo_nmda_threshold=hp.nmda_threshold,
        hippo_nmda_steepness=hp.nmda_steepness,
        hippo_ampa_ratio=hp.ampa_ratio,
        hippo_ec_ca1_learning_rate=hp.ec_ca1_learning_rate,
        hippo_ca3_learning_rate=hp.ca3_learning_rate,
        hippo_dg_sparsity=hp.dg_sparsity,
        hippo_ca3_sparsity=hp.ca3_sparsity,
        hippo_ca1_sparsity=hp.ca1_sparsity,
        enable_stp=config.enable_stp,
        enable_bcm=config.enable_bcm,
        verbose=False,
        device=config.device,
    ))

    # DEBUG: Print initial D1/D2 weights and RPE normalization config
    print(f"[INIT] D1 mean={brain.striatum.d1_weights.mean():.4f}, D2 mean={brain.striatum.d2_weights.mean():.4f}")
    print(f"[INIT] DA config: burst={brain.striatum.striatum_config.dopamine_burst}, dip={brain.striatum.striatum_config.dopamine_dip}")
    print(f"[INIT] Sensitivity: d1={brain.striatum.striatum_config.d1_da_sensitivity}, d2={brain.striatum.striatum_config.d2_da_sensitivity}")
    print(f"[INIT] RPE Normalization: enabled={brain.striatum.striatum_config.normalize_rpe}, tau={brain.striatum.striatum_config.rpe_avg_tau}, clip={brain.striatum.striatum_config.rpe_clip}")

    epoch_results = []
    sleep_results = []

    for epoch in range(config.n_epochs):
        # Exploration decay: reduce tonic dopamine over epochs
        # This naturally reduces exploration as learning progresses
        # Start at 0.3, decay to 0.05 by end of training
        decay_factor = 1.0 - (epoch / max(1, config.n_epochs - 1))
        new_tonic = 0.05 + 0.25 * decay_factor  # Range: 0.30 -> 0.05
        brain.striatum.tonic_dopamine = new_tonic

        # Curriculum learning: boost dopamine signals in early epochs
        # This provides stronger learning signals at the start when
        # the network doesn't know anything yet.
        if config.enable_curriculum and epoch < config.curriculum_warmup_epochs:
            # Boost learning rate during warmup (decays from 2x to 1x)
            warmup_progress = epoch / config.curriculum_warmup_epochs
            boost_factor = 2.0 - warmup_progress  # 2.0 -> 1.0
            old_burst = brain.striatum.striatum_config.dopamine_burst
            old_dip = brain.striatum.striatum_config.dopamine_dip
            brain.striatum.striatum_config.dopamine_burst = old_burst * boost_factor
            brain.striatum.striatum_config.dopamine_dip = old_dip * boost_factor
        else:
            # Reset to normal dopamine levels after warmup
            brain.striatum.striatum_config.dopamine_burst = 1.5  # From StriatumConfig
            brain.striatum.striatum_config.dopamine_dip = -1.5  # Symmetric with burst

        # Training epoch (verbose on first and last epochs)
        verbose_epoch = (epoch == 0 or epoch == config.n_epochs - 1)
        epoch_result = run_epoch(
            brain, patterns,
            config.trials_per_epoch,
            config.match_probability,
            verbose=verbose_epoch,
        )
        epoch_results.append(epoch_result)

        # Get exploration probability from last trial for logging
        exploration_prob = brain.striatum._last_exploration_prob

        # Print basic epoch info
        print(f"  Epoch {epoch+1:2d}: Acc={epoch_result['accuracy']:.1%} "
              f"(M:{epoch_result['match_accuracy']:.1%}, "
              f"NM:{epoch_result['nomatch_accuracy']:.1%}) "
              f"exp_prob={exploration_prob:.2f} tonic={new_tonic:.2f} "
              f"[Actions: M={epoch_result['n_action_0_selected']}, NM={epoch_result['n_action_1_selected']}]")

        # Centralized debug: use striatum's debug method
        brain.striatum.debug_state(f"END EPOCH {epoch+1}")

        # Sleep consolidation (if applicable)
        if (epoch + 1) % config.sleep_every_n_epochs == 0:
            if condition == "NO_SLEEP":
                pass  # No sleep debug output
            elif condition == "REALISTIC_SLEEP":
                sleep_result = brain.realistic_sleep(
                    n_cycles=config.sleep_cycles,
                    replays_per_cycle=config.replays_per_cycle,
                    verbose=False,
                )
                sleep_results.append(sleep_result)
                print(f"    [Realistic sleep: {sleep_result['total_replays']} replays, "
                      f"N2={sleep_result['n2_replays']}, SWS={sleep_result['sws_replays']}, "
                      f"REM={sleep_result['rem_replays']}]")

    # Final evaluation
    print(f"\n  Final Evaluation ({config.n_test_trials} trials):")

    # Centralized debug: use striatum's debug method
    brain.striatum.debug_state("BEFORE EVAL")

    final_eval = evaluate(brain, patterns, config.n_test_trials, config.match_probability)
    print(f"    Accuracy: {final_eval['accuracy']:.1%} "
          f"(Match: {final_eval['match_accuracy']:.1%}, "
          f"NoMatch: {final_eval['nomatch_accuracy']:.1%})")

    # Check if weights changed during evaluation (they shouldn't!)
    brain.striatum.debug_state("AFTER EVAL")

    # Print comprehensive diagnostics using the new system
    diag = brain.get_diagnostics()
    print_diagnostics_summary(diag, f"FINAL DIAGNOSTICS: {condition}")

    return {
        "condition": condition,
        "epoch_results": epoch_results,
        "sleep_results": sleep_results,
        "final_evaluation": final_eval,
        "diagnostics": diag,
    }


def run_experiment(config: ExperimentConfig, conditions: List[str]) -> Dict[str, Any]:
    """Run the full experiment."""

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print("="*60)
    print("EXPERIMENT: Sleep Importance for Learning")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Patterns: {config.n_patterns} x {config.pattern_size}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Trials/epoch: {config.trials_per_epoch}")
    print(f"  Sleep every: {config.sleep_every_n_epochs} epochs")
    print(f"  Sleep cycles: {config.sleep_cycles}")
    hp = config.hippo_params
    print(f"\nHippocampus Parameters:")
    print(f"  NMDA: tau={hp.nmda_tau}, threshold={hp.nmda_threshold}, steepness={hp.nmda_steepness}")
    print(f"  AMPA ratio: {hp.ampa_ratio}")
    print(f"  Learning: EC→CA1={hp.ec_ca1_learning_rate}, CA3={hp.ca3_learning_rate}")
    print(f"  Sparsity: DG={hp.dg_sparsity}, CA3={hp.ca3_sparsity}, CA1={hp.ca1_sparsity}")

    # Generate patterns
    patterns = generate_patterns(config.n_patterns, config.pattern_size)

    # Run each condition
    results = {}

    start_time = time.time()

    for condition in conditions:
        # Reset seed for fair comparison
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        results[condition] = run_condition(condition, config, patterns)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Final Accuracy by Condition")
    print("="*60)

    for condition in conditions:
        acc = results[condition]["final_evaluation"]["accuracy"]
        match_acc = results[condition]["final_evaluation"]["match_accuracy"]
        nomatch_acc = results[condition]["final_evaluation"]["nomatch_accuracy"]
        print(f"  {condition:20s}: {acc:.1%} (M:{match_acc:.1%}, NM:{nomatch_acc:.1%})")

    print(f"\nTotal time: {elapsed:.1f}s")

    return {
        "config": asdict(config),
        "results": results,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    """Main entry point."""

    # Parse CLI arguments
    args = parse_args()

    # Convert to config
    config = args_to_config(args)

    # Get conditions from CLI
    conditions = args.conditions

    results = run_experiment(config, conditions)

    # Save results
    results_dir = Path(__file__).parent.parent.parent / "results" / "integration"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp01_sleep_importance_{timestamp}.json"

    with open(results_file, "w") as f:
        # Convert non-serializable objects
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            else:
                return str(obj)

        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
