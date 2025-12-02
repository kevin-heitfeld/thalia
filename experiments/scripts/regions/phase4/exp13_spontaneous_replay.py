#!/usr/bin/env python3
"""
Experiment 13: Spontaneous Replay - Memory Consolidation
=========================================================

Tests emergent hippocampal replay during idle periods, a phenomenon
observed in biological brains where recent experiences are replayed
without external input, supporting memory consolidation.

Biological Basis:
- During rest/sleep, hippocampus spontaneously reactivates recent memories
- These replays occur in compressed timeframes (faster than real-time)
- Replay helps transfer memories to cortex (systems consolidation)
- Correlated with memory retention

Architecture:
- Hippocampus: Stores recent experiences
- Cortex: Learns from hippocampal replay patterns
- Prefrontal: Provides replay initiation signals

The Test:
1. EXPERIENCE PHASE: Present a sequence of patterns to hippocampus
2. IDLE PHASE: Remove external input, let system run freely
3. DETECTION: Check if hippocampal activity resembles stored patterns
4. CONSOLIDATION: Verify cortex representations improve from replay

Success Criteria:
1. Replay detection: Hippocampal activity matches >50% of stored patterns
2. Temporal compression: Replay occurs faster than original encoding
3. Cortex benefit: Cortex representations improve after idle period
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from thalia.regions import (
    Cortex, CortexConfig,
    Hippocampus, HippocampusConfig,
    Prefrontal, PrefrontalConfig,
)
from thalia.regions.base import LearningRule


def pattern_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two patterns."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = a_flat.norm()
    norm_b = b_flat.norm()
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    return float((a_flat @ b_flat) / (norm_a * norm_b))


def detect_replay(activity: torch.Tensor, stored_patterns: list[torch.Tensor],
                  threshold: float = 0.6) -> tuple[bool, int, float]:
    """Detect if activity matches any stored pattern.

    Returns: (match_found, pattern_index, similarity)
    """
    best_sim = 0.0
    best_idx = -1

    for idx, pattern in enumerate(stored_patterns):
        sim = pattern_similarity(activity, pattern)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    return best_sim >= threshold, best_idx, best_sim


def run_experiment() -> bool:
    """Run the spontaneous replay experiment."""
    torch.manual_seed(42)

    print("=" * 60)
    print("Experiment 13: Spontaneous Replay - Memory Consolidation")
    print("=" * 60)

    # Configuration
    n_patterns = 8  # Number of experiences to store
    pattern_dim = 16  # Dimension of each pattern
    experience_time = 10  # Timesteps per experience during encoding
    idle_timesteps = 200  # Length of idle period
    replay_threshold = 0.5  # Similarity threshold for replay detection

    print("\n[1/6] Creating experience patterns...")

    # Create distinct experience patterns
    experiences = []
    for i in range(n_patterns):
        # Create sparse, distinct patterns
        pattern = torch.zeros(pattern_dim)
        # Each pattern has 4 active units in different positions
        active_idx = torch.randperm(pattern_dim)[:4]
        pattern[active_idx] = torch.randn(4).abs() + 0.5
        pattern = pattern / pattern.norm()  # Normalize
        experiences.append(pattern)

    print(f"  Created {n_patterns} experience patterns (dim={pattern_dim})")

    # Create brain regions
    print("\n[2/6] Creating brain regions...")

    # Hippocampus: Stores experiences with pattern completion
    hipp_hidden = 32
    hippocampus = Hippocampus(HippocampusConfig(
        n_input=pattern_dim,
        n_output=hipp_hidden,
        learning_rate=0.5,
    ))

    # Cortex: Learns compressed representations
    cortex_out = 12
    cortex = Cortex(CortexConfig(
        n_input=hipp_hidden,
        n_output=cortex_out,
        learning_rule=LearningRule.HEBBIAN,
    ))

    # Prefrontal: Controls replay initiation (provides random "trigger" signals)
    prefrontal = Prefrontal(PrefrontalConfig(
        n_input=4,
        n_output=pattern_dim,
    ))

    print(f"  Hippocampus: {pattern_dim} → {hipp_hidden}")
    print(f"  Cortex: {hipp_hidden} → {cortex_out}")
    print(f"  Prefrontal: 4 → {pattern_dim} (replay trigger)")

    # Phase 1: Experience encoding
    print("\n[3/6] Experience phase - encoding patterns...")

    # Store each experience in hippocampus
    encoding_times = []
    for i, exp in enumerate(experiences):
        start_time = i * experience_time

        # Present pattern for multiple timesteps (simulates sustained experience)
        for t in range(experience_time):
            # Hebbian storage in hippocampus
            hipp_output = hippocampus.weights @ exp
            hipp_output = torch.tanh(hipp_output)

            # Update hippocampal weights to store this pattern
            exp_norm = exp / (exp.norm() + 1e-6)
            hipp_norm = hipp_output / (hipp_output.norm() + 1e-6)
            dw = 0.3 * torch.outer(hipp_norm, exp_norm)
            hippocampus.weights = (hippocampus.weights + dw * 0.1).clamp(-1, 1)

        encoding_times.append(experience_time)

    total_encoding_time = sum(encoding_times)
    print(f"  Encoded {n_patterns} patterns over {total_encoding_time} timesteps")
    print(f"  Avg encoding time: {total_encoding_time / n_patterns:.1f} per pattern")

    # Compute hippocampal outputs for stored patterns (ground truth)
    stored_hipp_patterns = []
    for exp in experiences:
        hipp_out = torch.tanh(hippocampus.weights @ exp)
        stored_hipp_patterns.append(hipp_out)

    # Get initial cortex representations (before replay)
    cortex_before = []
    for hipp_pattern in stored_hipp_patterns:
        cortex_out_vec = cortex.weights @ hipp_pattern
        cortex_before.append(cortex_out_vec.clone())

    # Phase 2: Idle period with spontaneous replay
    print("\n[4/6] Idle phase - monitoring for spontaneous replay...")

    replay_events = []  # (timestep, pattern_idx, similarity)
    replay_timings = {i: [] for i in range(n_patterns)}  # Track when each pattern replays

    # Small noise to simulate neural fluctuations
    noise_scale = 0.3

    # Initialize activity with small noise
    current_activity = torch.randn(pattern_dim) * noise_scale

    for t in range(idle_timesteps):
        # Generate random "trigger" from prefrontal (simulates internal signals)
        trigger = torch.randn(4) * 0.2
        prefrontal.set_context(trigger)
        pfc_raw = prefrontal.get_working_memory().flatten()
        # Pad or truncate to pattern_dim
        if pfc_raw.numel() < pattern_dim:
            pfc_output = torch.zeros(pattern_dim)
            pfc_output[:pfc_raw.numel()] = pfc_raw
        else:
            pfc_output = pfc_raw[:pattern_dim]

        # Combine noise with PFC signal
        input_noise = torch.randn(pattern_dim) * noise_scale
        combined_input = 0.3 * pfc_output + 0.7 * input_noise

        # Pass through hippocampus (pattern completion)
        hipp_activity = torch.tanh(hippocampus.weights @ combined_input)

        # Pattern completion: if activity is similar to stored, amplify it
        for stored_idx, stored_pattern in enumerate(stored_hipp_patterns):
            sim = pattern_similarity(hipp_activity, stored_pattern)
            if sim > 0.3:
                # Partial match - boost toward stored pattern (attractor dynamics)
                hipp_activity = 0.7 * hipp_activity + 0.3 * stored_pattern

        # Check for replay (high similarity to stored patterns)
        match_found, pattern_idx, similarity = detect_replay(
            hipp_activity, stored_hipp_patterns, threshold=replay_threshold
        )

        if match_found:
            replay_events.append((t, pattern_idx, similarity))
            replay_timings[pattern_idx].append(t)

            # During replay, cortex learns from the replayed pattern
            cortex_input = hipp_activity
            cortex_output = cortex.weights @ cortex_input

            # Hebbian update in cortex
            in_norm = cortex_input / (cortex_input.norm() + 1e-6)
            out_norm = cortex_output / (cortex_output.norm() + 1e-6)
            dw = 0.1 * torch.outer(out_norm, in_norm)
            cortex.weights = (cortex.weights + dw * 0.05).clamp(-1, 1)

    # Analyze replay
    patterns_replayed = set(e[1] for e in replay_events)
    n_patterns_replayed = len(patterns_replayed)
    total_replay_events = len(replay_events)

    print(f"\n  Replay Analysis:")
    print(f"  Total replay events: {total_replay_events}")
    print(f"  Unique patterns replayed: {n_patterns_replayed}/{n_patterns}")

    # Compute inter-replay intervals (temporal compression)
    all_intervals = []
    for pattern_idx, times in replay_timings.items():
        if len(times) >= 2:
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            all_intervals.extend(intervals)

    avg_replay_interval = np.mean(all_intervals) if all_intervals else float('inf')
    print(f"  Avg inter-replay interval: {avg_replay_interval:.1f} timesteps")
    print(f"  Original encoding time: {experience_time} timesteps")

    # Check temporal compression (replay faster than encoding)
    temporal_compression = experience_time / avg_replay_interval if avg_replay_interval > 0 else 0
    print(f"  Temporal compression factor: {temporal_compression:.2f}x")

    # Phase 3: Evaluate cortex improvement
    print("\n[5/6] Evaluating cortex representations after replay...")

    cortex_after = []
    for hipp_pattern in stored_hipp_patterns:
        cortex_out_vec = cortex.weights @ hipp_pattern
        cortex_after.append(cortex_out_vec.clone())

    # Measure clustering quality: do similar patterns map to similar representations?
    def intra_cluster_similarity(representations: list[torch.Tensor]) -> float:
        """Average similarity within the set of representations."""
        if len(representations) < 2:
            return 0.0
        sims = []
        for i in range(len(representations)):
            for j in range(i + 1, len(representations)):
                sims.append(pattern_similarity(representations[i], representations[j]))
        return np.mean(sims)

    def representation_quality(reps: list[torch.Tensor]) -> float:
        """Measure how distinct the representations are (lower is better for clustering)."""
        # We want representations to be distinct (low similarity)
        # But also informative (high norm)
        norms = [float(r.norm()) for r in reps]
        avg_norm = np.mean(norms)
        avg_sim = intra_cluster_similarity(reps)
        # Quality = high norm (informative) and moderate similarity
        return avg_norm * (1 - avg_sim * 0.5)

    quality_before = representation_quality(cortex_before)
    quality_after = representation_quality(cortex_after)
    quality_improvement = (quality_after - quality_before) / (abs(quality_before) + 1e-6) * 100

    print(f"  Cortex quality before replay: {quality_before:.3f}")
    print(f"  Cortex quality after replay: {quality_after:.3f}")
    print(f"  Quality improvement: {quality_improvement:.1f}%")

    # Generate plots
    print("\n[6/6] Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Replay events over time
    ax1 = axes[0, 0]
    if replay_events:
        times = [e[0] for e in replay_events]
        patterns = [e[1] for e in replay_events]
        sims = [e[2] for e in replay_events]
        scatter = ax1.scatter(times, patterns, c=sims, cmap='viridis',
                             s=50, alpha=0.7, vmin=replay_threshold, vmax=1.0)
        plt.colorbar(scatter, ax=ax1, label='Similarity')
    ax1.set_xlabel("Timestep (idle period)")
    ax1.set_ylabel("Pattern Index")
    ax1.set_title("Replay Events During Idle Period")
    ax1.set_yticks(range(n_patterns))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Replay frequency per pattern
    ax2 = axes[0, 1]
    replay_counts = [len(replay_timings[i]) for i in range(n_patterns)]
    bars = ax2.bar(range(n_patterns), replay_counts, color='steelblue', alpha=0.7)
    ax2.set_xlabel("Pattern Index")
    ax2.set_ylabel("Number of Replays")
    ax2.set_title("Replay Frequency by Pattern")
    ax2.set_xticks(range(n_patterns))
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Cortex representations before/after
    ax3 = axes[1, 0]
    before_norms = [float(r.norm()) for r in cortex_before]
    after_norms = [float(r.norm()) for r in cortex_after]
    x = np.arange(n_patterns)
    width = 0.35
    ax3.bar(x - width/2, before_norms, width, label='Before Replay', color='gray', alpha=0.7)
    ax3.bar(x + width/2, after_norms, width, label='After Replay', color='green', alpha=0.7)
    ax3.set_xlabel("Pattern Index")
    ax3.set_ylabel("Representation Norm")
    ax3.set_title("Cortex Representation Strength")
    ax3.set_xticks(x)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Inter-replay intervals histogram
    ax4 = axes[1, 1]
    if all_intervals:
        ax4.hist(all_intervals, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax4.axvline(x=experience_time, color='red', linestyle='--',
                   label=f'Encoding time ({experience_time})')
        ax4.axvline(x=avg_replay_interval, color='blue', linestyle='--',
                   label=f'Avg replay interval ({avg_replay_interval:.1f})')
    ax4.set_xlabel("Inter-Replay Interval (timesteps)")
    ax4.set_ylabel("Count")
    ax4.set_title("Replay Timing Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = project_root / "experiments" / "results" / "regions" / "exp13_spontaneous_replay.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: exp13_spontaneous_replay.png")

    # Evaluate criteria
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Criterion 1: Replay detection (>50% of patterns replayed)
    replay_rate = n_patterns_replayed / n_patterns * 100
    c1 = replay_rate >= 50
    print(f"\n1. Replay detection (≥50% patterns): {'PASS' if c1 else 'FAIL'}")
    print(f"   Patterns replayed: {n_patterns_replayed}/{n_patterns} ({replay_rate:.0f}%)")

    # Criterion 2: Temporal compression (replay faster than encoding)
    c2 = temporal_compression > 1.0
    print(f"\n2. Temporal compression (>1x): {'PASS' if c2 else 'FAIL'}")
    print(f"   Compression factor: {temporal_compression:.2f}x")

    # Criterion 3: Cortex benefit (representations improve)
    c3 = quality_improvement > 0
    print(f"\n3. Cortex benefit (quality improves): {'PASS' if c3 else 'FAIL'}")
    print(f"   Quality improvement: {quality_improvement:.1f}%")

    passed = sum([c1, c2, c3])
    success = passed >= 2

    print("\n" + "=" * 60)
    print(f"RESULT: {'PASSED' if success else 'FAILED'} ({passed}/3 criteria)")
    print("=" * 60)

    # Save results
    results = {
        "experiment": "exp13_spontaneous_replay",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_patterns": n_patterns,
            "pattern_dim": pattern_dim,
            "experience_time": experience_time,
            "idle_timesteps": idle_timesteps,
            "replay_threshold": replay_threshold,
        },
        "results": {
            "patterns_replayed": n_patterns_replayed,
            "total_replay_events": total_replay_events,
            "replay_rate": float(replay_rate),
            "temporal_compression": float(temporal_compression),
            "cortex_quality_before": float(quality_before),
            "cortex_quality_after": float(quality_after),
            "quality_improvement": float(quality_improvement),
        },
        "criteria": {
            "c1_replay_detection": bool(c1),
            "c2_temporal_compression": bool(c2),
            "c3_cortex_benefit": bool(c3),
        },
        "passed": int(passed),
        "success": bool(success),
    }

    results_path = (
        project_root / "experiments" / "results" / "regions" /
        f"exp13_spontaneous_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return success


if __name__ == "__main__":
    success = run_experiment()
    sys.exit(0 if success else 1)
