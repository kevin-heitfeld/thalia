"""
Streaming Trainer - Continuous online learning from data streams.

This module implements biologically-realistic online learning where the brain
processes continuous data streams without explicit training epochs or batch
boundaries. This matches natural learning more closely than traditional batch
training.

Key Differences from Batch Training:
====================================
- NO EPOCHS: Continuous stream of data, no restart/repeat
- NO TRAIN/EVAL SPLIT: Plasticity always enabled (like biological brains)
- CONTINUOUS ADAPTATION: Learning never stops
- ONLINE EVALUATION: Performance measured on-the-fly
- CATASTROPHIC FORGETTING: Real concern (use replay/consolidation)

Biological Motivation:
======================
Real brains learn continuously from experience streams:
- Visual input arrives frame-by-frame
- Auditory input is a continuous waveform
- Motor feedback is immediate and continuous
- No "training mode" vs "inference mode" distinction

Local learning rules (STDP, BCM, Hebbian) naturally support this:
- Weight updates happen during forward pass
- No need to accumulate gradients across batches
- Each experience immediately affects plasticity

Use Cases:
==========
1. Lifelong learning experiments
2. Online adaptation to environment changes
3. Real-time robotics/embodied AI
4. Streaming sensor data (audio, video)
5. Incremental knowledge acquisition

Example Usage:
==============

    from thalia.training.streaming import StreamingTrainer, StreamConfig
    from thalia.core.brain import EventDrivenBrain
    import torch

    # Create brain
    brain = EventDrivenBrain(config)

    # Create data stream (infinite generator)
    def mnist_stream():
        while True:
            # Cycle through MNIST continuously
            for batch in mnist_loader:
                for img, label in zip(*batch):
                    yield {'input': img, 'label': label}

    # Create streaming trainer
    trainer = StreamingTrainer(
        brain=brain,
        config=StreamConfig(
            eval_frequency=1000,  # Evaluate every 1000 samples
            checkpoint_frequency=10000,  # Save every 10k samples
            health_check_frequency=5000,  # Check health every 5k samples
            enable_replay=True,  # Use experience replay
            replay_buffer_size=10000,
            replay_frequency=10,  # Replay 1 sample every 10 new samples
        ),
        checkpoint_dir="checkpoints/streaming",
    )

    # Train continuously
    stats = trainer.train_online(
        data_stream=mnist_stream(),
        max_samples=1_000_000,  # Optional limit (None = infinite)
    )

Architecture:
=============

    StreamingTrainer
    ├── ExperienceBuffer (for replay-based consolidation)
    ├── OnlineEvaluator (continuous performance tracking)
    ├── DriftDetector (detect catastrophic forgetting)
    ├── HealthMonitor (ensure stable learning)
    └── CheckpointManager (periodic saves)

Strategies for Preventing Catastrophic Forgetting:
==================================================
1. Experience Replay: Periodically replay old samples
2. Elastic Weight Consolidation: Protect important weights
3. Neuromodulator Homeostasis: Maintain stable baseline DA/NE/ACh
4. Synaptic Consolidation: Slow time constants for important connections

References:
===========
- docs/design/curriculum_strategy.md - Training philosophy
- docs/architecture/CENTRALIZED_SYSTEMS.md - Current centralized systems

Author: Thalia Project
Date: December 12, 2025 (Tier 3 Implementation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Dict, Any, Callable, Deque
from pathlib import Path
from collections import deque
import time

import torch
import numpy as np

from thalia.core.brain import EventDrivenBrain
from thalia.diagnostics.health_monitor import HealthMonitor
from thalia.diagnostics.performance_profiler import PerformanceProfiler
from thalia.io.checkpoint import BrainCheckpoint


@dataclass
class StreamConfig:
    """Configuration for streaming trainer.

    Args:
        eval_frequency: Evaluate performance every N samples
        checkpoint_frequency: Save checkpoint every N samples
        health_check_frequency: Check brain health every N samples
        enable_replay: Whether to use experience replay
        replay_buffer_size: Max samples in replay buffer
        replay_frequency: Replay 1 sample every N new samples
        max_replay_age: Max age (samples) for replay samples (None = no limit)
        drift_detection_window: Window size for drift detection
        drift_threshold: Performance drop threshold to trigger alert
        enable_profiling: Track performance metrics (throughput, memory)
    """
    eval_frequency: int = 1000
    checkpoint_frequency: int = 10000
    health_check_frequency: int = 5000
    enable_replay: bool = True
    replay_buffer_size: int = 10000
    replay_frequency: int = 10  # Replay every 10 new samples
    max_replay_age: Optional[int] = None
    drift_detection_window: int = 5000
    drift_threshold: float = 0.1  # 10% performance drop
    enable_profiling: bool = True


@dataclass
class StreamingStats:
    """Statistics from streaming training session.

    Attributes:
        samples_processed: Total samples processed
        duration_seconds: Total training time
        throughput_samples_per_sec: Average throughput
        performance_history: List of (sample_idx, metrics_dict)
        drift_alerts: List of detected drift events
        health_issues: List of detected health problems
        checkpoints_saved: List of checkpoint paths
    """
    samples_processed: int = 0
    duration_seconds: float = 0.0
    throughput_samples_per_sec: float = 0.0
    performance_history: list = field(default_factory=list)
    drift_alerts: list = field(default_factory=list)
    health_issues: list = field(default_factory=list)
    checkpoints_saved: list = field(default_factory=list)


class ExperienceBuffer:
    """Circular buffer for experience replay.

    Stores recent samples for replay to prevent catastrophic forgetting.
    Uses reservoir sampling for large streams.
    """

    def __init__(self, capacity: int, max_age: Optional[int] = None):
        """Initialize buffer.

        Args:
            capacity: Max number of samples to store
            max_age: Max age (samples) for replay (None = no limit)
        """
        self.capacity = capacity
        self.max_age = max_age
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self.sample_counter = 0

    def add(self, sample: Dict[str, Any]) -> None:
        """Add sample to buffer with timestamp."""
        sample_with_time = {**sample, "_timestamp": self.sample_counter}
        self.buffer.append(sample_with_time)
        self.sample_counter += 1

    def sample_random(self) -> Optional[Dict[str, Any]]:
        """Sample random experience from buffer."""
        if len(self.buffer) == 0:
            return None

        # Filter by age if specified
        if self.max_age is not None:
            valid_samples = [
                s for s in self.buffer
                if self.sample_counter - s["_timestamp"] <= self.max_age
            ]
            if not valid_samples:
                return None
            idx = np.random.randint(len(valid_samples))
            sample = valid_samples[idx]
        else:
            idx = np.random.randint(len(self.buffer))
            sample = self.buffer[idx]

        # Remove timestamp before returning
        sample_copy = {k: v for k, v in sample.items() if k != "_timestamp"}
        return sample_copy

    def __len__(self) -> int:
        return len(self.buffer)


class DriftDetector:
    """Detect catastrophic forgetting during streaming.

    Monitors recent performance and alerts when significant drops occur.
    """

    def __init__(self, window_size: int, threshold: float):
        """Initialize drift detector.

        Args:
            window_size: Number of recent evaluations to track
            threshold: Fraction of performance drop to trigger alert
        """
        self.window_size = window_size
        self.threshold = threshold
        self.performance_history: Deque[float] = deque(maxlen=window_size)

    def update(self, performance: float) -> Optional[Dict[str, Any]]:
        """Update with new performance metric.

        Args:
            performance: Current performance (higher is better)

        Returns:
            Alert dict if drift detected, None otherwise
        """
        if len(self.performance_history) < self.window_size // 2:
            # Not enough history yet
            self.performance_history.append(performance)
            return None

        # Compare current to recent average
        recent_avg = np.mean(list(self.performance_history))
        drop = (recent_avg - performance) / (recent_avg + 1e-8)

        self.performance_history.append(performance)

        if drop > self.threshold:
            return {
                "type": "catastrophic_forgetting",
                "previous_avg": float(recent_avg),
                "current": float(performance),
                "drop_fraction": float(drop),
                "recommendation": "Increase replay frequency or enable consolidation"
            }
        return None


class StreamingTrainer:
    """Trainer for continuous online learning from data streams.

    Processes samples one-by-one with continuous plasticity, matching
    biological learning more closely than batch training.
    """

    def __init__(
        self,
        brain: EventDrivenBrain,
        config: StreamConfig,
        checkpoint_dir: str | Path = "checkpoints/streaming",
        evaluator: Optional[Callable[[EventDrivenBrain], Dict[str, float]]] = None,
    ):
        """Initialize streaming trainer.

        Args:
            brain: The brain to train
            config: Streaming configuration
            checkpoint_dir: Directory for saving checkpoints
            evaluator: Optional function to evaluate performance
                      Takes brain, returns dict of metrics
        """
        self.brain = brain
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator

        # Initialize subsystems
        if config.enable_replay:
            self.replay_buffer = ExperienceBuffer(
                capacity=config.replay_buffer_size,
                max_age=config.max_replay_age,
            )
        else:
            self.replay_buffer = None

        self.drift_detector = DriftDetector(
            window_size=config.drift_detection_window,
            threshold=config.drift_threshold,
        )

        self.health_monitor = HealthMonitor()

        if config.enable_profiling:
            self.profiler = PerformanceProfiler()
        else:
            self.profiler = None

        # Statistics
        self.stats = StreamingStats()

    def train_online(
        self,
        data_stream: Iterator[Dict[str, Any]],
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> StreamingStats:
        """Train continuously from data stream.

        Args:
            data_stream: Iterator yielding sample dicts
                        Each sample should have at least 'input' key
            max_samples: Max samples to process (None = infinite)
            verbose: Print progress updates

        Returns:
            StreamingStats with training statistics
        """
        start_time = time.time()
        sample_count = 0

        if verbose:
            print(f"🌊 Starting streaming training...")
            print(f"   Replay: {'enabled' if self.config.enable_replay else 'disabled'}")
            print(f"   Checkpoints: every {self.config.checkpoint_frequency:,} samples")
            print(f"   Evaluation: every {self.config.eval_frequency:,} samples")
            print()

        try:
            for sample_idx, sample in enumerate(data_stream):
                # Stop if reached max samples
                if max_samples is not None and sample_count >= max_samples:
                    break

                # Process sample
                if self.profiler:
                    self.profiler.start_step()

                self._process_sample(sample)

                if self.profiler:
                    self.profiler.end_step()

                sample_count += 1

                # Experience replay
                if self.replay_buffer and sample_count % self.config.replay_frequency == 0:
                    replay_sample = self.replay_buffer.sample_random()
                    if replay_sample:
                        self._process_sample(replay_sample, is_replay=True)

                # Add to replay buffer
                if self.replay_buffer:
                    self.replay_buffer.add(sample)

                # Periodic evaluation
                if sample_count % self.config.eval_frequency == 0:
                    self._evaluate(sample_count, verbose)

                # Health check
                if sample_count % self.config.health_check_frequency == 0:
                    self._check_health(sample_count, verbose)

                # Checkpoint
                if sample_count % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(sample_count, verbose)

        except KeyboardInterrupt:
            if verbose:
                print("\n⚠️  Training interrupted by user")

        # Final statistics
        duration = time.time() - start_time
        self.stats.samples_processed = sample_count
        self.stats.duration_seconds = duration
        self.stats.throughput_samples_per_sec = sample_count / duration if duration > 0 else 0.0

        if verbose:
            print(f"\n✅ Streaming training complete!")
            print(f"   Samples processed: {sample_count:,}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Throughput: {self.stats.throughput_samples_per_sec:.1f} samples/sec")
            if self.replay_buffer:
                print(f"   Replay buffer: {len(self.replay_buffer):,} samples")
            print(f"   Drift alerts: {len(self.stats.drift_alerts)}")
            print(f"   Health issues: {len(self.stats.health_issues)}")

        return self.stats

    def _process_sample(self, sample: Dict[str, Any], is_replay: bool = False) -> None:
        """Process single sample through brain.

        Args:
            sample: Sample dict with at least 'input' key
            is_replay: Whether this is a replay sample
        """
        # Get input spikes
        input_spikes = sample["input"]
        if not isinstance(input_spikes, torch.Tensor):
            input_spikes = torch.tensor(input_spikes, device=self.brain.device)

        # Move to correct device
        input_spikes = input_spikes.to(self.brain.device)

        # Forward pass (learning happens automatically via local rules)
        self.brain.forward(input_spikes)

        # Optional: Provide reward signal if available
        if "reward" in sample:
            reward = sample["reward"]
            if not isinstance(reward, (int, float)):
                reward = float(reward)
            # Modulate dopamine based on reward
            self.brain.set_dopamine(reward)

    def _evaluate(self, sample_count: int, verbose: bool) -> None:
        """Evaluate current performance.

        Args:
            sample_count: Current sample count
            verbose: Print results
        """
        if self.evaluator is None:
            return

        try:
            metrics = self.evaluator(self.brain)
            self.stats.performance_history.append((sample_count, metrics))

            # Check for drift (use first metric as primary)
            if metrics:
                primary_metric = list(metrics.values())[0]
                drift_alert = self.drift_detector.update(primary_metric)
                if drift_alert:
                    drift_alert["sample_count"] = sample_count
                    self.stats.drift_alerts.append(drift_alert)
                    if verbose:
                        print(f"\n⚠️  Drift detected at sample {sample_count:,}:")
                        print(f"   Previous avg: {drift_alert['previous_avg']:.3f}")
                        print(f"   Current: {drift_alert['current']:.3f}")
                        print(f"   Drop: {drift_alert['drop_fraction']:.1%}")
                        print(f"   {drift_alert['recommendation']}\n")

            if verbose:
                metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                print(f"[{sample_count:,}] Eval: {metrics_str}")

        except Exception as e:
            if verbose:
                print(f"⚠️  Evaluation failed: {e}")

    def _check_health(self, sample_count: int, verbose: bool) -> None:
        """Check brain health.

        Args:
            sample_count: Current sample count
            verbose: Print results
        """
        try:
            report = self.health_monitor.check_brain_health(self.brain)

            # Check for critical issues
            critical_issues = [
                issue for issue in report.issues
                if issue.severity in ["critical", "high"]
            ]

            if critical_issues:
                for issue in critical_issues:
                    issue_dict = {
                        "sample_count": sample_count,
                        "component": issue.component,
                        "issue": issue.issue,
                        "severity": issue.severity,
                        "recommendation": issue.recommendation,
                    }
                    self.stats.health_issues.append(issue_dict)

                if verbose:
                    print(f"\n⚠️  Health issues at sample {sample_count:,}:")
                    for issue in critical_issues:
                        print(f"   [{issue.severity.upper()}] {issue.component}: {issue.issue}")
                        print(f"   → {issue.recommendation}")
                    print()

        except Exception as e:
            if verbose:
                print(f"⚠️  Health check failed: {e}")

    def _save_checkpoint(self, sample_count: int, verbose: bool) -> None:
        """Save checkpoint.

        Args:
            sample_count: Current sample count
            verbose: Print confirmation
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"streaming_sample_{sample_count:09d}.pkl"

            metadata = {
                "sample_count": sample_count,
                "config": self.config,
                "stats": self.stats,
            }

            BrainCheckpoint.save(self.brain, checkpoint_path, metadata=metadata)
            self.stats.checkpoints_saved.append(str(checkpoint_path))

            if verbose:
                print(f"💾 Checkpoint saved: {checkpoint_path.name}")

        except Exception as e:
            if verbose:
                print(f"⚠️  Checkpoint save failed: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics.

        Returns:
            Dict with aggregated metrics
        """
        if not self.stats.performance_history:
            return {}

        # Aggregate metrics across all evaluations
        all_metrics = [metrics for _, metrics in self.stats.performance_history]
        metric_keys = all_metrics[0].keys()

        summary = {}
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))

        return summary
