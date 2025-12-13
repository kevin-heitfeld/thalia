"""
Live Diagnostics for Training Sessions.

Provides real-time visualization of neural activity, health metrics,
and learning dynamics during training.

Usage:
    from thalia.training.visualization.live_diagnostics import LiveDiagnostics

    diagnostics = LiveDiagnostics()

    # During training loop:
    diagnostics.update(brain, metrics)

    # Display current state
    diagnostics.show()

    # Save snapshot
    diagnostics.save_snapshot("outputs/step_1000.png")

Author: Thalia Project
Date: December 10, 2025
"""

from typing import Dict, Any, Optional
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import torch

from thalia.components.coding.spike_utils import compute_firing_rate
from thalia.training.visualization.constants import (
    TEXT_POSITION_CENTER,
    ALPHA_SEMI_TRANSPARENT,
    PERFORMANCE_EXCELLENT,
    PERFORMANCE_GOOD,
    PERFORMANCE_ACCEPTABLE,
)


class LiveDiagnostics:
    """
    Real-time diagnostics for training sessions.

    Features:
    - Neural activity visualization (spike rasters)
    - Health metrics dashboard
    - Weight evolution heatmaps
    - Firing rate distributions
    - Task performance tracking
    - Performance timing (steps/sec, forward pass time)
    - Memory usage (CPU/GPU)
    - Weight distribution histograms
    """

    def __init__(self, history_size: int = 100, figsize: tuple = (16, 10)):
        """
        Initialize live diagnostics.

        Args:
            history_size: Number of steps to keep in history
            figsize: Figure size for visualizations
        """
        self.history_size = history_size
        self.figsize = figsize

        # History buffers
        self.step_history = deque(maxlen=history_size)
        self.firing_rate_history = {
            'cortex': deque(maxlen=history_size),
            'hippocampus': deque(maxlen=history_size),
            'pfc': deque(maxlen=history_size),
            'striatum': deque(maxlen=history_size),
        }
        self.health_history = {
            'is_healthy': deque(maxlen=history_size),
            'runaway_count': deque(maxlen=history_size),
            'silent_count': deque(maxlen=history_size),
        }
        self.performance_history = {
            'motor_control': deque(maxlen=history_size),
            'reaching': deque(maxlen=history_size),
            'manipulation': deque(maxlen=history_size),
        }
        # NEW: Performance timing history
        self.timing_history = {
            'steps_per_sec': deque(maxlen=history_size),
            'forward_ms': deque(maxlen=history_size),
        }
        # NEW: Memory history
        self.memory_history = {
            'cpu_mb': deque(maxlen=history_size),
            'gpu_mb': deque(maxlen=history_size),
        }
        # NEW: Latest weight data
        self.latest_weights: Dict[str, Any] = {}

        # Latest spike data for raster plot
        self.latest_spikes = {}
        self.current_step = 0

        # Figure setup
        self.fig = None
        self.axes = None

    def update(
        self,
        step: int,
        brain: Any,
        metrics: Optional[Dict[str, float]] = None,
        spikes: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        Update diagnostics with latest data.

        Args:
            step: Current training step
            brain: Brain object with regions
            metrics: Performance metrics
            spikes: Latest spike data from regions
        """
        self.current_step = step
        self.step_history.append(step)

        # Update firing rates
        for region_name, region in [
            ('cortex', getattr(brain, 'cortex', None)),
            ('hippocampus', getattr(brain, 'hippocampus', None)),
            ('pfc', getattr(brain, 'prefrontal', None)),
            ('striatum', getattr(brain, 'striatum', None)),
        ]:
            if region and hasattr(region, 'state') and region.state.spikes is not None:
                firing_rate = compute_firing_rate(region.state.spikes)
                self.firing_rate_history[region_name].append(firing_rate)
            else:
                self.firing_rate_history[region_name].append(0.0)

        # Update health
        if hasattr(brain, 'check_health'):
            health = brain.check_health()
            self.health_history['is_healthy'].append(1.0 if health.is_healthy else 0.0)
            self.health_history['runaway_count'].append(
                sum(1 for issue in health.issues if 'runaway' in issue.lower())
            )
            self.health_history['silent_count'].append(
                sum(1 for issue in health.issues if 'silent' in issue.lower())
            )
        else:
            self.health_history['is_healthy'].append(1.0)
            self.health_history['runaway_count'].append(0)
            self.health_history['silent_count'].append(0)

        # Update performance
        if metrics:
            self.performance_history['motor_control'].append(
                metrics.get('motor_control_accuracy', 0.0)
            )
            self.performance_history['reaching'].append(
                metrics.get('reaching_accuracy', 0.0)
            )
            self.performance_history['manipulation'].append(
                metrics.get('manipulation_success', 0.0)
            )

            # NEW: Update timing metrics
            self.timing_history['steps_per_sec'].append(
                metrics.get('performance/steps_per_sec', 0.0)
            )
            self.timing_history['forward_ms'].append(
                metrics.get('performance/avg_forward_ms', 0.0)
            )

            # NEW: Update memory metrics
            self.memory_history['cpu_mb'].append(
                metrics.get('memory/cpu_mb', 0.0)
            )
            self.memory_history['gpu_mb'].append(
                metrics.get('memory/gpu_mb', 0.0)
            )

            # NEW: Extract weight statistics for distribution plots
            self.latest_weights = {
                k: v for k, v in metrics.items()
                if k.startswith('weights/') and '_mean' in k
            }

        # Store latest spikes for raster
        if spikes:
            self.latest_spikes = spikes

    def show(self, save_path: Optional[str] = None) -> None:
        """
        Display current diagnostics.

        Args:
            save_path: If provided, save figure to this path
        """
        # Create figure with subplots (4 rows x 3 columns)
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

        # Row 1: Neural activity
        ax_spikes = fig.add_subplot(gs[0, :2])
        self._plot_spike_raster(ax_spikes)

        ax_firing_rates = fig.add_subplot(gs[0, 2])
        self._plot_firing_rate_distribution(ax_firing_rates)

        # Row 2: Performance timing and memory
        ax_timing = fig.add_subplot(gs[1, :2])
        self._plot_performance_timing(ax_timing)

        ax_memory = fig.add_subplot(gs[1, 2])
        self._plot_memory_usage(ax_memory)

        # Row 3: Health metrics
        ax_health = fig.add_subplot(gs[2, :])
        self._plot_health_metrics(ax_health)

        # Row 4: Task performance
        ax_performance = fig.add_subplot(gs[3, :])
        self._plot_performance(ax_performance)

        # Add title
        fig.suptitle(
            f'Thalia Live Diagnostics (Step {self.current_step})',
            fontsize=16,
            fontweight='bold'
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Diagnostics saved: {save_path}")
        else:
            plt.show()

    def _plot_spike_raster(self, ax) -> None:
        """Plot spike raster for latest timesteps."""
        if not self.latest_spikes:
            ax.text(TEXT_POSITION_CENTER, TEXT_POSITION_CENTER, 'No spike data yet', ha='center', va='center')
            ax.axis('off')
            return

        # Combine spikes from all regions
        y_offset = 0
        colors = plt.cm.Set3(range(len(self.latest_spikes)))

        for idx, (region_name, spikes) in enumerate(self.latest_spikes.items()):
            if spikes is None or not isinstance(spikes, torch.Tensor):
                continue

            # Get spike locations
            if spikes.dim() == 2:  # (timesteps, neurons)
                spike_times, spike_neurons = torch.where(spikes > 0)
                spike_times = spike_times.cpu().numpy()
                spike_neurons = spike_neurons.cpu().numpy() + y_offset

                ax.scatter(spike_times, spike_neurons, s=1, alpha=0.6,
                          color=colors[idx], label=region_name)

                y_offset += spikes.shape[1]

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron ID')
        ax.set_title('Spike Raster (Recent Activity)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

    def _plot_firing_rate_distribution(self, ax) -> None:
        """Plot firing rate distribution across regions."""
        if not self.firing_rate_history['cortex']:
            ax.text(TEXT_POSITION_CENTER, TEXT_POSITION_CENTER, 'No data yet', ha='center', va='center')
            ax.axis('off')
            return

        # Get latest firing rates
        regions = []
        rates = []
        colors = []

        color_map = {
            'cortex': '#FF6B6B',
            'hippocampus': '#4ECDC4',
            'pfc': '#95E1D3',
            'striatum': '#FFA07A',
        }

        for region_name, history in self.firing_rate_history.items():
            if history:
                regions.append(region_name.upper())
                rates.append(history[-1])
                colors.append(color_map.get(region_name, '#999999'))

        # Bar chart
        y_pos = np.arange(len(regions))
        ax.barh(y_pos, rates, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(regions)
        ax.set_xlabel('Firing Rate')
        ax.set_title('[DIAGNOSTICS] Current Firing Rates', fontweight='bold', fontsize=10)
        ax.set_xlim(0, 0.3)
        ax.grid(True, axis='x', alpha=0.3)

        # Add target range
        ax.axvspan(0.05, 0.15, alpha=0.2, color='green', label='Target')
        ax.legend(fontsize=7)

    def _plot_health_metrics(self, ax) -> None:
        """Plot health metrics over time."""
        if not self.step_history:
            ax.text(TEXT_POSITION_CENTER, TEXT_POSITION_CENTER, 'No data yet', ha='center', va='center')
            ax.axis('off')
            return

        steps = list(self.step_history)

        # Plot health status
        ax.fill_between(
            steps,
            self.health_history['is_healthy'],
            alpha=0.3,
            color='green',
            label='Healthy'
        )

        # Plot issue counts
        if self.health_history['runaway_count']:
            runaway = list(self.health_history['runaway_count'])
            ax.plot(steps, runaway, color='red', linewidth=2,
                   marker='x', label='Runaway Events')

        if self.health_history['silent_count']:
            silent = list(self.health_history['silent_count'])
            ax.plot(steps, silent, color='blue', linewidth=2,
                   marker='o', label='Silent Events')

        ax.set_xlabel('Step')
        ax.set_ylabel('Status / Count')
        ax.set_title('Health Metrics', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_performance(self, ax) -> None:
        """Plot task performance over time."""
        if not self.step_history:
            ax.text(TEXT_POSITION_CENTER, TEXT_POSITION_CENTER, 'No data yet', ha='center', va='center')
            ax.axis('off')
            return

        steps = list(self.step_history)

        # Plot each task performance
        for task_name, history in self.performance_history.items():
            if history:
                values = list(history)
                ax.plot(steps, values, linewidth=2, marker='o',
                       markersize=3, label=task_name.replace('_', ' ').title(),
                       alpha=0.8)

        # Add target lines
        ax.axhline(y=PERFORMANCE_EXCELLENT, color='green', linestyle='--', alpha=ALPHA_SEMI_TRANSPARENT,
                  linewidth=1, label='Motor Target (95%)')
        ax.axhline(y=PERFORMANCE_GOOD, color='blue', linestyle='--', alpha=ALPHA_SEMI_TRANSPARENT,
                  linewidth=1, label='Reaching Target (90%)')
        ax.axhline(y=PERFORMANCE_ACCEPTABLE, color='orange', linestyle='--', alpha=ALPHA_SEMI_TRANSPARENT,
                  linewidth=1, label='Manipulation Target (85%)')

        ax.set_xlabel('Step')
        ax.set_ylabel('Performance')
        ax.set_ylim(0, 1.05)
        ax.set_title('Task Performance', fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    def _plot_performance_timing(self, ax) -> None:
        """Plot performance timing metrics (steps/sec, forward time)."""
        if not self.step_history:
            ax.text(TEXT_POSITION_CENTER, TEXT_POSITION_CENTER, 'No data yet', ha='center', va='center')
            ax.axis('off')
            return

        steps = list(self.step_history)

        # Create dual-axis plot
        ax2 = ax.twinx()

        line1 = []
        line2 = []

        # Steps per second (left axis)
        if self.timing_history['steps_per_sec']:
            steps_per_sec = list(self.timing_history['steps_per_sec'])
            line1 = ax.plot(steps, steps_per_sec, color='#2E86AB', linewidth=2,
                           marker='o', markersize=3, label='Steps/sec', alpha=0.8)

        # Forward pass time (right axis)
        if self.timing_history['forward_ms']:
            forward_ms = list(self.timing_history['forward_ms'])
            line2 = ax2.plot(steps, forward_ms, color='#A23B72', linewidth=2,
                            marker='s', markersize=3, label='Forward (ms)', alpha=0.8)

        # Styling
        ax.set_xlabel('Step')
        ax.set_ylabel('Steps per Second', color='#2E86AB')
        ax.tick_params(axis='y', labelcolor='#2E86AB')
        ax2.set_ylabel('Forward Pass (ms)', color='#A23B72')
        ax2.tick_params(axis='y', labelcolor='#A23B72')
        ax.set_title('⚡ Performance Timing', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Combined legend
        if line1 and line2:
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=9)

    def _plot_memory_usage(self, ax) -> None:
        """Plot memory usage (CPU/GPU)."""
        if not self.step_history:
            ax.text(TEXT_POSITION_CENTER, TEXT_POSITION_CENTER, 'No data yet', ha='center', va='center')
            ax.axis('off')
            return

        # Get latest memory values
        memory_types = []
        memory_values = []
        colors = []

        if self.memory_history['cpu_mb']:
            memory_types.append('CPU')
            memory_values.append(self.memory_history['cpu_mb'][-1])
            colors.append('#FF6B6B')

        if self.memory_history['gpu_mb']:
            memory_types.append('GPU')
            memory_values.append(self.memory_history['gpu_mb'][-1])
            colors.append('#4ECDC4')

        if not memory_types:
            ax.text(TEXT_POSITION_CENTER, TEXT_POSITION_CENTER, 'No memory data', ha='center', va='center')
            ax.axis('off')
            return

        # Horizontal bar chart
        y_pos = np.arange(len(memory_types))
        _bars = ax.barh(y_pos, memory_values, color=colors, alpha=0.7)

        # Add value labels on bars
        for i, value in enumerate(memory_values):
            ax.text(value, i, f' {value:.0f} MB', va='center', fontsize=9, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(memory_types)
        ax.set_xlabel('Memory (MB)')
        ax.set_title('Memory Usage', fontweight='bold', fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)

        # Set reasonable x-limit
        if memory_values:
            max_mem = max(memory_values)
            ax.set_xlim(0, max_mem * 1.3)

    def save_snapshot(self, path: str) -> None:
        """Save current diagnostics to file."""
        self.show(save_path=path)

    def create_animated_gif(
        self,
        output_path: str,
        fps: int = 5,
        duration_seconds: int = 10
    ) -> None:
        """
        Create animated GIF of training progress.

        Args:
            output_path: Path to save GIF
            fps: Frames per second
            duration_seconds: Total duration
        """
        print(f"Creating animated diagnostics... ({duration_seconds}s at {fps} fps)")

        # TODO: Implement animated GIF creation
        # Would require storing snapshots at regular intervals
        # and combining them into an animation

        print("⚠️  Animated GIF creation not yet implemented")
        print("   Use save_snapshot() at regular intervals instead")


def quick_diagnostics(brain: Any, step: int = 0) -> None:
    """
    Quick diagnostics display.

    Args:
        brain: Brain object
        step: Current training step
    """
    diag = LiveDiagnostics()
    diag.update(step, brain)
    diag.show()
