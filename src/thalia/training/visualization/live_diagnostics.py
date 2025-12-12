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

from thalia.core.spike_utils import compute_firing_rate
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

        # Store latest spikes for raster
        if spikes:
            self.latest_spikes = spikes

    def show(self, save_path: Optional[str] = None) -> None:
        """
        Display current diagnostics.

        Args:
            save_path: If provided, save figure to this path
        """
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Row 1: Neural activity
        ax_spikes = fig.add_subplot(gs[0, :2])
        self._plot_spike_raster(ax_spikes)

        ax_firing_rates = fig.add_subplot(gs[0, 2])
        self._plot_firing_rate_distribution(ax_firing_rates)

        # Row 2: Health metrics
        ax_health = fig.add_subplot(gs[1, :])
        self._plot_health_metrics(ax_health)

        # Row 3: Performance
        ax_performance = fig.add_subplot(gs[2, :])
        self._plot_performance(ax_performance)

        # Add title
        fig.suptitle(
            f'🧠 Thalia Live Diagnostics (Step {self.current_step})',
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
        ax.set_title('🔥 Spike Raster (Recent Activity)', fontweight='bold')
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
        ax.set_title('🏥 Health Metrics', fontweight='bold')
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
        ax.set_title('📈 Task Performance', fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

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
