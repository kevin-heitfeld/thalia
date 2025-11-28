#!/usr/bin/env python3
"""THALIA GPU Neural Activity Visualization

Real-time visualization of spiking neural network activity on GPU.
Shows:
1. Live spike raster plots
2. Membrane potential heatmaps
3. Network activity statistics
4. Attractor dynamics

Requires: matplotlib, numpy, torch
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
import time
from typing import Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thalia.core import LIFNeuron, LIFConfig
from thalia.dynamics import AttractorNetwork, AttractorConfig


@dataclass
class VisualizationConfig:
    """Configuration for the visualization."""
    n_neurons: int = 256
    n_attractors: int = 4
    attractor_dim: int = 64
    timesteps: int = 200
    update_interval: int = 50  # ms between updates
    history_length: int = 100  # Number of timesteps to show


class NeuralVisualizer:
    """Real-time GPU visualization of spiking neural networks."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing Neural Visualizer on {self.device}")
        
        # Create neural networks
        self._create_networks()
        
        # History buffers for visualization
        self.spike_history: list[np.ndarray] = []
        self.membrane_history: list[np.ndarray] = []
        self.activity_history: list[float] = []
        self.current_step = 0
        
        # Performance tracking
        self.fps_history: list[float] = []
        self.last_time = time.time()
        
    def _create_networks(self) -> None:
        """Create the neural networks for visualization."""
        # LIF neurons for main display
        lif_config = LIFConfig(
            tau_mem=20.0,
            v_threshold=1.0,
            noise_std=0.1
        )
        self.lif_neurons = LIFNeuron(
            n_neurons=self.config.n_neurons,
            config=lif_config
        ).to(self.device)
        
        # Attractor network for pattern dynamics
        attractor_config = AttractorConfig(
            n_neurons=self.config.attractor_dim,
            noise_std=0.05
        )
        self.attractor = AttractorNetwork(config=attractor_config).to(self.device)
        
        # Initialize states
        self.lif_neurons.reset_state(batch_size=1)
        self.attractor.reset_state(batch_size=1)
        
        # Input generator - creates varying patterns
        self.input_phase = 0.0
        
    def generate_input(self) -> torch.Tensor:
        """Generate time-varying input for the network."""
        # Multi-frequency oscillating input
        t = self.current_step * 0.1
        
        # Create spatial pattern that changes over time
        x = torch.linspace(0, 2 * np.pi, self.config.n_neurons, device=self.device)
        
        # Combination of waves at different frequencies
        pattern = (
            0.3 * torch.sin(x + t * 0.5) +
            0.2 * torch.sin(2 * x + t * 0.3) +
            0.15 * torch.cos(3 * x + t * 0.7) +
            0.1 * torch.sin(5 * x + t * 1.1) +
            0.1 * torch.randn(self.config.n_neurons, device=self.device)
        )
        
        # Add occasional "bursts"
        if self.current_step % 50 < 5:
            burst_center = (self.current_step // 50) % 4
            burst_start = burst_center * (self.config.n_neurons // 4)
            burst_end = burst_start + (self.config.n_neurons // 4)
            pattern[burst_start:burst_end] += 0.5
        
        return pattern.unsqueeze(0)  # Add batch dimension
    
    def step(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Run one simulation step and return data for visualization."""
        with torch.no_grad():
            # Generate input
            input_pattern = self.generate_input()
            
            # Run LIF neurons
            spikes, membrane = self.lif_neurons(input_pattern)
            
            # Run attractor dynamics (influenced by spike activity)
            spike_rate = spikes.mean().item()
            attractor_input = torch.randn(1, self.config.attractor_dim, device=self.device) * spike_rate * 0.5
            attractor_spikes, attractor_membrane = self.attractor(attractor_input)
            
            # Store attractor state for visualization
            self.attractor_state = attractor_membrane
            
            # Convert to numpy for plotting
            spikes_np = spikes.cpu().numpy().flatten()
            membrane_np = membrane.cpu().numpy().flatten()
            activity = spike_rate
            
            # Store history
            self.spike_history.append(spikes_np)
            self.membrane_history.append(membrane_np)
            self.activity_history.append(activity)
            
            # Trim history
            if len(self.spike_history) > self.config.history_length:
                self.spike_history.pop(0)
                self.membrane_history.pop(0)
                self.activity_history.pop(0)
            
            self.current_step += 1
            
            return spikes_np, membrane_np, activity
    
    def create_visualization(self) -> tuple[plt.Figure, list]:
        """Create the matplotlib figure and axes for visualization."""
        # Create figure with custom layout
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a2e')
        
        gs = gridspec.GridSpec(3, 3, figure=fig, 
                               height_ratios=[2, 1.5, 1],
                               hspace=0.3, wspace=0.3)
        
        # Create axes
        ax_raster = fig.add_subplot(gs[0, :2])  # Spike raster (large)
        ax_membrane = fig.add_subplot(gs[1, :2])  # Membrane heatmap
        ax_attractor = fig.add_subplot(gs[0, 2])  # Attractor state
        ax_activity = fig.add_subplot(gs[1, 2])  # Activity trace
        ax_stats = fig.add_subplot(gs[2, :])  # Statistics bar
        
        # Style all axes
        for ax in [ax_raster, ax_membrane, ax_attractor, ax_activity, ax_stats]:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#0f3460')
        
        # Titles with styling
        title_style = {'color': '#e94560', 'fontsize': 12, 'fontweight': 'bold'}
        ax_raster.set_title('Spike Raster (Live)', **title_style)
        ax_membrane.set_title('Membrane Potential Heatmap', **title_style)
        ax_attractor.set_title('Attractor State', **title_style)
        ax_activity.set_title('Network Activity', **title_style)
        ax_stats.set_title('Statistics', **title_style)
        
        # Labels
        label_style = {'color': 'white', 'fontsize': 10}
        ax_raster.set_xlabel('Time Step', **label_style)
        ax_raster.set_ylabel('Neuron Index', **label_style)
        ax_membrane.set_xlabel('Time Step', **label_style)
        ax_membrane.set_ylabel('Neuron Index', **label_style)
        ax_activity.set_xlabel('Time Step', **label_style)
        ax_activity.set_ylabel('Firing Rate', **label_style)
        
        return fig, [ax_raster, ax_membrane, ax_attractor, ax_activity, ax_stats]
    
    def run_animation(self, duration: Optional[int] = None) -> None:
        """Run the real-time animation."""
        fig, axes = self.create_visualization()
        ax_raster, ax_membrane, ax_attractor, ax_activity, ax_stats = axes
        
        # Custom colormap for neural activity
        colors = ['#0f3460', '#16213e', '#1a1a2e', '#e94560', '#ff6b6b']
        neural_cmap = LinearSegmentedColormap.from_list('neural', colors)
        
        # Initialize plots
        raster_data = np.zeros((self.config.n_neurons, self.config.history_length))
        membrane_data = np.zeros((self.config.n_neurons, self.config.history_length))
        
        im_raster = ax_raster.imshow(raster_data, aspect='auto', cmap='hot',
                                      vmin=0, vmax=1, origin='lower')
        im_membrane = ax_membrane.imshow(membrane_data, aspect='auto', cmap=neural_cmap,
                                          vmin=-0.5, vmax=1.5, origin='lower')
        
        # Attractor scatter
        attractor_scatter = ax_attractor.scatter([], [], c=[], cmap='viridis', s=50)
        ax_attractor.set_xlim(-3, 3)
        ax_attractor.set_ylim(-3, 3)
        
        # Activity line
        activity_line, = ax_activity.plot([], [], color='#e94560', linewidth=2)
        ax_activity.set_xlim(0, self.config.history_length)
        ax_activity.set_ylim(0, 0.5)
        
        # Stats text
        stats_text = ax_stats.text(0.5, 0.5, '', transform=ax_stats.transAxes,
                                    ha='center', va='center', fontsize=11,
                                    color='white', family='monospace')
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis('off')
        
        # Frame counter for duration
        frame_count = [0]
        max_frames = duration * (1000 // self.config.update_interval) if duration else None
        
        def update(frame: int) -> list:
            """Update function for animation."""
            # Check if we should stop
            if max_frames and frame_count[0] >= max_frames:
                plt.close(fig)
                return []
            
            frame_count[0] += 1
            
            # Run simulation step
            spikes, membrane, activity = self.step()
            
            # Update FPS tracking
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            fps = 1.0 / dt if dt > 0 else 0
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            avg_fps = np.mean(self.fps_history)
            
            # Build raster data from history
            if len(self.spike_history) > 0:
                raster_data = np.array(self.spike_history).T
                if raster_data.shape[1] < self.config.history_length:
                    pad = np.zeros((self.config.n_neurons, 
                                   self.config.history_length - raster_data.shape[1]))
                    raster_data = np.hstack([pad, raster_data])
                im_raster.set_array(raster_data)
            
            # Build membrane data from history
            if len(self.membrane_history) > 0:
                membrane_data = np.array(self.membrane_history).T
                if membrane_data.shape[1] < self.config.history_length:
                    pad = np.zeros((self.config.n_neurons,
                                   self.config.history_length - membrane_data.shape[1]))
                    membrane_data = np.hstack([pad, membrane_data])
                im_membrane.set_array(membrane_data)
            
            # Update attractor visualization (2D projection)
            with torch.no_grad():
                attractor_np = self.attractor_state.cpu().numpy().flatten()
                # Use first two principal components as x,y
                x_proj = attractor_np[:len(attractor_np)//2].mean()
                y_proj = attractor_np[len(attractor_np)//2:].mean()
                
                # Plot recent trajectory
                if hasattr(self, 'attractor_trajectory'):
                    self.attractor_trajectory.append((x_proj, y_proj))
                    if len(self.attractor_trajectory) > 50:
                        self.attractor_trajectory.pop(0)
                else:
                    self.attractor_trajectory = [(x_proj, y_proj)]
                
                traj = np.array(self.attractor_trajectory)
                colors = np.linspace(0.2, 1, len(traj))
                attractor_scatter.set_offsets(traj)
                attractor_scatter.set_array(colors)
            
            # Update activity trace
            if len(self.activity_history) > 0:
                x_data = np.arange(len(self.activity_history))
                activity_line.set_data(x_data, self.activity_history)
                ax_activity.set_ylim(0, max(0.5, max(self.activity_history) * 1.2))
            
            # Update statistics
            total_spikes = sum(s.sum() for s in self.spike_history) if self.spike_history else 0
            avg_membrane = np.mean([m.mean() for m in self.membrane_history]) if self.membrane_history else 0
            current_activity = self.activity_history[-1] if self.activity_history else 0
            
            stats_str = (
                f"Step: {self.current_step:5d} | "
                f"FPS: {avg_fps:5.1f} | "
                f"Device: {self.device} | "
                f"Neurons: {self.config.n_neurons} | "
                f"Spikes/step: {spikes.sum():.0f} | "
                f"Avg Membrane: {avg_membrane:.3f} | "
                f"Activity: {current_activity:.3f}"
            )
            stats_text.set_text(stats_str)
            
            return [im_raster, im_membrane, attractor_scatter, activity_line, stats_text]
        
        # Run animation
        print("\n" + "=" * 60)
        print("Starting Neural Activity Visualization")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Neurons: {self.config.n_neurons}")
        print(f"Attractors: {self.config.n_attractors}")
        print(f"Update interval: {self.config.update_interval}ms")
        if duration:
            print(f"Duration: {duration}s")
        print("\nClose the window to stop.")
        print("=" * 60 + "\n")
        
        ani = FuncAnimation(fig, update, interval=self.config.update_interval,
                           blit=True, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()


def run_text_visualization(duration: int = 10) -> None:
    """Run a text-based visualization for environments without display."""
    print("=" * 60)
    print("THALIA Neural Activity (Text Mode)")
    print("=" * 60)
    
    config = VisualizationConfig(n_neurons=64, history_length=20)
    viz = NeuralVisualizer(config)
    
    print(f"Device: {viz.device}")
    print(f"Neurons: {config.n_neurons}")
    print("\nRunning simulation...\n")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        spikes, membrane, activity = viz.step()
        
        # Create text-based raster
        spike_chars = ''.join(['█' if s > 0.5 else '░' for s in spikes[:40]])
        
        # Activity bar
        bar_len = int(activity * 50)
        activity_bar = '█' * bar_len + '░' * (50 - bar_len)
        
        # Print status
        print(f"\rStep {viz.current_step:5d} | "
              f"Spikes: [{spike_chars}] | "
              f"Activity: [{activity_bar}] {activity:.3f}", end='', flush=True)
        
        time.sleep(0.05)
    
    print("\n\n" + "=" * 60)
    print("Simulation Complete!")
    print(f"Total steps: {viz.current_step}")
    print(f"Average activity: {np.mean(viz.activity_history):.4f}")
    print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="THALIA GPU Neural Activity Visualization")
    parser.add_argument("--text", action="store_true", help="Use text-based visualization")
    parser.add_argument("--duration", type=int, default=None, 
                       help="Duration in seconds (None for infinite)")
    parser.add_argument("--neurons", type=int, default=256, 
                       help="Number of neurons to simulate")
    parser.add_argument("--interval", type=int, default=50,
                       help="Update interval in milliseconds")
    args = parser.parse_args()
    
    if args.text:
        run_text_visualization(duration=args.duration or 10)
    else:
        config = VisualizationConfig(
            n_neurons=args.neurons,
            update_interval=args.interval
        )
        viz = NeuralVisualizer(config)
        viz.run_animation(duration=args.duration)


if __name__ == "__main__":
    main()
