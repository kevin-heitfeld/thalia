#!/usr/bin/env python3
"""Interactive THALIA Demo

An interactive visualization of the THALIA framework's cognitive processes:
- Real-time spiking neuron activity
- Attractor network dynamics
- Daydreaming/spontaneous thought

Run with: python examples/interactive_demo.py --text
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import sys
import time
from collections import deque

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thalia.core import LIFNeuron, LIFConfig
from thalia.dynamics import AttractorNetwork, AttractorConfig


class ThaliaVisualizer:
    """Real-time visualization of THALIA cognitive processes."""
    
    def __init__(self, n_neurons: int = 50, n_patterns: int = 3):
        self.n_neurons = n_neurons
        self.n_patterns = n_patterns
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create attractor network
        self.attractor_config = AttractorConfig(
            n_neurons=n_neurons,
            tau_mem=20.0,
            noise_std=0.05,
            sparsity=0.15
        )
        self.attractor_net = AttractorNetwork(config=self.attractor_config)
        
        # Store some random patterns
        self.patterns = []
        for i in range(n_patterns):
            pattern = (torch.rand(n_neurons) < self.attractor_config.sparsity).float()
            self.attractor_net.store_pattern(pattern)
            self.patterns.append(pattern)
        
        # Move to device
        self.attractor_net = self.attractor_net.to(self.device)
        
        # History buffers
        self.spike_history = deque(maxlen=100)
        self.membrane_history = deque(maxlen=100)
        self.pattern_similarity_history = deque(maxlen=200)
        
        # State
        self.mode = "observe"
        self.timestep = 0
        
        # Reset
        self.attractor_net.reset_state(batch_size=1)
        
    def step(self):
        """Perform one simulation step."""
        self.timestep += 1
        
        # Generate input based on mode
        if self.mode == "stimulate":
            # Use stored pattern as cue
            idx = (self.timestep // 20) % self.n_patterns
            pattern = self.patterns[idx].to(self.device)
            external_input = pattern.unsqueeze(0) * 0.5
            external_input[:, self.n_neurons//2:] = 0  # Partial cue
        elif self.mode == "daydream":
            external_input = torch.randn(1, self.n_neurons, device=self.device) * 0.1
        else:
            external_input = torch.randn(1, self.n_neurons, device=self.device) * 0.05
        
        # Run attractor network
        spikes, membrane = self.attractor_net(external_input)
        
        # Compute similarity to stored patterns
        similarities = []
        with torch.no_grad():
            for pattern in self.patterns:
                pattern = pattern.to(self.device)
                sim = torch.cosine_similarity(spikes.squeeze(), pattern, dim=0)
                similarities.append(sim.item())
        
        best_pattern = int(np.argmax(similarities))
        best_similarity = max(similarities)
        
        # Store history
        self.spike_history.append(spikes.cpu().detach().numpy().flatten())
        self.membrane_history.append(membrane.cpu().detach().numpy().flatten())
        self.pattern_similarity_history.append((best_pattern, best_similarity))
        
        return spikes, membrane, best_similarity, best_pattern
    
    def set_mode(self, mode: str):
        """Set simulation mode."""
        self.mode = mode
        if mode == "daydream":
            self.attractor_net.reset_state(1)
    
    def get_statistics(self):
        """Get current statistics."""
        if len(self.spike_history) == 0:
            return {}
        
        recent_spikes = np.array(list(self.spike_history)[-20:])
        firing_rate = recent_spikes.mean() * 100
        
        recent_sims = [s for _, s in list(self.pattern_similarity_history)[-20:]]
        mean_similarity = np.mean(recent_sims) if recent_sims else 0
        
        recent_patterns = [p for p, _ in list(self.pattern_similarity_history)[-50:]]
        transitions = sum(1 for i in range(1, len(recent_patterns)) 
                         if recent_patterns[i] != recent_patterns[i-1])
        
        return {
            "firing_rate": firing_rate,
            "mean_similarity": mean_similarity,
            "pattern_transitions": transitions,
            "current_pattern": recent_patterns[-1] if recent_patterns else 0,
            "timestep": self.timestep,
        }


def run_text_demo():
    """Run a text-based demo."""
    print("=" * 60)
    print("THALIA Text-Based Demo")
    print("=" * 60)
    
    viz = ThaliaVisualizer(n_neurons=30, n_patterns=3)
    
    print(f"\nDevice: {viz.device}")
    print(f"Neurons: {viz.n_neurons}")
    print(f"Patterns: {viz.n_patterns}")
    
    modes = ['observe', 'stimulate', 'daydream']
    
    for mode in modes:
        print(f"\n{'=' * 40}")
        print(f"Mode: {mode.upper()}")
        print("=" * 40)
        
        viz.set_mode(mode)
        
        for step in range(25):
            spikes, membrane, similarity, pattern = viz.step()
            
            # Create spike visualization (ASCII art)
            spike_counts = spikes.sum().item()
            spike_bar = "█" * int(spike_counts * 3) + "░" * (30 - int(spike_counts * 3))
            
            print(f"  t={viz.timestep:3d} |{spike_bar[:30]}| "
                  f"sim={similarity:.2f} P{pattern}")
        
        stats = viz.get_statistics()
        print(f"\n  Summary: FR={stats['firing_rate']:.1f}%, "
              f"Sim={stats['mean_similarity']:.3f}, "
              f"Transitions={stats['pattern_transitions']}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return viz.get_statistics()


def run_interactive_demo():
    """Run the interactive demo with matplotlib animation."""
    print("=" * 60)
    print("THALIA Interactive Demo")
    print("=" * 60)
    print("\nInitializing neural networks...")
    
    viz = ThaliaVisualizer(n_neurons=50, n_patterns=4)
    
    print(f"Device: {viz.device}")
    print(f"Neurons: {viz.n_neurons}")
    print(f"Patterns: {viz.n_patterns}")
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("THALIA: Real-Time Cognitive Visualization", fontsize=14, fontweight='bold')
    
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax_spikes = fig.add_subplot(gs[0, :2])
    ax_membrane = fig.add_subplot(gs[1, :2])
    ax_sim = fig.add_subplot(gs[0, 2])
    ax_pattern = fig.add_subplot(gs[1, 2])
    ax_stats = fig.add_subplot(gs[2, 0])
    ax_controls = fig.add_subplot(gs[2, 1])
    ax_info = fig.add_subplot(gs[2, 2])
    
    # Initialize spike raster
    spike_image = np.zeros((viz.n_neurons, 100))
    im_spikes = ax_spikes.imshow(spike_image, aspect='auto', cmap='binary', 
                                  vmin=0, vmax=1, interpolation='nearest')
    ax_spikes.set_xlabel("Time (steps)")
    ax_spikes.set_ylabel("Neuron ID")
    ax_spikes.set_title("Spike Raster")
    
    # Initialize membrane plot
    membrane_image = np.zeros((viz.n_neurons, 100))
    im_membrane = ax_membrane.imshow(membrane_image, aspect='auto', cmap='RdBu_r', 
                                      vmin=-1, vmax=2, interpolation='bilinear')
    ax_membrane.set_xlabel("Time (steps)")
    ax_membrane.set_ylabel("Neuron ID")
    ax_membrane.set_title("Membrane Potentials")
    plt.colorbar(im_membrane, ax=ax_membrane, label='Voltage')
    
    # Similarity plot
    sim_line, = ax_sim.plot([], [], 'b-', linewidth=2)
    ax_sim.set_xlim(0, 200)
    ax_sim.set_ylim(-0.5, 1)
    ax_sim.set_xlabel("Time")
    ax_sim.set_ylabel("Similarity")
    ax_sim.set_title("Pattern Similarity")
    ax_sim.grid(True, alpha=0.3)
    ax_sim.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Pattern distribution (pie chart)
    pattern_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ax_pattern.set_title("Pattern Distribution")
    
    # Statistics text
    ax_stats.axis('off')
    stats_text = ax_stats.text(0.1, 0.9, "", transform=ax_stats.transAxes,
                               fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax_stats.set_title("Statistics")
    
    # Mode indicator
    ax_controls.axis('off')
    mode_colors = {'observe': '#4ECDC4', 'stimulate': '#FF6B6B', 'daydream': '#45B7D1'}
    mode_text = ax_controls.text(0.5, 0.5, "OBSERVE", transform=ax_controls.transAxes,
                                  fontsize=20, ha='center', va='center', 
                                  fontweight='bold', color=mode_colors['observe'])
    ax_controls.set_title("Current Mode")
    
    # Info
    ax_info.axis('off')
    info_str = """Press keys to change mode:
O - Observe (default)
S - Stimulate neurons
D - Daydream mode
Q - Quit"""
    ax_info.text(0.1, 0.9, info_str, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax_info.set_title("Controls")
    
    current_mode = ['observe']
    running = [True]
    
    def on_key(event):
        if event.key == 'o':
            current_mode[0] = 'observe'
            viz.set_mode('observe')
        elif event.key == 's':
            current_mode[0] = 'stimulate'
            viz.set_mode('stimulate')
        elif event.key == 'd':
            current_mode[0] = 'daydream'
            viz.set_mode('daydream')
        elif event.key == 'q':
            running[0] = False
            plt.close()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    def update(frame):
        if not running[0]:
            return []
        
        viz.step()
        
        # Update spike raster
        if len(viz.spike_history) > 0:
            spike_data = np.array(list(viz.spike_history)).T
            if spike_data.shape[1] < 100:
                padded = np.zeros((viz.n_neurons, 100))
                padded[:, -spike_data.shape[1]:] = spike_data
                spike_data = padded
            im_spikes.set_array(spike_data)
        
        # Update membrane potentials
        if len(viz.membrane_history) > 0:
            membrane_data = np.array(list(viz.membrane_history)).T
            if membrane_data.shape[1] < 100:
                padded = np.zeros((viz.n_neurons, 100))
                padded[:, -membrane_data.shape[1]:] = membrane_data
                membrane_data = padded
            im_membrane.set_array(membrane_data)
        
        # Update similarity plot
        sim_data = [s for _, s in list(viz.pattern_similarity_history)]
        sim_line.set_data(range(len(sim_data)), sim_data)
        
        # Update pattern distribution
        ax_pattern.clear()
        if len(viz.pattern_similarity_history) > 0:
            pattern_counts = [0] * viz.n_patterns
            for p, _ in viz.pattern_similarity_history:
                pattern_counts[p] += 1
            total = sum(pattern_counts)
            if total > 0:
                sizes = [c/total for c in pattern_counts]
                labels = [f'P{i}' for i in range(viz.n_patterns)]
                ax_pattern.pie(sizes, labels=labels, colors=pattern_colors[:viz.n_patterns],
                              autopct='%1.0f%%', startangle=90)
        ax_pattern.set_title("Pattern Distribution")
        
        # Update statistics
        stats = viz.get_statistics()
        stats_str = f"""Timestep: {stats.get('timestep', 0)}
Firing Rate: {stats.get('firing_rate', 0):.1f}%
Similarity: {stats.get('mean_similarity', 0):.3f}
Transitions: {stats.get('pattern_transitions', 0)}
Current Pattern: P{stats.get('current_pattern', 0)}"""
        stats_text.set_text(stats_str)
        
        # Update mode display
        mode = current_mode[0]
        mode_text.set_text(mode.upper())
        mode_text.set_color(mode_colors.get(mode, 'black'))
        
        return [im_spikes, im_membrane, sim_line, stats_text, mode_text]
    
    print("\nStarting visualization...")
    print("Press O=Observe, S=Stimulate, D=Daydream, Q=Quit")
    
    anim = FuncAnimation(fig, update, frames=None, interval=50, blit=False, cache_frame_data=False)
    
    plt.tight_layout()
    plt.show()
    
    print("\nDemo finished!")
    return viz.get_statistics()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="THALIA Interactive Demo")
    parser.add_argument("--text", action="store_true", help="Run text-only demo")
    args = parser.parse_args()
    
    if args.text:
        run_text_demo()
    else:
        try:
            run_interactive_demo()
        except Exception as e:
            print(f"Interactive mode failed ({e}), falling back to text mode...")
            run_text_demo()
