#!/usr/bin/env python3
"""
Hierarchical SNN Demo

Demonstrates the HierarchicalSNN - a multi-level spiking neural network
where different levels operate at different temporal scales:

- Sensory layer (τ=5ms): Fast processing of raw input
- Feature layer (τ=10ms): Mid-level feature extraction
- Concept layer (τ=50ms): Slower conceptual processing
- Abstract layer (τ=200ms): Very slow abstract reasoning

The key insight is that thoughts at different levels evolve at different
speeds - perceptions are fleeting while abstract concepts persist.
"""

import torch
import matplotlib.pyplot as plt
from thalia.hierarchy import HierarchicalSNN, HierarchicalConfig, LayerConfig

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print()


def demo_basic_hierarchy():
    """Demonstrate basic hierarchical processing."""
    print("=" * 60)
    print("Demo 1: Basic Hierarchical Processing")
    print("=" * 60)
    
    # Create default 4-layer hierarchy
    config = HierarchicalConfig()
    net = HierarchicalSNN(config)
    net.to(device)
    net.set_input_size(256)  # Match sensory layer size
    net.reset_state(batch_size=1)
    
    print(f"\n--- Layer Configuration ---")
    for i, layer_cfg in enumerate(config.layers):
        print(f"  Layer {i} ({layer_cfg.name:8s}): "
              f"{layer_cfg.n_neurons:3d} neurons, τ={layer_cfg.tau_mem:5.1f}ms")
    
    print(f"\n--- Processing Sensory Input ---")
    
    # Simulate 100 timesteps of sensory input
    for t in range(100):
        # Sensory input: random patterns
        sensory_input = torch.randn(1, 256, device=device) * 0.5
        result = net(sensory_input)
        
        if t % 20 == 0:
            spike_counts = [s.sum().item() for s in result["spikes"]]
            print(f"  t={t:3d}: spikes per layer = {spike_counts}")
    
    # Get temporal profile
    print(f"\n--- Temporal Profile ---")
    profile = net.get_temporal_profile()
    for name, stats in profile.items():
        print(f"  {name:8s}: τ={stats['tau']:5.1f}ms, "
              f"mean_rate={stats['mean_rate']:.4f}, "
              f"variance={stats['variance']:.6f}")


def demo_temporal_scales():
    """Demonstrate different temporal scales at different levels."""
    print("\n" + "=" * 60)
    print("Demo 2: Temporal Scales")
    print("=" * 60)
    
    # Create a simpler 2-layer hierarchy for clarity
    config = HierarchicalConfig(
        layers=[
            LayerConfig(name="fast", n_neurons=64, tau_mem=5.0, noise_std=0.1),
            LayerConfig(name="slow", n_neurons=32, tau_mem=100.0, noise_std=0.1),
        ],
        enable_feedback=False,  # One-way for simpler analysis
    )
    net = HierarchicalSNN(config)
    net.to(device)
    net.set_input_size(64)
    net.reset_state(batch_size=1)
    
    print(f"\n--- Processing with Step Input ---")
    
    # First phase: strong input
    print("  Phase 1 (t=0-50): Strong input")
    for t in range(50):
        inp = torch.ones(1, 64, device=device) * 2.0
        result = net(inp)
    
    fast_activity_phase1 = net.get_layer_activity("fast")[-10:].mean().item()
    slow_activity_phase1 = net.get_layer_activity("slow")[-10:].mean().item()
    
    # Second phase: no input
    print("  Phase 2 (t=50-150): No input (observe decay)")
    for t in range(100):
        result = net(None)
    
    fast_activity_phase2 = net.get_layer_activity("fast")[-10:].mean().item()
    slow_activity_phase2 = net.get_layer_activity("slow")[-10:].mean().item()
    
    print(f"\n--- Activity Comparison ---")
    print(f"  Fast layer: Phase 1 = {fast_activity_phase1:.4f}, "
          f"Phase 2 = {fast_activity_phase2:.4f}")
    print(f"  Slow layer: Phase 1 = {slow_activity_phase1:.4f}, "
          f"Phase 2 = {slow_activity_phase2:.4f}")
    print(f"\n  The slow layer retains activity longer due to larger τ")


def demo_top_down_modulation():
    """Demonstrate top-down feedback modulating lower layers."""
    print("\n" + "=" * 60)
    print("Demo 3: Top-Down Modulation")
    print("=" * 60)
    
    # Create hierarchy with feedback
    config = HierarchicalConfig(
        layers=[
            LayerConfig(name="sensory", n_neurons=64, tau_mem=5.0),
            LayerConfig(name="concept", n_neurons=32, tau_mem=50.0),
        ],
        feedback_strength=0.5,
        enable_feedback=True,
    )
    net = HierarchicalSNN(config)
    net.to(device)
    net.set_input_size(64)
    net.reset_state(batch_size=1)
    
    print(f"\n--- Processing with Feedback Enabled ---")
    
    # Process some input
    for t in range(50):
        inp = torch.randn(1, 64, device=device) * 0.5
        net(inp)
    
    # Now inject a pattern into the concept layer
    print("  Injecting pattern into concept layer...")
    concept_pattern = torch.zeros(1, 32, device=device)
    concept_pattern[0, :16] = 1.0  # Activate first half
    net.inject_to_layer("concept", concept_pattern, strength=2.0)
    
    # Continue processing - top-down should influence sensory
    print("  Processing with injected concept...")
    for t in range(50):
        result = net(torch.randn(1, 64, device=device) * 0.2)
    
    sensory_activity = net.get_layer_activity("sensory")
    concept_activity = net.get_layer_activity("concept")
    
    print(f"\n  Sensory layer mean activity: {sensory_activity.mean():.4f}")
    print(f"  Concept layer mean activity: {concept_activity.mean():.4f}")
    print(f"\n  Top-down feedback from concept layer influences sensory processing")


def demo_layer_activity_visualization():
    """Show how to visualize activity across layers."""
    print("\n" + "=" * 60)
    print("Demo 4: Layer Activity Analysis")
    print("=" * 60)
    
    config = HierarchicalConfig(
        layers=[
            LayerConfig(name="L1", n_neurons=32, tau_mem=5.0),
            LayerConfig(name="L2", n_neurons=32, tau_mem=20.0),
            LayerConfig(name="L3", n_neurons=32, tau_mem=80.0),
        ]
    )
    net = HierarchicalSNN(config)
    net.to(device)
    net.set_input_size(32)
    net.reset_state(batch_size=1)
    
    # Run with pulsed input
    print("\n--- Pulsed Input Experiment ---")
    pulse_times = [10, 30, 50, 70, 90]
    
    for t in range(100):
        if t in pulse_times:
            inp = torch.ones(1, 32, device=device) * 3.0
            print(f"  t={t}: Pulse!")
        else:
            inp = torch.zeros(1, 32, device=device)
        net(inp)
    
    # Get activity for each layer
    print("\n--- Activity Statistics ---")
    for layer_name in ["L1", "L2", "L3"]:
        activity = net.get_layer_activity(layer_name)
        tau = [l.tau_mem for l in config.layers if l.name == layer_name][0]
        
        # Compute autocorrelation timescale (simplified)
        mean_activity = activity.mean(dim=(1, 2))
        
        print(f"  {layer_name} (τ={tau:4.0f}ms): "
              f"total_spikes={activity.sum():.0f}, "
              f"max_per_step={mean_activity.max():.2f}")


def demo_custom_hierarchy():
    """Demonstrate creating custom hierarchical architectures."""
    print("\n" + "=" * 60)
    print("Demo 5: Custom Hierarchy")
    print("=" * 60)
    
    # Create a 3-level hierarchy with specific properties
    config = HierarchicalConfig(
        layers=[
            LayerConfig(
                name="perception",
                n_neurons=100,
                tau_mem=10.0,
                noise_std=0.05,
                recurrent=True,
                recurrent_strength=0.3,
            ),
            LayerConfig(
                name="reasoning",
                n_neurons=50,
                tau_mem=100.0,
                noise_std=0.1,
                recurrent=True,
                recurrent_strength=0.5,
            ),
            LayerConfig(
                name="metacognition",
                n_neurons=25,
                tau_mem=500.0,  # Very slow!
                noise_std=0.2,
                recurrent=True,
                recurrent_strength=0.7,
            ),
        ],
        feedforward_strength=1.0,
        feedback_strength=0.4,
    )
    
    net = HierarchicalSNN(config)
    net.to(device)
    net.set_input_size(100)
    net.reset_state(batch_size=1)
    
    print(f"\n--- Custom Hierarchy ---")
    for cfg in config.layers:
        print(f"  {cfg.name:15s}: {cfg.n_neurons:3d} neurons, "
              f"τ={cfg.tau_mem:5.0f}ms, recurrent={cfg.recurrent_strength:.1f}")
    
    # Process
    print(f"\n--- Processing ---")
    for t in range(200):
        inp = torch.randn(1, 100, device=device) * 0.3
        result = net(inp)
        
        if t % 40 == 0:
            spike_counts = [s.sum().item() for s in result["spikes"]]
            print(f"  t={t:3d}: spikes = {spike_counts}")
    
    # Final profile
    print(f"\n--- Final Temporal Profile ---")
    profile = net.get_temporal_profile()
    for name, stats in profile.items():
        print(f"  {name:15s}: mean_rate={stats['mean_rate']:.4f}")


def demo_spontaneous_activity():
    """Demonstrate noise-driven spontaneous activity."""
    print("\n" + "=" * 60)
    print("Demo 6: Spontaneous Activity (No Input)")
    print("=" * 60)
    
    config = HierarchicalConfig(
        layers=[
            LayerConfig(name="low", n_neurons=64, tau_mem=10.0, noise_std=0.3),
            LayerConfig(name="mid", n_neurons=32, tau_mem=50.0, noise_std=0.3),
            LayerConfig(name="high", n_neurons=16, tau_mem=200.0, noise_std=0.3),
        ],
        enable_feedback=True,
        feedback_strength=0.5,
    )
    net = HierarchicalSNN(config)
    net.to(device)
    net.reset_state(batch_size=1)
    
    print(f"\n--- Running Without External Input ---")
    
    total_spikes = {"low": 0, "mid": 0, "high": 0}
    
    for t in range(200):
        result = net(None)  # No external input!
        
        for i, name in enumerate(["low", "mid", "high"]):
            total_spikes[name] += result["spikes"][i].sum().item()
        
        if t % 50 == 0:
            spike_counts = [s.sum().item() for s in result["spikes"]]
            print(f"  t={t:3d}: instantaneous spikes = {spike_counts}")
    
    print(f"\n--- Total Spontaneous Spikes ---")
    for name, count in total_spikes.items():
        print(f"  {name:5s}: {count:.0f} spikes in 200 timesteps")
    
    print(f"\n  The network generates activity purely from noise and recurrence!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   THALIA - Hierarchical SNN Demo")
    print("   Multi-level temporal processing")
    print("=" * 60)
    
    demo_basic_hierarchy()
    demo_temporal_scales()
    demo_top_down_modulation()
    demo_layer_activity_visualization()
    demo_custom_hierarchy()
    demo_spontaneous_activity()
    
    print("\n" + "=" * 60)
    print("   Demo Complete!")
    print("=" * 60)
