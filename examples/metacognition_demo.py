#!/usr/bin/env python3
"""Demo: Metacognitive Self-Monitoring in THALIA.

This demo shows how the metacognition module enables self-monitoring:
- Tracking confidence in processing
- Estimating uncertainty
- Detecting errors and conflicts
- Adapting processing based on cognitive state

The metacognitive system monitors its own processing and adjusts
behavior accordingly - a key capability for robust cognition.
"""

import time
import torch
import numpy as np

from thalia.metacognition import (
    MetacognitiveNetwork,
    MetacognitiveConfig,
    CognitiveState,
    ConfidenceLevel,
    ErrorType,
)


def demo_basic_monitoring():
    """Demonstrate basic self-monitoring capabilities."""
    print("=" * 60)
    print("DEMO 1: Basic Self-Monitoring")
    print("=" * 60)

    # Create metacognitive network
    config = MetacognitiveConfig(hidden_dim=256)
    network = MetacognitiveNetwork(input_dim=128, config=config)

    print(f"\nCreated MetacognitiveNetwork:")
    print(f"  Input dimension: 128")
    print(f"  Hidden dimension: 256")

    # Simulate some neural activity
    activity = torch.randn(128) * 0.5

    # Observe the activity
    state = network.observe(activity)

    print(f"\nAfter observing neural activity:")
    print(f"  Cognitive load: {state.load:.3f}")
    print(f"  Confidence: {state.confidence:.3f}")
    print(f"  Uncertainty: {state.uncertainty:.3f}")
    print(f"  Processing mode: {state.processing_mode}")
    print(f"  Is overloaded: {state.is_overloaded()}")
    print(f"  Is confident: {state.is_confident()}")
    print(f"  Has errors: {state.has_errors()}")

    # Get summary
    summary = network.get_summary()
    print(f"\nCognitive Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


def demo_uncertainty_tracking():
    """Show how uncertainty is tracked over time."""
    print("\n" + "=" * 60)
    print("DEMO 2: Uncertainty Tracking")
    print("=" * 60)

    network = MetacognitiveNetwork(input_dim=64)

    print("\nProcessing familiar patterns (low uncertainty):")
    # Process similar patterns
    base_pattern = torch.randn(64)

    for i in range(5):
        # Add small noise to base pattern
        pattern = base_pattern + torch.randn(64) * 0.1
        state = network.observe(pattern)
        print(f"  Step {i+1}: uncertainty = {state.uncertainty:.3f}")

    print("\nProcessing novel patterns (high uncertainty):")
    # Now process very different patterns
    for i in range(5):
        pattern = torch.randn(64) * 2  # Very different
        state = network.observe(pattern)
        print(f"  Step {i+1}: uncertainty = {state.uncertainty:.3f}")

    print("\nThe system tracks how familiar inputs are!")


def demo_confidence_dynamics():
    """Show confidence building and decay."""
    print("\n" + "=" * 60)
    print("DEMO 3: Confidence Dynamics")
    print("=" * 60)

    network = MetacognitiveNetwork(input_dim=64)

    print("\nBuilding confidence through consistent activity:")
    # Consistent patterns build confidence
    for i in range(10):
        # Same pattern = consistent = confidence builds
        pattern = torch.ones(64) * 0.5 + torch.randn(64) * 0.05
        confidence = network.get_confidence()
        level = ConfidenceLevel.from_value(confidence)
        print(f"  Step {i+1}: confidence = {confidence:.3f} ({level.name})")
        network.observe(pattern)

    print("\nConfidence erodes with conflicting input:")
    # Different patterns reduce confidence
    for i in range(5):
        pattern = torch.randn(64)  # Random = inconsistent
        confidence = network.get_confidence()
        level = ConfidenceLevel.from_value(confidence)
        print(f"  Step {i+1}: confidence = {confidence:.3f} ({level.name})")
        network.observe(pattern)
def demo_error_detection():
    """Show error detection and handling."""
    print("\n" + "=" * 60)
    print("DEMO 4: Error Detection")
    print("=" * 60)

    network = MetacognitiveNetwork(input_dim=64)

    print("\nNormal processing (no errors):")
    activity = torch.randn(64) * 0.5
    state = network.observe(activity)
    print(f"  Errors detected: {len(state.active_errors)}")

    print("\nProcessing with prediction error:")
    # Create mismatch between prediction and actual
    predicted = torch.ones(64) * 0.5
    actual = -torch.ones(64) * 0.5  # Opposite!

    state = network.observe(actual, prediction=predicted)
    print(f"  Errors detected: {len(state.active_errors)}")
    for error in state.active_errors:
        print(f"    - {error.error_type.name}: magnitude={error.magnitude:.3f}")

    # Get recommendations based on errors
    recommendations = network.get_recommendations()
    print(f"\n  Recommendations:")
    for rec in recommendations:
        print(f"    • {rec}")


def demo_adaptive_control():
    """Show how processing adapts based on cognitive state."""
    print("\n" + "=" * 60)
    print("DEMO 5: Adaptive Control")
    print("=" * 60)

    network = MetacognitiveNetwork(input_dim=64)

    print("\nNormal state adjustments:")
    activity = torch.randn(64) * 0.3
    state = network.observe(activity)
    adjustments = network.get_adjustments()
    strategy = network.get_strategy()

    print(f"  State: load={state.load:.2f}, conf={state.confidence:.2f}")
    print(f"  Strategy: {strategy}")
    print(f"  Adjustments:")
    for key, value in adjustments.items():
        print(f"    {key}: {value:.3f}")

    print("\nOverloaded state adjustments:")
    # High activity = high load
    activity = torch.ones(64) * 2.0  # High activity
    state = network.observe(activity)
    adjustments = network.get_adjustments()
    strategy = network.get_strategy()

    print(f"  State: load={state.load:.2f}, conf={state.confidence:.2f}")
    print(f"  Strategy: {strategy}")
    print(f"  Adjustments:")
    for key, value in adjustments.items():
        print(f"    {key}: {value:.3f}")

    print("\nNote: When overloaded, noise is REDUCED (0.2) and attention FOCUSED (0.9)")


def demo_continuous_monitoring():
    """Show continuous cognitive state monitoring."""
    print("\n" + "=" * 60)
    print("DEMO 6: Continuous Monitoring Simulation")
    print("=" * 60)

    network = MetacognitiveNetwork(input_dim=64)

    print("\nSimulating cognitive task with varying difficulty...")

    phases = [
        ("Easy", 0.3, 0.1),
        ("Medium", 0.6, 0.3),
        ("Hard", 1.0, 0.5),
        ("Overload", 1.5, 0.8),
        ("Recovery", 0.4, 0.2),
    ]

    for phase_name, intensity, noise in phases:
        print(f"\n  Phase: {phase_name}")

        state = None
        for _ in range(5):
            activity = torch.randn(64) * intensity + torch.randn(64) * noise
            state = network.observe(activity)

        # Report state at end of phase
        if state is not None:
            print(f"    Load: {state.load:.3f} | Conf: {state.confidence:.3f} | Uncert: {state.uncertainty:.3f}")
            print(f"    Overloaded: {state.is_overloaded()} | Strategy: {network.get_strategy()}")


def demo_gpu_performance():
    """Benchmark metacognitive processing on GPU."""
    print("\n" + "=" * 60)
    print("DEMO 7: GPU Performance Benchmark")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create network on device
    config = MetacognitiveConfig(hidden_dim=512)
    network = MetacognitiveNetwork(input_dim=256, config=config).to(device)

    # Warmup
    for _ in range(10):
        activity = torch.randn(256, device=device)
        network.observe(activity)

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_iters = 1000
    start = time.perf_counter()

    for _ in range(n_iters):
        activity = torch.randn(256, device=device)
        state = network.observe(activity)
        adjustments = network.get_adjustments()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    rate = n_iters / elapsed

    print(f"\nMetacognitive cycles: {n_iters}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Rate: {rate:.1f} observations/sec")
    print(f"Latency: {1000/rate:.2f}ms per observation")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  THALIA: Metacognitive Self-Monitoring Demo")
    print("=" * 60)
    print("\nMetacognition enables 'thinking about thinking' -")
    print("monitoring confidence, uncertainty, and errors to")
    print("adapt processing for robust cognition.")

    demo_basic_monitoring()
    demo_uncertainty_tracking()
    demo_confidence_dynamics()
    demo_error_detection()
    demo_adaptive_control()
    demo_continuous_monitoring()
    demo_gpu_performance()

    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  • Confidence tracks processing consistency")
    print("  • Uncertainty distinguishes familiar vs novel inputs")
    print("  • Errors trigger adaptive adjustments")
    print("  • The system 'knows what it knows'")
    print()


if __name__ == "__main__":
    main()
