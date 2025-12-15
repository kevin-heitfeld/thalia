"""
Example: Cross-Region Phase Synchrony Metrics

This example demonstrates how to use cross-region phase synchrony metrics
to validate curriculum mechanisms:
- Working memory: Hippocampus-PFC theta coherence (Stage 1+)
- Cross-modal binding: Visual-auditory gamma coherence (Stage 2+)
"""

import math
from thalia.diagnostics import OscillatorHealthMonitor
from thalia.coordination.oscillator import OscillatorManager


def example_working_memory_synchrony():
    """Validate working memory via hippocampus-PFC theta coherence."""
    print("=== Working Memory Synchrony (Curriculum Stage 1+) ===\n")

    monitor = OscillatorHealthMonitor()

    # Simulate brain state during working memory task
    # GOOD: High theta coherence between hippocampus and PFC
    region_phases_good = {
        'hippocampus': {'theta': 1.0, 'gamma': 2.0},
        'prefrontal': {'theta': 1.05, 'gamma': 2.2},  # Very close theta phases
    }

    print("Scenario 1: Good Working Memory")
    coherence_map = monitor.compute_region_pair_coherence(
        region_phases_good,
        region_pairs=[('hippocampus', 'prefrontal')],
        oscillators=['theta', 'gamma']
    )

    for pair_key, osc_coherence in coherence_map.items():
        print(f"  {pair_key}:")
        for osc, coherence in osc_coherence.items():
            print(f"    {osc}: {coherence:.3f} {'✓' if coherence > 0.6 else '✗'}")

    issues = monitor.check_cross_region_synchrony(region_phases_good)
    print(f"  Issues: {len(issues)} {'✓ PASS' if len(issues) == 0 else '✗ FAIL'}\n")

    # BAD: Low theta coherence (opposite phases)
    region_phases_bad = {
        'hippocampus': {'theta': 0.0, 'gamma': 2.0},
        'prefrontal': {'theta': math.pi, 'gamma': 2.2},  # Opposite theta phases
    }

    print("Scenario 2: Poor Working Memory")
    coherence_map = monitor.compute_region_pair_coherence(
        region_phases_bad,
        region_pairs=[('hippocampus', 'prefrontal')],
        oscillators=['theta', 'gamma']
    )

    for pair_key, osc_coherence in coherence_map.items():
        print(f"  {pair_key}:")
        for osc, coherence in osc_coherence.items():
            print(f"    {osc}: {coherence:.3f} {'✓' if coherence > 0.6 else '✗'}")

    issues = monitor.check_cross_region_synchrony(region_phases_bad)
    print(f"  Issues: {len(issues)} {'✗ FAIL' if len(issues) > 0 else '✓ PASS'}")
    if issues:
        for issue in issues:
            print(f"    - {issue.description}")
    print()


def example_cross_modal_binding_synchrony():
    """Validate cross-modal binding via visual-auditory gamma coherence."""
    print("=== Cross-Modal Binding Synchrony (Curriculum Stage 2+) ===\n")

    monitor = OscillatorHealthMonitor()

    # GOOD: High gamma coherence for binding
    region_phases_good = {
        'visual_cortex': {'gamma': 2.0, 'theta': 1.0},
        'auditory_cortex': {'gamma': 2.05, 'theta': 1.5},  # Close gamma phases
    }

    print("Scenario 1: Good Cross-Modal Binding")
    coherence_map = monitor.compute_region_pair_coherence(
        region_phases_good,
        region_pairs=[('visual_cortex', 'auditory_cortex')],
        oscillators=['gamma']
    )

    for pair_key, osc_coherence in coherence_map.items():
        print(f"  {pair_key}:")
        for osc, coherence in osc_coherence.items():
            print(f"    {osc}: {coherence:.3f} {'✓' if coherence > 0.4 else '✗'}")

    issues = monitor.check_cross_region_synchrony(region_phases_good)
    print(f"  Issues: {len(issues)} {'✓ PASS' if len(issues) == 0 else '✗ FAIL'}\n")

    # BAD: Low gamma coherence (desynchronized)
    region_phases_bad = {
        'visual_cortex': {'gamma': 0.0, 'theta': 1.0},
        'auditory_cortex': {'gamma': math.pi, 'theta': 1.5},  # Opposite gamma
    }

    print("Scenario 2: Poor Cross-Modal Binding")
    coherence_map = monitor.compute_region_pair_coherence(
        region_phases_bad,
        region_pairs=[('visual_cortex', 'auditory_cortex')],
        oscillators=['gamma']
    )

    for pair_key, osc_coherence in coherence_map.items():
        print(f"  {pair_key}:")
        for osc, coherence in osc_coherence.items():
            print(f"    {osc}: {coherence:.3f} {'✓' if coherence > 0.4 else '✗'}")

    issues = monitor.check_cross_region_synchrony(region_phases_bad)
    print(f"  Issues: {len(issues)} {'✗ FAIL' if len(issues) > 0 else '✓ PASS'}")
    if issues:
        for issue in issues:
            print(f"    - {issue.description}")
    print()


def example_multi_region_synchrony():
    """Analyze synchrony across multiple regions simultaneously."""
    print("=== Multi-Region Synchrony Analysis ===\n")

    monitor = OscillatorHealthMonitor()

    # Simulate complex brain state with 4 regions
    region_phases = {
        'hippocampus': {'theta': 1.0, 'gamma': 2.0},
        'prefrontal': {'theta': 1.05, 'gamma': 2.2},    # Good theta sync with hippo
        'cortex': {'theta': 1.1, 'gamma': 2.5},         # Moderate sync
        'striatum': {'theta': 2.0, 'gamma': 3.0},       # Poor sync
    }

    print("Computing coherence for all region pairs...")
    coherence_map = monitor.compute_region_pair_coherence(
        region_phases,
        oscillators=['theta', 'gamma']
    )

    print(f"\nFound {len(coherence_map)} region pairs:\n")
    for pair_key, osc_coherence in coherence_map.items():
        print(f"{pair_key}:")
        for osc, coherence in osc_coherence.items():
            status = '✓' if coherence > 0.6 else ('~' if coherence > 0.4 else '✗')
            print(f"  {osc}: {coherence:.3f} {status}")
        print()


def example_real_time_monitoring():
    """Monitor synchrony changes over time."""
    print("=== Real-Time Synchrony Monitoring ===\n")

    monitor = OscillatorHealthMonitor()
    oscillators = OscillatorManager(dt_ms=1.0, device="cpu")

    print("Monitoring hippocampus-PFC synchrony over 50 timesteps...\n")

    # Track coherence over time
    theta_coherence_history = []

    for step in range(50):
        oscillators.advance(dt_ms=1.0)

        # Simulate slight phase drift between regions
        region_phases = {
            'hippocampus': {'theta': oscillators.theta.phase},
            'prefrontal': {
                'theta': oscillators.theta.phase + (step * 0.01)  # Gradual drift
            },
        }

        coherence = monitor.compute_phase_coherence(
            region_phases['hippocampus'],
            region_phases['prefrontal'],
            'theta'
        )
        theta_coherence_history.append(coherence)

        if step % 10 == 0:
            print(f"Step {step:2d}: Theta coherence = {coherence:.3f}")

    # Analyze trend
    initial_coherence = sum(theta_coherence_history[:10]) / 10
    final_coherence = sum(theta_coherence_history[-10:]) / 10

    print(f"\nInitial coherence (steps 0-9):   {initial_coherence:.3f}")
    print(f"Final coherence (steps 40-49):   {final_coherence:.3f}")
    print(f"Change: {final_coherence - initial_coherence:+.3f}")

    if final_coherence < 0.5:
        print("⚠ Warning: Synchrony degraded significantly!")
    else:
        print("✓ Synchrony maintained")


if __name__ == "__main__":
    print("Cross-Region Phase Synchrony Metrics")
    print("=" * 50)
    print()

    example_working_memory_synchrony()
    print("=" * 50)
    print()

    example_cross_modal_binding_synchrony()
    print("=" * 50)
    print()

    example_multi_region_synchrony()
    print("=" * 50)
    print()

    example_real_time_monitoring()
