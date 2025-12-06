"""
Demo: Health Dashboard

This script demonstrates the health monitoring dashboard by:
1. Simulating various network conditions (synthetic diagnostics)
2. Displaying real-time health metrics
3. Saving a report

To run: python experiments/scripts/demo_dashboard.py

Note: This demo uses synthetic diagnostics rather than a full brain
to make it easy to demonstrate all health monitoring features.
"""

import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt

from thalia.diagnostics import Dashboard, HealthConfig


def generate_synthetic_diagnostics(phase: str, timestep: int) -> dict:
    """Generate synthetic diagnostics for different simulation phases."""

    if phase == "normal":
        # Healthy network state
        return {
            "spike_counts": {"cortex": 25, "hippocampus": 20, "striatum": 15},
            "cortex": {
                "l23_w_mean": 0.5,
                "l4_w_mean": 0.6,
                "l5_w_mean": 0.55,
            },
            "robustness_ei_ratio": 4.0,
            "criticality": {
                "enabled": True,
                "branching_ratio": 1.0,
            },
            "dopamine": {
                "global": 0.5,
                "tonic": 0.1,
                "phasic": 0.0,
            },
        }

    elif phase == "increasing_excitation":
        # Gradually increasing activity (seizure risk)
        spike_base = 25 + timestep * 2
        return {
            "spike_counts": {"cortex": spike_base, "hippocampus": spike_base, "striatum": spike_base},
            "cortex": {
                "l23_w_mean": 0.5 + timestep * 0.05,
                "l4_w_mean": 0.6,
                "l5_w_mean": 0.55,
            },
            "robustness_ei_ratio": 4.0 + timestep * 0.2,  # E/I imbalance
            "criticality": {
                "enabled": True,
                "branching_ratio": 1.0 + timestep * 0.01,  # Going supercritical
            },
            "dopamine": {"global": 0.5, "tonic": 0.1, "phasic": 0.0},
        }

    elif phase == "collapse":
        # Very low activity
        return {
            "spike_counts": {"cortex": 1, "hippocampus": 0, "striatum": 1},
            "cortex": {
                "l23_w_mean": 0.005,  # Weight collapse
                "l4_w_mean": 0.006,
                "l5_w_mean": 0.004,
            },
            "robustness_ei_ratio": 0.5,  # Over-inhibited
            "criticality": {
                "enabled": True,
                "branching_ratio": 0.3,  # Subcritical
            },
            "dopamine": {"global": 0.1, "tonic": 0.05, "phasic": 0.0},
        }

    elif phase == "recovery":
        # Gradually recovering
        spike_base = 1 + timestep * 1
        return {
            "spike_counts": {"cortex": spike_base, "hippocampus": spike_base, "striatum": spike_base},
            "cortex": {
                "l23_w_mean": 0.1 + timestep * 0.02,
                "l4_w_mean": 0.15 + timestep * 0.02,
                "l5_w_mean": 0.12 + timestep * 0.02,
            },
            "robustness_ei_ratio": 1.0 + timestep * 0.1,
            "criticality": {
                "enabled": True,
                "branching_ratio": 0.5 + timestep * 0.02,
            },
            "dopamine": {"global": 0.2 + timestep * 0.01, "tonic": 0.1, "phasic": 0.0},
        }

    else:
        # Default to normal
        return generate_synthetic_diagnostics("normal", 0)


def run_simulation_with_monitoring():
    """Run a simulated network with real-time health monitoring."""

    print("Initializing health dashboard...")
    dashboard = Dashboard(
        health_config=HealthConfig(),
        window_size=50,  # Show last 50 timesteps
    )

    print("\nRunning simulation with synthetic diagnostics...")
    print("Watch the dashboard window for real-time health updates!")
    print("Close the dashboard window or press Ctrl+C to stop.\n")

    # Simulation phases
    phases = [
        ("normal", 50, "Normal activity"),
        ("increasing_excitation", 30, "Increasing excitation (seizure risk)"),
        ("collapse", 20, "Activity collapse"),
        ("recovery", 40, "Recovery"),
    ]

    try:
        for phase_key, duration, phase_name in phases:
            print(f"Phase: {phase_name} ({duration} steps)")

            for t in range(duration):
                # Generate synthetic diagnostics
                diagnostics = generate_synthetic_diagnostics(phase_key, t)

                # Update dashboard
                dashboard.update(diagnostics)

                # Show every 3 steps
                if (dashboard._current_timestep - 1) % 3 == 0:
                    dashboard.show(block=False)

            print(f"  Completed {duration} steps\n")

    except KeyboardInterrupt:
        print("\nStopping simulation...")

    finally:
        # Print summary
        print("\nSimulation complete!")
        dashboard.print_summary()

        # Save report
        output_path = "experiments/results/health_dashboard_demo.png"
        dashboard.save_report(output_path)

        print(f"\nDashboard saved to {output_path}")
        print("\nShowing final dashboard...")
        print("Close the window to exit the demo.")
        
        # Final display - show non-blocking first to create the window
        dashboard.show(block=False)
        
        # Then block with plt.show() to keep window open until user closes it
        plt.show()
        
        dashboard.close()


if __name__ == "__main__":
    run_simulation_with_monitoring()
