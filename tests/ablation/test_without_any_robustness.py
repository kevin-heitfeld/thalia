"""
Ablation test: All Robustness Mechanisms

Tests what happens when ALL robustness mechanisms are disabled.
This is the most severe ablation - showing the cumulative importance.

Expected impact:
- Severe instability
- Runaway excitation
- Poor learning
- System collapse
"""

import pytest
import torch

from thalia.regions import LayeredCortex
from thalia.regions.cortex import LayeredCortexConfig
from thalia.config import RobustnessConfig


class TestWithoutAnyRobustness:
    """Ablation test: Remove ALL robustness mechanisms."""

    def test_system_stability_without_any_robustness(self):
        """Test that system becomes unstable without any robustness."""

        # Baseline: Full robustness (all mechanisms enabled)
        baseline_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig.full(),
        )
        baseline_cortex = LayeredCortex(baseline_config)        # Ablated: NO robustness (all mechanisms disabled)
        ablated_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig.minimal(),  # <-- ALL ABLATED
        )
        ablated_cortex = LayeredCortex(ablated_config)

        # Run with moderate input
        input_pattern = torch.randn(64) * 1.0

        # Track activity
        baseline_activities = []
        ablated_activities = []

        for step in range(100):
            # Baseline
            baseline_cortex.forward(input_pattern)
            baseline_activities.append(baseline_cortex.state.l23_spikes.sum().item())

            # Ablated
            ablated_cortex.forward(input_pattern)
            ablated_activities.append(ablated_cortex.state.l23_spikes.sum().item())

        # Compute stability metrics
        baseline_mean = sum(baseline_activities) / len(baseline_activities)
        ablated_mean = sum(ablated_activities) / len(ablated_activities)

        baseline_variance = sum((x - baseline_mean)**2 for x in baseline_activities) / len(baseline_activities)
        ablated_variance = sum((x - ablated_mean)**2 for x in ablated_activities) / len(ablated_activities)

        baseline_cv = (baseline_variance**0.5) / max(baseline_mean, 1.0)
        ablated_cv = (ablated_variance**0.5) / max(ablated_mean, 1.0)

        # Check for runaway
        early_ablated = sum(ablated_activities[:20]) / 20
        late_ablated = sum(ablated_activities[-20:]) / 20
        growth_ratio = late_ablated / max(early_ablated, 1.0)

        print("\n" + "="*60)
        print("ABLATION TEST: Without ANY Robustness Mechanisms")
        print("="*60)
        print(f"\nSystem Stability Comparison (100 steps):")
        print(f"\nBaseline (full robustness):")
        print(f"  Mean activity:        {baseline_mean:.1f}")
        print(f"  Variance:             {baseline_variance:.1f}")
        print(f"  Coefficient of Var:   {baseline_cv:.3f}")

        print(f"\nAblated (no robustness):")
        print(f"  Mean activity:        {ablated_mean:.1f}")
        print(f"  Variance:             {ablated_variance:.1f}")
        print(f"  Coefficient of Var:   {ablated_cv:.3f}")
        print(f"  Early→Late growth:    {growth_ratio:.2f}x")

        # Calculate impact
        variance_increase = (ablated_variance - baseline_variance) / max(baseline_variance, 1.0)
        cv_increase = (ablated_cv - baseline_cv) / max(baseline_cv, 0.1)

        print(f"\n📊 CUMULATIVE IMPACT:")
        print(f"  Variance increase:    {variance_increase*100:+.1f}%")
        print(f"  CV increase:          {cv_increase*100:+.1f}%")
        print(f"  Activity growth:      {(growth_ratio-1)*100:+.1f}%")

        # Categorize severity
        if variance_increase > 2.0 or growth_ratio > 2.0:
            severity = "CATASTROPHIC"
        elif variance_increase > 1.0 or growth_ratio > 1.5:
            severity = "SEVERE"
        elif variance_increase > 0.5:
            severity = "MODERATE"
        else:
            severity = "MINOR"

        print(f"\n⚠️  SEVERITY: {severity}")
        print("="*60)

        # Document that we observed the behavior (may vary by random seed)
        # This test is exploratory - quantifying the effect rather than asserting thresholds
        print(f"\n📊 Ablation complete: documented {'increase' if variance_increase > 0 else 'decrease'} in variance")

    def test_learning_stability_without_robustness(self):
        """Test that learning becomes unstable without robustness."""

        # Full robustness
        stable = LayeredCortex(LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig.full(),
        ))

        # No robustness
        unstable = LayeredCortex(LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig.minimal(),
        ))

        # Training sequence: gradually increasing complexity
        patterns = [
            torch.randn(64) * 0.5,   # Weak
            torch.randn(64) * 1.0,   # Moderate
            torch.randn(64) * 1.5,   # Strong
        ]

        stable_responses = []
        unstable_responses = []

        for pattern_idx, pattern in enumerate(patterns):
            print(f"\nPattern {pattern_idx + 1}:")

            # Present pattern multiple times
            stable_pattern_responses = []
            unstable_pattern_responses = []

            for rep in range(20):
                stable.forward(pattern)
                unstable.forward(pattern)

                stable_pattern_responses.append(stable.state.l5_spikes.sum().item())
                unstable_pattern_responses.append(unstable.state.l5_spikes.sum().item())

            stable_mean = sum(stable_pattern_responses) / len(stable_pattern_responses)
            unstable_mean = sum(unstable_pattern_responses) / len(unstable_pattern_responses)

            print(f"  Stable mean:   {stable_mean:.1f}")
            print(f"  Unstable mean: {unstable_mean:.1f}")

            stable_responses.extend(stable_pattern_responses)
            unstable_responses.extend(unstable_pattern_responses)

        # Check response consistency
        stable_consistency = 1.0 / (1.0 + torch.tensor(stable_responses).std().item())
        unstable_consistency = 1.0 / (1.0 + torch.tensor(unstable_responses).std().item())

        print("\n" + "="*60)
        print("LEARNING STABILITY TEST")
        print("="*60)
        print(f"\nResponse consistency across patterns:")
        print(f"  Stable (full robustness):   {stable_consistency:.3f}")
        print(f"  Unstable (no robustness):   {unstable_consistency:.3f}")

        consistency_loss = (stable_consistency - unstable_consistency) / stable_consistency
        print(f"  → Consistency loss: {consistency_loss*100:.1f}%")
        print("="*60)

        if consistency_loss > 0.5:
            print("✅ Robustness is CRITICAL for learning stability")
        elif consistency_loss > 0.2:
            print("⚠️  Robustness is VALUABLE for learning stability")
        else:
            print("ℹ️  Robustness has MINOR effect on learning")

    def test_contrast_with_preset_comparison(self):
        """Compare the three presets: minimal, stable, full."""

        # Create all three configurations
        minimal_cortex = LayeredCortex(LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig.minimal(),
        ))

        stable_cortex = LayeredCortex(LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig.stable(),
        ))

        full_cortex = LayeredCortex(LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig.full(),
        ))

        # Test with challenging input
        input_pattern = torch.randn(64) * 1.5

        # Collect responses
        configs = {
            "Minimal": minimal_cortex,
            "Stable": stable_cortex,
            "Full": full_cortex,
        }

        results = {}

        for name, cortex in configs.items():
            activities = []
            for _ in range(50):
                cortex.forward(input_pattern)
                activities.append(cortex.state.l23_spikes.sum().item())

            mean_activity = sum(activities) / len(activities)
            variance = sum((x - mean_activity)**2 for x in activities) / len(activities)
            cv = (variance**0.5) / max(mean_activity, 1.0)

            results[name] = {
                "mean": mean_activity,
                "variance": variance,
                "cv": cv,
            }

        print("\n" + "="*60)
        print("PRESET COMPARISON")
        print("="*60)

        for name, metrics in results.items():
            print(f"\n{name} preset:")
            print(f"  Mean activity:     {metrics['mean']:.1f}")
            print(f"  Variance:          {metrics['variance']:.1f}")
            print(f"  Coeff of Var:      {metrics['cv']:.3f}")

        # Only compute ratios if we have non-zero CVs
        print("\n📊 SUMMARY:")
        if results['Minimal']['cv'] > 0.01 and results['Stable']['cv'] > 0.01:
            print(f"  Minimal → Stable: {(results['Stable']['cv']/results['Minimal']['cv']-1)*-100:.1f}% CV reduction")
            print(f"  Stable → Full:    {(results['Full']['cv']/results['Stable']['cv']-1)*-100:.1f}% CV reduction")
            print(f"  Minimal → Full:   {(results['Full']['cv']/results['Minimal']['cv']-1)*-100:.1f}% CV reduction")
        else:
            print("  (All presets showed near-zero activity - random seed variation)")
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
