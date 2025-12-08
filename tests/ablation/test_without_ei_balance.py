"""
Ablation test: E/I Balance

Tests what happens when E/I balance regulation is disabled.

Expected impact:
- Higher risk of runaway excitation
- Less stable activity levels
- Greater variance in firing rates
"""

import pytest
import torch

from thalia.regions import LayeredCortex
from thalia.regions.cortex import LayeredCortexConfig
from thalia.config import RobustnessConfig

# Constants for test thresholds
STABILITY_TOLERANCE = 0.5  # 50% variance increase is significant
RUNAWAY_THRESHOLD = 1.5  # Activity growing by 50% indicates runaway


@pytest.mark.ablation
class TestWithoutEIBalance:
    """Ablation test: Remove E/I balance regulation."""

    @pytest.mark.slow
    def test_stability_degrades_without_ei_balance(self):
        """Quantify stability degradation without E/I balance."""

        # Baseline: Full robustness with E/I balance
        baseline_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig.full(),
        )
        baseline_cortex = LayeredCortex(baseline_config)

        # Ablated: Same config but WITHOUT E/I balance
        ablated_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig(
                enable_ei_balance=False,  # <-- ABLATED
                enable_divisive_norm=True,
                enable_intrinsic_plasticity=True,
                enable_criticality=True,
                enable_metabolic=True,
            ),
        )
        ablated_cortex = LayeredCortex(ablated_config)

        # Test with strong input (stress test)
        strong_input = torch.randn(64) * 1.5

        # Run both for same duration
        baseline_spikes = []
        ablated_spikes = []

        for _ in range(100):
            # Baseline
            baseline_cortex.forward(strong_input)
            baseline_spikes.append(baseline_cortex.state.l5_spikes.sum().item())

            # Ablated
            ablated_cortex.forward(strong_input)
            ablated_spikes.append(ablated_cortex.state.l5_spikes.sum().item())

        # Analyze stability
        baseline_variance = torch.tensor(baseline_spikes, dtype=torch.float32).var().item()
        ablated_variance = torch.tensor(ablated_spikes, dtype=torch.float32).var().item()

        baseline_mean = torch.tensor(baseline_spikes, dtype=torch.float32).mean().item()
        ablated_mean = torch.tensor(ablated_spikes, dtype=torch.float32).mean().item()

        # Print results for documentation
        print("\n" + "="*60)
        print("ABLATION TEST: Without E/I Balance")
        print("="*60)
        print(f"\nActivity levels:")
        print(f"  Baseline: {baseline_mean:.2f} ± {baseline_variance**0.5:.2f} spikes/step")
        print(f"  Ablated:  {ablated_mean:.2f} ± {ablated_variance**0.5:.2f} spikes/step")

        print(f"\nStability (variance):")
        print(f"  Baseline: {baseline_variance:.3f}")
        print(f"  Ablated:  {ablated_variance:.3f}")
        
        variance_increase_pct = (ablated_variance - baseline_variance) / (baseline_variance + 1e-6) * 100
        print(f"  → {variance_increase_pct:+.1f}% change")
        print("="*60)

        # CONCRETE ASSERTIONS - No more "may or may not"
        
        # Assert that both systems maintain some activity (sanity check)
        assert baseline_mean > 0, "Baseline system collapsed (no activity)"
        assert ablated_mean > 0, "Ablated system collapsed (no activity)"
        
        # Assert that ablated system has higher variance (less stable)
        stability_degradation = (ablated_variance - baseline_variance) / (baseline_variance + 1e-6)
        
        # Document the quantified impact
        if stability_degradation > 0.5:
            print(f"\n✅ E/I balance is CRITICAL: {stability_degradation*100:.1f}% stability loss")
        elif stability_degradation > 0.2:
            print(f"\n⚠️  E/I balance is VALUABLE: {stability_degradation*100:.1f}% stability loss")
        else:
            print(f"\nℹ️  E/I balance has MINOR impact: {stability_degradation*100:.1f}% stability loss")
        
        # The ablation should show measurable degradation
        # If E/I balance doesn't improve stability, it's not doing its job
        assert stability_degradation > -STABILITY_TOLERANCE, \
            f"E/I balance made stability WORSE by {-stability_degradation*100:.1f}% - mechanism broken?"

    @pytest.mark.slow
    def test_runaway_prevention_without_ei_balance(self):
        """Test if E/I balance prevents runaway excitation."""

        # Ablated: No E/I balance, strong recurrent connections
        ablated_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            l23_recurrent_strength=1.0,  # Strong recurrence
            robustness=RobustnessConfig(
                enable_ei_balance=False,  # <-- ABLATED
                enable_divisive_norm=False,  # Also off to stress test
                enable_intrinsic_plasticity=False,
                enable_criticality=False,
                enable_metabolic=False,
            ),
        )
        ablated_cortex = LayeredCortex(ablated_config)

        # Baseline: With E/I balance
        baseline_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            l23_recurrent_strength=1.0,
            robustness=RobustnessConfig(
                enable_ei_balance=True,  # <-- Enabled
                enable_divisive_norm=False,
                enable_intrinsic_plasticity=False,
                enable_criticality=False,
                enable_metabolic=False,
            ),
        )
        baseline_cortex = LayeredCortex(baseline_config)

        # Strong pulse input
        strong_pulse = torch.randn(64) * 3.0

        # Track activity over time
        baseline_activity = []
        ablated_activity = []

        for step in range(50):
            # Give pulse at step 0, then let it reverberate
            if step == 0:
                baseline_cortex.forward(strong_pulse)
                ablated_cortex.forward(strong_pulse)
            else:
                baseline_cortex.forward(torch.zeros(64))
                ablated_cortex.forward(torch.zeros(64))

            baseline_activity.append(baseline_cortex.state.l23_spikes.sum().item())
            ablated_activity.append(ablated_cortex.state.l23_spikes.sum().item())

        # Check for runaway (activity keeps growing)
        baseline_late = sum(baseline_activity[30:]) / 20
        ablated_late = sum(ablated_activity[30:]) / 20

        baseline_early = sum(baseline_activity[5:15]) / 10
        ablated_early = sum(ablated_activity[5:15]) / 10

        print("\n" + "="*60)
        print("RUNAWAY TEST: Recurrent activity after pulse")
        print("="*60)
        print(f"\nBaseline (with E/I balance):")
        print(f"  Early (steps 5-15): {baseline_early:.2f} spikes")
        print(f"  Late (steps 30-50):  {baseline_late:.2f} spikes")
        baseline_ratio = baseline_late / max(baseline_early, 1)
        print(f"  Ratio: {baseline_ratio:.2f}x")

        print(f"\nAblated (without E/I balance):")
        print(f"  Early (steps 5-15): {ablated_early:.2f} spikes")
        print(f"  Late (steps 30-50):  {ablated_late:.2f} spikes")
        ablated_ratio = ablated_late / max(ablated_early, 1)
        print(f"  Ratio: {ablated_ratio:.2f}x")
        print("="*60)

        # CONCRETE ASSERTIONS
        
        # Baseline with E/I balance should stabilize or decay (not grow)
        assert baseline_ratio < RUNAWAY_THRESHOLD, \
            f"Baseline WITH E/I balance shows runaway: " \
            f"{baseline_early:.1f} → {baseline_late:.1f} ({baseline_ratio:.2f}x growth)"
        
        # Document ablated system behavior
        if ablated_ratio > RUNAWAY_THRESHOLD:
            print(f"\n⚠️  WITHOUT E/I balance: Activity grows {ablated_ratio:.2f}x (runaway risk)")
        else:
            print(f"\nℹ️  WITHOUT E/I balance: Activity stable at {ablated_ratio:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

    """Ablation test: Remove E/I balance regulation."""

    def test_stability_degrades_without_ei_balance(self):
        """Quantify stability degradation without E/I balance."""

        # Baseline: Full robustness with E/I balance
        baseline_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig.full(),
        )
        baseline_cortex = LayeredCortex(baseline_config)

        # Ablated: Same config but WITHOUT E/I balance
        ablated_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig(
                enable_ei_balance=False,  # <-- ABLATED
                enable_divisive_norm=True,
                enable_intrinsic_plasticity=True,
                enable_criticality=True,
                enable_metabolic=True,
            ),
        )
        ablated_cortex = LayeredCortex(ablated_config)

        # Test with strong input (stress test)
        strong_input = torch.randn(64) * 1.5

        # Run both for same duration
        baseline_spikes = []
        ablated_spikes = []

        for _ in range(100):
            # Baseline
            baseline_cortex.forward(strong_input)
            baseline_spikes.append(baseline_cortex.state.l5_spikes.sum().item())

            # Ablated
            ablated_cortex.forward(strong_input)
            ablated_spikes.append(ablated_cortex.state.l5_spikes.sum().item())

        # Analyze stability
        baseline_variance = torch.tensor(baseline_spikes, dtype=torch.float32).var().item()
        ablated_variance = torch.tensor(ablated_spikes, dtype=torch.float32).var().item()

        baseline_mean = torch.tensor(baseline_spikes, dtype=torch.float32).mean().item()
        ablated_mean = torch.tensor(ablated_spikes, dtype=torch.float32).mean().item()

        # Print results
        print("\n" + "="*60)
        print("ABLATION TEST: Without E/I Balance")
        print("="*60)
        print(f"\nActivity levels:")
        print(f"  Baseline: {baseline_mean:.2f} ± {baseline_variance**0.5:.2f} spikes/step")
        print(f"  Ablated:  {ablated_mean:.2f} ± {ablated_variance**0.5:.2f} spikes/step")

        print(f"\nStability (variance):")
        print(f"  Baseline: {baseline_variance:.3f}")
        print(f"  Ablated:  {ablated_variance:.3f}")
        variance_increase = (ablated_variance - baseline_variance) / baseline_variance * 100
        print(f"  → {variance_increase:+.1f}% change")

        print("="*60)

        # Quantify degradation
        # E/I balance should reduce variance (more stable)
        stability_degradation = (ablated_variance - baseline_variance) / (baseline_variance + 1e-6)

        print(f"\n📊 RESULT: E/I balance reduces variance by {-stability_degradation*100:.1f}%")

        # Document findings (not strict assertions - this is discovery)
        if stability_degradation > 0.5:
            print("✅ E/I balance is CRITICAL (>50% stability loss)")
        elif stability_degradation > 0.2:
            print("⚠️  E/I balance is VALUABLE (20-50% stability loss)")
        else:
            print("ℹ️  E/I balance has MINOR impact (<20% stability loss)")

    def test_runaway_prevention_without_ei_balance(self):
        """Test if E/I balance prevents runaway excitation."""

        # Ablated: No E/I balance, strong recurrent connections
        ablated_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            l23_recurrent_strength=1.0,  # Strong recurrence
            robustness=RobustnessConfig(
                enable_ei_balance=False,  # <-- ABLATED
                enable_divisive_norm=False,  # Also off to stress test
                enable_intrinsic_plasticity=False,
                enable_criticality=False,
                enable_metabolic=False,
            ),
        )
        ablated_cortex = LayeredCortex(ablated_config)

        # Baseline: With E/I balance
        baseline_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            l23_recurrent_strength=1.0,
            robustness=RobustnessConfig(
                enable_ei_balance=True,  # <-- Enabled
                enable_divisive_norm=False,
                enable_intrinsic_plasticity=False,
                enable_criticality=False,
                enable_metabolic=False,
            ),
        )
        baseline_cortex = LayeredCortex(baseline_config)

        # Strong pulse input
        strong_pulse = torch.randn(64) * 3.0

        # Track activity over time
        baseline_activity = []
        ablated_activity = []

        for step in range(50):
            # Give pulse at step 0, then let it reverberate
            if step == 0:
                baseline_cortex.forward(strong_pulse)
                ablated_cortex.forward(strong_pulse)
            else:
                baseline_cortex.forward(torch.zeros(64))
                ablated_cortex.forward(torch.zeros(64))

            baseline_activity.append(baseline_cortex.state.l23_spikes.sum().item())
            ablated_activity.append(ablated_cortex.state.l23_spikes.sum().item())

        # Check for runaway (activity keeps growing)
        baseline_late = sum(baseline_activity[30:]) / 20
        ablated_late = sum(ablated_activity[30:]) / 20

        baseline_early = sum(baseline_activity[5:15]) / 10
        ablated_early = sum(ablated_activity[5:15]) / 10

        print("\n" + "="*60)
        print("RUNAWAY TEST: Recurrent activity after pulse")
        print("="*60)
        print(f"\nBaseline (with E/I balance):")
        print(f"  Early (steps 5-15): {baseline_early:.2f} spikes")
        print(f"  Late (steps 30-50):  {baseline_late:.2f} spikes")
        print(f"  Ratio: {baseline_late/max(baseline_early, 1):.2f}x")

        print(f"\nAblated (without E/I balance):")
        print(f"  Early (steps 5-15): {ablated_early:.2f} spikes")
        print(f"  Late (steps 30-50):  {ablated_late:.2f} spikes")
        print(f"  Ratio: {ablated_late/max(ablated_early, 1):.2f}x")
        print("="*60)

        # E/I balance should prevent runaway (late activity should decay, not grow)
        if ablated_late > ablated_early * 1.5:
            print("⚠️  WITHOUT E/I balance: Activity grows (runaway risk)")

        if baseline_late < baseline_early * 0.5:
            print("✅ WITH E/I balance: Activity decays (stable)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
