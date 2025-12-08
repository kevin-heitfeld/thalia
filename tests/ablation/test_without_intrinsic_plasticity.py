"""
Ablation test: Intrinsic Plasticity

Tests what happens when intrinsic plasticity is disabled.

Expected impact:
- Less firing rate adaptation
- Poor adjustment to input statistics
- Reduced excitability control
"""

import pytest
import torch

from thalia.regions import LayeredCortex
from thalia.regions.cortex import LayeredCortexConfig
from thalia.config import RobustnessConfig


class TestWithoutIntrinsicPlasticity:
    """Ablation test: Remove intrinsic plasticity."""

    def test_firing_rate_adaptation_without_ip(self):
        """Test that firing rate adaptation fails without intrinsic plasticity."""

        # Baseline: With intrinsic plasticity
        baseline_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig(
                enable_ei_balance=False,  # Isolate IP effect
                enable_divisive_norm=False,
                enable_intrinsic_plasticity=True,
                enable_criticality=False,
                enable_metabolic=False,
            ),
        )
        baseline_cortex = LayeredCortex(baseline_config)

        # Ablated: WITHOUT intrinsic plasticity
        ablated_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig(
                enable_ei_balance=False,
                enable_divisive_norm=False,
                enable_intrinsic_plasticity=False,  # <-- ABLATED
                enable_criticality=False,
                enable_metabolic=False,
            ),
        )
        ablated_cortex = LayeredCortex(ablated_config)

        # Run with consistent input for adaptation
        input_pattern = torch.randn(64) * 1.0

        # Track firing rates over time
        baseline_rates = []
        ablated_rates = []

        for step in range(100):
            baseline_cortex.forward(input_pattern)
            ablated_cortex.forward(input_pattern)

            # Track L2/3 firing rates (main adaptive layer)
            baseline_rate = baseline_cortex.state.l23_spikes.float().mean().item()
            ablated_rate = ablated_cortex.state.l23_spikes.float().mean().item()

            baseline_rates.append(baseline_rate)
            ablated_rates.append(ablated_rate)

        # Compare early vs late firing rates
        early_baseline = sum(baseline_rates[:20]) / 20
        late_baseline = sum(baseline_rates[-20:]) / 20

        early_ablated = sum(ablated_rates[:20]) / 20
        late_ablated = sum(ablated_rates[-20:]) / 20

        # Compute adaptation
        baseline_adaptation = abs(late_baseline - early_baseline)
        ablated_adaptation = abs(late_ablated - early_ablated)

        # Compute convergence to target (10%)
        target_rate = 0.1
        baseline_error = abs(late_baseline - target_rate)
        ablated_error = abs(late_ablated - target_rate)

        print("\n" + "="*60)
        print("ABLATION TEST: Without Intrinsic Plasticity")
        print("="*60)
        print(f"\nFiring Rate Adaptation Test (100 steps):")
        print(f"\nBaseline (with IP):")
        print(f"  Early rate (steps 0-20):   {early_baseline:.3f}")
        print(f"  Late rate (steps 80-100):  {late_baseline:.3f}")
        print(f"  Adaptation magnitude:      {baseline_adaptation:.3f}")
        print(f"  Distance from target 10%:  {baseline_error:.3f}")

        print(f"\nAblated (without IP):")
        print(f"  Early rate (steps 0-20):   {early_ablated:.3f}")
        print(f"  Late rate (steps 80-100):  {late_ablated:.3f}")
        print(f"  Adaptation magnitude:      {ablated_adaptation:.3f}")
        print(f"  Distance from target 10%:  {ablated_error:.3f}")

        # Compute impact
        if ablated_adaptation > 0:
            adaptation_loss = (baseline_adaptation - ablated_adaptation) / ablated_adaptation
        else:
            adaptation_loss = 1.0 if baseline_adaptation > 0 else 0.0

        target_convergence = (ablated_error - baseline_error) / max(ablated_error, 0.01)

        print(f"\n📊 RESULT:")
        print(f"  Adaptation loss: {adaptation_loss*100:.1f}%")
        print(f"  Target convergence improvement: {target_convergence*100:.1f}%")
        print("="*60)

        # Document findings
        if adaptation_loss > 0.5:
            print("✅ IP is CRITICAL for firing rate adaptation (>50% loss)")
        elif adaptation_loss > 0.2:
            print("⚠️  IP is VALUABLE for firing rate adaptation (20-50% loss)")
        else:
            print("ℹ️  IP has MINOR impact on current setup (<20% loss)")

    def test_excitability_control_without_ip(self):
        """Test that excitability control degrades without IP."""

        # Create both configurations with different input statistics
        with_ip = LayeredCortex(LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig(
                enable_ei_balance=False,
                enable_divisive_norm=False,
                enable_intrinsic_plasticity=True,
                enable_criticality=False,
                enable_metabolic=False,
            ),
        ))

        without_ip = LayeredCortex(LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig(
                enable_ei_balance=False,
                enable_divisive_norm=False,
                enable_intrinsic_plasticity=False,  # <-- ABLATED
                enable_criticality=False,
                enable_metabolic=False,
            ),
        ))

        # Test response to input distribution shift
        # Phase 1: Weak inputs (adapt to low input)
        weak_input = torch.randn(64) * 0.5

        with_ip_weak_rates = []
        without_ip_weak_rates = []

        for _ in range(50):
            with_ip.forward(weak_input)
            without_ip.forward(weak_input)

            with_ip_weak_rates.append(with_ip.state.l23_spikes.float().mean().item())
            without_ip_weak_rates.append(without_ip.state.l23_spikes.float().mean().item())

        # Phase 2: Strong inputs (test excitability adjustment)
        strong_input = weak_input * 3.0

        with_ip_strong_rates = []
        without_ip_strong_rates = []

        for _ in range(50):
            with_ip.forward(strong_input)
            without_ip.forward(strong_input)

            with_ip_strong_rates.append(with_ip.state.l23_spikes.float().mean().item())
            without_ip_strong_rates.append(without_ip.state.l23_spikes.float().mean().item())

        # Compute rate changes after input shift
        with_ip_weak = sum(with_ip_weak_rates[-10:]) / 10
        with_ip_strong = sum(with_ip_strong_rates[-10:]) / 10
        with_ip_change = with_ip_strong / max(with_ip_weak, 0.01)

        without_ip_weak = sum(without_ip_weak_rates[-10:]) / 10
        without_ip_strong = sum(without_ip_strong_rates[-10:]) / 10
        without_ip_change = without_ip_strong / max(without_ip_weak, 0.01)

        print("\n" + "="*60)
        print("EXCITABILITY CONTROL TEST")
        print("="*60)
        print(f"\nResponse to 3x input increase:")
        print(f"\nWith IP:")
        print(f"  Weak input phase:   {with_ip_weak:.3f}")
        print(f"  Strong input phase: {with_ip_strong:.3f}")
        print(f"  Rate change:        {with_ip_change:.2f}x")

        print(f"\nWithout IP:")
        print(f"  Weak input phase:   {without_ip_weak:.3f}")
        print(f"  Strong input phase: {without_ip_strong:.3f}")
        print(f"  Rate change:        {without_ip_change:.2f}x")

        # IP should provide more stable rates despite input shift
        with_ip_stability = 3.0 - with_ip_change  # Lower change = more stable
        without_ip_stability = 3.0 - without_ip_change

        stability_loss = (with_ip_stability - without_ip_stability) / max(with_ip_stability, 0.1)

        print(f"\n📊 RESULT:")
        print(f"  With IP provides {with_ip_stability:.2f} stability units")
        print(f"  Without IP provides {without_ip_stability:.2f} stability units")
        print(f"  → Stability loss: {stability_loss*100:.1f}%")
        print("="*60)

        if stability_loss > 0.5:
            print("✅ IP SIGNIFICANTLY stabilizes excitability (>50%)")
        elif stability_loss > 0.2:
            print("⚠️  IP MODERATELY stabilizes excitability (20-50%)")
        else:
            print("ℹ️  IP has MINOR effect on excitability (<20%)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
