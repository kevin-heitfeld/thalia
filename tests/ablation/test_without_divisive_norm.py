"""
Ablation test: Divisive Normalization

Tests what happens when divisive normalization is disabled.

Expected impact:
- Reduced contrast invariance
- Less gain control
- Activity may saturate with strong inputs
"""

import pytest
import torch

from thalia.regions import LayeredCortex
from thalia.regions.cortex import LayeredCortexConfig
from thalia.config import RobustnessConfig


class TestWithoutDivisiveNorm:
    """Ablation test: Remove divisive normalization."""

    def test_gain_control_lost_without_divisive_norm(self):
        """Test that gain control degrades without divisive normalization."""

        # Baseline: With divisive normalization
        baseline_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig(
                enable_ei_balance=False,  # Isolate divisive norm effect
                enable_divisive_norm=True,
                enable_intrinsic_plasticity=False,
                enable_criticality=False,
                enable_metabolic=False,
            ),
        )
        baseline_cortex = LayeredCortex(baseline_config)

        # Ablated: WITHOUT divisive normalization
        ablated_config = LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig(
                enable_ei_balance=False,
                enable_divisive_norm=False,  # <-- ABLATED
                enable_intrinsic_plasticity=False,
                enable_criticality=False,
                enable_metabolic=False,
            ),
        )
        ablated_cortex = LayeredCortex(ablated_config)

        # Test with different input magnitudes
        weak_input = torch.randn(1, 64) * 0.5
        strong_input = weak_input * 3.0  # 3x stronger, same pattern

        # Baseline responses
        baseline_cortex.reset_state()
        baseline_cortex.forward(weak_input)
        baseline_weak = baseline_cortex.state.l5_spikes.sum().item()

        baseline_cortex.reset_state()
        baseline_cortex.forward(strong_input)
        baseline_strong = baseline_cortex.state.l5_spikes.sum().item()

        # Ablated responses
        ablated_cortex.reset_state()
        ablated_cortex.forward(weak_input)
        ablated_weak = ablated_cortex.state.l5_spikes.sum().item()

        ablated_cortex.reset_state()
        ablated_cortex.forward(strong_input)
        ablated_strong = ablated_cortex.state.l5_spikes.sum().item()

        # Compute scaling ratios
        baseline_ratio = baseline_strong / max(baseline_weak, 1.0)
        ablated_ratio = ablated_strong / max(ablated_weak, 1.0)

        print("\n" + "="*60)
        print("ABLATION TEST: Without Divisive Normalization")
        print("="*60)
        print(f"\nGain Control Test (3x input scaling):")
        print(f"\nBaseline (with divisive norm):")
        print(f"  Weak input:   {baseline_weak:.1f} spikes")
        print(f"  Strong input: {baseline_strong:.1f} spikes")
        print(f"  Output ratio: {baseline_ratio:.2f}x (vs 3x input)")

        print(f"\nAblated (without divisive norm):")
        print(f"  Weak input:   {ablated_weak:.1f} spikes")
        print(f"  Strong input: {ablated_strong:.1f} spikes")
        print(f"  Output ratio: {ablated_ratio:.2f}x (vs 3x input)")

        # Gain control metric: how sublinear is the response?
        baseline_gain_control = 3.0 - baseline_ratio  # Closer to 3 = less gain control
        ablated_gain_control = 3.0 - ablated_ratio

        gain_control_loss = (baseline_gain_control - ablated_gain_control) / max(baseline_gain_control, 0.1)

        print(f"\n📊 RESULT:")
        print(f"  Baseline provides {baseline_gain_control:.2f} units of gain control")
        print(f"  Ablated provides {ablated_gain_control:.2f} units of gain control")
        print(f"  → Loss: {gain_control_loss*100:.1f}%")
        print("="*60)

        # Document findings
        if gain_control_loss > 0.5:
            print("✅ Divisive norm is CRITICAL for gain control (>50% loss)")
        elif gain_control_loss > 0.2:
            print("⚠️  Divisive norm is VALUABLE for gain control (20-50% loss)")
        else:
            print("ℹ️  Divisive norm has MINOR impact (<20% loss)")

    def test_contrast_invariance_without_divisive_norm(self):
        """Test that contrast invariance degrades without divisive norm."""

        # Create both configurations
        with_norm = LayeredCortex(LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig(
                enable_ei_balance=False,
                enable_divisive_norm=True,
                enable_intrinsic_plasticity=False,
                enable_criticality=False,
                enable_metabolic=False,
            ),
        ))

        without_norm = LayeredCortex(LayeredCortexConfig(
            n_input=64,
            n_output=32,
            robustness=RobustnessConfig(
                enable_ei_balance=False,
                enable_divisive_norm=False,  # <-- ABLATED
                enable_intrinsic_plasticity=False,
                enable_criticality=False,
                enable_metabolic=False,
            ),
        ))

        # Test with multiple contrast levels
        contrasts = [0.2, 0.5, 1.0, 1.5, 2.0]

        with_norm_responses = []
        without_norm_responses = []

        for contrast in contrasts:
            input_pattern = torch.randn(1, 64) * contrast

            # With norm
            with_norm.reset_state()
            with_norm.forward(input_pattern)
            with_norm_responses.append(with_norm.state.l5_spikes.sum().item())

            # Without norm
            without_norm.reset_state()
            without_norm.forward(input_pattern)
            without_norm_responses.append(without_norm.state.l5_spikes.sum().item())

        # Compute response variance across contrasts
        with_norm_variance = torch.tensor(with_norm_responses).var().item()
        without_norm_variance = torch.tensor(without_norm_responses).var().item()

        print("\n" + "="*60)
        print("CONTRAST INVARIANCE TEST")
        print("="*60)
        print(f"\nResponse across contrast levels:")
        for i, contrast in enumerate(contrasts):
            print(f"  Contrast {contrast:.1f}x:")
            print(f"    With norm:    {with_norm_responses[i]:.1f} spikes")
            print(f"    Without norm: {without_norm_responses[i]:.1f} spikes")

        print(f"\nResponse variance (lower = more invariant):")
        print(f"  With norm:    {with_norm_variance:.2f}")
        print(f"  Without norm: {without_norm_variance:.2f}")

        variance_increase = (without_norm_variance - with_norm_variance) / max(with_norm_variance, 1.0)
        print(f"  → {variance_increase*100:+.1f}% variance increase without norm")
        print("="*60)

        # Divisive norm should reduce response variance (more invariant)
        if variance_increase > 0.5:
            print("✅ Divisive norm SIGNIFICANTLY improves contrast invariance (>50%)")
        elif variance_increase > 0.2:
            print("⚠️  Divisive norm MODERATELY improves contrast invariance (20-50%)")
        else:
            print("ℹ️  Divisive norm has MINOR effect on contrast invariance (<20%)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
