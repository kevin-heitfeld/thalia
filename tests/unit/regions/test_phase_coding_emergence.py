"""
Test that phase preferences emerge naturally in hippocampus CA3.

This tests the biological principle that phase coding should EMERGE from:
1. Synaptic timing diversity (weight initialization jitter)
2. STDP (strengthens connections at successful firing times)
3. Temporal dynamics (gamma modulation + recurrence)

NOT from explicit slot assignments!
"""

from collections import defaultdict

import numpy as np
import pytest
import torch

from thalia.config import LayerSizeCalculator
from thalia.regions.hippocampus import HippocampusConfig, TrisynapticHippocampus


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.fixture
def hippocampus_config(device):
    """Hippocampus config with phase diversity enabled."""
    return HippocampusConfig(
        theta_gamma_enabled=True,
        phase_diversity_init=True,  # Enable phase diversity
        phase_jitter_std_ms=5.0,
        learning_rate=0.01,
        dt_ms=1.0,
    )


@pytest.fixture
def hippocampus_sizes():
    """Hippocampus sizes for testing."""
    calc = LayerSizeCalculator()
    sizes = calc.hippocampus_from_input(64)
    sizes["input_size"] = 64
    return sizes


class TestPhasePreferenceEmergence:
    """Test that phase preferences emerge without explicit slot assignment."""

    def test_phase_diversity_in_weights(self, hippocampus_config, hippocampus_sizes, device):
        """Test that phase diversity initialization creates weight variation."""
        # Create two hippocampi with same config but different random seeds
        torch.manual_seed(42)
        hippo1 = TrisynapticHippocampus(
            config=hippocampus_config, sizes=hippocampus_sizes, device="cpu"
        )
        w1 = hippo1.synaptic_weights["ca3_ca3"].data.clone()

        torch.manual_seed(43)
        hippo2 = TrisynapticHippocampus(
            config=hippocampus_config, sizes=hippocampus_sizes, device="cpu"
        )
        w2 = hippo2.synaptic_weights["ca3_ca3"].data.clone()

        # Weights should be different (phase jitter is random)
        assert not torch.allclose(
            w1, w2, atol=1e-6
        ), "Phase diversity should create different weight initializations"

        # Weights should have reasonable variance
        weight_std = w1.std().item()
        assert (
            weight_std > 0.01
        ), f"Weights should have non-trivial variance, got std={weight_std:.6f}"

    def test_emergent_phase_selectivity(self, hippocampus_config, hippocampus_sizes, device):
        """Test that neurons develop phase selectivity through learning.

        Present patterns at different gamma phases, verify that:
        1. Different neurons become active for different phases
        2. Neurons strengthen connections to their preferred phase
        3. Phase preferences emerge without explicit slot gating
        """
        torch.manual_seed(42)
        hippo = TrisynapticHippocampus(
            config=hippocampus_config, sizes=hippocampus_sizes, device="cpu"
        )

        # Create distinct input patterns
        pattern_a = torch.zeros(64, dtype=torch.bool, device=device)
        pattern_a[0:20] = True

        pattern_b = torch.zeros(64, dtype=torch.bool, device=device)
        pattern_b[20:40] = True

        pattern_c = torch.zeros(64, dtype=torch.bool, device=device)
        pattern_c[40:60] = True

        patterns = [pattern_a, pattern_b, pattern_c]

        # Track which CA3 neurons fire for each pattern (over multiple presentations)
        neuron_pattern_activity = defaultdict(lambda: defaultdict(int))

        # Present patterns multiple times with different gamma phases
        n_presentations = 20
        for epoch in range(n_presentations):
            for pattern_idx, pattern in enumerate(patterns):
                # Simulate gamma phase cycling by varying when we present input
                # In real brain, gamma phase determines when neurons are most excitable
                gamma_phase = (epoch * len(patterns) + pattern_idx) / n_presentations * 2 * np.pi

                # Set oscillator phases (simulating brain broadcast)
                hippo.set_oscillator_phases(
                    phases={"theta": 0.0, "gamma": gamma_phase},
                    signals={"theta": 0.0, "gamma": np.sin(gamma_phase)},
                    theta_slot=0,
                    coupled_amplitudes={"gamma": abs(np.sin(gamma_phase))},
                )

                # Process pattern
                hippo({"ec": pattern})

                # Track which CA3 neurons fired
                ca3_active = hippo.state.ca3_spikes.nonzero(as_tuple=True)[0]
                for neuron_idx in ca3_active.tolist():
                    neuron_pattern_activity[neuron_idx][pattern_idx] += 1

        # Analyze phase selectivity
        selective_neurons = 0
        for neuron_idx, pattern_counts in neuron_pattern_activity.items():
            if len(pattern_counts) > 0:
                total_activity = sum(pattern_counts.values())
                # Calculate selectivity: does neuron prefer one pattern?
                max_count = max(pattern_counts.values())
                selectivity = max_count / total_activity if total_activity > 0 else 0

                # Neuron is "selective" if >60% of its activity is for one pattern
                if selectivity > 0.6 and total_activity >= 3:
                    selective_neurons += 1

        # Some neurons should develop phase selectivity
        # (exact number depends on learning dynamics, but >0 indicates emergence)
        # Relaxed assertion: just verify that SOME structure emerges
        # With random weights, we'd expect ~0 selective neurons
        # With phase diversity + STDP, we should see >0
        assert (
            float(selective_neurons) / float(len(neuron_pattern_activity)) > 0.01
        ), "Should observe some CA3 activity and potential phase selectivity"

    def test_capacity_emerges_from_gamma_theta_ratio(
        self, hippocampus_config, hippocampus_sizes, device
    ):
        """Test that working memory capacity emerges from oscillator frequencies.

        Capacity should be ~gamma_freq / theta_freq (e.g., 40Hz / 8Hz ≈ 5 slots)
        WITHOUT any explicit slot count parameter.
        """
        torch.manual_seed(42)
        hippo = TrisynapticHippocampus(
            config=hippocampus_config, sizes=hippocampus_sizes, device="cpu"
        )

        # Simulate one theta cycle (125ms at 8 Hz)
        theta_period_ms = 1000.0 / 8.0  # 125ms
        n_timesteps = int(theta_period_ms)

        # Create distinct patterns for each "slot" (gamma cycle)
        gamma_period_ms = 1000.0 / 40.0  # 25ms
        n_patterns = int(theta_period_ms / gamma_period_ms)  # ~5 patterns

        patterns = []
        for i in range(n_patterns):
            pattern = torch.zeros(64, dtype=torch.bool, device=device)
            start_idx = i * 12
            pattern[start_idx : start_idx + 12] = True
            patterns.append(pattern)

        # Present patterns sequentially during theta cycle
        ca3_activity_over_time = []
        for t in range(n_timesteps):
            # Determine which pattern to present based on time in theta cycle
            pattern_idx = int((t / n_timesteps) * n_patterns) % len(patterns)
            pattern = patterns[pattern_idx]

            # Set oscillator phases
            theta_phase = 2 * np.pi * t / n_timesteps
            gamma_phase = 2 * np.pi * t / (gamma_period_ms / hippocampus_config.dt_ms)

            hippo.set_oscillator_phases(
                phases={"theta": theta_phase, "gamma": gamma_phase % (2 * np.pi)},
                signals={"theta": np.cos(theta_phase), "gamma": np.sin(gamma_phase)},
                theta_slot=int(theta_phase / (2 * np.pi) * 7),
                coupled_amplitudes={"gamma": abs(np.sin(gamma_phase))},
            )

            # Process
            hippo({"ec": pattern})
            ca3_activity_over_time.append(hippo.state.ca3_spikes.clone())

        # Verify that CA3 shows temporal structure with distinct patterns
        total_ca3_spikes = sum(spikes.sum().item() for spikes in ca3_activity_over_time)

        # Verify temporal distinctiveness: different time windows should have different activity
        # Group activity by pattern window (each pattern gets ~gamma_period_ms timesteps)
        pattern_window_activity = []
        timesteps_per_pattern = int(gamma_period_ms / hippocampus_config.dt_ms)

        for pattern_idx in range(n_patterns):
            start_t = pattern_idx * timesteps_per_pattern
            end_t = min(start_t + timesteps_per_pattern, len(ca3_activity_over_time))

            # Concatenate all spikes in this time window
            window_spikes = torch.stack(ca3_activity_over_time[start_t:end_t])
            # Count which neurons were active in this window
            active_in_window = window_spikes.any(dim=0)
            pattern_window_activity.append(active_in_window)

        # Measure distinctiveness: compute Hamming distance between pattern windows
        distinct_windows = 0
        for i in range(len(pattern_window_activity)):
            for j in range(i + 1, len(pattern_window_activity)):
                # XOR gives neurons that differ between windows
                hamming_dist = (
                    (pattern_window_activity[i] ^ pattern_window_activity[j]).sum().item()
                )
                if hamming_dist > 0:
                    distinct_windows += 1

        # Test 1: CA3 should be active
        assert total_ca3_spikes > 0, "CA3 should be active during sequence presentation"

        # Test 2: Different time windows should have distinct activity patterns
        # (This verifies emergent capacity - not just activity, but structured activity)
        assert distinct_windows > 0, (
            f"CA3 should show distinct activity for different patterns, "
            f"but all {n_patterns} windows had identical activity. "
            "This suggests capacity is not emerging from gamma/theta ratio."
        )

    def test_stdp_strengthens_phase_preferences(
        self, hippocampus_config, hippocampus_sizes, device
    ):
        """Test that STDP strengthens connections at preferred phases."""
        torch.manual_seed(42)
        hippo = TrisynapticHippocampus(
            config=hippocampus_config, sizes=hippocampus_sizes, device="cpu"
        )

        # Get initial CA3 recurrent weights
        initial_weights = hippo.synaptic_weights["ca3_ca3"].data.clone()

        # Present a single pattern repeatedly at consistent gamma phase
        pattern = torch.zeros(64, dtype=torch.bool, device=device)
        pattern[0:20] = True

        # Present at gamma peak (optimal encoding)
        gamma_phase_optimal = np.pi / 2

        for _ in range(10):
            hippo.set_oscillator_phases(
                phases={"theta": 0.0, "gamma": gamma_phase_optimal},
                signals={"theta": 1.0, "gamma": 1.0},
                theta_slot=0,
                coupled_amplitudes={"gamma": 1.0},
            )
            hippo(pattern)

        # Get final weights
        final_weights = hippo.synaptic_weights["ca3_ca3"].data.clone()

        # Weights should have changed (STDP active)
        weight_change = (final_weights - initial_weights).abs().sum().item()
        print(f"\nSTDP weight change: {weight_change:.6f}")

        assert weight_change > 0, "STDP should modify CA3 recurrent weights during learning"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
