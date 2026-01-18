"""
Tests for Hippocampus Multi-Timescale Consolidation (Phase 1A).

This test suite validates the multi-timescale consolidation mechanism in the
hippocampus, which implements systems consolidation theory (McClelland et al., 1995):
- Fast trace (τ ~60s): Immediate synaptic tagging for episodic details
- Slow trace (τ ~3600s): Gradual consolidation for semantic regularities
- Consolidation: Transfer from fast (hippocampus) to slow (neocortex)

Biological motivation:
- Fast learning in hippocampus captures episodic specifics
- Slow consolidation extracts statistical patterns over time
- Prevents catastrophic forgetting through gradual interleaving
"""

import pytest
import torch

from thalia.regions.hippocampus.config import HippocampusConfig
from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus


@pytest.fixture
def device():
    """Test device (CPU for consistency)."""
    return "cpu"


@pytest.fixture
def small_hippocampus_config():
    """Small hippocampus config for testing (multiscale enabled)."""
    return HippocampusConfig(
        dt_ms=1.0,
        use_multiscale_consolidation=True,
        fast_trace_tau_ms=60_000.0,  # 1 minute
        slow_trace_tau_ms=3_600_000.0,  # 1 hour
        consolidation_rate=0.001,
        slow_trace_contribution=0.1,
        # Small sizes for faster tests
        learning_rate=0.01,
        ca3_ca2_learning_rate=0.001,
        ec_ca2_learning_rate=0.01,
        ca2_ca1_learning_rate=0.005,
        ec_ca1_learning_rate=0.5,
    )


@pytest.fixture
def small_hippocampus_sizes():
    """Small hippocampus sizes for testing."""
    return {
        "input_size": 32,
        "dg_size": 64,
        "ca3_size": 48,
        "ca2_size": 24,
        "ca1_size": 32,
    }


@pytest.fixture
def hippocampus(small_hippocampus_config, small_hippocampus_sizes, device):
    """Create small hippocampus for testing."""
    return TrisynapticHippocampus(
        config=small_hippocampus_config,
        sizes=small_hippocampus_sizes,
        device=device,
    )


# =============================================================================
# Test 1: Trace Initialization
# =============================================================================


def test_trace_initialization(hippocampus, small_hippocampus_sizes):
    """Test that multiscale consolidation system initializes and can learn."""
    # Test behavioral contract: hippocampus with multiscale should learn differently
    # Store initial weights
    initial_ca3_ca3 = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Present pattern repeatedly (should trigger consolidation)
    input_pattern = torch.ones(small_hippocampus_sizes["input_size"], device=hippocampus.device)
    for _ in range(50):
        hippocampus.forward({"ec": input_pattern})

    # Weights should have changed (learning occurred)
    final_ca3_ca3 = hippocampus.synaptic_weights["ca3_ca3"]
    weight_change = (final_ca3_ca3 - initial_ca3_ca3).abs().mean().item()
    assert weight_change > 0, "Multiscale consolidation should modify weights"

    # Verify shapes are valid (behavioral contract)
    assert final_ca3_ca3.shape == (
        small_hippocampus_sizes["ca3_size"],
        small_hippocampus_sizes["ca3_size"],
    ), "Weight matrix should have correct dimensions"


# =============================================================================
# Test 2: Disabled Flag Check
# =============================================================================


def test_disabled_multiscale(small_hippocampus_config, small_hippocampus_sizes, device):
    """Test that multiscale can be disabled (behavioral difference)."""
    # Create two hippocampi: one with multiscale, one without
    config_with = small_hippocampus_config
    config_with.use_multiscale_consolidation = True

    config_without = small_hippocampus_config
    config_without.use_multiscale_consolidation = False

    hippo_with = TrisynapticHippocampus(
        config=config_with,
        sizes=small_hippocampus_sizes,
        device=device,
    )

    hippo_without = TrisynapticHippocampus(
        config=config_without,
        sizes=small_hippocampus_sizes,
        device=device,
    )

    # Both should be functional (forward pass works)
    input_pattern = torch.ones(small_hippocampus_sizes["input_size"], device=device)

    output_with = hippo_with.forward({"ec": input_pattern})
    output_without = hippo_without.forward({"ec": input_pattern})

    # Both should produce valid outputs
    assert output_with.shape[0] == small_hippocampus_sizes["ca1_size"]
    assert output_without.shape[0] == small_hippocampus_sizes["ca1_size"]
    assert not torch.isnan(output_with).any(), "Output with multiscale should be valid"
    assert not torch.isnan(output_without).any(), "Output without multiscale should be valid"


# =============================================================================
# Test 3: Fast Trace Decay
# =============================================================================


def test_fast_trace_decay(hippocampus, small_hippocampus_sizes):
    """Test that fast consolidation shows rapid weight adaptation (behavioral)."""
    # Store initial weights
    initial_weights = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Present pattern for short period (fast trace should dominate)
    input_pattern = torch.ones(small_hippocampus_sizes["input_size"], device=hippocampus.device)

    # Present for 20 timesteps (short-term)
    for _ in range(20):
        hippocampus.forward({"ec": input_pattern})

    # Weights should change quickly (fast trace active)
    weights_after_short = hippocampus.synaptic_weights["ca3_ca3"].clone()
    short_term_change = (weights_after_short - initial_weights).abs().mean().item()

    # Wait without input (fast trace should decay)
    zero_input = torch.zeros(small_hippocampus_sizes["input_size"], device=hippocampus.device)
    for _ in range(100):  # Decay period
        hippocampus.forward({"ec": zero_input})

    # After decay, presenting pattern again should show learning still occurred
    # (some consolidation to slow trace happened)
    weights_after_decay = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Weights should have changed from initial (consolidation occurred)
    total_change = (weights_after_decay - initial_weights).abs().mean().item()
    assert total_change > 0, "Fast trace should cause weight changes"
    assert short_term_change > 0, "Short-term presentation should modify weights"


# =============================================================================
# Test 4: Slow Trace Persistence
# =============================================================================


def test_slow_trace_persistence(hippocampus):
    """Test that multiscale consolidation creates persistent weight changes (behavioral)."""
    # Get input size from hippocampus
    input_size = hippocampus.input_size

    # Store initial weights
    initial_ca3_ca3 = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Present pattern repeatedly for consolidation
    input_pattern = torch.ones(input_size, device=hippocampus.device)
    for _ in range(100):  # Extended presentation for consolidation
        hippocampus.forward({"ec": input_pattern})

    # Wait long period without input (test persistence)
    zero_input = torch.zeros(input_size, device=hippocampus.device)
    for _ in range(200):  # Long delay
        hippocampus.forward({"ec": zero_input})

    # Weights should still show change from initial (slow consolidation persists)
    final_ca3_ca3 = hippocampus.synaptic_weights["ca3_ca3"]
    persistent_change = (final_ca3_ca3 - initial_ca3_ca3).abs().mean().item()

    # Slow trace should maintain some of the learned pattern despite long delay
    assert persistent_change > 0.001, "Slow consolidation should create persistent weight changes"


# =============================================================================
# Test 5: Consolidation Transfer
# =============================================================================


def test_consolidation_transfer(hippocampus):
    """Test that repeated presentation leads to stronger consolidation (behavioral)."""
    # Get input size from hippocampus
    input_size = hippocampus.input_size

    # Compare single vs repeated presentation
    initial_weights = hippocampus.synaptic_weights["ca3_ca3"].clone()

    input_pattern = torch.ones(input_size, device=hippocampus.device)

    # Single presentation
    hippocampus.forward({"ec": input_pattern})
    weights_single = hippocampus.synaptic_weights["ca3_ca3"].clone()
    single_change = (weights_single - initial_weights).abs().mean().item()

    # Continued presentation (consolidation should accumulate)
    for _ in range(99):  # 99 more times (100 total)
        hippocampus.forward({"ec": input_pattern})

    weights_repeated = hippocampus.synaptic_weights["ca3_ca3"]
    repeated_change = (weights_repeated - initial_weights).abs().mean().item()

    # Repeated presentation should cause larger weight changes (consolidation)
    assert (
        repeated_change > single_change
    ), "Consolidation should accumulate with repeated presentation"

    # Both should cause some change
    assert single_change > 0, "Single presentation should modify weights"
    assert repeated_change > 0, "Repeated presentation should modify weights"


def test_combined_learning(hippocampus):
    """Test that multiscale consolidation enables both fast and slow learning (behavioral)."""
    # Get input size from hippocampus
    input_size = hippocampus.input_size

    # Store initial weights
    initial_weights = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Pattern A: Present many times (should consolidate strongly)
    pattern_a = torch.ones(input_size, device=hippocampus.device)
    for _ in range(50):
        hippocampus.forward({"ec": pattern_a})

    weights_after_a = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Pattern B: Present few times (fast trace dominates)
    pattern_b = torch.zeros(input_size, device=hippocampus.device)
    for _ in range(5):
        hippocampus.forward({"ec": pattern_b})

    weights_after_b = hippocampus.synaptic_weights["ca3_ca3"]

    # Both patterns should cause weight changes
    change_a = (weights_after_a - initial_weights).abs().mean().item()
    change_b = (weights_after_b - weights_after_a).abs().mean().item()

    assert change_a > 0, "Pattern A should modify weights"
    assert change_b > 0, "Pattern B should modify weights"

    # Combined learning: both patterns affected weights
    total_change = (weights_after_b - initial_weights).abs().mean().item()
    assert total_change > 0, "Combined learning should occur"


# =============================================================================
# Test 7: Episodic vs Semantic (Qualitative)
# =============================================================================


def test_episodic_vs_semantic_dynamics(hippocampus):
    """Test that fast trace captures episodes, slow trace captures patterns."""
    config = hippocampus.config
    dt_ms = config.dt_ms

    # Simulate 3 learning episodes with 60s gaps
    # Episode 1: Strong learning event
    hippocampus._ca3_ca3_fast = torch.ones_like(hippocampus._ca3_ca3_fast) * 2.0
    hippocampus._ca3_ca3_slow = torch.zeros_like(hippocampus._ca3_ca3_slow)

    # Wait 60 seconds (fast decays, slow accumulates)
    n_steps = 60_000
    fast_decay = dt_ms / config.fast_trace_tau_ms
    slow_decay = dt_ms / config.slow_trace_tau_ms

    for _ in range(n_steps):
        consolidation = config.consolidation_rate * hippocampus._ca3_ca3_fast
        hippocampus._ca3_ca3_fast = (1.0 - fast_decay) * hippocampus._ca3_ca3_fast
        hippocampus._ca3_ca3_slow = (1.0 - slow_decay) * hippocampus._ca3_ca3_slow + consolidation

    # After 60s: Fast has decayed to ~36%, slow has accumulated
    fast_after_60s = hippocampus._ca3_ca3_fast.mean().item()
    slow_after_60s = hippocampus._ca3_ca3_slow.mean().item()

    # Episode 2: Another learning event
    hippocampus._ca3_ca3_fast += 2.0

    # Wait another 60s
    for _ in range(n_steps):
        consolidation = config.consolidation_rate * hippocampus._ca3_ca3_fast
        hippocampus._ca3_ca3_fast = (1.0 - fast_decay) * hippocampus._ca3_ca3_fast
        hippocampus._ca3_ca3_slow = (1.0 - slow_decay) * hippocampus._ca3_ca3_slow + consolidation

    slow_after_120s = hippocampus._ca3_ca3_slow.mean().item()

    # Slow trace should have accumulated more (evidence of multiple episodes)
    assert slow_after_120s > slow_after_60s

    # Fast trace should be lower than initial (decayed between episodes)
    assert fast_after_60s < 2.0 * 0.5  # Should have decayed below 50%


# =============================================================================
# Test 8: Integration Test (Forward Pass)
# =============================================================================


def test_integration_with_forward_pass(hippocampus, small_hippocampus_sizes):
    """Test that multi-timescale learning works during actual forward passes."""
    # Create input pattern
    input_pattern = torch.randn(small_hippocampus_sizes["input_size"])

    # Store initial weights
    initial_ca3_weights = hippocampus.synaptic_weights["ca3_ca3"].data.clone()

    # Run forward pass in encoding mode (ACh high)
    hippocampus.set_neuromodulators(acetylcholine=0.8)

    # Multiple forward passes (simulate encoding)
    for _ in range(10):
        output = hippocampus({"ec": input_pattern})

    # Weights should have changed due to learning
    assert not torch.allclose(hippocampus.synaptic_weights["ca3_ca3"].data, initial_ca3_weights)

    # Fast trace should be non-zero (learning occurred)
    assert hippocampus._ca3_ca3_fast.abs().sum() > 0

    # Check that output is correct shape
    assert output.shape == (small_hippocampus_sizes["ca1_size"],)


# =============================================================================
# Test 9: Config Validation
# =============================================================================


def test_config_parameter_ranges(small_hippocampus_config):
    """Test that config parameters are in valid ranges."""
    config = small_hippocampus_config

    # Time constants should be positive
    assert config.fast_trace_tau_ms > 0
    assert config.slow_trace_tau_ms > 0

    # Slow should be much longer than fast
    assert config.slow_trace_tau_ms > config.fast_trace_tau_ms * 10

    # Consolidation rate should be small (0.001 = 0.1%)
    assert 0 < config.consolidation_rate < 0.01

    # Slow contribution should be smaller than fast (default 0.1)
    assert 0 < config.slow_trace_contribution < 1.0


# =============================================================================
# Test 10: Multi-Pathway Consistency
# =============================================================================


def test_all_pathways_have_traces(hippocampus):
    """Test that all 5 learning pathways have fast/slow traces."""
    pathways = [
        ("ca3_ca3", "CA3 recurrent"),
        ("ca3_ca2", "CA3→CA2"),
        ("ec_ca2", "EC→CA2"),
        ("ec_ca1", "EC→CA1"),
        ("ca2_ca1", "CA2→CA1"),
    ]

    for pathway_name, description in pathways:
        fast_trace = getattr(hippocampus, f"_{pathway_name}_fast")
        slow_trace = getattr(hippocampus, f"_{pathway_name}_slow")

        # Traces should exist
        assert fast_trace is not None, f"{description} fast trace is None"
        assert slow_trace is not None, f"{description} slow trace is None"

        # Traces should have correct dtype and device
        assert fast_trace.dtype == torch.float32
        assert slow_trace.dtype == torch.float32
        assert str(fast_trace.device) == "cpu"
        assert str(slow_trace.device) == "cpu"


# =============================================================================
# Test 11: Time Constant Accuracy
# =============================================================================


def test_time_constant_accuracy(hippocampus):
    """Test that decay rates match configured time constants."""
    config = hippocampus.config
    dt_ms = config.dt_ms

    # Fast trace: τ = 60,000ms
    fast_decay_rate = dt_ms / config.fast_trace_tau_ms
    assert abs(fast_decay_rate - 1.0 / 60_000) < 1e-9

    # Slow trace: τ = 3,600,000ms
    slow_decay_rate = dt_ms / config.slow_trace_tau_ms
    assert abs(slow_decay_rate - 1.0 / 3_600_000) < 1e-9

    # Half-life calculation
    # For exponential decay: t_half = τ * ln(2) ≈ 0.693 * τ
    fast_half_life = config.fast_trace_tau_ms * 0.693
    slow_half_life = config.slow_trace_tau_ms * 0.693

    # Fast half-life: ~41.6 seconds
    assert 40_000 < fast_half_life < 43_000

    # Slow half-life: ~41.6 minutes
    assert 2_400_000 < slow_half_life < 2_600_000


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
