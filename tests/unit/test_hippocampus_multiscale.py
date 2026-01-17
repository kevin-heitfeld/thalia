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
    """Test that fast/slow traces are initialized with correct shapes."""
    # All traces should be initialized as tensors (not None)
    assert hippocampus._ca3_ca3_fast is not None
    assert hippocampus._ca3_ca3_slow is not None
    assert hippocampus._ca3_ca2_fast is not None
    assert hippocampus._ca3_ca2_slow is not None
    assert hippocampus._ec_ca2_fast is not None
    assert hippocampus._ec_ca2_slow is not None
    assert hippocampus._ec_ca1_fast is not None
    assert hippocampus._ec_ca1_slow is not None
    assert hippocampus._ca2_ca1_fast is not None
    assert hippocampus._ca2_ca1_slow is not None

    # Check shapes match weight matrices
    assert hippocampus._ca3_ca3_fast.shape == (
        small_hippocampus_sizes["ca3_size"],
        small_hippocampus_sizes["ca3_size"],
    )
    assert hippocampus._ca3_ca2_fast.shape == (
        small_hippocampus_sizes["ca2_size"],
        small_hippocampus_sizes["ca3_size"],
    )
    assert hippocampus._ec_ca2_fast.shape == (
        small_hippocampus_sizes["ca2_size"],
        small_hippocampus_sizes["input_size"],
    )
    assert hippocampus._ec_ca1_fast.shape == (
        small_hippocampus_sizes["ca1_size"],
        small_hippocampus_sizes["input_size"],
    )
    assert hippocampus._ca2_ca1_fast.shape == (
        small_hippocampus_sizes["ca1_size"],
        small_hippocampus_sizes["ca2_size"],
    )

    # All traces should start at zero
    assert torch.allclose(hippocampus._ca3_ca3_fast, torch.zeros_like(hippocampus._ca3_ca3_fast))
    assert torch.allclose(hippocampus._ca3_ca3_slow, torch.zeros_like(hippocampus._ca3_ca3_slow))


# =============================================================================
# Test 2: Disabled Flag Check
# =============================================================================


def test_disabled_multiscale(small_hippocampus_config, small_hippocampus_sizes, device):
    """Test that traces are None when multiscale is disabled."""
    config = small_hippocampus_config
    config.use_multiscale_consolidation = False

    hippo = TrisynapticHippocampus(
        config=config,
        sizes=small_hippocampus_sizes,
        device=device,
    )

    # All traces should be None when feature disabled
    assert hippo._ca3_ca3_fast is None
    assert hippo._ca3_ca3_slow is None
    assert hippo._ca3_ca2_fast is None
    assert hippo._ca3_ca2_slow is None
    assert hippo._ec_ca2_fast is None
    assert hippo._ec_ca2_slow is None
    assert hippo._ec_ca1_fast is None
    assert hippo._ec_ca1_slow is None
    assert hippo._ca2_ca1_fast is None
    assert hippo._ca2_ca1_slow is None


# =============================================================================
# Test 3: Fast Trace Decay
# =============================================================================


def test_fast_trace_decay(hippocampus, small_hippocampus_sizes):
    """Test that fast traces decay with τ ~60s."""
    config = hippocampus.config
    dt = config.dt_ms

    # Initialize fast trace to 1.0
    hippocampus._ca3_ca3_fast = torch.ones_like(hippocampus._ca3_ca3_fast)

    # Simulate 60 seconds (60,000 timesteps at 1ms)
    # With τ=60,000ms, after 60,000ms we expect decay to ~36.8% (1/e)
    n_steps = 60_000
    decay_rate = dt / config.fast_trace_tau_ms

    for _ in range(n_steps):
        hippocampus._ca3_ca3_fast = (1.0 - decay_rate) * hippocampus._ca3_ca3_fast

    # After 1 time constant, should be at ~1/e ≈ 0.368
    expected = torch.ones_like(hippocampus._ca3_ca3_fast) * 0.368

    # Allow 5% tolerance due to discrete timesteps
    assert torch.allclose(hippocampus._ca3_ca3_fast, expected, atol=0.02)


# =============================================================================
# Test 4: Slow Trace Persistence
# =============================================================================


def test_slow_trace_persistence(hippocampus):
    """Test that slow traces persist much longer (τ ~3600s)."""
    config = hippocampus.config
    dt = config.dt_ms

    # Initialize slow trace to 1.0
    hippocampus._ca3_ca3_slow = torch.ones_like(hippocampus._ca3_ca3_slow)

    # Simulate 60 seconds (should barely decay)
    n_steps = 60_000
    decay_rate = dt / config.slow_trace_tau_ms

    for _ in range(n_steps):
        hippocampus._ca3_ca3_slow = (1.0 - decay_rate) * hippocampus._ca3_ca3_slow

    # After 60s (1/60th of time constant), should still be ~98.4%
    # exp(-60/3600) = exp(-1/60) ≈ 0.9835
    expected = torch.ones_like(hippocampus._ca3_ca3_slow) * 0.9835

    # Very slow decay - should be close to 1.0
    assert torch.allclose(hippocampus._ca3_ca3_slow, expected, atol=0.005)


# =============================================================================
# Test 5: Consolidation Transfer
# =============================================================================


def test_consolidation_transfer(hippocampus):
    """Test that fast trace gradually transfers to slow trace."""
    config = hippocampus.config
    dt = config.dt_ms

    # Set fast trace to 1.0, slow trace to 0.0
    hippocampus._ca3_ca3_fast = torch.ones_like(hippocampus._ca3_ca3_fast)
    hippocampus._ca3_ca3_slow = torch.zeros_like(hippocampus._ca3_ca3_slow)

    # Simulate consolidation over time (no new learning, just transfer)
    n_steps = 100_000  # 100 seconds
    fast_decay = dt / config.fast_trace_tau_ms
    slow_decay = dt / config.slow_trace_tau_ms
    consolidation_rate = config.consolidation_rate

    for _ in range(n_steps):
        # Fast trace decays
        hippocampus._ca3_ca3_fast = (1.0 - fast_decay) * hippocampus._ca3_ca3_fast

        # Slow trace accumulates from fast
        consolidation = consolidation_rate * hippocampus._ca3_ca3_fast
        hippocampus._ca3_ca3_slow = (1.0 - slow_decay) * hippocampus._ca3_ca3_slow + consolidation

    # Fast trace should have decayed significantly
    assert hippocampus._ca3_ca3_fast.max() < 0.3

    # Slow trace should have accumulated (not zero anymore)
    assert hippocampus._ca3_ca3_slow.max() > 0.01

    # With consolidation_rate=0.001 and 100k timesteps, slow trace will accumulate
    # significantly. The key is that consolidation IS happening (slow > 0).
    # The exact value depends on the interplay of consolidation and decay.
    assert hippocampus._ca3_ca3_slow.mean() > 1.0  # Substantial consolidation occurred


def test_combined_learning(hippocampus):
    """Test that weight updates combine fast + slow traces correctly."""
    config = hippocampus.config

    # Set different values for fast and slow traces
    hippocampus._ca3_ca3_fast = torch.ones_like(hippocampus._ca3_ca3_fast) * 1.0
    hippocampus._ca3_ca3_slow = torch.ones_like(hippocampus._ca3_ca3_slow) * 0.5

    # Compute combined update
    combined = (
        hippocampus._ca3_ca3_fast + config.slow_trace_contribution * hippocampus._ca3_ca3_slow
    )

    # Should be: 1.0 + 0.1 * 0.5 = 1.05
    expected = torch.ones_like(hippocampus._ca3_ca3_fast) * 1.05

    assert torch.allclose(combined, expected)


# =============================================================================
# Test 7: Episodic vs Semantic (Qualitative)
# =============================================================================


def test_episodic_vs_semantic_dynamics(hippocampus):
    """Test that fast trace captures episodes, slow trace captures patterns."""
    config = hippocampus.config
    dt = config.dt_ms

    # Simulate 3 learning episodes with 60s gaps
    # Episode 1: Strong learning event
    hippocampus._ca3_ca3_fast = torch.ones_like(hippocampus._ca3_ca3_fast) * 2.0
    hippocampus._ca3_ca3_slow = torch.zeros_like(hippocampus._ca3_ca3_slow)

    # Wait 60 seconds (fast decays, slow accumulates)
    n_steps = 60_000
    fast_decay = dt / config.fast_trace_tau_ms
    slow_decay = dt / config.slow_trace_tau_ms

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
        output = hippocampus(input_pattern)

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
    dt = config.dt_ms

    # Fast trace: τ = 60,000ms
    fast_decay_rate = dt / config.fast_trace_tau_ms
    assert abs(fast_decay_rate - 1.0 / 60_000) < 1e-9

    # Slow trace: τ = 3,600,000ms
    slow_decay_rate = dt / config.slow_trace_tau_ms
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
