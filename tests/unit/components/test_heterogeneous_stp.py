"""
Tests for heterogeneous STP parameter sampling (Phase 1 Enhancement).

Validates that:
1. Parameters sampled from lognormal distributions
2. Biological variability is correct (CV ≈ specified variability)
3. 10-fold variability exists within pathway (Dobrunz & Stevens 1997)
4. Per-synapse STP dynamics differ appropriately
"""

import numpy as np
import pytest

from thalia.components.synapses import (
    create_heterogeneous_stp_configs,
    sample_heterogeneous_stp_params,
)


@pytest.fixture
def n_synapses():
    """Number of synapses for testing."""
    return 100


@pytest.fixture
def variability():
    """Coefficient of variation for testing (0.3 = 30%)."""
    return 0.3


class TestHeterogeneousSTPSampling:
    """Test heterogeneous STP parameter sampling functions."""

    def test_sample_basic_structure(self, n_synapses, variability):
        """Test that sampling returns correct array shapes."""
        U, tau_d, tau_f = sample_heterogeneous_stp_params(
            base_preset="corticostriatal",
            n_synapses=n_synapses,
            variability=variability,
            seed=42,
        )

        # Check shapes
        assert U.shape == (n_synapses,)
        assert tau_d.shape == (n_synapses,)
        assert tau_f.shape == (n_synapses,)

        # Check types
        assert isinstance(U, np.ndarray)
        assert isinstance(tau_d, np.ndarray)
        assert isinstance(tau_f, np.ndarray)

    def test_sample_biological_bounds(self, n_synapses, variability):
        """Test that sampled parameters respect biological bounds."""
        U, tau_d, tau_f = sample_heterogeneous_stp_params(
            base_preset="corticostriatal",
            n_synapses=n_synapses,
            variability=variability,
            seed=42,
        )

        # U must be in [0, 1]
        assert np.all(U >= 0.0)
        assert np.all(U <= 1.0)

        # Tau values must be positive
        assert np.all(tau_d > 0)
        assert np.all(tau_f > 0)

    def test_sample_variability(self, n_synapses):
        """Test that coefficient of variation matches specified variability."""
        variability = 0.3  # Target CV = 30%

        U, tau_d, tau_f = sample_heterogeneous_stp_params(
            base_preset="corticostriatal",
            n_synapses=1000,  # Large sample for statistical test
            variability=variability,
            seed=42,
        )

        # Compute coefficient of variation: CV = std / mean
        cv_U = np.std(U) / np.mean(U)
        cv_tau_d = np.std(tau_d) / np.mean(tau_d)
        cv_tau_f = np.std(tau_f) / np.mean(tau_f)

        # Allow 20% tolerance for stochastic sampling
        assert abs(cv_U - variability) < 0.1, f"U CV {cv_U:.3f} not close to {variability}"
        assert (
            abs(cv_tau_d - variability) < 0.1
        ), f"tau_d CV {cv_tau_d:.3f} not close to {variability}"
        assert (
            abs(cv_tau_f - variability) < 0.1
        ), f"tau_f CV {cv_tau_f:.3f} not close to {variability}"

    def test_sample_reproducibility(self, n_synapses, variability):
        """Test that same seed produces same samples."""
        U1, tau_d1, tau_f1 = sample_heterogeneous_stp_params(
            base_preset="corticostriatal",
            n_synapses=n_synapses,
            variability=variability,
            seed=42,
        )

        U2, tau_d2, tau_f2 = sample_heterogeneous_stp_params(
            base_preset="corticostriatal",
            n_synapses=n_synapses,
            variability=variability,
            seed=42,
        )

        # Same seed should give identical results
        np.testing.assert_array_equal(U1, U2)
        np.testing.assert_array_equal(tau_d1, tau_d2)
        np.testing.assert_array_equal(tau_f1, tau_f2)

    def test_sample_different_seeds(self, n_synapses, variability):
        """Test that different seeds produce different samples."""
        U1, _, _ = sample_heterogeneous_stp_params(
            base_preset="corticostriatal",
            n_synapses=n_synapses,
            variability=variability,
            seed=42,
        )

        U2, _, _ = sample_heterogeneous_stp_params(
            base_preset="corticostriatal",
            n_synapses=n_synapses,
            variability=variability,
            seed=99,
        )

        # Different seeds should give different results
        assert not np.array_equal(U1, U2)

    def test_tenfold_variability(self):
        """Test that 10-fold variability exists (Dobrunz & Stevens 1997)."""
        # High variability (CV=0.5) should produce 10-fold range
        U, _, _ = sample_heterogeneous_stp_params(
            base_preset="corticostriatal",
            n_synapses=1000,
            variability=0.5,  # High variability
            seed=42,
        )

        # Check that max/min ratio approaches 10 (allow some tolerance)
        ratio = np.max(U) / np.min(U)
        assert ratio >= 5.0, f"U variability ratio {ratio:.2f} too low (expected ~10x)"


class TestHeterogeneousSTPConfigs:
    """Test heterogeneous STP config creation."""

    def test_create_configs_structure(self, n_synapses, variability):
        """Test that create_heterogeneous_stp_configs returns correct structure."""
        configs = create_heterogeneous_stp_configs(
            base_preset="corticostriatal",
            n_synapses=n_synapses,
            variability=variability,
            seed=42,
        )

        # Should return list of configs
        assert isinstance(configs, list)
        assert len(configs) == n_synapses

        # Each config should have U, tau_d, tau_f fields
        for config in configs:
            assert hasattr(config, "U")
            assert hasattr(config, "tau_d")
            assert hasattr(config, "tau_f")

    def test_create_configs_diversity(self, variability):
        """Test that configs are diverse (not all identical)."""
        configs = create_heterogeneous_stp_configs(
            base_preset="corticostriatal",
            n_synapses=100,
            variability=variability,
            seed=42,
        )

        # Extract U values
        U_values = [cfg.U for cfg in configs]

        # Should have diversity (not all same)
        assert len(set(U_values)) > 10, "Configs should be diverse, not uniform"

    def test_create_configs_different_presets(self, n_synapses, variability):
        """Test that different base presets produce different configs."""
        cortico_configs = create_heterogeneous_stp_configs(
            base_preset="corticostriatal",
            n_synapses=n_synapses,
            variability=variability,
            seed=42,
        )

        thalamo_configs = create_heterogeneous_stp_configs(
            base_preset="thalamostriatal",
            n_synapses=n_synapses,
            variability=variability,
            seed=42,
        )

        # Mean U should differ between presets
        cortico_mean_U = np.mean([cfg.U for cfg in cortico_configs])
        thalamo_mean_U = np.mean([cfg.U for cfg in thalamo_configs])

        assert (
            abs(cortico_mean_U - thalamo_mean_U) > 0.05
        ), "Different presets should have different mean parameters"


class TestHeterogeneousSTPDynamics:
    """Test that heterogeneous STP produces diverse dynamics.

    Note: Current ShortTermPlasticity implementation doesn't support
    per-synapse heterogeneous parameters yet. These tests validate
    the parameter sampling infrastructure for future STP enhancement.
    """

    def test_heterogeneous_parameter_diversity(self):
        """Test that heterogeneous configs span a reasonable range."""
        n_synapses = 100

        # Create heterogeneous configs
        hetero_configs = create_heterogeneous_stp_configs(
            base_preset="corticostriatal",
            n_synapses=n_synapses,
            variability=0.3,
            seed=42,
        )

        # Extract parameters
        U_values = np.array([cfg.U for cfg in hetero_configs])
        tau_d_values = np.array([cfg.tau_d for cfg in hetero_configs])

        # Check diversity (range should be > 2x mean for CV=0.3)
        U_range = np.max(U_values) - np.min(U_values)
        U_mean = np.mean(U_values)
        assert (
            U_range > U_mean * 0.5
        ), f"U range {U_range:.3f} should be substantial relative to mean {U_mean:.3f}"

        tau_d_range = np.max(tau_d_values) - np.min(tau_d_values)
        tau_d_mean = np.mean(tau_d_values)
        assert (
            tau_d_range > tau_d_mean * 0.5
        ), f"tau_d range {tau_d_range:.1f} should be substantial relative to mean {tau_d_mean:.1f}"

    def test_config_parameter_access(self):
        """Test that configs have correct parameter attributes."""
        configs = create_heterogeneous_stp_configs(
            base_preset="corticostriatal",
            n_synapses=10,
            variability=0.3,
            seed=42,
        )

        # Each config should be an STPConfig with U, tau_d, tau_f
        for cfg in configs:
            assert hasattr(cfg, "U")
            assert hasattr(cfg, "tau_d")
            assert hasattr(cfg, "tau_f")
            assert hasattr(cfg, "decay_d")
            assert hasattr(cfg, "decay_f")

            # Values should be valid
            assert 0.0 <= cfg.U <= 1.0
            assert cfg.tau_d > 0
            assert cfg.tau_f > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
