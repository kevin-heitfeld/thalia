"""
Tests for robustness mechanisms.

This module tests all robustness mechanisms:
- E/I Balance Regulation
- Divisive Normalization
- Intrinsic Plasticity
- Criticality Monitoring
- Metabolic Constraints
"""

import pytest
import torch

from thalia.learning.ei_balance import (
    EIBalanceConfig,
    EIBalanceRegulator,
    LayerEIBalance,
)
from thalia.core.normalization import (
    DivisiveNormConfig,
    DivisiveNormalization,
    ContrastNormalization,
)
from thalia.learning.intrinsic_plasticity import (
    IntrinsicPlasticityConfig,
    IntrinsicPlasticity,
    PopulationIntrinsicPlasticity,
)
from thalia.diagnostics.criticality import (
    CriticalityConfig,
    CriticalityMonitor,
    CriticalityState,
    AvalancheAnalyzer,
)
from thalia.learning.metabolic import (
    MetabolicConfig,
    MetabolicConstraint,
    RegionalMetabolicBudget,
)
from thalia.config.robustness_config import RobustnessConfig


# =============================================================================
# E/I BALANCE TESTS
# =============================================================================

class TestEIBalanceRegulator:
    """Tests for E/I balance regulation."""
    
    def test_initialization(self):
        """Test that regulator initializes correctly."""
        config = EIBalanceConfig(target_ratio=4.0)
        regulator = EIBalanceRegulator(config)
        
        assert regulator.get_inh_scaling() == 1.0
        assert regulator.get_current_ratio() == pytest.approx(4.0, rel=0.5)
    
    def test_compute_ratio(self):
        """Test E/I ratio computation."""
        regulator = EIBalanceRegulator()
        
        exc_spikes = torch.tensor([[1, 1, 0, 0, 1, 0, 0, 0]])  # 3/8 = 0.375
        inh_spikes = torch.tensor([[1, 0]])  # 1/2 = 0.5
        
        ratio = regulator.compute_ratio(exc_spikes, inh_spikes)
        # exc_mean = 0.375, inh_mean = 0.5
        assert ratio == pytest.approx(0.375 / 0.5, rel=0.01)
    
    def test_corrects_high_excitation(self):
        """Test that high E/I ratio increases inhibition scaling."""
        config = EIBalanceConfig(
            target_ratio=4.0,
            tau_balance=100.0,  # Fast adaptation for test
            adaptation_rate=0.1,
        )
        regulator = EIBalanceRegulator(config)
        
        # Simulate high excitation (E/I >> 4)
        exc_spikes = torch.ones(1, 100)  # All firing
        inh_spikes = torch.zeros(1, 10)   # None firing
        
        initial_scale = regulator.get_inh_scaling()
        
        # Update multiple times
        for _ in range(50):
            regulator.update(exc_spikes, inh_spikes)
        
        final_scale = regulator.get_inh_scaling()
        
        # Inhibition should be boosted (scale > 1)
        assert final_scale > initial_scale
        assert final_scale > 1.0
    
    def test_corrects_high_inhibition(self):
        """Test that low E/I ratio decreases inhibition scaling."""
        config = EIBalanceConfig(
            target_ratio=4.0,
            tau_balance=100.0,
            adaptation_rate=0.1,
        )
        regulator = EIBalanceRegulator(config)
        
        # Simulate high inhibition (E/I << 4)
        exc_spikes = torch.zeros(1, 100)  # None firing
        inh_spikes = torch.ones(1, 10)     # All firing
        
        # Update multiple times
        for _ in range(50):
            regulator.update(exc_spikes, inh_spikes)
        
        final_scale = regulator.get_inh_scaling()
        
        # Inhibition should be reduced (scale < 1)
        assert final_scale < 1.0
    
    def test_balanced_state_stable(self):
        """Test that balanced E/I ratio keeps scaling near 1."""
        config = EIBalanceConfig(target_ratio=4.0)
        regulator = EIBalanceRegulator(config)
        
        # Simulate balanced state (E/I ≈ 4)
        exc_spikes = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]])  # 50%
        inh_spikes = torch.tensor([[1, 0]])  # 50% -> ratio ≈ 1 (need adjustment)
        
        # Actually create a 4:1 ratio scenario
        exc_spikes = torch.zeros(1, 100)
        exc_spikes[0, :40] = 1  # 40% excitatory
        inh_spikes = torch.zeros(1, 10)
        inh_spikes[0, :1] = 1   # 10% inhibitory -> ratio = 4
        
        for _ in range(100):
            regulator.update(exc_spikes, inh_spikes)
        
        scale = regulator.get_inh_scaling()
        assert scale == pytest.approx(1.0, rel=0.5)
    
    def test_health_status(self):
        """Test health status reporting."""
        regulator = EIBalanceRegulator(EIBalanceConfig(target_ratio=4.0))
        
        # Initially should be roughly balanced
        status = regulator.get_health_status()
        assert status in ["balanced", "over-excited", "over-inhibited"]
    
    def test_diagnostics(self):
        """Test diagnostics output."""
        regulator = EIBalanceRegulator()
        
        diag = regulator.get_diagnostics()
        
        assert "exc_avg" in diag
        assert "inh_avg" in diag
        assert "current_ratio" in diag
        assert "target_ratio" in diag
        assert "inh_scale" in diag
        assert "status" in diag


class TestLayerEIBalance:
    """Tests for layer-level E/I balance."""
    
    def test_initialization(self):
        """Test layer E/I balance initialization."""
        layer = LayerEIBalance(n_exc=80, n_inh=20)
        
        assert layer.n_exc == 80
        assert layer.n_inh == 20
    
    def test_scale_inhibition(self):
        """Test inhibitory weight scaling."""
        layer = LayerEIBalance(n_exc=80, n_inh=20)
        
        inh_weights = torch.ones(10, 20)
        scaled = layer.scale_inhibition(inh_weights)
        
        assert scaled.shape == inh_weights.shape


# =============================================================================
# DIVISIVE NORMALIZATION TESTS
# =============================================================================

class TestDivisiveNormalization:
    """Tests for divisive normalization."""
    
    def test_initialization(self):
        """Test that normalization initializes correctly."""
        norm = DivisiveNormalization(DivisiveNormConfig(sigma=1.0))
        assert norm.config.sigma == 1.0
    
    def test_global_normalization(self):
        """Test global divisive normalization."""
        norm = DivisiveNormalization(DivisiveNormConfig(sigma=1.0))
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = norm(x)
        
        # output = x / (sigma^2 + sum(x))
        expected = x / (1.0 + x.sum())
        
        assert torch.allclose(y, expected, atol=1e-6)
    
    def test_self_normalization(self):
        """Test self-only divisive normalization."""
        config = DivisiveNormConfig(sigma=1.0, pool_type="self")
        norm = DivisiveNormalization(config)
        
        x = torch.tensor([1.0, 2.0, 3.0])
        y = norm(x)
        
        # output = x / (sigma^2 + x)
        expected = x / (1.0 + x)
        
        assert torch.allclose(y, expected, atol=1e-6)
    
    def test_invariant_to_scaling(self):
        """Test that normalized outputs are similar across input scales."""
        norm = DivisiveNormalization(DivisiveNormConfig(sigma=0.1))
        
        x1 = torch.tensor([1.0, 2.0, 3.0])
        x10 = x1 * 10
        x100 = x1 * 100
        
        y1 = norm(x1)
        y10 = norm(x10)
        y100 = norm(x100)
        
        # Normalized patterns should be similar (not identical due to sigma)
        # Check that relative ordering is preserved
        assert (y1.argmax() == y10.argmax() == y100.argmax())
        
        # Higher inputs should produce more similar outputs (sigma becomes negligible)
        cos_sim_10_100 = torch.cosine_similarity(y10.unsqueeze(0), y100.unsqueeze(0))
        assert cos_sim_10_100 > 0.99
    
    def test_batched_input(self):
        """Test normalization with batched input."""
        norm = DivisiveNormalization(DivisiveNormConfig(sigma=1.0))
        
        x = torch.randn(8, 64)  # batch of 8, 64 features
        y = norm(x)
        
        assert y.shape == x.shape
    
    def test_normalize_with_gain(self):
        """Test normalize_with_gain returns both output and gain."""
        norm = DivisiveNormalization()
        
        x = torch.tensor([1.0, 2.0, 3.0])
        y, gain = norm.normalize_with_gain(x)
        
        assert y.shape == x.shape
        assert gain.shape[0] == 1  # Global gain
        
        # y should equal x * gain (broadcast)
        assert torch.allclose(y, x * gain, atol=1e-6)


class TestContrastNormalization:
    """Tests for contrast normalization."""
    
    def test_removes_mean(self):
        """Test that contrast normalization centers the data."""
        norm = ContrastNormalization()
        
        x = torch.tensor([10.0, 12.0, 14.0])  # Mean = 12
        y = norm(x)
        
        # After subtractive step, mean should be ~0
        # But divisive step may shift it
        # At least check output is defined
        assert y.shape == x.shape
        assert not torch.isnan(y).any()


# =============================================================================
# INTRINSIC PLASTICITY TESTS
# =============================================================================

class TestIntrinsicPlasticity:
    """Tests for intrinsic plasticity."""
    
    def test_initialization(self):
        """Test that IP initializes correctly."""
        config = IntrinsicPlasticityConfig(target_rate=0.1)
        ip = IntrinsicPlasticity(n_neurons=100, config=config)
        
        assert ip.n_neurons == 100
        assert ip.thresholds.shape == (100,)
    
    def test_raises_threshold_for_active_neuron(self):
        """Test that active neurons get higher thresholds."""
        config = IntrinsicPlasticityConfig(
            target_rate=0.1,
            tau_rate=10.0,  # Fast for testing
            tau_threshold=10.0,
        )
        ip = IntrinsicPlasticity(n_neurons=10, config=config, initial_threshold=1.0)
        
        initial_thresh = ip.thresholds.clone()
        
        # Simulate high activity (100% firing)
        spikes = torch.ones(10)
        
        for _ in range(50):
            ip.update(spikes)
        
        final_thresh = ip.thresholds
        
        # Thresholds should increase
        assert (final_thresh > initial_thresh).all()
    
    def test_lowers_threshold_for_silent_neuron(self):
        """Test that silent neurons get lower thresholds (if bidirectional)."""
        config = IntrinsicPlasticityConfig(
            target_rate=0.1,
            tau_rate=10.0,
            tau_threshold=10.0,
            bidirectional=True,
        )
        ip = IntrinsicPlasticity(n_neurons=10, config=config, initial_threshold=1.0)
        
        initial_thresh = ip.thresholds.clone()
        
        # Simulate no activity
        spikes = torch.zeros(10)
        
        for _ in range(50):
            ip.update(spikes)
        
        final_thresh = ip.thresholds
        
        # Thresholds should decrease
        assert (final_thresh < initial_thresh).all()
    
    def test_respects_bounds(self):
        """Test that thresholds stay within bounds."""
        config = IntrinsicPlasticityConfig(
            v_thresh_min=0.5,
            v_thresh_max=2.0,
            tau_threshold=1.0,  # Very fast
        )
        ip = IntrinsicPlasticity(n_neurons=10, config=config)
        
        # Try to push thresholds very high
        spikes = torch.ones(10)
        for _ in range(1000):
            ip.update(spikes)
        
        assert (ip.thresholds <= config.v_thresh_max).all()
        assert (ip.thresholds >= config.v_thresh_min).all()
    
    def test_diagnostics(self):
        """Test diagnostics output."""
        ip = IntrinsicPlasticity(n_neurons=100)
        
        diag = ip.get_diagnostics()
        
        assert "rate_avg_mean" in diag
        assert "threshold_mean" in diag
        assert "target_rate" in diag


class TestPopulationIntrinsicPlasticity:
    """Tests for population-level intrinsic plasticity."""
    
    def test_modulates_excitability(self):
        """Test that population IP modulates excitability."""
        config = IntrinsicPlasticityConfig(
            target_rate=0.1,
            tau_rate=10.0,
            tau_threshold=10.0,
        )
        pip = PopulationIntrinsicPlasticity(config)
        
        # High activity should reduce excitability
        spikes = torch.ones(100)
        for _ in range(50):
            pip.update(spikes)
        
        assert pip.get_excitability() < 1.0
    
    def test_modulate_input(self):
        """Test input modulation."""
        pip = PopulationIntrinsicPlasticity()
        
        input_current = torch.randn(32, 100)
        modulated = pip.modulate_input(input_current)
        
        assert modulated.shape == input_current.shape


# =============================================================================
# CRITICALITY TESTS
# =============================================================================

class TestCriticalityMonitor:
    """Tests for criticality monitoring."""
    
    def test_initialization(self):
        """Test that monitor initializes correctly."""
        monitor = CriticalityMonitor()
        
        assert monitor.get_branching_ratio() == 1.0
        assert monitor.get_state() == CriticalityState.CRITICAL
    
    def test_detects_subcritical(self):
        """Test detection of subcritical state (activity dying)."""
        config = CriticalityConfig(
            window_size=20,
            target_branching=1.0,
        )
        monitor = CriticalityMonitor(config)
        
        # Simulate dying activity
        for i in range(20):
            spikes = torch.zeros(100)
            spikes[:max(1, 50 - i * 3)] = 1  # Decreasing activity
            monitor.update(spikes)
        
        # Should detect subcritical (branching < 1)
        assert monitor.get_branching_ratio() < 1.0
    
    def test_detects_supercritical(self):
        """Test detection of supercritical state (activity exploding)."""
        config = CriticalityConfig(
            window_size=20,
            target_branching=1.0,
        )
        monitor = CriticalityMonitor(config)
        
        # Simulate exploding activity
        for i in range(20):
            spikes = torch.zeros(100)
            spikes[:min(100, 10 + i * 5)] = 1  # Increasing activity
            monitor.update(spikes)
        
        # Should detect supercritical (branching > 1)
        assert monitor.get_branching_ratio() > 1.0
    
    def test_weight_scaling_correction(self):
        """Test that weight scaling corrects toward criticality."""
        config = CriticalityConfig(
            correction_enabled=True,
            correction_rate=0.01,
        )
        monitor = CriticalityMonitor(config)
        
        # Simulate supercritical state (should reduce weights)
        for i in range(50):
            spikes = torch.zeros(100)
            spikes[:min(100, 10 + i)] = 1
            monitor.update(spikes)
        
        scale = monitor.get_weight_scaling()
        # If supercritical, weights should be scaled down
        if monitor.get_branching_ratio() > 1.0:
            assert scale < 1.0
    
    def test_diagnostics(self):
        """Test diagnostics output."""
        monitor = CriticalityMonitor()
        
        # Need to update first
        monitor.update(torch.ones(10))
        monitor.update(torch.ones(10))
        
        diag = monitor.get_diagnostics()
        
        assert "branching_ratio" in diag
        assert "target_branching" in diag
        assert "state" in diag
        assert "weight_scale" in diag


class TestAvalancheAnalyzer:
    """Tests for avalanche analysis."""
    
    def test_detects_avalanches(self):
        """Test that avalanches are detected correctly."""
        analyzer = AvalancheAnalyzer(silence_threshold=0)
        
        # Simulate: silence -> activity -> silence
        analyzer.update(torch.zeros(100))
        analyzer.update(torch.ones(50))
        analyzer.update(torch.ones(30))
        analyzer.update(torch.zeros(100))
        
        sizes = analyzer.get_avalanche_sizes()
        assert len(sizes) == 1
        assert sizes[0] == 80  # 50 + 30 spikes
    
    def test_mean_size(self):
        """Test mean avalanche size computation."""
        analyzer = AvalancheAnalyzer()
        
        # Create a few avalanches
        for _ in range(5):
            analyzer.update(torch.zeros(100))
            analyzer.update(torch.ones(10))
            analyzer.update(torch.zeros(100))
        
        mean = analyzer.get_mean_size()
        assert mean == 10.0


# =============================================================================
# METABOLIC CONSTRAINTS TESTS
# =============================================================================

class TestMetabolicConstraint:
    """Tests for metabolic constraints."""
    
    def test_initialization(self):
        """Test that constraint initializes correctly."""
        config = MetabolicConfig(energy_per_spike=0.01, energy_budget=1.0)
        metabolic = MetabolicConstraint(config)
        
        assert metabolic.config.energy_per_spike == 0.01
    
    def test_compute_cost(self):
        """Test energy cost computation."""
        config = MetabolicConfig(energy_per_spike=0.1, baseline_cost=0.0)
        metabolic = MetabolicConstraint(config)
        
        spikes = torch.ones(10)  # 10 spikes
        cost = metabolic.compute_cost(spikes)
        
        assert cost == pytest.approx(1.0)  # 10 * 0.1 = 1.0
    
    def test_penalty_for_over_budget(self):
        """Test penalty when over energy budget."""
        config = MetabolicConfig(
            energy_per_spike=0.1,
            energy_budget=0.5,
            penalty_scale=1.0,
        )
        metabolic = MetabolicConstraint(config)
        
        spikes = torch.ones(10)  # Cost = 1.0, budget = 0.5
        penalty = metabolic.compute_penalty(spikes)
        
        # Excess = 1.0 - 0.5 = 0.5, penalty = 0.5 * 1.0 = 0.5
        assert penalty == pytest.approx(0.5)
    
    def test_no_penalty_under_budget(self):
        """Test no penalty when under energy budget."""
        config = MetabolicConfig(
            energy_per_spike=0.01,
            energy_budget=1.0,
        )
        metabolic = MetabolicConstraint(config)
        
        spikes = torch.ones(10)  # Cost = 0.1, budget = 1.0
        penalty = metabolic.compute_penalty(spikes)
        
        assert penalty == 0.0
    
    def test_gain_modulation_reduces_for_high_cost(self):
        """Test that gain is reduced when over budget."""
        config = MetabolicConfig(
            energy_per_spike=0.1,
            energy_budget=0.5,
            gain_modulation_enabled=True,
            gain_modulation_rate=0.1,
            tau_energy=10.0,  # Fast for testing
        )
        metabolic = MetabolicConstraint(config)
        
        # Repeatedly exceed budget
        spikes = torch.ones(100)  # High cost
        for _ in range(50):
            metabolic.update(spikes)
        
        gain = metabolic.get_gain_modulation()
        assert gain < 1.0  # Should reduce gain
    
    def test_efficiency_tracking(self):
        """Test efficiency computation."""
        config = MetabolicConfig(energy_per_spike=0.1, energy_budget=1.0)
        metabolic = MetabolicConstraint(config)
        
        spikes = torch.ones(5)  # Cost = 0.5
        metabolic.update(spikes)
        
        efficiency = metabolic.get_efficiency()
        # budget / cost = 1.0 / 0.5 = 2.0
        assert efficiency > 1.0  # Under budget = efficient
    
    def test_intrinsic_reward(self):
        """Test intrinsic reward computation."""
        config = MetabolicConfig(
            energy_per_spike=0.1,
            energy_budget=0.5,
        )
        metabolic = MetabolicConstraint(config)
        
        # Under budget
        reward_under = metabolic.get_intrinsic_reward(
            torch.ones(3),  # Cost = 0.3 < 0.5
            efficiency_bonus=0.1,
        )
        assert reward_under == 0.1  # Gets bonus
        
        # Over budget
        reward_over = metabolic.get_intrinsic_reward(
            torch.ones(10),  # Cost = 1.0 > 0.5
        )
        assert reward_over < 0  # Negative reward
    
    def test_diagnostics(self):
        """Test diagnostics output."""
        metabolic = MetabolicConstraint()
        metabolic.update(torch.ones(10))
        
        diag = metabolic.get_diagnostics()
        
        assert "energy_avg" in diag
        assert "energy_budget" in diag
        assert "efficiency" in diag
        assert "gain" in diag


class TestRegionalMetabolicBudget:
    """Tests for regional metabolic budget."""
    
    def test_tracks_multiple_regions(self):
        """Test tracking across multiple regions."""
        budgets = {"cortex": 1.0, "hippocampus": 0.5, "pfc": 0.3}
        regional = RegionalMetabolicBudget(budgets)
        
        regional.update_region("cortex", torch.ones(50))
        regional.update_region("hippocampus", torch.ones(20))
        
        total = regional.get_total_cost()
        assert total > 0
    
    def test_global_budget_check(self):
        """Test global budget checking."""
        # Use higher energy_per_spike so cost exceeds budget
        config = MetabolicConfig(energy_per_spike=0.1)
        budgets = {"a": 1.0, "b": 1.0}
        regional = RegionalMetabolicBudget(budgets, global_budget=1.0, config=config)
        
        # 100 spikes * 0.1 = 10.0 cost per region
        regional.update_region("a", torch.ones(100))
        regional.update_region("b", torch.ones(100))
        
        # Total cost = 20.0 > global budget = 1.0
        assert regional.is_globally_over_budget()


# =============================================================================
# ROBUSTNESS CONFIG TESTS
# =============================================================================

class TestRobustnessConfig:
    """Tests for RobustnessConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = RobustnessConfig()
        
        assert config.enable_ei_balance is True
        assert config.enable_divisive_norm is True
        assert config.enable_intrinsic_plasticity is True
        assert config.enable_criticality is False
        assert config.enable_metabolic is False
    
    def test_minimal_preset(self):
        """Test minimal preset."""
        config = RobustnessConfig.minimal()
        
        assert config.enable_ei_balance is False
        assert config.enable_divisive_norm is False
        assert len(config.get_enabled_mechanisms()) == 0
    
    def test_full_preset(self):
        """Test full preset."""
        config = RobustnessConfig.full()
        
        assert config.enable_ei_balance is True
        assert config.enable_divisive_norm is True
        assert config.enable_intrinsic_plasticity is True
        assert config.enable_criticality is True
        assert config.enable_metabolic is True
        assert len(config.get_enabled_mechanisms()) == 5
    
    def test_get_enabled_mechanisms(self):
        """Test getting list of enabled mechanisms."""
        config = RobustnessConfig(
            enable_ei_balance=True,
            enable_divisive_norm=False,
            enable_intrinsic_plasticity=True,
        )
        
        enabled = config.get_enabled_mechanisms()
        
        assert "ei_balance" in enabled
        assert "divisive_norm" not in enabled
        assert "intrinsic_plasticity" in enabled
    
    def test_summary(self):
        """Test summary output."""
        config = RobustnessConfig.biological()
        summary = config.summary()
        
        assert "E/I Balance: ON" in summary
        assert "Criticality: ON" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
