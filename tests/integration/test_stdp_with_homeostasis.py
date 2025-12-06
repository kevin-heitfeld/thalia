"""
Integration tests for STDP + Homeostasis interaction.

Tests the interaction between STDP learning and homeostatic
plasticity mechanisms, demonstrating how stability emerges from
their combination.

Complexity Level: 1 (Learning Rules) + 2 (Stability)
"""

import pytest
import torch

from thalia.learning.strategies import STDPWithHomeostasis
from thalia.learning.bcm import BCMRule
from thalia.learning.unified_homeostasis import UnifiedHomeostasis


class TestSTDPWithHomeostasis:
    """Test STDP learning with homeostatic regulation."""
    
    def test_homeostasis_prevents_runaway_potentiation(self, health_monitor):
        """Test that homeostasis prevents unlimited weight growth from STDP."""
        # Create STDP with homeostasis
        stdp_homeostasis = STDPWithHomeostasis(
            n_neurons=64,
            stdp_a_plus=0.01,
            stdp_a_minus=0.01,
            homeostasis_target_rate=0.05,
        )
        
        # Create random input that would normally cause runaway
        pre_spikes = torch.rand(100, 32) > 0.9  # 10% firing rate
        post_spikes = torch.rand(100, 64) > 0.9
        
        weights = torch.randn(64, 32) * 0.1
        
        # Run learning for many steps
        for t in range(100):
            weights = stdp_homeostasis.update(
                weights=weights,
                pre_spikes=pre_spikes[t],
                post_spikes=post_spikes[t],
            )
        
        # Check that weights are bounded (not exploded)
        assert weights.abs().max() < 2.0, "Weights exploded despite homeostasis"
        assert weights.abs().min() > 0.001, "Weights collapsed despite STDP"
        
        # Check health
        diagnostics = {
            "spike_counts": {"test_region": int(post_spikes.sum())},
            "cortex": {"test_w_mean": float(weights.mean())},
            "dopamine": {"global": 0.0},
        }
        report = health_monitor.check_health(diagnostics)
        
        # Should be healthy or have only minor issues
        assert report.overall_severity < 50, f"Unhealthy state: {report.summary}"
    
    def test_homeostasis_rescues_silent_neurons(self):
        """Test that homeostasis increases excitability of silent neurons."""
        homeostasis = UnifiedHomeostasis(
            target_rate=0.05,
            tau_homeostasis=100.0,
            adaptation_lr=0.01,
        )
        
        # Create scenario with silent neurons
        weights = torch.zeros(64, 32)
        weights[:32, :] = 0.5  # Only first half active
        
        spike_history = []
        
        # Simulate with input
        for t in range(200):
            pre_spikes = torch.rand(32) > 0.9
            pre_input = weights @ pre_spikes.float()
            
            # Apply homeostasis
            modulated_input = homeostasis.modulate_input(pre_input, neuron_idx=None)
            
            # Neurons spike if input > threshold
            post_spikes = modulated_input > 0.5
            
            # Update homeostasis
            homeostasis.update(post_spikes.float())
            
            spike_history.append(post_spikes)
        
        # Check that initially silent neurons (32-64) eventually spike
        early_spikes = torch.stack(spike_history[:50]).sum(dim=0)
        late_spikes = torch.stack(spike_history[150:]).sum(dim=0)
        
        # Silent neurons should have low early activity
        assert early_spikes[32:].float().mean() < early_spikes[:32].float().mean()
        
        # But homeostasis should increase their activity over time
        assert late_spikes[32:].sum() > early_spikes[32:].sum()
    
    def test_stdp_homeostasis_converges_to_target_rate(self):
        """Test that combined STDP+homeostasis converges to target rate."""
        stdp_homeostasis = STDPWithHomeostasis(
            n_neurons=64,
            stdp_a_plus=0.005,
            stdp_a_minus=0.005,
            homeostasis_target_rate=0.08,  # 8% target
            homeostasis_tau=1000.0,
        )
        
        # Random input patterns
        weights = torch.randn(64, 32) * 0.1
        firing_rates = []
        
        for t in range(500):
            pre_spikes = torch.rand(32) > 0.85  # 15% pre rate
            pre_input = weights @ pre_spikes.float()
            
            # Apply homeostasis modulation
            modulated = stdp_homeostasis.modulate_input(pre_input)
            post_spikes = modulated > 0.5
            
            # Update weights with STDP
            weights = stdp_homeostasis.update(
                weights=weights,
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
            )
            
            firing_rates.append(post_spikes.float().mean().item())
        
        # Check convergence to target
        final_rate = sum(firing_rates[-100:]) / 100
        assert abs(final_rate - 0.08) < 0.03, \
            f"Did not converge to target rate: {final_rate:.3f} vs 0.08"


class TestBCMWithHomeostasis:
    """Test BCM learning rule with homeostatic mechanisms."""
    
    def test_bcm_adapts_threshold(self):
        """Test that BCM sliding threshold prevents saturation."""
        bcm = BCMRule(
            learning_rate=0.001,
            tau_theta=1000.0,
        )
        
        # Simulate high activity that should raise threshold
        high_activity = torch.ones(64) * 0.8
        
        initial_theta = bcm.theta.clone()
        
        # Update threshold with high activity
        for _ in range(100):
            bcm.update_threshold(high_activity)
        
        final_theta = bcm.theta
        
        # Threshold should increase
        assert (final_theta > initial_theta).all()
        
        # Now simulate low activity
        low_activity = torch.ones(64) * 0.1
        
        for _ in range(100):
            bcm.update_threshold(low_activity)
        
        # Threshold should decrease
        assert (bcm.theta < final_theta).all()
    
    def test_bcm_selective_strengthening(self):
        """Test that BCM strengthens active synapses, weakens inactive."""
        bcm = BCMRule(learning_rate=0.01, tau_theta=100.0)
        
        weights = torch.ones(64, 32) * 0.1
        
        # Pattern with selective activation
        pre_pattern = torch.zeros(32)
        pre_pattern[:16] = 1.0  # Only first half active
        
        post_activity = weights @ pre_pattern
        
        # Update multiple times
        for _ in range(50):
            dw = bcm.compute_update(
                pre_spikes=pre_pattern,
                post_activity=post_activity,
                weights=weights,
            )
            weights += dw
            bcm.update_threshold(post_activity)
        
        # Active synapses (first half) should be stronger
        active_weights = weights[:, :16].mean()
        inactive_weights = weights[:, 16:].mean()
        
        assert active_weights > inactive_weights, \
            "BCM did not selectively strengthen active synapses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
