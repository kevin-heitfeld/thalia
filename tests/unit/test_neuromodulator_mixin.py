"""Tests for NeuromodulatorMixin.

Validates that the mixin provides consistent neuromodulator handling
across brain regions.
"""

import pytest
import torch
import math

from thalia.core.neuromodulator_mixin import NeuromodulatorMixin
from thalia.regions.base import RegionState


class MockRegionWithMixin(NeuromodulatorMixin):
    """Mock region for testing the mixin."""
    
    def __init__(self):
        self.state = RegionState()
        self.base_learning_rate = 0.01


class TestNeuromodulatorMixin:
    """Test suite for NeuromodulatorMixin."""
    
    @pytest.fixture
    def region(self):
        """Create mock region with mixin."""
        return MockRegionWithMixin()
    
    def test_set_dopamine(self, region):
        """Test setting dopamine level."""
        region.set_dopamine(0.5)
        assert region.state.dopamine == 0.5
        
        region.set_dopamine(-0.3)
        assert region.state.dopamine == -0.3
    
    def test_set_acetylcholine(self, region):
        """Test setting acetylcholine level."""
        region.set_acetylcholine(0.7)
        assert region.state.acetylcholine == 0.7
    
    def test_set_norepinephrine(self, region):
        """Test setting norepinephrine level."""
        region.set_norepinephrine(0.4)
        assert region.state.norepinephrine == 0.4
    
    def test_set_neuromodulator_generic(self, region):
        """Test generic neuromodulator setter."""
        region.set_neuromodulator('dopamine', 0.8)
        assert region.state.dopamine == 0.8
        
        region.set_neuromodulator('acetylcholine', 0.6)
        assert region.state.acetylcholine == 0.6
        
        region.set_neuromodulator('norepinephrine', 0.5)
        assert region.state.norepinephrine == 0.5
    
    def test_set_neuromodulator_invalid(self, region):
        """Test that invalid neuromodulator name raises error."""
        with pytest.raises(ValueError, match="Unknown neuromodulator"):
            region.set_neuromodulator('serotonin', 0.5)
    
    def test_decay_neuromodulators_default_tau(self, region):
        """Test neuromodulator decay with default tau constants."""
        # Set initial levels
        region.set_dopamine(1.0)
        region.set_acetylcholine(1.0)
        region.set_norepinephrine(1.0)
        
        # Decay for 1ms
        region.decay_neuromodulators(dt_ms=1.0)
        
        # Expected decay factors with default tau
        # dopamine: tau=200ms → exp(-1/200) ≈ 0.995
        # acetylcholine: tau=50ms → exp(-1/50) ≈ 0.980
        # norepinephrine: tau=100ms → exp(-1/100) ≈ 0.990
        
        assert region.state.dopamine == pytest.approx(math.exp(-1.0 / 200.0), abs=1e-6)
        assert region.state.acetylcholine == pytest.approx(math.exp(-1.0 / 50.0), abs=1e-6)
        assert region.state.norepinephrine == pytest.approx(math.exp(-1.0 / 100.0), abs=1e-6)
    
    def test_decay_neuromodulators_custom_tau(self, region):
        """Test neuromodulator decay with custom tau constants."""
        region.set_dopamine(1.0)
        region.set_acetylcholine(1.0)
        region.set_norepinephrine(1.0)
        
        # Custom tau values
        region.decay_neuromodulators(
            dt_ms=10.0,
            dopamine_tau_ms=100.0,
            acetylcholine_tau_ms=25.0,
            norepinephrine_tau_ms=50.0
        )
        
        assert region.state.dopamine == pytest.approx(math.exp(-10.0 / 100.0), abs=1e-6)
        assert region.state.acetylcholine == pytest.approx(math.exp(-10.0 / 25.0), abs=1e-6)
        assert region.state.norepinephrine == pytest.approx(math.exp(-10.0 / 50.0), abs=1e-6)
    
    def test_decay_toward_zero(self, region):
        """Test that decay goes toward zero (baseline)."""
        region.set_dopamine(1.0)
        
        # Decay many times
        for _ in range(1000):
            region.decay_neuromodulators(dt_ms=1.0)
        
        # Should be close to zero after ~5 tau (1000ms)
        assert region.state.dopamine < 0.01
    
    def test_get_effective_learning_rate_baseline(self, region):
        """Test learning rate with zero dopamine (baseline)."""
        region.state.dopamine = 0.0
        lr = region.get_effective_learning_rate()
        assert lr == 0.01  # base_learning_rate
    
    def test_get_effective_learning_rate_positive_dopamine(self, region):
        """Test learning rate with positive dopamine (reward)."""
        region.state.dopamine = 1.0
        lr = region.get_effective_learning_rate()
        assert lr == pytest.approx(0.02, abs=1e-6)  # 0.01 * (1 + 1.0) = 0.02
    
    def test_get_effective_learning_rate_negative_dopamine(self, region):
        """Test learning rate with negative dopamine (punishment)."""
        region.state.dopamine = -0.5
        lr = region.get_effective_learning_rate()
        assert lr == pytest.approx(0.005, abs=1e-6)  # 0.01 * (1 - 0.5) = 0.005
    
    def test_get_effective_learning_rate_clamped(self, region):
        """Test that learning rate doesn't go negative."""
        region.state.dopamine = -2.0  # Extreme negative
        lr = region.get_effective_learning_rate()
        assert lr == 0.0  # Clamped to zero
    
    def test_get_effective_learning_rate_custom_base(self, region):
        """Test learning rate with custom base LR."""
        region.state.dopamine = 0.5
        lr = region.get_effective_learning_rate(base_lr=0.02)
        assert lr == pytest.approx(0.03, abs=1e-6)  # 0.02 * (1 + 0.5) = 0.03
    
    def test_get_effective_learning_rate_with_sensitivity(self, region):
        """Test learning rate with dopamine sensitivity parameter."""
        region.state.dopamine = 1.0
        
        # Full sensitivity
        lr = region.get_effective_learning_rate(dopamine_sensitivity=1.0)
        assert lr == pytest.approx(0.02, abs=1e-6)  # 0.01 * (1 + 1.0*1.0) = 0.02
        
        # Half sensitivity
        lr = region.get_effective_learning_rate(dopamine_sensitivity=0.5)
        assert lr == pytest.approx(0.015, abs=1e-6)  # 0.01 * (1 + 0.5*1.0) = 0.015
        
        # No sensitivity
        lr = region.get_effective_learning_rate(dopamine_sensitivity=0.0)
        assert lr == pytest.approx(0.01, abs=1e-6)  # 0.01 * (1 + 0.0*1.0) = 0.01
    
    def test_get_neuromodulator_state(self, region):
        """Test getting neuromodulator state for diagnostics."""
        region.set_dopamine(0.5)
        region.set_acetylcholine(0.7)
        region.set_norepinephrine(0.3)
        
        state = region.get_neuromodulator_state()
        
        assert state['dopamine'] == 0.5
        assert state['acetylcholine'] == 0.7
        assert state['norepinephrine'] == 0.3
    
    def test_default_tau_constants(self, region):
        """Test that default tau constants can be overridden."""
        # Default values
        assert region.DEFAULT_DOPAMINE_TAU_MS == 200.0
        assert region.DEFAULT_ACETYLCHOLINE_TAU_MS == 50.0
        assert region.DEFAULT_NOREPINEPHRINE_TAU_MS == 100.0
        
        # Create subclass with custom defaults
        class CustomRegion(MockRegionWithMixin):
            DEFAULT_DOPAMINE_TAU_MS = 150.0
            DEFAULT_ACETYLCHOLINE_TAU_MS = 30.0
            DEFAULT_NOREPINEPHRINE_TAU_MS = 75.0
        
        custom = CustomRegion()
        custom.set_dopamine(1.0)
        custom.decay_neuromodulators(dt_ms=1.0)
        
        # Should use custom tau
        assert custom.state.dopamine == pytest.approx(math.exp(-1.0 / 150.0), abs=1e-6)


class TestNeuromodulatorMixinIntegration:
    """Test that mixin integrates correctly with real BrainRegion."""
    
    def test_brain_region_has_mixin_methods(self):
        """Test that BrainRegion inherits mixin methods."""
        from thalia.regions.base import BrainRegion
        
        # Check that mixin methods are available
        assert hasattr(BrainRegion, 'set_dopamine')
        assert hasattr(BrainRegion, 'set_acetylcholine')
        assert hasattr(BrainRegion, 'set_norepinephrine')
        assert hasattr(BrainRegion, 'set_neuromodulator')
        assert hasattr(BrainRegion, 'decay_neuromodulators')
        assert hasattr(BrainRegion, 'get_effective_learning_rate')
        assert hasattr(BrainRegion, 'get_neuromodulator_state')
    
    def test_striatum_uses_mixin(self, striatum):
        """Test that Striatum (concrete region) can use mixin."""
        # Set via mixin
        striatum.set_dopamine(0.8)
        assert striatum.state.dopamine == 0.8
        
        # Decay via mixin
        striatum.decay_neuromodulators(dt_ms=1.0)
        assert striatum.state.dopamine < 0.8  # Decayed
        
        # Get effective LR via mixin
        lr = striatum.get_effective_learning_rate()
        assert lr > 0.0
    
    def test_prefrontal_uses_mixin(self, prefrontal):
        """Test that Prefrontal (concrete region) can use mixin."""
        prefrontal.set_acetylcholine(0.9)
        assert prefrontal.state.acetylcholine == 0.9
        
        prefrontal.decay_neuromodulators(dt_ms=5.0)
        assert prefrontal.state.acetylcholine < 0.9
    
    def test_mixin_state_dict(self, striatum):
        """Test neuromodulator state diagnostics."""
        striatum.set_dopamine(0.5)
        striatum.set_acetylcholine(0.3)
        striatum.set_norepinephrine(0.7)
        
        state = striatum.get_neuromodulator_state()
        
        assert state['dopamine'] == 0.5
        assert state['acetylcholine'] == 0.3
        assert state['norepinephrine'] == 0.7
