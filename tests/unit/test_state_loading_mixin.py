"""Tests for StateLoadingMixin."""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.core.region_state import BaseRegionState
from thalia.mixins.state_loading_mixin import StateLoadingMixin


@dataclass
class MockRegionState(BaseRegionState):
    """Mock state class for testing."""

    # Common fields
    spikes: Optional[torch.Tensor] = None
    membrane: Optional[torch.Tensor] = None
    g_exc: Optional[torch.Tensor] = None
    g_inh: Optional[torch.Tensor] = None
    g_adaptation: Optional[torch.Tensor] = None

    # Learning traces
    eligibility_trace: Optional[torch.Tensor] = None
    bcm_threshold: Optional[torch.Tensor] = None

    # Neuromodulators
    dopamine: Optional[float] = None
    acetylcholine: Optional[float] = None
    norepinephrine: Optional[float] = None

    # STP state
    stp_state: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "spikes": self.spikes,
            "membrane": self.membrane,
            "g_exc": self.g_exc,
            "g_inh": self.g_inh,
            "g_adaptation": self.g_adaptation,
            "eligibility_trace": self.eligibility_trace,
            "bcm_threshold": self.bcm_threshold,
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
            "stp_state": self.stp_state,
        }


class MockRegion(StateLoadingMixin):
    """Mock region for testing StateLoadingMixin."""

    def __init__(self, n_neurons: int, device: torch.device):
        self.device = device
        self.n_neurons = n_neurons

        # Create neurons
        config = ConductanceLIFConfig()
        self.neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
        self.neurons.to(device)
        self.neurons.reset_state()  # Initialize state tensors

        # Learning traces
        self.eligibility_trace = torch.zeros(n_neurons, device=device)
        self.bcm_threshold = torch.ones(n_neurons, device=device)

        # Mock state object (like prefrontal/cerebellum pattern)
        class State:
            dopamine = 0.0
            acetylcholine = 0.0
            norepinephrine = 0.0

        self.state = State()


class TestStateLoadingMixin:
    """Test suite for StateLoadingMixin."""

    @pytest.fixture
    def device(self):
        """Provide device for tests."""
        return torch.device("cpu")

    @pytest.fixture
    def region(self, device):
        """Create mock region."""
        return MockRegion(n_neurons=100, device=device)

    def test_restore_neuron_state(self, region, device):
        """Test _restore_neuron_state() method."""
        # Create mock state
        v_mem = torch.randn(100, device=device)
        state_dict = {"membrane": v_mem, "v_mem": v_mem}

        # Restore
        region._restore_neuron_state(state_dict)

        # Verify
        assert torch.allclose(region.neurons.membrane, v_mem)

    def test_restore_conductances(self, region, device):
        """Test _restore_conductances() method."""
        # Create mock state
        g_exc = torch.rand(100, device=device)
        g_inh = torch.rand(100, device=device)
        g_adapt = torch.rand(100, device=device)

        state_dict = {
            "g_exc": g_exc,
            "g_inh": g_inh,
            "g_adaptation": g_adapt,
        }

        # Restore
        region._restore_conductances(state_dict)

        # Verify (note: ConductanceLIF uses g_adapt, not g_adaptation)
        assert torch.allclose(region.neurons.g_E, g_exc)
        assert torch.allclose(region.neurons.g_I, g_inh)
        assert torch.allclose(region.neurons.g_adapt, g_adapt)

    def test_restore_learning_traces(self, region, device):
        """Test _restore_learning_traces() method."""
        # Create mock state
        eligibility = torch.rand(100, device=device)
        bcm_threshold = torch.rand(100, device=device)

        state_dict = {
            "eligibility_trace": eligibility,
            "bcm_threshold": bcm_threshold,
        }

        # Restore
        region._restore_learning_traces(state_dict)

        # Verify
        assert torch.allclose(region.eligibility_trace, eligibility)
        assert torch.allclose(region.bcm_threshold, bcm_threshold)

    def test_restore_neuromodulators(self, region):
        """Test _restore_neuromodulators() method."""
        state_dict = {
            "dopamine": 0.5,
            "acetylcholine": 0.7,
            "norepinephrine": 0.3,
        }

        # Restore
        region._restore_neuromodulators(state_dict)

        # Verify (using self.state.{modulator} pattern)
        assert region.state.dopamine == 0.5
        assert region.state.acetylcholine == 0.7
        assert region.state.norepinephrine == 0.3

    def test_load_state_integration(self, region, device):
        """Test full load_state() integration."""
        # Create mock state
        state = MockRegionState(
            spikes=torch.zeros(100, device=device),
            membrane=torch.randn(100, device=device),
            g_exc=torch.rand(100, device=device),
            g_inh=torch.rand(100, device=device),
            eligibility_trace=torch.rand(100, device=device),
            bcm_threshold=torch.rand(100, device=device),
            dopamine=0.5,
            acetylcholine=0.7,
            norepinephrine=0.3,
        )

        # Load
        region.load_state(state)

        # Verify all components restored
        assert torch.allclose(region.neurons.membrane, state.membrane)
        assert torch.allclose(region.neurons.g_E, state.g_exc)
        assert torch.allclose(region.neurons.g_I, state.g_inh)
        assert torch.allclose(region.eligibility_trace, state.eligibility_trace)
        assert torch.allclose(region.bcm_threshold, state.bcm_threshold)
        assert region.state.dopamine == 0.5
        assert region.state.acetylcholine == 0.7
        assert region.state.norepinephrine == 0.3

    def test_device_transfer(self, region):
        """Test device transfer during restoration."""
        # Create state on CPU
        state = MockRegionState(
            membrane=torch.randn(100),  # CPU
            g_exc=torch.rand(100),
        )

        # Load (should transfer to region.device)
        region.load_state(state)

        # Verify tensors on correct device
        assert region.neurons.membrane.device == region.device
        assert region.neurons.g_E.device == region.device

    def test_missing_state_fields(self, region, device):
        """Test handling of missing/None state fields."""
        # Create partial state
        state = MockRegionState(
            membrane=torch.randn(100, device=device),
            g_exc=None,  # Missing
            eligibility_trace=None,
        )

        # Should not raise - missing fields gracefully handled
        region.load_state(state)

        # Verify present field restored
        assert torch.allclose(region.neurons.membrane, state.membrane)
