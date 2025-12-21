"""Tests for Striatum RegionState protocol implementation.

Tests the StriatumState dataclass implementation of the RegionState protocol,
including serialization, device transfer, and integration with the Striatum region.

Phase 3.2 of state management refactoring.
"""

import pytest
import torch

from thalia.regions.striatum.striatum import Striatum
from thalia.regions.striatum.config import StriatumConfig, StriatumState
from thalia.core.region_state import save_region_state, load_region_state


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def device() -> str:
    """Test device (CPU)."""
    return "cpu"


@pytest.fixture
def striatum_config(device: str) -> StriatumConfig:
    """Create minimal Striatum configuration."""
    return StriatumConfig(
        n_input=20,
        n_output=5,  # 5 actions
        population_coding=True,
        neurons_per_action=4,  # 20 total neurons
        rpe_enabled=True,
        use_goal_conditioning=True,
        pfc_size=16,
        device=device,
        dt_ms=1.0,
    )


@pytest.fixture
def striatum_region(striatum_config: StriatumConfig) -> Striatum:
    """Create Striatum region instance."""
    region = Striatum(striatum_config)
    return region


@pytest.fixture
def sample_state(device: str) -> StriatumState:
    """Create a sample StriatumState with non-zero values."""
    n_neurons = 20
    n_input = 20
    pfc_size = 16

    return StriatumState(
        # Base state
        spikes=torch.randint(0, 2, (n_neurons,), device=device, dtype=torch.float32),
        membrane=torch.rand(n_neurons, device=device) * (-70.0) + (-70.0),

        # D1/D2 pathways (mock states)
        d1_pathway_state={"weights": torch.rand(n_neurons, n_input, device=device)},
        d2_pathway_state={"weights": torch.rand(n_neurons, n_input, device=device)},

        # Vote accumulation
        d1_votes_accumulated=torch.rand(n_neurons, device=device),
        d2_votes_accumulated=torch.rand(n_neurons, device=device),

        # Action selection
        last_action=2,
        recent_spikes=torch.randint(0, 2, (n_neurons,), device=device, dtype=torch.float32),

        # Exploration
        exploring=True,
        last_uncertainty=0.35,
        last_exploration_prob=0.15,
        exploration_manager_state={"action_counts": [5, 3, 7, 2, 4]},

        # Value/RPE
        value_estimates=torch.rand(5, device=device),  # 5 actions
        last_rpe=0.42,
        last_expected=0.68,

        # Goal modulation
        pfc_modulation_d1=torch.rand(n_neurons, pfc_size, device=device),
        pfc_modulation_d2=torch.rand(n_neurons, pfc_size, device=device),

        # Delay buffers (15ms and 25ms at 1ms dt)
        d1_delay_buffer=torch.rand(15, n_neurons, device=device),
        d2_delay_buffer=torch.rand(25, n_neurons, device=device),
        d1_delay_ptr=7,
        d2_delay_ptr=12,

        # Homeostasis
        activity_ema=0.45,
        trial_spike_count=142,
        trial_timesteps=200,
        homeostatic_scaling_applied=True,
        homeostasis_manager_state={"baseline": 0.02},

        # Neuromodulators
        dopamine=0.6,
        acetylcholine=0.3,
        norepinephrine=0.25,
    )


# ============================================================================
# TEST PROTOCOL COMPLIANCE
# ============================================================================


class TestStriatumStateProtocol:
    """Test StriatumState implements RegionState protocol."""

    def test_to_dict_basic(self, sample_state: StriatumState):
        """Test to_dict() serializes all fields."""
        state_dict = sample_state.to_dict()

        # Check base state
        assert "spikes" in state_dict
        assert "membrane" in state_dict

        # Check pathways
        assert "d1_pathway_state" in state_dict
        assert "d2_pathway_state" in state_dict

        # Check vote accumulation
        assert "d1_votes_accumulated" in state_dict
        assert "d2_votes_accumulated" in state_dict

        # Check action selection
        assert "last_action" in state_dict
        assert "recent_spikes" in state_dict

        # Check exploration
        assert "exploring" in state_dict
        assert "last_uncertainty" in state_dict
        assert "last_exploration_prob" in state_dict
        assert "exploration_manager_state" in state_dict

        # Check value/RPE
        assert "value_estimates" in state_dict
        assert "last_rpe" in state_dict
        assert "last_expected" in state_dict

        # Check goal modulation
        assert "pfc_modulation_d1" in state_dict
        assert "pfc_modulation_d2" in state_dict

        # Check delay buffers
        assert "d1_delay_buffer" in state_dict
        assert "d2_delay_buffer" in state_dict
        assert "d1_delay_ptr" in state_dict
        assert "d2_delay_ptr" in state_dict

        # Check homeostasis
        assert "activity_ema" in state_dict
        assert "trial_spike_count" in state_dict
        assert "trial_timesteps" in state_dict
        assert "homeostatic_scaling_applied" in state_dict
        assert "homeostasis_manager_state" in state_dict

        # Check neuromodulators
        assert "dopamine" in state_dict
        assert "acetylcholine" in state_dict
        assert "norepinephrine" in state_dict

        # Check tensor types
        assert isinstance(state_dict["spikes"], torch.Tensor)
        assert isinstance(state_dict["membrane"], torch.Tensor)
        assert isinstance(state_dict["d1_votes_accumulated"], torch.Tensor)
        assert isinstance(state_dict["value_estimates"], torch.Tensor)

        # Check nested dicts
        assert isinstance(state_dict["d1_pathway_state"], dict)
        assert isinstance(state_dict["exploration_manager_state"], dict)

    def test_from_dict_basic(self, sample_state: StriatumState, device: str):
        """Test from_dict() deserializes correctly."""
        state_dict = sample_state.to_dict()
        restored_state = StriatumState.from_dict(state_dict, device=device)

        # Check tensors restored
        assert torch.allclose(restored_state.spikes, sample_state.spikes)
        assert torch.allclose(restored_state.membrane, sample_state.membrane)
        assert torch.allclose(restored_state.d1_votes_accumulated, sample_state.d1_votes_accumulated)
        assert torch.allclose(restored_state.d2_votes_accumulated, sample_state.d2_votes_accumulated)
        assert torch.allclose(restored_state.value_estimates, sample_state.value_estimates)

        # Check scalars restored
        assert restored_state.last_action == sample_state.last_action
        assert restored_state.exploring == sample_state.exploring
        assert restored_state.last_uncertainty == sample_state.last_uncertainty
        assert restored_state.last_rpe == sample_state.last_rpe
        assert restored_state.dopamine == sample_state.dopamine

        # Check pointers restored
        assert restored_state.d1_delay_ptr == sample_state.d1_delay_ptr
        assert restored_state.d2_delay_ptr == sample_state.d2_delay_ptr

    def test_reset_clears_state(self, sample_state: StriatumState):
        """Test reset() clears all state tensors in-place."""
        sample_state.reset()

        # Check base state cleared
        assert sample_state.spikes is None
        assert sample_state.membrane is None

        # Check votes cleared (should be zero)
        assert torch.allclose(sample_state.d1_votes_accumulated, torch.zeros_like(sample_state.d1_votes_accumulated))
        assert torch.allclose(sample_state.d2_votes_accumulated, torch.zeros_like(sample_state.d2_votes_accumulated))

        # Check action selection cleared
        assert sample_state.last_action is None
        assert torch.allclose(sample_state.recent_spikes, torch.zeros_like(sample_state.recent_spikes))

        # Check exploration reset
        assert sample_state.exploring is False
        assert sample_state.last_uncertainty is None
        assert sample_state.last_exploration_prob is None

        # Check value/RPE cleared
        assert torch.allclose(sample_state.value_estimates, torch.zeros_like(sample_state.value_estimates))
        assert sample_state.last_rpe is None
        assert sample_state.last_expected is None

        # Check delay buffers cleared
        assert torch.allclose(sample_state.d1_delay_buffer, torch.zeros_like(sample_state.d1_delay_buffer))
        assert torch.allclose(sample_state.d2_delay_buffer, torch.zeros_like(sample_state.d2_delay_buffer))
        assert sample_state.d1_delay_ptr == 0
        assert sample_state.d2_delay_ptr == 0

        # Check homeostasis reset
        assert sample_state.activity_ema == 0.0
        assert sample_state.trial_spike_count == 0
        assert sample_state.trial_timesteps == 0
        assert sample_state.homeostatic_scaling_applied is False

        # Check neuromodulators reset
        assert sample_state.dopamine == 0.0
        assert sample_state.acetylcholine == 0.0
        assert sample_state.norepinephrine == 0.0

    def test_device_transfer_cpu_to_cpu(self, sample_state: StriatumState):
        """Test device transfer from CPU to CPU (no-op)."""
        state_dict = sample_state.to_dict()
        restored_state = StriatumState.from_dict(state_dict, device="cpu")

        assert restored_state.spikes.device.type == "cpu"
        assert restored_state.membrane.device.type == "cpu"
        assert restored_state.d1_votes_accumulated.device.type == "cpu"
        assert torch.allclose(restored_state.spikes, sample_state.spikes)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_transfer_cpu_to_cuda(self, sample_state: StriatumState):
        """Test device transfer from CPU to CUDA."""
        state_dict = sample_state.to_dict()
        restored_state = StriatumState.from_dict(state_dict, device="cuda")

        assert restored_state.spikes.device.type == "cuda"
        assert restored_state.membrane.device.type == "cuda"
        assert restored_state.d1_votes_accumulated.device.type == "cuda"
        assert torch.allclose(restored_state.spikes.cpu(), sample_state.spikes)

    def test_roundtrip_serialization(self, sample_state: StriatumState, device: str):
        """Test serialize → deserialize → serialize produces identical results."""
        dict1 = sample_state.to_dict()
        restored = StriatumState.from_dict(dict1, device=device)
        dict2 = restored.to_dict()

        # Check all tensor keys match
        tensor_keys = [
            "spikes", "membrane", "d1_votes_accumulated", "d2_votes_accumulated",
            "recent_spikes", "value_estimates", "pfc_modulation_d1", "pfc_modulation_d2",
            "d1_delay_buffer", "d2_delay_buffer"
        ]
        for key in tensor_keys:
            assert torch.allclose(dict1[key], dict2[key])

        # Check scalar keys match
        scalar_keys = [
            "last_action", "exploring", "last_uncertainty", "last_rpe",
            "dopamine", "d1_delay_ptr", "d2_delay_ptr", "activity_ema"
        ]
        for key in scalar_keys:
            assert dict1[key] == dict2[key]

    def test_optional_fields_none(self, device: str):
        """Test state works with optional fields set to None."""
        minimal_state = StriatumState(
            # Only base state
            spikes=None,
            membrane=None,

            # All optional fields None
            d1_pathway_state=None,
            d2_pathway_state=None,
            d1_votes_accumulated=torch.zeros(10, device=device),
            d2_votes_accumulated=torch.zeros(10, device=device),
            last_action=None,
            recent_spikes=torch.zeros(10, device=device),
            exploring=False,
            last_uncertainty=None,
            last_exploration_prob=None,
            exploration_manager_state=None,
            value_estimates=None,
            last_rpe=None,
            last_expected=None,
            pfc_modulation_d1=None,
            pfc_modulation_d2=None,
            d1_delay_buffer=None,
            d2_delay_buffer=None,
            d1_delay_ptr=0,
            d2_delay_ptr=0,
            activity_ema=0.0,
            trial_spike_count=0,
            trial_timesteps=0,
            homeostatic_scaling_applied=False,
            homeostasis_manager_state=None,
            dopamine=0.0,
            acetylcholine=0.0,
            norepinephrine=0.0,
        )

        # Should serialize and deserialize without errors
        state_dict = minimal_state.to_dict()
        restored = StriatumState.from_dict(state_dict, device=device)

        assert restored.spikes is None
        assert restored.value_estimates is None
        assert restored.pfc_modulation_d1 is None

    def test_delay_buffer_wraparound(self, device: str):
        """Test delay buffer circular pointer wraparound."""
        state = StriatumState(
            d1_delay_buffer=torch.rand(15, 10, device=device),
            d2_delay_buffer=torch.rand(25, 10, device=device),
            d1_delay_ptr=14,  # At end
            d2_delay_ptr=24,  # At end
        )

        state_dict = state.to_dict()
        restored = StriatumState.from_dict(state_dict, device=device)

        assert restored.d1_delay_ptr == 14
        assert restored.d2_delay_ptr == 24
        assert torch.allclose(restored.d1_delay_buffer, state.d1_delay_buffer)


# ============================================================================
# TEST INTEGRATION WITH STRIATUM REGION
# ============================================================================


class TestStriatumStateIntegration:
    """Test StriatumState integration with Striatum region."""

    def test_get_state_returns_valid_state(self, striatum_region: Striatum):
        """Test get_state() returns a valid StriatumState."""
        state = striatum_region.get_state()

        assert isinstance(state, StriatumState)
        assert state.d1_votes_accumulated is not None
        assert state.d2_votes_accumulated is not None
        assert state.recent_spikes is not None
        assert state.dopamine == 0.3  # Initial value (tonic dopamine from config)

    def test_load_state_restores_correctly(self, striatum_region: Striatum):
        """Test load_state() restores state correctly."""
        # Get initial state
        initial_state = striatum_region.get_state()

        # Modify some state values
        initial_state.d1_votes_accumulated.fill_(0.5)
        initial_state.d2_votes_accumulated.fill_(0.3)
        initial_state.last_action = 2
        initial_state.exploring = True
        initial_state.dopamine = 0.7

        # Load modified state
        striatum_region.load_state(initial_state)

        # Verify restoration
        restored_state = striatum_region.get_state()
        assert torch.allclose(restored_state.d1_votes_accumulated, initial_state.d1_votes_accumulated)
        assert torch.allclose(restored_state.d2_votes_accumulated, initial_state.d2_votes_accumulated)
        assert restored_state.last_action == initial_state.last_action
        assert restored_state.exploring == initial_state.exploring
        assert restored_state.dopamine == initial_state.dopamine

    def test_state_roundtrip_through_region(self, striatum_region: Striatum):
        """Test get_state() → load_state() preserves all state."""
        # Run forward pass to generate some activity
        input_spikes = torch.randint(0, 2, (striatum_region.config.n_input,),
                                     dtype=torch.float32, device=striatum_region.device)
        striatum_region(input_spikes)
        striatum_region.set_neuromodulators(dopamine=0.6)

        # Capture state
        state1 = striatum_region.get_state()

        # Run more activity
        striatum_region(input_spikes)

        # Load original state
        striatum_region.load_state(state1)

        # Verify restoration
        state2 = striatum_region.get_state()

        # Compare key tensors
        assert torch.allclose(state1.d1_votes_accumulated, state2.d1_votes_accumulated)
        assert torch.allclose(state1.d2_votes_accumulated, state2.d2_votes_accumulated)
        assert torch.allclose(state1.recent_spikes, state2.recent_spikes)
        assert state1.last_action == state2.last_action
        assert state1.dopamine == state2.dopamine

    def test_state_after_reset(self, striatum_region: Striatum):
        """Test state is cleared after region reset (not implemented yet)."""
        # Run forward pass
        input_spikes = torch.randint(0, 2, (striatum_region.config.n_input,),
                                     dtype=torch.float32, device=striatum_region.device)
        striatum_region(input_spikes)

        # Get state
        state = striatum_region.get_state()

        # Reset state object (not region)
        state.reset()

        # Verify state cleared
        if state.d1_votes_accumulated is not None:
            assert torch.allclose(state.d1_votes_accumulated, torch.zeros_like(state.d1_votes_accumulated))
        assert state.last_action is None
        assert state.exploring is False


# ============================================================================
# TEST FILE I/O
# ============================================================================


class TestStriatumStateFileIO:
    """Test saving and loading StriatumState to/from files."""

    def test_save_and_load_state(self, sample_state: StriatumState, tmp_path, device: str):
        """Test save_region_state() and load_region_state()."""
        checkpoint_path = tmp_path / "striatum_state.pt"

        # Save state
        save_region_state(sample_state, checkpoint_path)
        assert checkpoint_path.exists()

        # Load state
        loaded_state = load_region_state(StriatumState, checkpoint_path, device=device)

        # Verify restoration
        assert torch.allclose(loaded_state.spikes, sample_state.spikes)
        assert torch.allclose(loaded_state.d1_votes_accumulated, sample_state.d1_votes_accumulated)
        assert loaded_state.last_action == sample_state.last_action
        assert loaded_state.dopamine == sample_state.dopamine
        assert loaded_state.d1_delay_ptr == sample_state.d1_delay_ptr

    def test_save_and_load_region_state(self, striatum_region: Striatum, tmp_path):
        """Test save/load with real Striatum region."""
        checkpoint_path = tmp_path / "striatum_checkpoint.pt"

        # Run some activity
        input_spikes = torch.randint(0, 2, (striatum_region.config.n_input,),
                                     dtype=torch.float32, device=striatum_region.device)
        striatum_region(input_spikes)
        striatum_region.set_neuromodulators(dopamine=0.55)

        # Save state
        state1 = striatum_region.get_state()
        save_region_state(state1, checkpoint_path)

        # Load into new region
        new_config = StriatumConfig(
            n_input=striatum_region.config.n_input,
            n_output=striatum_region.config.n_output,
            population_coding=True,
            neurons_per_action=4,
            rpe_enabled=True,
            use_goal_conditioning=True,
            pfc_size=16,
            device=striatum_region.device,
            dt_ms=1.0,
        )
        new_region = Striatum(new_config)

        loaded_state = load_region_state(StriatumState, checkpoint_path, device=new_region.device)
        new_region.load_state(loaded_state)

        # Verify restoration
        state2 = new_region.get_state()
        assert torch.allclose(state1.d1_votes_accumulated, state2.d1_votes_accumulated)
        assert state1.dopamine == state2.dopamine


# ============================================================================
# TEST EDGE CASES
# ============================================================================


class TestStriatumStateEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_state_serialization(self, device: str):
        """Test serialization with all None/zero values."""
        empty_state = StriatumState(
            d1_votes_accumulated=torch.zeros(10, device=device),
            d2_votes_accumulated=torch.zeros(10, device=device),
            recent_spikes=torch.zeros(10, device=device),
        )

        state_dict = empty_state.to_dict()
        restored = StriatumState.from_dict(state_dict, device=device)

        assert restored.spikes is None
        assert restored.last_action is None
        assert torch.allclose(restored.d1_votes_accumulated, torch.zeros(10, device=device))

    def test_large_delay_buffers(self, device: str):
        """Test with large delay buffers (stress test)."""
        large_state = StriatumState(
            d1_votes_accumulated=torch.zeros(100, device=device),
            d2_votes_accumulated=torch.zeros(100, device=device),
            recent_spikes=torch.zeros(100, device=device),
            d1_delay_buffer=torch.rand(50, 100, device=device),  # 50ms delay
            d2_delay_buffer=torch.rand(100, 100, device=device),  # 100ms delay
            d1_delay_ptr=25,
            d2_delay_ptr=67,
        )

        # Should serialize/deserialize without memory issues
        state_dict = large_state.to_dict()
        restored = StriatumState.from_dict(state_dict, device=device)

        assert restored.d1_delay_buffer.shape == (50, 100)
        assert restored.d2_delay_buffer.shape == (100, 100)
        assert torch.allclose(restored.d1_delay_buffer, large_state.d1_delay_buffer)

    def test_partial_optional_features(self, device: str):
        """Test with some optional features enabled, others disabled."""
        partial_state = StriatumState(
            d1_votes_accumulated=torch.zeros(10, device=device),
            d2_votes_accumulated=torch.zeros(10, device=device),
            recent_spikes=torch.zeros(10, device=device),
            # RPE enabled
            value_estimates=torch.rand(5, device=device),
            last_rpe=0.42,
            # Goal conditioning disabled
            pfc_modulation_d1=None,
            pfc_modulation_d2=None,
            # Delays enabled
            d1_delay_buffer=torch.rand(15, 10, device=device),
            d2_delay_buffer=torch.rand(25, 10, device=device),
        )

        state_dict = partial_state.to_dict()
        restored = StriatumState.from_dict(state_dict, device=device)

        assert restored.value_estimates is not None
        assert restored.pfc_modulation_d1 is None
        assert restored.d1_delay_buffer is not None

    def test_mismatched_tensor_sizes(self, device: str):
        """Test that state can handle different tensor sizes (for growth)."""
        # State with different neuron counts (simulating growth)
        state_small = StriatumState(
            d1_votes_accumulated=torch.zeros(10, device=device),
            d2_votes_accumulated=torch.zeros(10, device=device),
            recent_spikes=torch.zeros(10, device=device),
        )

        state_large = StriatumState(
            d1_votes_accumulated=torch.zeros(20, device=device),
            d2_votes_accumulated=torch.zeros(20, device=device),
            recent_spikes=torch.zeros(20, device=device),
        )

        # Should serialize both without issues
        dict_small = state_small.to_dict()
        dict_large = state_large.to_dict()

        assert dict_small["d1_votes_accumulated"].shape[0] == 10
        assert dict_large["d1_votes_accumulated"].shape[0] == 20
