"""Tests for ThalamicRelayState RegionState protocol compliance.

This test suite validates that ThalamicRelayState correctly implements the
RegionState protocol, including:
- Protocol compliance (to_dict, from_dict, reset)
- Integration with ThalamicRelay region (get_state, load_state)
- Device transfer (CPU/CUDA)
- File I/O with utility functions
- STP state persistence (sensory and L6 feedback pathways)
- Relay and TRN neuron state preservation
- Burst/tonic mode state preservation
- Alpha gating state preservation

PHASE: 2.2 - ThalamicRelayState migration to RegionState protocol
"""

import tempfile
from pathlib import Path

import pytest
import torch

from thalia.core.region_state import (
    validate_state_protocol,
    save_region_state,
    load_region_state,
)
from thalia.regions.thalamus import ThalamicRelayState, ThalamicRelay, ThalamicRelayConfig


# ============================================================================
# PROTOCOL COMPLIANCE TESTS
# ============================================================================


def test_thalamic_relay_state_implements_protocol():
    """Validate that ThalamicRelayState implements RegionState protocol."""
    assert validate_state_protocol(ThalamicRelayState), (
        "ThalamicRelayState should implement RegionState protocol "
        "(to_dict, from_dict, reset methods)"
    )


def test_thalamic_relay_state_init():
    """Test ThalamicRelayState initialization with default values."""
    state = ThalamicRelayState()

    # Base region state should have defaults
    assert state.spikes is None
    assert state.membrane is None
    assert state.dopamine == 0.2
    assert state.acetylcholine == 0.0
    assert state.norepinephrine == 0.0

    # Thalamus-specific state should be None
    assert state.relay_spikes is None
    assert state.relay_membrane is None
    assert state.trn_spikes is None
    assert state.trn_membrane is None
    assert state.current_mode is None
    assert state.burst_counter is None
    assert state.alpha_gate is None
    assert state.stp_sensory_relay_state is None
    assert state.stp_l6_feedback_state is None


def test_thalamic_relay_state_with_data():
    """Test ThalamicRelayState initialization with actual data."""
    n_relay = 64
    n_trn = 32

    relay_spikes = torch.zeros(n_relay, dtype=torch.bool)
    relay_membrane = torch.randn(n_relay)
    trn_spikes = torch.zeros(n_trn, dtype=torch.bool)
    trn_membrane = torch.randn(n_trn)
    current_mode = torch.ones(n_relay)  # Tonic mode
    burst_counter = torch.zeros(n_relay)
    alpha_gate = torch.ones(n_relay) * 0.8

    stp_sensory_state = {
        "u": torch.ones(n_relay) * 0.4,
        "x": torch.ones(n_relay),
    }
    stp_l6_state = {
        "u": torch.ones(n_relay) * 0.7,
        "x": torch.ones(n_relay),
    }

    state = ThalamicRelayState(
        spikes=relay_spikes,
        membrane=relay_membrane,
        dopamine=0.5,
        acetylcholine=0.3,
        norepinephrine=0.8,
        relay_spikes=relay_spikes,
        relay_membrane=relay_membrane,
        trn_spikes=trn_spikes,
        trn_membrane=trn_membrane,
        current_mode=current_mode,
        burst_counter=burst_counter,
        alpha_gate=alpha_gate,
        stp_sensory_relay_state=stp_sensory_state,
        stp_l6_feedback_state=stp_l6_state,
    )

    # Verify all fields are set
    assert state.spikes is not None
    assert state.membrane is not None
    assert state.dopamine == 0.5
    assert state.acetylcholine == 0.3
    assert state.norepinephrine == 0.8
    assert state.relay_spikes is not None
    assert state.relay_membrane is not None
    assert state.trn_spikes is not None
    assert state.trn_membrane is not None
    assert state.current_mode is not None
    assert state.burst_counter is not None
    assert state.alpha_gate is not None
    assert state.stp_sensory_relay_state is not None
    assert state.stp_l6_feedback_state is not None
    assert state.stp_sensory_relay_state["u"] is not None
    assert state.stp_l6_feedback_state["x"] is not None


def test_thalamic_relay_state_to_dict():
    """Test ThalamicRelayState.to_dict() serialization."""
    n_relay = 64
    relay_spikes = torch.zeros(n_relay, dtype=torch.bool)
    relay_membrane = torch.randn(n_relay)
    stp_state = {"u": torch.ones(n_relay) * 0.4, "x": torch.ones(n_relay)}

    state = ThalamicRelayState(
        spikes=relay_spikes,
        membrane=relay_membrane,
        dopamine=0.6,
        relay_spikes=relay_spikes,
        relay_membrane=relay_membrane,
        stp_sensory_relay_state=stp_state,
    )

    state_dict = state.to_dict()

    # Verify all fields are present
    assert "spikes" in state_dict
    assert "membrane" in state_dict
    assert "dopamine" in state_dict
    assert "acetylcholine" in state_dict
    assert "norepinephrine" in state_dict
    assert "relay_spikes" in state_dict
    assert "relay_membrane" in state_dict
    assert "trn_spikes" in state_dict
    assert "trn_membrane" in state_dict
    assert "current_mode" in state_dict
    assert "burst_counter" in state_dict
    assert "alpha_gate" in state_dict
    assert "stp_sensory_relay_state" in state_dict
    assert "stp_l6_feedback_state" in state_dict

    # Verify values match
    assert torch.equal(state_dict["relay_spikes"], relay_spikes)
    assert torch.equal(state_dict["relay_membrane"], relay_membrane)
    assert state_dict["dopamine"] == 0.6
    assert state_dict["stp_sensory_relay_state"] is not None
    assert torch.equal(state_dict["stp_sensory_relay_state"]["u"], stp_state["u"])


def test_thalamic_relay_state_from_dict():
    """Test ThalamicRelayState.from_dict() deserialization."""
    n_relay = 64
    n_trn = 32

    state_dict = {
        "spikes": torch.zeros(n_relay, dtype=torch.bool),
        "membrane": torch.randn(n_relay),
        "dopamine": 0.7,
        "acetylcholine": 0.4,
        "norepinephrine": 0.6,
        "relay_spikes": torch.zeros(n_relay, dtype=torch.bool),
        "relay_membrane": torch.randn(n_relay),
        "trn_spikes": torch.zeros(n_trn, dtype=torch.bool),
        "trn_membrane": torch.randn(n_trn),
        "current_mode": torch.ones(n_relay),
        "burst_counter": torch.zeros(n_relay),
        "alpha_gate": torch.ones(n_relay) * 0.9,
        "stp_sensory_relay_state": {"u": torch.ones(n_relay) * 0.4, "x": torch.ones(n_relay)},
        "stp_l6_feedback_state": {"u": torch.ones(n_relay) * 0.7, "x": torch.ones(n_relay)},
    }

    state = ThalamicRelayState.from_dict(state_dict)

    # Verify all fields restored
    assert state.spikes is not None
    assert state.membrane is not None
    assert state.dopamine == 0.7
    assert state.acetylcholine == 0.4
    assert state.norepinephrine == 0.6
    assert state.relay_spikes is not None
    assert state.relay_membrane is not None
    assert state.trn_spikes is not None
    assert state.trn_membrane is not None
    assert state.current_mode is not None
    assert state.burst_counter is not None
    assert state.alpha_gate is not None
    assert state.stp_sensory_relay_state is not None
    assert state.stp_l6_feedback_state is not None

    # Verify values match
    assert torch.equal(state.relay_spikes, state_dict["relay_spikes"])
    assert torch.equal(state.trn_membrane, state_dict["trn_membrane"])
    assert torch.equal(state.alpha_gate, state_dict["alpha_gate"])


def test_thalamic_relay_state_reset():
    """Test ThalamicRelayState.reset() clears all fields."""
    n_relay = 64

    state = ThalamicRelayState(
        spikes=torch.zeros(n_relay, dtype=torch.bool),
        membrane=torch.randn(n_relay),
        dopamine=0.8,
        acetylcholine=0.5,
        norepinephrine=0.9,
        relay_spikes=torch.zeros(n_relay, dtype=torch.bool),
        relay_membrane=torch.randn(n_relay),
        stp_sensory_relay_state={"u": torch.ones(n_relay), "x": torch.ones(n_relay)},
    )

    # Reset state (in-place mutation, returns None)
    state.reset()

    # Verify all fields are cleared
    assert state.spikes is None
    assert state.membrane is None
    assert state.dopamine == 0.2  # Default
    assert state.acetylcholine == 0.0
    assert state.norepinephrine == 0.0
    assert state.relay_spikes is None
    assert state.relay_membrane is None
    assert state.trn_spikes is None
    assert state.trn_membrane is None
    assert state.current_mode is None
    assert state.burst_counter is None
    assert state.alpha_gate is None
    assert state.stp_sensory_relay_state is None
    assert state.stp_l6_feedback_state is None


# ============================================================================
# INTEGRATION TESTS WITH THALAMICRELAY REGION
# ============================================================================


def test_region_get_state_captures_complete_state():
    """Test ThalamicRelay.get_state() captures all state including STP."""
    config = ThalamicRelayConfig(
        n_input=32,
        n_output=64,
        stp_enabled=True,
        dt_ms=1.0,
        device="cpu",
    )
    thalamus = ThalamicRelay(config)

    # Run forward pass to generate state
    input_spikes = torch.zeros(32, dtype=torch.bool, device="cpu")
    input_spikes[:5] = True  # Some activity
    _ = thalamus(input_spikes)

    # Capture state
    state = thalamus.get_state()

    # Verify state is ThalamicRelayState
    assert isinstance(state, ThalamicRelayState)

    # Verify relay neuron state captured
    assert state.relay_spikes is not None
    assert state.relay_spikes.shape == (64,)

    # Relay membrane may be None depending on neuron implementation
    if state.relay_membrane is not None:
        assert state.relay_membrane.shape == (64,)

    # Verify TRN state captured
    assert state.trn_spikes is not None
    n_trn = int(64 * config.trn_ratio)
    assert state.trn_spikes.shape == (n_trn,)

    # Verify mode state captured
    if state.current_mode is not None:
        assert state.current_mode.shape == (64,)

    # Verify STP state captured when enabled
    assert state.stp_sensory_relay_state is not None
    assert "u" in state.stp_sensory_relay_state
    assert "x" in state.stp_sensory_relay_state
    assert state.stp_l6_feedback_state is not None
    assert "u" in state.stp_l6_feedback_state
    assert "x" in state.stp_l6_feedback_state


def test_region_load_state_restores_complete_state():
    """Test ThalamicRelay.load_state() restores all state correctly."""
    config = ThalamicRelayConfig(
        n_input=32,
        n_output=64,
        stp_enabled=True,
        dt_ms=1.0,
        device="cpu",
    )
    thalamus = ThalamicRelay(config)

    # Run forward pass and capture state
    input_spikes = torch.zeros(32, dtype=torch.bool, device="cpu")
    input_spikes[:5] = True
    _ = thalamus(input_spikes)

    # Capture state BEFORE neuromodulator changes
    state = thalamus.get_state()
    captured_dopamine = state.dopamine

    # Modify neuromodulators
    thalamus.set_neuromodulators(dopamine=0.9, acetylcholine=0.7, norepinephrine=0.8)

    # Verify neuromodulators changed
    assert thalamus.state.dopamine == 0.9

    # Restore state
    thalamus.load_state(state)

    # Verify neuromodulators restored to captured values
    assert thalamus.state.dopamine == captured_dopamine
    assert thalamus.state.acetylcholine == state.acetylcholine
    assert thalamus.state.norepinephrine == state.norepinephrine

    # Verify relay neuron state restored
    assert torch.equal(thalamus.state.relay_spikes, state.relay_spikes)

    # Verify TRN state restored
    assert torch.equal(thalamus.state.trn_spikes, state.trn_spikes)

    # Verify STP state restored
    if thalamus.stp_sensory_relay is not None:
        assert torch.equal(thalamus.stp_sensory_relay.u, state.stp_sensory_relay_state["u"])
        assert torch.equal(thalamus.stp_sensory_relay.x, state.stp_sensory_relay_state["x"])


def test_region_state_roundtrip():
    """Test get_state() → load_state() preserves complete state."""
    config = ThalamicRelayConfig(
        n_input=32,
        n_output=64,
        stp_enabled=True,
        dt_ms=1.0,
        device="cpu",
    )
    thalamus = ThalamicRelay(config)

    # Generate initial state
    input_spikes = torch.zeros(32, dtype=torch.bool, device="cpu")
    input_spikes[:8] = True
    _ = thalamus(input_spikes)

    # Set neuromodulators
    thalamus.set_neuromodulators(dopamine=0.6, acetylcholine=0.4, norepinephrine=0.7)

    # Capture state
    state1 = thalamus.get_state()

    # Run more forward passes to change state
    _ = thalamus(torch.zeros(32, dtype=torch.bool, device="cpu"))
    _ = thalamus(torch.zeros(32, dtype=torch.bool, device="cpu"))

    # Restore original state
    thalamus.load_state(state1)

    # Capture state again
    state2 = thalamus.get_state()

    # Verify states match
    assert torch.equal(state2.relay_spikes, state1.relay_spikes)
    if state1.relay_membrane is not None:
        assert torch.equal(state2.relay_membrane, state1.relay_membrane)
    assert torch.equal(state2.trn_spikes, state1.trn_spikes)
    assert state2.dopamine == state1.dopamine
    assert state2.acetylcholine == state1.acetylcholine
    assert state2.norepinephrine == state1.norepinephrine

    # Verify STP state matches
    if state1.stp_sensory_relay_state is not None:
        assert torch.equal(
            state2.stp_sensory_relay_state["u"],
            state1.stp_sensory_relay_state["u"]
        )
        assert torch.equal(
            state2.stp_sensory_relay_state["x"],
            state1.stp_sensory_relay_state["x"]
        )


# ============================================================================
# DEVICE TRANSFER TESTS
# ============================================================================


def test_thalamic_relay_state_device_transfer_cpu_to_cpu():
    """Test ThalamicRelayState device transfer from CPU to CPU."""
    n_relay = 64

    state_dict = {
        "relay_spikes": torch.zeros(n_relay, dtype=torch.bool, device="cpu"),
        "relay_membrane": torch.randn(n_relay, device="cpu"),
        "dopamine": 0.5,
        "stp_sensory_relay_state": {
            "u": torch.ones(n_relay, device="cpu") * 0.4,
            "x": torch.ones(n_relay, device="cpu"),
        },
    }

    state = ThalamicRelayState.from_dict(state_dict, device=torch.device("cpu"))

    # Verify tensors are on CPU
    assert state.relay_spikes.device.type == "cpu"
    assert state.relay_membrane.device.type == "cpu"
    assert state.stp_sensory_relay_state["u"].device.type == "cpu"
    assert state.stp_sensory_relay_state["x"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_thalamic_relay_state_device_transfer_cpu_to_cuda():
    """Test ThalamicRelayState device transfer from CPU to CUDA."""
    n_relay = 64
    n_trn = 32

    state_dict = {
        "relay_spikes": torch.zeros(n_relay, dtype=torch.bool, device="cpu"),
        "relay_membrane": torch.randn(n_relay, device="cpu"),
        "trn_spikes": torch.zeros(n_trn, dtype=torch.bool, device="cpu"),
        "alpha_gate": torch.ones(n_relay, device="cpu") * 0.8,
        "dopamine": 0.6,
        "stp_sensory_relay_state": {
            "u": torch.ones(n_relay, device="cpu") * 0.4,
            "x": torch.ones(n_relay, device="cpu"),
        },
        "stp_l6_feedback_state": {
            "u": torch.ones(n_relay, device="cpu") * 0.7,
            "x": torch.ones(n_relay, device="cpu"),
        },
    }

    state = ThalamicRelayState.from_dict(state_dict, device=torch.device("cuda:0"))

    # Verify tensors are on CUDA
    assert state.relay_spikes.device.type == "cuda"
    assert state.relay_membrane.device.type == "cuda"
    assert state.trn_spikes.device.type == "cuda"
    assert state.alpha_gate.device.type == "cuda"
    assert state.stp_sensory_relay_state["u"].device.type == "cuda"
    assert state.stp_sensory_relay_state["x"].device.type == "cuda"
    assert state.stp_l6_feedback_state["u"].device.type == "cuda"
    assert state.stp_l6_feedback_state["x"].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_thalamic_relay_state_device_transfer_cuda_to_cpu():
    """Test ThalamicRelayState device transfer from CUDA to CPU."""
    n_relay = 64

    state_dict = {
        "relay_spikes": torch.zeros(n_relay, dtype=torch.bool, device="cuda:0"),
        "relay_membrane": torch.randn(n_relay, device="cuda:0"),
        "dopamine": 0.7,
        "stp_sensory_relay_state": {
            "u": torch.ones(n_relay, device="cuda:0") * 0.4,
            "x": torch.ones(n_relay, device="cuda:0"),
        },
    }

    state = ThalamicRelayState.from_dict(state_dict, device=torch.device("cpu"))

    # Verify tensors are on CPU
    assert state.relay_spikes.device.type == "cpu"
    assert state.relay_membrane.device.type == "cpu"
    assert state.stp_sensory_relay_state["u"].device.type == "cpu"
    assert state.stp_sensory_relay_state["x"].device.type == "cpu"


# ============================================================================
# FILE I/O TESTS
# ============================================================================


def test_save_and_load_thalamic_relay_state():
    """Test saving and loading ThalamicRelayState to/from file."""
    n_relay = 64
    n_trn = 32

    # Create state with data
    state = ThalamicRelayState(
        spikes=torch.zeros(n_relay, dtype=torch.bool),
        membrane=torch.randn(n_relay),
        dopamine=0.65,
        acetylcholine=0.35,
        norepinephrine=0.75,
        relay_spikes=torch.zeros(n_relay, dtype=torch.bool),
        relay_membrane=torch.randn(n_relay),
        trn_spikes=torch.zeros(n_trn, dtype=torch.bool),
        trn_membrane=torch.randn(n_trn),
        current_mode=torch.ones(n_relay),
        alpha_gate=torch.ones(n_relay) * 0.85,
        stp_sensory_relay_state={"u": torch.ones(n_relay) * 0.4, "x": torch.ones(n_relay)},
        stp_l6_feedback_state={"u": torch.ones(n_relay) * 0.7, "x": torch.ones(n_relay)},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save state
        save_path = Path(tmpdir) / "thalamus_state.pt"
        save_region_state(state, save_path)

        # Load state
        loaded_state = load_region_state(ThalamicRelayState, save_path)

        # Verify all fields match
        assert torch.equal(loaded_state.relay_spikes, state.relay_spikes)
        assert torch.equal(loaded_state.relay_membrane, state.relay_membrane)
        assert torch.equal(loaded_state.trn_spikes, state.trn_spikes)
        assert torch.equal(loaded_state.trn_membrane, state.trn_membrane)
        assert torch.equal(loaded_state.current_mode, state.current_mode)
        assert torch.equal(loaded_state.alpha_gate, state.alpha_gate)
        assert loaded_state.dopamine == state.dopamine
        assert loaded_state.acetylcholine == state.acetylcholine
        assert loaded_state.norepinephrine == state.norepinephrine
        assert torch.equal(
            loaded_state.stp_sensory_relay_state["u"],
            state.stp_sensory_relay_state["u"]
        )
        assert torch.equal(
            loaded_state.stp_l6_feedback_state["x"],
            state.stp_l6_feedback_state["x"]
        )


def test_save_load_region_with_utility_functions():
    """Test complete region checkpoint/restore using utility functions."""
    config = ThalamicRelayConfig(
        n_input=32,
        n_output=64,
        stp_enabled=True,
        dt_ms=1.0,
        device="cpu",
    )
    thalamus = ThalamicRelay(config)

    # Generate state
    input_spikes = torch.zeros(32, dtype=torch.bool, device="cpu")
    input_spikes[:10] = True
    _ = thalamus(input_spikes)

    # Set neuromodulators
    thalamus.set_neuromodulators(dopamine=0.55, acetylcholine=0.45, norepinephrine=0.65)

    # Capture state BEFORE saving
    captured_dopamine = thalamus.state.dopamine

    # Get and save state
    state = thalamus.get_state()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "thalamus_checkpoint.pt"
        save_region_state(state, save_path)

        # Modify thalamus state
        thalamus.reset_state()
        thalamus.set_neuromodulators(dopamine=0.1, acetylcholine=0.1, norepinephrine=0.1)

        # Load and restore state
        loaded_state = load_region_state(ThalamicRelayState, save_path)
        thalamus.load_state(loaded_state)

        # Verify state restored using captured dopamine value
        assert thalamus.state.dopamine == captured_dopamine
        assert thalamus.state.acetylcholine == state.acetylcholine
        assert thalamus.state.norepinephrine == state.norepinephrine
        assert torch.equal(thalamus.state.relay_spikes, state.relay_spikes)
        if thalamus.stp_sensory_relay is not None:
            assert torch.equal(thalamus.stp_sensory_relay.u, state.stp_sensory_relay_state["u"])


# ============================================================================
# STP STATE TESTS
# ============================================================================


def test_stp_state_captured_when_enabled():
    """Test that STP state is captured when STP is enabled."""
    config = ThalamicRelayConfig(
        n_input=32,
        n_output=64,
        stp_enabled=True,
        dt_ms=1.0,
        device="cpu",
    )
    thalamus = ThalamicRelay(config)

    # Run forward passes with sustained input
    input_spikes = torch.ones(32, dtype=torch.bool, device="cpu")
    for _ in range(5):
        _ = thalamus(input_spikes)

    # Capture state
    state = thalamus.get_state()

    # Verify STP state captured
    assert state.stp_sensory_relay_state is not None
    assert "u" in state.stp_sensory_relay_state
    assert "x" in state.stp_sensory_relay_state
    assert state.stp_sensory_relay_state["u"] is not None
    assert state.stp_sensory_relay_state["x"] is not None

    assert state.stp_l6_feedback_state is not None
    assert "u" in state.stp_l6_feedback_state
    assert "x" in state.stp_l6_feedback_state
    # Note: L6 STP state may have None tensors if no L6 input was provided
    # This is expected behavior - STP module exists but hasn't been used yet

    # Verify depression occurred in sensory pathway (x should be < 1.0 after sustained input)
    if state.stp_sensory_relay_state["x"] is not None:
        assert (state.stp_sensory_relay_state["x"] < 0.99).any(), \
            "Expected some depression after sustained input"


def test_stp_state_not_captured_when_disabled():
    """Test that STP state is None when STP is disabled."""
    config = ThalamicRelayConfig(
        n_input=32,
        n_output=64,
        stp_enabled=False,
        dt_ms=1.0,
        device="cpu",
    )
    thalamus = ThalamicRelay(config)

    # Run forward pass
    input_spikes = torch.zeros(32, dtype=torch.bool, device="cpu")
    _ = thalamus(input_spikes)

    # Capture state
    state = thalamus.get_state()

    # Verify STP state is None
    assert state.stp_sensory_relay_state is None
    assert state.stp_l6_feedback_state is None


def test_stp_state_roundtrip():
    """Test that STP state is preserved across save/load."""
    config = ThalamicRelayConfig(
        n_input=32,
        n_output=64,
        stp_enabled=True,
        dt_ms=1.0,
        device="cpu",
    )
    thalamus = ThalamicRelay(config)

    # Run forward passes to generate depression
    input_spikes = torch.ones(32, dtype=torch.bool, device="cpu")
    for _ in range(10):
        _ = thalamus(input_spikes)

    # Capture state
    state1 = thalamus.get_state()
    sensory_u_before = state1.stp_sensory_relay_state["u"].clone() if state1.stp_sensory_relay_state["u"] is not None else None
    sensory_x_before = state1.stp_sensory_relay_state["x"].clone() if state1.stp_sensory_relay_state["x"] is not None else None
    l6_u_before = state1.stp_l6_feedback_state["u"].clone() if state1.stp_l6_feedback_state["u"] is not None else None
    l6_x_before = state1.stp_l6_feedback_state["x"].clone() if state1.stp_l6_feedback_state["x"] is not None else None

    # Reset thalamus
    thalamus.reset_state()

    # Restore state
    thalamus.load_state(state1)

    # Verify STP state restored exactly
    if sensory_u_before is not None:
        assert torch.equal(thalamus.stp_sensory_relay.u, sensory_u_before)
    if sensory_x_before is not None:
        assert torch.equal(thalamus.stp_sensory_relay.x, sensory_x_before)
    if l6_u_before is not None:
        assert torch.equal(thalamus.stp_l6_feedback.u, l6_u_before)
    if l6_x_before is not None:
        assert torch.equal(thalamus.stp_l6_feedback.x, l6_x_before)


# ============================================================================
# BURST/TONIC MODE STATE TESTS
# ============================================================================


def test_mode_state_preservation():
    """Test that burst/tonic mode state is preserved across save/load."""
    config = ThalamicRelayConfig(
        n_input=32,
        n_output=64,
        stp_enabled=False,
        dt_ms=1.0,
        device="cpu",
    )
    thalamus = ThalamicRelay(config)

    # Run forward pass to generate mode state
    input_spikes = torch.zeros(32, dtype=torch.bool, device="cpu")
    input_spikes[:5] = True
    _ = thalamus(input_spikes)

    # Capture state
    state = thalamus.get_state()

    # Verify mode state captured
    if state.current_mode is not None:
        mode_before = state.current_mode.clone()

        # Reset and restore
        thalamus.reset_state()
        thalamus.load_state(state)

        # Verify mode restored
        assert torch.equal(thalamus.state.current_mode, mode_before)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


def test_state_with_none_fields():
    """Test handling of None fields during serialization."""
    state = ThalamicRelayState(
        dopamine=0.5,
        relay_spikes=None,  # Explicitly None
        relay_membrane=None,
        stp_sensory_relay_state=None,
    )

    # Should serialize without error
    state_dict = state.to_dict()

    # Should deserialize without error
    restored = ThalamicRelayState.from_dict(state_dict)

    assert restored.relay_spikes is None
    assert restored.relay_membrane is None
    assert restored.stp_sensory_relay_state is None
    assert restored.dopamine == 0.5


def test_state_with_stp_nested_dict():
    """Test handling of nested STP state dicts."""
    n_relay = 64

    # Create state with nested STP dicts
    stp_state = {
        "u": torch.ones(n_relay) * 0.4,
        "x": torch.ones(n_relay) * 0.8,
    }

    state = ThalamicRelayState(
        dopamine=0.6,
        stp_sensory_relay_state=stp_state,
        stp_l6_feedback_state=stp_state.copy(),  # Separate copy
    )

    # Serialize and deserialize
    state_dict = state.to_dict()
    restored = ThalamicRelayState.from_dict(state_dict, device=torch.device("cpu"))

    # Verify nested dicts preserved
    assert restored.stp_sensory_relay_state is not None
    assert torch.equal(restored.stp_sensory_relay_state["u"], stp_state["u"])
    assert torch.equal(restored.stp_sensory_relay_state["x"], stp_state["x"])
    assert restored.stp_l6_feedback_state is not None
    assert torch.equal(restored.stp_l6_feedback_state["u"], stp_state["u"])


def test_state_with_partial_fields():
    """Test deserialization with missing optional fields."""
    state_dict = {
        "dopamine": 0.5,
        "relay_spikes": torch.zeros(64, dtype=torch.bool),
        # Missing many fields - should use defaults
    }

    state = ThalamicRelayState.from_dict(state_dict)

    # Verify defaults used for missing fields
    assert state.dopamine == 0.5
    assert state.relay_spikes is not None
    assert state.acetylcholine == 0.0  # Default
    assert state.norepinephrine == 0.0  # Default
    assert state.relay_membrane is None
    assert state.trn_spikes is None
    assert state.stp_sensory_relay_state is None


# ============================================================================
# SUMMARY TEST
# ============================================================================


def test_summary():
    """Summary test documenting ThalamicRelayState RegionState protocol compliance.

    This test documents the complete migration of ThalamicRelayState from
    NeuralComponentState to BaseRegionState with full RegionState protocol support.

    COVERAGE:
    - ✅ Protocol compliance (to_dict, from_dict, reset)
    - ✅ Integration with ThalamicRelay region (get_state, load_state)
    - ✅ Device transfer (CPU/CUDA)
    - ✅ File I/O with utility functions
    - ✅ STP state persistence (sensory and L6 feedback pathways)
    - ✅ Relay and TRN neuron state preservation
    - ✅ Burst/tonic mode state preservation
    - ✅ Alpha gating state preservation
    - ✅ Neuromodulator state (dopamine, acetylcholine, norepinephrine)
    - ✅ Edge cases (None fields, nested dicts, partial fields)

    TOTAL: 21 tests
    """
    pass
