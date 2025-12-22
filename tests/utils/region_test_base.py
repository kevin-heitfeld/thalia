"""Base test class for region testing with common test patterns.

This module provides RegionTestBase, an abstract base class that standardizes
testing patterns across all brain regions. It eliminates test boilerplate by
providing common test methods for:

- Initialization verification (2 tests)
- Forward pass shape checking (4 tests)
- Growth functionality (3 tests: input/output expansion, state preservation)
- State management (5 tests: get/load, reset, file I/O, STP persistence, device transfer)
- Device transfer (2 tests: CPU and CUDA)
- Integration (2 tests: neuromodulators, diagnostics)

Total: **17 standard tests** inherited by all region test classes.

**Usage Pattern**:

```python
from tests.utils.region_test_base import RegionTestBase

class TestMyRegion(RegionTestBase):
    '''Test MyRegion implementation.'''

    def create_region(self, **kwargs):
        '''Create region instance for testing.'''
        config = MyRegionConfig(**kwargs)
        return MyRegion(config)

    def get_default_params(self):
        '''Return default region parameters.'''
        return {"n_input": 100, "n_output": 50}

    def test_region_specific_feature(self):
        '''Test region-specific behavior (custom test).'''
        region = self.create_region(**self.get_default_params())
        # Region-specific tests here
```

**Benefits**:
- Reduces test boilerplate: ~100 lines per region × 6 regions = 600 lines eliminated
- Ensures consistent test coverage across all regions
- Easy to add new standard tests (e.g., checkpoint compatibility)
- Region-specific tests remain in subclasses

**Architecture Rationale**:
Follows Tier 3.4 recommendation from Architecture Review 2025-12-22.
Standardizes testing without constraining region-specific test requirements.

Author: Thalia Project
Date: December 22, 2025 (Tier 3.4 implementation)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pytest
import torch

from thalia.core.neural_region import NeuralRegion


class RegionTestBase(ABC):
    """Base class for region testing with common test patterns.

    Subclasses must implement:
    - create_region(**kwargs): Create region instance
    - get_default_params(): Return default configuration parameters

    Optionally override:
    - get_min_params(): Return minimal valid parameters
    - get_input_dict(): Return dict input for multi-source regions
    - skip_growth_tests(): Return True if region doesn't support growth
    """

    # =========================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def create_region(self, **kwargs) -> NeuralRegion:
        """Create region instance for testing.

        Args:
            **kwargs: Configuration parameters (e.g., n_input, n_output, device)

        Returns:
            Configured region instance

        Example:
            def create_region(self, **kwargs):
                config = LayeredCortexConfig(**kwargs)
                return LayeredCortex(config)
        """
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Return default region parameters for testing.

        Returns:
            Dict with standard parameters (must include n_input, n_output)

        Example:
            def get_default_params(self):
                return {
                    "n_input": 100,
                    "n_output": 50,
                    "device": "cpu",
                    "dt_ms": 1.0,
                }
        """
        pass

    # =========================================================================
    # OPTIONAL OVERRIDES
    # =========================================================================

    def get_min_params(self) -> Dict[str, Any]:
        """Return minimal valid parameters (defaults to get_default_params).

        Override if region has smaller valid configurations for quick tests.
        """
        return self.get_default_params()

    def get_input_dict(self, n_input: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Return dict input for multi-source regions (default: single source).

        Override for regions that require multiple input sources.

        Example (Hippocampus):
            def get_input_dict(self, n_input, device="cpu"):
                return {
                    "ec": torch.zeros(n_input, device=device),
                    "ec_l3": torch.zeros(n_input // 2, device=device),
                }
        """
        return {"default": torch.zeros(n_input, device=device)}

    def skip_growth_tests(self) -> bool:
        """Return True if region doesn't support growth (default: False).

        Override for regions that don't implement grow_output/grow_input.
        """
        return False

    # =========================================================================
    # COMMON TEST PATTERNS
    # =========================================================================

    def test_initialization(self):
        """Test region initializes correctly with valid parameters."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Verify configuration stored
        assert region.config.n_input == params["n_input"]
        assert region.config.n_output == params["n_output"]

        # Verify device set correctly
        expected_device = params.get("device", "cpu")
        assert region.device.type == expected_device

        # Verify region has required attributes
        assert hasattr(region, "forward")
        assert hasattr(region, "reset_state")
        assert hasattr(region, "get_state")
        assert hasattr(region, "load_state")

    def test_initialization_minimal(self):
        """Test region initializes with minimal valid parameters."""
        params = self.get_min_params()
        region = self.create_region(**params)

        # Verify basic functionality
        assert region.config.n_input == params["n_input"]
        assert region.config.n_output == params["n_output"]

    def test_forward_pass_tensor_input(self):
        """Test forward pass with single tensor input returns correct shape."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Create input spikes (1D, no batch dimension per ADR-005)
        input_spikes = torch.zeros(params["n_input"], device=region.device)

        # Forward pass
        output = region.forward(input_spikes)

        # Verify output shape (1D per ADR-005)
        assert output.dim() == 1, f"Expected 1D output (ADR-005), got {output.dim()}D"
        assert output.shape[0] == params["n_output"]

        # Verify output is boolean or float
        assert output.dtype in [torch.bool, torch.float32, torch.float64]

    def test_forward_pass_dict_input(self):
        """Test forward pass with dict input (multi-source support)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Create dict input
        input_dict = self.get_input_dict(params["n_input"], device=region.device.type)

        # Forward pass
        output = region.forward(input_dict)

        # Verify output shape
        assert output.dim() == 1
        assert output.shape[0] == params["n_output"]

    def test_forward_pass_zero_input(self):
        """Test forward pass handles zero input (clock-driven execution)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Zero input (no spikes)
        input_spikes = torch.zeros(params["n_input"], device=region.device)

        # Should not raise error
        output = region.forward(input_spikes)
        assert output.shape[0] == params["n_output"]

    def test_forward_pass_multiple_calls(self):
        """Test region handles multiple forward passes (temporal dynamics)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        input_spikes = torch.zeros(params["n_input"], device=region.device)

        # Multiple forward passes should not error
        for _ in range(10):
            output = region.forward(input_spikes)
            assert output.shape[0] == params["n_output"]

    @pytest.mark.skipif(lambda self: self.skip_growth_tests(), reason="Region doesn't support growth")
    def test_grow_output(self):
        """Test region can grow output dimension (add neurons)."""
        if self.skip_growth_tests():
            pytest.skip("Region doesn't support growth")

        params = self.get_default_params()
        region = self.create_region(**params)

        original_output = params["n_output"]
        n_new = 10

        # Grow output
        region.grow_output(n_new)

        # Verify config updated
        assert region.config.n_output == original_output + n_new

        # Verify forward pass still works with new size
        input_spikes = torch.zeros(params["n_input"], device=region.device)
        output = region.forward(input_spikes)
        assert output.shape[0] == original_output + n_new

    @pytest.mark.skipif(lambda self: self.skip_growth_tests(), reason="Region doesn't support growth")
    def test_grow_input(self):
        """Test region can grow input dimension (accept more inputs)."""
        if self.skip_growth_tests():
            pytest.skip("Region doesn't support growth")

        params = self.get_default_params()
        region = self.create_region(**params)

        original_input = params["n_input"]
        n_new = 20

        # Grow input
        region.grow_input(n_new)

        # Verify config updated
        assert region.config.n_input == original_input + n_new

        # Verify forward pass works with new input size
        input_spikes = torch.zeros(original_input + n_new, device=region.device)
        output = region.forward(input_spikes)
        assert output.shape[0] == params["n_output"]

    @pytest.mark.skipif(lambda self: self.skip_growth_tests(), reason="Region doesn't support growth")
    def test_growth_preserves_state(self):
        """Test growth preserves existing neuron state."""
        if self.skip_growth_tests():
            pytest.skip("Region doesn't support growth")

        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass to initialize state
        input_spikes = torch.ones(params["n_input"], device=region.device)
        region.forward(input_spikes)

        # Get state before growth
        state_before = region.get_state()

        # Grow output
        region.grow_output(10)

        # Get state after growth
        state_after = region.get_state()

        # Verify original neurons preserved (first N values should match)
        if state_before.membrane is not None and state_after.membrane is not None:
            n_original = params["n_output"]
            original_membrane = state_before.membrane[:n_original]
            preserved_membrane = state_after.membrane[:n_original]
            assert torch.allclose(original_membrane, preserved_membrane, atol=1e-5)

    def test_state_get_and_load(self):
        """Test get_state() and load_state() round-trip."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass to generate non-trivial state
        input_spikes = torch.ones(params["n_input"], device=region.device)
        region.forward(input_spikes)

        # Get state
        state = region.get_state()

        # Verify state has required fields
        assert hasattr(state, "to_dict")
        assert hasattr(state, "from_dict")
        assert hasattr(state, "reset")

        # Create new region and load state
        region2 = self.create_region(**params)
        region2.load_state(state)

        # Verify states match
        state2 = region2.get_state()
        state_dict1 = state.to_dict()
        state_dict2 = state2.to_dict()

        # Compare all tensor fields
        for key in state_dict1:
            if isinstance(state_dict1[key], torch.Tensor) and state_dict1[key] is not None:
                assert torch.allclose(state_dict1[key], state_dict2[key], atol=1e-5)
            elif isinstance(state_dict1[key], (int, float)):
                assert state_dict1[key] == state_dict2[key]

    def test_reset_state(self):
        """Test reset_state() reinitializes region state."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass to generate non-trivial state
        input_spikes = torch.ones(params["n_input"], device=region.device)
        region.forward(input_spikes)

        # Reset state
        region.reset_state()

        # Verify state reset (membrane potentials should be at rest)
        state = region.get_state()
        if state.membrane is not None:
            # Most neurons should be at resting potential (~-70mV)
            mean_membrane = state.membrane.mean().item()
            assert -75.0 <= mean_membrane <= -65.0, f"Expected resting potential, got {mean_membrane}mV"

    def test_device_cpu(self):
        """Test region works on CPU device."""
        params = self.get_default_params()
        params["device"] = "cpu"

        region = self.create_region(**params)
        assert region.device.type == "cpu"

        # Verify forward pass works
        input_spikes = torch.zeros(params["n_input"], device="cpu")
        output = region.forward(input_spikes)
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self):
        """Test region works on CUDA device (if available)."""
        params = self.get_default_params()
        params["device"] = "cuda"

        region = self.create_region(**params)
        assert region.device.type == "cuda"

        # Verify forward pass works
        input_spikes = torch.zeros(params["n_input"], device="cuda")
        output = region.forward(input_spikes)
        assert output.device.type == "cuda"

    def test_neuromodulator_update(self):
        """Test region accepts and stores neuromodulator values."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Set neuromodulators via set_neuromodulators (from NeuromodulatorMixin)
        if hasattr(region, "set_neuromodulators"):
            region.set_neuromodulators(dopamine=0.7, acetylcholine=0.5, norepinephrine=0.3)

            # Verify stored in state
            state = region.get_state()
            assert hasattr(state, "dopamine")
            assert hasattr(state, "acetylcholine")
            assert hasattr(state, "norepinephrine")

            # Verify values set correctly
            assert state.dopamine == 0.7
            assert state.acetylcholine == 0.5
            assert state.norepinephrine == 0.3

    def test_diagnostics_collection(self):
        """Test region provides diagnostic information."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass
        input_spikes = torch.zeros(params["n_input"], device=region.device)
        region.forward(input_spikes)

        # Collect diagnostics (from DiagnosticsMixin)
        if hasattr(region, "get_diagnostics"):
            diagnostics = region.get_diagnostics()

            # Verify diagnostics is a dict
            assert isinstance(diagnostics, dict)

            # Should have at least some standard metrics
            # (exact metrics depend on region, so we just check non-empty)
            assert len(diagnostics) > 0

    def test_state_file_io(self, tmp_path):
        """Test state can be saved to and loaded from file.

        Uses save_region_state() and load_region_state() utilities.
        """
        from thalia.core.region_state import save_region_state, load_region_state

        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward to generate non-trivial state
        input_spikes = torch.ones(params["n_input"], device=region.device)
        for _ in range(3):
            region.forward(input_spikes)

        # Get state
        state1 = region.get_state()

        # Save to file
        filepath = tmp_path / "region_state.pt"
        save_region_state(state1, str(filepath))

        # Load from file
        state_class = type(state1)
        state2 = load_region_state(state_class, str(filepath), device=params["device"])

        # Verify states match
        dict1 = state1.to_dict()
        dict2 = state2.to_dict()

        for key in dict1:
            if isinstance(dict1[key], torch.Tensor) and dict1[key] is not None:
                assert torch.allclose(dict1[key], dict2[key], atol=1e-6), \
                    f"Mismatch in tensor field: {key}"
            elif isinstance(dict1[key], (int, float)) and dict1[key] is not None:
                assert dict1[key] == dict2[key], f"Mismatch in scalar field: {key}"

    def test_stp_state_persistence(self):
        """Test STP state is captured in get_state() if region uses STP.

        Verifies that STP components (if present) have their state preserved
        in the region's state object.
        """
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward to activate STP
        input_spikes = torch.ones(params["n_input"], device=region.device)
        for _ in range(5):
            region.forward(input_spikes)

        # Get state
        state = region.get_state()
        state_dict = state.to_dict()

        # Check if region has STP components
        has_stp = False
        for attr_name in dir(region):
            if "stp_" in attr_name.lower() and not attr_name.startswith("_"):
                attr = getattr(region, attr_name)
                if attr is not None and hasattr(attr, "get_state"):
                    has_stp = True
                    # Verify corresponding state field exists
                    state_field_name = f"{attr_name}_state"
                    assert state_field_name in state_dict, \
                        f"STP component {attr_name} should have state field {state_field_name}"

                    # If STP was active, state should contain u and x
                    stp_state = state_dict[state_field_name]
                    if stp_state is not None:
                        assert "u" in stp_state or "x" in stp_state, \
                            f"STP state should contain u/x fields"

        # If no STP found, test passes (not all regions use STP)
        if not has_stp:
            pytest.skip("Region does not use STP")


# =============================================================================
# STANDARD TEST SUMMARY
# =============================================================================
# Total: 17 standard tests inherited by all region test classes
#
# Initialization (2 tests):
#   - test_initialization: Verify with default params
#   - test_initialization_minimal: Verify with minimal params
#
# Forward pass (4 tests):
#   - test_forward_pass_tensor_input: Single tensor input
#   - test_forward_pass_dict_input: Multi-source dict input
#   - test_forward_pass_zero_input: Zero/empty input handling
#   - test_forward_pass_multiple_calls: Temporal consistency
#
# Growth (3 tests):
#   - test_grow_output: Add output neurons
#   - test_grow_input: Expand input dimension
#   - test_growth_preserves_state: State preservation during growth
#
# State management (5 tests):
#   - test_state_get_and_load: Round-trip serialization
#   - test_reset_state: State reinitialization
#   - test_state_file_io: Save/load from disk
#   - test_stp_state_persistence: STP state capture
#   - test_device_cuda: CUDA device transfer (within device tests)
#
# Device support (2 tests):
#   - test_device_cpu: CPU execution
#   - test_device_cuda: CUDA execution and state transfer
#
# Integration (2 tests):
#   - test_neuromodulator_update: Neuromodulator application
#   - test_diagnostics_collection: Diagnostic metrics
# =============================================================================
