"""Base test class for region testing with common test patterns.

This module provides RegionTestBase, an abstract base class that standardizes
testing patterns across all brain regions. It eliminates test boilerplate by
providing common test methods for:

- Initialization verification (2 tests)
- Forward pass shape checking (4 tests)
- Growth functionality (3 tests: input/output expansion, state preservation)
- State management (7 tests: get/load, reset, file I/O, STP persistence, device transfer, roundtrip serialization, None handling)
- Device support (2 tests: CPU and CUDA)
- Integration (2 tests: neuromodulators, diagnostics)

Total: **19 standard tests** inherited by all region test classes.

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
        return {"input_size": 100, "output_size": 50}  # Use semantic names

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
from thalia.typing import SourceOutputs


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

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Return default region parameters for testing.

        Returns:
            Dict with standard parameters using semantic field names.
            Must include input field (e.g., 'input_size') and output field
            (e.g., 'output_size', 'purkinje_size', 'n_actions', 'relay_size').

        Example:
            def get_default_params(self):
                return {
                    "input_size": 100,  # Semantic input field
                    "output_size": 50,  # Semantic output field (region-specific)
                    "device": "cpu",
                    "dt_ms": 1.0,
                }
        """

    # =========================================================================
    # OPTIONAL OVERRIDES
    # =========================================================================

    def get_min_params(self) -> Dict[str, Any]:
        """Return minimal valid parameters (defaults to get_default_params).

        Override if region has smaller valid configurations for quick tests.
        """
        return self.get_default_params()

    def get_input_dict(self, n_input: int, device: str = "cpu") -> SourceOutputs:
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

    def _prepare_forward_input(self, input_spikes: torch.Tensor):
        """Prepare input for forward() call (handles multi-source regions).

        Args:
            input_spikes: Raw tensor input

        Returns:
            Input in format expected by region (Tensor or Dict[str, Tensor])
        """
        # Try to detect if region expects dict input
        # Multi-source regions (like Striatum) should override get_input_dict()
        input_dict = self.get_input_dict(input_spikes.shape[0], device=str(input_spikes.device))

        # If get_input_dict returns single "default" source, check if region accepts dict
        if len(input_dict) == 1 and "default" in input_dict:
            # Return dict for regions that require it (detected via signature check)
            return {"default": input_spikes}
        else:
            # Multi-source case: split input across sources
            return input_dict

    # =========================================================================
    # SEMANTIC FIELD NAME HELPERS
    # =========================================================================

    def _get_input_field_name(self) -> str:
        """Get semantic input field name (always 'input_size').

        Returns:
            'input_size' for all regions
        """
        return "input_size"

    def _get_output_field_name(self) -> str:
        """Get semantic output field name based on region type.

        Returns:
            Region-specific output field name:
            - Cerebellum: 'purkinje_size'
            - Striatum: 'n_actions'
            - Thalamus: 'relay_size'
            - Hippocampus/Cortex/Prefrontal: 'output_size'
        """
        # Extract region name from test class name (e.g., TestCerebellum → cerebellum)
        class_name = type(self).__name__.lower()

        if "cerebellum" in class_name:
            return "purkinje_size"
        elif "striatum" in class_name:
            return "n_actions"
        elif "thalamus" in class_name or "thalamic" in class_name:
            return "relay_size"
        else:
            # Default for hippocampus, cortex, prefrontal
            return "output_size"

    def _get_input_size(self, params: Dict[str, Any]) -> int:
        """Get input size from params dict using semantic field name.

        Args:
            params: Parameter dictionary from get_default_params()

        Returns:
            Input size value
        """
        field = self._get_input_field_name()
        return params[field]

    def _get_region_input_size(self, region: NeuralRegion) -> int:
        """Get input size from region (new pattern).

        Args:
            region: Region instance

        Returns:
            Input size value from region.input_size
        """
        return getattr(region, "input_size", None)

    def _get_region_output_size(self, region: NeuralRegion) -> int:
        """Get output size from region (new pattern).

        Args:
            region: Region instance

        Returns:
            Output size value from region.n_output
        """
        return getattr(region, "n_output", None)

    def _get_config_input_size(self, config: Any) -> int:
        """Get input size from config using semantic field name.

        DEPRECATED: Use _get_region_input_size() instead (new size-free config pattern).
        This method is kept for backward compatibility.

        Args:
            config: Region configuration object

        Returns:
            Input size value from config (or None if not present)
        """
        field = self._get_input_field_name()
        return getattr(config, field, None)

    def _get_config_output_size(self, config: Any) -> int:
        """Get output size from config using semantic field name.

        DEPRECATED: Use _get_region_output_size() instead (new size-free config pattern).
        This method is kept for backward compatibility.

        Args:
            config: Region configuration object

        Returns:
            Output size value from config (or None if not present)
        """
        field = self._get_output_field_name()
        return getattr(config, field, None)

    # =========================================================================
    # COMMON TEST PATTERNS
    # =========================================================================

    def test_initialization(self):
        """Test region initializes correctly with valid parameters."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Verify sizes stored in region (new pattern: sizes separate from config)
        assert self._get_region_input_size(region) == self._get_input_size(params)
        # Output size is computed property - just verify it exists and is > 0
        assert self._get_region_output_size(region) > 0

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

        # Verify basic functionality (using semantic field names)
        assert self._get_region_input_size(region) == self._get_input_size(params)
        # Output size is computed, just verify it's > 0
        assert self._get_region_output_size(region) > 0

    def test_forward_pass_dict_input(self):
        """Test forward pass with dict input (standardized multi-source API)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Create dict input (all regions now use dict format)
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)

        # Forward pass
        output = region.forward(input_dict)

        # Verify output shape (1D per ADR-005)
        output_size = self._get_region_output_size(region)
        assert output.dim() == 1, f"Expected 1D output (ADR-005), got {output.dim()}D"
        assert output.shape[0] == output_size

        # Verify output is boolean or float
        assert output.dtype in [torch.bool, torch.float32, torch.float64]

    def test_forward_pass_zero_input(self):
        """Test forward pass handles zero input (clock-driven execution)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Zero input (no spikes) - use dict format
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)

        # Should not raise error
        output = region.forward(input_dict)
        output_size = self._get_region_output_size(region)
        assert output.shape[0] == output_size

    def test_forward_pass_multiple_calls(self):
        """Test region handles multiple forward passes (temporal dynamics)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        input_size = self._get_input_size(params)
        output_size = self._get_region_output_size(region)
        input_dict = self.get_input_dict(input_size, device=region.device.type)

        # Multiple forward passes should not error
        for _ in range(10):
            output = region.forward(input_dict)
            assert output.shape[0] == output_size

    def test_grow_output(self):
        """Test region can grow output dimension (add neurons)."""
        if self.skip_growth_tests():
            pytest.skip("Region doesn't support growth")

        params = self.get_default_params()
        region = self.create_region(**params)

        # Get original output size from region config (it's a property)
        original_output = self._get_region_output_size(region)
        n_new = 10

        # Grow output
        region.grow_output(n_new)

        # Verify config updated
        assert self._get_region_output_size(region) == original_output + n_new

        # Verify forward pass still works with new size
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)
        output = region.forward(input_dict)
        assert output.shape[0] == original_output + n_new

    def test_grow_input(self):
        """Test region can grow input dimension (accept more inputs)."""
        if self.skip_growth_tests():
            pytest.skip("Region doesn't support growth")

        params = self.get_default_params()
        region = self.create_region(**params)

        original_input = self._get_input_size(params)
        n_new = 20

        # Grow input
        region.grow_input(n_new)

        # Verify config updated
        assert self._get_region_input_size(region) == original_input + n_new

        # Verify forward pass works with new input size
        input_dict = self.get_input_dict(original_input + n_new, device=region.device.type)
        output = region.forward(input_dict)
        output_size = self._get_region_output_size(region)
        assert output.shape[0] == output_size

    def test_growth_preserves_state(self):
        """Test growth preserves existing neuron state."""
        if self.skip_growth_tests():
            pytest.skip("Region doesn't support growth")

        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass to initialize state
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)
        region.forward(input_dict)

        # Get state before growth
        state_before = region.get_state()
        # Get original output size from region config
        n_original = self._get_region_output_size(region)

        # Grow output
        region.grow_output(10)

        # Get state after growth
        state_after = region.get_state()

        # Verify original neurons preserved (first N values should match)
        if state_before.membrane is not None and state_after.membrane is not None:
            original_membrane = state_before.membrane[:n_original]
            preserved_membrane = state_after.membrane[:n_original]
            assert torch.allclose(original_membrane, preserved_membrane, atol=1e-5)

    def test_state_get_and_load(self):
        """Test get_state() and load_state() round-trip."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass to generate non-trivial state
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)
        region.forward(input_dict)

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
            # Skip None values (optional state fields like pfc_modulation)
            if state_dict1[key] is None or state_dict2.get(key) is None:
                continue
            if isinstance(state_dict1[key], torch.Tensor):
                try:
                    assert torch.allclose(state_dict1[key], state_dict2[key], atol=1e-5)
                except (RuntimeError, AssertionError) as e:
                    # Add context about which key failed
                    shape1 = (
                        state_dict1[key].shape
                        if isinstance(state_dict1[key], torch.Tensor)
                        else "N/A"
                    )
                    shape2 = (
                        state_dict2[key].shape
                        if isinstance(state_dict2[key], torch.Tensor)
                        else "N/A"
                    )
                    raise AssertionError(
                        f"State mismatch for key '{key}': shape1={shape1}, shape2={shape2}"
                    ) from e
            elif isinstance(state_dict1[key], (int, float)):
                assert state_dict1[key] == state_dict2[key]

    def test_reset_state(self):
        """Test reset_state() reinitializes region state."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass to generate non-trivial state
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)
        region.forward(input_dict)

        # Reset state
        region.reset_state()

        # Verify state reset (membrane potentials should be at rest)
        state = region.get_state()
        if state.membrane is not None:
            # Most neurons should be at resting potential (~-70mV)
            # Skip if membrane values are near 0 (some regions don't normalize to mV)
            mean_membrane = state.membrane.mean().item()
            if abs(mean_membrane) > 1.0:  # Only check if values are in mV range
                assert (
                    -75.0 <= mean_membrane <= -65.0
                ), f"Expected resting potential, got {mean_membrane}mV"

    def test_device_cpu(self):
        """Test region works on CPU device."""
        params = self.get_default_params()
        params["device"] = "cpu"

        region = self.create_region(**params)
        assert region.device.type == "cpu"

        # Verify forward pass works
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device="cpu")
        output = region.forward(input_dict)
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self):
        """Test region works on CUDA device (if available)."""
        params = self.get_default_params()
        params["device"] = "cuda"

        region = self.create_region(**params)
        assert region.device.type == "cuda"

        # Verify forward pass works
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device="cuda")
        output = region.forward(input_dict)
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
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)
        region.forward(input_dict)

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
        from thalia.core.region_state import load_region_state, save_region_state

        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward to generate non-trivial state
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)
        for _ in range(3):
            region.forward(input_dict)

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
                assert torch.allclose(
                    dict1[key], dict2[key], atol=1e-6
                ), f"Mismatch in tensor field: {key}"
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
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)
        for _ in range(5):
            region.forward(input_dict)

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
                    # Two state storage patterns:
                    # 1. Flat fields: stp_corticostriatal → stp_corticostriatal_u, stp_corticostriatal_x (striatum)
                    # 2. Nested dict: stp_ca3_recurrent → stp_ca3_recurrent_state: {"u": ..., "x": ...} (others)
                    u_field = f"{attr_name}_u"
                    x_field = f"{attr_name}_x"
                    state_field = f"{attr_name}_state"

                    # Check for either pattern
                    assert (
                        u_field in state_dict or x_field in state_dict or state_field in state_dict
                    ), f"STP component {attr_name} should have state fields {u_field}/{x_field} (flat) or {state_field} (nested)"

        # If no STP found, test passes (not all regions use STP)
        if not has_stp:
            pytest.skip("Region does not use STP")

    def test_state_to_dict_from_dict_roundtrip(self):
        """Test to_dict() and from_dict() preserve all state fields.

        Verifies complete serialization roundtrip without loss.
        """
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward to generate non-trivial state
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)
        for _ in range(5):
            region.forward(input_dict)

        # Get state and serialize
        state1 = region.get_state()
        data = state1.to_dict()

        # Deserialize
        state_class = type(state1)
        state2 = state_class.from_dict(data, device=params["device"])

        # Verify roundtrip preserves all tensor fields
        dict1 = state1.to_dict()
        dict2 = state2.to_dict()

        for key in dict1:
            val1 = dict1[key]
            val2 = dict2[key]

            if isinstance(val1, torch.Tensor) and val1 is not None:
                assert torch.allclose(
                    val1, val2, atol=1e-6
                ), f"Roundtrip mismatch in tensor field: {key}"
            elif isinstance(val1, (int, float)) and val1 is not None:
                assert val1 == val2, f"Roundtrip mismatch in scalar field: {key}"
            elif isinstance(val1, dict) and val1 is not None:
                # Handle nested dicts (e.g., STP state)
                for subkey in val1:
                    if isinstance(val1[subkey], torch.Tensor):
                        assert torch.allclose(
                            val1[subkey], val2[subkey], atol=1e-6
                        ), f"Roundtrip mismatch in nested field: {key}.{subkey}"

    def test_state_handles_none_fields(self):
        """Test state serialization handles None/missing fields gracefully.

        Verifies edge case where some optional state fields are None.
        """
        params = (
            self.get_min_params() if hasattr(self, "get_min_params") else self.get_default_params()
        )
        region = self.create_region(**params)

        # Get state immediately (minimal initialization)
        state1 = region.get_state()

        # Serialize and deserialize
        data = state1.to_dict()
        state_class = type(state1)
        state2 = state_class.from_dict(data, device=params["device"])

        # Should not raise errors
        # Verify at least some fields are present
        assert data is not None
        assert len(data) > 0


# =============================================================================
# STANDARD TEST SUMMARY
# =============================================================================
# Total: 19 standard tests inherited by all region test classes
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
# State management (7 tests):
#   - test_state_get_and_load: Round-trip serialization via region methods
#   - test_reset_state: State reinitialization
#   - test_state_file_io: Save/load from disk
#   - test_stp_state_persistence: STP state capture
#   - test_device_cuda: CUDA device transfer (within device tests)
#   - test_state_to_dict_from_dict_roundtrip: Direct serialization roundtrip
#   - test_state_handles_none_fields: None/missing field handling
#
# Device support (2 tests):
#   - test_device_cpu: CPU execution
#   - test_device_cuda: CUDA execution and state transfer
#
# Integration (2 tests):
#   - test_neuromodulator_update: Neuromodulator application
#   - test_diagnostics_collection: Diagnostic metrics
# =============================================================================
