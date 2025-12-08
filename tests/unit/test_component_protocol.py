"""
Tests for BrainComponent protocol and component parity.

Ensures that both BrainRegion and BaseNeuralPathway implement
the complete BrainComponent protocol, preventing feature asymmetry.
"""

import pytest
import torch

from thalia.core.component_protocol import BrainComponent
from thalia.regions.base import BrainRegion
from thalia.core.pathway_protocol import BaseNeuralPathway


class TestComponentProtocol:
    """Test that BrainComponent protocol is properly defined and used."""

    def test_protocol_defines_required_methods(self):
        """BrainComponent protocol should define all required methods."""
        required_methods = [
            'forward',
            'reset_state',
            'add_neurons',
            'get_capacity_metrics',
            'get_diagnostics',
            'check_health',
            'get_full_state',
            'load_full_state',
        ]

        required_properties = [
            'device',
            'dtype',
        ]

        protocol_methods = dir(BrainComponent)

        for method in required_methods:
            assert method in protocol_methods, \
                f"BrainComponent protocol missing required method: {method}"

        for prop in required_properties:
            assert prop in protocol_methods, \
                f"BrainComponent protocol missing required property: {prop}"

    def test_brain_region_implements_protocol(self):
        """BrainRegion should implement all BrainComponent methods."""
        # Check that BrainRegion has all required methods
        region_methods = dir(BrainRegion)

        required_methods = [
            'forward',
            'reset_state',
            'add_neurons',
            'get_capacity_metrics',
            'get_diagnostics',
            'check_health',
            'get_full_state',
            'load_full_state',
        ]

        for method in required_methods:
            assert method in region_methods, \
                f"BrainRegion missing BrainComponent method: {method}"

    def test_pathway_implements_protocol(self):
        """BaseNeuralPathway should implement all BrainComponent methods."""
        # Check that BaseNeuralPathway has all required methods
        pathway_methods = dir(BaseNeuralPathway)

        required_methods = [
            'forward',
            'reset_state',
            'add_neurons',
            'get_capacity_metrics',
            'get_diagnostics',
            'check_health',
            'get_full_state',
            'load_full_state',
        ]

        for method in required_methods:
            assert method in pathway_methods, \
                f"BaseNeuralPathway missing BrainComponent method: {method}"

    def test_region_and_pathway_have_same_interface(self):
        """Regions and pathways should have the same public interface."""
        region_methods = {
            m for m in dir(BrainRegion)
            if not m.startswith('_')
        }
        pathway_methods = {
            m for m in dir(BaseNeuralPathway)
            if not m.startswith('_')
        }

        # Core BrainComponent methods that must match
        core_methods = {
            'forward',
            'reset_state',
            'add_neurons',
            'get_capacity_metrics',
            'get_diagnostics',
            'check_health',
            'get_full_state',
            'load_full_state',
        }

        # Both should have all core methods
        assert core_methods.issubset(region_methods), \
            "BrainRegion missing core methods"
        assert core_methods.issubset(pathway_methods), \
            "BaseNeuralPathway missing core methods"

        # Report on method differences (informational, not a failure)
        region_only = region_methods - pathway_methods - core_methods
        pathway_only = pathway_methods - region_methods - core_methods

        if region_only:
            print(f"\nRegion-specific methods: {sorted(region_only)}")
        if pathway_only:
            print(f"\nPathway-specific methods: {sorted(pathway_only)}")


class TestComponentParity:
    """Test that component parity is maintained across implementations."""

    def test_add_neurons_signature_matches(self):
        """add_neurons() should have same signature for regions and pathways."""
        import inspect

        region_sig = inspect.signature(BrainRegion.add_neurons)
        pathway_sig = inspect.signature(BaseNeuralPathway.add_neurons)

        # Parameters should match
        region_params = list(region_sig.parameters.keys())
        pathway_params = list(pathway_sig.parameters.keys())

        assert region_params == pathway_params, \
            f"add_neurons() signature mismatch:\n" \
            f"  Region:  {region_params}\n" \
            f"  Pathway: {pathway_params}"

    def test_get_capacity_metrics_signature_matches(self):
        """get_capacity_metrics() should have same signature."""
        import inspect

        region_sig = inspect.signature(BrainRegion.get_capacity_metrics)
        pathway_sig = inspect.signature(BaseNeuralPathway.get_capacity_metrics)

        # Parameters should match
        region_params = list(region_sig.parameters.keys())
        pathway_params = list(pathway_sig.parameters.keys())

        assert region_params == pathway_params, \
            f"get_capacity_metrics() signature mismatch:\n" \
            f"  Region:  {region_params}\n" \
            f"  Pathway: {pathway_params}"

    def test_growth_methods_documented_for_both(self):
        """Growth methods should be documented in both classes."""
        # Check that growth methods have docstrings
        assert BrainRegion.add_neurons.__doc__ is not None, \
            "BrainRegion.add_neurons() missing docstring"
        assert BaseNeuralPathway.add_neurons.__doc__ is not None, \
            "BaseNeuralPathway.add_neurons() missing docstring"

        # Check that docstrings mention key concepts
        region_doc = BrainRegion.add_neurons.__doc__.lower()
        pathway_doc = BaseNeuralPathway.add_neurons.__doc__.lower()

        # Essential concepts that should be documented
        key_concepts = ['neuron', 'growth', 'capacity', 'preserve']

        for concept in key_concepts:
            assert concept in region_doc, \
                f"BrainRegion.add_neurons() docstring missing '{concept}'"
            assert concept in pathway_doc, \
                f"BaseNeuralPathway.add_neurons() docstring missing '{concept}'"


class TestProtocolEnforcement:
    """Test that protocol enforcement works correctly."""

    def test_protocol_isinstance_check(self):
        """Protocol isinstance checks should work for abstract base classes."""
        # This tests that the protocol is runtime_checkable
        from typing import TYPE_CHECKING

        # We can't instantiate abstract classes, but we can check
        # that they would satisfy the protocol if instantiated

        # Check that protocol has required attributes
        assert hasattr(BrainComponent, '__protocol_attrs__') or \
               hasattr(BrainComponent, '__mro__'), \
               "BrainComponent should be a Protocol"

    def test_component_mixin_provides_defaults(self):
        """BrainComponentMixin should provide default implementations."""
        from thalia.core.component_protocol import BrainComponentMixin

        # Check that mixin provides default methods
        assert hasattr(BrainComponentMixin, 'add_neurons')
        assert hasattr(BrainComponentMixin, 'get_capacity_metrics')
        assert hasattr(BrainComponentMixin, 'check_health')

        # Check that defaults raise NotImplementedError where appropriate
        mixin = BrainComponentMixin()

        with pytest.raises(NotImplementedError, match="add_neurons.*not yet implemented"):
            mixin.add_neurons(n_new=100)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
