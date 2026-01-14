"""
Tests for DynamicBrain PathwayManager integration (Phase 1.7.1).

Tests that DynamicBrain correctly integrates PathwayManager for:
- Pathway diagnostics collection
- Coordinated pathway growth
- Pathway state save/load

Author: Thalia Project
Date: December 15, 2025
"""

import pytest
import torch

from thalia.config import GlobalConfig
from thalia.core.brain_builder import BrainBuilder
from thalia.regions.cortex import calculate_layer_sizes


class TestPathwayManagerIntegration:
    """Test PathwayManager integration in DynamicBrain."""

    @pytest.fixture
    def global_config(self):
        """Create test global config."""
        return GlobalConfig(device="cpu", dt_ms=1.0)

    @pytest.fixture
    def simple_brain(self, global_config):
        """Create simple brain for testing."""
        brain = (
            BrainBuilder(global_config)
            .add_component("input", "thalamic_relay", input_size=64, relay_size=64, trn_size=0)
            .add_component("cortex", "layered_cortex", input_size=64, **calculate_layer_sizes(32))
            .connect("input", "cortex", pathway_type="axonal_projection")  # Routing pathway (no weights)
            .build()
        )
        return brain

    def test_pathway_manager_exists(self, simple_brain):
        """Test that PathwayManager is initialized."""
        assert hasattr(simple_brain, 'pathway_manager')
        assert simple_brain.pathway_manager is not None

    def test_pathway_diagnostics(self, simple_brain):
        """Test pathway diagnostics collection."""
        # Get diagnostics
        diag = simple_brain.get_diagnostics()

        # Should include pathways diagnostics (from PathwayManager)
        assert 'pathways' in diag
        pathway_diag = diag['pathways']

        # Should have pathway entries
        assert len(pathway_diag) > 0

        # Check for input_to_cortex pathway
        assert 'input_to_cortex' in pathway_diag

        # Should have weight statistics
        cortex_pathway_diag = pathway_diag['input_to_cortex']
        assert isinstance(cortex_pathway_diag, dict)

    def test_pathway_manager_get_all_pathways(self, simple_brain):
        """Test get_all_pathways returns correct format."""
        pathways = simple_brain.pathway_manager.get_all_pathways()

        assert isinstance(pathways, dict)
        assert 'input_to_cortex' in pathways

        # Should be pathway instance
        pathway = pathways['input_to_cortex']
        assert hasattr(pathway, 'forward')

    def test_full_state_includes_pathways(self, simple_brain):
        """Test that get_full_state includes pathway states."""
        # Run forward pass
        input_data = {"input": torch.randn(64)}
        simple_brain.forward(input_data, n_timesteps=5)

        # Get full state
        state = simple_brain.get_full_state()

        # Should have pathways key
        assert 'pathways' in state
        assert isinstance(state['pathways'], dict)
        assert len(state['pathways']) > 0

    def test_auto_grow_uses_pathway_manager(self, simple_brain, monkeypatch):
        """Test that auto_grow uses PathwayManager for pathway growth."""
        # Track if pathway manager was called
        grow_called = {'called': False, 'component': None, 'amount': None}

        original_grow = simple_brain.pathway_manager.grow_connected_pathways

        def tracked_grow(component_name, growth_amount, **kwargs):
            grow_called['called'] = True
            grow_called['component'] = component_name
            grow_called['amount'] = growth_amount
            return original_grow(component_name, growth_amount, **kwargs)

        monkeypatch.setattr(
            simple_brain.pathway_manager,
            'grow_connected_pathways',
            tracked_grow
        )

        # Force growth by making component report high utilization
        def mock_check_growth_needs(self):
            return {
                'cortex': {
                    'growth_recommended': True,
                    'growth_reason': 'High utilization',
                    'firing_rate': 0.9,
                    'weight_saturation': 0.85,
                }
            }

        monkeypatch.setattr(simple_brain.__class__, 'check_growth_needs', mock_check_growth_needs)

        # Trigger auto-grow
        _ = simple_brain.auto_grow(threshold=0.8)  # noqa: F841

        # Should have called pathway manager
        assert grow_called['called']
        assert grow_called['component'] == 'cortex'
        assert grow_called['amount'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
