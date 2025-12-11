"""Tests for pathway registration in ComponentRegistry."""

import pytest
from thalia.core.component_registry import ComponentRegistry

# Import pathways to trigger registration
import thalia.sensory.pathways  # noqa: F401
import thalia.integration.spiking_pathway  # noqa: F401
import thalia.integration.pathways.spiking_attention  # noqa: F401
import thalia.integration.pathways.spiking_replay  # noqa: F401


def test_spiking_pathway_registration():
    """Test that SpikingPathway is registered."""
    assert "spiking" in ComponentRegistry.list_components("pathway")
    
    info = ComponentRegistry.get_component_info("pathway", "spiking")
    assert info is not None
    assert "spiking inter-region pathway" in info["description"].lower()
    assert info["version"] == "2.0"


def test_visual_pathway_registration():
    """Test that VisualPathway is registered."""
    assert "visual" in ComponentRegistry.list_components("pathway")
    
    info = ComponentRegistry.get_component_info("pathway", "visual")
    assert info is not None
    assert "visual pathway" in info["description"].lower()


def test_auditory_pathway_registration():
    """Test that AuditoryPathway is registered."""
    assert "auditory" in ComponentRegistry.list_components("pathway")
    
    info = ComponentRegistry.get_component_info("pathway", "auditory")
    assert info is not None
    assert "auditory pathway" in info["description"].lower()


def test_language_pathway_registration():
    """Test that LanguagePathway is registered."""
    assert "language" in ComponentRegistry.list_components("pathway")
    
    info = ComponentRegistry.get_component_info("pathway", "language")
    assert info is not None
    assert "language pathway" in info["description"].lower()


def test_attention_pathway_registration():
    """Test that SpikingAttentionPathway is registered."""
    assert "attention" in ComponentRegistry.list_components("pathway")
    
    # Test primary name
    info = ComponentRegistry.get_component_info("pathway", "attention")
    assert info is not None
    assert "attention" in info["description"].lower()
    
    # Test alias
    aliases = ComponentRegistry.list_aliases("pathway")
    assert "spiking_attention" in aliases


def test_replay_pathway_registration():
    """Test that SpikingReplayPathway is registered."""
    assert "replay" in ComponentRegistry.list_components("pathway")
    
    # Test primary name
    info = ComponentRegistry.get_component_info("pathway", "replay")
    assert info is not None
    assert "replay" in info["description"].lower() or "consolidation" in info["description"].lower()
    
    # Test aliases
    aliases = ComponentRegistry.list_aliases("pathway")
    assert "spiking_replay" in aliases
    assert "consolidation" in aliases


def test_list_all_pathways():
    """Test listing all registered pathways."""
    pathways = ComponentRegistry.list_components("pathway")
    
    # Check that all expected pathways are registered
    expected_pathways = {
        "spiking",
        "visual", 
        "auditory",
        "language",
        "attention",
        "replay"
    }
    
    for pathway_name in expected_pathways:
        assert pathway_name in pathways, f"Pathway '{pathway_name}' not registered"


def test_pathway_namespace_isolation():
    """Test that pathway namespace is isolated from region namespace."""
    # These should exist in pathway namespace
    assert "spiking" in ComponentRegistry.list_components("pathway")
    assert "visual" in ComponentRegistry.list_components("pathway")
    
    # These should exist in region namespace
    assert "cortex" in ComponentRegistry.list_components("region")
    assert "striatum" in ComponentRegistry.list_components("region")
    
    # No cross-contamination
    pathway_list = ComponentRegistry.list_components("pathway")
    assert "cortex" not in pathway_list
    assert "striatum" not in pathway_list


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
