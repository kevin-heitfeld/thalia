"""
Tests for Unified Component Registry.

Tests the ComponentRegistry system for registering and creating
brain components (regions, pathways, modules).

Author: Thalia Project
Date: December 11, 2025
"""

import pytest
import torch
from dataclasses import dataclass

from thalia.core.component_registry import (
    ComponentRegistry,
    register_region,
    register_pathway,
    register_module,
)
from thalia.core.component_protocol import BrainComponent
from thalia.config.base import BaseConfig


# Test fixtures - Mock components
@dataclass
class MockRegionConfig(BaseConfig):
    """Mock region configuration."""
    n_input: int = 64
    n_output: int = 32


@dataclass
class MockPathwayConfig(BaseConfig):
    """Mock pathway configuration."""
    source_size: int = 64
    target_size: int = 32


class MockRegion(BrainComponent):
    """Mock region for testing."""
    
    def __init__(self, config: MockRegionConfig):
        self.config = config
        self.n_input = config.n_input
        self.n_output = config.n_output
    
    def forward(self, x):
        return torch.zeros(self.n_output)
    
    def reset_state(self):
        pass
    
    def get_diagnostics(self):
        return {}
    
    def get_capacity_metrics(self):
        from thalia.core.growth import CapacityMetrics
        return CapacityMetrics(
            utilization=0.5,
            total_neurons=self.n_output,
            active_neurons=0,
        )
    
    def check_health(self):
        from thalia.core.diagnostics_mixin import HealthStatus
        return HealthStatus(
            is_healthy=True,
            overall_severity="normal",
            issues=[],
            summary="Mock region is healthy",
            metrics={}
        )
    
    def add_neurons(self, n_new: int, **kwargs):
        self.n_output += n_new
    
    def get_full_state(self):
        return {
            "config": self.config,
            "weights": {},
            "region_state": {},
        }
    
    def load_full_state(self, state):
        pass


class MockPathway(BrainComponent):
    """Mock pathway for testing."""
    
    def __init__(self, config: MockPathwayConfig):
        self.config = config
        self.source_size = config.source_size
        self.target_size = config.target_size
    
    def forward(self, x):
        return torch.zeros(self.target_size)
    
    def reset_state(self):
        pass
    
    def get_diagnostics(self):
        return {}
    
    def get_capacity_metrics(self):
        from thalia.core.growth import CapacityMetrics
        return CapacityMetrics(
            utilization=0.3,
            total_neurons=self.target_size,
            active_neurons=0,
        )
    
    def check_health(self):
        from thalia.core.diagnostics_mixin import HealthStatus
        return HealthStatus(
            is_healthy=True,
            overall_severity="normal",
            issues=[],
            summary="Mock pathway is healthy",
            metrics={}
        )
    
    def add_neurons(self, n_new: int, **kwargs):
        self.target_size += n_new
    
    def get_full_state(self):
        return {
            "config": self.config,
            "weights": {},
            "region_state": {},
        }
    
    def load_full_state(self, state):
        pass


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test."""
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


class TestComponentRegistryBasics:
    """Test basic registry operations."""
    
    def test_register_region_decorator(self):
        """Test region registration via decorator."""
        @register_region("test_region")
        class TestRegion(MockRegion):
            pass
        
        assert ComponentRegistry.is_registered("region", "test_region")
        assert ComponentRegistry.get("region", "test_region") == TestRegion
    
    def test_register_pathway_decorator(self):
        """Test pathway registration via decorator."""
        @register_pathway("test_pathway")
        class TestPathway(MockPathway):
            pass
        
        assert ComponentRegistry.is_registered("pathway", "test_pathway")
        assert ComponentRegistry.get("pathway", "test_pathway") == TestPathway
    
    def test_register_module_decorator(self):
        """Test module registration via decorator."""
        @register_module("test_module")
        class TestModule(BrainComponent):
            def __init__(self, config):
                self.config = config
            
            def forward(self, x):
                return x
            
            def reset_state(self):
                pass
            
            def get_diagnostics(self):
                return {}
            
            def get_capacity_metrics(self):
                from thalia.core.growth import CapacityMetrics
                return CapacityMetrics(
                    utilization=0.0,
                    total_neurons=0,
                    active_neurons=0,
                )
            
            def check_health(self):
                from thalia.core.diagnostics_mixin import HealthStatus
                return HealthStatus(
                    is_healthy=True,
                    overall_severity="normal",
                    issues=[],
                    summary="Mock module is healthy",
                    metrics={}
                )
            
            def add_neurons(self, n_new: int, **kwargs):
                pass
            
            def get_full_state(self):
                return {"config": self.config}
            
            def load_full_state(self, state):
                pass
        
        assert ComponentRegistry.is_registered("module", "test_module")
        assert ComponentRegistry.get("module", "test_module") == TestModule
    
    def test_register_with_aliases(self):
        """Test registration with aliases."""
        @register_region("cortex", aliases=["layered_cortex", "ctx"])
        class TestCortex(MockRegion):
            pass
        
        # Primary name works
        assert ComponentRegistry.get("region", "cortex") == TestCortex
        
        # Aliases work
        assert ComponentRegistry.get("region", "layered_cortex") == TestCortex
        assert ComponentRegistry.get("region", "ctx") == TestCortex
    
    def test_register_duplicate_name_error(self):
        """Test that registering duplicate name raises error."""
        @register_region("duplicate")
        class FirstRegion(MockRegion):
            pass
        
        with pytest.raises(ValueError, match="already registered"):
            @register_region("duplicate")
            class SecondRegion(MockRegion):
                pass
    
    def test_register_duplicate_alias_error(self):
        """Test that duplicate alias raises error."""
        @register_region("first", aliases=["shared"])
        class FirstRegion(MockRegion):
            pass
        
        with pytest.raises(ValueError, match="Alias 'shared' already registered"):
            @register_region("second", aliases=["shared"])
            class SecondRegion(MockRegion):
                pass
    
    def test_register_same_class_idempotent(self):
        """Test that registering same class twice is idempotent."""
        @register_region("idempotent")
        class TestRegion(MockRegion):
            pass
        
        # Create instance of decorator to register same class again
        # Should not raise error for same class
        register_region("idempotent")(TestRegion)
        
        assert ComponentRegistry.get("region", "idempotent") == TestRegion


class TestComponentCreation:
    """Test component instantiation via registry."""
    
    def test_create_region(self):
        """Test creating a region instance."""
        @register_region("test_region")
        class TestRegion(MockRegion):
            pass
        
        config = MockRegionConfig(n_input=128, n_output=64)
        region = ComponentRegistry.create("region", "test_region", config)
        
        assert isinstance(region, TestRegion)
        assert region.n_input == 128
        assert region.n_output == 64
    
    def test_create_pathway(self):
        """Test creating a pathway instance."""
        @register_pathway("test_pathway")
        class TestPathway(MockPathway):
            pass
        
        config = MockPathwayConfig(source_size=256, target_size=128)
        pathway = ComponentRegistry.create("pathway", "test_pathway", config)
        
        assert isinstance(pathway, TestPathway)
        assert pathway.source_size == 256
        assert pathway.target_size == 128
    
    def test_create_via_alias(self):
        """Test creating component via alias."""
        @register_region("cortex", aliases=["ctx"])
        class TestCortex(MockRegion):
            pass
        
        config = MockRegionConfig()
        region = ComponentRegistry.create("region", "ctx", config)
        
        assert isinstance(region, TestCortex)
    
    def test_create_unregistered_error(self):
        """Test that creating unregistered component raises error."""
        config = MockRegionConfig()
        
        with pytest.raises(ValueError, match="not registered"):
            ComponentRegistry.create("region", "nonexistent", config)
    
    def test_create_with_kwargs(self):
        """Test creating component with additional kwargs."""
        @register_region("parameterized")
        class ParameterizedRegion(MockRegion):
            def __init__(self, config, extra_param=None):
                super().__init__(config)
                self.extra_param = extra_param
        
        config = MockRegionConfig()
        region = ComponentRegistry.create(
            "region", "parameterized", config,
            extra_param="test_value"
        )
        
        assert region.extra_param == "test_value"


class TestComponentDiscovery:
    """Test component discovery and introspection."""
    
    def test_list_regions(self):
        """Test listing all registered regions."""
        @register_region("cortex")
        class Cortex(MockRegion):
            pass
        
        @register_region("hippocampus")
        class Hippocampus(MockRegion):
            pass
        
        regions = ComponentRegistry.list_components("region")
        assert "cortex" in regions
        assert "hippocampus" in regions
        assert len(regions) == 2
    
    def test_list_pathways(self):
        """Test listing all registered pathways."""
        @register_pathway("spiking")
        class SpikingPathway(MockPathway):
            pass
        
        @register_pathway("attention")
        class AttentionPathway(MockPathway):
            pass
        
        pathways = ComponentRegistry.list_components("pathway")
        assert "spiking" in pathways
        assert "attention" in pathways
        assert len(pathways) == 2
    
    def test_list_all_components(self):
        """Test listing all component types."""
        @register_region("cortex")
        class Cortex(MockRegion):
            pass
        
        @register_pathway("spiking")
        class SpikingPathway(MockPathway):
            pass
        
        all_components = ComponentRegistry.list_components()
        
        assert "region" in all_components
        assert "pathway" in all_components
        assert "cortex" in all_components["region"]
        assert "spiking" in all_components["pathway"]
    
    def test_list_aliases(self):
        """Test getting alias mappings."""
        @register_region("cortex", aliases=["layered_cortex", "ctx"])
        class Cortex(MockRegion):
            pass
        
        aliases = ComponentRegistry.list_aliases("region")
        
        assert aliases["layered_cortex"] == "cortex"
        assert aliases["ctx"] == "cortex"
        assert len(aliases) == 2
    
    def test_get_component_info(self):
        """Test getting component metadata."""
        @register_region(
            "cortex",
            description="Multi-layer cortical microcircuit",
            version="2.0",
            author="Thalia Project"
        )
        class Cortex(MockRegion):
            pass
        
        info = ComponentRegistry.get_component_info("region", "cortex")
        
        assert info is not None
        assert "Multi-layer" in info["description"]
        assert info["version"] == "2.0"
        assert info["author"] == "Thalia Project"
        assert info["class"] == "Cortex"
    
    def test_get_component_info_uses_docstring(self):
        """Test that component info uses docstring if no description."""
        @register_region("documented")
        class DocumentedRegion(MockRegion):
            """This is a documented region."""
            pass
        
        info = ComponentRegistry.get_component_info("region", "documented")
        
        assert "documented region" in info["description"]
    
    def test_validate_component_valid(self):
        """Test validating a component with valid config."""
        @register_region("valid")
        class ValidRegion(MockRegion):
            pass
        
        config = MockRegionConfig()
        is_valid, error = ComponentRegistry.validate_component(
            "region", "valid", config
        )
        
        assert is_valid
        assert error is None
    
    def test_validate_component_unregistered(self):
        """Test validating unregistered component."""
        config = MockRegionConfig()
        is_valid, error = ComponentRegistry.validate_component(
            "region", "nonexistent", config
        )
        
        assert not is_valid
        assert "not registered" in error


class TestRegistryManagement:
    """Test registry management operations."""
    
    def test_clear_specific_type(self):
        """Test clearing specific component type."""
        @register_region("cortex")
        class Cortex(MockRegion):
            pass
        
        @register_pathway("spiking")
        class SpikingPathway(MockPathway):
            pass
        
        # Clear only regions
        ComponentRegistry.clear("region")
        
        assert not ComponentRegistry.is_registered("region", "cortex")
        assert ComponentRegistry.is_registered("pathway", "spiking")
    
    def test_clear_all(self):
        """Test clearing all component types."""
        @register_region("cortex")
        class Cortex(MockRegion):
            pass
        
        @register_pathway("spiking")
        class SpikingPathway(MockPathway):
            pass
        
        ComponentRegistry.clear()
        
        assert not ComponentRegistry.is_registered("region", "cortex")
        assert not ComponentRegistry.is_registered("pathway", "spiking")
    
    def test_is_registered_with_type(self):
        """Test checking registration status."""
        @register_region("cortex")
        class Cortex(MockRegion):
            pass
        
        assert ComponentRegistry.is_registered("region", "cortex")
        assert not ComponentRegistry.is_registered("region", "nonexistent")
        assert not ComponentRegistry.is_registered("pathway", "cortex")


class TestNamespaceIsolation:
    """Test that different component types don't interfere."""
    
    def test_same_name_different_types(self):
        """Test that same name can be used for different types."""
        @register_region("transformer")
        class TransformerRegion(MockRegion):
            pass
        
        @register_pathway("transformer")
        class TransformerPathway(MockPathway):
            pass
        
        # Both should be registered without conflict
        assert ComponentRegistry.get("region", "transformer") == TransformerRegion
        assert ComponentRegistry.get("pathway", "transformer") == TransformerPathway
    
    def test_isolated_namespaces(self):
        """Test that component types have isolated namespaces."""
        @register_region("test")
        class TestRegion(MockRegion):
            pass
        
        # Should not appear in pathway namespace
        assert ComponentRegistry.is_registered("region", "test")
        assert not ComponentRegistry.is_registered("pathway", "test")


class TestBackwardCompatibility:
    """Test backward compatibility with existing patterns."""
    
    def test_register_region_shorthand(self):
        """Test that register_region() works as shorthand."""
        @register_region("cortex")
        class Cortex(MockRegion):
            pass
        
        # Should be accessible via ComponentRegistry
        assert ComponentRegistry.is_registered("region", "cortex")
        assert ComponentRegistry.get("region", "cortex") == Cortex
    
    def test_multiple_registrations_coexist(self):
        """Test that new and old registration patterns coexist."""
        # Old pattern (shorthand)
        @register_region("old_style")
        class OldStyle(MockRegion):
            pass
        
        # New pattern (explicit)
        @ComponentRegistry.register("new_style", "region")
        class NewStyle(MockRegion):
            pass
        
        # Both should work
        assert ComponentRegistry.is_registered("region", "old_style")
        assert ComponentRegistry.is_registered("region", "new_style")


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_component_type(self):
        """Test that invalid component type raises error."""
        with pytest.raises(ValueError, match="Invalid component_type"):
            @ComponentRegistry.register("test", "invalid_type")
            class TestComponent(MockRegion):
                pass
    
    def test_register_non_class(self):
        """Test that registering non-class raises error."""
        with pytest.raises(ValueError, match="must be a class"):
            decorator = ComponentRegistry.register("test", "region")
            decorator("not_a_class")
    
    def test_get_invalid_type(self):
        """Test getting component with invalid type returns None."""
        result = ComponentRegistry.get("invalid_type", "test")
        assert result is None
    
    def test_list_invalid_type(self):
        """Test listing components of invalid type returns empty list."""
        result = ComponentRegistry.list_components("invalid_type")
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
