"""
Tests for Region Factory and Registry.

Tests verify that:
1. Regions can be registered and retrieved
2. Factory creates regions correctly
3. Aliases work as expected
4. Error handling is appropriate
"""

import pytest
import torch

from thalia.regions import (
    RegionFactory,
    RegionRegistry,
    register_region,
    BrainRegion,
    RegionConfig,
    LayeredCortex,
    LayeredCortexConfig,
    Cerebellum,
    CerebellumConfig,
    Striatum,
    StriatumConfig,
    Prefrontal,
    PrefrontalConfig,
    TrisynapticHippocampus,
    TrisynapticConfig,
)


class TestRegionRegistry:
    """Tests for RegionRegistry."""
    
    def test_registry_has_standard_regions(self):
        """Test that standard regions are registered."""
        assert RegionRegistry.is_registered("cortex")
        assert RegionRegistry.is_registered("cerebellum")
        assert RegionRegistry.is_registered("striatum")
        assert RegionRegistry.is_registered("prefrontal")
        assert RegionRegistry.is_registered("hippocampus")
    
    def test_aliases_work(self):
        """Test that aliases resolve to correct regions."""
        # Cortex aliases
        assert RegionRegistry.get("cortex") == LayeredCortex
        assert RegionRegistry.get("layered_cortex") == LayeredCortex
        
        # PFC aliases
        assert RegionRegistry.get("prefrontal") == Prefrontal
        assert RegionRegistry.get("pfc") == Prefrontal
        
        # Hippocampus aliases
        assert RegionRegistry.get("hippocampus") == TrisynapticHippocampus
        assert RegionRegistry.get("trisynaptic") == TrisynapticHippocampus
    
    def test_list_regions(self):
        """Test listing all registered regions."""
        regions = RegionRegistry.list_regions()
        assert "cortex" in regions
        assert "predictive_cortex" in regions
        assert "cerebellum" in regions
        assert "striatum" in regions
        assert "prefrontal" in regions
        assert "hippocampus" in regions
    
    def test_get_nonexistent_region(self):
        """Test getting a non-existent region returns None."""
        assert RegionRegistry.get("nonexistent") is None


class TestRegionFactory:
    """Tests for RegionFactory."""
    
    def test_create_cortex(self):
        """Test creating a cortex region."""
        config = LayeredCortexConfig(n_input=256, n_output=128)
        cortex = RegionFactory.create("cortex", config)
        
        assert isinstance(cortex, LayeredCortex)
        assert isinstance(cortex, BrainRegion)
        assert cortex.config.n_input == 256
        # LayeredCortex computes internal layer sizes based on n_input, not n_output
        assert cortex.config.n_output > 0
    
    def test_create_with_alias(self):
        """Test creating region using alias."""
        config = LayeredCortexConfig(n_input=256, n_output=128)
        cortex = RegionFactory.create("layered_cortex", config)
        
        assert isinstance(cortex, LayeredCortex)
    
    def test_create_cerebellum(self):
        """Test creating a cerebellum region."""
        config = CerebellumConfig(n_input=128, n_output=64)
        cerebellum = RegionFactory.create("cerebellum", config)
        
        assert isinstance(cerebellum, Cerebellum)
        assert cerebellum.config.n_input == 128
        assert cerebellum.config.n_output == 64
    
    def test_create_striatum(self):
        """Test creating a striatum region."""
        config = StriatumConfig(n_input=192, n_output=4)
        striatum = RegionFactory.create("striatum", config)
        
        assert isinstance(striatum, Striatum)
        assert striatum.config.n_input == 192
        # Striatum uses population_coding with neurons_per_action=10
        # So n_output=4 actions * 10 neurons_per_action = 40 total neurons
        assert striatum.config.n_output == 40
    
    def test_create_prefrontal(self):
        """Test creating a prefrontal region."""
        config = PrefrontalConfig(n_input=192, n_output=32)
        pfc = RegionFactory.create("prefrontal", config)
        
        assert isinstance(pfc, Prefrontal)
        assert pfc.config.n_input == 192
        assert pfc.config.n_output == 32
    
    def test_create_hippocampus(self):
        """Test creating a hippocampus region."""
        config = TrisynapticConfig(n_input=128, n_output=64)
        hippocampus = RegionFactory.create("hippocampus", config)
        
        assert isinstance(hippocampus, TrisynapticHippocampus)
        assert hippocampus.config.n_input == 128
        assert hippocampus.config.n_output == 64
    
    def test_create_nonexistent_region(self):
        """Test creating a non-existent region raises error."""
        config = RegionConfig(n_input=10, n_output=5)
        
        with pytest.raises(ValueError) as exc_info:
            RegionFactory.create("nonexistent", config)
        
        assert "Unknown region 'nonexistent'" in str(exc_info.value)
        assert "Available regions:" in str(exc_info.value)
    
    def test_create_batch(self):
        """Test creating multiple regions at once."""
        region_specs = {
            "cortex": LayeredCortexConfig(n_input=256, n_output=128),
            "striatum": StriatumConfig(n_input=192, n_output=4),
            "cerebellum": CerebellumConfig(n_input=128, n_output=64),
        }
        
        regions = RegionFactory.create_batch(region_specs)
        
        assert len(regions) == 3
        assert isinstance(regions["cortex"], LayeredCortex)
        assert isinstance(regions["striatum"], Striatum)
        assert isinstance(regions["cerebellum"], Cerebellum)
    
    def test_is_available(self):
        """Test checking region availability."""
        assert RegionFactory.is_available("cortex")
        assert RegionFactory.is_available("pfc")
        assert not RegionFactory.is_available("nonexistent")
    
    def test_list_available(self):
        """Test listing available regions."""
        available = RegionFactory.list_available()
        assert "cortex" in available
        assert "cerebellum" in available
        assert "striatum" in available


class TestRegisterDecorator:
    """Tests for @register_region decorator."""
    
    def test_register_custom_region(self):
        """Test registering a custom region."""
        
        @register_region("test_region")
        class TestRegion(BrainRegion):
            def _get_learning_rule(self):
                from thalia.regions.base import LearningRule
                return LearningRule.HEBBIAN
            
            def _initialize_weights(self):
                return torch.zeros(self.config.n_output, self.config.n_input)
            
            def _create_neurons(self):
                return None
            
            def forward(self, input_spikes, **kwargs):
                return torch.zeros(1, self.config.n_output)
        
        # Check it's registered
        assert RegionRegistry.is_registered("test_region")
        assert RegionRegistry.get("test_region") == TestRegion
        
        # Can create it
        config = RegionConfig(n_input=10, n_output=5)
        region = RegionFactory.create("test_region", config)
        assert isinstance(region, TestRegion)
        
        # Cleanup
        del RegionRegistry._registry["test_region"]
    
    def test_register_with_aliases(self):
        """Test registering region with aliases."""
        
        @register_region("test_region2", aliases=["alias1", "alias2"])
        class TestRegion2(BrainRegion):
            def _get_learning_rule(self):
                from thalia.regions.base import LearningRule
                return LearningRule.HEBBIAN
            
            def _initialize_weights(self):
                return torch.zeros(self.config.n_output, self.config.n_input)
            
            def _create_neurons(self):
                return None
            
            def forward(self, input_spikes, **kwargs):
                return torch.zeros(1, self.config.n_output)
        
        # Check all names work
        assert RegionRegistry.is_registered("test_region2")
        assert RegionRegistry.is_registered("alias1")
        assert RegionRegistry.is_registered("alias2")
        
        assert RegionRegistry.get("test_region2") == TestRegion2
        assert RegionRegistry.get("alias1") == TestRegion2
        assert RegionRegistry.get("alias2") == TestRegion2
        
        # Cleanup
        del RegionRegistry._registry["test_region2"]
        del RegionRegistry._aliases["alias1"]
        del RegionRegistry._aliases["alias2"]


class TestDynamicBrainConstruction:
    """Tests for dynamic brain construction patterns."""
    
    def test_loop_based_construction(self):
        """Test building brain via loop over region names."""
        region_configs = {
            "cortex": LayeredCortexConfig(n_input=256, n_output=128),
            "striatum": StriatumConfig(n_input=128, n_output=4),
        }
        
        # Loop-based construction (as proposed in refactoring doc)
        regions = {}
        for region_name, config in region_configs.items():
            regions[region_name] = RegionFactory.create(region_name, config)
        
        assert len(regions) == 2
        assert isinstance(regions["cortex"], LayeredCortex)
        assert isinstance(regions["striatum"], Striatum)
    
    def test_selective_region_activation(self):
        """Test activating only selected regions."""
        all_configs = {
            "cortex": LayeredCortexConfig(n_input=256, n_output=128),
            "hippocampus": TrisynapticConfig(n_input=128, n_output=64),
            "striatum": StriatumConfig(n_input=192, n_output=4),
            "cerebellum": CerebellumConfig(n_input=128, n_output=64),
        }
        
        # Only activate cortex and striatum
        active_regions = ["cortex", "striatum"]
        
        regions = {}
        for region_name in active_regions:
            if region_name in all_configs:
                regions[region_name] = RegionFactory.create(
                    region_name,
                    all_configs[region_name]
                )
        
        assert len(regions) == 2
        assert "cortex" in regions
        assert "striatum" in regions
        assert "hippocampus" not in regions
        assert "cerebellum" not in regions
