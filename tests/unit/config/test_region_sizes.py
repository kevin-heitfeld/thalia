"""
Tests for region size computation helpers and explicit size configs.

Tests the helper functions and explicit size fields added during the
December 2025 explicit size refactoring (commits 9b636ed, 29afce6).

Test Coverage:
- Helper functions return correct size dictionaries
- All regions initialize with correct explicit sizes
- Growth methods preserve explicit size fields in configs
- Edge cases (zero sizes, first growth, large growth)
"""

import pytest
import torch

from thalia.config import (
    compute_hippocampus_sizes,
    compute_cortex_layer_sizes,
    compute_thalamus_sizes,
    compute_multisensory_sizes,
    compute_cerebellum_sizes,
    compute_striatum_sizes,
    THALAMUS_TRN_RATIO,
    CEREBELLUM_GRANULE_EXPANSION,
)
from thalia.regions.hippocampus import HippocampusConfig, Hippocampus
from thalia.regions.cortex import LayeredCortexConfig, LayeredCortex
from thalia.regions.thalamus import ThalamicRelayConfig, ThalamicRelay
from thalia.regions.multisensory import MultimodalIntegrationConfig, MultimodalIntegration
from thalia.regions.cerebellum_region import CerebellumConfig, Cerebellum
from thalia.regions.striatum import StriatumConfig, Striatum


class TestHelperFunctions:
    """Test region size computation helper functions."""

    def test_compute_hippocampus_sizes(self):
        """Test hippocampus size computation from EC input."""
        ec_size = 100
        sizes = compute_hippocampus_sizes(ec_size)

        # Check all required keys present
        assert "dg_size" in sizes
        assert "ca3_size" in sizes
        assert "ca2_size" in sizes
        assert "ca1_size" in sizes

        # Check ratios are correct
        assert sizes["dg_size"] == 400  # 4x expansion
        assert sizes["ca3_size"] == 200  # 0.5 of DG
        assert sizes["ca2_size"] == 100  # 0.25 of DG
        assert sizes["ca1_size"] == 200  # Equal to CA3

    def test_compute_cortex_layer_sizes(self):
        """Test cortex layer size computation from input."""
        input_size = 100
        sizes = compute_cortex_layer_sizes(input_size)

        # Check all required keys present
        assert "l4_size" in sizes
        assert "l23_size" in sizes
        assert "l5_size" in sizes
        assert "total_size" in sizes

        # Check ratios are correct
        assert sizes["l4_size"] == 150  # 1.5x input
        assert sizes["l23_size"] == 300  # 2x L4
        assert sizes["l5_size"] == 150  # 0.5x L23
        assert sizes["total_size"] == 600

    def test_compute_thalamus_sizes(self):
        """Test thalamus size computation."""
        relay_size = 100
        sizes = compute_thalamus_sizes(relay_size)

        # Check all required keys present
        assert "relay_size" in sizes
        assert "trn_size" in sizes

        # Check default ratio
        assert sizes["relay_size"] == 100
        assert sizes["trn_size"] == 30  # 0.3 of relay

        # Check custom ratio
        sizes_custom = compute_thalamus_sizes(relay_size, trn_ratio=0.5)
        assert sizes_custom["trn_size"] == 50

    def test_compute_multisensory_sizes(self):
        """Test multisensory pool size computation."""
        total_size = 1000
        sizes = compute_multisensory_sizes(total_size)

        # Check all required keys present
        assert "visual_size" in sizes
        assert "auditory_size" in sizes
        assert "language_size" in sizes
        assert "integration_size" in sizes
        assert "total_size" in sizes

        # Check default ratios
        assert sizes["visual_size"] == 300  # 0.3
        assert sizes["auditory_size"] == 300  # 0.3
        assert sizes["language_size"] == 200  # 0.2
        assert sizes["integration_size"] == 200  # Remainder (0.2)
        assert sizes["total_size"] == 1000

        # Check they sum to total
        computed_total = (
            sizes["visual_size"] +
            sizes["auditory_size"] +
            sizes["language_size"] +
            sizes["integration_size"]
        )
        assert computed_total == total_size

    def test_compute_cerebellum_sizes(self):
        """Test cerebellum size computation."""
        purkinje_size = 100
        sizes = compute_cerebellum_sizes(purkinje_size)

        # Check all required keys present
        assert "granule_size" in sizes
        assert "purkinje_size" in sizes

        # Check default expansion
        assert sizes["purkinje_size"] == 100
        assert sizes["granule_size"] == 400  # 4x expansion

        # Check custom expansion
        sizes_custom = compute_cerebellum_sizes(purkinje_size, granule_expansion=2.0)
        assert sizes_custom["granule_size"] == 200

    def test_compute_striatum_sizes(self):
        """Test striatum size computation."""
        n_actions = 4
        neurons_per_action = 10
        sizes = compute_striatum_sizes(n_actions, neurons_per_action)

        # Check all required keys present
        assert "d1_size" in sizes
        assert "d2_size" in sizes
        assert "total_size" in sizes
        assert "n_actions" in sizes

        # Check sizes
        assert sizes["total_size"] == 40  # 4 * 10
        assert sizes["d1_size"] == 20  # Half (0.5 ratio)
        assert sizes["d2_size"] == 20  # Half
        assert sizes["d1_size"] + sizes["d2_size"] == sizes["total_size"]
        assert sizes["n_actions"] == 4


class TestRegionInitialization:
    """Test that regions initialize with correct explicit sizes."""

    def test_hippocampus_explicit_sizes(self):
        """Test hippocampus uses explicit sizes from config."""
        ec_size = 100
        sizes = compute_hippocampus_sizes(ec_size)

        # Calculate total neurons
        n_neurons = sizes["dg_size"] + sizes["ca3_size"] + sizes["ca2_size"] + sizes["ca1_size"]

        config = HippocampusConfig(
            n_input=ec_size,
            n_output=sizes["ca1_size"],
            n_neurons=n_neurons,
            dg_size=sizes["dg_size"],
            ca3_size=sizes["ca3_size"],
            ca2_size=sizes["ca2_size"],
            ca1_size=sizes["ca1_size"],
            device="cpu",
        )

        hippocampus = Hippocampus(config)

        # Check sizes match config
        assert hippocampus.dg_size == sizes["dg_size"]
        assert hippocampus.ca3_size == sizes["ca3_size"]
        assert hippocampus.ca2_size == sizes["ca2_size"]
        assert hippocampus.ca1_size == sizes["ca1_size"]

    def test_cortex_explicit_sizes(self):
        """Test cortex uses explicit sizes from config."""
        input_size = 100
        sizes = compute_cortex_layer_sizes(input_size)

        # L6 split: L6a and L6b are each half of L5 size
        l6a_size = sizes["l5_size"] // 2
        l6b_size = sizes["l5_size"] - l6a_size  # Handle odd sizes

        # LayeredCortex expects n_output = l23_size + l5_size (L4 is input layer)
        n_output = sizes["l23_size"] + sizes["l5_size"]
        n_neurons = sizes["l4_size"] + sizes["l23_size"] + sizes["l5_size"] + l6a_size + l6b_size

        config = LayeredCortexConfig(
            n_input=input_size,
            n_output=n_output,
            n_neurons=n_neurons,
            l4_size=sizes["l4_size"],
            l23_size=sizes["l23_size"],
            l5_size=sizes["l5_size"],
            l6a_size=l6a_size,
            l6b_size=l6b_size,
            device="cpu",
        )

        cortex = LayeredCortex(config)

        # Check sizes match config
        assert cortex.l4_size == sizes["l4_size"]
        assert cortex.l23_size == sizes["l23_size"]
        assert cortex.l5_size == sizes["l5_size"]

    def test_thalamus_explicit_sizes(self):
        """Test thalamus uses explicit sizes from config."""
        relay_size = 100
        sizes = compute_thalamus_sizes(relay_size)
        n_neurons = sizes["relay_size"] + sizes["trn_size"]

        config = ThalamicRelayConfig(
            n_input=50,
            n_output=relay_size,
            n_neurons=n_neurons,
            relay_size=sizes["relay_size"],
            trn_size=sizes["trn_size"],
            device="cpu",
        )

        thalamus = ThalamicRelay(config)

        # Check sizes match config
        assert thalamus.n_relay == sizes["relay_size"]
        assert thalamus.n_trn == sizes["trn_size"]

    def test_multisensory_explicit_sizes(self):
        """Test multisensory uses explicit sizes from config."""
        total_size = 1000
        sizes = compute_multisensory_sizes(total_size)

        config = MultimodalIntegrationConfig(
            n_input=300,
            n_output=total_size,
            n_neurons=total_size,
            visual_pool_size=sizes["visual_size"],
            auditory_pool_size=sizes["auditory_size"],
            language_pool_size=sizes["language_size"],
            integration_pool_size=sizes["integration_size"],
            visual_input_size=100,
            auditory_input_size=100,
            language_input_size=100,
            device="cpu",
        )

        multisensory = MultimodalIntegration(config)

        # Check sizes match config
        assert multisensory.visual_pool_size == sizes["visual_size"]
        assert multisensory.auditory_pool_size == sizes["auditory_size"]
        assert multisensory.language_pool_size == sizes["language_size"]
        assert multisensory.integration_pool_size == sizes["integration_size"]

    def test_cerebellum_explicit_sizes(self):
        """Test cerebellum uses explicit sizes from config."""
        purkinje_size = 100
        sizes = compute_cerebellum_sizes(purkinje_size)
        n_neurons = sizes["granule_size"] + sizes["purkinje_size"]

        config = CerebellumConfig(
            n_input=50,
            n_output=purkinje_size,
            n_neurons=n_neurons,
            granule_size=sizes["granule_size"],
            purkinje_size=sizes["purkinje_size"],
            use_enhanced_microcircuit=True,
            device="cpu",
        )

        cerebellum = Cerebellum(config)

        # Check sizes match config (when enhanced)
        assert cerebellum.cerebellum_config.purkinje_size == sizes["purkinje_size"]
        assert cerebellum.cerebellum_config.granule_size == sizes["granule_size"]

    def test_striatum_explicit_sizes(self):
        """Test striatum uses explicit sizes from config."""
        n_actions = 4
        neurons_per_action = 10
        sizes = compute_striatum_sizes(n_actions, neurons_per_action)

        config = StriatumConfig(
            n_input=100,
            n_output=sizes["total_size"],
            n_neurons=sizes["total_size"],
            d1_size=sizes["d1_size"],
            d2_size=sizes["d2_size"],
            n_actions=sizes["n_actions"],
            neurons_per_action=sizes["neurons_per_action"],
            population_coding=True,
            device="cpu",
        )

        striatum = Striatum(config)

        # Check sizes stored in config
        assert striatum.striatum_config.d1_size == sizes["d1_size"]
        assert striatum.striatum_config.d2_size == sizes["d2_size"]
        assert striatum.n_actions == sizes["n_actions"]
        assert striatum.neurons_per_action == sizes["neurons_per_action"]


class TestGrowthMethods:
    """Test that growth methods preserve explicit size fields."""

    def test_thalamus_grow_output_updates_sizes(self):
        """Test thalamus grow_output updates relay_size and trn_size."""
        relay_size = 100
        sizes = compute_thalamus_sizes(relay_size)

        config = ThalamicRelayConfig(
            n_input=50,
            n_output=relay_size,
            relay_size=sizes["relay_size"],
            trn_size=sizes["trn_size"],
            device="cpu",
        )

        thalamus = ThalamicRelay(config)
        initial_relay = thalamus.n_relay
        initial_trn = thalamus.n_trn

        # Grow output
        n_new = 20
        thalamus.grow_output(n_new)

        # Check instance variables updated
        assert thalamus.n_relay == initial_relay + n_new
        # TRN grows proportionally
        expected_trn = int((initial_relay + n_new) * initial_trn / initial_relay)
        assert thalamus.n_trn == expected_trn

        # Check config fields updated
        assert thalamus.config.relay_size == thalamus.n_relay
        assert thalamus.config.trn_size == thalamus.n_trn
        assert thalamus.thalamus_config.relay_size == thalamus.n_relay
        assert thalamus.thalamus_config.trn_size == thalamus.n_trn

    def test_multisensory_grow_output_updates_pool_sizes(self):
        """Test multisensory grow_output updates pool size fields."""
        total_size = 1000
        sizes = compute_multisensory_sizes(total_size)

        config = MultimodalIntegrationConfig(
            n_input=300,
            n_output=total_size,
            visual_pool_size=sizes["visual_size"],
            auditory_pool_size=sizes["auditory_size"],
            language_pool_size=sizes["language_size"],
            integration_pool_size=sizes["integration_size"],
            visual_input_size=100,
            auditory_input_size=100,
            language_input_size=100,
            device="cpu",
        )

        multisensory = MultimodalIntegration(config)
        initial_visual = multisensory.visual_pool_size
        initial_auditory = multisensory.auditory_pool_size
        initial_language = multisensory.language_pool_size
        initial_integration = multisensory.integration_pool_size

        # Grow output
        n_new = 100
        multisensory.grow_output(n_new)

        # Check instance variables updated (growth distributed by ratio)
        assert multisensory.visual_pool_size > initial_visual
        assert multisensory.auditory_pool_size > initial_auditory
        assert multisensory.language_pool_size > initial_language
        assert multisensory.integration_pool_size > initial_integration

        # Check config fields match instance variables
        assert multisensory.config.visual_pool_size == multisensory.visual_pool_size
        assert multisensory.config.auditory_pool_size == multisensory.auditory_pool_size
        assert multisensory.config.language_pool_size == multisensory.language_pool_size
        assert multisensory.config.integration_pool_size == multisensory.integration_pool_size

        # Check total is correct
        total_pool = (
            multisensory.visual_pool_size +
            multisensory.auditory_pool_size +
            multisensory.language_pool_size +
            multisensory.integration_pool_size
        )
        assert total_pool == total_size + n_new

    def test_cerebellum_grow_output_updates_purkinje_size(self):
        """Test cerebellum grow_output updates purkinje_size."""
        purkinje_size = 100
        sizes = compute_cerebellum_sizes(purkinje_size)

        config = CerebellumConfig(
            n_input=50,
            n_output=purkinje_size,
            granule_size=sizes["granule_size"],
            purkinje_size=sizes["purkinje_size"],
            use_enhanced_microcircuit=True,
            device="cpu",
        )

        cerebellum = Cerebellum(config)

        # Grow output
        n_new = 20
        cerebellum.grow_output(n_new)

        # Check config field updated
        assert cerebellum.config.purkinje_size == purkinje_size + n_new
        assert cerebellum.cerebellum_config.purkinje_size == purkinje_size + n_new

    def test_striatum_grow_output_updates_d1_d2_sizes(self):
        """Test striatum grow_output updates d1_size and d2_size."""
        n_actions = 4
        neurons_per_action = 10
        sizes = compute_striatum_sizes(n_actions, neurons_per_action)

        config = StriatumConfig(
            n_input=100,
            n_output=sizes["total_size"],
            d1_size=sizes["d1_size"],
            d2_size=sizes["d2_size"],
            n_actions=sizes["n_actions"],
            neurons_per_action=sizes["neurons_per_action"],
            population_coding=True,
            device="cpu",
        )

        striatum = Striatum(config)
        initial_d1 = striatum.striatum_config.d1_size
        initial_d2 = striatum.striatum_config.d2_size
        initial_total = initial_d1 + initial_d2

        # Grow output (add 1 action = 10 neurons)
        n_new = 1
        striatum.grow_output(n_new)

        # Check config fields updated
        # New total = initial + (neurons_per_action * n_new)
        new_total = initial_total + (neurons_per_action * n_new)
        new_d1 = striatum.striatum_config.d1_size
        new_d2 = striatum.striatum_config.d2_size

        # Verify growth happened
        assert new_d1 > initial_d1
        assert new_d2 > initial_d2
        assert new_d1 + new_d2 == new_total

        # Verify configs match
        assert striatum.config.d1_size == new_d1
        assert striatum.config.d2_size == new_d2
        assert striatum.striatum_config.d1_size == new_d1
        assert striatum.striatum_config.d2_size == new_d2
        assert striatum.striatum_config.d2_size == new_d2


class TestEdgeCases:
    """Test edge cases for size computation and growth."""

    def test_zero_input_size(self):
        """Test helper functions with zero input size."""
        # Should return all zeros
        sizes = compute_hippocampus_sizes(0)
        assert sizes["dg_size"] == 0
        assert sizes["ca3_size"] == 0
        assert sizes["ca2_size"] == 0
        assert sizes["ca1_size"] == 0

    def test_large_growth(self):
        """Test that large growth operations work correctly."""
        relay_size = 50
        sizes = compute_thalamus_sizes(relay_size)

        config = ThalamicRelayConfig(
            n_input=50,
            n_output=relay_size,
            relay_size=sizes["relay_size"],
            trn_size=sizes["trn_size"],
            device="cpu",
        )

        thalamus = ThalamicRelay(config)

        # Grow by 1000 (20x original size)
        thalamus.grow_output(1000)

        # Should not crash, sizes should be reasonable
        assert thalamus.n_relay == 1050
        assert thalamus.config.relay_size == 1050
        assert thalamus.n_trn > 0  # Should grow proportionally

    def test_minimal_sizes(self):
        """Test that minimal sizes (1 neuron) work."""
        config = ThalamicRelayConfig(
            n_input=1,
            n_output=1,
            relay_size=1,
            trn_size=1,
            device="cpu",
        )

        thalamus = ThalamicRelay(config)

        # Should initialize without error
        assert thalamus.n_relay == 1
        assert thalamus.n_trn == 1

        # Forward pass should work
        input_spikes = torch.zeros(1, device="cpu")
        output = thalamus(input_spikes)
        assert output.shape == (1,)

    def test_multisensory_zero_input_growth(self):
        """Test multisensory handles zero input sizes correctly."""
        # Create with zero inputs
        config = MultimodalIntegrationConfig(
            n_input=0,
            n_output=1000,
            visual_pool_size=300,
            auditory_pool_size=300,
            language_pool_size=200,
            integration_pool_size=200,
            visual_input_size=0,
            auditory_input_size=0,
            language_input_size=0,
            device="cpu",
        )

        multisensory = MultimodalIntegration(config)

        # Grow input (edge case: no current input)
        n_new = 100
        multisensory.grow_input(n_new)

        # Should distribute evenly when starting from zero
        assert multisensory.config.visual_input_size > 0
        assert multisensory.config.auditory_input_size > 0
        assert multisensory.config.language_input_size > 0
