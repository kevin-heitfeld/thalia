"""
Tests for brain surgery tools.

Tests lesion, ablation, and growth operations on trained brains.
"""

import pytest
import torch

from thalia.surgery import (
    lesion_region,
    partial_lesion,
    temporary_lesion,
    restore_region,
    ablate_pathway,
    restore_pathway,
    freeze_region,
    unfreeze_region,
    freeze_pathway,
    unfreeze_pathway,
)
from thalia.core.brain import EventDrivenBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes


@pytest.fixture
def test_brain():
    """Create minimal brain for testing."""
    config = ThaliaConfig(
        global_=GlobalConfig(device="cpu", dt_ms=1.0),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=10,
                thalamus_size=20,
                cortex_size=30,
                hippocampus_size=40,
                pfc_size=20,
                n_actions=5,
            ),
        ),
    )
    return EventDrivenBrain.from_config(config)


def test_lesion_region_silences_region(test_brain):
    """Test complete region lesion."""
    # Get initial weights
    initial_weights = {
        name: param.data.clone()
        for name, param in test_brain.hippocampus.impl.named_parameters()
    }

    # Lesion hippocampus
    lesion_region(test_brain, "hippocampus")

    # Check weights are zeroed
    for name, param in test_brain.hippocampus.impl.named_parameters():
        if param.requires_grad:
            assert torch.all(param.data == 0.0), f"{name} not zeroed"

    # Check plasticity disabled
    assert test_brain.hippocampus.impl.plasticity_enabled is False


def test_lesion_region_alternate_names(test_brain):
    """Test lesion with alternate region names."""
    # Should work with both names
    lesion_region(test_brain, "pfc")
    assert test_brain.pfc.impl.plasticity_enabled is False

    restore_region(test_brain, "pfc")

    lesion_region(test_brain, "prefrontal")
    assert test_brain.pfc.impl.plasticity_enabled is False


def test_partial_lesion(test_brain):
    """Test partial region lesion."""
    # Lesion 30% of cortex
    partial_lesion(test_brain, "cortex", lesion_fraction=0.3)

    # Should still have some non-zero weights
    has_nonzero = False
    for param in test_brain.cortex.impl.parameters():
        if param.requires_grad and torch.any(param.data != 0):
            has_nonzero = True
            break

    assert has_nonzero, "All weights zeroed (should be partial)"


def test_temporary_lesion(test_brain):
    """Test temporary lesion with context manager."""
    # Save initial state
    initial_weights = list(test_brain.striatum.impl.parameters())[0].data.clone()

    # Temporary lesion
    with temporary_lesion(test_brain, "striatum"):
        # Inside context: lesioned
        lesioned_weights = list(test_brain.striatum.impl.parameters())[0].data.clone()
        assert torch.all(lesioned_weights == 0.0)

    # Outside context: restored
    restored_weights = list(test_brain.striatum.impl.parameters())[0].data.clone()
    assert torch.allclose(restored_weights, initial_weights, atol=1e-6)


def test_restore_region(test_brain):
    """Test region restoration after lesion."""
    # Save initial weights
    initial_weights = {
        name: param.data.clone()
        for name, param in test_brain.cerebellum.impl.named_parameters()
    }

    # Lesion and restore
    lesion_region(test_brain, "cerebellum")
    restore_region(test_brain, "cerebellum")

    # Check weights restored
    for name, param in test_brain.cerebellum.impl.named_parameters():
        if name in initial_weights:
            assert torch.allclose(
                param.data,
                initial_weights[name],
                atol=1e-6,
            ), f"{name} not restored"


def test_restore_without_lesion_raises(test_brain):
    """Test restore fails if no lesion was saved."""
    with pytest.raises(ValueError, match="No saved state"):
        restore_region(test_brain, "thalamus")


def test_ablate_pathway(test_brain):
    """Test pathway ablation."""
    # Get pathway
    pathway = test_brain.pathways.pathways["cortex_to_hippocampus"]

    # Ablate
    ablate_pathway(test_brain, "cortex_to_hippocampus")

    # Check weights zeroed
    for param in pathway.parameters():
        if param.requires_grad:
            assert torch.all(param.data == 0.0)

    # Check plasticity disabled
    assert pathway.plasticity_enabled is False


def test_ablate_pathway_alternate_names(test_brain):
    """Test ablation with alternate pathway names."""
    ablate_pathway(test_brain, "cortex_to_hippo")
    pathway = test_brain.pathways.pathways["cortex_to_hippocampus"]
    assert pathway.plasticity_enabled is False


def test_restore_pathway(test_brain):
    """Test pathway restoration."""
    pathway_name = "cortex_to_pfc"
    pathway = test_brain.pathways.pathways[pathway_name]

    # Save initial weights
    initial_weights = {
        name: param.data.clone()
        for name, param in pathway.named_parameters()
    }

    # Ablate and restore
    ablate_pathway(test_brain, pathway_name)
    restore_pathway(test_brain, pathway_name)

    # Check restored
    for name, param in pathway.named_parameters():
        if name in initial_weights:
            assert torch.allclose(
                param.data,
                initial_weights[name],
                atol=1e-6,
            )


def test_freeze_region(test_brain):
    """Test freezing region plasticity."""
    freeze_region(test_brain, "cortex")

    # Check plasticity disabled
    assert test_brain.cortex.impl.plasticity_enabled is False

    # Check parameters frozen
    for param in test_brain.cortex.impl.parameters():
        assert param.requires_grad is False


def test_unfreeze_region(test_brain):
    """Test unfreezing region plasticity."""
    freeze_region(test_brain, "hippocampus")
    unfreeze_region(test_brain, "hippocampus")

    # Check plasticity enabled
    assert test_brain.hippocampus.impl.plasticity_enabled is True

    # Check parameters unfrozen
    for param in test_brain.hippocampus.impl.parameters():
        assert param.requires_grad is True


def test_freeze_pathway(test_brain):
    """Test freezing pathway plasticity."""
    pathway_name = "cortex_to_striatum"
    pathway = test_brain.pathways.pathways[pathway_name]

    freeze_pathway(test_brain, pathway_name)

    # Check plasticity disabled
    assert pathway.plasticity_enabled is False

    # Check parameters frozen
    for param in pathway.parameters():
        assert param.requires_grad is False


def test_unfreeze_pathway(test_brain):
    """Test unfreezing pathway plasticity."""
    pathway_name = "pfc_to_striatum"
    pathway = test_brain.pathways.pathways[pathway_name]

    freeze_pathway(test_brain, pathway_name)
    unfreeze_pathway(test_brain, pathway_name)

    # Check plasticity enabled
    assert pathway.plasticity_enabled is True

    # Check parameters unfrozen
    for param in pathway.parameters():
        assert param.requires_grad is True


def test_lesion_invalid_region_raises(test_brain):
    """Test lesion with invalid region name."""
    with pytest.raises(ValueError, match="Unknown region"):
        lesion_region(test_brain, "invalid_region")


def test_ablate_invalid_pathway_raises(test_brain):
    """Test ablation with invalid pathway name."""
    with pytest.raises(ValueError, match="Unknown pathway"):
        ablate_pathway(test_brain, "invalid_pathway")
