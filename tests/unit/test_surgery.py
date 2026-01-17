"""
Tests for brain surgery tools.

Tests lesion, ablation, and growth operations on trained brains.
"""

import pytest
import torch

from tests.utils import create_test_brain
from thalia.core.dynamic_brain import DynamicBrain
from thalia.surgery import (
    freeze_region,
    lesion_region,
    partial_lesion,
    restore_region,
    temporary_lesion,
    unfreeze_region,
)


@pytest.fixture
def test_brain() -> "DynamicBrain":
    """Create minimal brain for testing."""
    return create_test_brain(
        input_size=10,
        thalamus_size=20,
        cortex_size=30,
        hippocampus_size=40,
        pfc_size=20,
        n_actions=5,
    )


def test_lesion_region_silences_region(test_brain: "DynamicBrain"):
    """Test complete region lesion."""
    # Lesion hippocampus
    lesion_region(test_brain, "hippocampus")

    # Check weights are zeroed
    for name, param in test_brain.components["hippocampus"].named_parameters():
        if param.requires_grad:
            assert torch.all(param.data == 0.0), f"{name} not zeroed"

    # Check plasticity disabled
    assert test_brain.components["hippocampus"].plasticity_enabled is False


def test_lesion_region_alternate_names(test_brain: "DynamicBrain"):
    """Test lesion with alternate region names."""
    # Should work with both names
    lesion_region(test_brain, "pfc")
    assert test_brain.components["pfc"].plasticity_enabled is False

    restore_region(test_brain, "pfc")

    lesion_region(test_brain, "prefrontal")
    assert test_brain.components["pfc"].plasticity_enabled is False


def test_partial_lesion(test_brain: "DynamicBrain"):
    """Test partial region lesion."""
    # Lesion 30% of cortex
    partial_lesion(test_brain, "cortex", lesion_fraction=0.3)

    # Should still have some non-zero weights
    has_nonzero = False
    for param in test_brain.components["cortex"].parameters():
        if param.requires_grad and torch.any(param.data != 0):
            has_nonzero = True
            break

    assert has_nonzero, "All weights zeroed (should be partial)"


def test_temporary_lesion(test_brain: "DynamicBrain"):
    """Test temporary lesion with context manager."""
    # Save initial state
    initial_weights = list(test_brain.components["striatum"].parameters())[0].data.clone()

    # Temporary lesion
    with temporary_lesion(test_brain, "striatum"):
        # Inside context: lesioned
        lesioned_weights = list(test_brain.components["striatum"].parameters())[0].data.clone()
        assert torch.all(lesioned_weights == 0.0)

    # Outside context: restored
    restored_weights = list(test_brain.components["striatum"].parameters())[0].data.clone()
    assert torch.allclose(restored_weights, initial_weights, atol=1e-6)


def test_restore_region(test_brain: "DynamicBrain"):
    """Test region restoration after lesion."""
    # Save initial weights
    initial_weights = {
        name: param.data.clone()
        for name, param in test_brain.components["cerebellum"].named_parameters()
    }

    # Lesion and restore
    lesion_region(test_brain, "cerebellum")
    restore_region(test_brain, "cerebellum")

    # Check weights restored
    for name, param in test_brain.components["cerebellum"].named_parameters():
        if name in initial_weights:
            assert torch.allclose(
                param.data,
                initial_weights[name],
                atol=1e-6,
            ), f"{name} not restored"


def test_restore_without_lesion_raises(test_brain: "DynamicBrain"):
    """Test restore fails if no lesion was saved."""
    with pytest.raises(ValueError, match="No saved state"):
        restore_region(test_brain, "thalamus")


def test_freeze_region(test_brain: "DynamicBrain"):
    """Test freezing region plasticity."""
    freeze_region(test_brain, "cortex")

    # Check plasticity disabled
    assert test_brain.components["cortex"].plasticity_enabled is False

    # Check parameters frozen
    for param in test_brain.components["cortex"].parameters():
        assert param.requires_grad is False


def test_unfreeze_region(test_brain: "DynamicBrain"):
    """Test unfreezing region plasticity."""
    freeze_region(test_brain, "hippocampus")
    unfreeze_region(test_brain, "hippocampus")

    # Check plasticity enabled
    assert test_brain.components["hippocampus"].plasticity_enabled is True

    # Check parameters unfrozen
    for param in test_brain.components["hippocampus"].parameters():
        assert param.requires_grad is True


def test_lesion_invalid_region_raises(test_brain: "DynamicBrain"):
    """Test lesion with invalid region name."""
    with pytest.raises(ValueError, match="Unknown region"):
        lesion_region(test_brain, "invalid_region")
