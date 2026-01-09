"""Integration test to verify mixin inheritance from NeuralRegion works correctly."""

import torch

from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from thalia.regions.cerebellum_region import Cerebellum, CerebellumConfig
from thalia.mixins.state_loading_mixin import StateLoadingMixin
from thalia.learning.strategy_mixin import LearningStrategyMixin


def test_neural_region_has_both_mixins():
    """Verify NeuralRegion provides both StateLoadingMixin and LearningStrategyMixin."""
    # Create a simple Prefrontal region
    config = PrefrontalConfig(input_size=64, n_neurons=128, device="cpu")
    pfc = Prefrontal(config)

    # Verify it has StateLoadingMixin methods
    assert hasattr(pfc, 'load_state')
    assert hasattr(pfc, '_restore_neuron_state')
    assert hasattr(pfc, '_restore_conductances')
    assert hasattr(pfc, '_restore_neuromodulators')
    assert hasattr(pfc, '_load_custom_state')

    # Verify it has LearningStrategyMixin methods
    assert hasattr(pfc, 'apply_strategy_learning')
    assert hasattr(pfc, 'learning_strategy')

    # Verify it's an instance of both mixins (via NeuralRegion)
    assert isinstance(pfc, StateLoadingMixin)
    assert isinstance(pfc, LearningStrategyMixin)


def test_cerebellum_inherits_mixins_from_base():
    """Verify Cerebellum gets mixins from NeuralRegion, not direct inheritance."""
    config = CerebellumConfig(
        n_input=100,
        n_output=50,
        device="cpu",
        use_enhanced_microcircuit=False,  # Simpler test
    )
    cerebellum = Cerebellum(config)

    # Verify StateLoadingMixin methods available
    assert hasattr(cerebellum, 'load_state')
    assert hasattr(cerebellum, '_restore_neuron_state')
    assert hasattr(cerebellum, '_load_custom_state')

    # Verify inheritance chain
    assert isinstance(cerebellum, StateLoadingMixin)
    assert isinstance(cerebellum, LearningStrategyMixin)


def test_state_loading_works_with_base_inheritance():
    """Verify state loading still works with base class inheritance."""
    config = PrefrontalConfig(input_size=32, n_neurons=64, device="cpu")
    pfc = Prefrontal(config)
    pfc.reset_state()

    # Get state
    state = pfc.get_state()

    # Modify state
    state.working_memory = torch.ones(64)
    state.dopamine = 0.8

    # Load state (should use mixin's load_state → _load_custom_state chain)
    pfc.load_state(state)

    # Verify restored
    assert torch.allclose(pfc.state.working_memory, torch.ones(64))
    assert pfc.state.dopamine == 0.8


def test_no_duplicate_mixin_inheritance():
    """Ensure regions don't inherit mixins twice (once from base, once directly)."""
    # Check Prefrontal class definition doesn't mention mixins
    from thalia.regions.prefrontal import Prefrontal
    import inspect

    # Get Prefrontal's direct base classes (should only be NeuralRegion)
    bases = Prefrontal.__bases__
    assert len(bases) == 1, f"Prefrontal should only inherit from NeuralRegion, got: {bases}"
    assert bases[0].__name__ == "NeuralRegion"

    # Similarly for Cerebellum
    from thalia.regions.cerebellum_region import Cerebellum
    bases = Cerebellum.__bases__
    assert len(bases) == 1, f"Cerebellum should only inherit from NeuralRegion, got: {bases}"
    assert bases[0].__name__ == "NeuralRegion"
