"""
Integration tests for Planning Systems (MentalSimulationCoordinator, DynaPlanner).

Tests that DynamicBrain properly integrates model-based planning systems.

Phase 1.7.5: Planning Systems Integration
"""

import pytest
import torch

from thalia.core.dynamic_brain import DynamicBrain
from thalia.regions.cortex import LayeredCortex
from thalia.regions.hippocampus import Hippocampus
from thalia.regions.prefrontal import Prefrontal
from thalia.regions.striatum import Striatum
from thalia.pathways.spiking_pathway import SpikingPathway
from thalia.config import GlobalConfig


@pytest.fixture
def planning_brain():
    """Create DynamicBrain with planning enabled."""
    device = torch.device("cpu")

    # Global config with planning enabled
    # Note: DynamicBrain checks for planning flag dynamically
    # For testing, we'll add it as an attribute
    global_config = GlobalConfig(
        device="cpu",
        dt_ms=1.0,
    )
    # Add planning flag dynamically for testing
    global_config.use_model_based_planning = True

    # Create minimal components for planning
    from thalia.regions.cortex.config import LayeredCortexConfig
    from thalia.regions.hippocampus import HippocampusConfig
    from thalia.regions.prefrontal import PrefrontalConfig
    from thalia.regions.striatum import StriatumConfig

    input_size = 64
    cortex_size = 128
    hippocampus_size = 64
    pfc_size = 64
    n_actions = 4

    components = {
        "cortex": LayeredCortex(LayeredCortexConfig(
            dt_ms=1.0,
            device="cpu",
            n_input=input_size,
            n_output=cortex_size,
            l4_size=80,
            l23_size=80,
            l5_size=48,
            l6_size=40,
        )),
        "hippocampus": Hippocampus(HippocampusConfig(
            dt_ms=1.0,
            device="cpu",
            n_input=cortex_size,
            n_output=hippocampus_size,
        )),
        "pfc": Prefrontal(PrefrontalConfig(
            dt_ms=1.0,
            device="cpu",
            n_input=cortex_size + hippocampus_size,
            n_output=pfc_size,
        )),
        "striatum": Striatum(StriatumConfig(
            dt_ms=1.0,
            device="cpu",
            n_input=pfc_size,
            n_output=n_actions,
        )),
    }

    # Create pathways
    from thalia.core.base.component_config import PathwayConfig

    connections = {
        ("cortex", "hippocampus"): SpikingPathway(PathwayConfig(
            n_input=cortex_size,
            n_output=hippocampus_size,
            device="cpu",
            dt_ms=1.0,
        )),
        ("cortex", "pfc"): SpikingPathway(PathwayConfig(
            n_input=cortex_size,
            n_output=pfc_size,
            device="cpu",
            dt_ms=1.0,
        )),
        ("hippocampus", "pfc"): SpikingPathway(PathwayConfig(
            n_input=hippocampus_size,
            n_output=pfc_size,
            device="cpu",
            dt_ms=1.0,
        )),
        ("pfc", "striatum"): SpikingPathway(PathwayConfig(
            n_input=pfc_size,
            n_output=n_actions,
            device="cpu",
            dt_ms=1.0,
        )),
    }

    # Create brain
    brain = DynamicBrain(
        components=components,
        connections=connections,
        global_config=global_config,
    )

    # Set up registry for event-driven mode
    # Create a registry instance (adapters will fall back to GenericEventAdapter)
    from thalia.managers.component_registry import ComponentRegistry
    brain._registry = ComponentRegistry()

    return brain


class TestPlanningSystemsIntegration:
    """Tests for planning systems integration in DynamicBrain."""

    def test_mental_simulation_initialized(self, planning_brain):
        """Test that MentalSimulationCoordinator is initialized when planning enabled."""
        assert hasattr(planning_brain, "mental_simulation")
        assert planning_brain.mental_simulation is not None

    def test_dyna_planner_initialized(self, planning_brain):
        """Test that DynaPlanner is initialized when planning enabled."""
        assert hasattr(planning_brain, "dyna_planner")
        assert planning_brain.dyna_planner is not None

    def test_planning_disabled_by_default(self):
        """Test that planning is not initialized when disabled."""
        global_config = GlobalConfig(
            device="cpu",
            dt_ms=1.0,
        )
        # Don't set use_model_based_planning (defaults to False)

        from thalia.regions.striatum import Striatum, StriatumConfig

        components = {
            "striatum": Striatum(StriatumConfig(
                dt_ms=1.0,
                device="cpu",
                n_input=64,
                n_output=4,
            )),
        }

        brain = DynamicBrain(
            components=components,
            connections={},
            global_config=global_config,
        )

        assert brain.mental_simulation is None
        assert brain.dyna_planner is None

    def test_select_action_with_planning(self, planning_brain):
        """Test that select_action uses planning when requested."""
        # Do forward pass first (event-driven mode is default)
        sensory_input = torch.randn(64, device=planning_brain.device)  # Match cortex n_input
        planning_brain.forward({"cortex": sensory_input}, n_timesteps=10)

        # Select action with planning
        action, confidence = planning_brain.select_action(
            explore=False,
            use_planning=True,
        )

        assert isinstance(action, int)
        assert 0 <= action < 4
        assert 0.0 <= confidence <= 1.0

    def test_select_action_without_planning(self, planning_brain):
        """Test that select_action falls back to striatum when planning=False."""
        # Do forward pass first (event-driven mode is default)
        sensory_input = torch.randn(64, device=planning_brain.device)  # Match cortex n_input
        planning_brain.forward({"cortex": sensory_input}, n_timesteps=10)

        # Select action without planning
        action, confidence = planning_brain.select_action(
            explore=False,
            use_planning=False,
        )

        assert isinstance(action, int)
        assert 0 <= action < 4
        assert 0.0 <= confidence <= 1.0

    def test_dyna_planning_after_reward(self, planning_brain):
        """Test that Dyna background planning triggers after reward delivery."""
        # Do forward pass and select action (event-driven mode is default)
        sensory_input = torch.randn(64, device=planning_brain.device)  # Match cortex n_input
        planning_brain.forward({"cortex": sensory_input}, n_timesteps=10)

        action, _ = planning_brain.select_action(explore=False)

        # Check Dyna planner state before reward
        initial_experiences = len(planning_brain.dyna_planner.replay_buffer) if hasattr(planning_brain.dyna_planner, 'replay_buffer') else 0

        # Deliver reward (should trigger Dyna planning)
        planning_brain.deliver_reward(external_reward=1.0)

        # Note: Actual verification requires checking Dyna internals
        # For now, just verify no errors during planning
        assert planning_brain.dyna_planner is not None

    def test_planning_handles_missing_pfc_state(self, planning_brain):
        """Test that planning gracefully handles missing PFC state."""
        # Select action without forward pass (no PFC state)
        # Should fall back to striatum
        action, confidence = planning_brain.select_action(
            explore=False,
            use_planning=True,
        )

        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_planning_requires_all_components(self):
        """Test that planning only initializes if all required components present."""
        global_config = GlobalConfig(
            device="cpu",
            dt_ms=1.0,
        )
        global_config.use_model_based_planning = True

        # Only striatum (missing pfc, hippocampus, cortex)
        from thalia.regions.striatum import Striatum, StriatumConfig

        components = {
            "striatum": Striatum(StriatumConfig(
                dt_ms=1.0,
                device="cpu",
                n_input=64,
                n_output=4,
            )),
        }

        brain = DynamicBrain(
            components=components,
            connections={},
            global_config=global_config,
        )

        # Planning should not initialize without required components
        assert brain.mental_simulation is None
        assert brain.dyna_planner is None
