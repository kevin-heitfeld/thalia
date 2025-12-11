"""
Pathway Manager for EventDrivenBrain.

Handles creation, tracking, and coordinated growth of all inter-region pathways.
"""

from typing import Dict, List, Tuple, Any

from thalia.config.base import PathwayConfig
from thalia.integration.spiking_pathway import SpikingPathway
from thalia.integration.spiking_pathway import SpikingLearningRule, TemporalCoding
from thalia.integration.pathways.spiking_attention import (
    SpikingAttentionPathway,
    SpikingAttentionPathwayConfig,
)
from thalia.integration.pathways.spiking_replay import (
    SpikingReplayPathway,
    SpikingReplayPathwayConfig,
)


class PathwayManager:
    """Manages all inter-region neural pathways in the brain.

    Responsibilities:
    - Create and configure all pathways
    - Track region-pathway connections for coordinated growth
    - Provide unified access to pathways for checkpointing and diagnostics
    """

    def __init__(
        self,
        cortex_l23_size: int,
        cortex_l5_size: int,
        input_size: int,
        cortex_size: int,
        hippocampus_size: int,
        pfc_size: int,
        n_actions: int,
        neurons_per_action: int,
        dt_ms: float,
        device: str,
    ):
        """Initialize pathway manager.

        Args:
            cortex_l23_size: Size of cortex layer 2/3 output
            cortex_l5_size: Size of cortex layer 5 output
            input_size: Input size for attention pathway
            cortex_size: Cortex size for replay pathway
            hippocampus_size: Hippocampus output size
            pfc_size: PFC size
            n_actions: Number of possible actions
            neurons_per_action: Striatum neurons per action
            dt_ms: Simulation timestep in milliseconds
            device: Torch device
        """
        self.dt_ms = dt_ms
        self.device = device

        # Store sizes for growth coordination
        self._sizes = {
            'cortex_l23': cortex_l23_size,
            'cortex_l5': cortex_l5_size,
            'input': input_size,
            'cortex': cortex_size,
            'hippocampus': hippocampus_size,
            'pfc': pfc_size,
            'n_actions': n_actions,
            'neurons_per_action': neurons_per_action,
        }

        # Create pathways
        self._create_pathways()

        # Track region-pathway connections for coordinated growth
        self._setup_connection_tracking()

    def _create_pathways(self) -> None:
        """Create all inter-region pathways."""
        # 1. Cortex L2/3 → Hippocampus (encoding pathway)
        self.cortex_to_hippo = SpikingPathway(
            PathwayConfig(
                n_input=self._sizes['cortex_l23'],
                n_output=self._sizes['cortex_l23'],  # Match hippo input
                learning_rule=SpikingLearningRule.STDP,
                temporal_coding=TemporalCoding.PHASE,  # Theta phase coding
                stdp_lr=0.001,
                dt_ms=self.dt_ms,
                device=self.device,
            )
        )

        # 2. Cortex L5 → Striatum (action selection pathway)
        self.cortex_to_striatum = SpikingPathway(
            PathwayConfig(
                n_input=self._sizes['cortex_l5'],
                n_output=self._sizes['cortex_l5'],  # Match striatum input
                learning_rule=SpikingLearningRule.DOPAMINE_STDP,  # Reward-modulated
                temporal_coding=TemporalCoding.RATE,
                stdp_lr=0.002,
                dt_ms=self.dt_ms,
                device=self.device,
            )
        )

        # 3. Cortex L2/3 → PFC (working memory input)
        self.cortex_to_pfc = SpikingPathway(
            PathwayConfig(
                n_input=self._sizes['cortex_l23'],
                n_output=self._sizes['cortex_l23'],  # PFC receives cortex + hippo
                learning_rule=SpikingLearningRule.STDP,
                temporal_coding=TemporalCoding.SYNCHRONY,  # Binding via synchrony
                stdp_lr=0.0015,
                dt_ms=self.dt_ms,
                device=self.device,
            )
        )

        # 4. Hippocampus → PFC (episodic to working memory)
        self.hippo_to_pfc = SpikingPathway(
            PathwayConfig(
                n_input=self._sizes['hippocampus'],
                n_output=self._sizes['hippocampus'],
                learning_rule=SpikingLearningRule.STDP,
                temporal_coding=TemporalCoding.PHASE,  # Theta-coupled
                stdp_lr=0.001,
                dt_ms=self.dt_ms,
                device=self.device,
            )
        )

        # 5. Hippocampus → Striatum (context for action selection)
        self.hippo_to_striatum = SpikingPathway(
            PathwayConfig(
                n_input=self._sizes['hippocampus'],
                n_output=self._sizes['hippocampus'],
                learning_rule=SpikingLearningRule.DOPAMINE_STDP,  # Reward-modulated
                temporal_coding=TemporalCoding.PHASE,
                stdp_lr=0.0015,
                dt_ms=self.dt_ms,
                device=self.device,
            )
        )

        # 6. PFC → Striatum (goal-directed control)
        self.pfc_to_striatum = SpikingPathway(
            PathwayConfig(
                n_input=self._sizes['pfc'],
                n_output=self._sizes['pfc'],
                learning_rule=SpikingLearningRule.DOPAMINE_STDP,  # Reward-modulated
                temporal_coding=TemporalCoding.RATE,
                stdp_lr=0.002,
                dt_ms=self.dt_ms,
                device=self.device,
            )
        )

        # 7. Striatum → Cerebellum (action refinement)
        striatum_size = self._sizes['n_actions'] * self._sizes['neurons_per_action']
        self.striatum_to_cerebellum = SpikingPathway(
            PathwayConfig(
                n_input=striatum_size,
                n_output=striatum_size,
                learning_rule=SpikingLearningRule.STDP,
                temporal_coding=TemporalCoding.LATENCY,  # Precise timing for motor control
                stdp_lr=0.001,
                dt_ms=self.dt_ms,
                device=self.device,
            )
        )

        # 8. PFC → Cortex (top-down attention modulation) [SPECIALIZED]
        self.attention = SpikingAttentionPathway(
            SpikingAttentionPathwayConfig(
                n_input=self._sizes['pfc'],
                n_output=self._sizes['input'],
                device=self.device,
            )
        )

        # 9. Hippocampus → Cortex (replay/consolidation during sleep) [SPECIALIZED]
        self.replay = SpikingReplayPathway(
            SpikingReplayPathwayConfig(
                n_input=self._sizes['hippocampus'],
                n_output=self._sizes['cortex'],
                device=self.device,
            )
        )

    def _setup_connection_tracking(self) -> None:
        """Setup tracking of region-pathway connections for coordinated growth."""
        # Maps: region_name -> list of (pathway, dimension_type)
        # dimension_type: 'source' or 'target'
        self.region_connections: Dict[str, List[Tuple[Any, str]]] = {
            'cortex': [
                (self.cortex_to_hippo, 'source'),
                (self.cortex_to_striatum, 'source'),
                (self.cortex_to_pfc, 'source'),
                (self.replay, 'target'),
                (self.attention, 'target'),
            ],
            'hippocampus': [
                (self.cortex_to_hippo, 'target'),
                (self.hippo_to_pfc, 'source'),
                (self.hippo_to_striatum, 'source'),
                (self.replay, 'source'),
            ],
            'pfc': [
                (self.cortex_to_pfc, 'target'),
                (self.hippo_to_pfc, 'target'),
                (self.pfc_to_striatum, 'source'),
                (self.attention, 'source'),
            ],
            'striatum': [
                (self.cortex_to_striatum, 'target'),
                (self.hippo_to_striatum, 'target'),
                (self.pfc_to_striatum, 'target'),
                (self.striatum_to_cerebellum, 'source'),
            ],
            'cerebellum': [
                (self.striatum_to_cerebellum, 'target'),
            ],
        }

    def get_all_pathways(self) -> Dict[str, Any]:
        """Get dictionary of all pathways for iteration."""
        return {
            "cortex_to_hippo": self.cortex_to_hippo,
            "cortex_to_striatum": self.cortex_to_striatum,
            "cortex_to_pfc": self.cortex_to_pfc,
            "hippo_to_pfc": self.hippo_to_pfc,
            "hippo_to_striatum": self.hippo_to_striatum,
            "pfc_to_striatum": self.pfc_to_striatum,
            "striatum_to_cerebellum": self.striatum_to_cerebellum,
            "attention": self.attention,
            "replay": self.replay,
        }

    def grow_connected_pathways(
        self,
        region_name: str,
        growth_amount: int,
    ) -> None:
        """Grow all pathways connected to a region.

        Args:
            region_name: Name of the region that grew
            growth_amount: Number of neurons added to the region
        """
        if region_name not in self.region_connections:
            return

        for pathway, dimension_type in self.region_connections[region_name]:
            if hasattr(pathway, 'add_neurons'):
                if dimension_type == 'source':
                    pathway.add_neurons(source_growth=growth_amount, target_growth=0)
                else:  # target
                    pathway.add_neurons(source_growth=0, target_growth=growth_amount)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get pathway diagnostics."""
        diagnostics = {}
        for name, pathway in self.get_all_pathways().items():
            if hasattr(pathway, 'get_diagnostics'):
                diagnostics[name] = pathway.get_diagnostics()
        return diagnostics

    def get_state(self) -> Dict[str, Any]:
        """Get checkpoint state for all pathways."""
        state = {}
        for name, pathway in self.get_all_pathways().items():
            if hasattr(pathway, 'get_state'):
                state[name] = pathway.get_state()
        return state

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load checkpoint state for all pathways."""
        for name, pathway_state in state_dict.items():
            pathway = self.get_all_pathways().get(name)
            if pathway is not None and hasattr(pathway, 'load_state'):
                pathway.load_state(pathway_state)
