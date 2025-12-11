"""
EventDrivenBrain: High-level brain system using event-driven architecture.

This module provides a brain system that uses the event-driven infrastructure
for biologically realistic parallel computation.

Key Differences from BrainSystem:
=================================
1. EVENT-DRIVEN: Regions communicate via events with axonal delays
2. PARALLEL-READY: Can run regions on separate processes
3. NATURAL RHYTHMS: Theta arrives at regions with phase offsets (via delays)
4. ASYNCHRONOUS: No artificial synchronization barriers

Architecture:
=============

All inter-region connections are EXPLICIT PATHWAYS (not just event routing):

    Sensory Input
         │
         ▼ SpikingPathway (5ms delay)
    ┌─────────┐
    │  CORTEX │
    │  (L4→   │
    │   L2/3→ │
    │   L5)   │
    └────┬────┘
         │
    ┌────┴───────┬──────────────┐
    │            │              │
    │(pathway 1) │(pathway 2)   │(pathway 3)
    ▼(3ms)       ▼(6ms)         ▼(5ms)
┌───────────┐ ┌─────┐        ┌──────────┐
│HIPPOCAMPUS│ │ PFC │        │ STRIATUM │
│ (DG→CA3→  │ │     │        │  (D1/D2) │
│   CA1)    │ │     │        │          │
└─────┬─────┘ └──┬──┘        └────┬─────┘
      │          │                │
 (pw 4)│     (pw 5)          (pw 6)│
      ▼(5ms)     │                │
    ┌─────┐      │                │
    │ PFC │◄─────┘                │
    │     │                       │
    └──┬──┘                       │
       │(pw 6)                    │
       ▼(4ms)                     │
    ┌──────────┐                  │
    │ STRIATUM │◄─────────────────┘
    │  (PFC→)  │
    └────┬─────┘
         │(pathway 7)
         ▼
    ┌──────────┐
    │CEREBELLUM│
    └────┬─────┘
         │
         ▼
    Motor Output

Specialized Pathways:
(pw 8) PFC → Cortex L2/3: Top-down attention (SpikingAttentionPathway)
(pw 9) Hippocampus → Cortex: Memory replay during sleep (SpikingReplayPathway)

All 9 pathways:
1. Cortex L2/3 → Hippocampus (encoding)
2. Cortex L5 → Striatum (action selection)
3. Cortex L2/3 → PFC (working memory input)
4. Hippocampus → PFC (episodic → working memory)
5. Hippocampus → Striatum (context for action)
6. PFC → Striatum (goal-directed control)
7. Striatum → Cerebellum (action refinement)
8. PFC → Cortex (attention modulation) [SPECIALIZED]
9. Hippocampus → Cortex (replay/consolidation) [SPECIALIZED]

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, Optional, Any
import torch
import torch.nn as nn

from thalia.events import (
    EventType, EventScheduler,
    SpikePayload,
)
from thalia.events.adapters import (
    EventDrivenCortex, EventDrivenHippocampus, EventDrivenPFC, EventDrivenStriatum,
    EventDrivenCerebellum, EventRegionConfig,
)
from thalia.events.parallel import ParallelExecutor
from .pathway_manager import PathwayManager
from .neuromodulator_manager import NeuromodulatorManager
from .neuron_constants import INTRINSIC_LEARNING_THRESHOLD
from .spike_utils import compute_firing_rate
from .diagnostics import (
    DiagnosticsManager,
    StriatumDiagnostics,
    HippocampusDiagnostics,
    BrainSystemDiagnostics,
)

# Robustness mechanisms
from ..diagnostics.criticality import CriticalityMonitor

# IO utilities
from ..io import CheckpointManager

# Import actual region implementations
from ..regions.cortex import LayeredCortex, LayeredCortexConfig
from ..regions.cortex.predictive_cortex import PredictiveCortex, PredictiveCortexConfig
from ..regions.hippocampus import Hippocampus, HippocampusConfig
from ..regions.prefrontal import Prefrontal, PrefrontalConfig
from ..regions.striatum import Striatum, StriatumConfig
from ..regions.cerebellum import Cerebellum, CerebellumConfig
from ..regions.theta_dynamics import TemporalIntegrationLayer

# Import config types
from ..config.brain_config import CortexType

if TYPE_CHECKING:
    from thalia.config import ThaliaConfig


@dataclass
class EventDrivenBrain(nn.Module):
    """
    Event-driven brain system with biologically realistic timing.

    Provides high-level trial APIs similar to BrainSystem, but uses
    event-driven infrastructure for natural temporal dynamics.

    Key features:
    1. Regions communicate via events with axonal delays
    2. Theta rhythm arrives at regions with natural phase offsets
    3. Spikes propagate with realistic timing
    4. Can run in parallel mode for performance

    Example:
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cuda"),
            brain=BrainConfig(sizes=RegionSizes(cortex_size=256)),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Process input (encoding, maintenance, or retrieval)
        result = brain.forward(sample_pattern, n_timesteps=15)
        result = brain.forward(None, n_timesteps=10)  # Maintenance period
        result = brain.forward(test_pattern, n_timesteps=15)

        # Action selection and learning
        action, confidence = brain.select_action()
        brain.deliver_reward(external_reward=1.0)  # Combines with intrinsic rewards
    """

    def __init__(self, config: "ThaliaConfig"):
        """Initialize EventDrivenBrain from ThaliaConfig.

        Args:
            config: ThaliaConfig with all settings

        Example:
            from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig

            config = ThaliaConfig(
                global_=GlobalConfig(device="cuda"),
                brain=BrainConfig(sizes=RegionSizes(cortex_size=256)),
            )
            brain = EventDrivenBrain(config)

        Note:
            Prefer using EventDrivenBrain.from_thalia_config(config) for clarity.

        Raises:
            ConfigValidationError: If configuration is invalid
        """
        # Validate configuration before initialization
        from thalia.config import validate_thalia_config
        validate_thalia_config(config)

        super().__init__()

        # Store ThaliaConfig directly
        self.thalia_config = config
        # Create a simple namespace for backwards compatibility with code that accesses self.config
        from types import SimpleNamespace
        self.config = SimpleNamespace(
            input_size=config.brain.sizes.input_size,
            cortex_size=config.brain.sizes.cortex_size,
            hippocampus_size=config.brain.sizes.hippocampus_size,
            pfc_size=config.brain.sizes.pfc_size,
            n_actions=config.brain.sizes.n_actions,
            cortex_type=config.brain.cortex_type,
            cortex_config=config.brain.cortex,
            dt_ms=config.global_.dt_ms,
            theta_frequency_hz=config.global_.theta_frequency_hz,
            encoding_timesteps=config.brain.encoding_timesteps,
            delay_timesteps=config.brain.delay_timesteps,
            test_timesteps=config.brain.test_timesteps,
            neurons_per_action=config.brain.striatum.neurons_per_action,
            oscillator_couplings=config.brain.oscillator_couplings,
            parallel=config.brain.parallel,
            device=config.global_.device,
        )

        # Current simulation time
        self._current_time: float = 0.0

        # =====================================================================
        # CREATE BRAIN REGIONS
        # =====================================================================

        # 1. CORTEX: Feature extraction
        # Build cortex config by merging user config with computed sizes
        if self.config.cortex_config is not None:
            # Use provided config, but override sizes
            base_cortex_config = self.config.cortex_config
        else:
            # Create default config
            base_cortex_config = LayeredCortexConfig(n_input=0, n_output=0)

        # Merge with sizes from this config (sizes always come from here)
        cortex_config = replace(
            base_cortex_config,
            dt_ms=self.config.dt_ms,
            device=self.config.device,
            n_input=self.config.input_size,
            n_output=self.config.cortex_size,
        )

        # Select implementation based on config
        if self.config.cortex_type == CortexType.PREDICTIVE:
            # PredictiveCortex with local error learning
            # Convert LayeredCortexConfig to PredictiveCortexConfig
            pred_config = PredictiveCortexConfig(
                dt_ms=cortex_config.dt_ms,
                device=cortex_config.device,
                n_input=cortex_config.n_input,
                n_output=cortex_config.n_output,
                l4_ratio=cortex_config.l4_ratio,
                l23_ratio=cortex_config.l23_ratio,
                l5_ratio=cortex_config.l5_ratio,
                l4_sparsity=cortex_config.l4_sparsity,
                l23_sparsity=cortex_config.l23_sparsity,
                l5_sparsity=cortex_config.l5_sparsity,
                l23_recurrent_strength=cortex_config.l23_recurrent_strength,
                l23_recurrent_decay=cortex_config.l23_recurrent_decay,
                input_to_l4_strength=cortex_config.input_to_l4_strength,
                l4_to_l23_strength=cortex_config.l4_to_l23_strength,
                l23_to_l5_strength=cortex_config.l23_to_l5_strength,
                l23_top_down_strength=cortex_config.l23_top_down_strength,
                stdp_lr=cortex_config.stdp_lr,
                stdp_tau_plus=cortex_config.stdp_tau_plus,
                stdp_tau_minus=cortex_config.stdp_tau_minus,
                ffi_threshold=cortex_config.ffi_threshold,
                ffi_strength=cortex_config.ffi_strength,
                ffi_tau=cortex_config.ffi_tau,
                bcm_enabled=cortex_config.bcm_enabled,
                bcm_tau_theta=cortex_config.bcm_tau_theta,
                prediction_enabled=True,
            )
            _cortex_impl = PredictiveCortex(pred_config)
        else:
            # Default LayeredCortex (L4→L2/3→L5)
            _cortex_impl = LayeredCortex(cortex_config)
        _cortex_impl.reset_state()

        # Get cortex layer sizes
        self._cortex_l23_size = _cortex_impl.l23_size
        self._cortex_l5_size = _cortex_impl.l5_size
        cortex_to_hippo_size = self._cortex_l23_size

        # 2. HIPPOCAMPUS: Episodic memory
        _hippocampus_impl = Hippocampus(HippocampusConfig(
            dt_ms=self.config.dt_ms,
            device=self.config.device,
            n_input=cortex_to_hippo_size,
            n_output=self.config.hippocampus_size,
            ec_l3_input_size=self.config.input_size,
        ))

        # 3. PFC: Working memory
        pfc_input_size = cortex_to_hippo_size + self.config.hippocampus_size
        _pfc_impl = Prefrontal(PrefrontalConfig(
            dt_ms=self.config.dt_ms,
            device=self.config.device,
            n_input=pfc_input_size,
            n_output=self.config.pfc_size,
        ))
        _pfc_impl.reset_state()

        # 4. STRIATUM: Action selection
        # Receives: cortex L5 + hippocampus + PFC
        # NOTE: Pass n_output=n_actions (not n_actions*neurons_per_action)
        # The Striatum internally handles population coding expansion
        striatum_input = self._cortex_l5_size + self.config.hippocampus_size + self.config.pfc_size

        # Sync striatum pfc_size with actual PFC output size to prevent dimension mismatch
        # This ensures goal-conditioned learning works correctly
        striatum_config = config.brain.striatum
        if striatum_config.use_goal_conditioning:
            striatum_config = replace(striatum_config, pfc_size=self.config.pfc_size)

        _striatum_impl = Striatum(StriatumConfig(
            dt_ms=self.config.dt_ms,
            device=self.config.device,
            n_input=striatum_input,
            n_output=self.config.n_actions,  # Number of actions, NOT total neurons
            neurons_per_action=self.config.neurons_per_action,
            pfc_size=striatum_config.pfc_size,  # Now synced with PFC
            use_goal_conditioning=striatum_config.use_goal_conditioning,
        ))
        _striatum_impl.reset_state()

        # 5. CEREBELLUM: Motor refinement
        # Receives: striatum output (action signals)
        # After population coding, striatum outputs n_actions * neurons_per_action
        cerebellum_input = self.config.n_actions * self.config.neurons_per_action
        _cerebellum_impl = Cerebellum(CerebellumConfig(
            dt_ms=self.config.dt_ms,
            device=self.config.device,
            n_input=cerebellum_input,
            n_output=self.config.n_actions,
        ))

        # =====================================================================
        # CREATE EVENT-DRIVEN ADAPTERS
        # =====================================================================
        # These ARE the brain regions - access underlying implementations via .impl

        self.cortex = EventDrivenCortex(
            EventRegionConfig(
                name="cortex",
                output_targets=["hippocampus", "pfc", "striatum"],
            ),
            _cortex_impl,
            pfc_size=self.config.pfc_size,  # For top-down projection
        )

        self.hippocampus = EventDrivenHippocampus(
            EventRegionConfig(
                name="hippocampus",
                output_targets=["pfc", "striatum"],
            ),
            _hippocampus_impl,
        )

        self.pfc = EventDrivenPFC(
            EventRegionConfig(
                name="pfc",
                output_targets=["striatum", "cortex"],  # Top-down to cortex
            ),
            _pfc_impl,
            cortex_input_size=self._cortex_l23_size,
            hippocampus_input_size=self.config.hippocampus_size,
        )

        self.striatum = EventDrivenStriatum(
            EventRegionConfig(
                name="striatum",
                output_targets=["cerebellum"],  # Striatum -> Cerebellum
            ),
            _striatum_impl,
            cortex_input_size=self._cortex_l5_size,
            hippocampus_input_size=self.config.hippocampus_size,
            pfc_input_size=self.config.pfc_size,
            # PFC goal context now extracted from concatenated input via pathway
        )

        self.cerebellum = EventDrivenCerebellum(
            EventRegionConfig(
                name="cerebellum",
                output_targets=[],  # Final motor output
            ),
            _cerebellum_impl,
        )

        # Region lookup (now just references the adapters directly)
        self.adapters = {
            "cortex": self.cortex,
            "hippocampus": self.hippocampus,
            "pfc": self.pfc,
            "striatum": self.striatum,
            "cerebellum": self.cerebellum,
        }

        # =====================================================================
        # TEMPORAL INTEGRATION LAYER
        # =====================================================================

        self.cortex_to_hippo_integrator = TemporalIntegrationLayer(
            n_neurons=cortex_to_hippo_size,
            tau=50.0,
            threshold=0.5,
            gain=2.0,
            device=torch.device(self.config.device),
        )

        # =====================================================================
        # OSCILLATOR MANAGER (centralized, like dopamine)
        # =====================================================================
        # Manages all brain-wide oscillations (delta, theta, alpha, beta, gamma)
        # and broadcasts phases to regions. This ensures:
        # 1. Biological accuracy (EEG shows brain-wide synchronization)
        # 2. Efficiency (single oscillator per frequency)
        # 3. Consistent with dopamine architecture (centralized broadcast)
        # 4. Easy phase-amplitude coupling across regions
        from thalia.core.oscillator import OscillatorManager
        self.oscillators = OscillatorManager(
            dt_ms=self.config.dt_ms,
            device=self.config.device,
            theta_freq=self.config.theta_frequency_hz,
            couplings=self.config.oscillator_couplings,
        )

        # =====================================================================
        # PATHWAY MANAGER
        # =====================================================================
        # Manages all inter-region connections with:
        # - Independent STDP learning during forward passes
        # - Growth support (expand when connected regions grow)
        # - Checkpoint compatibility
        # - Health monitoring and diagnostics
        #
        # This enables curriculum learning and coordinated adaptation.

        self.pathway_manager = PathwayManager(
            cortex_l23_size=self._cortex_l23_size,
            cortex_l5_size=self._cortex_l5_size,
            input_size=self.config.input_size,
            cortex_size=self.config.cortex_size,
            hippocampus_size=self.config.hippocampus_size,
            pfc_size=self.config.pfc_size,
            n_actions=self.config.n_actions,
            neurons_per_action=self.config.neurons_per_action,
            dt_ms=self.config.dt_ms,
            device=self.config.device,
        )

        # Create shortcuts for backward compatibility
        self.cortex_to_hippo_pathway = self.pathway_manager.cortex_to_hippo
        self.cortex_to_striatum_pathway = self.pathway_manager.cortex_to_striatum
        self.cortex_to_pfc_pathway = self.pathway_manager.cortex_to_pfc
        self.hippo_to_pfc_pathway = self.pathway_manager.hippo_to_pfc
        self.hippo_to_striatum_pathway = self.pathway_manager.hippo_to_striatum
        self.pfc_to_striatum_pathway = self.pathway_manager.pfc_to_striatum
        self.striatum_to_cerebellum_pathway = self.pathway_manager.striatum_to_cerebellum
        self.attention_pathway = self.pathway_manager.attention
        self.replay_pathway = self.pathway_manager.replay

        # Pathway registry for iteration (growth, checkpointing, diagnostics)
        self.pathways = self.pathway_manager.get_all_pathways()

        # Pathway-region connection tracking (now managed by PathwayManager)
        self._region_pathway_connections = self.pathway_manager.region_connections

        # =====================================================================
        # EVENT SCHEDULER (for sequential mode)
        # =====================================================================

        self.scheduler = EventScheduler()

        # =====================================================================
        # PARALLEL EXECUTION (optional)
        # =====================================================================

        self._parallel_executor: Optional[ParallelExecutor] = None
        if self.config.parallel:
            self._init_parallel_executor()

        # State tracking
        self._last_cortex_output: Optional[torch.Tensor] = None
        self._last_hippo_output: Optional[torch.Tensor] = None
        self._last_pfc_output: Optional[torch.Tensor] = None
        self._last_action: Optional[int] = None

        # =====================================================================
        # NEUROMODULATOR MANAGER
        # =====================================================================
        # Coordinates VTA (dopamine), LC (norepinephrine), NB (acetylcholine)
        # Manages tonic + phasic signaling, broadcasts to all regions
        # Computes uncertainty, intrinsic reward, prediction error

        self.neuromodulator_manager = NeuromodulatorManager()

        # Create shortcuts for backward compatibility
        self.vta = self.neuromodulator_manager.vta
        self.locus_coeruleus = self.neuromodulator_manager.locus_coeruleus
        self.nucleus_basalis = self.neuromodulator_manager.nucleus_basalis
        self.neuromodulator_coordination = self.neuromodulator_manager.coordination

        # Monitoring
        self._spike_counts: Dict[str, int] = {name: 0 for name in self.adapters}
        self._events_processed: int = 0

        # Growth history tracking
        self._growth_history: list = []

        # Comparison signal state (for match/mismatch detection)
        self._ca1_accumulated: float = 0.0
        self._last_comparison_decision: Optional[str] = None
        self._last_similarity: float = 0.0
        self._novelty_signal: float = 0.0

        # =====================================================================
        # CRITICALITY MONITOR (Optional robustness diagnostic)
        # =====================================================================
        # Tracks network criticality via branching ratio.
        # Enabled when robustness config has criticality enabled.
        self.criticality_monitor: Optional[CriticalityMonitor] = None
        self._total_spikes_for_criticality: int = 0

        # =====================================================================
        # DIAGNOSTICS
        # =====================================================================

        self.diagnostics = DiagnosticsManager()
        self.diagnostics.configure_component("striatum", enabled=True)
        self.diagnostics.configure_component("hippocampus", enabled=True)
        self.diagnostics.configure_component("cortex", enabled=True)
        self.diagnostics.configure_component("pfc", enabled=True)
        self.diagnostics.configure_component("cerebellum", enabled=True)

        # =====================================================================
        # MODEL-BASED PLANNING (Phase 2)
        # =====================================================================
        # Mental simulation coordinator and Dyna background planning
        # Only initialized if use_model_based_planning=True
        self.mental_simulation: Optional["MentalSimulationCoordinator"] = None
        self.dyna_planner: Optional["DynaPlanner"] = None

        if self.thalia_config.brain.use_model_based_planning:
            from thalia.planning import (
                MentalSimulationCoordinator,
                SimulationConfig,
                DynaPlanner,
                DynaConfig,
            )

            # Create mental simulation coordinator
            self.mental_simulation = MentalSimulationCoordinator(
                pfc=self.pfc.impl,
                hippocampus=self.hippocampus,
                striatum=self.striatum.impl,
                cortex=self.cortex,
                config=SimulationConfig(),
            )

            # Create Dyna planner for background planning
            self.dyna_planner = DynaPlanner(
                coordinator=self.mental_simulation,
                striatum=self.striatum.impl,
                hippocampus=self.hippocampus,
                config=DynaConfig(),
            )

        # =====================================================================
        # TRIAL COORDINATOR
        # =====================================================================
        # Coordinates trial execution flow (forward, select_action, deliver_reward)
        # Follows existing manager pattern to reduce god object complexity
        from .trial_coordinator import TrialCoordinator

        # Create mutable container for shared time state
        # This allows coordinator to update brain's _current_time
        self._time_container = [self._current_time]

        self.trial_coordinator = TrialCoordinator(
            regions=self.adapters,
            pathways=self.pathway_manager,
            neuromodulators=self.neuromodulator_manager,
            oscillators=self.oscillators,
            config=self.config,
            spike_counts=self._spike_counts,
            vta=self.vta,
            brain_time=self._time_container,
            mental_simulation=self.mental_simulation,
            dyna_planner=self.dyna_planner,
        )

        # =====================================================================
        # CONSOLIDATION MANAGER
        # =====================================================================
        # Manages memory consolidation and offline replay
        # Follows existing manager pattern to reduce god object complexity
        from .consolidation_manager import ConsolidationManager

        # Create mutable container for shared last_action state
        self._last_action_container = [None]

        self.consolidation_manager = ConsolidationManager(
            hippocampus=self.hippocampus,
            striatum=self.striatum,
            cortex=self.cortex,
            pfc=self.pfc,
            config=self.config,
            deliver_reward_fn=self.deliver_reward,
        )

        # Set cortex L5 size (needed for state reconstruction)
        self.consolidation_manager.set_cortex_l5_size(self._cortex_l5_size)

        # =====================================================================
        # CHECKPOINT MANAGER
        # =====================================================================
        # Initialize checkpoint manager for convenient save/load operations
        self.checkpoint_manager = CheckpointManager(
            brain=self,
            default_compression='zstd'
        )

    @classmethod
    def from_thalia_config(cls, config: "ThaliaConfig") -> "EventDrivenBrain":
        """Create EventDrivenBrain from unified ThaliaConfig.

        This is the recommended way to create a brain, as it uses the
        unified configuration system that eliminates parameter duplication.

        Args:
            config: ThaliaConfig with all settings

        Returns:
            EventDrivenBrain instance

        Raises:
            ConfigValidationError: If configuration is invalid

        Example:
            from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig

            config = ThaliaConfig(
                global_=GlobalConfig(device="cuda"),
                brain=BrainConfig(sizes=RegionSizes(cortex_size=256)),
            )
            brain = EventDrivenBrain.from_thalia_config(config)
        """
        # Validate configuration before creation
        from thalia.config import validate_thalia_config
        validate_thalia_config(config)

        # Create brain via __init__ (which also validates, but this provides early feedback)
        brain = cls(config)

        # Initialize CriticalityMonitor if robustness config enables it
        if config.robustness.enable_criticality:
            brain.criticality_monitor = CriticalityMonitor(config.robustness.criticality)

        return brain

    @classmethod
    def create_from_config(cls, config_dict: Dict[str, Any]) -> "EventDrivenBrain":
        """Create brain dynamically from configuration dictionary.

        This method uses ComponentRegistry to build a brain from a declarative
        configuration, enabling flexible architecture construction without
        hardcoded region/pathway dependencies.

        Args:
            config_dict: Configuration dictionary with the following structure:
                {
                    "global": {
                        "device": "cuda",  # or "cpu"
                        "dt_ms": 1.0,
                        "theta_frequency_hz": 8.0,
                    },
                    "regions": {
                        "cortex": {
                            "type": "cortex",  # Registry name
                            "n_input": 784,
                            "n_output": 256,
                            "n_layers": 3,
                        },
                        "hippocampus": {
                            "type": "hippocampus",
                            "dg_size": 500,
                            "ca3_size": 300,
                            "ca1_size": 200,
                        },
                        "striatum": {
                            "type": "striatum",
                            "n_neurons": 256,
                            "n_actions": 10,
                        },
                    },
                    "pathways": {
                        "visual_input": {
                            "type": "visual",  # Registry name
                            "output_size": 256,
                        },
                        "cortex_to_hippocampus": {
                            "type": "spiking",
                            "source_size": 256,
                            "target_size": 500,
                        },
                    },
                }

        Returns:
            EventDrivenBrain instance with dynamically constructed regions/pathways

        Raises:
            KeyError: If required config keys are missing
            ValueError: If region/pathway types are not registered

        Example:
            >>> from thalia.core.component_registry import ComponentRegistry
            >>> config = {
            ...     "global": {"device": "cpu", "dt_ms": 1.0},
            ...     "regions": {
            ...         "cortex": {"type": "cortex", "n_input": 784, "n_output": 256},
            ...         "striatum": {"type": "striatum", "n_neurons": 256, "n_actions": 10},
            ...     },
            ... }
            >>> brain = EventDrivenBrain.create_from_config(config)

        Note:
            This is an advanced API for custom architectures. For standard
            brain configurations, prefer EventDrivenBrain.from_thalia_config().
        """
        from thalia.core.component_registry import ComponentRegistry
        from thalia.config import (
            ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes,
            LayeredCortexConfig, PrefrontalConfig,
            StriatumConfig, CerebellumConfig
        )

        # Extract global config
        global_dict = config_dict.get("global", {})
        device = global_dict.get("device", "cpu")
        dt_ms = global_dict.get("dt_ms", 1.0)
        theta_frequency_hz = global_dict.get("theta_frequency_hz", 8.0)

        # Build regions using ComponentRegistry
        regions = {}
        regions_dict = config_dict.get("regions", {})

        for region_name, region_config in regions_dict.items():
            region_type = region_config.pop("type")  # Extract registry name

            # Inject global config into region config
            region_config["device"] = device
            region_config["dt_ms"] = dt_ms

            # Get the appropriate config class from registry metadata
            region_class = ComponentRegistry.get("region", region_type)
            if region_class is None:
                raise ValueError(f"Region type '{region_type}' not registered")

            # Try to find the config class
            if region_type == "cortex" or region_type == "layered_cortex":
                config_obj = LayeredCortexConfig(**region_config)
            elif region_type == "hippocampus" or region_type == "trisynaptic":
                config_obj = HippocampusConfig(**region_config)
            elif region_type == "prefrontal" or region_type == "pfc":
                config_obj = PrefrontalConfig(**region_config)
            elif region_type == "striatum":
                config_obj = StriatumConfig(**region_config)
            elif region_type == "cerebellum":
                config_obj = CerebellumConfig(**region_config)
            else:
                # Generic fallback - create config from dict
                raise ValueError(
                    f"Unknown region type '{region_type}'. "
                    f"Supported types: cortex, hippocampus, prefrontal, striatum, cerebellum"
                )

            # Create region instance via registry
            regions[region_name] = ComponentRegistry.create("region", region_type, config_obj)

        # Build pathways using ComponentRegistry
        pathways = {}
        pathways_dict = config_dict.get("pathways", {})

        for pathway_name, pathway_config in pathways_dict.items():
            pathway_type = pathway_config.pop("type")  # Extract registry name

            # Inject global config into pathway config
            pathway_config["device"] = device
            pathway_config["dt_ms"] = dt_ms

            # Get pathway class from registry
            pathway_class = ComponentRegistry.get("pathway", pathway_type)
            if pathway_class is None:
                raise ValueError(f"Pathway type '{pathway_type}' not registered")

            # Determine config class based on pathway type
            # Most pathways use PathwayConfig or specialized subclasses
            if pathway_type in ["spiking", "spiking_stdp"]:
                # Base spiking pathway
                from thalia.config.base import PathwayConfig
                config_obj = PathwayConfig(**pathway_config)
            elif pathway_type == "attention" or pathway_type == "spiking_attention":
                # Attention pathway with specialized config
                from thalia.integration.pathways.spiking_attention import SpikingAttentionPathwayConfig
                config_obj = SpikingAttentionPathwayConfig(**pathway_config)
            elif pathway_type == "replay" or pathway_type == "spiking_replay":
                # Replay pathway with specialized config
                from thalia.integration.pathways.spiking_replay import SpikingReplayPathwayConfig
                config_obj = SpikingReplayPathwayConfig(**pathway_config)
            elif pathway_type == "visual":
                # Visual sensory pathway
                from thalia.sensory.pathways import VisualConfig
                config_obj = VisualConfig(**pathway_config)
            elif pathway_type == "auditory":
                # Auditory sensory pathway
                from thalia.sensory.pathways import AuditoryConfig
                config_obj = AuditoryConfig(**pathway_config)
            elif pathway_type == "language":
                # Language sensory pathway
                from thalia.sensory.pathways import LanguageConfig
                config_obj = LanguageConfig(**pathway_config)
            else:
                # Generic fallback - try PathwayConfig
                from thalia.config.base import PathwayConfig
                try:
                    config_obj = PathwayConfig(**pathway_config)
                except Exception as e:
                    raise ValueError(
                        f"Unknown pathway type '{pathway_type}' and failed to create "
                        f"with PathwayConfig: {e}"
                    ) from e

            # Create pathway instance via direct instantiation
            # (ComponentRegistry.create would work too, but this is more explicit)
            pathways[pathway_name] = pathway_class(config_obj)

        # For now, create a minimal ThaliaConfig and standard brain
        # Full dynamic construction would require more infrastructure
        # This provides the foundation for future expansion

        # Build ThaliaConfig from extracted values
        global_config = GlobalConfig(
            device=device,
            dt_ms=dt_ms,
            theta_frequency_hz=theta_frequency_hz,
        )

        # Extract sizes from regions if provided
        cortex_size = regions_dict.get("cortex", {}).get("n_output", 256)
        hippocampus_size = regions_dict.get("hippocampus", {}).get("ca1_size", 200)
        pfc_size = regions_dict.get("pfc", {}).get("n_neurons", 128)
        n_actions = regions_dict.get("striatum", {}).get("n_actions", 10)
        input_size = regions_dict.get("cortex", {}).get("n_input", 784)

        region_sizes = RegionSizes(
            input_size=input_size,
            cortex_size=cortex_size,
            hippocampus_size=hippocampus_size,
            pfc_size=pfc_size,
            n_actions=n_actions,
        )

        brain_config = BrainConfig(sizes=region_sizes)

        thalia_config = ThaliaConfig(
            global_=global_config,
            brain=brain_config,
        )

        # Create brain using standard constructor
        # In future, this could be extended to use the dynamically created regions
        brain = cls.from_thalia_config(thalia_config)

        return brain

    def _init_parallel_executor(self) -> None:
        """Initialize parallel executor with region creators.

        Note: Parallel mode is experimental. On Windows, multiprocessing uses
        "spawn" which requires pickleable region creators. This implementation
        uses the existing module-level creator functions from parallel.py
        for now. For full config customization in parallel mode, consider using
        the sequential mode (parallel=False).

        TODO: These creator functions need to be implemented in thalia.events.parallel
        """
        # TODO: Implement _create_real_* functions in thalia.events.parallel
        # from thalia.events.parallel import (
        #     _create_real_cortex, _create_real_hippocampus,
        #     _create_real_pfc, _create_real_striatum,
        # )

        # Create parallel executor with module-level creators
        # These are pickle-able because they're defined at module level
        self._parallel_executor = ParallelExecutor(
            region_creators={
                # "cortex": _create_real_cortex,
                # "hippocampus": _create_real_hippocampus,
                # "pfc": _create_real_pfc,
                # "striatum": _create_real_striatum,
            },
        )
        self._parallel_executor.start()

    def __del__(self):
        """Clean up parallel executor if active."""
        if hasattr(self, '_parallel_executor') and self._parallel_executor is not None:
            try:
                self._parallel_executor.stop()
            except Exception:
                pass  # Ignore errors during cleanup

    # =========================================================================
    # HIGH-LEVEL APIs
    # =========================================================================

    def forward(
        self,
        sensory_input: Optional[torch.Tensor] = None,
        n_timesteps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process sensory input through the brain for n timesteps.

        This is the standard forward pass - the brain automatically handles:
        - Encoding vs retrieval (hippocampus theta phase)
        - Working memory maintenance (PFC dopamine gating)
        - Prediction learning (cortex STDP)
        - Action preparation (striatum accumulation)

        No explicit mode switching needed - natural dynamics handle everything.

        Args:
            sensory_input: Input pattern [input_size], or None for maintenance
            n_timesteps: Number of timesteps to process (default: encoding_timesteps)

        Returns:
            Dict with region activities for monitoring

        Example:
            # Encoding
            brain.forward(sample_pattern, n_timesteps=15)

            # Maintenance (working memory)
            brain.forward(None, n_timesteps=10)

            # Retrieval/test
            brain.forward(test_pattern, n_timesteps=15)

            # Action selection
            action, confidence = brain.select_action()

            # Learning
            brain.deliver_reward(external_reward=1.0)
        """
        # Default timesteps
        n_timesteps = n_timesteps or self.config.encoding_timesteps

        # Delegate to trial coordinator
        result = self.trial_coordinator.forward(
            sensory_input=sensory_input,
            n_timesteps=n_timesteps,
            scheduler=self.scheduler,
            parallel_executor=self._parallel_executor,
            process_events_fn=self._process_events_until,
            update_neuromodulators_fn=self._update_neuromodulators,
            get_cortex_input_fn=self._get_cortex_input,
            criticality_monitor=self.criticality_monitor,
        )

        # Sync time from coordinator
        self._current_time = self._time_container[0]

        return result

    def select_action(self, explore: bool = True, use_planning: bool = True) -> tuple[int, float]:
        """Select action based on current striatum state.

        Uses the striatum's finalize_action method which handles:
        - Accumulated NET votes (D1-D2)
        - UCB exploration bonus
        - Softmax selection

        If use_planning=True and model-based planning is enabled:
        - Uses MentalSimulationCoordinator for tree search
        - Returns best action from simulated rollouts
        - Falls back to striatum if planning disabled

        Args:
            explore: Whether to allow exploration
            use_planning: Whether to use mental simulation

        Returns:
            (action, confidence): Selected action index and confidence [0, 1]
        """
        # Delegate to trial coordinator
        return self.trial_coordinator.select_action(explore=explore, use_planning=use_planning)

    def deliver_reward(self, external_reward: Optional[float] = None) -> None:
        """Deliver external reward signal for learning.

        Brain acts as VTA (ventral tegmental area):
        1. Combines external reward with current intrinsic reward
        2. Queries striatum for expected value of the action taken
        3. Computes reward prediction error (RPE = reward - expected)
        4. Normalizes RPE using adaptive scaling
        5. Broadcasts normalized dopamine to ALL regions

        Biologically accurate reward combination:
        - If external_reward is None: Use pure intrinsic reward
        - If external_reward is provided (including 0.0): Add to intrinsic reward
        - Real brains sum external + intrinsic, they don't replace each other

        Note on automatic learning:
        - Intrinsic rewards (from prediction errors) trigger learning AUTOMATICALLY
          during forward() when signals are strong (|reward| > 0.3 threshold)
        - This method is for EXTERNAL task feedback (correct/incorrect)
        - Both learning signals use the same striatal plasticity mechanisms
        - This enables both curiosity-driven AND task-driven learning

        Args:
            external_reward: Task-based reward value (-1 to +1), or None for pure intrinsic
        """
        # Delegate to trial coordinator
        self.trial_coordinator.deliver_reward(
            external_reward=external_reward,
            compute_intrinsic_reward_fn=self._compute_intrinsic_reward,
        )

        # Store experience automatically (for replay) via consolidation manager
        if self.trial_coordinator.get_last_action() is not None:
            # Compute total reward for experience storage
            intrinsic_reward = self._compute_intrinsic_reward()
            if external_reward is None:
                total_reward = intrinsic_reward
            else:
                total_reward = external_reward + intrinsic_reward
                total_reward = max(-2.0, min(2.0, total_reward))

            # Sync last_action to container for consolidation manager
            self._last_action_container[0] = self.trial_coordinator.get_last_action()

            self.consolidation_manager.store_experience(
                action=self.trial_coordinator.get_last_action(),
                reward=total_reward,
                last_action_holder=self._last_action_container,
            )            # Trigger background planning (Phase 2)
            if self.dyna_planner is not None:
                current_state = self.trial_coordinator.get_last_pfc_output()
                next_state = self.pfc.impl.state.spikes if self.pfc.impl.state else None

                if current_state is not None and next_state is not None:
                    goal_context = current_state

                    self.dyna_planner.process_real_experience(
                        state=current_state,
                        action=self.trial_coordinator.get_last_action(),
                        reward=total_reward,
                        next_state=next_state,
                        done=False,
                        goal_context=goal_context
                    )

    def _compute_uncertainty(self) -> float:
        """Compute current task uncertainty for arousal modulation.

        Uncertainty drives norepinephrine release from locus coeruleus.
        High uncertainty → high arousal → increased neural gain.

        Sources:
        1. Prediction error magnitude (cortex)
        2. Value estimate variance (striatum)
        3. Novelty detection

        Returns:
            Uncertainty estimate in [0, 1]
        """
        uncertainty = 0.0
        n_sources = 0

        # Cortex prediction error as uncertainty proxy
        if hasattr(self.cortex.impl, 'state') and hasattr(self.cortex.impl.state, 'free_energy'):
            free_energy = self.cortex.impl.state.free_energy
            # High FE → high uncertainty
            cortex_uncertainty = min(1.0, free_energy / 10.0)
            uncertainty += cortex_uncertainty
            n_sources += 1

        # Average across sources
        if n_sources > 0:
            uncertainty = uncertainty / n_sources
        else:
            # No signals → assume moderate uncertainty
            uncertainty = 0.3

        return max(0.0, min(1.0, uncertainty))

    def _compute_intrinsic_reward(self) -> float:
        """Compute intrinsic reward from the brain's internal objectives.

        This implements the free energy principle: the brain rewards itself
        for minimizing prediction error (surprise). Intrinsic reward is
        ALWAYS computed - it's the brain's continuous self-evaluation.

        Sources:
        1. **Cortex predictive coding**: Low free energy → good predictions → reward
        2. **Hippocampus pattern completion**: High similarity → successful recall → reward

        This is biologically plausible:
        - VTA dopamine neurons respond to internal prediction errors
        - Curiosity and "eureka" moments are intrinsically rewarding
        - The brain learns even without external feedback

        Returns:
            Intrinsic reward in range [-1, 1]
        """
        reward = 0.0
        n_sources = 0

        # =====================================================================
        # 1. CORTEX PREDICTIVE CODING (free energy minimization)
        # =====================================================================
        # Low prediction error = good model of the world = reward
        if hasattr(self.cortex.impl, 'state') and hasattr(self.cortex.impl.state, 'free_energy'):
            free_energy = self.cortex.impl.state.free_energy

            # Free energy is typically 0-10, lower is better
            # Map: 0 → +1 (perfect prediction), 5 → 0, 10+ → -1 (bad prediction)
            cortex_reward = 1.0 - 0.2 * min(free_energy, 10.0)
            cortex_reward = max(-1.0, min(1.0, cortex_reward))
            reward += cortex_reward
            n_sources += 1

        # Fallback: check for accumulated free energy in PredictiveCortex
        elif hasattr(self.cortex.impl, '_total_free_energy'):
            total_fe = self.cortex.impl._total_free_energy
            cortex_reward = 1.0 - 0.1 * min(total_fe, 20.0)
            cortex_reward = max(-1.0, min(1.0, cortex_reward))
            reward += cortex_reward
            n_sources += 1

        # =====================================================================
        # 2. HIPPOCAMPUS PATTERN COMPLETION (memory recall quality)
        # =====================================================================
        # High pattern similarity = successful memory retrieval = reward
        # Biology: VTA observes CA1 output activity. Strong coherent firing = successful recall.
        # We infer similarity from CA1 spike rate (observable signal).
        if (hasattr(self.hippocampus.impl, 'state') and
            self.hippocampus.impl.state.ca1_spikes is not None):

            # CA1 firing rate as proxy for retrieval quality
            # High rate = strong recall, low rate = weak/no recall
            ca1_activity = compute_firing_rate(self.hippocampus.impl.state.ca1_spikes)

            # Map CA1 activity [0, 1] to reward [-1, 1]
            # 0.5 activity = neutral (0 reward), >0.5 = positive, <0.5 = negative
            hippo_reward = 2.0 * ca1_activity - 1.0
            # Weight slightly less than cortex (memory is secondary to prediction)
            reward += 0.5 * hippo_reward
            n_sources += 1

        # =====================================================================
        # Average across sources
        # =====================================================================
        if n_sources > 0:
            reward = reward / n_sources
        else:
            # No intrinsic signals available - neutral
            reward = 0.0

        return max(-1.0, min(1.0, reward))

    def _compute_prediction_error(self) -> float:
        """Compute current prediction error for ACh modulation.

        Prediction error drives ACh release from nucleus basalis.
        High PE → novelty → ACh burst → encoding mode.

        Sources:
        1. Cortex free energy (prediction error magnitude)
        2. Hippocampus retrieval mismatch

        Returns:
            Prediction error estimate in [0, 1]
        """
        prediction_error = 0.0
        n_sources = 0

        # Cortex predictive coding error
        if hasattr(self.cortex.impl, 'state') and hasattr(self.cortex.impl.state, 'free_energy'):
            free_energy = self.cortex.impl.state.free_energy
            # Map FE to [0, 1]: 0 → 0, 5 → 0.5, 10+ → 1.0
            cortex_pe = min(1.0, free_energy / 10.0)
            prediction_error += cortex_pe
            n_sources += 1

        # Average across sources
        if n_sources > 0:
            prediction_error = prediction_error / n_sources
        else:
            # No signals → assume low PE (familiar context)
            prediction_error = 0.2

        return max(0.0, min(1.0, prediction_error))

    def _update_neuromodulators(self) -> None:
        """Update all centralized neuromodulator systems every timestep.

        Updates:
        1. VTA dopamine (tonic from intrinsic reward, phasic decays)
        2. Locus coeruleus NE (arousal from uncertainty)
        3. Nucleus basalis ACh (encoding from prediction error)
        4. Broadcasts all signals to regions and pathways via manager

        Called every timestep to maintain neuromodulator dynamics.
        """
        # =====================================================================
        # 1. UPDATE VTA (DOPAMINE)
        # =====================================================================
        intrinsic_reward = self._compute_intrinsic_reward()
        self.vta.update(dt_ms=self.config.dt_ms, intrinsic_reward=intrinsic_reward)

        # =====================================================================
        # 2. UPDATE LOCUS COERULEUS (NOREPINEPHRINE)
        # =====================================================================
        uncertainty = self._compute_uncertainty()
        self.locus_coeruleus.update(dt_ms=self.config.dt_ms, uncertainty=uncertainty)

        # =====================================================================
        # 3. UPDATE NUCLEUS BASALIS (ACETYLCHOLINE)
        # =====================================================================
        prediction_error = self._compute_prediction_error()
        self.nucleus_basalis.update(dt_ms=self.config.dt_ms, prediction_error=prediction_error)

        # =====================================================================
        # 4. APPLY DA-NE COORDINATION (requires prediction error context)
        # =====================================================================
        # This coordination happens here because it needs the PE context
        # Other coordinations (NE-ACh, DA-ACh) are handled in manager
        dopamine = self.vta.get_global_dopamine()
        norepinephrine = self.locus_coeruleus.get_norepinephrine()

        dopamine, norepinephrine = self.neuromodulator_manager.coordination.coordinate_da_ne(
            dopamine, norepinephrine, prediction_error
        )

        # Update the systems with coordinated values
        # (These are small adjustments to the raw signals)
        # Note: We don't call set_dopamine/set_norepinephrine because those
        # would override internal state. The coordination is applied during broadcast.

        # =====================================================================
        # 5. BROADCAST TO ALL REGIONS (via manager with coordination)
        # =====================================================================
        regions = {
            'cortex': self.cortex.impl,
            'hippocampus': self.hippocampus.impl,
            'pfc': self.pfc.impl,
            'striatum': self.striatum.impl,
            'cerebellum': self.cerebellum.impl,
        }

        # Manager handles NE-ACh and DA-ACh coordination, then broadcasts
        self.neuromodulator_manager.broadcast_to_regions(regions)

        # =====================================================================
        # 6. BROADCAST TO ALL PATHWAYS (biologically accurate!)
        # =====================================================================
        # In real brains, neuromodulators affect ALL synapses, not just those
        # within regions. Pathways (inter-region connections) also receive
        # dopamine, norepinephrine, and acetylcholine modulation.
        #
        # This affects:
        # - Pathway learning rates (DA modulates STDP)
        # - Signal gain (NE increases responsiveness)
        # - LTP/LTD balance (ACh favors encoding)
        dopamine = self.vta.get_global_dopamine()
        norepinephrine = self.locus_coeruleus.get_norepinephrine()
        acetylcholine = self.nucleus_basalis.get_acetylcholine()

        for pathway in self.pathways.values():
            if hasattr(pathway, 'set_neuromodulators'):
                pathway.set_neuromodulators(dopamine, norepinephrine, acetylcholine)

        # =====================================================================
        # 7. INTRINSIC REWARD LEARNING (CONTINUOUS CURIOSITY-DRIVEN LEARNING)
        # =====================================================================
        # If intrinsic reward is significant, trigger striatum learning automatically.
        # This enables pure curiosity-driven learning without external rewards.
        #
        # Biologically: Dopamine responses to prediction errors (intrinsic) DO cause
        # striatal plasticity, not just external rewards. This is how exploration and
        # curiosity drive learning.
        #
        # Threshold: Only trigger learning for strong intrinsic signals (|reward| > 0.3)
        # to avoid learning from noise. This mirrors dopamine neuron firing thresholds.
        last_action = self.trial_coordinator.get_last_action()
        if abs(intrinsic_reward) > INTRINSIC_LEARNING_THRESHOLD and last_action is not None:
            # Trigger striatum learning from intrinsic reward
            # Note: This uses the SAME learning mechanism as external rewards,
            # just with intrinsic signal as the teaching signal
            self.striatum.impl.deliver_reward(intrinsic_reward)
            self.striatum.impl.update_value_estimate(last_action, intrinsic_reward)

        # =====================================================================
        # 8. PHASE 3: UPDATE COGNITIVE LOAD (HYPERBOLIC DISCOUNTING)
        # =====================================================================
        # Automatically update cognitive load based on PFC working memory usage
        # This modulates temporal discounting (high load → more impulsive)
        if hasattr(self.pfc.impl, 'discounter') and self.pfc.impl.discounter is not None:
            cognitive_load = self._compute_cognitive_load()
            self.pfc.impl.update_cognitive_load(cognitive_load)

        # =====================================================================
        # 9. BROADCAST OSCILLATOR PHASES
        # =====================================================================
        self._broadcast_oscillator_phases()

    def _compute_cognitive_load(self) -> float:
        """Compute current cognitive load from PFC working memory usage.

        Phase 3 functionality: Cognitive load drives temporal discounting.
        High working memory usage → high load → more impulsive choices.

        Returns:
            Cognitive load (0-1), where 1 = maximum capacity
        """
        # Measure PFC activity from spike output, not internal WM state
        if self.pfc.impl.state.spikes is None:
            return 0.0

        # Measure working memory load from sustained spike activity
        wm_activity = compute_firing_rate(self.pfc.impl.state.spikes)

        # Also consider number of active goals (if hierarchical goals enabled)
        goal_load = 0.0
        if (hasattr(self.pfc.impl, 'goal_manager') and
            self.pfc.impl.goal_manager is not None):
            n_active = len(self.pfc.impl.goal_manager.active_goals)
            max_goals = self.pfc.impl.goal_manager.config.max_active_goals
            goal_load = n_active / max(max_goals, 1)

        # Combine WM activity and goal count (weighted average)
        cognitive_load = 0.7 * wm_activity + 0.3 * goal_load

        # Clamp to [0, 1]
        return max(0.0, min(1.0, cognitive_load))

    def _broadcast_oscillator_phases(self) -> None:
        """Broadcast oscillator phases and effective amplitudes to all regions.

        Effective amplitudes implement automatic multiplicative coupling:
        Each oscillator's amplitude reflects the combined effect of ALL
        phase-amplitude couplings. For example, if gamma is modulated by
        theta (×0.8) and beta (×0.6), the effective gamma amplitude is 0.48.

        This enables emergent higher-order coupling without explicit programming.

        Regions use oscillator information for:
        - Phase-dependent gating (theta encoding vs retrieval)
        - Attention modulation (alpha suppression)
        - Motor preparation (beta synchrony)
        - Feature binding (gamma synchrony)
        - Amplitude-dependent learning (effective_amplitudes)

        Called every timestep, similar to dopamine broadcast.
        """
        phases = self.oscillators.get_phases()
        signals = self.oscillators.get_signals()

        # Compute effective amplitudes (automatic multiplicative coupling)
        effective_amplitudes = self.oscillators.get_effective_amplitudes()

        # Get theta slot for sequence encoding (working memory)
        theta_slot = self.oscillators.get_theta_slot(n_slots=7)

        # Pass to regions that implement set_oscillator_phases
        # (optional - regions can ignore if not needed)
        if hasattr(self.cortex.impl, 'set_oscillator_phases'):
            self.cortex.impl.set_oscillator_phases(
                phases, signals, theta_slot, effective_amplitudes
            )

        if hasattr(self.hippocampus.impl, 'set_oscillator_phases'):
            self.hippocampus.impl.set_oscillator_phases(
                phases, signals, theta_slot, effective_amplitudes
            )

        if hasattr(self.pfc.impl, 'set_oscillator_phases'):
            self.pfc.impl.set_oscillator_phases(
                phases, signals, theta_slot, effective_amplitudes
            )

        if hasattr(self.striatum.impl, 'set_oscillator_phases'):
            self.striatum.impl.set_oscillator_phases(
                phases, signals, theta_slot, effective_amplitudes
            )

        if hasattr(self.cerebellum.impl, 'set_oscillator_phases'):
            self.cerebellum.impl.set_oscillator_phases(
                phases, signals, theta_slot, effective_amplitudes
            )

        # Broadcast to pathways (component parity)
        # Pathways can use oscillator info for:
        # - Phase-dependent transmission efficiency
        # - Attention modulation (beta/alpha coupling)
        # - State-dependent plasticity (sleep/wake)
        # - Temporal coordination (align inter-region communication)
        for _pathway_name, pathway in self.pathways.items():
            if hasattr(pathway, 'set_oscillator_phases'):
                pathway.set_oscillator_phases(
                    phases, signals, theta_slot, effective_amplitudes
                )

    def consolidate(self, n_cycles: int = 5, batch_size: int = 32, verbose: bool = False) -> Dict[str, Any]:
        """Perform memory consolidation (replay) automatically.

        This simulates sleep/offline replay where hippocampus replays stored
        episodes to strengthen cortical representations. Each replayed experience
        triggers actual learning via dopamine delivery.

        Biologically accurate consolidation:
        1. Sample experiences from hippocampal memory
        2. Replay state through brain (reactivate patterns)
        3. Deliver stored reward → dopamine → striatum learning
        4. HER automatically augments if enabled

        This is why consolidation works: replayed experiences trigger the SAME
        learning signals as real experiences, strengthening action values offline.

        Args:
            n_cycles: Number of replay cycles to run
            batch_size: Number of experiences per cycle
            verbose: Whether to print progress

        Returns:
            Dict with consolidation statistics
        """
        # Sync coordinator's last_action to container before consolidation
        self._last_action_container[0] = self.trial_coordinator.get_last_action()

        # Delegate to consolidation manager
        return self.consolidation_manager.consolidate(
            n_cycles=n_cycles,
            batch_size=batch_size,
            verbose=verbose,
            last_action_holder=self._last_action_container,
        )

    def reset_state(self) -> None:
        """Reset brain state for new episode.

        This is a HARD reset - use for completely new, unrelated episodes.
        For starting a new sequence within the same session, use new_sequence().
        """
        self._current_time = 0.0
        self._time_container[0] = 0.0  # Sync coordinator's time
        self.scheduler = EventScheduler()

        # Reset regions (full state reset)
        self.cortex.impl.reset_state()
        self.pfc.impl.reset_state()
        self.striatum.impl.reset_state()
        self.hippocampus.impl.new_trial()

        # Reset monitoring
        self._spike_counts = {name: 0 for name in self.adapters}
        self._events_processed = 0
        self._last_action = None

    def new_sequence(self) -> None:
        """Prepare for a new sequence while preserving learned representations.

        Unlike reset(), this only clears what's needed for a new sequence:
        - Hippocampus stored patterns (for new comparisons)
        - Does NOT reset neural states (let natural decay handle transitions)
        - Does NOT reset weights (preserves learning)

        Call this between unrelated text sequences during training, but NOT
        between tokens within a sequence.
        """
        self.hippocampus.impl.new_trial()

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete brain state for checkpointing.

        Returns state dictionary with keys:
        - regions: State from each brain region (cortex, hippocampus, pfc, striatum, cerebellum)
        - pathways: State from pathway manager (all 9 pathways)
        - neuromodulators: State from neuromodulator manager (VTA, LC, NB)
        - theta: Theta oscillator state
        - scheduler: Event scheduler state (current time, pending events)
        - trial_state: Current trial phase and counters
        - config: Configuration for validation

        This captures the COMPLETE state needed to resume from checkpoint.
        """
        state_dict = {
            "regions": {
                "cortex": self.cortex.impl.get_full_state(),
                "hippocampus": self.hippocampus.impl.get_full_state(),
                "pfc": self.pfc.impl.get_full_state(),
                "striatum": self.striatum.impl.get_full_state(),
                "cerebellum": self.cerebellum.impl.get_full_state(),
            },
            "pathways": self.pathway_manager.get_state(),
            "neuromodulators": self.neuromodulator_manager.get_state(),
            "oscillators": {
                # Store oscillator states - manager has delta, theta, alpha, beta, gamma
                # Each has get_state() method for checkpointing
                "delta": self.oscillators.delta.get_state() if hasattr(self.oscillators, 'delta') else None,
                "theta": self.oscillators.theta.get_state() if hasattr(self.oscillators, 'theta') else None,
                "alpha": self.oscillators.alpha.get_state() if hasattr(self.oscillators, 'alpha') else None,
                "beta": self.oscillators.beta.get_state() if hasattr(self.oscillators, 'beta') else None,
                "gamma": self.oscillators.gamma.get_state() if hasattr(self.oscillators, 'gamma') else None,
            },
            "scheduler": {
                "current_time": self._current_time,
                "events_processed": self._events_processed,
            },
            "trial_state": {
                "spike_counts": self._spike_counts.copy(),
                "last_action": self.trial_coordinator.get_last_action(),
            },
            "config": {
                "input_size": self.config.input_size,
                "cortex_size": self.config.cortex_size,
                "hippocampus_size": self.config.hippocampus_size,
                "pfc_size": self.config.pfc_size,
                "n_actions": self.config.n_actions,
                "cortex_type": self.config.cortex_type.name,
            },
        }

        return state_dict

    def load_full_state(self, state_dict: Dict[str, Any]) -> None:
        """Load complete brain state from checkpoint.

        Args:
            state_dict: State dictionary from get_full_state()

        Raises:
            ValueError: If config dimensions don't match

        Note:
            This restores the complete brain state including:
            - All region weights and states
            - Pathway manager state (all 9 pathways)
            - Neuromodulator manager state (VTA, LC, NB)
            - Theta oscillator phase
            - Event scheduler time
            - Trial phase and counters
        """
        # Validate config compatibility
        config = state_dict.get("config", {})
        if config.get("input_size") != self.config.input_size:
            raise ValueError(f"Config mismatch: input_size {config.get('input_size')} != {self.config.input_size}")
        if config.get("cortex_size") != self.config.cortex_size:
            raise ValueError(f"Config mismatch: cortex_size {config.get('cortex_size')} != {self.config.cortex_size}")
        if config.get("hippocampus_size") != self.config.hippocampus_size:
            raise ValueError(f"Config mismatch: hippocampus_size {config.get('hippocampus_size')} != {self.config.hippocampus_size}")
        if config.get("pfc_size") != self.config.pfc_size:
            raise ValueError(f"Config mismatch: pfc_size {config.get('pfc_size')} != {self.config.pfc_size}")
        if config.get("n_actions") != self.config.n_actions:
            raise ValueError(f"Config mismatch: n_actions {config.get('n_actions')} != {self.config.n_actions}")

        # Restore region states
        regions = state_dict["regions"]
        self.cortex.impl.load_full_state(regions["cortex"])
        self.hippocampus.impl.load_full_state(regions["hippocampus"])
        self.pfc.impl.load_full_state(regions["pfc"])
        self.striatum.impl.load_full_state(regions["striatum"])
        self.cerebellum.impl.load_full_state(regions["cerebellum"])

        # Restore manager states
        if "pathways" in state_dict:
            self.pathway_manager.load_state(state_dict["pathways"])
        if "neuromodulators" in state_dict:
            self.neuromodulator_manager.load_state(state_dict["neuromodulators"])

        # Restore oscillator phases
        if "oscillators" in state_dict:
            osc_state = state_dict["oscillators"]
            phases = osc_state.get("phases", {})
            # Set phases directly on oscillators
            for freq_name, phase in phases.items():
                if hasattr(self.oscillators, f"{freq_name}_phase"):
                    setattr(self.oscillators, f"{freq_name}_phase", phase)

        # Restore scheduler state
        scheduler_state = state_dict["scheduler"]
        self._current_time = scheduler_state["current_time"]
        self._events_processed = scheduler_state["events_processed"]

        # Restore trial state
        trial_state = state_dict["trial_state"]
        self._spike_counts = trial_state["spike_counts"].copy()
        self._last_action = trial_state["last_action"]

        # Note: Event queue is NOT restored - assumes checkpoint at clean state
        # For mid-trial checkpointing, would need to serialize pending events

    # =========================================================================
    # COMPARISON SIGNAL (MATCH/MISMATCH DETECTION)
    # =========================================================================

    def _compute_comparison_signal(
        self,
        hippo_activity: torch.Tensor,
        n_timesteps: int,
        current_timestep: int,
    ) -> torch.Tensor:
        """Compute match/mismatch comparison signal using temporal burst coding.

        Accumulates CA1 activity over the test phase, then generates synchronized
        bursts in the decision window to signal match vs mismatch.

        Args:
            hippo_activity: Current hippocampal output [batch, hippo_size]
            n_timesteps: Total timesteps in phase
            current_timestep: Current timestep index

        Returns:
            Comparison signal [batch, comparison_size] for striatum
        """
        comparison_size = getattr(self.config, 'comparison_size', 4)
        batch_size = hippo_activity.shape[0] if hippo_activity.dim() > 1 else 1

        # Accumulate CA1 activity
        if not hasattr(self, '_ca1_accumulated'):
            self._ca1_accumulated = 0.0

        ca1_sum = hippo_activity.sum().item()
        self._ca1_accumulated += ca1_sum

        # Decision window: last 20% of phase
        decision_start = int(n_timesteps * 0.8)

        if current_timestep >= decision_start:
            # Compute similarity from accumulated activity
            # High activity = match (CA3 retrieval succeeded)
            # Low activity = mismatch (pattern not in CA3)

            avg_activity = self._ca1_accumulated / (current_timestep + 1)

            # Threshold for match/mismatch decision
            match_threshold = 0.5  # Tunable parameter

            if avg_activity > match_threshold:
                # MATCH: Strong synchronized burst to match neurons
                signal = torch.zeros(batch_size, comparison_size)
                signal[:, :comparison_size // 2] = 1.0  # Match neurons fire
                self._last_comparison_decision = 'MATCH'
            else:
                # MISMATCH: Strong synchronized burst to mismatch neurons
                signal = torch.zeros(batch_size, comparison_size)
                signal[:, comparison_size // 2:] = 1.0  # Mismatch neurons fire
                self._last_comparison_decision = 'MISMATCH'

            self._last_similarity = avg_activity
            return signal.squeeze() if batch_size == 1 else signal
        else:
            # Not in decision window - return zeros
            return torch.zeros(comparison_size)

    # =========================================================================
    # COUNTERFACTUAL LEARNING
    # =========================================================================

    def deliver_reward_with_counterfactual(
        self,
        reward: float,
        is_match: bool,
        selected_action: int,
        counterfactual_scale: float = 0.5,
    ) -> Dict[str, Any]:
        """Deliver reward with counterfactual learning for non-selected action.

        Implements model-based RL: after experiencing a real outcome, we also
        simulate "what would have happened if I had chosen differently?"

        This solves asymmetric learning where only the selected action updates.
        Now BOTH actions learn on every trial:
        - Selected action: learns from actual outcome
        - Non-selected action: learns from counterfactual (imagined) outcome

        Args:
            reward: Actual reward received
            is_match: Whether this was a match trial
            selected_action: Action that was actually taken (0=MATCH, 1=NOMATCH)
            counterfactual_scale: How much to scale counterfactual learning (0-1)

        Returns:
            Dict with both real and counterfactual learning metrics
        """
        # Delegate to trial coordinator for counterfactual learning
        other_action = 1 - selected_action
        self.trial_coordinator.deliver_reward_with_counterfactual(
            external_reward=reward,
            counterfactual_action=other_action,
            compute_intrinsic_reward_fn=self._compute_intrinsic_reward,
        )

        # Get novelty boost and apply to specialized pathways
        novelty_boost = self._get_novelty_boost()
        modulated_reward = reward * novelty_boost

        # Update specialized pathways (most pathways already learned during forward)
        attention_result = {}
        if hasattr(self.attention_pathway, 'learn'):
            attention_result = self.attention_pathway.learn(
                source_activity=torch.zeros(self.config.pfc_size),
                target_activity=torch.zeros(self.config.cortex_size),
                dopamine=modulated_reward,
            )

        # Determine counterfactual reward for metrics
        correct_action = 0 if is_match else 1
        counterfactual_reward = 1.0 if (other_action == correct_action) else -1.0

        return {
            "real": {},  # Metrics now handled by coordinator
            "counterfactual": {},
            "selected_action": selected_action,
            "other_action": other_action,
            "counterfactual_reward": counterfactual_reward,
            "attention_pathway": attention_result,
            "novelty_boost": novelty_boost,
        }

    def _get_novelty_boost(self) -> float:
        """Get novelty-based learning rate multiplier."""
        # Simple implementation - can be enhanced with actual novelty detection
        if not hasattr(self, '_novelty_signal'):
            self._novelty_signal = 1.0
        return max(1.0, self._novelty_signal)

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _get_cortex_input(self, sensory_input: Optional[torch.Tensor]) -> torch.Tensor:
        """Get cortex input, defaulting to zeros for consolidation.

        During consolidation (sensory_input=None), we still need to call
        cortex.forward() with zero input so that:
        1. L2/3 recurrent dynamics can continue
        2. Eligibility traces can decay properly
        3. The cortex state is updated each timestep

        Args:
            sensory_input: External input or None for consolidation

        Returns:
            Tensor of shape [input_size] - either the input or zeros
        """
        if sensory_input is not None:
            return sensory_input
        return torch.zeros(self.config.input_size)

    def _process_events_until(self, end_time: float) -> None:
        """Process all scheduled events up to end_time."""
        while True:
            event = self.scheduler.pop_next()
            if event is None or event.time > end_time:
                # Put event back if we overshot
                if event is not None:
                    self.scheduler.schedule(event)
                break

            # Route event to target region
            if event.target in self.adapters:
                region_impl = self.adapters[event.target]
                output_events = region_impl.process_event(event)

                # Schedule output events
                for out_event in output_events:
                    self.scheduler.schedule(out_event)

                    # Track spikes
                    if out_event.event_type == EventType.SPIKE:
                        payload = out_event.payload
                        if isinstance(payload, SpikePayload):
                            spike_count = int(payload.spikes.sum().item())
                            self._spike_counts[event.target] += spike_count

                            # Update criticality monitor if enabled
                            if self.criticality_monitor is not None:
                                self.criticality_monitor.update(payload.spikes)

                self._events_processed += 1

    def _process_pending_events(self) -> None:
        """Process all pending events (used for reward delivery)."""
        max_time = self._current_time + 100.0  # Process up to 100ms ahead
        self._process_events_until(max_time)

    # =========================================================================
    # GROWTH MANAGEMENT
    # =========================================================================

    def check_growth_needs(self) -> Dict[str, Any]:
        """Check if any brain regions need growth based on capacity metrics.

        Returns:
            Dictionary with region names as keys and growth recommendations
        """
        from thalia.core.growth import GrowthManager

        growth_report = {}

        # Check each major region
        for region_name in ['striatum', 'hippocampus', 'cortex', 'pfc', 'cerebellum']:
            if hasattr(self, region_name):
                region = getattr(self, region_name)
                manager = GrowthManager(region_name=region_name)
                metrics = manager.get_capacity_metrics(region)

                growth_report[region_name] = {
                    'firing_rate': metrics.firing_rate,
                    'weight_saturation': metrics.weight_saturation,
                    'synapse_usage': metrics.synapse_usage,
                    'neuron_count': metrics.neuron_count,
                    'growth_recommended': metrics.growth_recommended,
                    'growth_reason': metrics.growth_reason,
                }

        return growth_report

    def auto_grow(self, threshold: float = 0.8) -> Dict[str, int]:
        """Automatically grow regions that need more capacity.

        When a region grows, this method also updates all connected pathways
        to maintain proper connectivity. This ensures pathway dimensions stay
        synchronized with region sizes.

        Args:
            threshold: Capacity threshold for triggering growth (0.0-1.0)

        Returns:
            Dictionary mapping region names to number of neurons added
        """
        from datetime import datetime

        growth_actions = {}
        report = self.check_growth_needs()

        for region_name, metrics in report.items():
            if metrics['growth_recommended']:
                # Calculate growth amount based on current size
                region = getattr(self, region_name)
                current_size = region.config.n_output
                growth_amount = max(int(current_size * 0.1), 8)  # 10% or minimum 8

                # Add neurons to region
                region.add_neurons(n_new=growth_amount)
                growth_actions[region_name] = growth_amount

                # Update all pathways connected to this region
                self._grow_connected_pathways(region_name, growth_amount)

                # Track growth history
                self._growth_history.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'region': region_name,
                    'neurons_added': growth_amount,
                    'old_size': current_size,
                    'new_size': current_size + growth_amount,
                    'reason': metrics['growth_reason'],
                })

        return growth_actions

    def _grow_connected_pathways(self, region_name: str, growth_amount: int) -> None:
        """Grow all pathways connected to a region that has grown.

        Delegates to PathwayManager for coordinated growth.

        Args:
            region_name: Name of region that grew
            growth_amount: Number of neurons added to region
        """
        self.pathway_manager.grow_connected_pathways(region_name, growth_amount)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def _collect_striatum_diagnostics(self) -> StriatumDiagnostics:
        """Collect structured diagnostics from striatum."""
        striatum = self.striatum
        n_actions = self.config.n_actions
        neurons_per = self.config.neurons_per_action

        # Per-action weight means
        d1_per_action = []
        d2_per_action = []
        net_per_action = []

        for a in range(n_actions):
            start = a * neurons_per
            end = start + neurons_per
            d1_mean = striatum.d1_weights[start:end].mean().item()
            d2_mean = striatum.d2_weights[start:end].mean().item()
            d1_per_action.append(d1_mean)
            d2_per_action.append(d2_mean)
            net_per_action.append(d1_mean - d2_mean)

        # Eligibility traces
        d1_elig_per_action = []
        d2_elig_per_action = []
        for a in range(n_actions):
            start = a * neurons_per
            end = start + neurons_per
            d1_elig_per_action.append(striatum.d1_eligibility[start:end].abs().mean().item())
            d2_elig_per_action.append(striatum.d2_eligibility[start:end].abs().mean().item())

        # UCB and exploration
        action_counts = [int(c) for c in striatum._action_counts.tolist()] if hasattr(striatum, '_action_counts') else []
        total_trials = int(striatum._total_trials) if hasattr(striatum, '_total_trials') else 0
        exploration_prob = getattr(striatum, '_last_exploration_prob', 0.0)

        return StriatumDiagnostics(
            d1_per_action=d1_per_action,
            d2_per_action=d2_per_action,
            net_per_action=net_per_action,
            d1_elig_per_action=d1_elig_per_action,
            d2_elig_per_action=d2_elig_per_action,
            last_action=self.trial_coordinator.get_last_action(),
            exploring=getattr(striatum, '_last_exploring', False),
            exploration_prob=exploration_prob,
            action_counts=action_counts,
            total_trials=total_trials,
        )

    def _collect_hippocampus_diagnostics(self) -> HippocampusDiagnostics:
        """Collect structured diagnostics from hippocampus."""
        hippo = self.hippocampus

        # CA1 activity (key for match/mismatch)
        ca1_spikes = hippo.state.ca1_spikes
        ca1_total = ca1_spikes.sum().item() if ca1_spikes is not None else 0.0

        # Normalize by hippocampus size
        ca1_normalized = ca1_total / max(1, self.config.hippocampus_size)

        # Layer activity
        dg_spikes = hippo.state.dg_spikes.sum().item() if hippo.state.dg_spikes is not None else 0.0
        ca3_spikes = hippo.state.ca3_spikes.sum().item() if hippo.state.ca3_spikes is not None else 0.0

        # Memory metrics
        n_stored = len(hippo.episode_buffer) if hasattr(hippo, 'episode_buffer') else 0

        # Get comparison decision if available
        comparison_decision = getattr(self, '_last_comparison_decision', 'UNKNOWN')

        return HippocampusDiagnostics(
            ca1_total_spikes=ca1_total,
            ca1_normalized=ca1_normalized,
            ca1_similarity=getattr(self, '_last_similarity', 0.0),
            comparison_decision=comparison_decision,
            dg_spikes=dg_spikes,
            ca3_spikes=ca3_spikes,
            ca1_spikes=ca1_total,
            n_stored_episodes=n_stored,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about brain state.

        Returns both structured component diagnostics and raw metrics,
        including pathway and neuromodulator diagnostics.
        """
        # Collect structured component diagnostics
        striatum_diag = self._collect_striatum_diagnostics()
        hippo_diag = self._collect_hippocampus_diagnostics()

        # Record to diagnostics manager
        self.diagnostics.record("striatum", striatum_diag.to_dict())
        self.diagnostics.record("hippocampus", hippo_diag.to_dict())

        diag = {
            # Brain state
            "current_time": self._current_time,
            "theta_phase": self.oscillators.get_phases().get('theta', 0.0),
            "theta_frequency": self.oscillators.theta_freq,
            "spike_counts": self._spike_counts.copy(),
            "events_processed": self._events_processed,
            "last_action": self._last_action,

            # VTA Dopamine System (centralized)
            "dopamine": {
                "global": self.vta.get_global_dopamine(),  # Combined signal to regions
                "tonic": self.vta.get_tonic_dopamine(),    # Slow baseline (intrinsic)
                "phasic": self.vta.get_phasic_dopamine(),  # Fast bursts (external rewards)
            },
            # VTA state for monitoring
            "vta": self.vta.get_state(),

            # Locus Coeruleus (norepinephrine/arousal)
            "locus_coeruleus": self.locus_coeruleus.get_state(),

            # Nucleus Basalis (acetylcholine/encoding)
            "nucleus_basalis": self.nucleus_basalis.get_state(),

            # Legacy key for backwards compatibility
            "global_dopamine": self.vta.get_global_dopamine(),

            # Structured component diagnostics
            "striatum": striatum_diag.to_dict(),
            "hippocampus": hippo_diag.to_dict(),

            # Manager diagnostics
            "neuromodulator_manager": self.neuromodulator_manager.get_diagnostics(),
            "pathway_manager": self.pathway_manager.get_diagnostics(),

            # Summary for quick access
            "summary": {
                "last_action": self._last_action,
                "exploring": striatum_diag.exploring,
                "net_weight_means": striatum_diag.net_per_action,
                "ca1_spikes": hippo_diag.ca1_spikes,
                "dopamine_global": self.vta.get_global_dopamine(),
                "dopamine_tonic": self.vta.get_tonic_dopamine(),
                "dopamine_phasic": self.vta.get_phasic_dopamine(),
                "norepinephrine": self.locus_coeruleus.get_norepinephrine(),
                "arousal": self.locus_coeruleus.get_arousal(),
                "acetylcholine": self.nucleus_basalis.get_acetylcholine(),
                "encoding_mode": self.nucleus_basalis.is_encoding_mode(),
                "encoding_strength": self.nucleus_basalis.get_encoding_strength(),
            },

            # Robustness/Criticality diagnostics
            "criticality": self._get_criticality_diagnostics(),
        }

        # Add pathway diagnostics for all 9 inter-region pathways (backward compatibility)
        diag["pathways"] = {}
        for pathway_name, pathway in self.pathways.items():
            if hasattr(pathway, 'get_diagnostics'):
                diag["pathways"][pathway_name] = pathway.get_diagnostics()

        return diag

    def _get_criticality_diagnostics(self) -> Dict[str, Any]:
        """Get criticality diagnostics if monitor is enabled."""
        if self.criticality_monitor is None:
            return {"enabled": False}

        diag = self.criticality_monitor.get_diagnostics()
        diag["enabled"] = True
        return diag

    def get_structured_diagnostics(self) -> BrainSystemDiagnostics:
        """Get fully structured diagnostics as a dataclass.

        Returns:
            BrainSystemDiagnostics with all component data
        """
        return BrainSystemDiagnostics(
            trial_num=self.diagnostics.trial_count,
            is_match=getattr(self, '_last_is_match', False),
            selected_action=self._last_action or 0,
            correct=getattr(self, '_last_correct', False),
            striatum=self._collect_striatum_diagnostics(),
            hippocampus=self._collect_hippocampus_diagnostics(),
        )

    # =====================================================================
    # CHECKPOINT METHODS
    # =====================================================================

    def save_checkpoint(
        self,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        compression: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save brain checkpoint with metadata.

        This is a convenience method that delegates to CheckpointManager.
        Provides unified save/load interface with validation and logging.

        Args:
            path: Path to save checkpoint file (.pt or .pth)
            metadata: Optional metadata to store (e.g., training info)
            compression: Compression algorithm ('zstd', 'gzip', or None)

        Returns:
            Dict with save info:
                - path: Saved file path
                - size_mb: File size in megabytes
                - components: List of saved component names
                - save_time_s: Time taken to save

        Example:
            info = brain.save_checkpoint(
                "checkpoint_epoch10.pt",
                metadata={"epoch": 10, "accuracy": 0.95},
                compression='zstd'
            )
            print(f"Saved {info['size_mb']:.2f}MB in {info['save_time_s']:.2f}s")
        """
        return self.checkpoint_manager.save(
            path=path,
            metadata=metadata,
            compression=compression,
        )

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load brain checkpoint from file.

        This is a convenience method that delegates to CheckpointManager.
        Validates config compatibility and loads all components.

        Args:
            path: Path to checkpoint file
            strict: If True, enforce config compatibility checks

        Returns:
            Dict with checkpoint info:
                - metadata: Stored metadata dict
                - components: List of loaded component names
                - config_compatible: Whether configs match

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If config incompatible and strict=True

        Example:
            info = brain.load_checkpoint("checkpoint_epoch10.pt")
            print(f"Loaded {len(info['components'])} components")
            print(f"Metadata: {info['metadata']}")
        """
        return self.checkpoint_manager.load(
            path=path,
            strict=strict,
        )
