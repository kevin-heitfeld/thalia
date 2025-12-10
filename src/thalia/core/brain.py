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
from typing import Dict, Optional, Any
import torch
import torch.nn as nn

from .event_system import (
    Event, EventType, EventScheduler,
    SpikePayload,
    get_axonal_delay,
)
from .event_regions import (
    EventDrivenCortex, EventDrivenHippocampus, EventDrivenPFC, EventDrivenStriatum,
    EventDrivenCerebellum, EventRegionConfig,
)
from .vta import VTADopamineSystem, VTAConfig
from .locus_coeruleus import LocusCoeruleusSystem, LocusCoeruleusConfig
from .nucleus_basalis import NucleusBasalisSystem, NucleusBasalisConfig
from .homeostatic_regulation import NeuromodulatorCoordination
from .parallel_executor import ParallelExecutor
from .diagnostics import (
    DiagnosticsManager,
    StriatumDiagnostics,
    HippocampusDiagnostics,
    BrainSystemDiagnostics,
)

# Robustness mechanisms
from ..diagnostics.criticality import CriticalityMonitor

# Import actual region implementations
from ..regions.cortex import LayeredCortex, LayeredCortexConfig
from ..regions.cortex.predictive_cortex import PredictiveCortex, PredictiveCortexConfig
from ..regions.hippocampus import TrisynapticHippocampus, TrisynapticConfig
from ..regions.prefrontal import Prefrontal, PrefrontalConfig
from ..regions.striatum import Striatum, StriatumConfig
from ..regions.cerebellum import Cerebellum, CerebellumConfig
from ..regions.theta_dynamics import TemporalIntegrationLayer

# Import config types
from ..config.brain_config import CortexType

# Import pathways for top-down modulation and consolidation
from ..integration.pathways.spiking_attention import (
    SpikingAttentionPathway, SpikingAttentionPathwayConfig
)
from ..integration.pathways.spiking_replay import (
    SpikingReplayPathway, SpikingReplayPathwayConfig
)
from ..integration.spiking_pathway import (
    SpikingPathway, SpikingPathwayConfig, TemporalCoding, SpikingLearningRule
)


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

        # Process sample (encoding phase)
        sample_result = brain.process_sample(pattern, n_timesteps=15)

        # Delay period
        delay_result = brain.delay(n_timesteps=10)

        # Test and respond
        test_result = brain.process_test(test_pattern, n_timesteps=15)

        # Get action selection
        action, confidence = brain.select_action()

        # Learn from external reward (intrinsic rewards are computed continuously)
        brain.deliver_reward(external_reward=1.0)
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
        """
        from thalia.config import ThaliaConfig

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
            n_input=self.config.input_size,
            n_output=self.config.cortex_size,
            dt=self.config.dt_ms,  # RegionConfigBase uses 'dt' not 'dt_ms'
            device=self.config.device,
        )

        # Select implementation based on config
        if self.config.cortex_type == CortexType.PREDICTIVE:
            # PredictiveCortex with local error learning
            # Convert LayeredCortexConfig to PredictiveCortexConfig
            pred_config = PredictiveCortexConfig(
                n_input=cortex_config.n_input,
                n_output=cortex_config.n_output,
                dt_ms=cortex_config.dt_ms,
                device=cortex_config.device,
                # Copy layer parameters
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
                ffi_enabled=cortex_config.ffi_enabled,
                ffi_threshold=cortex_config.ffi_threshold,
                ffi_strength=cortex_config.ffi_strength,
                ffi_tau=cortex_config.ffi_tau,
                bcm_enabled=cortex_config.bcm_enabled,
                bcm_tau_theta=cortex_config.bcm_tau_theta,
                output_layer=cortex_config.output_layer,
                dual_output=cortex_config.dual_output,
                # Predictive-specific defaults
                prediction_enabled=True,
                use_attention=True,
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
        _hippocampus_impl = TrisynapticHippocampus(TrisynapticConfig(
            n_input=cortex_to_hippo_size,
            n_output=config.hippocampus_size,
            dt=config.dt_ms,
            ec_l3_input_size=config.input_size,
            device=config.device,
        ))

        # 3. PFC: Working memory
        pfc_input_size = cortex_to_hippo_size + config.hippocampus_size
        _pfc_impl = Prefrontal(PrefrontalConfig(
            n_input=pfc_input_size,
            n_output=config.pfc_size,
            dt=config.dt_ms,
            device=config.device,
        ))
        _pfc_impl.reset_state()

        # 4. STRIATUM: Action selection
        # Receives: cortex L5 + hippocampus + PFC
        # NOTE: Pass n_output=n_actions (not n_actions*neurons_per_action)
        # The Striatum internally handles population coding expansion
        striatum_input = self._cortex_l5_size + config.hippocampus_size + config.pfc_size
        _striatum_impl = Striatum(StriatumConfig(
            n_input=striatum_input,
            n_output=config.n_actions,  # Number of actions, NOT total neurons
            neurons_per_action=config.neurons_per_action,
            device=config.device,
        ))
        _striatum_impl.reset_state()

        # 5. CEREBELLUM: Motor refinement
        # Receives: striatum output (action signals)
        # After population coding, striatum outputs n_actions * neurons_per_action
        cerebellum_input = config.n_actions * config.neurons_per_action
        _cerebellum_impl = Cerebellum(CerebellumConfig(
            n_input=cerebellum_input,
            n_output=config.n_actions,  # Refined motor output
            device=config.device,
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
            pfc_size=config.pfc_size,  # For top-down projection
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
            hippocampus_input_size=config.hippocampus_size,
        )

        self.striatum = EventDrivenStriatum(
            EventRegionConfig(
                name="striatum",
                output_targets=["cerebellum"],  # Striatum -> Cerebellum
            ),
            _striatum_impl,
            cortex_input_size=self._cortex_l5_size,
            hippocampus_input_size=config.hippocampus_size,
            pfc_input_size=config.pfc_size,
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
            device=torch.device(config.device),
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
            dt_ms=config.dt_ms,
            device=config.device,
            theta_freq=config.theta_frequency_hz,
            couplings=config.oscillator_couplings,
        )

        # =====================================================================
        # LEARNABLE PATHWAYS
        # =====================================================================
        # All inter-region connections are explicit pathways with:
        # - Independent STDP learning during forward passes
        # - Growth support (expand when connected regions grow)
        # - Checkpoint compatibility
        # - Health monitoring and diagnostics
        #
        # This enables curriculum learning and coordinated adaptation.

        # 1. Cortex L2/3 → Hippocampus (encoding pathway)
        self.cortex_to_hippo_pathway = SpikingPathway(
            SpikingPathwayConfig(
                source_size=self._cortex_l23_size,
                target_size=cortex_to_hippo_size,
                learning_rule=SpikingLearningRule.STDP,
                temporal_coding=TemporalCoding.PHASE,  # Theta phase coding
                stdp_lr=0.001,
                device=config.device,
            )
        )

        # 2. Cortex L5 → Striatum (action selection pathway)
        self.cortex_to_striatum_pathway = SpikingPathway(
            SpikingPathwayConfig(
                source_size=self._cortex_l5_size,
                target_size=self._cortex_l5_size,  # Match striatum input expectation
                learning_rule=SpikingLearningRule.DOPAMINE_STDP,  # Reward-modulated
                temporal_coding=TemporalCoding.RATE,
                stdp_lr=0.002,
                device=config.device,
            )
        )

        # 3. Cortex L2/3 → PFC (working memory input)
        self.cortex_to_pfc_pathway = SpikingPathway(
            SpikingPathwayConfig(
                source_size=self._cortex_l23_size,
                target_size=self._cortex_l23_size,  # PFC receives cortex + hippo
                learning_rule=SpikingLearningRule.STDP,
                temporal_coding=TemporalCoding.SYNCHRONY,  # Binding via synchrony
                stdp_lr=0.0015,
                device=config.device,
            )
        )

        # 4. Hippocampus → PFC (episodic to working memory)
        self.hippo_to_pfc_pathway = SpikingPathway(
            SpikingPathwayConfig(
                source_size=config.hippocampus_size,
                target_size=config.hippocampus_size,
                learning_rule=SpikingLearningRule.STDP,
                temporal_coding=TemporalCoding.PHASE,  # Theta-coupled
                stdp_lr=0.001,
                device=config.device,
            )
        )

        # 5. Hippocampus → Striatum (context for action selection)
        self.hippo_to_striatum_pathway = SpikingPathway(
            SpikingPathwayConfig(
                source_size=config.hippocampus_size,
                target_size=config.hippocampus_size,
                learning_rule=SpikingLearningRule.DOPAMINE_STDP,  # Reward-modulated
                temporal_coding=TemporalCoding.PHASE,
                stdp_lr=0.0015,
                device=config.device,
            )
        )

        # 6. PFC → Striatum (goal-directed control)
        self.pfc_to_striatum_pathway = SpikingPathway(
            SpikingPathwayConfig(
                source_size=config.pfc_size,
                target_size=config.pfc_size,
                learning_rule=SpikingLearningRule.DOPAMINE_STDP,  # Reward-modulated
                temporal_coding=TemporalCoding.RATE,
                stdp_lr=0.002,
                device=config.device,
            )
        )

        # 7. Striatum → Cerebellum (action refinement)
        self.striatum_to_cerebellum_pathway = SpikingPathway(
            SpikingPathwayConfig(
                source_size=config.n_actions * config.neurons_per_action,
                target_size=config.n_actions * config.neurons_per_action,
                learning_rule=SpikingLearningRule.STDP,
                temporal_coding=TemporalCoding.LATENCY,  # Precise timing for motor control
                stdp_lr=0.001,
                device=config.device,
            )
        )

        # 8. PFC → Cortex (top-down attention modulation) [SPECIALIZED]
        self.attention_pathway = SpikingAttentionPathway(
            SpikingAttentionPathwayConfig(
                source_size=config.pfc_size,
                target_size=config.input_size,
                device=config.device,
            )
        )

        # 9. Hippocampus → Cortex (replay/consolidation during sleep) [SPECIALIZED]
        self.replay_pathway = SpikingReplayPathway(
            SpikingReplayPathwayConfig(
                source_size=config.hippocampus_size,
                target_size=config.cortex_size,
                device=config.device,
            )
        )

        # Pathway registry for iteration (growth, checkpointing, diagnostics)
        self.pathways = {
            "cortex_to_hippo": self.cortex_to_hippo_pathway,
            "cortex_to_striatum": self.cortex_to_striatum_pathway,
            "cortex_to_pfc": self.cortex_to_pfc_pathway,
            "hippo_to_pfc": self.hippo_to_pfc_pathway,
            "hippo_to_striatum": self.hippo_to_striatum_pathway,
            "pfc_to_striatum": self.pfc_to_striatum_pathway,
            "striatum_to_cerebellum": self.striatum_to_cerebellum_pathway,
            "attention": self.attention_pathway,
            "replay": self.replay_pathway,
        }

        # Pathway-region connection tracking for coordinated growth
        # Maps: region_name -> list of (pathway, dimension_type)
        # dimension_type: 'source' or 'target'
        self._region_pathway_connections = {
            'cortex': [
                (self.cortex_to_hippo_pathway, 'source'),
                (self.cortex_to_striatum_pathway, 'source'),
                (self.cortex_to_pfc_pathway, 'source'),
                (self.replay_pathway, 'target'),
                (self.attention_pathway, 'target'),
            ],
            'hippocampus': [
                (self.cortex_to_hippo_pathway, 'target'),
                (self.hippo_to_pfc_pathway, 'source'),
                (self.hippo_to_striatum_pathway, 'source'),
                (self.replay_pathway, 'source'),
            ],
            'pfc': [
                (self.cortex_to_pfc_pathway, 'target'),
                (self.hippo_to_pfc_pathway, 'target'),
                (self.pfc_to_striatum_pathway, 'source'),
                (self.attention_pathway, 'source'),
            ],
            'striatum': [
                (self.cortex_to_striatum_pathway, 'target'),
                (self.hippo_to_striatum_pathway, 'target'),
                (self.pfc_to_striatum_pathway, 'target'),
                (self.striatum_to_cerebellum_pathway, 'source'),
            ],
            'cerebellum': [
                (self.striatum_to_cerebellum_pathway, 'target'),
            ],
        }

        # =====================================================================
        # EVENT SCHEDULER (for sequential mode)
        # =====================================================================

        self.scheduler = EventScheduler()

        # =====================================================================
        # PARALLEL EXECUTION (optional)
        # =====================================================================

        self._parallel_executor: Optional[ParallelExecutor] = None
        if config.parallel:
            self._init_parallel_executor()

        # State tracking
        self._last_cortex_output: Optional[torch.Tensor] = None
        self._last_hippo_output: Optional[torch.Tensor] = None
        self._last_pfc_output: Optional[torch.Tensor] = None
        self._last_action: Optional[int] = None

        # =====================================================================
        # CENTRALIZED NEUROMODULATOR SYSTEMS
        # =====================================================================
        # VTA DOPAMINE SYSTEM (reward prediction error)
        # Manages tonic + phasic dopamine, broadcasts to all regions
        self.vta = VTADopamineSystem(VTAConfig())

        # LOCUS COERULEUS (norepinephrine arousal)
        # Manages arousal/uncertainty, broadcasts NE to all regions
        self.locus_coeruleus = LocusCoeruleusSystem(LocusCoeruleusConfig())

        # NUCLEUS BASALIS (acetylcholine attention/encoding)
        # Manages encoding/retrieval mode, broadcasts ACh to cortex/hippocampus
        self.nucleus_basalis = NucleusBasalisSystem(NucleusBasalisConfig())

        # NEUROMODULATOR COORDINATION
        # Implements biological interactions between systems (DA-ACh, NE-ACh, DA-NE)
        self.neuromodulator_coordination = NeuromodulatorCoordination()

        # Monitoring
        self._spike_counts: Dict[str, int] = {name: 0 for name in self.adapters}
        self._events_processed: int = 0

        # Growth history tracking
        self._growth_history: list = []

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

        self.diagnostics = DiagnosticsManager(level=config.diagnostic_level)
        self.diagnostics.configure_component("striatum", enabled=True)
        self.diagnostics.configure_component("hippocampus", enabled=True)
        self.diagnostics.configure_component("cortex", enabled=True)
        self.diagnostics.configure_component("pfc", enabled=True)
        self.diagnostics.configure_component("cerebellum", enabled=True)

    @classmethod
    def from_thalia_config(cls, config: "ThaliaConfig") -> "EventDrivenBrain":
        """Create EventDrivenBrain from unified ThaliaConfig.

        This is the recommended way to create a brain, as it uses the
        unified configuration system that eliminates parameter duplication.

        Args:
            config: ThaliaConfig with all settings

        Returns:
            EventDrivenBrain instance

        Example:
            from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig

            config = ThaliaConfig(
                global_=GlobalConfig(device="cuda"),
                brain=BrainConfig(sizes=RegionSizes(cortex_size=256)),
            )
            brain = EventDrivenBrain.from_thalia_config(config)
        """
        # Create brain via __init__
        brain = cls(config)

        # Initialize CriticalityMonitor if robustness config enables it
        if config.robustness.enable_criticality:
            brain.criticality_monitor = CriticalityMonitor(config.robustness.criticality)

        return brain

    def _init_parallel_executor(self) -> None:
        """Initialize parallel executor with region creators.

        Note: Parallel mode is experimental. On Windows, multiprocessing uses
        "spawn" which requires pickleable region creators. This implementation
        uses the existing module-level creator functions from parallel_executor.py
        for now. For full config customization in parallel mode, consider using
        the sequential mode (parallel=False).
        """
        from .parallel_executor import (
            _create_real_cortex, _create_real_hippocampus,
            _create_real_pfc, _create_real_striatum,
        )

        # Create parallel executor with module-level creators
        # These are pickle-able because they're defined at module level
        self._parallel_executor = ParallelExecutor(
            region_creators={
                "cortex": _create_real_cortex,
                "hippocampus": _create_real_hippocampus,
                "pfc": _create_real_pfc,
                "striatum": _create_real_striatum,
            },
            theta_frequency=self.config.theta_frequency_hz,
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
    # HIGH-LEVEL TRIAL APIs
    # =========================================================================

    def process_sample(
        self,
        sample_pattern: torch.Tensor,
        n_timesteps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process sample pattern (encoding phase).

        Routes information through:
        1. Sensory → Cortex (feature extraction)
        2. Cortex → Hippocampus (encoding via CA3 attractors)
        3. Cortex + Hippo → PFC (working memory)

        With continuous learning, state transitions happen via natural dynamics
        (decay, FFI) rather than explicit resets. Call new_sequence() only when
        starting a completely new, unrelated sequence.

        Args:
            sample_pattern: Input pattern to encode [input_size]
            n_timesteps: Number of timesteps to process

        Returns:
            Dict with region activities for monitoring
        """
        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert sample_pattern.shape[-1] == self.config.input_size, (
            f"EventDrivenBrain.process_sample: sample_pattern has shape {sample_pattern.shape} "
            f"but input_size={self.config.input_size}. Check that input matches brain config."
        )

        n_timesteps = n_timesteps or self.config.encoding_timesteps

        # Note: No new_trial()/clear() calls here - continuous processing
        # State transitions happen via natural dynamics (decay, FFI)
        # Call new_sequence() explicitly when starting unrelated sequences
        # Gamma slot auto-advances in hippocampus forward() - no explicit position needed

        # Process timesteps
        results = self._run_timesteps(
            sensory_input=sample_pattern,
            n_timesteps=n_timesteps,
        )

        # Capture PFC output for decoder (language model uses this)
        if hasattr(self.pfc.impl, 'state') and self.pfc.impl.state is not None:
            if self.pfc.impl.state.working_memory is not None:
                self._last_pfc_output = self.pfc.impl.state.working_memory.squeeze(0).clone()
            elif self.pfc.impl.state.spikes is not None:
                self._last_pfc_output = self.pfc.impl.state.spikes.squeeze(0).clone()

        return results

    def delay(
        self,
        n_timesteps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Delay period (maintenance phase).

        PFC maintains working memory, other regions decay.

        Args:
            n_timesteps: Number of delay timesteps

        Returns:
            Dict with region activities
        """
        n_timesteps = n_timesteps or self.config.delay_timesteps

        results = self._run_timesteps(
            sensory_input=None,
            n_timesteps=n_timesteps,
        )

        return results

    def process_test(
        self,
        test_pattern: torch.Tensor,
        n_timesteps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process test pattern (retrieval/comparison phase).

        Hippocampus compares test to stored sample via NMDA mechanism.

        Args:
            test_pattern: Test pattern to compare [input_size]
            n_timesteps: Number of timesteps

        Returns:
            Dict with region activities and comparison result
        """
        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert test_pattern.shape[-1] == self.config.input_size, (
            f"EventDrivenBrain.process_test: test_pattern has shape {test_pattern.shape} "
            f"but input_size={self.config.input_size}. Check that input matches brain config."
        )

        n_timesteps = n_timesteps or self.config.test_timesteps

        results = self._run_timesteps(
            sensory_input=test_pattern,
            n_timesteps=n_timesteps,
        )

        return results

    def select_action(self, explore: bool = True) -> tuple[int, float]:
        """Select action based on current striatum state.

        Uses the striatum's finalize_action method which handles:
        - Accumulated NET votes (D1-D2)
        - UCB exploration bonus
        - Softmax selection

        Args:
            explore: Whether to allow exploration

        Returns:
            (action, confidence): Selected action index and confidence [0, 1]
        """
        # Use striatum's finalize_action method
        result = self.striatum.impl.finalize_action(explore=explore)

        action = result.get("selected_action", 0)
        probs = result.get("probs", None)

        if probs is not None:
            confidence = float(probs[action].item())
        else:
            confidence = 1.0

        self._last_action = action

        return action, confidence

    def deliver_reward(self, external_reward: float = 0.0) -> None:
        """Deliver external reward signal for learning.

        Brain acts as VTA (ventral tegmental area):
        1. Combines external reward with current intrinsic reward
        2. Queries striatum for expected value of the action taken
        3. Computes reward prediction error (RPE = reward - expected)
        4. Normalizes RPE using adaptive scaling
        5. Broadcasts normalized dopamine to ALL regions

        Note: Intrinsic reward (from prediction errors) is computed CONTINUOUSLY
        during `_run_timesteps_sequential`. This method adds external reward
        on top for task-based learning.

        Args:
            external_reward: Task-based reward value (-1 to +1), default 0.0
        """
        # =====================================================================
        # STEP 1: Compute phasic dopamine from external reward
        # =====================================================================
        # External rewards create PHASIC bursts/dips that add to tonic baseline.
        # The tonic component flows continuously via _update_neuromodulators().
        #
        # If no external reward (0.0), we just compute the phasic component from
        # the current state for the striatum value update.

        # =====================================================================
        # STEP 2: Get expected value from striatum
        # =====================================================================
        expected = self.striatum.impl.get_expected_value(self._last_action)

        # =====================================================================
        # STEP 3: Compute RPE and deliver to VTA
        # =====================================================================
        # VTA handles normalization, phasic burst computation, and decay
        if external_reward != 0.0:
            # External reward
            normalized_rpe = self.vta.deliver_reward(
                external_reward=external_reward,
                expected_value=expected
            )
        else:
            # Pure intrinsic reward
            intrinsic = self._compute_intrinsic_reward()
            normalized_rpe = self.vta.deliver_reward(
                external_reward=intrinsic,
                expected_value=expected
            )

        # =====================================================================
        # STEP 4: Get dopamine signal from VTA and broadcast to ALL regions
        # =====================================================================
        dopamine = self.vta.get_global_dopamine()

        self.cortex.impl.set_dopamine(dopamine)
        self.hippocampus.impl.set_dopamine(dopamine)
        self.pfc.impl.set_dopamine(dopamine)
        self.striatum.impl.set_dopamine(dopamine)
        self.cerebellum.impl.set_dopamine(dopamine)

        # =====================================================================
        # STEP 7: Trigger striatum learning (D1/D2 plasticity)
        # =====================================================================
        # Use the external reward for striatum value updates
        reward_for_striatum = external_reward if external_reward != 0.0 else self._compute_intrinsic_reward()
        if self._last_action is not None:
            self.striatum.impl.deliver_reward(reward_for_striatum)

        # =====================================================================
        # STEP 8: Update value estimate in striatum
        # =====================================================================
        if self._last_action is not None:
            self.striatum.impl.update_value_estimate(self._last_action, reward_for_striatum)

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
        if hasattr(self.hippocampus.impl, 'get_pattern_similarity'):
            similarity = self.hippocampus.impl.get_pattern_similarity()
            if similarity is not None:
                # Similarity is 0-1, map to [-1, 1]
                hippo_reward = 2.0 * similarity - 1.0
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
        4. Broadcasts all signals to regions

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
        # 4. BROADCAST TO ALL REGIONS (with coordination)
        # =====================================================================
        # Get raw neuromodulator signals
        dopamine = self.vta.get_global_dopamine()
        norepinephrine = self.locus_coeruleus.get_norepinephrine()
        acetylcholine = self.nucleus_basalis.get_acetylcholine()

        # Apply biological coordination between systems
        # 1. NE-ACh: Optimal encoding at moderate arousal (inverted-U)
        acetylcholine = self.neuromodulator_coordination.coordinate_ne_ach(
            norepinephrine, acetylcholine
        )

        # 2. DA-ACh: High reward without novelty suppresses encoding
        acetylcholine = self.neuromodulator_coordination.coordinate_da_ach(
            dopamine, acetylcholine
        )

        # 3. DA-NE: High uncertainty + reward enhances both
        dopamine, norepinephrine = self.neuromodulator_coordination.coordinate_da_ne(
            dopamine, norepinephrine, prediction_error
        )

        # Broadcast coordinated signals to all regions
        self.cortex.impl.set_dopamine(dopamine)
        self.cortex.impl.set_norepinephrine(norepinephrine)
        self.cortex.impl.set_acetylcholine(acetylcholine)

        self.hippocampus.impl.set_dopamine(dopamine)
        self.hippocampus.impl.set_norepinephrine(norepinephrine)
        self.hippocampus.impl.set_acetylcholine(acetylcholine)

        self.pfc.impl.set_dopamine(dopamine)
        self.pfc.impl.set_norepinephrine(norepinephrine)
        self.pfc.impl.set_acetylcholine(acetylcholine)

        self.striatum.impl.set_dopamine(dopamine)
        self.striatum.impl.set_norepinephrine(norepinephrine)
        self.striatum.impl.set_acetylcholine(acetylcholine)

        self.cerebellum.impl.set_dopamine(dopamine)
        self.cerebellum.impl.set_norepinephrine(norepinephrine)
        self.cerebellum.impl.set_acetylcholine(acetylcholine)

        # =====================================================================
        # 5. BROADCAST OSCILLATOR PHASES
        # =====================================================================
        self._broadcast_oscillator_phases()

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

    def store_experience(
        self,
        is_match: bool,
        selected_action: int,
        correct: bool,
        reward: float,
        sample_pattern: Optional[torch.Tensor] = None,
        test_pattern: Optional[torch.Tensor] = None,
    ) -> None:
        """Store experience for later replay.

        Delegates to hippocampus with priority boosting for
        rare/important experiences. This should be called after
        deliver_reward() to store the completed trial.

        Args:
            is_match: Whether trial was a match
            selected_action: Action that was taken (0=MATCH, 1=NO-MATCH)
            correct: Whether action was correct
            reward: Reward received
            sample_pattern: Original sample (optional)
            test_pattern: Test pattern (optional)
        """
        priority_boost = 0.0

        # Boost correct NOMATCH selections (rare!)
        if correct and selected_action == 1:
            priority_boost += 3.0

        # Boost NO-MATCH trials generally (minority class)
        if not is_match:
            priority_boost += 1.5

        # Construct state from current brain activity
        # Combine cortex L5 + hippocampus + PFC as the state
        cortex_L5 = self.cortex.impl.state.l5_spikes
        if cortex_L5 is None:
            cortex_L5 = torch.zeros(1, self._cortex_l5_size)

        hippo_out = self.hippocampus.impl.state.ca1_spikes
        if hippo_out is None:
            hippo_out = torch.zeros(1, self.config.hippocampus_size)

        pfc_out = self.pfc.impl.state.spikes
        if pfc_out is None:
            pfc_out = torch.zeros(1, self.config.pfc_size)

        combined_state = torch.cat([
            cortex_L5.view(-1),
            hippo_out.view(-1),
            pfc_out.view(-1),
        ])

        self.hippocampus.impl.store_episode(
            state=combined_state,
            action=selected_action,
            reward=reward,
            correct=correct,
            context=sample_pattern,
            metadata={
                "is_match": is_match,
                "test_pattern": test_pattern.clone() if test_pattern is not None else None,
            },
            priority_boost=priority_boost,
        )

    def reset_state(self) -> None:
        """Reset brain state for new episode.

        This is a HARD reset - use for completely new, unrelated episodes.
        For starting a new sequence within the same session, use new_sequence().
        """
        self._current_time = 0.0
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
        - pathways: State from all inter-region pathways (9 pathways total)
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
            "pathways": {},
            "oscillators": {
                "theta_frequency_hz": self.oscillators.theta_freq,
                "phases": self.oscillators.get_phases(),
            },
            "scheduler": {
                "current_time": self._current_time,
                "events_processed": self._events_processed,
            },
            "trial_state": {
                "spike_counts": self._spike_counts.copy(),
                "last_action": self._last_action,
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

        # Add pathway states if they exist
        if hasattr(self, 'attention_pathway') and self.attention_pathway is not None:
            state_dict["pathways"]["attention"] = self.attention_pathway.get_state()

        if hasattr(self, 'replay_pathway') and self.replay_pathway is not None:
            state_dict["pathways"]["replay"] = self.replay_pathway.get_state()

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
            - Pathway configurations
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

        # Restore pathway states if they exist
        pathways = state_dict.get("pathways", {})
        for pathway_name, pathway_state in pathways.items():
            if pathway_name in self.pathways and self.pathways[pathway_name] is not None:
                self.pathways[pathway_name].load_state(pathway_state)

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
        # Get novelty-based learning boost if available
        novelty_boost = self._get_novelty_boost()
        modulated_reward = reward * novelty_boost

        # Deliver to VTA and broadcast
        expected = self.striatum.impl.get_expected_value(selected_action)
        self.vta.deliver_reward(
            external_reward=modulated_reward,
            expected_value=expected
        )
        dopamine = self.vta.get_global_dopamine()

        # Broadcast to all regions
        self.cortex.impl.set_dopamine(dopamine)
        self.hippocampus.impl.set_dopamine(dopamine)
        self.pfc.impl.set_dopamine(dopamine)
        self.striatum.impl.set_dopamine(dopamine)
        self.cerebellum.impl.set_dopamine(dopamine)

        # 1. Real learning: update striatum for SELECTED action
        real_result = self.striatum.impl.deliver_reward(modulated_reward)

        # 2. Counterfactual: what would the OTHER action have gotten?
        other_action = 1 - selected_action

        # Determine counterfactual reward:
        # - If trial is MATCH: MATCH action (0) would get +1, NOMATCH (1) would get -1
        # - If trial is NOMATCH: NOMATCH action (1) would get +1, MATCH (0) would get -1
        correct_action = 0 if is_match else 1
        counterfactual_reward = 1.0 if (other_action == correct_action) else -1.0

        # Apply counterfactual learning
        counterfactual_result = {}
        if hasattr(self.striatum.impl, 'deliver_counterfactual_reward'):
            counterfactual_result = self.striatum.impl.deliver_counterfactual_reward(
                reward=counterfactual_reward,
                action=other_action,
                counterfactual_scale=counterfactual_scale,
            )

        # Reset eligibility after both learnings
        if hasattr(self.striatum.impl, 'reset_eligibility'):
            self.striatum.impl.reset_eligibility()

        # Update specialized pathways (most pathways already learned during forward)
        attention_result = {}
        if hasattr(self.attention_pathway, 'learn'):
            attention_result = self.attention_pathway.learn(
                source_activity=torch.zeros(self.config.pfc_size),
                target_activity=torch.zeros(self.config.cortex_size),
                dopamine=modulated_reward,
            )

        return {
            "real": real_result,
            "counterfactual": counterfactual_result,
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

    def _run_timesteps(
        self,
        sensory_input: Optional[torch.Tensor],
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Run simulation for specified timesteps.

        Delegates to parallel executor if parallel mode is enabled,
        otherwise runs sequentially in the main process.

        Note: Trial phase for hippocampus is determined automatically
        from oscillator states (theta encoding/retrieval strength).
        """
        if self._parallel_executor is not None:
            return self._run_timesteps_parallel(
                sensory_input, n_timesteps
            )
        return self._run_timesteps_sequential(
            sensory_input, n_timesteps
        )

    def _run_timesteps_parallel(
        self,
        sensory_input: Optional[torch.Tensor],
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Run simulation using parallel executor."""
        assert self._parallel_executor is not None

        end_time = self._current_time + n_timesteps * self.config.dt_ms

        # Get effective input (zero tensor for consolidation, allows recurrent dynamics)
        cortex_input = self._get_cortex_input(sensory_input)

        # Inject sensory input (always - even zeros for consolidation)
        self._parallel_executor.inject_sensory_input(
            cortex_input,
            target="cortex",
            time=self._current_time,
        )

        # Run parallel simulation
        result = self._parallel_executor.run_until(end_time)

        # Update local state
        self._current_time = end_time
        self._spike_counts = result["spike_counts"]
        self._events_processed = result["events_processed"]

        return {
            "cortex_activity": torch.zeros(self._cortex_l23_size),
            "hippocampus_activity": torch.zeros(self.config.hippocampus_size),
            "pfc_activity": torch.zeros(self.config.pfc_size),
            "spike_counts": self._spike_counts.copy(),
            "events_processed": self._events_processed,
            "final_time": self._current_time,
        }

    def _run_timesteps_sequential(
        self,
        sensory_input: Optional[torch.Tensor],
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Run simulation sequentially in main process."""

        # Track activities for monitoring
        cortex_total = torch.zeros(self._cortex_l23_size)
        hippo_total = torch.zeros(self.config.hippocampus_size)
        pfc_total = torch.zeros(self.config.pfc_size)

        for t in range(n_timesteps):
            step_time = self._current_time + t * self.config.dt_ms

            # Advance oscillators (once per timestep, like dopamine)
            self.oscillators.advance(self.config.dt_ms)

            # Schedule sensory input (or zero input for consolidation)
            cortex_input = self._get_cortex_input(sensory_input)
            delay = get_axonal_delay("sensory", "cortex")
            event = Event(
                time=step_time + delay,
                event_type=EventType.SENSORY,
                source="sensory_input",
                target="cortex",
                payload=SpikePayload(spikes=cortex_input),
            )
            self.scheduler.schedule(event)

            # Dopamine is broadcast directly to all regions via set_dopamine() above

            # =========================================================
            # CONTINUOUS NEUROMODULATOR UPDATES
            # =========================================================
            # Update all centralized neuromodulator systems:
            # - VTA dopamine (tonic from intrinsic reward, phasic decay)
            # - Locus coeruleus NE (arousal from uncertainty)
            # This happens every timestep for continuous modulation.
            self._update_neuromodulators()

        # Update time
        end_time = self._current_time + n_timesteps * self.config.dt_ms

        # Process all events up to end_time
        self._process_events_until(end_time)

        self._current_time = end_time

        return {
            "cortex_activity": cortex_total,
            "hippocampus_activity": hippo_total,
            "pfc_activity": pfc_total,
            "spike_counts": self._spike_counts.copy(),
            "events_processed": self._events_processed,
            "final_time": self._current_time,
        }

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

        When a region adds neurons, connected pathways need to expand their
        weight matrices to accommodate the new connections.

        Args:
            region_name: Name of region that grew
            growth_amount: Number of neurons added to region
        """
        if region_name not in self._region_pathway_connections:
            return

        connections = self._region_pathway_connections[region_name]

        for pathway, dimension_type in connections:
            if dimension_type == 'source':
                # Region is source → pathway needs more input connections
                # This would require expanding pathway's source_size and input weights
                # For now, we don't support this (would need pathway.expand_source())
                pass
            elif dimension_type == 'target':
                # Region is target → pathway needs more output connections
                # Pathway's target_size should grow
                pathway.add_neurons(n_new=growth_amount)

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
            last_action=self._last_action,
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
        including pathway diagnostics for all 9 inter-region connections.
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

        # Add pathway diagnostics for all 9 inter-region pathways
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


# =============================================================================
# SIMPLE TEST
# =============================================================================

def test_event_driven_brain():
    """Basic test of EventDrivenBrain."""
    print("\n=== Test: EventDrivenBrain ===")

    from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

    config = ThaliaConfig(
        global_=GlobalConfig(device="cpu"),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=100,
                cortex_size=64,
                hippocampus_size=40,
                pfc_size=20,
                n_actions=2,
            ),
        ),
    )

    brain = EventDrivenBrain.from_thalia_config(config)
    print(f"  Created brain with {len(brain.adapters)} adapters")

    # Create sample pattern
    sample = (torch.rand(100) > 0.5).float()
    print(f"  Sample pattern: {sample.sum().item():.0f}/100 active")

    # Process sample
    print("\n  Processing sample (encoding)...")
    result = brain.process_sample(sample, n_timesteps=10)
    print(f"    Events processed: {result['events_processed']}")
    print(f"    Spike counts: {result['spike_counts']}")

    # Delay
    print("\n  Delay period...")
    result = brain.delay(n_timesteps=5)
    print(f"    Events processed: {result['events_processed']}")

    # Process test (same pattern = match)
    print("\n  Processing test (retrieval)...")
    result = brain.process_test(sample, n_timesteps=10)
    print(f"    Events processed: {result['events_processed']}")

    # Select action
    action, confidence = brain.select_action()
    print(f"\n  Selected action: {action} (confidence: {confidence:.2f})")

    # Deliver external reward (intrinsic rewards flow continuously)
    print("  Delivering external reward...")
    brain.deliver_reward(external_reward=1.0)

    # Diagnostics
    diag = brain.get_diagnostics()
    print(f"\n  Final state:")
    print(f"    Time: {diag['current_time']:.1f}ms")
    print(f"    Phase: {diag['trial_phase']}")
    print(f"    Total events: {diag['events_processed']}")

    print("\n  PASSED: EventDrivenBrain works correctly")


if __name__ == "__main__":
    test_event_driven_brain()
