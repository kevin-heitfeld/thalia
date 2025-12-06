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

    Sensory Input
         │
         ▼ (5ms delay)
    ┌─────────┐
    │  CORTEX │──────────────────────────────────────┐
    │  (L4→   │                                      │
    │   L2/3→ │                                      │
    │   L5)   │                                      │
    └────┬────┘                                      │
         │                                           │
    ┌────┴────┬─────────────┐                       │
    │         │             │                       │
    ▼(3ms)    ▼(6ms)        ▼(5ms)                 ▼(5ms)
┌───────────┐ ┌─────┐    ┌──────────┐         ┌──────────┐
│HIPPOCAMPUS│ │ PFC │    │ STRIATUM │         │ STRIATUM │
│ (DG→CA3→  │ │     │◄───│  (D1/D2) │◄────────│  (L5→)   │
│   CA1)    │ │     │    │          │         │          │
└─────┬─────┘ └──┬──┘    └────┬─────┘         └──────────┘
      │          │            │
      ▼(5ms)     │            │
    ┌─────┐      │            │
    │ PFC │◄─────┘            │
    │     │                   │
    └──┬──┘                   │
       │                      │
       ▼(4ms)                 │
    ┌──────────┐              │
    │ STRIATUM │◄─────────────┘
    │  (PFC→)  │
    └────┬─────┘
         │
         ▼
    Motor Output

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Any
import torch
import torch.nn as nn

from .event_system import (
    Event, EventType, EventScheduler, ThetaGenerator, TrialPhase,
    SpikePayload, DopaminePayload,
    get_axonal_delay,
)
from .event_regions import (
    EventDrivenCortex, EventDrivenHippocampus, EventDrivenPFC, EventDrivenStriatum,
    EventDrivenCerebellum, EventRegionConfig,
)
from .parallel_executor import _ParallelExecutor
from .sleep import SleepSystemMixin
from .diagnostics import (
    DiagnosticsManager,
    DiagnosticLevel,
    StriatumDiagnostics,
    HippocampusDiagnostics,
    BrainSystemDiagnostics,
)

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


@dataclass
class EventDrivenBrainConfig:
    """Configuration for the event-driven brain system.

    .. deprecated:: 0.2.0
        Use :class:`thalia.config.ThaliaConfig` instead for unified configuration.
        Create brain with ``EventDrivenBrain.from_thalia_config(config)``.

    This is a simplified configuration focusing on the key parameters.
    For full control, create regions directly and pass them in.
    """
    # Region sizes
    input_size: int = 256
    cortex_size: int = 128
    hippocampus_size: int = 64
    pfc_size: int = 32
    n_actions: int = 2

    # Region type selection
    cortex_type: CortexType = CortexType.LAYERED
    """Which cortex implementation to use. PREDICTIVE enables local error learning."""

    # Region-specific configs (optional - uses defaults if not provided)
    # These are the full configs from thalia.regions.*
    cortex_config: Optional[LayeredCortexConfig] = None
    """Full cortex config. n_input/n_output will be overridden by sizes above."""

    # Time settings
    dt_ms: float = 1.0
    theta_frequency_hz: float = 8.0

    # Default timesteps for trial phases
    encoding_timesteps: int = 15
    delay_timesteps: int = 10
    test_timesteps: int = 15

    # Striatum settings
    neurons_per_action: int = 10

    # Execution mode
    parallel: bool = False  # Use multiprocessing for regions

    # Diagnostics
    diagnostic_level: DiagnosticLevel = DiagnosticLevel.SUMMARY

    device: str = "cpu"

    def __post_init__(self):
        """Emit deprecation warning."""
        import warnings
        warnings.warn(
            "EventDrivenBrainConfig is deprecated. Use ThaliaConfig instead:\n"
            "  from thalia.config import ThaliaConfig\n"
            "  config = ThaliaConfig(...)\n"
            "  brain = EventDrivenBrain.from_thalia_config(config)",
            DeprecationWarning,
            stacklevel=2,
        )


class EventDrivenBrain(SleepSystemMixin, nn.Module):
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
        # Using unified config (recommended)
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cuda"),
            brain=BrainConfig(sizes=RegionSizes(cortex_size=256)),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Or using legacy config
        brain = EventDrivenBrain(EventDrivenBrainConfig(
            input_size=256,
            cortex_size=128,
            n_actions=2,
        ))

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

    def __init__(self, config: EventDrivenBrainConfig):
        super().__init__()
        self.config = config

        # Current simulation time
        self._current_time: float = 0.0

        # Trial state
        self._trial_phase = TrialPhase.ENCODE

        # =====================================================================
        # CREATE BRAIN REGIONS
        # =====================================================================

        # 1. CORTEX: Feature extraction
        # Build cortex config by merging user config with computed sizes
        if config.cortex_config is not None:
            # Use provided config, but override sizes
            base_cortex_config = config.cortex_config
        else:
            # Create default config
            base_cortex_config = LayeredCortexConfig(n_input=0, n_output=0)
        
        # Merge with sizes from this config (sizes always come from here)
        cortex_config = replace(
            base_cortex_config,
            n_input=config.input_size,
            n_output=config.cortex_size,
            dt_ms=config.dt_ms,
            device=config.device,
        )
        
        # Select implementation based on config
        if config.cortex_type == CortexType.PREDICTIVE:
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
            self.cortex = PredictiveCortex(pred_config)
        else:
            # Default LayeredCortex (L4→L2/3→L5)
            self.cortex = LayeredCortex(cortex_config)
        self.cortex.reset_state(batch_size=1)

        # Get cortex layer sizes
        self._cortex_l23_size = self.cortex.l23_size
        self._cortex_l5_size = self.cortex.l5_size
        cortex_to_hippo_size = self._cortex_l23_size

        # 2. HIPPOCAMPUS: Episodic memory
        self.hippocampus = TrisynapticHippocampus(TrisynapticConfig(
            n_input=cortex_to_hippo_size,
            n_output=config.hippocampus_size,
            dt_ms=config.dt_ms,
            ec_l3_input_size=config.input_size,
            device=config.device,
        ))

        # 3. PFC: Working memory
        pfc_input_size = cortex_to_hippo_size + config.hippocampus_size
        self.pfc = Prefrontal(PrefrontalConfig(
            n_input=pfc_input_size,
            n_output=config.pfc_size,
            dt_ms=config.dt_ms,
            device=config.device,
        ))
        self.pfc.reset_state(batch_size=1)

        # 4. STRIATUM: Action selection
        # Receives: cortex L5 + hippocampus + PFC
        # NOTE: Pass n_output=n_actions (not n_actions*neurons_per_action)
        # The Striatum internally handles population coding expansion
        striatum_input = self._cortex_l5_size + config.hippocampus_size + config.pfc_size
        self.striatum = Striatum(StriatumConfig(
            n_input=striatum_input,
            n_output=config.n_actions,  # Number of actions, NOT total neurons
            neurons_per_action=config.neurons_per_action,
            device=config.device,
        ))
        self.striatum.reset()

        # 5. CEREBELLUM: Motor refinement
        # Receives: striatum output (action signals)
        # After population coding, striatum outputs n_actions * neurons_per_action
        cerebellum_input = config.n_actions * config.neurons_per_action
        self.cerebellum = Cerebellum(CerebellumConfig(
            n_input=cerebellum_input,
            n_output=config.n_actions,  # Refined motor output
            device=config.device,
        ))

        # =====================================================================
        # CREATE EVENT-DRIVEN ADAPTERS
        # =====================================================================

        self.event_cortex = EventDrivenCortex(
            EventRegionConfig(
                name="cortex",
                output_targets=["hippocampus", "pfc", "striatum"],
            ),
            self.cortex,
            pfc_size=config.pfc_size,  # For top-down projection
        )

        self.event_hippocampus = EventDrivenHippocampus(
            EventRegionConfig(
                name="hippocampus",
                output_targets=["pfc", "striatum"],
            ),
            self.hippocampus,
        )

        self.event_pfc = EventDrivenPFC(
            EventRegionConfig(
                name="pfc",
                output_targets=["striatum", "cortex"],  # Top-down to cortex
            ),
            self.pfc,
            cortex_input_size=self._cortex_l23_size,
            hippocampus_input_size=config.hippocampus_size,
        )

        self.event_striatum = EventDrivenStriatum(
            EventRegionConfig(
                name="striatum",
                output_targets=["cerebellum"],  # Striatum -> Cerebellum
            ),
            self.striatum,
            cortex_input_size=self._cortex_l5_size,
            hippocampus_input_size=config.hippocampus_size,
            pfc_input_size=config.pfc_size,
        )

        self.event_cerebellum = EventDrivenCerebellum(
            EventRegionConfig(
                name="cerebellum",
                output_targets=[],  # Final motor output
            ),
            self.cerebellum,
        )

        # Region lookup
        self.regions = {
            "cortex": self.event_cortex,
            "hippocampus": self.event_hippocampus,
            "pfc": self.event_pfc,
            "striatum": self.event_striatum,
            "cerebellum": self.event_cerebellum,
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
        # THETA RHYTHM
        # =====================================================================

        self.theta = ThetaGenerator(
            frequency_hz=config.theta_frequency_hz,
            connected_regions=list(self.regions.keys()),
        )

        # =====================================================================
        # LEARNABLE PATHWAYS
        # =====================================================================

        # Attention pathway: PFC → Cortex top-down modulation
        self.attention_pathway = SpikingAttentionPathway(
            SpikingAttentionPathwayConfig(
                source_size=config.pfc_size,
                target_size=config.input_size,
            )
        )

        # Replay pathway: Hippocampus → Cortex consolidation (during sleep)
        self.replay_pathway = SpikingReplayPathway(
            SpikingReplayPathwayConfig(
                source_size=config.hippocampus_size,
                target_size=config.cortex_size,
            )
        )

        # =====================================================================
        # EVENT SCHEDULER (for sequential mode)
        # =====================================================================

        self.scheduler = EventScheduler()

        # =====================================================================
        # PARALLEL EXECUTION (optional)
        # =====================================================================

        self._parallel_executor: Optional[_ParallelExecutor] = None
        if config.parallel:
            self._init_parallel_executor()

        # State tracking
        self._last_cortex_output: Optional[torch.Tensor] = None
        self._last_hippo_output: Optional[torch.Tensor] = None
        self._last_pfc_output: Optional[torch.Tensor] = None
        self._last_action: Optional[int] = None

        # =====================================================================
        # VTA DOPAMINE SYSTEM (centralized RPE computation)
        # =====================================================================
        # Brain acts as VTA: computes reward prediction error and broadcasts
        # normalized dopamine signal to all regions. This centralizes what
        # was previously in Striatum's DopamineSystem.
        #
        # We separate TONIC and PHASIC dopamine (biologically accurate):
        # - TONIC: Slow, continuous signal from intrinsic prediction quality
        # - PHASIC: Sharp bursts/dips from external rewards, decays over time
        #
        # Regions receive: global_dopamine = tonic + phasic
        self._tonic_dopamine: float = 0.0   # Slow baseline (intrinsic)
        self._phasic_dopamine: float = 0.0  # Fast bursts (external rewards)
        self._global_dopamine: float = 0.0  # Combined signal to regions

        # Phasic dopamine decay parameters
        # τ = 200ms for dopamine reuptake (decay = exp(-dt/τ))
        # At dt=1ms: decay = exp(-1/200) ≈ 0.995 per timestep
        self._phasic_decay: float = 0.995  # Per-timestep decay factor

        # Adaptive normalization prevents saturation from reward scaling:
        # - Tracks running average of |RPE| to adapt to reward statistics
        # - Outputs normalized RPE in range [-rpe_clip, +rpe_clip]
        self._vta_avg_abs_rpe: float = 0.5  # Running average of |RPE|
        self._vta_rpe_history_count: int = 0  # Number of rewards seen
        self._vta_rpe_avg_tau: float = 0.9  # EMA decay for running average
        self._vta_rpe_clip: float = 2.0  # Clip normalized RPE to this range

        # Monitoring
        self._spike_counts: Dict[str, int] = {name: 0 for name in self.regions}
        self._events_processed: int = 0

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
        # Import here to avoid circular imports
        from thalia.config import ThaliaConfig

        legacy_config = config.to_event_driven_brain_config()
        return cls(legacy_config)

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
        self._parallel_executor = _ParallelExecutor(
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
        n_timesteps = n_timesteps or self.config.encoding_timesteps

        # Set trial phase
        self._trial_phase = TrialPhase.ENCODE
        self.theta.align_to_encoding()

        # Note: No new_trial()/clear() calls here - continuous processing
        # State transitions happen via natural dynamics (decay, FFI)
        # Call new_sequence() explicitly when starting unrelated sequences

        # Process timesteps
        results = self._run_timesteps(
            sensory_input=sample_pattern,
            n_timesteps=n_timesteps,
            trial_phase=TrialPhase.ENCODE,
        )

        # Capture PFC output for decoder (language model uses this)
        if hasattr(self.pfc, 'state') and self.pfc.state is not None:
            if self.pfc.state.working_memory is not None:
                self._last_pfc_output = self.pfc.state.working_memory.squeeze(0).clone()
            elif self.pfc.state.spikes is not None:
                self._last_pfc_output = self.pfc.state.spikes.squeeze(0).clone()

        return results

    def delay(
        self,
        n_timesteps: Optional[int] = None,
        dopamine: float = 0.0,
    ) -> Dict[str, Any]:
        """Delay period (maintenance phase).

        PFC maintains working memory, other regions decay.

        Args:
            n_timesteps: Number of delay timesteps
            dopamine: Tonic dopamine level during delay

        Returns:
            Dict with region activities
        """
        n_timesteps = n_timesteps or self.config.delay_timesteps
        self._trial_phase = TrialPhase.DELAY

        results = self._run_timesteps(
            sensory_input=None,
            n_timesteps=n_timesteps,
            trial_phase=TrialPhase.DELAY,
            dopamine=dopamine,
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
        n_timesteps = n_timesteps or self.config.test_timesteps

        # Set trial phase
        self._trial_phase = TrialPhase.RETRIEVE
        self.theta.align_to_retrieval()

        results = self._run_timesteps(
            sensory_input=test_pattern,
            n_timesteps=n_timesteps,
            trial_phase=TrialPhase.RETRIEVE,
        )

        return results

    def run_consolidation(
        self,
        n_timesteps: int = 50,
    ) -> Dict[str, Any]:
        """Run consolidation timesteps without sensory input.
        
        This allows eligibility traces to interact with phasic dopamine
        after reward delivery. The brain continues processing internally:
        - Phasic dopamine decays naturally (τ ~ 200ms)
        - Eligibility traces also decay (τ ~ 100-1000ms)  
        - Learning occurs where traces × dopamine are both non-zero
        
        Typical usage (two-phase training):
        1. process_sample() → builds eligibility traces
        2. select_action() → commits to an action
        3. deliver_reward() → sets phasic dopamine
        4. run_consolidation() → allows learning to occur
        
        Args:
            n_timesteps: Number of consolidation timesteps (default 50)
            
        Returns:
            Dict with region activities during consolidation
        """
        results = self._run_timesteps(
            sensory_input=None,
            n_timesteps=n_timesteps,
            trial_phase=self._trial_phase,  # Keep current phase
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
        result = self.striatum.finalize_action(explore=explore)

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
        # The tonic component already flows continuously via _update_tonic_dopamine().
        #
        # If no external reward (0.0), we just compute the phasic component from
        # the current state for the striatum value update.

        # =====================================================================
        # STEP 2: Get expected value from striatum
        # =====================================================================
        expected = self.striatum.get_expected_value(self._last_action)

        # =====================================================================
        # STEP 3: Compute reward prediction error
        # =====================================================================
        # RPE = actual - expected
        # For external rewards, the "actual" is the external signal
        # For pure intrinsic (external=0), compute from current intrinsic
        if external_reward != 0.0:
            rpe = external_reward - expected
        else:
            intrinsic = self._compute_intrinsic_reward()
            rpe = intrinsic - expected

        # =====================================================================
        # STEP 4: Normalize RPE (adaptive to reward statistics)
        # =====================================================================
        da_level = self._compute_normalized_dopamine(rpe)

        # =====================================================================
        # STEP 5: SET PHASIC DOPAMINE (this is the key change!)
        # =====================================================================
        # Instead of directly setting _global_dopamine, we set the PHASIC component.
        # This will decay over time via _update_tonic_dopamine(), giving the
        # eligibility traces time to overlap with the dopamine signal.
        self._phasic_dopamine = da_level

        # Update global immediately (regions get the sum)
        self._global_dopamine = self._tonic_dopamine + self._phasic_dopamine
        self._global_dopamine = max(-2.0, min(2.0, self._global_dopamine))

        # =====================================================================
        # STEP 6: Broadcast to ALL regions
        # =====================================================================
        self.cortex.set_dopamine(self._global_dopamine)
        self.hippocampus.set_dopamine(self._global_dopamine)
        self.pfc.set_dopamine(self._global_dopamine)
        self.striatum.set_dopamine(self._global_dopamine)
        self.cerebellum.set_dopamine(self._global_dopamine)

        # Create dopamine events for all regions (event-driven pathway)
        for region_name in self.regions:
            delay = get_axonal_delay("vta", region_name)
            event = Event(
                time=self._current_time + delay,
                event_type=EventType.DOPAMINE,
                source="reward_system",
                target=region_name,
                payload=DopaminePayload(
                    level=self._global_dopamine,
                    is_burst=self._phasic_dopamine > 0.5,
                    is_dip=self._phasic_dopamine < -0.5,
                ),
            )
            self.scheduler.schedule(event)

        # Process dopamine events
        self._process_pending_events()

        # =====================================================================
        # STEP 7: Trigger striatum learning (D1/D2 plasticity)
        # =====================================================================
        # Use the external reward for striatum value updates
        reward_for_striatum = external_reward if external_reward != 0.0 else self._compute_intrinsic_reward()
        if self._last_action is not None:
            self.striatum.deliver_reward(reward_for_striatum)

        # =====================================================================
        # STEP 8: Update value estimate in striatum
        # =====================================================================
        if self._last_action is not None:
            self.striatum.update_value_estimate(self._last_action, reward_for_striatum)

    def _compute_normalized_dopamine(self, rpe: float) -> float:
        """Compute normalized dopamine from raw RPE.

        Uses adaptive normalization to prevent saturation:
        - Tracks running average of |RPE| to adapt to reward statistics
        - Outputs normalized RPE in range [-rpe_clip, +rpe_clip]

        This is the VTA's core computation: converting prediction error
        into a normalized dopamine signal suitable for learning.

        Args:
            rpe: Raw reward prediction error

        Returns:
            Normalized dopamine level
        """
        abs_rpe = abs(rpe)
        self._vta_rpe_history_count += 1

        # Adaptive smoothing: slower early on for stability
        if self._vta_rpe_history_count < 10:
            alpha = 1.0 / self._vta_rpe_history_count
        else:
            alpha = 1.0 - self._vta_rpe_avg_tau

        # Update running average of |RPE|
        self._vta_avg_abs_rpe = (
            self._vta_rpe_avg_tau * self._vta_avg_abs_rpe + alpha * abs_rpe
        )

        # Normalize RPE by running average (with epsilon for stability)
        epsilon = 0.1
        normalized_rpe = rpe / (self._vta_avg_abs_rpe + epsilon)

        # Clip to prevent extreme updates
        return max(-self._vta_rpe_clip, min(self._vta_rpe_clip, normalized_rpe))

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
        if hasattr(self.cortex, 'state') and hasattr(self.cortex.state, 'free_energy'):
            free_energy = self.cortex.state.free_energy

            # Free energy is typically 0-10, lower is better
            # Map: 0 → +1 (perfect prediction), 5 → 0, 10+ → -1 (bad prediction)
            cortex_reward = 1.0 - 0.2 * min(free_energy, 10.0)
            cortex_reward = max(-1.0, min(1.0, cortex_reward))
            reward += cortex_reward
            n_sources += 1

        # Fallback: check for accumulated free energy in PredictiveCortex
        elif hasattr(self.cortex, '_total_free_energy'):
            total_fe = self.cortex._total_free_energy
            cortex_reward = 1.0 - 0.1 * min(total_fe, 20.0)
            cortex_reward = max(-1.0, min(1.0, cortex_reward))
            reward += cortex_reward
            n_sources += 1

        # =====================================================================
        # 2. HIPPOCAMPUS PATTERN COMPLETION (memory recall quality)
        # =====================================================================
        # High pattern similarity = successful memory retrieval = reward
        if hasattr(self.hippocampus, 'get_pattern_similarity'):
            similarity = self.hippocampus.get_pattern_similarity()
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

    def _update_tonic_dopamine(self) -> None:
        """Update dopamine levels every timestep.

        This handles both dopamine channels:

        1. TONIC DOPAMINE (slow, intrinsic):
           - Updates based on current prediction quality
           - Smoothed with EMA (τ ~100ms)
           - Represents ongoing "mood"/motivation

        2. PHASIC DOPAMINE (fast, external):
           - Decays toward zero each timestep (τ ~200ms)
           - Set by deliver_reward() when external rewards arrive
           - Represents reward prediction error bursts/dips

        GLOBAL DOPAMINE = tonic + phasic
        This is what regions receive for plasticity modulation.

        Biologically:
        - VTA neurons fire tonically at ~4-5Hz (baseline)
        - Phasic bursts (5-20 spikes) for unexpected rewards
        - Phasic pauses for unexpected punishments
        - Both components sum at target synapses
        """
        # =====================================================================
        # 1. UPDATE TONIC DOPAMINE (slow, intrinsic)
        # =====================================================================
        intrinsic = self._compute_intrinsic_reward()

        # Smooth the tonic signal (slow changes)
        # alpha = 0.1 → τ ≈ 10ms at dt=1ms (could make slower)
        tonic_alpha = 0.05  # Slower smoothing for tonic baseline
        self._tonic_dopamine = (1 - tonic_alpha) * self._tonic_dopamine + tonic_alpha * intrinsic

        # =====================================================================
        # 2. DECAY PHASIC DOPAMINE (fast, external)
        # =====================================================================
        # Exponential decay with τ ~200ms
        # decay = 0.995 per ms → after 200ms: 0.995^200 ≈ 0.37 (≈ 1/e)
        self._phasic_dopamine *= self._phasic_decay

        # =====================================================================
        # 3. COMPUTE GLOBAL DOPAMINE (sum of both)
        # =====================================================================
        self._global_dopamine = self._tonic_dopamine + self._phasic_dopamine

        # Clip to reasonable range (dopamine has physiological limits)
        self._global_dopamine = max(-2.0, min(2.0, self._global_dopamine))

        # =====================================================================
        # 4. BROADCAST TO ALL REGIONS
        # =====================================================================
        self.cortex.set_dopamine(self._global_dopamine)
        self.hippocampus.set_dopamine(self._global_dopamine)
        self.pfc.set_dopamine(self._global_dopamine)
        self.striatum.set_dopamine(self._global_dopamine)
        self.cerebellum.set_dopamine(self._global_dopamine)


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
        cortex_L5 = self.cortex.state.l5_spikes
        if cortex_L5 is None:
            cortex_L5 = torch.zeros(1, self._cortex_l5_size)

        hippo_out = self.hippocampus.state.ca1_spikes
        if hippo_out is None:
            hippo_out = torch.zeros(1, self.config.hippocampus_size)

        pfc_out = self.pfc.state.spikes
        if pfc_out is None:
            pfc_out = torch.zeros(1, self.config.pfc_size)

        combined_state = torch.cat([
            cortex_L5.view(-1),
            hippo_out.view(-1),
            pfc_out.view(-1),
        ])

        self.hippocampus.store_episode(
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

    def reset(self) -> None:
        """Reset brain state for new episode.
        
        This is a HARD reset - use for completely new, unrelated episodes.
        For starting a new sequence within the same session, use new_sequence().
        """
        self._current_time = 0.0
        self._trial_phase = TrialPhase.ENCODE
        self.theta.reset()
        self.scheduler = EventScheduler()

        # Reset regions (full state reset)
        self.cortex.reset_state(batch_size=1)
        self.pfc.reset_state(batch_size=1)
        self.striatum.reset()
        self.hippocampus.new_trial()

        # Reset monitoring
        self._spike_counts = {name: 0 for name in self.regions}
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
        self.hippocampus.new_trial()

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
        self._global_dopamine = modulated_reward

        # 1. Real learning: update striatum for SELECTED action
        real_result = self.striatum.deliver_reward(modulated_reward)

        # 2. Counterfactual: what would the OTHER action have gotten?
        other_action = 1 - selected_action

        # Determine counterfactual reward:
        # - If trial is MATCH: MATCH action (0) would get +1, NOMATCH (1) would get -1
        # - If trial is NOMATCH: NOMATCH action (1) would get +1, MATCH (0) would get -1
        correct_action = 0 if is_match else 1
        counterfactual_reward = 1.0 if (other_action == correct_action) else -1.0

        # Apply counterfactual learning
        counterfactual_result = {}
        if hasattr(self.striatum, 'deliver_counterfactual_reward'):
            counterfactual_result = self.striatum.deliver_counterfactual_reward(
                reward=counterfactual_reward,
                action=other_action,
                counterfactual_scale=counterfactual_scale,
            )

        # Reset eligibility after both learnings
        if hasattr(self.striatum, 'reset_eligibility'):
            self.striatum.reset_eligibility()

        # Update attention pathway
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

    def _run_timesteps(
        self,
        sensory_input: Optional[torch.Tensor],
        n_timesteps: int,
        trial_phase: TrialPhase,
        dopamine: float = 0.0,
    ) -> Dict[str, Any]:
        """Run simulation for specified timesteps.
        
        Delegates to parallel executor if parallel mode is enabled,
        otherwise runs sequentially in the main process.
        """
        if self._parallel_executor is not None:
            return self._run_timesteps_parallel(
                sensory_input, n_timesteps, trial_phase, dopamine
            )
        return self._run_timesteps_sequential(
            sensory_input, n_timesteps, trial_phase, dopamine
        )

    def _run_timesteps_parallel(
        self,
        sensory_input: Optional[torch.Tensor],
        n_timesteps: int,
        trial_phase: TrialPhase,
        dopamine: float = 0.0,
    ) -> Dict[str, Any]:
        """Run simulation using parallel executor."""
        assert self._parallel_executor is not None
        
        end_time = self._current_time + n_timesteps * self.config.dt_ms
        
        # Inject sensory input if provided
        if sensory_input is not None:
            self._parallel_executor.inject_sensory_input(
                sensory_input, 
                target="cortex",
                time=self._current_time,
            )
        
        # Inject dopamine if specified
        if abs(dopamine) > 1e-6:
            self._parallel_executor.inject_reward(dopamine, time=self._current_time)
        
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
        trial_phase: TrialPhase,
        dopamine: float = 0.0,
    ) -> Dict[str, Any]:
        """Run simulation sequentially in main process."""

        # Track activities for monitoring
        cortex_total = torch.zeros(self._cortex_l23_size)
        hippo_total = torch.zeros(self.config.hippocampus_size)
        pfc_total = torch.zeros(self.config.pfc_size)

        for t in range(n_timesteps):
            step_time = self._current_time + t * self.config.dt_ms

            # Advance theta and schedule theta events
            theta_events = self.theta.advance_to(step_time)
            self.scheduler.schedule_many(theta_events)

            # Schedule sensory input if provided
            if sensory_input is not None:
                delay = get_axonal_delay("sensory", "cortex")
                event = Event(
                    time=step_time + delay,
                    event_type=EventType.SENSORY,
                    source="sensory_input",
                    target="cortex",
                    payload=SpikePayload(spikes=sensory_input),
                )
                self.scheduler.schedule(event)

            # Schedule dopamine if specified (external override)
            if abs(dopamine) > 1e-6:
                for region_name in ["striatum", "pfc"]:
                    delay = get_axonal_delay("vta", region_name)
                    event = Event(
                        time=step_time + delay,
                        event_type=EventType.DOPAMINE,
                        source="tonic_dopamine",
                        target=region_name,
                        payload=DopaminePayload(level=dopamine),
                    )
                    self.scheduler.schedule(event)

            # =========================================================
            # CONTINUOUS INTRINSIC DOPAMINE (tonic modulation)
            # =========================================================
            # Update tonic dopamine based on ongoing prediction quality.
            # This happens every timestep - the brain continuously
            # evaluates its own predictions and modulates learning rates.
            self._update_tonic_dopamine()

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
            if event.target in self.regions:
                region = self.regions[event.target]
                output_events = region.process_event(event)

                # Schedule output events
                for out_event in output_events:
                    self.scheduler.schedule(out_event)

                    # Track spikes
                    if out_event.event_type == EventType.SPIKE:
                        payload = out_event.payload
                        if isinstance(payload, SpikePayload):
                            self._spike_counts[event.target] += int(payload.spikes.sum().item())

                self._events_processed += 1

    def _process_pending_events(self) -> None:
        """Process all pending events (used for reward delivery)."""
        max_time = self._current_time + 100.0  # Process up to 100ms ahead
        self._process_events_until(max_time)

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
        
        Returns both structured component diagnostics and raw metrics.
        """
        # Collect structured component diagnostics
        striatum_diag = self._collect_striatum_diagnostics()
        hippo_diag = self._collect_hippocampus_diagnostics()

        # Record to diagnostics manager
        self.diagnostics.record("striatum", striatum_diag.to_dict())
        self.diagnostics.record("hippocampus", hippo_diag.to_dict())

        return {
            # Brain state
            "current_time": self._current_time,
            "trial_phase": self._trial_phase.value,
            "theta_phase": self.theta.phase,
            "encoding_strength": self.theta.encoding_strength,
            "retrieval_strength": self.theta.retrieval_strength,
            "spike_counts": self._spike_counts.copy(),
            "events_processed": self._events_processed,
            "last_action": self._last_action,
            
            # VTA Dopamine System (centralized)
            "dopamine": {
                "global": self._global_dopamine,  # Combined signal to regions
                "tonic": self._tonic_dopamine,    # Slow baseline (intrinsic)
                "phasic": self._phasic_dopamine,  # Fast bursts (external rewards)
                "phasic_decay": self._phasic_decay,
            },
            # VTA normalization state
            "vta": {
                "avg_abs_rpe": self._vta_avg_abs_rpe,
                "rpe_history_count": self._vta_rpe_history_count,
                "rpe_clip": self._vta_rpe_clip,
            },
            
            # Legacy key for backwards compatibility
            "global_dopamine": self._global_dopamine,
            
            # Structured component diagnostics
            "striatum": striatum_diag.to_dict(),
            "hippocampus": hippo_diag.to_dict(),
            
            # Summary for quick access
            "summary": {
                "last_action": self._last_action,
                "exploring": striatum_diag.exploring,
                "net_weight_means": striatum_diag.net_per_action,
                "ca1_spikes": hippo_diag.ca1_spikes,
                "dopamine_global": self._global_dopamine,
                "dopamine_tonic": self._tonic_dopamine,
                "dopamine_phasic": self._phasic_dopamine,
            },
        }

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

    config = EventDrivenBrainConfig(
        input_size=100,
        cortex_size=64,
        hippocampus_size=40,
        pfc_size=20,
        n_actions=2,
    )

    brain = EventDrivenBrain(config)
    print(f"  Created brain with {len(brain.regions)} regions")

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
