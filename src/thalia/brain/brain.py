"""DynamicBrain: Flexible, graph-based brain architecture with modular regions."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Type, TypeVar, cast

import torch
import torch.nn as nn

from thalia.typing import (
    BrainOutput,
    NeuromodulatorInput,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import SynapseIdModuleDict, generate_reward_spikes, validate_spike_tensor

from .axonal_tract import AxonalTract
from .configs import BrainConfig, NeuralRegionConfig
from .neuromodulator_hub import NeuromodulatorHub

from .regions import NeuralRegion


RegionT = TypeVar('RegionT', bound=NeuralRegion[NeuralRegionConfig])


# =============================================================================
# TYPED-KEY CONTAINER FOR AXONAL TRACTS
# =============================================================================

class AxonalTractDict(SynapseIdModuleDict):
    """Type-narrowed :class:`~thalia.utils.SynapseIdModuleDict` for :class:`AxonalTract` values.

    Inherits all SynapseId-keyed bookkeeping from the shared mixin.  The only
    purpose of this subclass is to narrow the value type from ``nn.Module`` to
    ``AxonalTract`` so callers get correct type inference without casts.
    """

    def __getitem__(self, key: SynapseId) -> AxonalTract:  # type: ignore[override]
        return super().__getitem__(key)  # type: ignore[return-value]

    def items(self) -> Iterator[tuple[SynapseId, AxonalTract]]:  # type: ignore[override]
        return super().items()  # type: ignore[return-value]

    def values(self) -> Iterator[AxonalTract]:  # type: ignore[override]
        return super().values()  # type: ignore[return-value]


class DynamicBrain(nn.Module):
    """Dynamic brain constructed from graph."""

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Device where tensors are located."""
        return torch.device(self.config.device)

    @property
    def dt_ms(self) -> float:
        """Timestep duration in milliseconds."""
        return self.config.dt_ms

    @property
    def current_time(self) -> float:
        """Get current simulation time in milliseconds."""
        return self._current_time

    def get_region_by_name(self, region_name: RegionName) -> Optional[NeuralRegion[NeuralRegionConfig]]:
        """Get region by name, or None if not present."""
        return self.regions[region_name] if region_name in self.regions else None

    def get_region_by_name_and_type(self, region_name: RegionName, region_type: Type[RegionT]) -> Optional[RegionT]:
        """Get region by name and type, or None if not present."""
        region = self.get_region_by_name(region_name)
        if region is not None and isinstance(region, region_type):
            return cast(RegionT, region)
        return None

    def get_first_region_of_type(self, region_type: Type[RegionT]) -> Optional[RegionT]:
        """Get the first region of a given type, or None if not present."""
        for region in self.regions.values():
            if isinstance(region, region_type):
                return cast(RegionT, region)
        return None

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: BrainConfig,
        regions: Dict[RegionName, NeuralRegion[NeuralRegionConfig]],
        axonal_tracts: Dict[SynapseId, AxonalTract],
    ):
        """Initialize DynamicBrain from graph.

        Args:
            config: Brain configuration (device, dt_ms, etc.)
            regions: Dict mapping region names to instances
            axonal_tracts: Dict mapping SynapseId to AxonalTract instances
        """
        # =================================================================
        # INITIALIZE BRAIN STATE
        # =================================================================
        super().__init__()

        self.config = config

        # Store regions as nn.ModuleDict for proper parameter tracking
        self.regions: Dict[RegionName, NeuralRegion[NeuralRegionConfig]] = nn.ModuleDict(regions)

        # Store axonal tracts in a typed nn.ModuleDict wrapper so .to(), .state_dict(),
        # and .parameters() track all delay-buffer state correctly.
        self.axonal_tracts: AxonalTractDict = AxonalTractDict()
        for key, tract in axonal_tracts.items():
            self.axonal_tracts[key] = tract

        # Current simulation time
        self._current_time: float = 0.0

        # Last timestep's brain output used to build the neuromodulator broadcast
        # signal (one-step lag is unavoidable with clock-driven simulation).
        # The per-region NeuromodulatorReceptor handles rise/decay dynamics.
        self._last_brain_output: Optional[BrainOutput] = None

        # NeuromodulatorHub: replaces the inline neuromodulator-signal-building loop
        # in forward().  Holds a live reference to self.regions so that regions
        # added after construction are automatically tracked.
        self.neuromodulator_hub = NeuromodulatorHub(self.regions)

        # =================================================================
        # INITIALIZE TEMPORAL PARAMETERS
        # =================================================================
        # Broadcast dt_ms to all regions for initial setup
        # Regions compute decay factors, phase increments, etc.
        self.set_timestep(self.dt_ms)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: Optional[SynapticInput] = None) -> BrainOutput:
        """Run one timestep of the brain."""
        # === CLOCK-DRIVEN EXECUTION (ADR-003) ===
        # All regions execute every timestep.
        # Axonal delays are handled by axonal tract CircularDelayBuffers.
        # This ensures continuous dynamics: membrane decay, recurrent connections,
        # and short-term plasticity all evolve every timestep.

        # =====================================================================
        # SPIKE PROPAGATION
        # =====================================================================
        # At timestep T:
        # 1. Read delayed outputs from buffers (spikes sent at T-delay)
        # 2. Inject external input spikes (e.g., sensory)
        # 3. Execute regions → produce outputs at T
        # 4. Write outputs to buffers → will be read at T+delay
        #
        # At T=0: buffers are empty, so delayed_outputs are zeros
        # At T>=delay: buffers have data, delayed_outputs have actual spikes
        # =====================================================================

        # STEP 1: Read delayed outputs from all axonal tracts (spikes from T-delay)
        region_inputs: Dict[RegionName, SynapticInput] = {}

        for axonal_tract in self.axonal_tracts.values():
            # read_delayed_outputs() returns a flat SynapticInput dict keyed by
            # the pre-built SynapseId, which already encodes target region,
            # target population, and polarity — no reconstruction needed.
            for synapse_id, delayed_source_spikes in axonal_tract.read_delayed_outputs().items():
                target_region = synapse_id.target_region

                if target_region not in region_inputs:
                    region_inputs[target_region] = {}

                if synapse_id in region_inputs[target_region]:
                    raise ValueError(
                        f"Duplicate synapse_id {synapse_id} from axonal tracts. "
                        f"Check for overlapping source populations or routing errors."
                    )

                region_inputs[target_region][synapse_id] = delayed_source_spikes

        # STEP 2: Inject external input spikes (e.g., sensory) - these have FULL routing keys already
        # External inputs are added on top of delayed spikes (logical OR) since both contribute to activation
        if synaptic_inputs is not None:
            for synapse_id, synapse_input in synaptic_inputs.items():
                validate_spike_tensor(synapse_input, tensor_name=str(synapse_id))

                if synapse_id.target_region not in region_inputs:
                    region_inputs[synapse_id.target_region] = {}

                # Logical OR if key already exists
                # (e.g., external sensory input + delayed thalamic input to same target)
                if synapse_id in region_inputs[synapse_id.target_region]:
                    region_inputs[synapse_id.target_region][synapse_id] = region_inputs[synapse_id.target_region][synapse_id] | synapse_input
                else:
                    region_inputs[synapse_id.target_region][synapse_id] = synapse_input

        # STEP 3: Collect neuromodulator signals from the previous timestep's output.
        # Each neuromodulator region declares outputs via a ``neuromodulator_outputs``
        # ClassVar (e.g. VTA→'da', LC→'ne', NB→'ach').
        # Raw spike tensors are broadcast to all regions; the per-region
        # NeuromodulatorReceptor handles rise/decay dynamics internally.
        # One-timestep lag is unavoidable with clock-driven simulation (ADR-003).
        neuromodulator_signals: NeuromodulatorInput = self.neuromodulator_hub.build(self._last_brain_output)

        # STEP 4: Execute all regions with synaptic inputs AND neuromodulator broadcast → produce outputs at T
        brain_output: BrainOutput = {}
        for region_name, region in self.regions.items():
            synaptic_inputs_for_region: SynapticInput = region_inputs.get(region_name, {})
            region_output: RegionOutput = region.forward(synaptic_inputs_for_region, neuromodulator_signals)
            brain_output[region_name] = region_output

        # STEP 5: Write current outputs to axonal tracts and advance buffers (available at T+delay)
        for axonal_tract in self.axonal_tracts.values():
            # Write ALL source regions' current outputs to buffer (for T+delay reads)
            axonal_tract.write_and_advance(brain_output)

        # Store this timestep's output for the neuromodulator broadcast on the next step.
        self._last_brain_output = brain_output

        # Advance simulation time
        self._current_time += self.dt_ms

        return brain_output

    def consolidate(self, duration_ms: float) -> Dict[str, Any]:
        """Trigger memory consolidation via spontaneous replay.

        No explicit coordination - just set acetylcholine low and run brain forward.
        Hippocampus spontaneously replays high-priority patterns via sharp-wave ripples.

        Biological mechanism:
        1. Lower acetylcholine (0.1) → sleep/rest mode
        2. Hippocampus CA3 spontaneously reactivates stored patterns
        3. Replay probability ∝ synaptic tag strength (Frey-Morris tagging)
        4. Ripples occur at ~1-3 Hz during low ACh
        5. Restore acetylcholine (0.7) → awake/encoding mode

        Args:
            duration_ms: Consolidation duration in milliseconds (default 1000ms = 1 second)

        Returns:
            Dict with consolidation statistics:
                - ripples: Number of sharp-wave ripples detected
                - duration_ms: Total consolidation duration
                - ripple_rate_hz: Ripples per second

        Raises:
            ValueError: If hippocampus not present in brain
        """
        # Check for hippocampus by name (avoids importing region-specific types here)
        hippocampus = self.get_region_by_name("hippocampus")
        if hippocampus is None:
            raise ValueError(
                "Hippocampus required for consolidation. "
                "Brain must include a region named 'hippocampus'."
            )

        # During consolidation there is no sensory input, so NucleusBasalis is
        # silent and ACh concentration decays to near-zero naturally.
        # Sharp-wave ripples emerge from CA3 attractor dynamics (low-ACh disinhibits
        # recurrence; GABA_B-terminated bursts produce the inter-ripple interval).
        # hippocampus.ripple_detected is set True whenever >5% of CA3 fires synchronously.
        # No region-specific enter/exit API is required.
        n_timesteps = int(duration_ms / self.dt_ms)
        ripple_count = 0

        for _ in range(n_timesteps):
            self.forward(None)  # No sensory input during sleep
            if hippocampus.ripple_detected:
                ripple_count += 1

        # Compute ripple rate
        ripple_rate_hz = ripple_count / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

        return {
            "ripples": ripple_count,
            "duration_ms": duration_ms,
            "ripple_rate_hz": ripple_rate_hz,
        }

    def deliver_reward(self, external_reward: Optional[float] = None) -> None:
        """Deliver reward signal via RewardEncoder region for VTA processing.

        **Spiking Dopamine Architecture**: Reward is encoded as population-coded
        spikes by :func:`~thalia.utils.generate_reward_spikes` and injected
        directly into VTA via :meth:`forward`.  VTA DA spikes are then broadcast
        to all regions with DA receptors (Striatum, PFC, Hippocampus, CorticalColumn, etc.)
        where they are converted to concentration dynamics via
        NeuromodulatorReceptor.

        Args:
            external_reward: Task-based reward value (typically -1 to +1),
                           or None for pure intrinsic reward

        Raises:
            ValueError: If vta or striatum not found

        Note:
            Actual learning happens continuously in region forward() passes using
            the spiking dopamine signal. This method only delivers reward to VTA
            - the rest happens automatically through DA receptors.
        """
        striatum = self.get_region_by_name("striatum")
        if striatum is None or not hasattr(striatum, 'update_performance'):
            raise ValueError(
                "Striatum with update_performance() method not found. Cannot deliver reward. "
                "Brain must include a region named 'striatum' with an update_performance() method for RL tasks."
            )

        vta = self.get_region_by_name("vta")
        if vta is None or not hasattr(vta, 'neuromodulator_outputs') or not any('da' in k for k in vta.neuromodulator_outputs.keys()):
            raise ValueError(
                "VTA with DA output not found. Cannot deliver reward. "
                "Brain must include a region named 'vta' whose neuromodulator_outputs "
                "contains at least one channel key that includes 'da' (e.g. 'da_mesolimbic')."
            )

        # Use external reward directly.
        # TODO (Phase 2): Replace with biologically-plausible intrinsic reward from LHb/VTA
        # once the habenula RPE circuit is implemented.  For now the training loop should
        # always supply an explicit external_reward.
        total_reward = external_reward if external_reward is not None else 0.0

        # n_reward_neurons: read from VTA's weight matrix so it always matches the
        # exact n_input registered by add_external_input_source() in BrainBuilder.
        reward_synapse = SynapseId.external_reward_to_vta_da(vta.region_name)
        n_reward_neurons: int = vta.get_synaptic_weights(reward_synapse).shape[1]
        spikes = generate_reward_spikes(
            reward=total_reward,
            n_neurons=n_reward_neurons,
            device=self.device,
        )
        self.forward(synaptic_inputs={reward_synapse: spikes})

        # Update exploration statistics based on reward.
        # Striatum applies continuous learning automatically in forward() using
        # spiking DA; this call only updates the per-trial performance history
        # used to adapt exploration rate (tonic dopamine).  It is intentionally
        # NOT routed through forward() because it must run exactly once per
        # trial, whereas forward() can run many times per trial.
        striatum.update_performance(total_reward)

    # =========================================================================
    # ADAPTIVE TIMESTEP AND OSCILLATOR COORDINATION
    # =========================================================================

    def set_timestep(self, new_dt_ms: float) -> None:
        """Change simulation timestep adaptively during training.

        Updates dt_ms and propagates temporal parameter updates to:
        - All regions (neurons, STP, learning strategies)
        - All axonal tracts (axonal delay buffers)

        Args:
            new_dt_ms: New timestep in milliseconds (must be positive)

        Raises:
            ValueError: If new_dt_ms <= 0
        """
        if new_dt_ms <= 0:
            raise ValueError(f"dt_ms must be positive, got {new_dt_ms}")

        # Update brain dt
        self.config.dt_ms = new_dt_ms

        # Update all regions
        for region in self.regions.values():
            region.update_temporal_parameters(new_dt_ms)

        # Update all axonal tracts
        for axonal_tract in self.axonal_tracts.values():
            axonal_tract.update_temporal_parameters(new_dt_ms)
