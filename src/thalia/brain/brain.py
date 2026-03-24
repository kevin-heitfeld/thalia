"""Brain: Flexible, graph-based brain architecture with modular regions."""

from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, Iterator, Optional, Type, TypeVar, Union, cast

import torch
import torch.nn as nn

from thalia.global_config import GlobalConfig
from thalia.typing import (
    BrainOutput,
    NeuromodulatorChannel,
    NeuromodulatorInput,
    PopulationKey,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import SynapseIdModuleDict, generate_reward_spikes, validate_spike_tensor

from .axonal_tract import AxonalTract
from .biophysics_registry import BiophysicsRegistry
from .configs import BrainConfig, NeuralRegionConfig
from .neuromodulator_hub import NeuromodulatorHub
from .neuron_index_registry import NeuronIndexRegistry
from .neurons.conductance_lif_batch import ConductanceLIFBatch
from .neurons.conductance_lif_neuron import ConductanceLIF
from .sparse_synaptic_matrix import GlobalSparseMatrix
from .synapses.stp_batch import STPBatch

from .regions import NeuralRegion


RegionT = TypeVar('RegionT', bound=NeuralRegion[NeuralRegionConfig])


# =============================================================================
# REGION THREAD POOL FOR PARALLEL EXECUTION
# =============================================================================


class _RegionThreadPool:
    """Lazy-initialised reusable thread pool for parallel region execution."""

    def __init__(self) -> None:
        self._pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._max_workers: int = 0

    def get(self, max_workers: int) -> concurrent.futures.ThreadPoolExecutor:
        if self._pool is None or self._max_workers != max_workers:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            self._max_workers = max_workers
        return self._pool


_region_thread_pool = _RegionThreadPool()


def _execute_region(
    region: NeuralRegion[NeuralRegionConfig],
    synaptic_inputs_for_region: SynapticInput,
    neuromodulator_signals: NeuromodulatorInput,
) -> RegionOutput:
    """Execute a single region's forward pass (thread-safe)."""
    return region.forward(synaptic_inputs_for_region, neuromodulator_signals)


# =============================================================================
# TYPED-KEY CONTAINER FOR AXONAL TRACTS
# =============================================================================

class AxonalTractDict(SynapseIdModuleDict):
    """Type-narrowed :class:`~thalia.utils.SynapseIdModuleDict` for :class:`AxonalTract` values.

    Inherits all SynapseId-keyed bookkeeping from the shared mixin.  The only
    purpose of this subclass is to narrow the value type from ``nn.Module`` to
    ``AxonalTract`` so callers get correct type inference without casts.
    """

    def __getitem__(self, key: SynapseId) -> AxonalTract:
        return super().__getitem__(key)  # type: ignore[return-value]

    def items(self) -> Iterator[tuple[SynapseId, AxonalTract]]:
        return super().items()  # type: ignore[return-value]

    def values(self) -> Iterator[AxonalTract]:
        return super().values()  # type: ignore[return-value]


class Brain(nn.Module):
    """Brain constructed from graph."""

    # =========================================================================
    # PROPERTIES
    # =========================================================================

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
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize Brain from graph.

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
        self.device = torch.device(device)

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

        # Pending CA1 mismatch level for next-step VTA novelty injection.
        # Hippocampus and VTA execute in the same forward() step, so the
        # mismatch signal from step T is stored here and injected at step T+1.
        # One-step causal delay maps to ~1 ms axonal conduction (biologically correct).
        self._pending_novelty_level: float = 0.0

        # Reward spikes queued by deliver_reward() for injection into the *next*
        # forward() call.
        self._pending_reward_inputs: Optional[SynapticInput] = None

        # =================================================================
        # INITIALIZE TEMPORAL PARAMETERS
        # =================================================================
        # Broadcast dt_ms to all regions for initial setup
        # Regions compute decay factors, phase increments, etc.
        self.set_timestep(self.dt_ms)

        # =================================================================
        # BATCHED STP
        # =================================================================
        # Collect all STP modules across all regions into one batched kernel.
        # Must be created after set_timestep() so decay factors are computed.
        self._stp_batch = self._create_stp_batch()

        # =================================================================
        # BATCHED NEURON STATE
        # =================================================================
        # Collect all eligible ConductanceLIF populations into one global batch.
        # After this call, eligible neurons' forward() writes inputs to batch
        # buffers instead of computing individually.
        self._neuron_batch = self._create_neuron_batch()

        # =================================================================
        # GLOBAL NEURON INDEX
        # =================================================================
        # Brain-wide registry mapping every (region, population) to a
        # contiguous global index range.  Unifies the index scheme used for
        # Philox RNG seeding with the sparse synaptic matrix row/column
        # spaces introduced in Step 5.
        self._neuron_index = NeuronIndexRegistry(
            self.regions, self._neuron_batch, device=self.device,
        )

        # =================================================================
        # GLOBAL SPARSE SYNAPTIC MATRIX (Step 5)
        # =================================================================
        # Four CSR sparse matrices (one per receptor type) replacing all
        # inter-region dense matmuls for eligible targets (ConductanceLIF
        # + subclasses).  TwoCompartmentLIF targets stay on the per-region
        # batched dendrite path.  Must be created before
        # build_batched_dendrite_weights so we can exclude sparse connections.
        self._sparse_matrix = GlobalSparseMatrix(
            self.regions, self._neuron_index, self._neuron_batch, self.device,
        )
        # Give each region a reference to the sparse matrix for sparse learning.
        for region in self.regions.values():
            region._sparse_matrix = self._sparse_matrix

        # =================================================================
        # BATCHED SYNAPTIC INTEGRATION
        # =================================================================
        # Precompute concatenated weight matrices per (target_pop, receptor_type)
        # for connections NOT in the global sparse matrix (intra-region
        # connections and TwoCompartmentLIF targets).
        _sparse_sids = frozenset(self._sparse_matrix.connections.keys())
        for region in self.regions.values():
            region.build_batched_dendrite_weights(exclude_synapse_ids=_sparse_sids)

        # =================================================================
        # BIOPHYSICS REGISTRY (read-only query layer)
        # =================================================================
        self.biophysics = BiophysicsRegistry.from_brain(self)

    def _create_stp_batch(self) -> STPBatch:
        """Collect all STP modules across regions into one STPBatch.

        Connections flagged as ``_manually_stepped_stp`` on their region are
        excluded — those are stepped by the region itself with multi-step
        delayed spikes and must not be double-updated.
        """
        entries: list[tuple[SynapseId, object]] = []
        for region in self.regions.values():
            excluded = region._manually_stepped_stp
            for synapse_id, stp_module in region.stp_modules.items():
                if synapse_id not in excluded:
                    entries.append((synapse_id, stp_module))
        return STPBatch(entries, device=self.device)

    def _create_neuron_batch(self) -> ConductanceLIFBatch:
        """Collect all eligible ConductanceLIF populations into one ConductanceLIFBatch.

        **Eligibility**: Only exact ``ConductanceLIF`` instances are batched.
        Subclasses (SerotoninNeuron, NorepinephrineNeuron, AcetylcholineNeuron)
        and ``TwoCompartmentLIF`` are excluded.

        After this call, eligible neurons' ``forward()`` writes inputs into the
        batch's global buffers and returns a deferred spike view.  ``Brain.forward()``
        then calls ``batch.step()`` once per timestep.
        """
        entries: list[tuple[PopulationKey, ConductanceLIF]] = []
        for region_name, region in self.regions.items():
            for pop_name, neuron in region.neuron_populations.items():
                # Exact type match — subclasses are excluded
                if type(neuron) is ConductanceLIF:
                    entries.append(((region_name, pop_name), neuron))

        batch = ConductanceLIFBatch(entries, device=self.device)
        batch.update_temporal_parameters(self.dt_ms)
        return batch

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @staticmethod
    def _merge_inputs(region_inputs: Dict[RegionName, SynapticInput], inputs_to_inject: SynapticInput) -> None:
        """Merge external input spikes into region_inputs with logical-OR."""
        for synapse_id, spikes in inputs_to_inject.items():
            target_region = synapse_id.target_region
            if target_region not in region_inputs:
                region_inputs[target_region] = {}
            if synapse_id in region_inputs[target_region]:
                region_inputs[target_region][synapse_id] = region_inputs[target_region][synapse_id] | spikes
            else:
                region_inputs[target_region][synapse_id] = spikes

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

        if GlobalConfig.DEBUG:
            for synapse_id, spikes in synaptic_inputs.items():
                validate_spike_tensor(spikes, tensor_name=synapse_id)

        # Cache nn.Module-registered dicts to avoid __getattr__ overhead
        # (searches _parameters, _buffers, _modules) on every access.
        _regions = self.regions           # nn.ModuleDict — 6 loops/step
        _axonal_tracts = self.axonal_tracts  # SynapseIdModuleDict — 2 loops/step

        # STEP 1: Read delayed outputs from all axonal tracts (spikes from T-delay)
        region_inputs: Dict[RegionName, SynapticInput] = {}

        for axonal_tract in _axonal_tracts.values():
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
                else:
                    region_inputs[target_region][synapse_id] = delayed_source_spikes

        # STEP 2: Inject external input spikes — sensory inputs passed by the caller AND
        # any reward spikes queued by deliver_reward() in the previous call.
        # All sources are merged with logical-OR so they layer on top of delayed tract outputs.
        if synaptic_inputs is not None:
            Brain._merge_inputs(region_inputs, synaptic_inputs)

        if self._pending_reward_inputs is not None:
            Brain._merge_inputs(region_inputs, self._pending_reward_inputs)
            self._pending_reward_inputs = None

        # STEP 3: HIPPOCAMPAL-VTA NOVELTY ROUTING (one-step causal delay)
        # CA1 mismatch from the *previous* step drives a VTA novelty burst *this* step.
        # Hippocampus and VTA execute in the same forward() call, so the mismatch scalar
        # computed at step T is stored as _pending_novelty_level and injected here at T+1.
        # Ref: Lisman & Grace 2005 — Hippocampal-VTA loop.
        if self._pending_novelty_level > 0.05:
            _vta = self.get_region_by_name("vta")
            if _vta is not None:
                _novelty_synapse = SynapseId.external_novelty_to_vta_da(_vta.region_name)
                try:
                    _n_novelty: int = _vta.get_synaptic_weights(_novelty_synapse).shape[1]
                except (KeyError, AttributeError):
                    _n_novelty = 0
                if _n_novelty > 0:
                    _novelty_spikes = generate_reward_spikes(
                        reward=self._pending_novelty_level,
                        n_neurons=_n_novelty,
                        device=self.device,
                    )
                    _novelty_target = _novelty_synapse.target_region
                    if _novelty_target not in region_inputs:
                        region_inputs[_novelty_target] = {}
                    if _novelty_synapse in region_inputs[_novelty_target]:
                        region_inputs[_novelty_target][_novelty_synapse] = (
                            region_inputs[_novelty_target][_novelty_synapse] | _novelty_spikes
                        )
                    else:
                        region_inputs[_novelty_target][_novelty_synapse] = _novelty_spikes

        # STEP 4: Collect neuromodulator signals from the previous timestep's output.
        # Each neuromodulator region declares outputs via a ``neuromodulator_outputs``
        # ClassVar (e.g. VTA→'da', LC→'ne', NB→'ach').
        # Raw spike tensors are broadcast to all regions; the per-region
        # NeuromodulatorReceptor handles rise/decay dynamics internally.
        # One-timestep lag is unavoidable with clock-driven simulation (ADR-003).
        neuromodulator_signals: NeuromodulatorInput = self.neuromodulator_hub.build(self._last_brain_output)

        # STEP 4b: Convert all region_inputs to float (ADR-004 relaxation).
        # Axonal tracts store bool for 8× memory savings, but all downstream
        # consumers (STP, sparse matmul, dendrite integration, learning) need
        # float.  Converting once here eliminates ~100k redundant .float() calls.
        for _ri_dict in region_inputs.values():
            for _sid in _ri_dict:
                _t = _ri_dict[_sid]
                if not _t.is_floating_point():
                    _ri_dict[_sid] = _t.float()

        # STEP 5: Execute all regions and batch-fire ConductanceLIF neurons.
        # =====================================================================
        # Four-phase execution:
        # Phase 1: All regions run _step() → ConductanceLIF populations write
        #          inputs to batch buffers (returning deferred spike views);
        #          TwoCompartmentLIF / subclass neurons fire normally.
        # Phase 2: _neuron_batch.step() executes ONE fused kernel call for all
        #          batched ConductanceLIF neurons, filling deferred spike views
        #          in-place so that region_output dicts now hold correct spikes.
        # Phase 3: Apply inter-region learning (using correct post-step spikes).
        # Phase 4: Write outputs to axonal tracts.
        # =====================================================================
        stp_efficacy = self._stp_batch.step(region_inputs, self._last_brain_output)

        # Global sparse synaptic integration (Step 5): 4 sparse matmuls replace
        # hundreds of per-connection dense matmuls for ConductanceLIF targets.
        # This fills batch input buffers (g_ampa_input, etc.) for batched pops
        # and prepares subclass conductances for region bypass.
        self._sparse_matrix.integrate(region_inputs, self._last_brain_output, stp_efficacy)
        self._sparse_matrix.scatter_to_neuron_batch()

        # Phase 1: Run all regions' _step().
        brain_output: BrainOutput = {}
        region_synaptic_cache: Dict[RegionName, SynapticInput] = {}

        # Pre-set precomputed fields for ALL regions before execution
        # (required for thread-safety — each thread reads its own region's fields).
        for region_name, region in _regions.items():
            region._precomputed_stp_efficacy = stp_efficacy
            region._precomputed_sparse_conductances = self._sparse_matrix.get_region_conductances(region_name)

        if self.config.execute_regions_in_parallel and len(_regions) > 1:
            # Parallel path: run regions concurrently via thread pool.
            # Safe because: regions write only to disjoint state (own neurons,
            # own batch buffer slices, own RegionOutput), and read from
            # pre-materialized immutable inputs.  C++ kernels release the GIL.
            pool = _region_thread_pool.get(
                max_workers=min(self.config.parallel_regions_max_workers, len(_regions)),
            )
            futures: Dict[concurrent.futures.Future[RegionOutput], RegionName] = {}
            for region_name, region in _regions.items():
                synaptic_inputs_for_region: SynapticInput = region_inputs.get(region_name, {})
                region_synaptic_cache[region_name] = synaptic_inputs_for_region
                future = pool.submit(
                    _execute_region,
                    region,
                    synaptic_inputs_for_region,
                    neuromodulator_signals,
                )
                futures[future] = region_name

            for future in concurrent.futures.as_completed(futures):
                region_name = futures[future]
                brain_output[region_name] = future.result()
        else:
            # Sequential path (fallback / single-region).
            for region_name, region in _regions.items():
                synaptic_inputs_for_region = region_inputs.get(region_name, {})
                region_synaptic_cache[region_name] = synaptic_inputs_for_region
                brain_output[region_name] = region.forward(synaptic_inputs_for_region, neuromodulator_signals)

        # Clear precomputed fields after execution
        for region_name, region in _regions.items():
            region._precomputed_stp_efficacy = None
            region._precomputed_sparse_conductances = None

        # Phase 2: Execute all batched ConductanceLIF neurons in one kernel call.
        # This fills the deferred spike views in-place, so brain_output now
        # contains correct current-step spikes for all batched populations.
        self._neuron_batch.step()

        # Phase 2b: Convert all brain_output spikes to float (ADR-004 relaxation).
        # Neurons produce bool spikes, but learning rules, neuromodulator hub,
        # and _last_brain_output consumers all need float.  One bulk conversion
        # here replaces scattered per-consumer .float() calls.
        for _region_out in brain_output.values():
            for _pop_name in _region_out:
                _t = _region_out[_pop_name]
                if not _t.is_floating_point():
                    _region_out[_pop_name] = _t.float()

        # Phase 3: Apply inter-region learning (post-step spikes are now valid).
        if not GlobalConfig.LEARNING_DISABLED:
            for region_name, region in _regions.items():
                region.apply_learning(region_synaptic_cache[region_name], brain_output[region_name])

        # Update pending novelty level from hippocampus CA1 mismatch (injected next step)
        _hippocampus = self.get_region_by_name("hippocampus")
        if _hippocampus is not None and hasattr(_hippocampus, "ca1_mismatch_level"):
            self._pending_novelty_level = float(_hippocampus.ca1_mismatch_level)
        else:
            self._pending_novelty_level = 0.0

        # STEP 6: Write current outputs to axonal tracts and advance buffers (available at T+delay)
        for axonal_tract in _axonal_tracts.values():
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

    def deliver_reward(self, external_reward: float) -> None:
        """Queue a reward signal for VTA on the next :meth:`forward` call.

        **Spiking Dopamine Architecture**: Reward is encoded as population-coded
        spikes by :func:`~thalia.utils.generate_reward_spikes` and queued for
        injection into VTA on the **next** call to :meth:`forward`.  This keeps
        reward delivery inside the normal clock-driven timestep and avoids the
        double-stepping bug that arose when this method used to call
        ``forward()`` directly.

        Typical training-loop pattern::

            for t in range(trial_steps):
                brain.forward(sensory_input)   # perception + action
            brain.deliver_reward(reward)       # queues reward spikes
            brain.forward()                    # VTA sees reward + normal dynamics

        Args:
            external_reward: Task-based reward value (typically -1 to +1).

        Raises:
            ValueError: If VTA or Striatum are not present in the brain.

        Note:
            The reward spikes reach VTA on the *next* ``forward()`` call, so
            there is a one-timestep lag relative to the moment
            ``deliver_reward()`` is invoked.  This matches the biological
            reality that RPE signals are computed on a ~10–100 ms timescale,
            not instantaneously.
        """
        striatum = self.get_region_by_name("striatum")
        if striatum is None or not hasattr(striatum, 'update_performance'):
            raise ValueError(
                "Striatum with update_performance() method not found. Cannot deliver reward. "
                "Brain must include a region named 'striatum' with an update_performance() method for RL tasks."
            )

        vta = self.get_region_by_name("vta")
        _DA_CHANNELS = {
            NeuromodulatorChannel.DA_MESOLIMBIC,
            NeuromodulatorChannel.DA_MESOCORTICAL,
            NeuromodulatorChannel.DA_NIGROSTRIATAL,
        }
        if vta is None or not hasattr(vta, 'neuromodulator_outputs') or not any(k in _DA_CHANNELS for k in vta.neuromodulator_outputs.keys()):
            raise ValueError(
                "VTA with DA output not found. Cannot deliver reward. "
                "Brain must include a region named 'vta' whose neuromodulator_outputs "
                "contains at least one DA channel (NeuromodulatorChannel.DA_MESOLIMBIC, "
                "DA_MESOCORTICAL, or DA_NIGROSTRIATAL)."
            )

        # Build reward spikes and store them; forward() will drain this on the next call.
        # n_reward_neurons is read from VTA's weight matrix so it always matches the
        # exact n_input registered by add_external_input_source() in BrainBuilder.
        reward_synapse = SynapseId.external_reward_to_vta_da(vta.region_name)
        n_reward_neurons: int = vta.get_synaptic_weights(reward_synapse).shape[1]
        spikes = generate_reward_spikes(reward=external_reward, n_neurons=n_reward_neurons, device=self.device)
        self._pending_reward_inputs = {reward_synapse: spikes}

        # Update exploration statistics based on reward.
        # Striatum applies continuous learning automatically in forward() using
        # spiking DA; this call only updates the per-trial performance history
        # used to adapt exploration rate (tonic dopamine).  It is intentionally
        # NOT routed through forward() because it must run exactly once per
        # trial, whereas forward() can run many times per trial.
        striatum.update_performance(external_reward)

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

        # Update batched STP (if already initialised)
        if hasattr(self, '_stp_batch'):
            self._stp_batch.update_temporal_parameters(new_dt_ms)

        # Update batched neuron state (if already initialised)
        if hasattr(self, '_neuron_batch'):
            self._neuron_batch.update_temporal_parameters(new_dt_ms)

        # Update all axonal tracts
        for axonal_tract in self.axonal_tracts.values():
            axonal_tract.update_temporal_parameters(new_dt_ms)

    # =========================================================================
    # WEIGHT CALIBRATION
    # =========================================================================

    @torch.no_grad()
    def calibrate_weights(
        self,
        *,
        n_steps: int,
        n_iterations: int,
        warmup_steps: int,
        correction_gain: float,
        max_correction: float,
    ) -> Dict[str, Any]:
        """Calibrate excitatory weights so spontaneous firing matches homeostatic targets.

        Runs ``n_iterations`` calibration cycles.  Each cycle:

        1. Runs ``warmup_steps`` spontaneous-activity timesteps (no external input)
           to let post-correction transients decay.
        2. Runs ``n_steps`` timesteps and accumulates per-neuron spike counts.
        3. For each population with a registered homeostatic target, computes a
           multiplicative correction factor and applies it uniformly to all
           incoming excitatory (AMPA/NMDA) weight matrices — both dense intra-region
           weights and sparse inter-region weights in the global CSR matrices.
        4. Rebuilds the precomputed batched dendrite weight matrices so the next
           iteration sees the corrected weights.

        **Correction formula** (geometric damping to prevent oscillation)::

            scale = (target_rate / actual_rate) ** correction_gain
            scale = clamp(scale, 1 / max_correction, max_correction)

        A gain of 0.5 takes the square-root of the ideal correction each cycle,
        converging in ~5 iterations while avoiding overshoot.

        Args:
            n_steps: Measurement timesteps per calibration cycle.
            n_iterations: Number of calibration cycles.
            warmup_steps: Transient-discard steps before each measurement phase.
            correction_gain: Exponent for geometric damping (0 < gain ≤ 1).
                1.0 applies the full correction; 0.5 (default) halves the log-error.
            max_correction: Per-iteration clamp for the scale factor
                (clips to ``[1/max_correction, max_correction]``).

        Returns:
            Dict with:
                ``'iterations'``: list of per-cycle dicts with keys ``'rates_hz'``
                    (``"region/pop"`` → Hz) and ``'corrections'`` (``"region/pop"``
                    → ``{'actual_hz', 'target_hz', 'scale'}``).
                ``'final_rates_hz'``: rates measured in the final cycle.
        """
        iteration_results = []
        final_rates: Dict[str, float] = {}

        for iteration in range(n_iterations):
            # ------------------------------------------------------------------
            # 1. Warm-up: discard transients from the previous correction.
            # ------------------------------------------------------------------
            for _ in range(warmup_steps):
                self.forward(None)

            # ------------------------------------------------------------------
            # 2. Measurement: accumulate {(region, pop): [total_spikes, n_neurons]}.
            # ------------------------------------------------------------------
            spike_accum: Dict[tuple[str, str], list[float]] = {}
            for _ in range(n_steps):
                brain_output = self.forward(None)
                for region_name, region_output in brain_output.items():
                    for pop_name, spikes in region_output.items():
                        key = (region_name, pop_name)
                        total = float(spikes.float().sum().item())
                        n = float(spikes.numel())
                        if key in spike_accum:
                            spike_accum[key][0] += total
                        else:
                            spike_accum[key] = [total, n]

            # Convert to Hz for the output dict.
            # actual_rate_per_timestep = total_spikes / n_neurons / n_steps
            # actual_hz = actual_rate_per_timestep * 1000.0 / dt_ms
            hz_factor = 1000.0 / (n_steps * self.dt_ms)
            actual_rates_hz: Dict[str, float] = {
                f"{rn}/{pn}": (v[0] / v[1]) * hz_factor
                for (rn, pn), v in spike_accum.items()
            }

            # ------------------------------------------------------------------
            # 3. Correction: scale excitatory weights for homeostasis populations.
            # ------------------------------------------------------------------
            corrections: Dict[str, Dict[str, float]] = {}

            for region_name, region in self.regions.items():
                for pop_name, state in region._homeostasis.items():
                    key = (region_name, pop_name)
                    if key not in spike_accum:
                        continue  # Population produced no output — skip.

                    total, n = spike_accum[key]
                    actual_rate_per_step = total / n / n_steps  # spikes/timestep
                    if actual_rate_per_step <= 0.0:
                        continue  # Completely silent — can't form a finite ratio.

                    # target_firing_rate is in spikes/timestep (calibrated at dt=1ms).
                    # The ratio target/actual is dt-independent when both are in the
                    # same units, so no explicit dt conversion is needed for the scale.
                    target_rate_per_step = state.target_firing_rate
                    if target_rate_per_step <= 0.0:
                        continue

                    raw_scale = target_rate_per_step / actual_rate_per_step
                    damped_scale = raw_scale ** correction_gain
                    clamped_scale = max(1.0 / max_correction, min(max_correction, damped_scale))

                    corrections[f"{region_name}/{pop_name}"] = {
                        "actual_hz": actual_rate_per_step * 1000.0 / self.dt_ms,
                        "target_hz": target_rate_per_step * 1000.0 / self.dt_ms,
                        "scale": clamped_scale,
                    }

                    self._scale_excitatory_weights(region, pop_name, clamped_scale)

            # ------------------------------------------------------------------
            # 4. Rebuild batched dendrite matrices to reflect corrected weights.
            # ------------------------------------------------------------------
            self._rebuild_batched_dendrite_weights()

            iteration_results.append({
                "iteration": iteration,
                "rates_hz": actual_rates_hz,
                "corrections": corrections,
            })
            final_rates = actual_rates_hz

        return {
            "iterations": iteration_results,
            "final_rates_hz": final_rates,
        }

    def _scale_excitatory_weights(
        self,
        region: "NeuralRegion[NeuralRegionConfig]",
        population_name: str,
        scale: float,
    ) -> None:
        """Uniformly scale all incoming excitatory weights for one population.

        Modifies both dense ``region.synaptic_weights`` (intra-region connections
        and any remaining dense inter-region connections) and the shared
        ``GlobalSparseMatrix`` (sparse inter-region connections), then clamps
        all values to ``[0, region.config.synaptic_scaling.w_max]``.

        Args:
            region: Target region whose population receives the scaled inputs.
            population_name: Name of the postsynaptic population to correct.
            scale: Multiplicative scale factor (> 1 boosts; < 1 suppresses).
        """
        w_max = region.config.synaptic_scaling.w_max

        # Dense weights (intra-region connections stored per-region).
        for synapse_id, weights in region.synaptic_weights.items():
            if (
                synapse_id.target_population == population_name
                and synapse_id.receptor_type.is_excitatory
            ):
                weights.data.mul_(scale).clamp_(max=w_max)

        # Sparse weights (inter-region connections in the global CSR matrix).
        for synapse_id, meta in self._sparse_matrix.connections.items():
            if (
                synapse_id.target_region == region.region_name
                and synapse_id.target_population == population_name
                and synapse_id.receptor_type.is_excitatory
                and meta.nnz > 0
            ):
                values = self._sparse_matrix.get_weight_values(synapse_id)
                values.mul_(scale).clamp_(max=w_max)
                self._sparse_matrix.set_weight_values(synapse_id, values)

    def _rebuild_batched_dendrite_weights(self) -> None:
        """Rebuild precomputed batched dendrite weight matrices after weight modification.

        Must be called after any in-place modification of ``synaptic_weights``
        so that the concatenated batched matrices used by
        ``_integrate_synaptic_inputs_at_dendrites`` reflect the updated values.
        The sparse CSR matrix is updated in-place by
        :meth:`_scale_excitatory_weights` and does not need rebuilding.
        """
        _sparse_sids = frozenset(self._sparse_matrix.connections.keys())
        for region in self.regions.values():
            region.build_batched_dendrite_weights(exclude_synapse_ids=_sparse_sids)
