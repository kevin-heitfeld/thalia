"""DynamicBrain: Flexible, graph-based brain architecture with modular regions."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple, Type, TypeVar, cast

import math

import torch
import torch.nn as nn

from thalia.typing import (
    BrainOutput,
    NeuromodulatorInput,
    PopulationName,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import compute_firing_rate, validate_spike_tensor

from .axonal_tract import AxonalTract
from .configs import BrainConfig, NeuralRegionConfig

from .regions import (
    NeuralRegion,
    Cortex,
    Hippocampus,
    RewardEncoder,
    Striatum,
)


RegionT = TypeVar('RegionT', bound=NeuralRegion[NeuralRegionConfig])


# =============================================================================
# TYPED-KEY CONTAINER FOR AXONAL TRACTS
# =============================================================================

class AxonalTractDict(nn.Module):
    """nn.ModuleDict wrapper keyed by (RegionName, PopulationName) tuples.

    nn.ModuleDict requires str keys; this class encodes the tuple to a stable
    string so that .to(), .state_dict(), and .parameters() all work correctly.
    """

    _SEP = "|"

    def __init__(self) -> None:
        super().__init__()
        self._md: nn.ModuleDict = nn.ModuleDict()

    @classmethod
    def _encode(cls, key: Tuple[RegionName, PopulationName]) -> str:
        region, population = key
        return f"{region}{cls._SEP}{population}"

    @classmethod
    def _decode(cls, raw: str) -> Tuple[RegionName, PopulationName]:
        region, population = raw.split(cls._SEP, 1)
        return region, population

    def __setitem__(self, key: Tuple[RegionName, PopulationName], value: AxonalTract) -> None:
        self._md[self._encode(key)] = value

    def __getitem__(self, key: Tuple[RegionName, PopulationName]) -> AxonalTract:
        return self._md[self._encode(key)]  # type: ignore[return-value]

    def __contains__(self, key: object) -> bool:
        try:
            return self._encode(key) in self._md
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self._md)

    def __iter__(self) -> Iterator[Tuple[RegionName, PopulationName]]:
        return (self._decode(k) for k in self._md)

    def items(self) -> Iterator[Tuple[Tuple[RegionName, PopulationName], AxonalTract]]:
        """Yield ((RegionName, PopulationName), AxonalTract) pairs."""
        return ((self._decode(k), v) for k, v in self._md.items())  # type: ignore[return-value]

    def values(self) -> Iterator[AxonalTract]:
        """Yield AxonalTract values."""
        return self._md.values()  # type: ignore[return-value]


# =============================================================================
# NEUROMODULATOR DIFFUSION TRACTS
# =============================================================================

# Default first-order low-pass time constants for neuromodulator volume diffusion.
# Biological refs: DA ~100–300 ms (Garris et al.); NE ~80 ms; ACh ~60 ms (muscarinic onset).
_NEUROMOD_DEFAULT_TAU_MS: Dict[str, float] = {
    'da': 150.0,
    'ne': 80.0,
    'ach': 60.0,
}


class NeuromodulatorTract(nn.Module):
    """First-order IIR low-pass filter modeling neuromodulator volume diffusion.

    Replaces the raw 1-step delay with a biologically realistic filter::

        filtered(t) = α · filtered(t-1) + (1 - α) · raw(t)

    where α = exp(-dt / tau_ms).  This models the slow extracellular spread of
    dopamine (τ ≈ 150 ms), norepinephrine (τ ≈ 80 ms), and acetylcholine
    (τ ≈ 60 ms).  The ``filtered`` buffer is registered via
    ``register_buffer`` so ``.to()``, ``state_dict()``, and
    ``load_state_dict()`` all work correctly.
    """

    def __init__(self, tau_ms: float, dt_ms: float) -> None:
        super().__init__()
        self.tau_ms = tau_ms
        self._alpha: float = math.exp(-dt_ms / tau_ms)

    def update(self, raw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Update filter state with the latest raw neuromodulator spike tensor.

        Args:
            raw: Spike tensor from the neuromodulator source population, or
                 ``None`` if the source region produced no output this step.

        Returns:
            Filtered tensor (same shape/device as *raw*), or ``None`` if the
            tract has not yet received any input.
        """
        if raw is None:
            return self._buffers.get('filtered', None)

        raw_f = raw.float()
        if 'filtered' not in self._buffers or self._buffers['filtered'] is None:
            # Lazy initialisation: allocate on first spike, matching shape & device.
            self.register_buffer('filtered', torch.zeros_like(raw_f))

        # Exponential moving average — models slow extracellular diffusion.
        self.filtered: torch.Tensor = self._alpha * self.filtered + (1.0 - self._alpha) * raw_f
        return self.filtered

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Recompute the IIR decay coefficient when the simulation timestep changes."""
        self._alpha = math.exp(-dt_ms / self.tau_ms)


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

    def get_region(self, region_name: RegionName, region_type: Type[RegionT]) -> Optional[RegionT]:
        """Get region by name and type, or None if not present."""
        region = self.regions[region_name] if region_name in self.regions else None
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
        axonal_tracts: Dict[Tuple[RegionName, PopulationName], AxonalTract],
    ):
        """Initialize DynamicBrain from graph.

        Args:
            config: Brain configuration (device, dt_ms, etc.)
            regions: Dict mapping region names to instances
            axonal_tracts: Dict mapping (target_region, target_population) to AxonalTract instances
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

        # Create one NeuromodulatorTract per modulator key, discovered by scanning
        # regions for the neuromodulator_outputs ClassVar.
        # Each tract models volume diffusion via a first-order IIR low-pass filter.
        self.neuromodulator_tracts: nn.ModuleDict = nn.ModuleDict()
        for region in regions.values():
            if hasattr(region, 'neuromodulator_outputs'):
                for mod_key in region.neuromodulator_outputs:
                    if mod_key not in self.neuromodulator_tracts:
                        tau = _NEUROMOD_DEFAULT_TAU_MS.get(mod_key, 100.0)
                        self.neuromodulator_tracts[mod_key] = NeuromodulatorTract(
                            tau_ms=tau, dt_ms=config.dt_ms
                        )

        # Current simulation time
        self._current_time: float = 0.0

        # Last timestep's output for neuromodulator broadcast (initialized to None)
        self._last_brain_output: Optional[BrainOutput] = None

        # =================================================================
        # INITIALIZE TEMPORAL PARAMETERS
        # =================================================================
        # Broadcast dt_ms to all regions for initial setup
        # Regions compute decay factors, phase increments, etc.
        self.set_timestep(self.dt_ms)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def __call__(self, *args, **kwds):
        assert False, f"{self.__class__.__name__} instances should not be called directly. Use forward() instead."
        return super().__call__(*args, **kwds)

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
        region_inputs: SynapticInput = {}

        for (target_region, target_population), axonal_tract in self.axonal_tracts.items():
            # Read delayed outputs WITHOUT writing or advancing
            delayed_outputs: BrainOutput = axonal_tract.read_delayed_outputs()

            for source_region, population_dict in delayed_outputs.items():
                for source_population, delayed_source_spikes in population_dict.items():
                    synapse_id = SynapseId(
                        source_region=source_region,
                        source_population=source_population,
                        target_region=target_region,
                        target_population=target_population,
                        is_inhibitory=False,  # TODO: Support inhibitory tracts in the future by adding is_inhibitory to AxonalTract and SynapseId
                    )

                    if synapse_id in region_inputs:
                        raise ValueError(
                            f"Duplicate synapse_id {synapse_id} from axonal tracts. "
                            f"Check for overlapping source populations or routing errors."
                        )

                    region_inputs[synapse_id] = delayed_source_spikes

        # STEP 2: Inject external input spikes (e.g., sensory) - these have FULL routing keys already
        # External inputs are added on top of delayed spikes (logical OR) since both contribute to activation
        if synaptic_inputs is not None:
            for synapse_id, synapse_input in synaptic_inputs.items():
                validate_spike_tensor(synapse_input, tensor_name=str(synapse_id))

                # Logical OR if key already exists
                # (e.g., external sensory input + delayed thalamic input to same target)
                if synapse_id in region_inputs:
                    region_inputs[synapse_id] = region_inputs[synapse_id] | synapse_input
                else:
                    region_inputs[synapse_id] = synapse_input

        # STEP 3: Collect neuromodulator signals via diffusion tracts.
        # Each neuromodulator region declares outputs via a `neuromodulator_outputs` ClassVar.
        # Raw spikes are fed through a NeuromodulatorTract IIR filter that models the
        # slow volume diffusion of DA/NE/ACh.
        neuromodulator_signals: NeuromodulatorInput = {}
        for region_name, region in self.regions.items():
            if hasattr(region, 'neuromodulator_outputs'):
                for mod_key, pop_name in region.neuromodulator_outputs.items():
                    raw: Optional[torch.Tensor] = None
                    if self._last_brain_output is not None:
                        raw = self._last_brain_output.get(region_name, {}).get(pop_name)
                    neuromodulator_signals[mod_key] = self.neuromodulator_tracts[mod_key].update(raw)

        # STEP 4: Execute all regions with synaptic inputs AND neuromodulator broadcast → produce outputs at T
        brain_output: BrainOutput = {}
        for region_name, region in self.regions.items():
            synaptic_inputs_for_region: SynapticInput = {k: v for k, v in region_inputs.items() if k.target_region == region_name}
            region_output: RegionOutput = region.forward(synaptic_inputs_for_region, neuromodulator_signals)
            brain_output[region_name] = region_output

        # Store for next timestep's neuromodulator broadcast
        self._last_brain_output = brain_output

        # STEP 5: Write current outputs to axonal tracts and advance buffers
        # CRITICAL: Each axonal tract may have MULTIPLE sources, so we iterate over axonal_tract.source_specs
        for _, axonal_tract in self.axonal_tracts.items():
            # Write ALL source regions' current outputs to buffer (for T+delay reads)
            # This handles multiple sources per tract correctly, as each source's spikes are extracted from brain_output
            axonal_tract.write_and_advance(brain_output)

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
        # Check for hippocampus
        hippocampus = self.get_first_region_of_type(Hippocampus)
        if hippocampus is None:
            raise ValueError(
                "Hippocampus required for consolidation. "
                "Brain must include 'hippocampus' region."
            )

        # Enter consolidation mode (low acetylcholine)
        hippocampus.enter_consolidation_mode()
        hippocampus.set_neuromodulators(acetylcholine=0.1)

        # Run brain forward - replay happens automatically
        n_timesteps = int(duration_ms / self.dt_ms)
        ripple_count = 0

        for _ in range(n_timesteps):
            self.forward(None)  # No sensory input during sleep
            if hippocampus.ripple_detected:
                ripple_count += 1

        # Return to encoding mode (high acetylcholine)
        hippocampus.set_neuromodulators(acetylcholine=0.7)
        hippocampus.exit_consolidation_mode()

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
        spikes in RewardEncoder, which feeds into VTA to modulate DA neuron firing.
        VTA DA spikes are then broadcast to all regions with DA receptors
        (Striatum, PFC, Hippocampus, Cortex) where they are converted to
        concentration dynamics via NeuromodulatorReceptor.

        Args:
            external_reward: Task-based reward value (typically -1 to +1),
                           or None for pure intrinsic reward

        Raises:
            ValueError: If reward_encoder or striatum not found

        Note:
            Actual learning happens continuously in region forward() passes using
            the spiking dopamine signal. This method only delivers reward to
            RewardEncoder - the rest happens automatically through VTA and DA receptors.
        """
        striatum = self.get_first_region_of_type(Striatum)
        if striatum is None:
            raise ValueError(
                "Striatum not found. Cannot deliver reward. "
                "Brain must include 'striatum' region for RL tasks."
            )

        # Get RewardEncoder region
        reward_encoder = self.get_first_region_of_type(RewardEncoder)
        if reward_encoder is None:
            raise ValueError(
                "RewardEncoder not found. Cannot deliver reward. "
                "Brain must include 'reward_encoder' region for spiking DA system."
            )

        # Compute total reward (external + intrinsic if available)
        intrinsic_reward = self._compute_intrinsic_reward()
        if external_reward is None:
            total_reward = intrinsic_reward
        else:
            total_reward = external_reward + intrinsic_reward

        # Deliver reward to RewardEncoder as population-coded spikes
        # RewardEncoder converts scalar → spike pattern, which VTA decodes
        reward_encoder.set_reward(total_reward)

        # Update exploration statistics based on reward
        # Striatum applies continuous learning automatically in forward() using spiking DA
        striatum.update_performance(total_reward)

    def _get_cortex_l4_activity(self) -> Optional[float]:
        """Helper to get cortex L4 activity for neuromodulator computations."""
        cortex = self.get_first_region_of_type(Cortex)
        if cortex is not None:
            return compute_firing_rate(cortex.l4_spikes)
        return None

    def _get_hippocampus_ca1_activity(self) -> Optional[float]:
        """Helper to get hippocampus CA1 activity for neuromodulator computations."""
        hippocampus = self.get_first_region_of_type(Hippocampus)
        if hippocampus is not None:
            return compute_firing_rate(hippocampus.ca1_spikes)
        return None

    def _compute_intrinsic_reward(self) -> float:
        """Compute intrinsic reward from the brain's internal objectives.

        This implements the free energy principle: the brain rewards itself
        for minimizing prediction error (surprise). Intrinsic reward is
        ALWAYS computed - it's the brain's continuous self-evaluation.

        Sources:
        1. **Cortex predictive coding**: Low prediction error → good model of the world → reward
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
        # L4 activity = prediction error
        # L4 = input - (L5+L6 prediction), so low L4 = good prediction = reward
        l4_activity = self._get_cortex_l4_activity()
        if l4_activity is not None:
            # L4 activity is prediction error (0 = perfect prediction, 1 = max error)
            # Map to reward: 0 → +1, 0.5 → 0, 1 → -1
            cortex_reward = 1.0 - 2.0 * l4_activity
            cortex_reward = max(-1.0, min(1.0, cortex_reward))
            reward += cortex_reward
            n_sources += 1

        # =====================================================================
        # 2. HIPPOCAMPUS PATTERN COMPLETION (memory recall quality)
        # =====================================================================
        # High pattern similarity = successful memory retrieval = reward
        # Biology: VTA observes CA1 output activity. Strong coherent firing = successful recall.
        # We infer similarity from CA1 spike rate (observable signal).
        ca1_activity = self._get_hippocampus_ca1_activity()
        if ca1_activity is not None:
            # CA1 firing rate as proxy for retrieval quality
            # High rate = strong recall, low rate = weak/no recall

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
            # No signals → assume moderate intrinsic reward
            reward = 0.0

        return max(-1.0, min(1.0, reward))

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

        # Update neuromodulator diffusion tracts
        for nm_tract in self.neuromodulator_tracts.values():
            nm_tract.update_temporal_parameters(new_dt_ms)
