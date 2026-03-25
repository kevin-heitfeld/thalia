"""NeuralRegion base class for brain regions with biologically accurate synaptic inputs.

This class implements the core functionality for neural regions, including:
1. Managing synaptic weights for multiple input sources (dendrites)
2. Integrating synaptic inputs at dendrites (multi-source summation)
3. Providing utilities for homeostatic plasticity and temporal parameter management

Regions with internal structure should subclass NeuralRegion and:
1. Call super().__init__() to initialize synaptic weight management
2. Define internal neuron populations and connections for within-region processing
3. Override forward() to apply synaptic weights and internal processing

Biological Inspiration:
- Neurons receive inputs from multiple sources via dendrites, each with its own synaptic weights
- Dendrites integrate these inputs through linear summation of conductances
- Homeostatic plasticity maintains stable activity levels through intrinsic excitability and synaptic scaling
- Temporal parameters (e.g., STP time constants) can be updated dynamically for biological realism
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Generic, List, NamedTuple, Optional, TypeVar, Union

import torch
import torch.nn as nn

from thalia import GlobalConfig
from thalia.brain.configs import NeuralRegionConfig
from thalia.brain.neurons import (
    AcetylcholineNeuronConfig,
    AcetylcholineNeuron,
    ConductanceLIFConfig,
    ConductanceLIF,
    NorepinephrineNeuronConfig,
    NorepinephrineNeuron,
    SerotoninNeuronConfig,
    SerotoninNeuron,
    TwoCompartmentLIFConfig,
    TwoCompartmentLIF,
)
from thalia.brain.synapses import (
    ConductanceScaledSpec,
    NMReceptorType,
    STPConfig,
    ShortTermPlasticity,
    WeightInitializer,
    make_neuromodulator_receptor,
)
from thalia.learning import LearningStrategy
from thalia.typing import (
    NeuromodulatorChannel,
    NeuromodulatorInput,
    PopulationName,
    PopulationPolarity,
    PopulationSizes,
    ReceptorType,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import (
    SynapseIdParameterDict,
    SynapseIdModuleDict,
    clamp_weights,
    validate_spike_tensor,
)


ConfigT = TypeVar('ConfigT', bound=NeuralRegionConfig)


@dataclass(frozen=True)
class InternalConnectionSpec:
    """Parameters for a sparse-random internal connection (E→I or I→E).

    Bundles the connectivity, weight scale, receptor type, and STP config
    for one directional connection within a region.  Used by
    :meth:`NeuralRegion._add_feedforward_inhibition` to reduce boilerplate
    when wiring standard E→I→E circuits.
    """

    connectivity: float
    weight_scale: float
    receptor_type: ReceptorType
    stp_config: Optional[STPConfig]
    learning_strategy: Optional[LearningStrategy] = None


@dataclass
class PopulationHomeostasisState:
    """Homeostatic tracking state for a single neuron population.

    Stores buffer attribute names (strings) rather than tensor references so that
    device moves via ``.to(device)`` keep the module's ``_buffers`` dict in sync
    without stale references.

    Attributes:
        population_name: Key in ``neuron_populations`` for neuron lookup.
        firing_rate_attr: Attribute name of the registered firing-rate EMA buffer.
        target_firing_rate: Per-population homeostatic target rate (spikes/ms).
    """

    population_name: PopulationName
    firing_rate_attr: str
    target_firing_rate: float


@dataclass(frozen=True)
class ReceptorSpec:
    """Declarative specification for a neuromodulator receptor.

    Used with :meth:`NeuralRegion._init_receptors` to create receptors and
    concentration buffers in a single call, and with
    :meth:`NeuralRegion._update_receptors` to batch-update all of them.

    Attributes:
        receptor_type: Canonical receptor subtype (determines kinetics).
        channel: Neuromodulatory broadcast channel to read spikes from.
        n_receptors: Number of postsynaptic receptors (typically population size).
        buffer_name: Attribute name for the registered concentration buffer.
        amplitude_scale: Multiplicative scale on canonical spike_amplitude.
        initial_value: Initial concentration value (default 0).
    """

    receptor_type: NMReceptorType
    channel: NeuromodulatorChannel
    n_receptors: int
    buffer_name: str
    amplitude_scale: float = 1.0
    initial_value: float = 0.0


class DendriteOutput(NamedTuple):
    """Output of `_integrate_synaptic_inputs_at_dendrites`.

    All tensors are non-negative conductances (clamped ≥ 0), separated by
    receptor type so callers can pass them to the correct neuron channels.

    Attributes:
        g_ampa:   AMPA conductance [n_neurons]   — fast excitatory
        g_nmda:   NMDA conductance [n_neurons]   — slow, voltage-gated excitatory
        g_gaba_a: GABA_A conductance [n_neurons] — fast inhibitory (Cl⁻, τ≈10 ms)
        g_gaba_b: GABA_B conductance [n_neurons] — slow inhibitory (K⁺, τ≈400 ms)
    """

    g_ampa:   torch.Tensor  # [n_neurons], non-negative
    g_nmda:   torch.Tensor  # [n_neurons], non-negative
    g_gaba_a: torch.Tensor  # [n_neurons], non-negative
    g_gaba_b: torch.Tensor  # [n_neurons], non-negative


# Scalar zero shared across unused DendriteOutput slots in the fast path.
# Callers of _integrate_single_synaptic_input always access only the one
# field matching the receptor type; the other three are never read.
_DENDRITE_ZERO = torch.tensor(0.0)


@dataclass(frozen=True)
class SynapseShape(NamedTuple):
    """Convenience struct for synapse shape metadata.

    Attributes:
        n_output: Number of post-synaptic neurons (output dimension).
        n_input: Number of pre-synaptic neurons (input dimension).
    """

    n_output: int
    n_input: int


class ConcatWeightBlock:
    """Precomputed concatenated weight matrix for one receptor type in a target population.

    Replaces N small matmuls with 1 larger one by horizontally stacking weight
    matrices from all sources of the same receptor type.
    Individual synaptic_weights entries become views into W_concat, so learning
    rule updates propagate automatically.
    """

    W_concat: torch.Tensor
    synapse_ids: tuple[SynapseId, ...]
    column_slices: tuple[slice, ...]
    total_sources: int
    spike_buffer: torch.Tensor

    __slots__ = ('W_concat', 'synapse_ids', 'column_slices', 'total_sources', 'spike_buffer')

    def __init__(
        self,
        W_concat: torch.Tensor,
        synapse_ids: tuple[SynapseId, ...],
        column_slices: tuple[slice, ...],
        total_sources: int,
        device: torch.device,
    ):
        self.W_concat = W_concat
        self.synapse_ids = synapse_ids
        self.column_slices = column_slices
        self.total_sources = total_sources
        self.spike_buffer = torch.zeros(total_sources, device=device)


class BatchedDendriteWeights:
    """Precomputed batched weights for all receptor types targeting one population.

    Used by the batched integration path in _integrate_synaptic_inputs_at_dendrites
    to replace per-source Python loops with one matmul per active receptor type.
    """

    n_target: int
    _blocks: Dict[ReceptorType, ConcatWeightBlock]

    __slots__ = ('n_target', '_blocks')

    def __init__(self, n_target: int, blocks: Dict[ReceptorType, ConcatWeightBlock]):
        self.n_target = n_target
        self._blocks = blocks

    def get_block(self, receptor_type: ReceptorType) -> Optional[ConcatWeightBlock]:
        return self._blocks.get(receptor_type)


class NeuralRegion(nn.Module, ABC, Generic[ConfigT]):
    """Base class for neural regions."""

    # Declared by subclasses that source neuromodulator volume-transmission signals.
    # Inherited from NeuromodulatorSource protocol check.
    # neuromodulator_outputs: ClassVar[Dict[NeuromodulatorChannel, PopulationName]]
    # (defined on source regions only — not all regions produce neuromodulators)

    # Declared by subclasses that *consume* neuromodulator channels in forward().
    # List the exact channel keys this region reads from NeuromodulatorInput.
    # NeuromodulatorHub.validate() raises at build time if any declared subscription
    # has no matching publisher, preventing silent signal loss.
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = []

    # =========================================================================
    # PROPERTIES AND UTILS
    # =========================================================================

    @property
    def dt_ms(self) -> float:
        """Timestep duration in milliseconds."""
        return self.config.dt_ms

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: ConfigT,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize NeuralRegion with config and layer sizes."""
        super().__init__()

        # Validate population sizes
        for source_name, size in population_sizes.items():
            if not isinstance(size, int) or size < 0:
                raise ValueError(
                    f"NeuralRegion.__init__: Invalid input size for source '{source_name}': "
                    f"expected non-negative int, got {size}."
                )

        # Store config and region name
        self.config: ConfigT = config
        self.region_name: RegionName = region_name
        self.device = torch.device(device)

        # EMA alpha for firing rate tracking
        self._firing_rate_alpha: float = self.dt_ms / self.config.homeostatic_gain.tau_ms

        # Neuron populations within this region.
        # PopulationName is str so nn.ModuleDict works directly; subclasses keep
        # their own typed references (self.l23, self.l4, …) for forward() calls.
        self.neuron_populations: nn.ModuleDict = nn.ModuleDict()  # Dict[PopulationName, ConductanceLIF | TwoCompartmentLIF]

        # Per-population polarity (EXCITATORY / INHIBITORY / ANY).
        # Populated by _create_and_register_neuron_population(); used for Dale's Law enforcement.
        self._population_polarities: Dict[PopulationName, PopulationPolarity] = {}

        # Per-population homeostatic state registry.
        self._homeostasis: Dict[PopulationName, PopulationHomeostasisState] = {}

        # Synaptic weights: one weight matrix per input source.
        # SynapseIdParameterDict wraps nn.ParameterDict so PyTorch tracks every
        # parameter correctly (.to(), .parameters(), .state_dict()) while the
        # public API remains SynapseId-typed.
        self.synaptic_weights: SynapseIdParameterDict = SynapseIdParameterDict()  # Dict[SynapseId, nn.Parameter]

        # Optional per-source STP modules for short-term plasticity (facilitation/depression)
        self.stp_modules: SynapseIdModuleDict = SynapseIdModuleDict()  # Dict[SynapseId, ShortTermPlasticity]

        # STP connections that the region steps manually (e.g. with multi-step
        # delayed spikes).  These are excluded from the global STPBatch to avoid
        # double-updating shared u/x state.
        self._manually_stepped_stp: set[SynapseId] = set()

        # Precomputed STP efficacy set by Brain.forward() before each region step.
        # When set, _integrate_synaptic_inputs_at_dendrites uses it instead of
        # calling individual STP.forward() — the batched kernel already ran.
        self._precomputed_stp_efficacy: Optional[Dict[SynapseId, torch.Tensor]] = None

        # Precomputed sparse conductances set by Brain.forward() for subclass
        # neuron populations (SerotoninNeuron, NorepinephrineNeuron, etc.)
        # whose inputs are handled by the global sparse matrix.
        # Maps pop_name → (g_ampa, g_nmda, g_gaba_a, g_gaba_b).
        self._precomputed_sparse_conductances: Optional[Dict[PopulationName, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None

        # Reference to the global sparse synaptic matrix (set by Brain.__init__
        # after construction).  Used by _apply_learning to dispatch sparse
        # weight updates for inter-region connections.
        self._sparse_matrix: Optional[Any] = None  # GlobalSparseMatrix, typed as Any to avoid circular import

        # Reverse index: target_population → list of SynapseIds targeting it.
        # Built incrementally by add_synapse(); used by _apply_synaptic_scaling
        # to avoid scanning all synaptic_weights entries every call.
        self._synapses_by_target_pop: Dict[PopulationName, List[SynapseId]] = {}

        # Step counter for synaptic scaling interval gating.
        self._synaptic_scaling_step: int = 0

        # Per-synapse learning strategies: registered as nn.Module so .to(device)
        # and state_dict() work automatically.
        self._learning_strategies: SynapseIdModuleDict = SynapseIdModuleDict()  # Dict[SynapseId, LearningStrategy]

        # Precomputed batched weight matrices for batched synaptic integration.
        # Built by build_batched_dendrite_weights() after all connections are added.
        # Maps target_population → BatchedDendriteWeights.
        self._batched_dendrite_weights: Optional[Dict[PopulationName, BatchedDendriteWeights]] = None

        # NOTE: Pre-allocated buffer caching by n_neurons was removed — it caused aliasing
        # when two populations share the same size (e.g., D1 and D2 both at 200 neurons).
        # The second call would clear the first call's result before it was consumed.

        # Previous region output for delayed spike access in the next step.  Updated at the end of forward().
        self._prev_region_output: Optional[RegionOutput] = None

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # NEURON POPULATION MANAGEMENT
    # =========================================================================

    def _create_and_register_neuron_population(
        self,
        population_name: PopulationName,
        n_neurons: int,
        polarity: PopulationPolarity,
        config: ConductanceLIFConfig,
    ) -> Union[ConductanceLIF, TwoCompartmentLIF]:
        """Convenience method to create a neuron population from config and register it."""
        if population_name in self.neuron_populations:
            raise ValueError(f"Population '{population_name}' already registered in {self.__class__.__name__}")

        # Infer neuron class from config type (ConductanceLIFConfig → ConductanceLIF, etc.)
        if isinstance(config, AcetylcholineNeuronConfig):
            neuron_cls = AcetylcholineNeuron
        elif isinstance(config, NorepinephrineNeuronConfig):
            neuron_cls = NorepinephrineNeuron
        elif isinstance(config, SerotoninNeuronConfig):
            neuron_cls = SerotoninNeuron
        elif isinstance(config, TwoCompartmentLIFConfig):
            neuron_cls = TwoCompartmentLIF
        else:
            assert isinstance(config, ConductanceLIFConfig)
            neuron_cls = ConductanceLIF

        # Create the neuron population
        population = neuron_cls(
            n_neurons=n_neurons,
            config=config,
            region_name=self.region_name,
            population_name=population_name,
            device=self.device,
        )

        # Register the population and its polarity for Dale's Law enforcement
        self.neuron_populations[population_name] = population
        self._population_polarities[population_name] = polarity

        # Auto-register homeostasis if a target rate is defined in config
        if population_name in self.config.homeostatic_target_rates:
            self._register_homeostasis(population_name)

        return population

    def get_neuron_population(self, population_name: PopulationName) -> Optional[Union[ConductanceLIF, TwoCompartmentLIF]]:
        """Get a registered neuron population by name."""
        population = self.neuron_populations[population_name] if population_name in self.neuron_populations else None
        assert population is None or isinstance(population, (ConductanceLIF, TwoCompartmentLIF)), (
            f"Registered population '{population_name}' is not a valid neuron type: {type(population)}"
        )
        return population

    def get_population_polarity(self, population_name: PopulationName) -> Optional[PopulationPolarity]:
        """Return the :class:`PopulationPolarity` registered for *population_name*.

        Returns :attr:`PopulationPolarity.ANY` for populations registered without
        an explicit polarity (e.g. legacy code or external inputs).
        """
        if population_name in self._population_polarities:
            return self._population_polarities[population_name]
        return None

    def get_population_size(self, population_name: PopulationName) -> int:
        """Get population size from population name."""
        population = self.get_neuron_population(population_name)
        if population is None:
            raise ValueError(
                f"Population '{population_name}' not found in {self.__class__.__name__}. "
                f"Registered populations: {self.neuron_populations.keys()}"
            )
        return population.n_neurons

    # =========================================================================
    # HOMEOSTATIC STATE REGISTRATION
    # =========================================================================

    def _register_homeostasis(
        self,
        population_name: PopulationName,
    ) -> None:
        """Register homeostatic state tracking for a neuron population.

        Looks up the target firing rate from ``self.config.homeostatic_target_rates``
        and derives population size from the already-registered neuron population.

        Must be called **after** ``_create_and_register_neuron_population()`` for
        this population.

        Buffer naming convention (using ``population_name`` directly)::

            {population_name}_firing_rate   # e.g. "dg_firing_rate"
        """
        if population_name not in self.neuron_populations:
            raise ValueError(
                f"Population '{population_name}' not registered in {self.__class__.__name__}. "
                f"Call _create_and_register_neuron_population() before registering homeostasis."
            )

        if population_name in self._homeostasis:
            raise ValueError(
                f"Homeostasis already registered for population '{population_name}' "
                f"in {self.__class__.__name__}."
            )

        target_rates = self.config.homeostatic_target_rates
        if population_name not in target_rates:
            raise ValueError(
                f"No homeostatic target rate for population '{population_name}' in "
                f"{self.config.__class__.__name__}.homeostatic_target_rates. "
                f"Available: {list(target_rates.keys())}"
            )

        n_neurons = self.get_population_size(population_name)
        fr_attr = f"{population_name}_firing_rate"
        self.register_buffer(fr_attr, torch.zeros(n_neurons, device=self.device))

        self._homeostasis[population_name] = PopulationHomeostasisState(
            population_name=population_name,
            firing_rate_attr=fr_attr,
            target_firing_rate=target_rates[population_name],
        )

    def _get_target_firing_rate(self, population_name: PopulationName) -> float:
        """Return the homeostatic target firing rate for a population."""
        return self._homeostasis[population_name].target_firing_rate

    def _apply_synaptic_scaling(self, population_name: PopulationName) -> None:
        """Apply bidirectional per-neuron multiplicative synaptic scaling (Turrigiano 2008).

        Each neuron independently scales ALL its incoming synaptic weights based on
        the deviation of its firing rate from target:

        - Underactive neurons (rate < target) → scale up incoming weights
        - Overactive neurons (rate > target)  → scale down incoming weights

        The scaling is multiplicative, preserving relative weight differences across
        synapses while stabilizing total drive.  This provides a second homeostatic
        axis complementary to intrinsic excitability (g_L) and threshold adaptation.

        Both dense (intra-region) and sparse (inter-region) weights are scaled.

        Reference: Turrigiano, G. G. (2008). The self-tuning neuron: synaptic scaling
        of excitatory synapses. *Cell*, 135(3), 422-435.

        Args:
            population_name: Key identifying the target population.
        """
        if self.config.homeostasis_disabled:
            return

        scaling_cfg = self.config.synaptic_scaling
        lr = scaling_cfg.lr_per_ms
        if lr == 0.0:
            return

        if population_name not in self._homeostasis:
            raise KeyError(
                f"No homeostasis registered for '{population_name}'. "
                f"Call _register_homeostasis() first."
            )

        state = self._homeostasis[population_name]
        firing_rate: torch.Tensor = getattr(self, state.firing_rate_attr)

        # Compensate learning rate for the interval: applying every N steps
        # with N× the rate is equivalent to applying every step.
        effective_lr = lr * scaling_cfg.interval_steps

        # Per-neuron scale factors: >1 when underactive, <1 when overactive
        scale_factors = 1.0 + effective_lr * (state.target_firing_rate - firing_rate)  # [n_neurons]

        w_min = scaling_cfg.w_min
        w_max = scaling_cfg.w_max

        # Scale dense weights using reverse index (only matching synapses)
        _weights_dict = self.synaptic_weights  # cache nn.ParameterDict lookup
        scale_unsqueezed = scale_factors.unsqueeze(1)
        for synapse_id in self._synapses_by_target_pop.get(population_name, ()):
            weights = _weights_dict[synapse_id]
            # weights.data is [n_post, n_pre]; scale each row by its neuron's factor
            weights.data.mul_(scale_unsqueezed)
            weights.data.clamp_(w_min, w_max)

        # Scale sparse weights (inter-region via GlobalSparseMatrix)
        sparse = self._sparse_matrix
        if sparse is not None:
            for synapse_id in sparse.get_synapse_ids_for_target(self.region_name, population_name):
                meta = sparse.connections[synapse_id]
                if meta.nnz > 0:
                    values = sparse.get_weight_values(synapse_id)
                    row_scales = scale_factors[meta.local_row_indices]
                    values.mul_(row_scales)
                    values.clamp_(w_min, w_max)
                    sparse.set_weight_values(synapse_id, values)

    def _update_homeostasis(self, population_name: PopulationName, spikes: torch.Tensor) -> None:
        """Update homeostatic intrinsic excitability and threshold adaptation.

        Biologically-accurate homeostasis through TWO mechanisms:

        1. **Intrinsic excitability** — modulate leak conductance (``g_L_scale``):
           lower g_L ↔ higher input resistance ↔ more excitable
           (Turrigiano & Nelson 2004).
        2. **Threshold adaptation** — lower spike threshold when underactive;
           complementary to intrinsic excitability with a faster time constant.

        Args:
            population_name: Key identifying the population.
            spikes: Boolean or float spike tensor for the current timestep
                ``[n_neurons]``.
        """
        if self.config.homeostasis_disabled:
            return

        if population_name not in self._homeostasis:
            raise KeyError(
                f"No homeostasis registered for '{population_name}' in "
                f"{self.__class__.__name__}. Call _register_homeostasis() first."
            )

        neurons = self.get_neuron_population(population_name)
        assert neurons is not None, (
            f"_update_homeostasis: neuron population '{population_name}' not found in "
            f"{self.__class__.__name__}.neuron_populations."
        )

        state = self._homeostasis[population_name]
        firing_rate: torch.Tensor = getattr(self, state.firing_rate_attr)

        # Update firing rate EMA with current spikes
        current_rate = spikes.float()
        firing_rate.mul_(1.0 - self._firing_rate_alpha).add_(current_rate * self._firing_rate_alpha)

        # Compute rate error: positive when underactive.
        pop_target = state.target_firing_rate
        rate_error = pop_target - firing_rate  # [n_neurons]

        # INTRINSIC EXCITABILITY: Modulate leak conductance
        # Underactive (rate_error > 0) → lower g_L_scale → lower leak → more excitable
        # Overactive  (rate_error < 0) → raise  g_L_scale → higher leak → less excitable
        g_L_update = -self.config.homeostatic_gain.lr_per_ms * rate_error
        neurons.g_L_scale.data.add_(g_L_update).clamp_(min=0.1, max=2.0)  # Biological range

        # THRESHOLD ADAPTATION: Lower threshold when underactive (faster than g_L)
        threshold_update = -self.config.homeostatic_threshold.lr_per_ms * rate_error
        neurons.adjust_thresholds(threshold_update, self.config.homeostatic_threshold.threshold_min, self.config.homeostatic_threshold.threshold_max)

    def _apply_all_population_homeostasis(self, region_outputs: RegionOutput) -> None:
        """Call homeostasis and synaptic scaling for every registered population.

        Regions that need non-standard homeostasis logic (e.g. Striatum joint D1/D2
        rate) can override this method or skip it entirely and call the per-population
        helpers directly.

        Args:
            region_outputs: Mapping from ``PopulationName`` to spike tensor produced
                in the current timestep.
        """
        # Firing-rate EMA and intrinsic excitability update every step
        for pop_name, _state in self._homeostasis.items():
            spikes = region_outputs.get(pop_name)
            if spikes is not None:
                self._update_homeostasis(pop_name, spikes)

        # Synaptic scaling runs every interval_steps (default 10)
        self._synaptic_scaling_step += 1
        if self._synaptic_scaling_step >= self.config.synaptic_scaling.interval_steps:
            self._synaptic_scaling_step = 0
            for pop_name in self._homeostasis:
                self._apply_synaptic_scaling(pop_name)

    # =========================================================================
    # SYNAPTIC INPUT MANAGEMENT
    # =========================================================================

    def get_synaptic_weights(self, synapse_id: SynapseId) -> nn.Parameter:
        """Get synaptic weights for a given input source, with validation."""
        if synapse_id not in self.synaptic_weights:
            raise ValueError(
                f"Synaptic weights for '{synapse_id}' not found in {self.__class__.__name__}. "
                f"Registered sources: {list(self.synaptic_weights.keys())}"
            )
        return self.synaptic_weights[synapse_id]

    def get_stp_module(self, synapse_id: SynapseId) -> Optional[ShortTermPlasticity]:
        """Get the ShortTermPlasticity module registered for a given synapse, or None if not present."""
        stp_module = self.stp_modules[synapse_id] if synapse_id in self.stp_modules else None
        if stp_module is not None:
            # Narrow the type from nn.Module → ShortTermPlasticity
            assert isinstance(stp_module, ShortTermPlasticity), f"Expected ShortTermPlasticity for '{synapse_id}', got {type(stp_module)}"
        return stp_module

    def get_learning_strategy(self, synapse_id: SynapseId) -> Optional[LearningStrategy]:
        """Return the :class:`~thalia.learning.LearningStrategy` for *synapse_id*, or ``None``."""
        strategy = self._learning_strategies[synapse_id] if synapse_id in self._learning_strategies else None
        if strategy is not None:
            # Narrow the type from nn.Module → LearningStrategy
            assert isinstance(strategy, LearningStrategy), f"Expected LearningStrategy for '{synapse_id}', got {type(strategy)}"
        return strategy

    def add_synapse(
        self,
        synapse_id: SynapseId,
        weights: torch.Tensor,
        stp_config: Optional[STPConfig],
        learning_strategy: Optional[LearningStrategy],
    ) -> None:
        """Add a new synaptic input source with weights, optional STP, and optional learning strategy.

        Enforces Dale's Law: the ``receptor_type`` in *synapse_id* must be consistent with the
        registered :class:`PopulationPolarity` of ``synapse_id.source_population``.  An
        ``EXCITATORY`` population cannot form inhibitory synapses and vice versa.

        Raises:
            ValueError: If Dale's Law is violated.
        """
        if synapse_id.target_region != self.region_name:
            raise ValueError(
                f"SynapseId target_region '{synapse_id.target_region}' does not match this region '{self.region_name}'."
            )

        # Enforce Dale's Law
        polarity = self._population_polarities.get(synapse_id.source_population, PopulationPolarity.ANY)
        if polarity == PopulationPolarity.EXCITATORY and synapse_id.receptor_type.is_inhibitory:
            raise ValueError(
                f"Dale's Law violation in region '{self.region_name}': "
                f"EXCITATORY population '{synapse_id.source_population}' cannot form an inhibitory "
                f"({synapse_id.receptor_type}) synapse onto '{synapse_id.target_population}'."
            )
        if polarity == PopulationPolarity.INHIBITORY and synapse_id.receptor_type.is_excitatory:
            raise ValueError(
                f"Dale's Law violation in region '{self.region_name}': "
                f"INHIBITORY population '{synapse_id.source_population}' cannot form an excitatory "
                f"({synapse_id.receptor_type}) synapse onto '{synapse_id.target_population}'."
            )

        device = weights.device
        n_output, n_input = weights.shape

        # Add synaptic weights
        if synapse_id in self.synaptic_weights:
            raise ValueError(f"Synaptic weights for '{synapse_id}' already exist in region '{self.region_name}'.")
        self.synaptic_weights[synapse_id] = nn.Parameter(weights, requires_grad=False)

        # Update reverse index: target_population → [synapse_ids]
        target_pop = synapse_id.target_population
        if target_pop not in self._synapses_by_target_pop:
            self._synapses_by_target_pop[target_pop] = []
        self._synapses_by_target_pop[target_pop].append(synapse_id)

        # Add STP module
        if stp_config is not None:
            if synapse_id in self.stp_modules:
                raise ValueError(f"STP module for '{synapse_id}' already exists in region '{self.region_name}'.")
            self.stp_modules[synapse_id] = ShortTermPlasticity(n_pre=n_input, config=stp_config, device=device)

        # Add learning strategy
        if learning_strategy is not None:
            self._add_learning_strategy(synapse_id, learning_strategy, n_pre=n_input, n_post=n_output, device=device)

    def add_input_source(
        self,
        synapse_id: SynapseId,
        n_input: int,
        connectivity: float,
        weight_scale: Union[float, ConductanceScaledSpec],
        *,
        stp_config: Optional[STPConfig],
        learning_strategy: Optional[LearningStrategy],
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> None:
        """Convenience method for adding a new synaptic input source with random sparse weights.

        Automatically initializes weights with the specified connectivity and weight scale.
        """
        if n_input < 0:
            raise ValueError("Number of input neurons must be non-negative")
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("Connectivity must be between 0 and 1")
        if isinstance(weight_scale, float) and weight_scale < 0:
            raise ValueError("Weight scale must be non-negative")

        # All registered inputs are synaptic (neuromodulators use separate broadcast system)
        n_output = self.get_population_size(synapse_id.target_population)

        if isinstance(weight_scale, ConductanceScaledSpec):
            spec = weight_scale
            weights = WeightInitializer.conductance_scaled(
                n_input=n_input,
                n_output=n_output,
                connectivity=connectivity,
                source_rate_hz=spec.source_rate_hz,
                target_g_L=spec.target_g_L,
                target_E_E=spec.target_E_E,
                target_tau_E_ms=spec.target_tau_E_ms,
                target_v_inf=spec.target_v_inf,
                fraction_of_drive=spec.fraction_of_drive,
                stp_utilization_factor=spec.stp_utilization_factor,
                inhibitory_load=spec.inhibitory_load,
                E_I=spec.E_I,
                device=device,
            )
        else:
            weights = WeightInitializer.sparse_random(
                n_input=n_input,
                n_output=n_output,
                connectivity=connectivity,
                weight_scale=weight_scale,
                device=device,
            )

        self.add_synapse(
            synapse_id=synapse_id,
            weights=weights,
            stp_config=stp_config,
            learning_strategy=learning_strategy,
        )

    def _add_internal_connection(
        self,
        source_population: PopulationName,
        target_population: PopulationName,
        weights: torch.Tensor,
        *,
        receptor_type: ReceptorType,
        stp_config: Optional[STPConfig],
        learning_strategy: Optional[LearningStrategy] = None,
    ) -> SynapseId:
        """Convenience method for adding synaptic connections between internal populations.

        Automatically constructs the appropriate SynapseId with source and target regions
        set to self.region_name, and the provided population names and receptor type.

        Returns:
            The SynapseId of the newly added internal connection, which can be used for
            later reference (e.g. applying learning updates).
        """
        synapse_id = SynapseId(
            source_region=self.region_name,
            source_population=source_population,
            target_region=self.region_name,
            target_population=target_population,
            receptor_type=receptor_type,
        )

        self.add_synapse(
            synapse_id=synapse_id,
            weights=weights,
            stp_config=stp_config,
            learning_strategy=learning_strategy,
        )

        return synapse_id

    def _add_feedforward_inhibition(
        self,
        exc_pop: PopulationName,
        inh_pop: PopulationName,
        e_to_i: InternalConnectionSpec,
        i_to_e: InternalConnectionSpec,
        device: Union[str, torch.device],
    ) -> tuple[SynapseId, SynapseId]:
        """Add a paired E→I + I→E feedforward inhibition circuit.

        Initializes both connections with sparse random weights using the
        population sizes already registered on this region.  Returns the
        SynapseId pair for later reference (e.g. diagnostics or learning).

        This helper covers the common case where both legs use
        :func:`WeightInitializer.sparse_random`.  Connections that need
        gaussian or uniform initialisation should call
        :meth:`_add_internal_connection` directly.
        """
        n_exc = self.get_population_size(exc_pop)
        n_inh = self.get_population_size(inh_pop)

        ei_syn = self._add_internal_connection(
            source_population=exc_pop,
            target_population=inh_pop,
            weights=WeightInitializer.sparse_random(
                n_input=n_exc,
                n_output=n_inh,
                connectivity=e_to_i.connectivity,
                weight_scale=e_to_i.weight_scale,
                device=device,
            ),
            receptor_type=e_to_i.receptor_type,
            stp_config=e_to_i.stp_config,
            learning_strategy=e_to_i.learning_strategy,
        )

        ie_syn = self._add_internal_connection(
            source_population=inh_pop,
            target_population=exc_pop,
            weights=WeightInitializer.sparse_random(
                n_input=n_inh,
                n_output=n_exc,
                connectivity=i_to_e.connectivity,
                weight_scale=i_to_e.weight_scale,
                device=device,
            ),
            receptor_type=i_to_e.receptor_type,
            stp_config=i_to_e.stp_config,
            learning_strategy=i_to_e.learning_strategy,
        )

        return ei_syn, ie_syn

    # =========================================================================
    # BATCHED DENDRITE WEIGHT PRECOMPUTATION
    # =========================================================================

    def build_batched_dendrite_weights(
        self,
        exclude_synapse_ids: frozenset[SynapseId] = frozenset(),
    ) -> None:
        """Precompute concatenated weight matrices for batched synaptic integration.

        For each target population, groups all registered synaptic weights by
        receptor type and horizontally concatenates them into one weight matrix.
        Individual synaptic_weights entries are then replaced with views into
        the concatenated matrix so that learning rule updates propagate
        automatically (shared storage).

        This enables ``_integrate_synaptic_inputs_at_dendrites`` to replace N
        small matmuls with 1 larger matmul per receptor type when filtering by
        target population.

        Connections in *exclude_synapse_ids* (typically those handled by the
        global sparse matrix) are skipped.

        Must be called after all ``add_input_source()`` / ``add_synapse()``
        calls are complete (typically in ``Brain.__init__``).
        """
        device = self.device

        # Group all synapse_ids by target_population
        pop_synapses: Dict[PopulationName, list[SynapseId]] = {}
        for synapse_id in self.synaptic_weights:
            if synapse_id in exclude_synapse_ids:
                continue
            pop_synapses.setdefault(synapse_id.target_population, []).append(synapse_id)

        batched: Dict[PopulationName, BatchedDendriteWeights] = {}

        for target_pop, synapse_ids in pop_synapses.items():
            if len(synapse_ids) < 2:
                continue  # Single synapse — no batching benefit

            n_target = self.synaptic_weights[synapse_ids[0]].shape[0]

            # Group by receptor type
            receptor_groups: Dict[ReceptorType, list[SynapseId]] = {}
            for sid in synapse_ids:
                receptor_groups.setdefault(sid.receptor_type, []).append(sid)

            blocks: Dict[ReceptorType, ConcatWeightBlock] = {}

            for receptor_type, sids in receptor_groups.items():
                # Sort for deterministic column ordering
                sids.sort(key=lambda s: s.to_key())

                # Collect weight data and compute column slices
                weight_list: list[torch.Tensor] = []
                column_slices: list[slice] = []
                offset = 0
                for sid in sids:
                    w = self.synaptic_weights[sid]
                    n_cols = w.shape[1]
                    weight_list.append(w.data)
                    column_slices.append(slice(offset, offset + n_cols))
                    offset += n_cols

                total_sources = offset

                # Concatenate weight matrices horizontally: [n_target, total_sources]
                W_concat = torch.cat(weight_list, dim=1)

                # Replace individual parameters with views into W_concat
                # so learning rule updates modify W_concat in-place.
                for sid, col_slice in zip(sids, column_slices):
                    self.synaptic_weights[sid] = nn.Parameter(
                        W_concat[:, col_slice], requires_grad=False
                    )

                blocks[receptor_type] = ConcatWeightBlock(
                    W_concat=W_concat,
                    synapse_ids=tuple(sids),
                    column_slices=tuple(column_slices),
                    total_sources=total_sources,
                    device=device,
                )

            batched[target_pop] = BatchedDendriteWeights(
                n_target=n_target,
                blocks=blocks,
            )

        self._batched_dendrite_weights = batched if batched else None

    # =========================================================================
    # LEARNING STRATEGY MANAGEMENT
    # =========================================================================

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        """Return extra keyword arguments for the learning strategy of *synapse_id*.

        Override in subclasses to supply neuromodulator concentrations or
        region-specific learning-rate multipliers.

        The base implementation returns an empty dict, which is correct for
        plain STDP with no neuromodulation.

        Example override (dopamine + ACh gate)::

            def _get_learning_kwargs(self, synapse_id):
                return {
                    "dopamine": self._da_concentration.mean().item(),
                    "acetylcholine": 1.0 - self._ach_concentration.mean().item(),
                }
        """
        return {}

    def _add_learning_strategy(
        self,
        synapse_id: SynapseId,
        strategy: LearningStrategy,
        n_pre: Optional[int] = None,
        n_post: Optional[int] = None,
        *,
        device: Union[str, torch.device],
    ) -> None:
        """Register *strategy* for *synapse_id*, optionally calling setup() immediately.

        If *n_pre* and *n_post* are both provided the strategy is set up eagerly
        (``strategy.setup(n_pre, n_post, device)``).  Otherwise setup is
        deferred until the first call to :meth:`_apply_learning`.
        """
        self._learning_strategies[synapse_id] = strategy
        if n_pre is not None and n_post is not None:
            strategy.setup(n_pre, n_post, device)

    def _apply_learning(
        self,
        synapse_id: SynapseId,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Look up the strategy for *synapse_id* and apply it, clamping weights in-place.

        No-ops when:
        - learning is disabled on this region's config
        - no strategy is registered for *synapse_id*

        Weight clamping uses ``self.config.synaptic_scaling.w_min`` / ``self.config.synaptic_scaling.w_max``.
        """
        if self.config.learning_disabled:
            return

        # Cache nn.Module submodule dicts once — avoids __getattr__ per access
        _learning_strategies = self._learning_strategies
        strategy = _learning_strategies[synapse_id] if synapse_id in _learning_strategies else None
        if strategy is None:
            return

        # Spike tensors arrive as bool from neuron models but learning rules
        # perform arithmetic (mean, outer product, subtraction).  Convert once
        # here instead of in every strategy.
        pre_spikes = pre_spikes.float()
        post_spikes = post_spikes.float()

        # ── Sparse path: connection managed by global sparse matrix ──
        sparse = self._sparse_matrix
        if sparse is not None and sparse.has_connection(synapse_id):
            meta = sparse.get_connection_meta(synapse_id)
            if meta.nnz == 0:
                return
            values = sparse.get_weight_values(synapse_id)
            updated = strategy.compute_update_sparse(
                values=values,
                row_indices=meta.local_row_indices,
                col_indices=meta.local_col_indices,
                n_post=meta.n_post,
                n_pre=meta.n_pre,
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                **kwargs,
            )
            clamp_weights(updated, self.config.synaptic_scaling.w_min, self.config.synaptic_scaling.w_max)
            # No connectivity mask needed — sparse entries ARE valid connections.
            sparse.set_weight_values(synapse_id, updated)
            return

        # ── Dense path: original per-connection weight matrix ──
        weights = self.synaptic_weights[synapse_id]
        # Capture connectivity mask BEFORE the update so that only
        # pre-existing synapses (weight > 0 at initialisation or after LTP)
        # can be modified.  Hebbian/STDP outer-products produce dense
        # [n_post × n_pre] matrices; without this mask they would create
        # de-novo connections at anatomically absent entries, violating
        # the sparse connectivity layout set at build time and causing the
        # same runaway potentiation seen in hippocampus (Fix 2, 2025-07).
        syn_mask = weights.data > 0.0
        updated = strategy.compute_update(
            weights=weights.data,
            pre_spikes=pre_spikes,
            post_spikes=post_spikes,
            **kwargs,
        )
        clamp_weights(updated, self.config.synaptic_scaling.w_min, self.config.synaptic_scaling.w_max)
        # Zero any newly-created entries; anatomically absent synapses stay absent.
        updated.mul_(syn_mask.float())
        weights.data = updated

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @abstractmethod
    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Perform a single simulation step for the region.

        Args:
            synaptic_inputs: Point-to-point synaptic connections from axonal tracts.
            neuromodulator_inputs: Broadcast neuromodulatory signals.
        """

    @torch.no_grad()
    def forward(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Process synaptic and neuromodulatory inputs through the region and produce outputs.

        Args:
            synaptic_inputs: Point-to-point synaptic connections
            neuromodulator_inputs: Broadcast neuromodulatory signals;
                every published channel is always a tensor — zero tensor when the source was silent.

        Returns:
            RegionOutput: Dict mapping population names to their output spike tensors
        """
        if GlobalConfig.DEBUG:
            for synapse_id, spikes in synaptic_inputs.items():
                validate_spike_tensor(spikes, tensor_name=f"synaptic_input_{synapse_id}")

                if synapse_id.target_region != self.region_name:
                    raise ValueError(
                        f"Synaptic input '{synapse_id}' has target_region '{synapse_id.target_region}' "
                        f"which does not match this region '{self.region_name}'."
                    )

                if synapse_id not in self.synaptic_weights:
                    raise ValueError(
                        f"Synaptic weights for input source '{synapse_id}' not found in {self.__class__.__name__}. "
                        f"Registered sources: {list(self.synaptic_weights.keys())}"
                    )

            for neuromod_type, spikes in neuromodulator_inputs.items():
                validate_spike_tensor(spikes, tensor_name=f"neuromodulator_{neuromod_type}")

        region_outputs: RegionOutput = self._step(synaptic_inputs, neuromodulator_inputs)
        self._prev_region_output = region_outputs

        if GlobalConfig.DEBUG:
            for population_name in self.neuron_populations.keys():
                if population_name not in region_outputs:
                    raise ValueError(
                        f"Output for population '{population_name}' not found in region_outputs. "
                        f"Expected keys: {list(region_outputs.keys())}"
                    )
                output_spikes = region_outputs[population_name]
                validate_spike_tensor(output_spikes, tensor_name=f"output_{population_name}")

        return region_outputs

    def apply_learning(
        self,
        synaptic_inputs: SynapticInput,
        region_outputs: RegionOutput,
    ) -> None:
        """Apply inter-region plasticity rules.

        Called by :meth:`forward` after ``_step()`` (and, in batched mode, after
        ``ConductanceLIFBatch.step()`` so that spike tensors are up-to-date).
        """
        if self.config.learning_disabled:
            return

        for synapse_id, pre_spikes in synaptic_inputs.items():
            post_spikes = region_outputs.get(synapse_id.target_population)
            if post_spikes is not None:
                learning_kwargs = self._get_learning_kwargs(synapse_id)
                self._apply_learning(synapse_id, pre_spikes, post_spikes, **learning_kwargs)

    def _prev_spikes(
        self,
        population: PopulationName,
    ) -> torch.Tensor:
        """Extract previous-step spikes for a population from _prev_region_output.

        Returns a zero tensor (all-silent) when no previous output is available
        (first timestep).
        """
        if self._prev_region_output is not None and population in self._prev_region_output:
            return self._prev_region_output[population].float()
        size = self.get_population_size(population)
        return torch.zeros(size, dtype=torch.float32, device=self.device)

    def _extract_neuromodulator(
        self,
        neuromodulator_inputs: NeuromodulatorInput,
        *channel_names: NeuromodulatorChannel,
    ) -> Optional[torch.Tensor]:
        """Return the tensor for the first channel key present in *neuromodulator_inputs*.

        Channels are tried in the order they are listed.  Returns ``None`` only when
        none of the named keys exist in the dict at all (i.e., no region in the
        current brain publishes that channel — caught at build time by
        ``NeuromodulatorHub.validate()``).

        Because ``NeuromodulatorHub`` now always emits a zero tensor for silent
        channels, a non-``None`` return is also guaranteed for every subscribed
        channel.  ``NeuromodulatorReceptor.update()`` handles zero tensors
        correctly (immediate early-return / decay-only path).

        Prefer passing a **single** channel name that exactly matches the published
        channel (declared in the region's ``neuromodulator_subscriptions``).
        Multi-key fallback is for forward-compat with brains that publish either
        of two channel variants.

        Example::

            da = self._extract_neuromodulator(nm, NeuromodulatorChannel.DA_MESOCORTICAL)

        Args:
            neuromodulator_inputs: The neuromodulator dict passed to ``forward()``.
            *channel_names: One or more keys to try in order.
        """
        for key in channel_names:
            if key in neuromodulator_inputs:
                return neuromodulator_inputs[key]
        return None

    def _init_receptors(
        self,
        specs: list[ReceptorSpec],
        device: Union[str, torch.device],
    ) -> None:
        """Create neuromodulator receptors and register concentration buffers.

        Call once in ``__init__`` to replace per-receptor boilerplate
        (``make_neuromodulator_receptor`` + ``register_buffer``).
        Use :meth:`_update_receptors` in ``_step`` to batch-update.
        """
        receptors = []
        for spec in specs:
            receptor = make_neuromodulator_receptor(
                spec.receptor_type,
                n_receptors=spec.n_receptors,
                dt_ms=self.dt_ms,
                device=device,
                amplitude_scale=spec.amplitude_scale,
            )
            receptors.append(receptor)
            if spec.initial_value != 0.0:
                buf = torch.full((spec.n_receptors,), spec.initial_value, device=torch.device(device))
            else:
                buf = torch.zeros(spec.n_receptors, device=torch.device(device))
            self.register_buffer(spec.buffer_name, buf)
        self._nm_receptor_modules = nn.ModuleList(receptors)
        self._nm_receptor_specs: list[ReceptorSpec] = specs

    def _init_receptors_from_config(self, device: Union[str, torch.device]) -> None:
        """Create neuromodulator receptors from ``self.config.neuromodulator_receptors``.

        Resolves *n_receptors* for each :class:`NMReceptorConfig` by summing the
        neuron counts of the listed populations (empty tuple → ``n_receptors=1``).
        Delegates to :meth:`_init_receptors` after building the runtime specs.
        """
        if not self.config.neuromodulator_receptors:
            return
        specs: list[ReceptorSpec] = []
        for rc in self.config.neuromodulator_receptors:
            n: int
            if rc.populations:
                n = sum(self.neuron_populations[pop].n_neurons for pop in rc.populations)
            else:
                n = 1
            specs.append(ReceptorSpec(
                receptor_type=rc.receptor_type,
                channel=rc.channel,
                n_receptors=n,
                buffer_name=rc.buffer_name,
                amplitude_scale=rc.amplitude_scale,
                initial_value=rc.initial_value,
            ))
        self._init_receptors(specs, device)

    def _update_receptors(self, nm_inputs: NeuromodulatorInput) -> None:
        """Extract spikes and update all declaratively registered receptors.

        Each unique channel is extracted once; the resulting concentration
        tensor is written to the corresponding registered buffer.
        """
        if self.config.neuromodulation_disabled:
            return

        spike_cache: Dict[NeuromodulatorChannel, Optional[torch.Tensor]] = {}
        for receptor, spec in zip(self._nm_receptor_modules, self._nm_receptor_specs):
            if spec.channel not in spike_cache:
                spike_cache[spec.channel] = self._extract_neuromodulator(
                    nm_inputs, spec.channel
                )
            setattr(self, spec.buffer_name, receptor.update(spike_cache[spec.channel]))

    def _integrate_synaptic_inputs_at_dendrites(
        self,
        synaptic_inputs: SynapticInput,
        n_neurons: int,
        *,
        filter_by_source_region: Optional[RegionName] = None,
        filter_by_target_population: Optional[PopulationName] = None,
    ) -> DendriteOutput:
        """Integrate synaptic inputs at dendrites (multi-source summation).

        Biological Process:
        1. Spikes arrive via axons (already delayed in AxonalTract)
        2. Synapses convert spikes to conductances: g = weights @ spikes
        3. Dendrites sum conductances from all sources: g_total = Σ g_i
        4. Soma integrates total conductance: neurons.forward(g_exc, g_inh)

        Excitatory and inhibitory conductances are accumulated separately so
        callers can pass them to the correct neuron channels (AMPA/NMDA vs GABA_A).

        When precomputed batched weights exist for the requested target population
        (built by ``build_batched_dendrite_weights``), a fast path replaces the
        per-source Python loop with one matmul per active receptor type.

        Args:
            synaptic_inputs: Dict mapping synapse IDs to spike tensors [n_source]
            n_neurons: Number of post-synaptic neurons (target population size)
            filter_by_source_region: If provided, only integrate inputs from this source region (e.g., "thalamus")
            filter_by_target_population: If provided, only integrate inputs targeting this population (e.g., "l4")

        Returns:
            DendriteOutput(g_ampa, g_nmda, g_gaba_a, g_gaba_b): All tensors are
            [n_neurons], non-negative conductances, separated by receptor type.

        Raises:
            AssertionError: If input tensors violate ADR-005 (not 1D)

        Note:
            Routing keys are used directly as synaptic weight keys.
            Region subclasses should initialize synaptic_weights with routing keys.
        """
        # ---- Sparse matrix bypass (Step 5) ----
        # When the global sparse matrix already computed conductances for this
        # target population (subclass neurons like SerotoninNeuron, NE, ACh),
        # return them directly — skip all per-source integration.
        if (
            filter_by_target_population is not None
            and self._precomputed_sparse_conductances is not None
            and filter_by_target_population in self._precomputed_sparse_conductances
        ):
            g_ampa, g_nmda, g_gaba_a, g_gaba_b = self._precomputed_sparse_conductances[filter_by_target_population]
            return DendriteOutput(g_ampa=g_ampa, g_nmda=g_nmda, g_gaba_a=g_gaba_a, g_gaba_b=g_gaba_b)

        # ---- Batched fast path ----
        # When filtering by target population with no other filters, and batched
        # weights have been precomputed, replace per-source loop with one matmul
        # per receptor type.
        if (
            filter_by_target_population is not None
            and filter_by_source_region is None
            and self._batched_dendrite_weights is not None
            and filter_by_target_population in self._batched_dendrite_weights
        ):
            return self._integrate_dendrites_batched(
                synaptic_inputs, filter_by_target_population, n_neurons
            )

        # ---- Original per-source loop path ----
        device = self.device

        # Cache module dicts once — avoids nn.Module.__getattr__ on every loop iteration
        _weights_dict = self.synaptic_weights
        _stp_dict     = self.stp_modules

        # Fresh tensors each call: buffer caching was removed because sharing the same
        # tensor across calls with equal n_neurons caused aliasing (the second call clears
        # the first call's result before it is consumed by the caller).
        g_ampa   = torch.zeros(n_neurons, device=device)
        g_nmda   = torch.zeros(n_neurons, device=device)
        g_gaba_a = torch.zeros(n_neurons, device=device)
        g_gaba_b = torch.zeros(n_neurons, device=device)

        for synapse_id, source_spikes in synaptic_inputs.items():
            # Apply filters to select which inputs to integrate
            if filter_by_source_region and synapse_id.source_region != filter_by_source_region:
                continue
            if filter_by_target_population and synapse_id.target_population != filter_by_target_population:
                continue

            # Validate that synaptic weights exist for this source
            if synapse_id not in _weights_dict:
                raise ValueError(
                    f"Synaptic weights for input source '{synapse_id}' not found in {self.__class__.__name__}. "
                    f"Registered sources: {list(_weights_dict.keys())}"
                )

            weights = _weights_dict[synapse_id]
            source_spikes_f = source_spikes.float()

            # OPTIONAL PER-SOURCE STP (SHORT-TERM PLASTICITY)
            # Models synaptic facilitation/depression (millisecond timescale)
            # Different sources can have different STP dynamics
            if synapse_id in _stp_dict:
                # Use precomputed efficacy from batched STP kernel if available;
                # otherwise fall back to per-module forward() call.
                _pre_eff = self._precomputed_stp_efficacy
                if _pre_eff is not None and synapse_id in _pre_eff:
                    stp_efficacy = _pre_eff[synapse_id]
                else:
                    stp_efficacy = _stp_dict[synapse_id].forward(source_spikes_f)
                source_conductance = weights @ (stp_efficacy * source_spikes_f)
            else:
                # SYNAPTIC CONDUCTANCE CALCULATION
                # Convert incoming spikes to conductance: g = W @ s
                source_conductance = weights @ source_spikes_f

            # Route by receptor type — four separate channels
            match synapse_id.receptor_type:
                case ReceptorType.AMPA:
                    g_ampa += source_conductance
                case ReceptorType.NMDA:
                    g_nmda += source_conductance
                case ReceptorType.GABA_A:
                    g_gaba_a += source_conductance
                case ReceptorType.GABA_B:
                    g_gaba_b += source_conductance

        # Clamp to non-negative (conductances cannot be negative!)
        g_ampa.clamp_(min=0.0)
        g_nmda.clamp_(min=0.0)
        g_gaba_a.clamp_(min=0.0)
        g_gaba_b.clamp_(min=0.0)
        return DendriteOutput(g_ampa=g_ampa, g_nmda=g_nmda, g_gaba_a=g_gaba_a, g_gaba_b=g_gaba_b)

    def _integrate_single_synaptic_input(
        self,
        synapse_id: SynapseId,
        source_spikes: torch.Tensor,
    ) -> DendriteOutput:
        """Integrate a single synaptic input source at the dendrites.

        Fast path that avoids the overhead of the general multi-source method:
        - Only allocates/clamps the one conductor matching the receptor type
        - Skips validation (done in forward() when DEBUG=True)
        - Accepts pre-converted float spikes to avoid redundant .float() calls
        """
        # Cache module dicts once — avoids nn.Module.__getattr__ per access
        _weights_dict = self.synaptic_weights
        _stp_dict     = self.stp_modules
        weights = _weights_dict[synapse_id]
        source_spikes_f = source_spikes.float()

        # Optional STP (short-term plasticity)
        stp = _stp_dict.get(synapse_id)
        if stp is not None:
            _pre_eff = self._precomputed_stp_efficacy
            if _pre_eff is not None and synapse_id in _pre_eff:
                efficacy = _pre_eff[synapse_id]
            else:
                efficacy = stp.forward(source_spikes_f)
            g = weights @ (efficacy * source_spikes_f)
        else:
            g = weights @ source_spikes_f

        g.clamp_(min=0.0)

        # Build DendriteOutput with only the active channel populated;
        # unused slots share a module-level scalar zero (callers never read them).
        _z = _DENDRITE_ZERO
        match synapse_id.receptor_type:
            case ReceptorType.AMPA:
                return DendriteOutput(g_ampa=g, g_nmda=_z, g_gaba_a=_z, g_gaba_b=_z)
            case ReceptorType.NMDA:
                return DendriteOutput(g_ampa=_z, g_nmda=g, g_gaba_a=_z, g_gaba_b=_z)
            case ReceptorType.GABA_A:
                return DendriteOutput(g_ampa=_z, g_nmda=_z, g_gaba_a=g, g_gaba_b=_z)
            case ReceptorType.GABA_B:
                return DendriteOutput(g_ampa=_z, g_nmda=_z, g_gaba_a=_z, g_gaba_b=g)

    def _integrate_dendrites_batched(
        self,
        synaptic_inputs: SynapticInput,
        target_pop: PopulationName,
        n_neurons: int,
    ) -> DendriteOutput:
        """Batched synaptic integration for one target population.

        Uses precomputed concatenated weight matrices (from
        ``build_batched_dendrite_weights``) to replace per-source matmuls with
        one matmul per active receptor type.  Spike gathering is still a Python
        loop but operates only on the sources that contribute to this population,
        not the entire synaptic_inputs dict.

        Args:
            synaptic_inputs: Dict mapping synapse IDs to spike tensors.
            target_pop: Target population name (must be in _batched_dendrite_weights).
            n_neurons: Expected target population size (for zero-fill).
        """
        assert self._batched_dendrite_weights is not None  # guaranteed by caller
        batch: BatchedDendriteWeights = self._batched_dendrite_weights[target_pop]
        device = self.device
        _pre_eff = self._precomputed_stp_efficacy
        _stp_dict = self.stp_modules

        g_ampa: Optional[torch.Tensor] = None
        g_nmda: Optional[torch.Tensor] = None
        g_gaba_a: Optional[torch.Tensor] = None
        g_gaba_b: Optional[torch.Tensor] = None

        for receptor_type in (ReceptorType.AMPA, ReceptorType.NMDA,
                              ReceptorType.GABA_A, ReceptorType.GABA_B):
            block = batch.get_block(receptor_type)
            if block is None:
                continue

            # Re-use pre-allocated spike buffer (zeroed each call)
            buf = block.spike_buffer
            buf.zero_()
            any_spikes = False

            for i, sid in enumerate(block.synapse_ids):
                if sid not in synaptic_inputs:
                    continue

                spikes = synaptic_inputs[sid]
                spikes_f = spikes.float()

                # Apply STP efficacy (precomputed by batched kernel or fallback)
                if sid in _stp_dict:
                    if _pre_eff is not None and sid in _pre_eff:
                        spikes_f = _pre_eff[sid] * spikes_f
                    else:
                        spikes_f = _stp_dict[sid].forward(spikes_f) * spikes_f

                buf[block.column_slices[i]] = spikes_f
                any_spikes = True

            if any_spikes:
                g = block.W_concat @ buf
                g.clamp_(min=0.0)
            else:
                g = torch.zeros(n_neurons, device=device)

            match receptor_type:
                case ReceptorType.AMPA:
                    g_ampa = g
                case ReceptorType.NMDA:
                    g_nmda = g
                case ReceptorType.GABA_A:
                    g_gaba_a = g
                case ReceptorType.GABA_B:
                    g_gaba_b = g

        z = torch.zeros(n_neurons, device=device)
        return DendriteOutput(
            g_ampa=g_ampa if g_ampa is not None else z,
            g_nmda=g_nmda if g_nmda is not None else z,
            g_gaba_a=g_gaba_a if g_gaba_a is not None else z,
            g_gaba_b=g_gaba_b if g_gaba_b is not None else z,
        )

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters for neurons and learning strategies.

        Args:
            dt_ms: New timestep in milliseconds
        """
        self.config.dt_ms = dt_ms

        self._firing_rate_alpha = dt_ms / self.config.homeostatic_gain.tau_ms

        for neurons in self.neuron_populations.values():
            neurons.update_temporal_parameters(dt_ms)

        for stp_module in self.stp_modules.values():
            stp_module.update_temporal_parameters(dt_ms)

        # Update per-synapse strategies; deduplicate by id() so shared strategies
        # (e.g. one CompositeStrategy registered for multiple SynapseIds) are only
        # updated once.
        seen: set[int] = set()
        for strategy in self._learning_strategies.values():
            sid = id(strategy)
            if sid not in seen:
                seen.add(sid)
                strategy.update_temporal_parameters(dt_ms)
