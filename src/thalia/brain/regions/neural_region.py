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
from typing import Any, Callable, ClassVar, Dict, Generic, List, NamedTuple, Optional, TypeVar, Union

import torch
import torch.nn as nn

from thalia import GlobalConfig
from thalia.brain.configs import NeuralRegionConfig
from thalia.brain.neurons import (
    ConductanceLIF,
    TwoCompartmentLIF,
)
from thalia.brain.synapses import (
    ConductanceScaledSpec,
    STPConfig,
    ShortTermPlasticity,
    WeightInitializer,
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
        weight_scale_attr: Attribute name of the synaptic weight-scale buffer, or
            ``None`` when synaptic scaling is not enabled for this population.
    """

    population_name: PopulationName
    firing_rate_attr: str
    target_firing_rate: float
    weight_scale_attr: Optional[str] = None


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


class SynapseShape(NamedTuple):
    """Convenience struct for synapse shape metadata.

    Attributes:
        n_output: Number of post-synaptic neurons (output dimension).
        n_input: Number of pre-synaptic neurons (input dimension).
    """

    n_output: int
    n_input: int


class SynapseInfo(NamedTuple):
    """Convenience struct for synapse metadata and parameters.

    Attributes:
        synapse_id: The SynapseId that indexes this synapse's weights and modules.
        weights: The synaptic weight matrix [n_post, n_pre].
        stp_module: Optional ShortTermPlasticity module for this synapse.
        learning_strategy: Optional LearningStrategy module for this synapse.
    """

    synapse_id: SynapseId
    weights: nn.Parameter
    stp_module: Optional[ShortTermPlasticity] = None
    learning_strategy: Optional[LearningStrategy] = None


class NeuralRegion(nn.Module, ABC, Generic[ConfigT]):
    """Base class for brain regions with biologically accurate synaptic inputs.

    Regions:
    1. Own their synaptic weights (one weight matrix per input source)
    2. Define learning rules (per-source plasticity)
    3. Integrate multi-source inputs naturally

    Subclassing:
        Regions with internal structure should:
        1. Call super().__init__() to get synaptic_weights dict
        2. Define internal neurons/weights for within-region processing
        3. Override forward() to apply synaptic weights then internal processing
    """

    config: ConfigT  # Type annotation for config

    # Declared by subclasses that source neuromodulator volume-transmission signals.
    # Inherited from NeuromodulatorSource protocol check.
    # neuromodulator_outputs: ClassVar[Dict[NeuromodulatorChannel, PopulationName]]
    # (defined on source regions only — not all regions produce neuromodulators)
    #
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

        # Neuron populations within this region.
        # PopulationName is str so nn.ModuleDict works directly; subclasses keep
        # their own typed references (self.l23, self.l4, …) for forward() calls.
        self.neuron_populations: nn.ModuleDict = nn.ModuleDict()  # Dict[PopulationName, ConductanceLIF | TwoCompartmentLIF]

        # Per-population polarity (EXCITATORY / INHIBITORY / ANY).
        # Populated by _register_neuron_population(); used for Dale's Law enforcement.
        self._population_polarities: Dict[PopulationName, PopulationPolarity] = {}

        # Synaptic weights: one weight matrix per input source.
        # SynapseIdParameterDict wraps nn.ParameterDict so PyTorch tracks every
        # parameter correctly (.to(), .parameters(), .state_dict()) while the
        # public API remains SynapseId-typed.
        self.synaptic_weights: SynapseIdParameterDict = SynapseIdParameterDict()  # Dict[SynapseId, nn.Parameter]

        # Optional per-source STP modules for short-term plasticity (facilitation/depression)
        self.stp_modules: SynapseIdModuleDict = SynapseIdModuleDict()  # Dict[SynapseId, ShortTermPlasticity]

        # Per-synapse learning strategies: registered as nn.Module so .to(device)
        # and state_dict() work automatically.
        self._learning_strategies: SynapseIdModuleDict = SynapseIdModuleDict()  # Dict[SynapseId, LearningStrategy]

        # Per-population homeostatic state registry.
        # Populated by _register_homeostasis(); consumed by _update_homeostasis()
        # and _apply_synaptic_scaling().
        self._homeostasis: dict[PopulationName, PopulationHomeostasisState] = {}

        # EMA alpha for firing rate tracking
        self._firing_rate_alpha: float = self.dt_ms / self.config.gain_tau_ms

        # NOTE: Pre-allocated buffer caching by n_neurons was removed — it caused aliasing
        # when two populations share the same size (e.g., D1 and D2 both at 200 neurons).
        # The second call would clear the first call's result before it was consumed.

        # -----------------------------------------------------------------------
        # LEARNING CONTROL
        # -----------------------------------------------------------------------
        # Synaptic inputs from the current forward() call; stored by _pre_forward
        # so that _post_forward can drive _apply_all_learning automatically.
        self._last_synaptic_inputs: SynapticInput = {}

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # NEURON POPULATION MANAGEMENT
    # =========================================================================

    def _register_neuron_population(
        self,
        population_name: PopulationName,
        population: Union[ConductanceLIF, TwoCompartmentLIF],
        polarity: PopulationPolarity = PopulationPolarity.ANY,
    ) -> None:
        """Register a neuron population within this region.

        Args:
            population_name: Unique name for the population within this region.
            population: The neuron group — either a :class:`ConductanceLIF` (single
                compartment) or a :class:`TwoCompartmentLIF` (soma + apical dendrite).
            polarity: Biological polarity of this population per Dale's Law.
                ``EXCITATORY`` populations may only form excitatory synapses;
                ``INHIBITORY`` populations may only form inhibitory synapses;
                ``ANY`` disables the polarity check (use for external inputs).
        """
        if population_name in self.neuron_populations:
            raise ValueError(f"Population '{population_name}' already registered in {self.__class__.__name__}")
        self.neuron_populations[population_name] = population
        self._population_polarities[population_name] = polarity

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
    # SYNAPTIC WEIGHT MANAGEMENT
    # =========================================================================

    def get_synaptic_weights(self, synapse_id: SynapseId) -> nn.Parameter:
        """Get synaptic weights for a given input source, with validation."""
        if synapse_id not in self.synaptic_weights:
            raise ValueError(
                f"Synaptic weights for '{synapse_id}' not found in {self.__class__.__name__}. "
                f"Registered sources: {list(self.synaptic_weights.keys())}"
            )
        return self.synaptic_weights[synapse_id]

    def get_shape_for_synapse(self, synapse_id: SynapseId) -> SynapseShape:
        """Convenience method to get the number of post-synaptic neurons for a given synapse."""
        weights = self.get_synaptic_weights(synapse_id)
        n_output, n_input = weights.shape
        return SynapseShape(n_output=n_output, n_input=n_input)

    def get_synapse_info(self, synapse_id: SynapseId) -> SynapseInfo:
        """Convenience method to get all relevant info for a given synapse."""
        if synapse_id not in self.synaptic_weights:
            raise ValueError(
                f"Synaptic weights for '{synapse_id}' not found in {self.__class__.__name__}. "
                f"Registered sources: {list(self.synaptic_weights.keys())}"
            )
        weights = self.get_synaptic_weights(synapse_id)
        stp_module = self.stp_modules[synapse_id] if synapse_id in self.stp_modules else None
        learning_strategy = self._learning_strategies[synapse_id] if synapse_id in self._learning_strategies else None
        return SynapseInfo(
            synapse_id=synapse_id,
            weights=weights,
            stp_module=stp_module,
            learning_strategy=learning_strategy,
        )

    def _add_synaptic_weights(self, synapse_id: SynapseId, weights: torch.Tensor) -> None:
        """Add synaptic weights for a given input source, with validation."""
        if synapse_id.target_region != self.region_name:
            raise ValueError(
                f"SynapseId target_region '{synapse_id.target_region}' does not match this region '{self.region_name}'."
            )
        if synapse_id in self.synaptic_weights:
            raise ValueError(f"Synaptic weights for '{synapse_id}' already exist in {self.__class__.__name__}")
        self.synaptic_weights[synapse_id] = nn.Parameter(weights, requires_grad=False)

    def _add_stp_module(
        self,
        synapse_id: SynapseId,
        n_pre: int,
        config: STPConfig,
        *,
        device: Union[str, torch.device],
    ) -> None:
        """Add a Short-Term Plasticity (STP) module for a specific input source."""
        if synapse_id not in self.synaptic_weights:
            raise ValueError(
                f"Cannot add STP module for '{synapse_id}' because synaptic weights do not exist. "
                f"Please add synaptic weights first using _add_synaptic_weights()."
            )
        if synapse_id in self.stp_modules:
            raise ValueError(f"STP module for '{synapse_id}' already exists in {self.__class__.__name__}")
        self.stp_modules[synapse_id] = ShortTermPlasticity(n_pre, config).to(device)

    def _add_synapse(
        self,
        synapse_id: SynapseId,
        weights: torch.Tensor,
        stp_config: Optional[STPConfig],
        learning_strategy: Optional[LearningStrategy],
    ) -> None:
        """Add a new synaptic input source with weights, optional STP, and optional learning strategy."""
        if synapse_id.target_region != self.region_name:
            raise ValueError(
                f"SynapseId target_region '{synapse_id.target_region}' does not match this region '{self.region_name}'."
            )

        device = weights.device

        self._add_synaptic_weights(synapse_id, weights)

        n_output, n_input = weights.shape

        if stp_config is not None:
            self._add_stp_module(
                synapse_id=synapse_id,
                n_pre=n_input,
                config=stp_config,
                device=device,
            )

        if learning_strategy is not None:
            self._add_learning_strategy(synapse_id, learning_strategy, n_pre=n_input, n_post=n_output, device=device)

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
        """Add internal synaptic weights between neuron populations within this region.

        Enforces Dale's Law: the ``receptor_type`` must be consistent with the
        registered :class:`PopulationPolarity` of ``source_population``.  An
        ``EXCITATORY`` population cannot form inhibitory synapses and vice versa.

        Args:
            source_population: Name of the pre-synaptic population within this region.
            target_population: Name of the post-synaptic population within this region.
            weights: Weight tensor ``[n_post, n_pre]``.
            receptor_type: Post-synaptic receptor type.
            stp_config: Short-term plasticity configuration (or ``None`` to skip STP).
            learning_strategy: Optional :class:`~thalia.learning.LearningStrategy`
                registered atomically with the weight matrix.  Equivalent to
                calling :meth:`_add_learning_strategy` immediately after this
                call but keeps weight and strategy registration co-located.

        Returns:
            The :class:`SynapseId` that indexes the registered weight matrix.

        Raises:
            ValueError: If Dale's Law is violated.
        """
        # Enforce Dale's Law
        polarity = self._population_polarities.get(source_population, PopulationPolarity.ANY)
        if polarity == PopulationPolarity.EXCITATORY and receptor_type.is_inhibitory:
            raise ValueError(
                f"Dale's Law violation in {self.__class__.__name__}: "
                f"EXCITATORY population '{source_population}' cannot form an inhibitory "
                f"({receptor_type}) synapse onto '{target_population}'."
            )
        if polarity == PopulationPolarity.INHIBITORY and receptor_type.is_excitatory:
            raise ValueError(
                f"Dale's Law violation in {self.__class__.__name__}: "
                f"INHIBITORY population '{source_population}' cannot form an excitatory "
                f"({receptor_type}) synapse onto '{target_population}'."
            )

        synapse_id = SynapseId(
            source_region=self.region_name,
            source_population=source_population,
            target_region=self.region_name,
            target_population=target_population,
            receptor_type=receptor_type,
        )

        self._add_synapse(
            synapse_id=synapse_id,
            weights=weights,
            stp_config=stp_config,
            learning_strategy=learning_strategy,
        )

        return synapse_id

    def add_input_source(
        self,
        synapse_id: SynapseId,
        n_input: int,
        connectivity: float,
        weight_scale: Union[float, ConductanceScaledSpec],
        *,
        stp_config: Optional[STPConfig] = None,
        learning_strategy: Optional[LearningStrategy] = None,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> None:
        """Add synaptic weights for a new input source.

        Args:
            synapse_id: Fully-typed routing key identifying the connection.
            n_input: Number of pre-synaptic neurons.
            connectivity: Sparse connection probability (0–1).
            weight_scale: Either a plain float scale for
                :meth:`~thalia.brain.synapses.WeightInitializer.sparse_random`, or a
                :class:`~thalia.brain.synapses.ConductanceScaledSpec` for physics-based
                calibration via
                :meth:`~thalia.brain.synapses.WeightInitializer.conductance_scaled`.
            stp_config: Optional short-term plasticity configuration.
            learning_strategy: Optional :class:`~thalia.learning.LearningStrategy`
                registered atomically with the weight matrix.  Equivalent to
                calling :meth:`_add_learning_strategy` immediately after this
                call but keeps weight and strategy registration co-located.
        """
        if synapse_id.target_region != self.region_name:
            raise ValueError(
                f"SynapseId target_region '{synapse_id.target_region}' does not match this region '{self.region_name}'."
            )
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
                target_E_L=spec.target_E_L,
                target_tau_E_ms=spec.target_tau_E_ms,
                target_v_inf=spec.target_v_inf,
                fraction_of_drive=spec.fraction_of_drive,
                stp_utilization_factor=spec.stp_utilization_factor,
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

        self._add_synapse(
            synapse_id=synapse_id,
            weights=weights,
            stp_config=stp_config,
            learning_strategy=learning_strategy,
        )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @abstractmethod
    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Perform a single simulation step for the region."""

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process synaptic and neuromodulatory inputs through the region and produce outputs.

        Args:
            synaptic_inputs: Point-to-point synaptic connections (Dict[SynapseId, torch.Tensor])
            neuromodulator_inputs: Broadcast neuromodulatory signals (Dict[NeuromodulatorChannel, torch.Tensor]);
                every published channel is always a tensor — zero tensor when the source was silent.

        Returns:
            RegionOutput: Dict mapping population names to their output spike tensors
        """
        # Cache inputs so _post_forward can drive the automatic learning step.
        self._last_synaptic_inputs = synaptic_inputs

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

        if not GlobalConfig.LEARNING_DISABLED:
            self._apply_all_learning(
                self._last_synaptic_inputs,
                region_outputs,
                kwargs_provider=self._get_learning_kwargs,
            )

        return region_outputs

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

    def _integrate_synaptic_inputs_at_dendrites(
        self,
        synaptic_inputs: SynapticInput,
        n_neurons: int,
        *,
        filter_by_source_region: Optional[RegionName] = None,
        filter_by_source_population: Optional[PopulationName] = None,
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

        Args:
            synaptic_inputs: Dict mapping synapse IDs to spike tensors [n_source]
            n_neurons: Number of post-synaptic neurons (target population size)
            filter_by_source_region: If provided, only integrate inputs from this source region (e.g., "thalamus")
            filter_by_source_population: If provided, only integrate inputs from this source population (e.g., "l4")
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
            validate_spike_tensor(source_spikes, tensor_name=synapse_id)

            # Apply filters to select which inputs to integrate
            if filter_by_source_region and synapse_id.source_region != filter_by_source_region:
                continue
            if filter_by_source_population and synapse_id.source_population != filter_by_source_population:
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
            source_spikes_float = source_spikes.float()

            # OPTIONAL PER-SOURCE STP (SHORT-TERM PLASTICITY)
            # Models synaptic facilitation/depression (millisecond timescale)
            # Different sources can have different STP dynamics
            if synapse_id in _stp_dict:
                # STP returns [n_pre] pre-spike efficacy u*x.
                # Apply element-wise to spikes before matmul: W @ (efficacy * spikes)
                # This avoids creating a [n_post, n_pre] intermediate weight matrix.
                stp_efficacy = _stp_dict[synapse_id].forward(source_spikes_float)
                source_conductance = weights @ (stp_efficacy * source_spikes_float)
            else:
                # SYNAPTIC CONDUCTANCE CALCULATION
                # Convert incoming spikes to conductance: g = W @ s
                source_conductance = weights @ source_spikes_float

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
        """Integrate a single synaptic input source at the dendrites."""
        return self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={synapse_id: source_spikes},
            n_neurons=self.get_shape_for_synapse(synapse_id).n_output,
        )

    # =========================================================================
    # HOMEOSTATIC STATE REGISTRATION
    # =========================================================================

    def _register_homeostasis(
        self,
        population_name: PopulationName,
        n_neurons: int,
        *,
        target_firing_rate: float,
        use_synaptic_scaling: bool = True,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> None:
        """Register homeostatic state tracking for a neuron population.

        Creates a firing-rate EMA buffer and, optionally, a synaptic weight-scale
        buffer for the given population.  May be called before
        ``_register_neuron_population()``; population neurons are looked up
        lazily at forward time inside ``_update_homeostasis()``.

        Buffer naming convention (using ``population_name`` directly)::

            {population_name}_firing_rate   # e.g. "dg_firing_rate"
            {population_name}_weight_scale  # e.g. "l23_pyr_weight_scale" (optional)

        Args:
            population_name: Key that identifies this population — must match the
                value passed to ``_register_neuron_population()`` exactly so that
                neuron lookup in ``_update_homeostasis()`` succeeds at runtime.
            n_neurons: Number of neurons in the population (needed to allocate the
                firing-rate tensor before the population itself is registered).
            use_synaptic_scaling: When ``True``, also registers a weight-scale
                buffer for multiplicative synaptic scaling (Turrigiano & Nelson 2004).
            target_firing_rate: Homeostatic target (spikes/ms) for this population.
                Populations within the same region may have biologically distinct
                activity levels (e.g. L23 vs L5 pyramidals, or DG vs CA1).
        """
        if population_name in self._homeostasis:
            raise ValueError(
                f"Homeostasis already registered for population '{population_name}' "
                f"in {self.__class__.__name__}."
            )

        fr_attr = f"{population_name}_firing_rate"
        ws_attr: Optional[str] = None

        self.register_buffer(fr_attr, torch.zeros(n_neurons, device=device))

        if use_synaptic_scaling:
            ws_attr = f"{population_name}_weight_scale"
            self.register_buffer(ws_attr, torch.ones(1, device=device))

        self._homeostasis[population_name] = PopulationHomeostasisState(
            population_name=population_name,
            firing_rate_attr=fr_attr,
            target_firing_rate=target_firing_rate,
            weight_scale_attr=ws_attr,
        )

    def _get_target_firing_rate(self, population_name: PopulationName) -> float:
        """Return the homeostatic target firing rate for a population.

        Returns the target registered via :meth:`_register_homeostasis`.

        Args:
            population_name: The population key (same value used in
                :meth:`_register_homeostasis`).
        """
        return self._homeostasis[population_name].target_firing_rate

    def _apply_synaptic_scaling(self, population_name: PopulationName) -> None:
        """Apply multiplicative synaptic scaling if a population is chronically underactive.

        Scales up ALL input synaptic weights targeting ``population_name`` when the
        population's time-averaged firing rate falls below
        ``config.synaptic_scaling_min_activity``.  This is a slow, global homeostatic
        mechanism distinct from per-neuron gain/threshold adaptation.

        ``_register_homeostasis(population_name, use_synaptic_scaling=True)`` must have
        been called before invoking this method.

        Args:
            population_name: Key identifying the target population (same value used
                in ``_register_homeostasis()``).
        """
        if population_name not in self._homeostasis:
            raise KeyError(
                f"No homeostasis registered for '{population_name}'. "
                f"Call _register_homeostasis() first."
            )
        state = self._homeostasis[population_name]
        if state.weight_scale_attr is None:
            raise ValueError(
                f"Synaptic scaling not enabled for '{population_name}'. "
                f"Pass use_synaptic_scaling=True when calling _register_homeostasis()."
            )

        firing_rate: torch.Tensor = getattr(self, state.firing_rate_attr)
        weight_scale: torch.Tensor = getattr(self, state.weight_scale_attr)

        # Compute layer-wide average activity (not per-neuron)
        layer_avg_rate = firing_rate.mean()

        # Scale up weights when chronically below threshold
        if layer_avg_rate < self.config.synaptic_scaling_min_activity:
            # Compute scaling update (slow, multiplicative)
            rate_deficit = self.config.synaptic_scaling_min_activity - layer_avg_rate
            scale_update = self.config.synaptic_scaling_lr * rate_deficit

            # Apply multiplicative scaling (1.0 → 1.001 → 1.002, etc.)
            weight_scale.data.mul_(1.0 + scale_update).clamp_(
                min=1.0, max=self.config.synaptic_scaling_max_factor
            )

            # Scale ALL input weights to this layer
            for synapse_id, weights in self.synaptic_weights.items():
                if synapse_id.target_population == population_name:
                    weights.data.mul_(1.0 + scale_update)

    def _update_homeostasis(self, population_name: PopulationName, spikes: torch.Tensor) -> None:
        """Update homeostatic intrinsic excitability and threshold adaptation.

        Biologically-accurate homeostasis through TWO mechanisms:

        1. **Intrinsic excitability** — modulate leak conductance (``g_L_scale``):
           lower g_L ↔ higher input resistance ↔ more excitable
           (Turrigiano & Nelson 2004).
        2. **Threshold adaptation** — lower spike threshold when underactive;
           complementary to intrinsic excitability with a faster time constant.

        CRITICAL: synaptic conductances are NOT multiplied by the homeostatic
        gain — synapses do not know about homeostasis (biologically correct).
        Synaptic scaling is a separate mechanism; see ``_apply_synaptic_scaling()``.

        ``_register_homeostasis(population_name)`` must have been called during
        ``__init__`` before invoking this method.

        Args:
            population_name: Key identifying the population — must match the value
                passed to ``_register_homeostasis()``.
            spikes: Boolean or float spike tensor for the current timestep
                ``[n_neurons]``.
        """
        if GlobalConfig.HOMEOSTASIS_DISABLED:
            return

        if population_name not in self._homeostasis:
            raise KeyError(
                f"No homeostasis registered for '{population_name}' in "
                f"{self.__class__.__name__}. Call _register_homeostasis() first."
            )

        state = self._homeostasis[population_name]
        neurons = self.get_neuron_population(population_name)
        assert neurons is not None, (
            f"_update_homeostasis: neuron population '{population_name}' not found in "
            f"{self.__class__.__name__}.neuron_populations."
        )

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
        g_L_update = -self.config.gain_learning_rate * rate_error
        neurons.g_L_scale.data.add_(g_L_update).clamp_(min=0.1, max=2.0)  # Biological range

        # THRESHOLD ADAPTATION: Lower threshold when underactive (faster than g_L)
        threshold_update = -self.config.threshold_learning_rate * rate_error
        neurons.adjust_thresholds(threshold_update, self.config.threshold_min, self.config.threshold_max)

    def _apply_all_population_homeostasis(self, region_outputs: RegionOutput) -> None:
        """Call homeostasis (and synaptic scaling if enabled) for every registered population.

        Iterates :attr:`_homeostasis` and calls :meth:`_update_homeostasis` for every
        population whose spikes appear in *region_outputs*.  If the population was
        registered with ``use_synaptic_scaling=True``,
        :meth:`_apply_synaptic_scaling` is also called.

        Regions that need non-standard homeostasis logic (e.g. Striatum joint D1/D2
        rate) can override this method or skip it entirely and call the per-population
        helpers directly.

        Args:
            region_outputs: Mapping from ``PopulationName`` to spike tensor produced
                in the current timestep.
        """
        for pop_name, state in self._homeostasis.items():
            spikes = region_outputs.get(pop_name)
            if spikes is not None:
                self._update_homeostasis(pop_name, spikes)
                if state.weight_scale_attr is not None:
                    self._apply_synaptic_scaling(pop_name)

    # =========================================================================
    # LEARNING STRATEGY MANAGEMENT
    # =========================================================================

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        """Return extra keyword arguments for the learning strategy of *synapse_id*.

        Override in subclasses to supply neuromodulator concentrations or
        region-specific learning-rate multipliers without repeating the
        ``_apply_all_learning`` call in every ``forward()``.

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

    def get_learning_strategy(self, synapse_id: SynapseId) -> Optional[LearningStrategy]:
        """Return the :class:`~thalia.learning.LearningStrategy` for *synapse_id*, or ``None``."""
        strategy = self._learning_strategies.get(synapse_id)
        if strategy is not None:
            # Narrow the type from nn.Module → LearningStrategy via local import
            # (local import avoids potential circular-import issues at module load time)
            assert isinstance(strategy, LearningStrategy), f"Expected LearningStrategy for '{synapse_id}', got {type(strategy)}"
        return strategy

    def _apply_learning(
        self,
        synapse_id: SynapseId,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Look up the strategy for *synapse_id* and apply it, clamping weights in-place.

        No-ops when:
        - global learning is disabled (:data:`GlobalConfig.LEARNING_DISABLED`)
        - no strategy is registered for *synapse_id*

        Weight clamping uses ``self.config.w_min`` / ``self.config.w_max``.
        """
        if GlobalConfig.LEARNING_DISABLED:
            return

        strategy = self.get_learning_strategy(synapse_id)
        if strategy is not None:
            weights = self.get_synaptic_weights(synapse_id)
            # Capture connectivity mask BEFORE the update so that only
            # pre-existing synapses (weight > 0 at initialisation or after LTP)
            # can be modified.  Hebbian/STDP outer-products produce dense
            # [n_post × n_pre] matrices; without this mask they would create
            # de-novo connections at anatomically absent entries, violating
            # the sparse connectivity layout set at build time and causing the
            # same runaway potentiation seen in hippocampus (Fix 2, 2025-07).
            syn_mask = weights.data > 0.0
            updated: torch.Tensor = strategy.compute_update(
                weights=weights.data,
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                **kwargs,
            )
            clamp_weights(updated, self.config.w_min, self.config.w_max)
            # Zero any newly-created entries; anatomically absent synapses stay absent.
            updated.mul_(syn_mask.float())
            weights.data = updated

    def _apply_all_learning(
        self,
        synaptic_inputs: SynapticInput,
        region_outputs: RegionOutput,
        *,
        kwargs_provider: Optional[Callable[[SynapseId], Dict[str, Any]]] = None,
    ) -> None:
        """Apply all registered learning strategies for external input synapses.

        Iterates ``self._learning_strategies`` and calls :meth:`_apply_learning`
        for every synapse whose **pre-spikes can be found in** *synaptic_inputs*.
        Synapses whose pre-spikes are computed internally (e.g. recurrent
        connections in cortex or hippocampus) are skipped automatically.

        This is the counterpart of :meth:`_apply_all_population_homeostasis` for
        the learning step: a single call in a region's ``forward()`` replaces a
        ``for synapse_id, source_spikes in synaptic_inputs.items():`` loop.

        Args:
            synaptic_inputs: Map of :class:`~thalia.typing.SynapseId` to pre-spike
                tensors received by the region this timestep.
            region_outputs: Map of population name → post-spike tensor produced in
                the current timestep.
            kwargs_provider: Optional callable ``(synapse_id) -> dict`` that returns
                extra keyword arguments (e.g. ``dopamine=``, ``learning_rate=``) to
                pass to :meth:`_apply_learning` for a specific synapse.  If ``None``
                no extra kwargs are passed.

        Example — simple region with no per-synapse neuromodulator kwargs::

            def forward(self, synaptic_inputs, neuromodulator_inputs):
                ...
                self._apply_all_learning(synaptic_inputs, region_outputs)

        Example — region with a scalar DA modulator::

            def _learning_kwargs(self, synapse_id):
                return {"dopamine": self._da_concentration.mean().item()}

            def forward(self, synaptic_inputs, neuromodulator_inputs):
                ...
                self._apply_all_learning(
                    synaptic_inputs, region_outputs,
                    kwargs_provider=self._learning_kwargs,
                )
        """
        if GlobalConfig.LEARNING_DISABLED:
            return

        for synapse_id, pre_spikes in synaptic_inputs.items():
            # Look up post-spikes for this synapse's target population.
            post_spikes = region_outputs.get(synapse_id.target_population)
            if post_spikes is None:
                # Target population not in outputs (e.g., PV inhibitory tracked elsewhere)
                continue

            # Lazy strategy registration falls through to _apply_learning if the
            # region doesn't pre-register strategies.  Regions that need explicit
            # strategy types (e.g., composite BCM+STDP) should call
            # _add_learning_strategy() before this point or override this method.
            extra = kwargs_provider(synapse_id) if kwargs_provider is not None else {}
            self._apply_learning(synapse_id, pre_spikes, post_spikes, **extra)

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters for neurons and learning strategies.

        Args:
            dt_ms: New timestep in milliseconds
        """
        self.config.dt_ms = dt_ms

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
                strategy.update_temporal_parameters(dt_ms)  # type: ignore[union-attr]

        self._firing_rate_alpha = self.dt_ms / self.config.gain_tau_ms
