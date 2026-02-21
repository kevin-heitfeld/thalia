"""NeuralRegion base class for brain regions with biologically accurate synaptic inputs.

This class implements the core functionality for neural regions, including:
1. Managing synaptic weights for multiple input sources (dendrites)
2. Integrating synaptic inputs at dendrites (multi-source summation)
3. Providing utilities for homeostatic plasticity and temporal parameter management

Regions with internal structure (like Cortex) should subclass NeuralRegion and:
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
from typing import TYPE_CHECKING, Dict, Optional, TypeVar, Generic

import torch
import torch.nn as nn

from thalia.brain.configs import NeuralRegionConfig
from thalia.components.synapses import (
    STPConfig,
    ShortTermPlasticity,
    WeightInitializer,
)
from thalia.typing import (
    NeuromodulatorInput,
    PopulationName,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import validate_spike_tensor, validate_spike_tensors

if TYPE_CHECKING:
    from thalia.components.neurons import ConductanceLIF, IzhikevichNeuron
    from thalia.learning import LearningStrategy


ConfigT = TypeVar('ConfigT', bound=NeuralRegionConfig)


class NeuralRegion(nn.Module, ABC, Generic[ConfigT]):
    """Base class for brain regions with biologically accurate synaptic inputs.

    Regions:
    1. Own their synaptic weights (one weight matrix per input source)
    2. Define learning rules (per-source plasticity)
    3. Integrate multi-source inputs naturally

    Subclassing:
        Regions with internal structure (like Cortex) should:
        1. Call super().__init__() to get _synaptic_weights dict
        2. Define internal neurons/weights for within-region processing
        3. Override forward() to apply synaptic weights then internal processing
    """

    config: ConfigT  # Type annotation for config

    # =========================================================================
    # PROPERTIES AND UTILS
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
    def dtype(self) -> torch.dtype:
        """Data type for floating point tensors."""
        return torch.float32

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: ConfigT, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize NeuralRegion with config and layer sizes."""
        super().__init__()

        # ====================================================================
        # VALIDATE INPUT SIZES
        # ====================================================================
        for source_name, size in population_sizes.items():
            if not isinstance(size, int) or size < 0:
                raise ValueError(
                    f"NeuralRegion.__init__: Invalid input size for source '{source_name}': "
                    f"expected non-negative int, got {size}."
                )

        # ====================================================================
        # INITIALIZE INSTANCE VARIABLES
        # ====================================================================
        self.config: ConfigT = config
        self.region_name: RegionName = region_name

        # Synaptic weights: one weight matrix per input source
        self._synaptic_weights: Dict[SynapseId, nn.Parameter] = {}  #nn.ParameterDict()

        # Optional per-source STP modules for short-term plasticity (facilitation/depression)
        self.stp_modules: Dict[SynapseId, ShortTermPlasticity] = {}  #nn.ModuleDict()

        # Track which sources have been added
        self.input_sources: Dict[SynapseId, int] = {}

        # Neuron populations within this region (e.g., L4, L2/3, L5 for Cortex)
        self.neuron_populations: Dict[PopulationName, ConductanceLIF | IzhikevichNeuron] = {}

        # Strategy instance (set by subclass)
        self.learning_strategy: Optional[LearningStrategy] = None

        # EMA alpha for firing rate tracking
        self._firing_rate_alpha = self.dt_ms / self.config.gain_tau_ms

    def __post_init__(self) -> None:
        """Post-initialization processing after dataclass __init__."""
        # Ensure all tensors are on the correct device
        self.to(self.device)

    def _register_neuron_population(
        self,
        population_name: PopulationName,
        population: ConductanceLIF | IzhikevichNeuron,
    ) -> None:
        """Register a neuron population within this region."""
        if population_name in self.neuron_populations:
            raise ValueError(f"Population '{population_name}' already registered in {self.__class__.__name__}")
        self.neuron_populations[population_name] = population

    def get_population_size(self, population: PopulationName) -> int:
        """Get population size from population name."""
        if population not in self.neuron_populations:
            raise ValueError(
                f"Population '{population}' not found in {self.__class__.__name__}. "
                f"Registered populations: {self.neuron_populations.keys()}"
            )
        return self.neuron_populations[population].n_neurons

    # =========================================================================
    # SYNAPTIC WEIGHT MANAGEMENT
    # =========================================================================

    def has_synaptic_weights(self, synapse_id: SynapseId) -> bool:
        """Check if synaptic weights exist for a given input source."""
        return synapse_id in self._synaptic_weights

    def get_synaptic_weights(self, synapse_id: SynapseId) -> nn.Parameter:
        """Get synaptic weights for a given input source, with validation."""
        if not self.has_synaptic_weights(synapse_id):
            raise ValueError(
                f"Synaptic weights for '{synapse_id}' not found in {self.__class__.__name__}. "
                f"Registered sources: {list(self._synaptic_weights.keys())}"
            )
        return self._synaptic_weights[synapse_id]

    def add_synaptic_weights(self, synapse_id: SynapseId, weights: torch.Tensor) -> None:
        """Add synaptic weights for a given input source, with validation."""
        if synapse_id in self._synaptic_weights:
            raise ValueError(f"Synaptic weights for '{synapse_id}' already exist in {self.__class__.__name__}")
        self._synaptic_weights[synapse_id] = nn.Parameter(weights, requires_grad=False)

    def add_stp_module(
        self,
        synapse_id: SynapseId,
        n_pre: int,
        n_post: int,
        config: STPConfig,
    ) -> None:
        """Add a Short-Term Plasticity (STP) module for a specific input source."""
        if synapse_id in self.stp_modules:
            raise ValueError(f"STP module for '{synapse_id}' already exists in {self.__class__.__name__}")
        if not self.has_synaptic_weights(synapse_id):
            raise ValueError(
                f"Cannot add STP module for '{synapse_id}' because synaptic weights do not exist. "
                f"Please add synaptic weights first using add_synaptic_weights()."
            )
        self.stp_modules[synapse_id] = ShortTermPlasticity(n_pre, n_post, config).to(self.device)

    def add_input_source(
        self,
        synapse_id: SynapseId,
        n_input: int,
        connectivity: float,
        weight_scale: float,
        *,
        stp_config: Optional[STPConfig] = None,
    ) -> None:
        """Add synaptic weights for a new input source."""

        if synapse_id in self.input_sources:
            raise ValueError(f"Input source '{synapse_id}' already exists")
        if n_input < 0:
            raise ValueError("Number of input neurons must be non-negative")
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("Connectivity must be between 0 and 1")
        if weight_scale < 0:
            raise ValueError("Weight scale must be non-negative")

        # Track source registration
        self.input_sources[synapse_id] = n_input

        # All registered inputs are synaptic (neuromodulators use separate broadcast system)
        n_output = self.get_population_size(synapse_id.target_population)

        self.add_synaptic_weights(
            synapse_id,
            WeightInitializer.sparse_random(
                n_input=n_input,
                n_output=n_output,
                connectivity=connectivity,
                weight_scale=weight_scale,
                device=self.device,
            ),
        )

        if stp_config is not None:
            self.add_stp_module(
                synapse_id=synapse_id,
                n_pre=n_input,
                n_post=n_output,
                config=stp_config,
            )

    def _add_internal_connection(
        self,
        source_population: PopulationName,
        target_population: PopulationName,
        weights: torch.Tensor,
        *,
        stp_config: STPConfig,
        is_inhibitory: bool = False,
    ) -> SynapseId:
        """Add internal synaptic weights between neuron populations within this region."""
        synapse_id = SynapseId(
            source_region=self.region_name,
            source_population=source_population,
            target_region=self.region_name,
            target_population=target_population,
            is_inhibitory=is_inhibitory,
        )

        if synapse_id in self.input_sources:
            raise ValueError(f"Input source '{synapse_id}' already exists")

        self.add_synaptic_weights(synapse_id, weights)

        if stp_config is not None:
            n_output, n_input = weights.shape
            self.add_stp_module(
                synapse_id=synapse_id,
                n_pre=n_input,
                n_post=n_output,
                config=stp_config,
            )

        return synapse_id

    def _split_excitatory_conductance(
        self,
        g_exc_total: torch.Tensor,
        nmda_ratio: float = 0.0,  # CRITICAL: Default 0.0 to prevent NMDA accumulation
        # TODO: Consider making nmda_ratio a per-source parameter for more biological realism
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split excitatory conductance into AMPA (fast) and NMDA (slow) components.

        Biology: Excitatory synapses contain both AMPA and NMDA receptors.
        AMPA provides fast transmission (tau~5ms), NMDA provides slow temporal
        integration (tau~100ms).

        CRITICAL: Due to tau_nmda=100ms vs tau_ampa=5ms (20x difference),
        NMDA accumulates much more than AMPA in steady state. With 30% NMDA input,
        steady-state conductance becomes 7-10x more NMDA than AMPA, causing
        pathological over-excitation. Reduced to 5% NMDA to prevent over-integration
        while still providing slow timescale dynamics.

        Args:
            g_exc_total: Total excitatory conductance to split
            nmda_ratio: Fraction of total conductance that is NMDA (default 0.05 = 5%)

        Returns:
            g_ampa: Fast AMPA conductance (95% of total)
            g_nmda: Slow NMDA conductance (5% of total)
        """
        ampa_ratio = 1.0 - nmda_ratio
        g_ampa = g_exc_total * ampa_ratio
        g_nmda = g_exc_total * nmda_ratio
        return g_ampa, g_nmda

    def _apply_synaptic_scaling(
        self,
        firing_rate: torch.Tensor,
        weight_scale: torch.Tensor,
        target_population: PopulationName,
    ) -> None:
        """Apply multiplicative synaptic scaling if layer is chronically underactive.

        All cortical layers should scale up input weights when firing rates
        are chronically below threshold. This is a slow, global homeostatic
        mechanism distinct from per-neuron gain/threshold adaptation.

        Args:
            firing_rate: Layer's firing rate EMA
            weight_scale: Layer's weight scaling factor
            population_name: Name of the target population (e.g., "l23") for scaling input weights
        """
        # Compute layer-wide average activity (not per-neuron)
        layer_avg_rate = firing_rate.mean()

        # Scale up weights when chronically below threshold
        if layer_avg_rate < self.config.synaptic_scaling_min_activity:
            # Compute scaling update (slow, multiplicative)
            rate_deficit = self.config.synaptic_scaling_min_activity - layer_avg_rate
            scale_update = self.config.synaptic_scaling_lr * rate_deficit

            # Apply multiplicative scaling (1.0 -> 1.001 -> 1.002, etc.)
            weight_scale.data.mul_(1.0 + scale_update).clamp_(
                min=1.0, max=self.config.synaptic_scaling_max_factor
            )

            # Scale ALL input weights to this layer
            for synapse_id, weights in self._synaptic_weights.items():
                if synapse_id.target_population == target_population:
                    weights.data.mul_(1.0 + scale_update)

    def _update_homeostasis(
        self,
        spikes: torch.Tensor,
        firing_rate: torch.Tensor,
        neurons: ConductanceLIF,
    ) -> None:
        """Update homeostatic intrinsic excitability and threshold adaptation.

        Biologically-accurate homeostasis through TWO mechanisms:
        1. INTRINSIC EXCITABILITY: Modulate leak conductance (g_L_scale)
           - Lower g_L → higher input resistance → more excitable
           - Implemented as neurons.g_L_scale (Turrigiano & Nelson 2004)
        2. THRESHOLD ADAPTATION: Lower threshold when underactive
           - Complementary to intrinsic excitability
           - Faster time constant than g_L modulation

        CRITICAL: We do NOT multiply synaptic conductances by gain!
        That's biologically incorrect - synapses don't know about homeostasis.

        Args:
            spikes: Current timestep's spikes [layer_size]
            firing_rate: Exponential moving average of firing rate [layer_size]
            neurons: Layer's neuron population (has g_L_scale, v_threshold)
        """
        # Update firing rate EMA with current spikes
        current_rate = spikes.float()
        firing_rate.mul_(1.0 - self._firing_rate_alpha).add_(current_rate * self._firing_rate_alpha)

        # Compute rate error: positive when underactive
        rate_error = self.config.target_firing_rate - firing_rate  # [layer_size]

        # INTRINSIC EXCITABILITY: Modulate leak conductance
        # When underactive (rate_error > 0): decrease g_L_scale → lower leak → more excitable
        # When overactive (rate_error < 0): increase g_L_scale → higher leak → less excitable
        # Inverse relationship: g_L ∝ 1/excitability
        g_L_update = -self.config.gain_learning_rate * rate_error  # Negative: underactive → lower g_L
        neurons.g_L_scale.data.add_(g_L_update).clamp_(min=0.1, max=2.0)  # Biological range

        # THRESHOLD ADAPTATION: Lower threshold when underactive (faster than g_L)
        threshold_update = -self.config.threshold_learning_rate * rate_error
        neurons.adjust_thresholds(threshold_update, self.config.threshold_min, self.config.threshold_max)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _pre_forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> None:
        """Pre-forward processing (input validation, etc.)."""
        # Validate synaptic input spike tensors
        validate_spike_tensors(synaptic_inputs, context=f"{self.__class__.__name__} forward synaptic inputs")

        # Check that all synaptic input sources are registered
        for synapse_id in synaptic_inputs.keys():
            if synapse_id not in self.input_sources:
                raise ValueError(
                    f"Input source '{synapse_id}' not registered in {self.__class__.__name__}. "
                    f"Registered sources: {list(self.input_sources.keys())}"
                )
            # All registered inputs must have weights (neuromodulators use separate system)
            if synapse_id not in self._synaptic_weights:
                raise ValueError(
                    f"Synaptic weights for input source '{synapse_id}' not found in {self.__class__.__name__}. "
                    f"Registered sources: {list(self._synaptic_weights.keys())}"
                )

        # Validate neuromodulator inputs (if any)
        for neuromod_type, spikes in neuromodulator_inputs.items():
            if spikes is not None:
                validate_spike_tensor(spikes, tensor_name=f"neuromodulator_{neuromod_type}")

    def _post_forward(self, region_outputs: RegionOutput) -> RegionOutput:
        """Post-forward processing (e.g., tracking output spikes)."""
        return region_outputs

    def __call__(self, *args, **kwds):
        assert False, f"{self.__class__.__name__} instances should not be called directly. Use forward() instead."
        return super().__call__(*args, **kwds)

    @abstractmethod
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process synaptic and neuromodulatory inputs through the region and produce outputs.

        Args:
            synaptic_inputs: Point-to-point synaptic connections (Dict[SynapseId, torch.Tensor])
            neuromodulator_inputs: Broadcast neuromodulatory signals (Dict[NeuromodulatorType, Optional[torch.Tensor]])

        Returns:
            RegionOutput: Dict mapping population names to their output spike tensors
        """

    def _integrate_synaptic_inputs_at_dendrites(
        self,
        synaptic_inputs: SynapticInput,
        n_neurons: int,
        *,
        filter_by_source_region: Optional[RegionName] = None,
        filter_by_source_population: Optional[PopulationName] = None,
        filter_by_target_population: Optional[PopulationName] = None,
    ) -> torch.Tensor:
        """Integrate synaptic inputs at dendrites (multi-source summation).

        Biological Process:
        1. Spikes arrive via axons (already delayed in AxonalTract)
        2. Synapses convert spikes to conductances: g = weights @ spikes
        3. Dendrites sum conductances from all sources: g_total = Σ g_i
        4. Soma integrates total conductance: neurons.forward(g_total)

        Args:
            synaptic_inputs: Dict mapping synapse IDs to spike tensors [n_source]
            n_neurons: Number of post-synaptic neurons (target population size)
            filter_by_source_region: If provided, only integrate inputs from this source region (e.g., "thalamus")
            filter_by_source_population: If provided, only integrate inputs from this source population (e.g., "l4")
            filter_by_target_population: If provided, only integrate inputs targeting this population (e.g., "l4")

        Returns:
            Total synaptic conductance [n_neurons] (float tensor, normalized by g_L)

        Raises:
            AssertionError: If input tensors violate ADR-005 (not 1D)

        Note:
            Routing keys are used directly as synaptic weight keys.
            Region subclasses should initialize _synaptic_weights with routing keys.
        """
        g_total = torch.zeros(n_neurons, device=self.device)

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
            if not self.has_synaptic_weights(synapse_id):
                raise ValueError(
                    f"Synaptic weights for input source '{synapse_id}' not found in {self.__class__.__name__}. "
                    f"Registered sources: {list(self._synaptic_weights.keys())}"
                )

            weights = self._synaptic_weights[synapse_id]
            source_spikes_float = source_spikes.float()

            if synapse_id.is_inhibitory:
                # TODO: Also integrate inhibitory conductance here?
                pass
            else:
                # OPTIONAL PER-SOURCE STP (SHORT-TERM PLASTICITY)
                # Models synaptic facilitation/depression (millisecond timescale)
                # Different sources can have different STP dynamics
                if synapse_id in self.stp_modules:
                    # Get STP efficacy modulation for this source
                    stp_efficacy = self.stp_modules[synapse_id].forward(source_spikes_float)
                    # Apply as multiplicative modulation of weights
                    # STP returns [n_pre, n_post], weights are [n_post, n_pre] → transpose
                    weights = weights * stp_efficacy.T

                # SYNAPTIC CONDUCTANCE CALCULATION
                # Convert incoming spikes to conductance: g = W @ s
                source_conductance = weights @ source_spikes_float

                # ACCUMULATE SYNAPTIC CONDUCTANCE
                # Biology: Dendritic conductances sum at soma (linear superposition)
                g_total += source_conductance

        # Clamp to non-negative (conductances cannot be negative!)
        g_total = torch.clamp(g_total, min=0.0)

        return g_total

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters for neurons and learning strategies.

        Args:
            dt_ms: New timestep in milliseconds
        """
        self.config.dt_ms = dt_ms

        for stp_module in self.stp_modules.values():
            stp_module.update_temporal_parameters(dt_ms)

        if self.learning_strategy is not None:
            self.learning_strategy.update_temporal_parameters(dt_ms)

        self._firing_rate_alpha = self.dt_ms / self.config.gain_tau_ms
