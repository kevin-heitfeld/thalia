"""
CorticalColumn - Multi-layer cortical column.

This implements a biologically realistic cortical column with distinct layers:

Architecture (based on canonical cortical microcircuit):
=========================================================

         Feedback from higher areas
                    │
                    ▼
    ┌───────────────────────────────────┐
    │          LAYER 2/3                │ ← Superficial pyramidal cells
    │   (Cortico-cortical output)       │ → To other cortical areas
    │   - Receives from L4              │ → Gamma attention gating
    │   - Lateral recurrent connections │ → Gap junctions for sync
    │   - Top-down feedback target      │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 4                  │ ← Spiny stellate cells
    │   (Feedforward input layer)       │ ← From thalamus/lower areas
    │   - Main sensory input recipient  │
    │   - No recurrent connections      │
    │   - Fast, feedforward processing  │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 5                  │ ← Deep pyramidal cells
    │   (Subcortical output layer)      │ → To striatum, brainstem, etc.
    │   - Receives from L2/3            │ → Motor/action-related output
    │   - Different output pathway      │
    │   - Burst-capable neurons         │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 6a                 │ ← CT Type I (low gamma)
    │   (Corticothalamic → TRN)         │ → Spatial attention via TRN
    │   - Receives from L2/3            │ → Inhibitory modulation
    │   - Projects to thalamic TRN      │
    │   - Sparse, low-gamma firing      │
    └───────────────────────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 6b                 │ ← CT Type II (high gamma)
    │   (Corticothalamic → Relay)       │ → Fast gain modulation
    │   - Receives from L2/3            │ → Direct relay excitation
    │   - Projects to thalamic relay    │
    │   - Dense, high-gamma firing      │
    └───────────────────────────────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from thalia import GlobalConfig
from thalia.brain.configs import CortexConfig, CortexLayer
from thalia.components import (
    NeuromodulatorReceptor,
    NeuronFactory,
    WeightInitializer,
)
from thalia.learning import (
    BCMConfig,
    BCMStrategy,
    STDPConfig,
    STDPStrategy,
    CompositeStrategy,
    PredictiveCodingConfig,
    PredictiveCodingStrategy,
    compute_excitability_modulation,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    ReceptorType,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import (
    CircularDelayBuffer,
    compute_ach_recurrent_suppression,
    compute_ne_gain,
    split_excitatory_conductance,
)

from .cortical_inhibitory_network import CorticalInhibitoryNetwork
from .neural_region import NeuralRegion
from .population_names import CortexPopulation
from .region_registry import register_region
from .stimulus_gating import StimulusGating

if TYPE_CHECKING:
    from thalia.components.neurons import ConductanceLIF, TwoCompartmentLIF


@register_region(
    "cortical_column",
    description="Multi-layer cortical microcircuit with L2/3/L4/L5/L6a/L6b structure",
    version="1.0",
    author="Thalia Project",
    config_class=CortexConfig,
)
class CorticalColumn(NeuralRegion[CortexConfig]):
    """Multi-layer cortical microcircuit with proper layer separation and routing.

    Implements a canonical cortical column with distinct computational layers:

    **Layer Architecture** (based on mammalian cortex):
    - **L2/3**: Processing layer - recurrent computation, outputs to other cortex
    - **L4**: Input layer - receives thalamic/sensory input, feedforward processing
    - **L5**: Output layer - projects to subcortical structures (striatum, etc.)
    - **L6a**: CT Type I - projects to thalamic TRN (spatial attention, low gamma)
    - **L6b**: CT Type II - projects to thalamic relay (gain modulation, high gamma)

    **Information Flow**:
    1. External input → L4 (feedforward processing)
    2. L4 → L2/3 (local integration with axonal delays)
    3. L2/3 → L2/3 (recurrent processing with STP and gap junctions)
    4. L2/3 → L5 (deep projection with axonal delays)
    5. L2/3 → L6a (corticothalamic type I with axonal delays)
    6. L2/3 → L6b (corticothalamic type II with axonal delays)
    7. L2/3 → Other cortex (cortico-cortical output)
    8. L5 → Subcortical (striatum, thalamus, brainstem)
    9. L6a → Thalamus TRN (spatial attention via inhibitory modulation)
    10. L6b → Thalamus Relay (fast gain modulation via excitatory drive)
    """

    # Mesocortical DA (VTA → PFC/association cortex) modulates layer-specific gain.
    # NE from LC sets broadband gain and signal-to-noise ratio.
    # ACh from nucleus basalis drives top-down attention (muscarinic M1 in L5, nicotinic in L2/3).
    neuromodulator_subscriptions: ClassVar[List[str]] = ['da_mesocortical', 'ne', 'ach']

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: CortexConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize cortex."""
        super().__init__(config, population_sizes, region_name)

        # =====================================================================
        # EXTRACT LAYER SIZES AND CALCULATE FSI POPULATIONS
        # =====================================================================
        # Layer sizes represent PYRAMIDAL neurons (excitatory, what projects out)
        # FSI (fast-spiking interneurons) are additive internal circuitry:
        #   - FSI = 25% of pyramidal count = 20% of total (80/20 split)
        #   - Example: 400 pyr + 100 FSI = 500 total (80% pyr, 20% FSI)

        # Pyramidal sizes
        self.l23_pyr_size: int = population_sizes[CortexPopulation.L23_PYR]
        self.l4_pyr_size: int = population_sizes[CortexPopulation.L4_PYR]
        self.l5_pyr_size: int = population_sizes[CortexPopulation.L5_PYR]
        self.l6a_pyr_size: int = population_sizes[CortexPopulation.L6A_PYR]
        self.l6b_pyr_size: int = population_sizes[CortexPopulation.L6B_PYR]

        # =====================================================================
        # INITIALIZE SUBCOMPONENTS
        # =====================================================================
        device = self.device

        # STDP+BCM learning strategies for each layer.
        # Created BEFORE _init_synaptic_weights() so they can be registered
        # atomically with their weight matrices (A8).
        # Custom STDP config using biologically appropriate amplitudes from config
        stdp_cfg = STDPConfig(
            learning_rate=config.learning_rate,
            a_plus=config.a_plus,
            a_minus=config.a_minus,
            tau_plus=config.tau_plus_ms,
            tau_minus=config.tau_minus_ms,
            w_min=config.w_min,
            w_max=config.w_max,
        )

        # BCM config for homeostatic modulation
        # CRITICAL: BCM LR must be much slower than STDP to prevent runaway potentiation
        # Use 2% of STDP learning rate (50:1 ratio) to reduce BCM influence
        # Previous 10:1 ratio caused runaway depression overwhelming STDP
        bcm_cfg = BCMConfig(
            learning_rate=config.learning_rate * 0.02,  # 50x slower than STDP
            w_min=config.w_min,
            w_max=config.w_max,
            tau_theta=100000.0,  # Very slow threshold adaptation (100s)
            theta_init=0.01,  # Higher initial threshold
            theta_max=0.5,  # Prevent saturation
            weight_decay=0.0002,  # L2 toward zero (proper weight decay)
            activity_threshold=0.01,  # 1% - allows LTD in sparse networks
        )

        # Create per-layer CompositeStrategy instances (STDP + BCM, nn.Module → tracked correctly)
        self.composite_l23: CompositeStrategy = CompositeStrategy([STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)])
        self.composite_l4:  CompositeStrategy = CompositeStrategy([STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)])
        self.composite_l5:  CompositeStrategy = CompositeStrategy([STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)])
        self.composite_l6a: CompositeStrategy = CompositeStrategy([STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)])
        self.composite_l6b: CompositeStrategy = CompositeStrategy([STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)])

        # Initialize layers and synaptic weights.
        # composite_* are now available for inline strategy registration.
        self._init_layers()
        self._init_synaptic_weights()

        # Intrinsic plasticity tracking
        self._l23_activity_history: torch.Tensor = torch.zeros(self.l23_pyr_size, device=device)
        self._l23_threshold_offset: torch.Tensor = torch.zeros(self.l23_pyr_size, device=device)

        # Initialize stimulus gating (transient inhibition)
        self.stimulus_gating: StimulusGating = StimulusGating(
            n_neurons=self.l4_pyr_size,
            device=device,
            threshold=config.ffi_threshold,
            max_inhibition=config.ffi_strength * 10.0,
            decay_rate=1.0 - (1.0 / config.ffi_tau),
        )

        # =====================================================================
        # DOPAMINE RECEPTOR (sparse 30% projection, L5-specific)
        # =====================================================================
        # CorticalColumn receives sparse DA innervation primarily to Layer 5 for action selection
        # L5 pyramidal cells project to striatum and benefit from DA modulation
        # Biological: VTA/SNc DA terminals concentrated in deep layers
        # Note: Only L5 neurons receive DA; other layers get zero concentration
        total_neurons = self.l23_pyr_size + self.l4_pyr_size + self.l5_pyr_size + self.l6a_pyr_size + self.l6b_pyr_size
        self.da_receptor = NeuromodulatorReceptor(
            n_receptors=total_neurons,
            tau_rise_ms=15.0,  # Medium kinetics
            tau_decay_ms=150.0,  # Longer clearance for sustained modulation
            spike_amplitude=0.1,  # Moderate amplitude for stable learning
            device=device,
        )
        # Per-layer DA concentration buffers
        # Only L5 receives DA (30% projection strength), others get zero
        self._da_concentration_l23: torch.Tensor
        self._da_concentration_l4: torch.Tensor
        self._da_concentration_l5: torch.Tensor
        self._da_concentration_l6a: torch.Tensor
        self._da_concentration_l6b: torch.Tensor
        self.register_buffer("_da_concentration_l23", torch.zeros(self.l23_pyr_size, device=device))
        self.register_buffer("_da_concentration_l4", torch.zeros(self.l4_pyr_size, device=device))
        self.register_buffer("_da_concentration_l5", torch.zeros(self.l5_pyr_size, device=device))
        self.register_buffer("_da_concentration_l6a", torch.zeros(self.l6a_pyr_size, device=device))
        self.register_buffer("_da_concentration_l6b", torch.zeros(self.l6b_pyr_size, device=device))

        # =====================================================================
        # NOREPINEPHRINE RECEPTOR (LC projection - all layers)
        # =====================================================================
        # CorticalColumn receives dense NE innervation from LC across all layers
        # NE modulates arousal, gain, and attention uniformly
        self.ne_receptor = NeuromodulatorReceptor(
            n_receptors=total_neurons,
            tau_rise_ms=8.0,
            tau_decay_ms=150.0,
            spike_amplitude=0.12,
            device=device,
        )
        # Per-layer NE concentration buffers (all layers receive NE)
        self._ne_concentration_l23: torch.Tensor
        self._ne_concentration_l4: torch.Tensor
        self._ne_concentration_l5: torch.Tensor
        self._ne_concentration_l6a: torch.Tensor
        self._ne_concentration_l6b: torch.Tensor
        self.register_buffer("_ne_concentration_l23", torch.zeros(self.l23_pyr_size, device=device))
        self.register_buffer("_ne_concentration_l4", torch.zeros(self.l4_pyr_size, device=device))
        self.register_buffer("_ne_concentration_l5", torch.zeros(self.l5_pyr_size, device=device))
        self.register_buffer("_ne_concentration_l6a", torch.zeros(self.l6a_pyr_size, device=device))
        self.register_buffer("_ne_concentration_l6b", torch.zeros(self.l6b_pyr_size, device=device))

        # =====================================================================
        # ACETYLCHOLINE RECEPTOR (NB projection - layer-specific)
        # =====================================================================
        # CorticalColumn receives ACh from nucleus basalis with layer-specific effects:
        # - L4: Enhance sensory processing (feedforward)
        # - L2/3: Enhance association processing
        # - L5/L6: Suppress recurrence (reduce feedback, enhance encoding)
        self.ach_receptor = NeuromodulatorReceptor(
            n_receptors=total_neurons,
            tau_rise_ms=5.0,
            tau_decay_ms=50.0,
            spike_amplitude=0.2,
            device=device,
        )
        # Per-layer ACh concentration buffers
        self._ach_concentration_l23: torch.Tensor
        self._ach_concentration_l4: torch.Tensor
        self._ach_concentration_l5: torch.Tensor
        self._ach_concentration_l6a: torch.Tensor
        self._ach_concentration_l6b: torch.Tensor
        self.register_buffer("_ach_concentration_l23", torch.zeros(self.l23_pyr_size, device=device))
        self.register_buffer("_ach_concentration_l4", torch.zeros(self.l4_pyr_size, device=device))
        self.register_buffer("_ach_concentration_l5", torch.zeros(self.l5_pyr_size, device=device))
        self.register_buffer("_ach_concentration_l6a", torch.zeros(self.l6a_pyr_size, device=device))
        self.register_buffer("_ach_concentration_l6b", torch.zeros(self.l6b_pyr_size, device=device))

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(CortexPopulation.L23_PYR, self.l23_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CortexPopulation.L4_PYR, self.l4_neurons, polarity=PopulationPolarity.EXCITATORY)
        # L5_PYR is EXCITATORY (Dale’s Law compliant). The predictive-coding
        # feedback is now disynaptic: L5_PYR (AMPA) → L4_SST_PRED → L4_PYR (GABA_A).
        self._register_neuron_population(CortexPopulation.L5_PYR, self.l5_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CortexPopulation.L6A_PYR, self.l6a_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CortexPopulation.L6B_PYR, self.l6b_neurons, polarity=PopulationPolarity.EXCITATORY)
        # Prediction-error SST interneuron driven by L5; inhibits L4 dendrites
        self._register_neuron_population(CortexPopulation.L4_SST_PRED, self.l4_sst_pred_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L23_INHIBITORY_PV, self.l23_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L23_INHIBITORY_SST, self.l23_inhibitory.sst_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L23_INHIBITORY_VIP, self.l23_inhibitory.vip_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L4_INHIBITORY_PV, self.l4_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L4_INHIBITORY_SST, self.l4_inhibitory.sst_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L4_INHIBITORY_VIP, self.l4_inhibitory.vip_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L5_INHIBITORY_PV, self.l5_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L5_INHIBITORY_SST, self.l5_inhibitory.sst_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L5_INHIBITORY_VIP, self.l5_inhibitory.vip_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L6A_INHIBITORY_PV, self.l6a_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L6A_INHIBITORY_SST, self.l6a_inhibitory.sst_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L6A_INHIBITORY_VIP, self.l6a_inhibitory.vip_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L6B_INHIBITORY_PV, self.l6b_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L6B_INHIBITORY_SST, self.l6b_inhibitory.sst_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L6B_INHIBITORY_VIP, self.l6b_inhibitory.vip_neurons, polarity=PopulationPolarity.INHIBITORY)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    def _init_layers(self) -> None:
        """Initialize conductance-based LIF neurons for each layer.

        Uses ConductanceLIF for biologically realistic gain control:
        - Separate excitatory and inhibitory conductances
        - Shunting inhibition (divisive effect)
        - Natural saturation at reversal potentials
        - No need for artificial divisive normalization

        **Layer-Specific Heterogeneity**:
        Applies distinct electrophysiological properties per layer
        (tau_mem, v_threshold, adaptation) reflecting biological
        diversity of cortical cell types.
        """
        cfg = self.config
        device = self.device

        # =====================================================================
        # LAYER-SPECIFIC HETEROGENEITY
        # =====================================================================
        self.l23_neurons = NeuronFactory.create_pyramidal_two_compartment(
            region_name=self.region_name,
            population_name=CortexPopulation.L23_PYR,
            n_neurons=self.l23_pyr_size,
            device=device,
            **cfg.layer_overrides[CortexLayer.L23]
        )
        self.l4_neurons = NeuronFactory.create_cortical_layer_neurons(
            region_name=self.region_name,
            population_name=CortexPopulation.L4_PYR,
            n_neurons=self.l4_pyr_size,
            device=device,
            **cfg.layer_overrides[CortexLayer.L4]
        )
        self.l5_neurons = NeuronFactory.create_pyramidal_two_compartment(
            region_name=self.region_name,
            population_name=CortexPopulation.L5_PYR,
            n_neurons=self.l5_pyr_size,
            device=device,
            **cfg.layer_overrides[CortexLayer.L5]
        )
        self.l6a_neurons = NeuronFactory.create_cortical_layer_neurons(
            region_name=self.region_name,
            population_name=CortexPopulation.L6A_PYR,
            n_neurons=self.l6a_pyr_size,
            device=device,
            **cfg.layer_overrides[CortexLayer.L6A]
        )
        self.l6b_neurons = NeuronFactory.create_cortical_layer_neurons(
            region_name=self.region_name,
            population_name=CortexPopulation.L6B_PYR,
            n_neurons=self.l6b_pyr_size,
            device=device,
            **cfg.layer_overrides[CortexLayer.L6B]
        )

        # =====================================================================
        # EXPLICIT INHIBITORY NETWORKS (PV, SST, VIP INTERNEURONS)
        # =====================================================================
        # Replace old FSI-only approach with explicit inhibitory networks containing
        # multiple cell types: PV (basket), SST (Martinotti), and VIP (disinhibitory)
        # Each layer gets its own inhibitory network with E→I, I→E, and I→I connectivity

        # L2/3 inhibitory network
        self.l23_inhibitory = CorticalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=CortexPopulation.L23_INHIBITORY,
            pyr_size=self.l23_pyr_size,
            total_inhib_fraction=0.25,
            dt_ms=cfg.dt_ms,
            device=device,
        )

        # L4 inhibitory network
        self.l4_inhibitory = CorticalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=CortexPopulation.L4_INHIBITORY,
            pyr_size=self.l4_pyr_size,
            total_inhib_fraction=0.25,
            dt_ms=cfg.dt_ms,
            device=device,
        )

        # L5 inhibitory network
        self.l5_inhibitory = CorticalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=CortexPopulation.L5_INHIBITORY,
            pyr_size=self.l5_pyr_size,
            total_inhib_fraction=0.25,
            dt_ms=cfg.dt_ms,
            device=device,
        )

        # L6a inhibitory network
        self.l6a_inhibitory = CorticalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=CortexPopulation.L6A_INHIBITORY,
            pyr_size=self.l6a_pyr_size,
            total_inhib_fraction=0.25,
            dt_ms=cfg.dt_ms,
            device=device,
        )

        # L6b inhibitory network
        self.l6b_inhibitory = CorticalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=CortexPopulation.L6B_INHIBITORY,
            pyr_size=self.l6b_pyr_size,
            total_inhib_fraction=0.25,
            dt_ms=cfg.dt_ms,
            device=device,
        )

        # =====================================================================
        # L4 SST PREDICTION-ERROR INTERNEURON (P1-03: disynaptic L5→L4 relay)
        # =====================================================================
        # Biology: L5 pyramidal cells drive SST+ Martinotti cells in L4 via AMPA.
        # These SST cells then inhibit L4 pyramidal dendrites (GABA_A), completing
        # the prediction pathway without violating Dale’s Law.
        # ~10% of L4 pyramidal count; fast Martinotti-like dynamics (tau_mem 15ms).
        self.l4_sst_pred_size = max(5, int(self.l4_pyr_size * 0.10))
        self.l4_sst_pred_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=CortexPopulation.L4_SST_PRED,
            n_neurons=self.l4_sst_pred_size,
            device=device,
            tau_mem=15.0,
            v_threshold=1.0,
            adapt_increment=0.05,
            tau_adapt=90.0,
        )


        # =====================================================================
        # INTER-LAYER AXONAL DELAYS
        # =====================================================================
        # Create delay buffers for biological signal propagation within layers
        # L4→L2/3 delay: Short vertical projection (~2ms biologically)
        # L2/3→L5: Longer vertical projection (~2ms biologically)
        # L2/3→L6a/L6b: Within column (~2ms biologically)

        # Calculate delay steps
        self._l5_l4_delay_steps: int = int(cfg.l5_to_l4_delay_ms / cfg.dt_ms)
        self._l6_l4_delay_steps: int = int(cfg.l6_to_l4_delay_ms / cfg.dt_ms)
        self._l4_l23_delay_steps: int = int(cfg.l4_to_l23_delay_ms / cfg.dt_ms)
        self._l23_l23_delay_steps: int = int(cfg.l23_to_l23_delay_ms / cfg.dt_ms)
        self._l23_l5_delay_steps: int = int(cfg.l23_to_l5_delay_ms / cfg.dt_ms)
        self._l23_l6a_delay_steps: int = int(cfg.l23_to_l6a_delay_ms / cfg.dt_ms)
        self._l23_l6b_delay_steps: int = int(cfg.l23_to_l6b_delay_ms / cfg.dt_ms)

        max_l23_delay = max(self._l23_l23_delay_steps, self._l23_l5_delay_steps, self._l23_l6a_delay_steps, self._l23_l6b_delay_steps)
        max_l4_delay = self._l4_l23_delay_steps
        max_l5_delay = self._l5_l4_delay_steps
        max_l6a_delay = self._l6_l4_delay_steps
        max_l6b_delay = self._l6_l4_delay_steps

        # Spike state buffers (for plasticity + inhibition)
        self._l23_spike_buffer = CircularDelayBuffer(max_delay=max_l23_delay, size=self.l23_pyr_size, device=device, dtype=torch.bool)
        self._l4_spike_buffer = CircularDelayBuffer(max_delay=max_l4_delay, size=self.l4_pyr_size, device=device, dtype=torch.bool)
        self._l5_spike_buffer = CircularDelayBuffer(max_delay=max_l5_delay, size=self.l5_pyr_size, device=device, dtype=torch.bool)
        self._l6a_spike_buffer = CircularDelayBuffer(max_delay=max_l6a_delay, size=self.l6a_pyr_size, device=device, dtype=torch.bool)
        self._l6b_spike_buffer = CircularDelayBuffer(max_delay=max_l6b_delay, size=self.l6b_pyr_size, device=device, dtype=torch.bool)

        # Membrane delay buffers (for inhibitory networks)
        self._l23_membrane_buffer = CircularDelayBuffer(max_delay=max_l23_delay, size=self.l23_pyr_size, device=device, dtype=torch.float32)
        self._l4_membrane_buffer = CircularDelayBuffer(max_delay=max_l4_delay, size=self.l4_pyr_size, device=device, dtype=torch.float32)
        self._l5_membrane_buffer = CircularDelayBuffer(max_delay=max_l5_delay, size=self.l5_pyr_size, device=device, dtype=torch.float32)
        self._l6a_membrane_buffer = CircularDelayBuffer(max_delay=max_l6a_delay, size=self.l6a_pyr_size, device=device, dtype=torch.float32)
        self._l6b_membrane_buffer = CircularDelayBuffer(max_delay=max_l6b_delay, size=self.l6b_pyr_size, device=device, dtype=torch.float32)

        # =====================================================================
        # HOMEOSTATIC PLASTICITY (Intrinsic Excitability + Synaptic Scaling)
        # =====================================================================
        # Per-layer homeostatic state: firing-rate EMA + synaptic weight-scale
        # buffer.  Unified via NeuralRegion._register_homeostasis() so that
        # _update_homeostasis() / _apply_synaptic_scaling() require only the
        # population name, eliminating the fragile tensor-passing calling
        # convention.  Neurons are looked up lazily (registered below).
        self._register_homeostasis(CortexPopulation.L23_PYR, self.l23_pyr_size, use_synaptic_scaling=True)
        self._register_homeostasis(CortexPopulation.L4_PYR,  self.l4_pyr_size,  use_synaptic_scaling=True)
        self._register_homeostasis(CortexPopulation.L5_PYR,  self.l5_pyr_size,  use_synaptic_scaling=True)
        self._register_homeostasis(CortexPopulation.L6A_PYR, self.l6a_pyr_size, use_synaptic_scaling=True)
        self._register_homeostasis(CortexPopulation.L6B_PYR, self.l6b_pyr_size, use_synaptic_scaling=True)

    def _init_synaptic_weights(self) -> None:
        """Initialize inter-layer weight matrices.

        Feedforward weights need to be strong enough to propagate sparse activity.
        With low initial activity, we use generous weight initialization to ensure
        signal propagation through the cortical layers.
        """
        # Expected number of active inputs based on sparse activity assumptions
        expected_active_l23 = max(1, int(self.l23_pyr_size * 0.10))
        expected_active_l4 = max(1, int(self.l4_pyr_size * 0.15))

        # Feedforward weights: Strong initialization for reliable propagation
        l23_std = 10.0 / expected_active_l23  # Stronger to ensure L2/3 activation from sparse L4 input
        l4_std = 15.0 / expected_active_l4  # Even stronger to ensure L4 can drive L2/3 with sparse input

        # L4 → L2/3: positive excitatory weights
        self._add_internal_connection(
            source_population=CortexPopulation.L4_PYR,
            target_population=CortexPopulation.L23_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l4_pyr_size,
                n_output=self.l23_pyr_size,
                connectivity=1.0,
                mean=0.0,
                std=l4_std,
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.AMPA,
            learning_strategy=self.composite_l23,
        )

        # L2/3 → L5: positive excitatory weights
        self._add_internal_connection(
            source_population=CortexPopulation.L23_PYR,
            target_population=CortexPopulation.L5_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l23_pyr_size,
                n_output=self.l5_pyr_size,
                connectivity=1.0,
                mean=0.0,
                std=l23_std,
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.AMPA,
            learning_strategy=self.composite_l5,
        )

        # L2/3 → L6a: positive excitatory weights (corticothalamic type I → TRN)
        self._add_internal_connection(
            source_population=CortexPopulation.L23_PYR,
            target_population=CortexPopulation.L6A_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l23_pyr_size,
                n_output=self.l6a_pyr_size,
                connectivity=1.0,
                mean=0.0,
                std=l23_std,
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.AMPA,
            learning_strategy=self.composite_l6a,
        )

        # L2/3 → L6b: positive excitatory weights (corticothalamic type II → relay)
        self._add_internal_connection(
            source_population=CortexPopulation.L23_PYR,
            target_population=CortexPopulation.L6B_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l23_pyr_size,
                n_output=self.l6b_pyr_size,
                connectivity=1.0,
                mean=0.0,
                std=l23_std,
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.AMPA,
            learning_strategy=self.composite_l6b,
        )

        # =====================================================================
        # PREDICTIVE CODING: L5/L6 → L4 FEEDBACK
        # =====================================================================
        # Deep layers generate predictions that inhibit L4 when correct.
        # P1-03: Dale's Law compliant disynaptic pathway:
        #   L5_PYR (AMPA) → L4_SST_PRED → L4_PYR (GABA_A)
        # The SST_PRED interneuron relays the prediction without L5 emitting GABA_A.
        #
        # A8: strategy config is created here so it can be passed inline to
        # _add_internal_connection (atomic weight + strategy registration).
        pred_cfg = PredictiveCodingConfig(
            learning_rate=self.config.prediction_learning_rate,
            prediction_delay_steps=1,
        )

        # L5 → L4_SST_PRED: Excitatory relay leg (fixed weights, no learning).
        self._l5_sst_pred_synapse = self._add_internal_connection(
            source_population=CortexPopulation.L5_PYR,
            target_population=CortexPopulation.L4_SST_PRED,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l5_pyr_size,
                n_output=self.l4_sst_pred_size,
                connectivity=1.0,
                mean=0.0,
                std=0.01,
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.AMPA,
        )

        # L4_SST_PRED → L4_PYR: Inhibitory prediction gate (anti-Hebbian).
        # PredictiveCodingStrategy registered atomically with the weight matrix.
        self._sst_pred_l4_synapse = self._add_internal_connection(
            source_population=CortexPopulation.L4_SST_PRED,
            target_population=CortexPopulation.L4_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l4_sst_pred_size,
                n_output=self.l4_pyr_size,
                connectivity=1.0,
                mean=0.0,
                std=0.01,
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.GABA_A,
            learning_strategy=PredictiveCodingStrategy(pred_cfg),
        )

        # L6 (combined) → L4: Inhibitory prediction (anti-Hebbian).
        self._l6_l4_synapse = self._add_internal_connection(
            source_population=CortexPopulation.L6_PYR,
            target_population=CortexPopulation.L4_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l6a_pyr_size + self.l6b_pyr_size,
                n_output=self.l4_pyr_size,
                connectivity=1.0,
                mean=0.0,
                std=0.01,  # Very weak initially to avoid suppressing L4
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.GABA_A,
            learning_strategy=PredictiveCodingStrategy(pred_cfg),
        )

        # =====================================================================
        # L2/3 RECURRENT WEIGHTS
        # =====================================================================
        # Biology: L2/3 pyramidal neurons have extensive recurrent connections
        # critical for attractor dynamics, working memory, and pattern completion.
        # These are the most plastic connections in cortex.
        self._add_internal_connection(
            source_population=CortexPopulation.L23_PYR,
            target_population=CortexPopulation.L23_PYR,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.l23_pyr_size,
                n_output=self.l23_pyr_size,
                connectivity=0.25,
                weight_scale=0.0008,
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.AMPA,
            learning_strategy=self.composite_l23,
        )

    # =========================================================================
    # CORTICAL LAYER PROCESSING HELPERS
    # =========================================================================
    # These methods consolidate common operations across all cortical layers
    # to ensure biological consistency and reduce code duplication.

    def _compute_layer_inhibition(
        self,
        g_exc: torch.Tensor,
        baseline_ratio: float,
        prev_spikes: Optional[torch.Tensor],
        prev_membrane: Optional[torch.Tensor],
        inhibitory_network: CorticalInhibitoryNetwork,
        additional_inhibition: Optional[torch.Tensor] = None,
        feedforward_fsi_excitation: Optional[torch.Tensor] = None,
        return_full_output: bool = False,
        ach_spikes: Optional[torch.Tensor] = None,
        long_range_excitation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Compute total inhibition for a cortical layer.

        All cortical layers follow the same pattern:
        1. Baseline inhibition (ratio of excitation for E-I balance)
        2. Network inhibition (PV/SST/VIP interneurons, causal from t-1)
        3. Optional feedforward inhibition (FSI driven by external input)
        4. Optional additional inhibition (e.g., predictive coding for L4)

        Args:
            g_exc: Excitatory conductance to the layer
            baseline_ratio: E-I balance ratio (typically 0.30, but varies by layer)
            prev_spikes: Previous timestep's spikes (for causal inhibition)
            prev_membrane: Previous timestep's membrane potentials
            inhibitory_network: Layer's inhibitory network (PV/SST/VIP)
            additional_inhibition: Extra inhibition (e.g., predictions)
            feedforward_fsi_excitation: Direct excitation to FSI for feedforward inhibition
            return_full_output: If True, return (g_inh_total, inhib_output_dict) for diagnostics
            ach_spikes: Raw NB ACh spikes for nicotinic receptors on VIP cells (optional).
            long_range_excitation: Top-down / long-range corticocortical input in pyramidal
                space [pyr_size], routed preferentially to VIP cells (L2/3 only).

        Returns:
            Total inhibitory conductance, or (g_inh_total, inhib_output_dict) if return_full_output=True
        """
        # Baseline inhibition: constant E-I ratio
        g_inh_baseline = g_exc * baseline_ratio

        # Network inhibition: causal (from previous timestep)
        if prev_spikes is not None:
            inhib_output = inhibitory_network.forward(
                pyr_spikes=prev_spikes,
                pyr_membrane=prev_membrane,
                external_excitation=g_exc,
                feedforward_excitation=feedforward_fsi_excitation,  # Direct FSI drive
                long_range_excitation=long_range_excitation,
                ach_spikes=ach_spikes,
            )
            g_inh_total = g_inh_baseline + inhib_output["total_inhibition"]
        else:
            # First timestep: baseline only
            g_inh_total = g_inh_baseline
            inhib_output = None

        # Add any additional inhibition (e.g., predictive coding)
        if additional_inhibition is not None:
            g_inh_total = g_inh_total + additional_inhibition

        # Return full output for diagnostics if requested
        if return_full_output:
            return g_inh_total, inhib_output
        else:
            return g_inh_total

    def _integrate_and_spike(
        self,
        g_exc: torch.Tensor,
        g_inh: torch.Tensor,
        neurons: "ConductanceLIF",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split conductances and run single-compartment neuron forward pass.

        All cortical layers use the same conductance-based integration:
        - Split excitation into AMPA (fast) and NMDA (slow)
        - Integrate with GABA_A inhibition
        - Return spikes and membrane potentials

        Args:
            g_exc: Total excitatory conductance
            g_inh: Total inhibitory conductance
            neurons: Layer's single-compartment neuron population

        Returns:
            (spikes, membrane_potentials)
        """
        g_ampa, g_nmda = split_excitatory_conductance(g_exc, nmda_ratio=0.2)
        spikes, membrane = neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda),
            g_gaba_a_input=ConductanceTensor(g_inh),
            g_gaba_b_input=None,
        )

        return spikes, membrane

    def _integrate_and_spike_two_compartment(
        self,
        g_exc_basal: torch.Tensor,
        g_inh_basal: torch.Tensor,
        g_exc_apical: Optional[torch.Tensor],
        neurons: "TwoCompartmentLIF",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split conductances and run two-compartment pyramidal neuron forward pass.

        Routes basal (feedforward / proximal) inputs to the somatic compartment
        and apical (top-down / distal) inputs to the dendritic compartment.
        NMDA Mg²⁺ block is applied at **dendritic voltage** for apical NMDA,
        implementing biologically correct coincidence detection.

        Args:
            g_exc_basal:  Excitatory conductance for basal compartment (L4→L2/3
                          feedforward, recurrent L2/3→L2/3, L2/3→L5).
            g_inh_basal:  Inhibitory conductance for basal compartment (local
                          PV/SST network output).
            g_exc_apical: Excitatory conductance for apical compartment
                          (top-down feedback from higher areas).
                          Pass ``None`` if no apical input this timestep.
            neurons: Two-compartment :class:`TwoCompartmentLIF` population.

        Returns:
            (spikes, V_soma, V_dend)
        """
        from thalia.components.neurons import TwoCompartmentLIF  # local import avoids circular
        assert isinstance(neurons, TwoCompartmentLIF), (
            f"_integrate_and_spike_two_compartment expects TwoCompartmentLIF, got {type(neurons).__name__}"
        )

        # Basal: feedforward / somatic inputs (20% NMDA — standard cortical ratio)
        g_ampa_b, g_nmda_b = split_excitatory_conductance(g_exc_basal, nmda_ratio=0.2)

        # Apical: top-down feedback inputs (30% NMDA apically — higher NMDA proportion
        # in distal dendrites matches biology: Bhatt et al., Spruston 2008)
        if g_exc_apical is not None:
            g_ampa_a, g_nmda_a = split_excitatory_conductance(g_exc_apical, nmda_ratio=0.3)
        else:
            zero = torch.zeros(neurons.n_neurons, device=neurons.device)
            g_ampa_a, g_nmda_a = zero, zero

        spikes, V_soma, V_dend = neurons.forward(
            g_ampa_basal=ConductanceTensor(g_ampa_b),
            g_nmda_basal=ConductanceTensor(g_nmda_b),
            g_gaba_a_basal=ConductanceTensor(g_inh_basal),
            g_gaba_b_basal=None,
            g_ampa_apical=ConductanceTensor(g_ampa_a),
            g_nmda_apical=ConductanceTensor(g_nmda_a),
            g_gaba_a_apical=None,
        )

        return spikes, V_soma, V_dend

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process input through layered cortical circuit.

        Args:
            synaptic_inputs: Point-to-point synaptic connections from thalamus, other cortical areas
            neuromodulator_inputs: Broadcast neuromodulatory signals (DA, NE, ACh)
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        cfg = self.config

        # Raw NB ACh spikes needed after the neuromodulation block for VIP nicotinic receptors.
        # Hoisted here so it is always defined regardless of NEUROMODULATION_DISABLED flag.
        nb_ach_spikes: Optional[torch.Tensor] = None

        if not GlobalConfig.NEUROMODULATION_DISABLED:
            # =====================================================================
            # DOPAMINE RECEPTOR PROCESSING (from VTA)
            # =====================================================================
            # Process VTA dopamine spikes → concentration dynamics
            # CorticalColumn receives layer-specific DA innervation with gradient:
            # - L5: Densest (30% of VTA output) - motor learning, prediction errors
            # - L6: Moderate (13.5%) - corticothalamic feedback modulation
            # - L2/3: Sparse (7.5%) - working memory, attentional modulation
            # - L4: Minimal (3%) - subtle sensory gain modulation
            #
            # Biology: While L5 is the primary target, other layers receive diffuse
            # DA innervation that modulates learning and plasticity. This gradient
            # reflects measured anatomical innervation density.
            vta_da_spikes = self._extract_neuromodulator(neuromodulator_inputs, 'da_mesocortical')
            # Update full receptor array
            da_concentration_full = self.da_receptor.update(vta_da_spikes)
            # Split into per-layer buffers with biological gradient
            offset = 0
            self._da_concentration_l23 = da_concentration_full[offset : offset + self.l23_pyr_size] * 0.075  # 7.5% (25% of L5)
            offset += self.l23_pyr_size
            self._da_concentration_l4 = da_concentration_full[offset : offset + self.l4_pyr_size] * 0.03  # 3% (10% of L5)
            offset += self.l4_pyr_size
            self._da_concentration_l5 = da_concentration_full[offset : offset + self.l5_pyr_size] * 0.30  # 30% (primary)
            offset += self.l5_pyr_size
            self._da_concentration_l6a = da_concentration_full[offset : offset + self.l6a_pyr_size] * 0.135  # 13.5% (45% of L5)
            offset += self.l6a_pyr_size
            self._da_concentration_l6b = da_concentration_full[offset :] * 0.135  # 13.5% (45% of L5)

            # =====================================================================
            # NOREPINEPHRINE RECEPTOR PROCESSING (from LC)
            # =====================================================================
            # Process LC norepinephrine spikes → gain and arousal modulation
            # All cortical layers receive NE innervation uniformly
            lc_ne_spikes = neuromodulator_inputs.get('ne', None)
            ne_concentration_full = self.ne_receptor.update(lc_ne_spikes)
            # Split into per-layer buffers (all layers receive NE)
            offset = 0
            self._ne_concentration_l23 = ne_concentration_full[offset : offset + self.l23_pyr_size]
            offset += self.l23_pyr_size
            self._ne_concentration_l4 = ne_concentration_full[offset : offset + self.l4_pyr_size]
            offset += self.l4_pyr_size
            self._ne_concentration_l5 = ne_concentration_full[offset : offset + self.l5_pyr_size]
            offset += self.l5_pyr_size
            self._ne_concentration_l6a = ne_concentration_full[offset : offset + self.l6a_pyr_size]
            offset += self.l6a_pyr_size
            self._ne_concentration_l6b = ne_concentration_full[offset :]

            # =====================================================================
            # ACETYLCHOLINE RECEPTOR PROCESSING (from NB)
            # =====================================================================
            # Process NB acetylcholine spikes → encoding/retrieval and attention
            # ACh has layer-specific effects on feedforward vs recurrent processing
            nb_ach_spikes = neuromodulator_inputs.get('ach', None)  # preserved for VIP nicotinic
            ach_concentration_full = self.ach_receptor.update(nb_ach_spikes)
            # Split into per-layer buffers
            offset = 0
            self._ach_concentration_l23 = ach_concentration_full[offset : offset + self.l23_pyr_size]
            offset += self.l23_pyr_size
            self._ach_concentration_l4 = ach_concentration_full[offset : offset + self.l4_pyr_size]
            offset += self.l4_pyr_size
            self._ach_concentration_l5 = ach_concentration_full[offset : offset + self.l5_pyr_size]
            offset += self.l5_pyr_size
            self._ach_concentration_l6a = ach_concentration_full[offset : offset + self.l6a_pyr_size]
            offset += self.l6a_pyr_size
            self._ach_concentration_l6b = ach_concentration_full[offset :]

        # =====================================================================
        # MULTI-SOURCE ROUTING: L4 (bottom-up) vs L2/3 (top-down)
        # =====================================================================
        # Biology: Different cortical layers receive inputs from different sources
        # - L4 receives thalamic/sensory inputs (bottom-up)
        # - L2/3 receives feedback from higher cortical areas (top-down)
        #
        # L4 has TWO explicit populations:
        # - l4_pyr: Pyramidal neurons (excitatory, main processing)
        # - l4_pv: PV interneurons (inhibitory, feedforward inhibition)
        #
        # Use routing key filtering:
        # - Keys starting with "l4_pyr:" target L4 pyramidal (e.g., "l4_pyr:thalamus:relay")
        # - Keys starting with "l4_pv:" target L4 PV interneurons
        # - Keys starting with "l23:" target L2/3 (e.g., "l23:hippocampus:ca1")

        # Integrate L4 pyramidal inputs
        l4_g_exc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l4_pyr_size,
            filter_by_target_population=CortexPopulation.L4_PYR,
        ).g_ampa

        # Integrate L4 PV inputs (feedforward inhibition drive)
        # Biology: Thalamic afferents synapse directly onto PV cells for fast inhibition
        l4_g_exc_pv = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l4_inhibitory.pv_size,
            filter_by_target_population=CortexPopulation.L4_INHIBITORY_PV,
        ).g_ampa

        # Add baseline noise (spontaneous miniature EPSPs)
        # Use stochastic noise to prevent inhibition-driven silence
        if cfg.baseline_noise_conductance_enabled:
            stochastic = 0.007 * torch.randn(self.l4_pyr_size, device=self.device)
            l4_g_exc = l4_g_exc + stochastic.abs()

        # =====================================================================
        # PREDICTIVE CODING: L5/L6 → L4 INHIBITORY PREDICTIONS
        # =====================================================================
        # Deep layers predict what L4 should receive
        # Good prediction → strong inhibition → L4 silent (no error)
        # Bad prediction → weak inhibition → L4 fires (error signal)
        # This makes L4 naturally compute: error = input - prediction

        # L5 → L4 prediction
        l5_delayed = self._l5_spike_buffer.read(self._l5_l4_delay_steps)

        # --- DISYNAPTIC PREDICTION PATHWAY (P1-03: Dale’s Law compliant) ---
        # Step 1: L5_PYR (AMPA) → L4_SST_PRED: drive the prediction interneuron.
        l5_sst_pred_conductance = torch.matmul(
            self.get_synaptic_weights(self._l5_sst_pred_synapse), l5_delayed.float()
        )
        sst_pred_g_ampa, sst_pred_g_nmda = split_excitatory_conductance(
            l5_sst_pred_conductance, nmda_ratio=0.1
        )
        l4_sst_pred_spikes, _ = self.l4_sst_pred_neurons.forward(
            g_ampa_input=ConductanceTensor(sst_pred_g_ampa),
            g_nmda_input=ConductanceTensor(sst_pred_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        # Step 2: L4_SST_PRED (GABA_A) → L4_PYR: apply prediction as inhibition.
        l5_l4_inhibition = torch.matmul(
            self.get_synaptic_weights(self._sst_pred_l4_synapse), l4_sst_pred_spikes.float()
        )

        # Precision weighting: Scale by L5 activity (confidence)
        l5_activity = torch.mean(l5_delayed.float()).item()  # Population activity [0, 1]
        l5_precision = cfg.precision_min + (cfg.precision_max - cfg.precision_min) * l5_activity
        l5_l4_inhibition = l5_l4_inhibition * l5_precision

        # L6 (combined) → L4 prediction (uses delayed L6 activity from buffer)
        l6a_delayed = self._l6a_spike_buffer.read(self._l6_l4_delay_steps)
        l6b_delayed = self._l6b_spike_buffer.read(self._l6_l4_delay_steps)
        l6_delayed = torch.cat([l6a_delayed, l6b_delayed], dim=-1)

        # Apply L6 prediction as inhibition
        l6_l4_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CortexPopulation.L6_PYR,
            target_region=self.region_name,
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.GABA_A,
        )
        l6_l4_inhibition = torch.matmul(self.get_synaptic_weights(l6_l4_synapse), l6_delayed.float())

        # Precision weighting: Scale by L6 activity (confidence)
        l6_activity = torch.mean(l6_delayed.float()).item()  # Population activity [0, 1]
        l6_precision = cfg.precision_min + (cfg.precision_max - cfg.precision_min) * l6_activity
        l6_l4_inhibition = l6_l4_inhibition * l6_precision

        # Combine L5 and L6 predictions for total predictive inhibition to L4
        l4_pred_inhibition = l5_l4_inhibition + l6_l4_inhibition

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more responsive
        # Low NE (baseline): Normal gain
        # Biological: β-adrenergic receptors increase neuronal excitability
        ne_level = self._ne_concentration_l4.mean().item()  # Average across L4 neurons
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = compute_ne_gain(ne_level)
        l4_g_exc = l4_g_exc * ne_gain

        # =====================================================================
        # L4 INHIBITORY NETWORK (PV, SST, VIP INTERNEURONS)
        # =====================================================================
        # CRITICAL: Compute inhibition from PREVIOUS timestep's activity (causal)
        # Inhibition prevents spikes from occurring, not post-hoc filtering
        #
        # Biology: PV cells fire in response to pyramidal activity at t-1,
        # then inhibit pyramidal neurons at t (1-2ms delay).
        # This creates competitive dynamics where winners suppress losers.
        #
        # FEEDFORWARD INHIBITION: FSI neurons also receive direct thalamic input
        # for rapid feedforward inhibition (1ms faster than feedback inhibition).

        # Compute total inhibition using helper (baseline + network + predictions)
        # Request full output to capture PV spikes for diagnostics
        # L4 uses stronger baseline inhibition (0.50) due to dense thalamic input
        l4_g_inh, l4_inhib_output = self._compute_layer_inhibition(
            g_exc=l4_g_exc,
            baseline_ratio=0.50,  # Higher than standard (0.30) to control strong thalamic drive
            prev_spikes=self._l4_spike_buffer.read(1),
            prev_membrane=self._l4_membrane_buffer.read(1),
            inhibitory_network=self.l4_inhibitory,
            additional_inhibition=l4_pred_inhibition,  # Predictive coding
            feedforward_fsi_excitation=l4_g_exc_pv,  # Direct thalamic input to PV cells
            return_full_output=True,  # Get full output for diagnostics
            ach_spikes=nb_ach_spikes,
        )

        # Store PV spikes for diagnostics (None on first timestep)
        if l4_inhib_output is not None:
            self._l4_pv_spikes = l4_inhib_output["pv_spikes"]
        else:
            self._l4_pv_spikes = torch.zeros(self.l4_inhibitory.pv_size, dtype=torch.bool, device=self.device)

        # Integrate conductances and generate spikes using helper
        l4_spikes, l4_membrane = self._integrate_and_spike(
            g_exc=l4_g_exc,
            g_inh=l4_g_inh,
            neurons=self.l4_neurons,
        )

        assert l4_spikes.shape == (self.l4_pyr_size,), (
            f"CorticalColumn: L4 spikes have shape {l4_spikes.shape} "
            f"but expected ({self.l4_pyr_size},). "
            f"Check input→L4 weights shape."
        )

        # =====================================================================
        # APPLY L4→L2/3 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for L4→L2/3 vertical projection
        l4_spikes_delayed = self._l4_spike_buffer.read(self._l4_l23_delay_steps)

        # L2/3: Processing with recurrence
        l4_l23_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CortexPopulation.L4_PYR,
            target_region=self.region_name,
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        )
        l23_ff = torch.matmul(self.get_synaptic_weights(l4_l23_synapse), l4_spikes_delayed.float())

        # Stimulus gating (transient inhibition - always enabled)
        # Compute total input activity from L4-targeted sources only
        # Biology: FFI is driven by thalamic inputs to L4, not all cortical inputs
        l4_source_activity = 0.0
        for synapse_id, source_spikes in synaptic_inputs.items():
            # Check if this input targets L4 (any L4 sub-population)
            if synapse_id.target_population.startswith(CortexPopulation.L4):
                l4_source_activity += source_spikes.float().sum()

        if l4_source_activity > 0:
            # Normalize by l4_pyr_size to get average activity per neuron
            # TODO: Why are we adding all sup-populations but normalizing by l4_pyr_size only? Should we consider inhibitory populations too?
            gating_input = torch.full((self.l4_pyr_size,), l4_source_activity / self.l4_pyr_size, device=self.device)
        else:
            gating_input = torch.zeros(self.l4_pyr_size, device=self.device)

        ffi = self.stimulus_gating.compute(gating_input, return_tensor=False)
        raw_ffi = ffi.item() if hasattr(ffi, "item") else float(ffi)
        effective_ffi_strength = min(1.0, raw_ffi / self.stimulus_gating.max_inhibition) * cfg.ffi_strength
        ffi_suppression = 1.0 - effective_ffi_strength

        # =====================================================================
        # RECURRENT PROCESSING IN L2/3 (Local Horizontal Connections)
        # =====================================================================
        # ACh modulation applied here (region-level neuromodulation):
        # High ACh (encoding): Suppress recurrence to prevent interference
        # Low ACh (retrieval): Enable recurrence for pattern completion
        ach_level = self._ach_concentration_l23.mean().item()  # Average across L2/3 neurons
        ach_recurrent_modulation = compute_ach_recurrent_suppression(ach_level)

        # =====================================================================
        # L2/3 RECURRENT INPUT (Internal Weights with Plasticity)
        # =====================================================================
        # Use internal recurrent weights (l23_l23) that undergo STDP+BCM learning
        # Biology: Recurrent connections are highly plastic and critical for
        # cortical dynamics (attractor states, working memory, prediction)
        l23_delayed = self._l23_spike_buffer.read(self._l23_l23_delay_steps)

        if l23_delayed is not None:
            # Apply recurrent weights: [l23_pyr_size, l23_pyr_size] @ [l23_pyr_size] → [l23_pyr_size]
            l23_l23_synapse = SynapseId(
                source_region=self.region_name,
                source_population=CortexPopulation.L23_PYR,
                target_region=self.region_name,
                target_population=CortexPopulation.L23_PYR,
                receptor_type=ReceptorType.AMPA,
            )
            l23_rec_raw = self.get_synaptic_weights(l23_l23_synapse) @ l23_delayed.float()
            # Apply modulations
            l23_rec = l23_rec_raw * ffi_suppression * ach_recurrent_modulation
        else:
            # First timestep: no previous activity
            l23_rec = torch.zeros_like(l23_ff)

        # =====================================================================
        # TOP-DOWN MODULATION TO L2/3 (Multi-Source)
        # =====================================================================
        # Biology: L2/3 receives direct feedback from higher cortical areas
        # (e.g., prefrontal, parietal) that modulates processing without going
        # through L4. This implements predictive coding and attentional modulation.

        # Use base class helper for top-down integration
        # Note: No alpha suppression on top-down (already processed by PFC)
        l23_td = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l23_pyr_size,
            filter_by_target_population=CortexPopulation.L23_PYR,
        ).g_ampa

        # =====================================================================
        # TWO-COMPARTMENT ROUTING: Basal (proximal) vs Apical (distal)
        # =====================================================================
        # Basal compartment: feedforward (L4→L2/3) + lateral recurrent (L2/3→L2/3)
        # These synapse on proximal/basal dendrites of L2/3 pyramidal cells.
        # Apical compartment: top-down feedback from higher cortical areas.
        # These synapse on distal apical tufts in L1.
        l23_basal_input = l23_ff + l23_rec

        # Add baseline noise to basal compartment (miniature EPSPs at proximal synapses)
        if cfg.baseline_noise_conductance_enabled:
            stochastic = 0.007 * torch.randn(self.l23_pyr_size, device=self.device)
            l23_basal_input = l23_basal_input + stochastic.abs()

        # Norepinephrine gain modulation applied to BOTH compartments
        ne_level_l23 = self._ne_concentration_l23.mean().item()
        ne_gain_l23 = compute_ne_gain(ne_level_l23)
        l23_basal_input  = l23_basal_input * ne_gain_l23
        l23_apical_input = l23_td * ne_gain_l23

        # Combined input for inhibition computation (inhibitory network doesn't
        # distinguish compartments; it responds to total somatic depolarisation)
        l23_input = l23_basal_input + l23_apical_input

        # INTRINSIC PLASTICITY: Apply per-neuron threshold adjustment
        # Under-firing neurons get lower thresholds (more excitable)
        # Over-firing neurons get higher thresholds (less excitable)
        # This is applied temporarily for this timestep only - doesn't modify stored thresholds

        # Temporarily adjust thresholds for this forward pass
        self.l23_neurons.adjust_thresholds(self._l23_threshold_offset, cfg.threshold_min, cfg.threshold_max)

        # =====================================================================
        # L2/3 INHIBITORY NETWORK (CAUSAL)
        # =====================================================================
        # Compute inhibition from PREVIOUS timestep, then integrate with excitation

        # Compute excitatory conductance from all L2/3 inputs
        l23_g_exc = F.relu(l23_input)

        # Compute inhibition using helper (baseline + network)
        l23_g_inh = self._compute_layer_inhibition(
            g_exc=l23_g_exc,
            baseline_ratio=0.30,
            prev_spikes=self._l23_spike_buffer.read(1),
            prev_membrane=self._l23_membrane_buffer.read(1),
            inhibitory_network=self.l23_inhibitory,
            ach_spikes=nb_ach_spikes,
            long_range_excitation=l23_td,  # S2-3: top-down input gates VIP disinhibition
        )

        # Two-compartment integration: feedforward+recurrent → basal, top-down → apical
        l23_spikes, l23_membrane, _l23_V_dend = self._integrate_and_spike_two_compartment(
            g_exc_basal=F.relu(l23_basal_input),
            g_inh_basal=l23_g_inh,
            g_exc_apical=F.relu(l23_apical_input),
            neurons=self.l23_neurons,
        )

        assert l23_spikes.shape == (self.l23_pyr_size,), (
            f"CorticalColumn: L2/3 spikes have shape {l23_spikes.shape} "
            f"but expected ({self.l23_pyr_size},). "
            f"Check L4→L2/3 weights shape."
        )

        # =====================================================================
        # APPLY L2/3→L5 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for L2/3→L5 vertical projection
        l23_spikes_delayed = self._l23_spike_buffer.read(self._l23_l5_delay_steps)

        # L5: Subcortical output (conductance-based)
        # NOTE: Use delayed L2/3 spikes for biological accuracy
        l23_l5_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CortexPopulation.L23_PYR,
            target_region=self.region_name,
            target_population=CortexPopulation.L5_PYR,
            receptor_type=ReceptorType.AMPA,
        )
        l5_g_exc = torch.matmul(self.get_synaptic_weights(l23_l5_synapse), l23_spikes_delayed.float())

        # Add baseline noise (spontaneous miniature EPSPs)
        # Use stochastic noise to prevent inhibition-driven silence
        if cfg.baseline_noise_conductance_enabled:
            stochastic = 0.007 * torch.randn(self.l5_pyr_size, device=self.device)
            l5_g_exc = l5_g_exc + stochastic.abs()

        # Norepinephrine gain modulation (arousal/uncertainty)
        ne_level_l5 = self._ne_concentration_l5.mean().item()
        ne_gain_l5 = compute_ne_gain(ne_level_l5)
        l5_g_exc = l5_g_exc * ne_gain_l5

        # Compute inhibition using helper (baseline + network)
        l5_g_inh = self._compute_layer_inhibition(
            g_exc=l5_g_exc,
            baseline_ratio=0.30,
            prev_spikes=self._l5_spike_buffer.read(1),
            prev_membrane=self._l5_membrane_buffer.read(1),
            inhibitory_network=self.l5_inhibitory,
            ach_spikes=nb_ach_spikes,
        )

        # TOP-DOWN MODULATION TO L5 (Apical dendrite)
        # Biology: L5 apical tufts in L1 receive top-down feedback from higher
        # cortical areas (PFC, association cortex, contralateral cortex), mirroring
        # the L2/3 apical compartment mechanism. NE gain is applied equally here.
        l5_td = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l5_pyr_size,
            filter_by_target_population=CortexPopulation.L5_PYR,
        ).g_ampa
        l5_apical_input = F.relu(l5_td * ne_gain_l5)

        # Integrate conductances and generate spikes using helper
        # L5 uses two-compartment neurons: L2/3 input targets basal dendrites,
        # top-down cortical feedback targets apical tufts.
        l5_spikes, l5_membrane, _l5_V_dend = self._integrate_and_spike_two_compartment(
            g_exc_basal=l5_g_exc,
            g_inh_basal=l5_g_inh,
            g_exc_apical=l5_apical_input,
            neurons=self.l5_neurons,
        )

        assert l5_spikes.shape == (self.l5_pyr_size,), (
            f"CorticalColumn: L5 spikes have shape {l5_spikes.shape} "
            f"but expected ({self.l5_pyr_size},). "
            f"Check L2/3→L5 weights shape."
        )

        # =====================================================================
        # LAYER 6: CORTICOTHALAMIC FEEDBACK (SPLIT INTO L6a AND L6b)
        # =====================================================================
        # L6a (corticothalamic type I): Projects to TRN (inhibitory modulation, low gamma)
        # L6b (corticothalamic type II): Projects to relay (excitatory modulation, high gamma)
        # This dual feedback implements both spatial attention (via TRN) and
        # fast gain modulation (direct relay excitation).

        # =====================================================================
        # L6a: Apply L2/3→L6a axonal delay
        # =====================================================================
        l23_spikes_for_l6a = self._l23_spike_buffer.read(self._l23_l6a_delay_steps)

        # L6a forward pass (corticothalamic type I → TRN)
        l23_l6a_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CortexPopulation.L23_PYR,
            target_region=self.region_name,
            target_population=CortexPopulation.L6A_PYR,
            receptor_type=ReceptorType.AMPA,
        )
        l6a_g_exc = torch.matmul(self.get_synaptic_weights(l23_l6a_synapse), l23_spikes_for_l6a.float())

        # Norepinephrine gain modulation (arousal/uncertainty)
        ne_level_l6a = self._ne_concentration_l6a.mean().item()
        ne_gain_l6a = compute_ne_gain(ne_level_l6a)
        l6a_g_exc = l6a_g_exc * ne_gain_l6a

        # Compute inhibition using helper (baseline + network)
        # L6a: Strong inhibition (0.8) for sparse, low-gamma firing
        l6a_g_inh = self._compute_layer_inhibition(
            g_exc=l6a_g_exc,
            baseline_ratio=0.80,  # Strong inhibition → sparse firing
            prev_spikes=self._l6a_spike_buffer.read(1),
            prev_membrane=self._l6a_membrane_buffer.read(1),
            inhibitory_network=self.l6a_inhibitory,
            ach_spikes=nb_ach_spikes,
        )

        # Integrate conductances and generate spikes using helper
        l6a_spikes, l6a_membrane = self._integrate_and_spike(
            g_exc=l6a_g_exc,
            g_inh=l6a_g_inh,
            neurons=self.l6a_neurons,
        )

        assert l6a_spikes.shape == (self.l6a_pyr_size,), (
            f"CorticalColumn: L6a spikes have shape {l6a_spikes.shape} "
            f"but expected ({self.l6a_pyr_size},)."
        )

        # =====================================================================
        # L6b: Apply L2/3→L6b axonal delay
        # =====================================================================
        l23_spikes_for_l6b = self._l23_spike_buffer.read(self._l23_l6b_delay_steps)

        # L6b forward pass (corticothalamic type II → relay)
        l23_l6b_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CortexPopulation.L23_PYR,
            target_region=self.region_name,
            target_population=CortexPopulation.L6B_PYR,
            receptor_type=ReceptorType.AMPA,
        )
        l6b_g_exc = torch.matmul(self.get_synaptic_weights(l23_l6b_synapse), l23_spikes_for_l6b.float())

        # Norepinephrine gain modulation (arousal/uncertainty)
        ne_level_l6b = self._ne_concentration_l6b.mean().item()
        ne_gain_l6b = compute_ne_gain(ne_level_l6b)
        l6b_g_exc = l6b_g_exc * ne_gain_l6b

        # Compute inhibition using helper (baseline + network)
        # L6b: Minimal inhibition (0.15) for dense, high-gamma firing
        l6b_g_inh = self._compute_layer_inhibition(
            g_exc=l6b_g_exc,
            baseline_ratio=0.15,  # Minimal inhibition → dense firing
            prev_spikes=self._l6b_spike_buffer.read(1),
            prev_membrane=self._l6b_membrane_buffer.read(1),
            inhibitory_network=self.l6b_inhibitory,
            ach_spikes=nb_ach_spikes,
        )

        # Integrate conductances and generate spikes using helper
        l6b_spikes, l6b_membrane = self._integrate_and_spike(
            g_exc=l6b_g_exc,
            g_inh=l6b_g_inh,
            neurons=self.l6b_neurons,
        )

        assert l6b_spikes.shape == (self.l6b_pyr_size,), (
            f"CorticalColumn: L6b spikes have shape {l6b_spikes.shape} "
            f"but expected ({self.l6b_pyr_size},)."
        )

        # =====================================================================
        # POST-PROCESSING: HOMEOSTASIS AND PLASTICITY
        # =====================================================================
        region_outputs: RegionOutput = {
            CortexPopulation.L23_PYR: l23_spikes,
            CortexPopulation.L4_PYR: l4_spikes,
            CortexPopulation.L5_PYR: l5_spikes,
            CortexPopulation.L6A_PYR: l6a_spikes,
            CortexPopulation.L6B_PYR: l6b_spikes,
        }

        # Homeostatic adaptation for all layers (intrinsic excitability + synaptic scaling)
        self._apply_all_population_homeostasis(region_outputs)

        # Apply synaptic plasticity based on current spikes and neuromodulator levels
        self._apply_plasticity(synaptic_inputs, region_outputs, l4_sst_pred_spikes)

        # =====================================================================
        # UPDATE STATE BUFFERS FOR NEXT TIMESTEP
        # =====================================================================
        # Write spikes to state buffers
        self._l23_spike_buffer.write_and_advance(l23_spikes)
        self._l4_spike_buffer.write_and_advance(l4_spikes)
        self._l5_spike_buffer.write_and_advance(l5_spikes)
        self._l6a_spike_buffer.write_and_advance(l6a_spikes)
        self._l6b_spike_buffer.write_and_advance(l6b_spikes)

        # Write membrane potentials to state buffers
        self._l23_membrane_buffer.write_and_advance(l23_membrane)
        self._l4_membrane_buffer.write_and_advance(l4_membrane)
        self._l5_membrane_buffer.write_and_advance(l5_membrane)
        self._l6a_membrane_buffer.write_and_advance(l6a_membrane)
        self._l6b_membrane_buffer.write_and_advance(l6b_membrane)

        return self._post_forward(region_outputs)

    def _apply_plasticity(self, synaptic_inputs: SynapticInput, region_outputs: RegionOutput, l4_sst_pred_spikes: torch.Tensor) -> None:
        """Apply continuous STDP learning with BCM modulation.

        This is called automatically at each forward() timestep.

        In biological cortex, synaptic plasticity happens continuously based on
        pre/post spike timing. Dopamine doesn't trigger learning - it modulates
        how much weight change occurs from the spike-timing-based plasticity.
        """
        if GlobalConfig.LEARNING_DISABLED:
            return

        cfg = self.config

        l23_spikes = region_outputs[CortexPopulation.L23_PYR]
        l4_spikes = region_outputs[CortexPopulation.L4_PYR]
        l5_spikes = region_outputs[CortexPopulation.L5_PYR]
        l6a_spikes = region_outputs[CortexPopulation.L6A_PYR]
        l6b_spikes = region_outputs[CortexPopulation.L6B_PYR]

        # =====================================================================
        # LAYER-SPECIFIC DOPAMINE MODULATION
        # =====================================================================
        l23_dopamine = self._da_concentration_l23.mean().item()
        l4_dopamine = self._da_concentration_l4.mean().item()
        l5_dopamine = self._da_concentration_l5.mean().item()
        l6a_dopamine = self._da_concentration_l6a.mean().item()
        l6b_dopamine = self._da_concentration_l6b.mean().item()

        # Route input spikes to appropriate layers for plasticity
        # NOTE: Only pyramidal neurons undergo BCM learning (PV weights are fixed)
        for synapse_id, source_spikes in synaptic_inputs.items():
            if synapse_id.target_population == CortexPopulation.L23_PYR:
                if self.get_learning_strategy(synapse_id) is None:
                    self._add_learning_strategy(synapse_id, self.composite_l23)

            elif synapse_id.target_population == CortexPopulation.L4_PYR:
                if self.get_learning_strategy(synapse_id) is None:
                    self._add_learning_strategy(synapse_id, self.composite_l4)

            elif synapse_id.target_population == CortexPopulation.L5_PYR:
                if self.get_learning_strategy(synapse_id) is None:
                    self._add_learning_strategy(synapse_id, self.composite_l5)

        # =====================================================================
        # L4 → L2/3 - L2/3-specific dopamine
        # =====================================================================
        l4_l23_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CortexPopulation.L4_PYR,
            target_region=self.region_name,
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        )
        self._apply_learning(
            l4_l23_synapse, l4_spikes, l23_spikes,
            dopamine=l23_dopamine,
            acetylcholine=self._ach_concentration_l23.mean().item(),
            norepinephrine=self._ne_concentration_l23.mean().item(),
        )

        # =====================================================================
        # L2/3 → L5 - L5-specific dopamine (highest modulation)
        # =====================================================================
        l23_l5_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CortexPopulation.L23_PYR,
            target_region=self.region_name,
            target_population=CortexPopulation.L5_PYR,
            receptor_type=ReceptorType.AMPA,
        )
        self._apply_learning(
            l23_l5_synapse, l23_spikes, l5_spikes,
            dopamine=l5_dopamine,
            acetylcholine=self._ach_concentration_l5.mean().item(),
            norepinephrine=self._ne_concentration_l5.mean().item(),
        )

        # =====================================================================
        # L2/3 → L6a (corticothalamic type I → TRN) - L6-specific dopamine
        # =====================================================================
        l23_l6a_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CortexPopulation.L23_PYR,
            target_region=self.region_name,
            target_population=CortexPopulation.L6A_PYR,
            receptor_type=ReceptorType.AMPA,
        )
        self._apply_learning(
            l23_l6a_synapse, l23_spikes, l6a_spikes,
            dopamine=l6a_dopamine,
            acetylcholine=self._ach_concentration_l6a.mean().item(),
            norepinephrine=self._ne_concentration_l6a.mean().item(),
        )

        # =====================================================================
        # L2/3 → L6b (corticothalamic type II → relay) - L6-specific dopamine
        # =====================================================================
        l23_l6b_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CortexPopulation.L23_PYR,
            target_region=self.region_name,
            target_population=CortexPopulation.L6B_PYR,
            receptor_type=ReceptorType.AMPA,
        )
        self._apply_learning(
            l23_l6b_synapse, l23_spikes, l6b_spikes,
            dopamine=l6b_dopamine,
            acetylcholine=self._ach_concentration_l6b.mean().item(),
            norepinephrine=self._ne_concentration_l6b.mean().item(),
        )

        # =====================================================================
        # L2/3 RECURRENT LEARNING (Critical for Cortical Plasticity)
        # =====================================================================
        # Biology: L2/3 recurrent synapses are the MOST plastic in cortex.
        # They undergo robust STDP and BCM-style homeostasis.
        # Critical for: attractor formation, working memory, predictive coding
        prev_l23_spikes = self._l23_spike_buffer.read(1)
        l23_l23_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CortexPopulation.L23_PYR,
            target_region=self.region_name,
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        )
        self._apply_learning(
            l23_l23_synapse, prev_l23_spikes, l23_spikes,
            dopamine=l23_dopamine,
            acetylcholine=self._ach_concentration_l23.mean().item(),
            norepinephrine=self._ne_concentration_l23.mean().item(),
        )
        self.get_synaptic_weights(l23_l23_synapse).data.fill_diagonal_(0.0)  # Maintain biological constraints (No self-connections)

        # =================================================================
        # INTRINSIC PLASTICITY: Update per-neuron threshold offsets
        # =================================================================
        # Update activity history (exponential moving average)
        self._l23_activity_history = 0.99 * self._l23_activity_history + (1.0 - 0.99) * l23_spikes.float()

        # Compute threshold modulation
        threshold_mod = compute_excitability_modulation(
            activity_history=self._l23_activity_history,
            activity_target=cfg.activity_target,
            tau=100.0,
        )
        # Convert gain modulation to threshold offset:
        # Under-firing → mod > 1.0 → negative offset → lower threshold → easier to spike
        # Over-firing → mod < 1.0 → positive offset → higher threshold → harder to spike
        # Formula: offset = (1.0 - modulation) inverts the sign correctly
        self._l23_threshold_offset = (1.0 - threshold_mod).clamp(-0.5, 0.5)

        # =====================================================================
        # PREDICTIVE CODING: ANTI-HEBBIAN LEARNING
        # =====================================================================
        # L5/L6 → L4 weights learn to predict and suppress L4 activity
        # When prediction correct: L4 silent → no weight change (good)
        # When prediction wrong: L4 fires → strengthen inhibition (learn)
        # This is anti-Hebbian: co-activation → increase inhibition

        # Dopamine modulation of prediction learning (scalar effective learning rate)
        pred_lr = cfg.prediction_learning_rate * (1.0 + l5_dopamine)

        # SST_PRED → L4 anti-Hebbian (PredictiveCodingStrategy on the inhibitory leg).
        # pre = SST_PRED spikes (expressed prediction); post = L4 spikes (prediction error).
        # Δw > 0 when SST_PRED fires and L4 still fires → strengthen prediction gate.
        self._apply_learning(
            self._sst_pred_l4_synapse,
            pre_spikes=l4_sst_pred_spikes,
            post_spikes=l4_spikes,
            learning_rate_override=pred_lr,
        )

        # L6 (combined) → L4 anti-Hebbian: concatenate current L6a+L6b as pre-spikes.
        l6_combined = torch.cat([l6a_spikes, l6b_spikes], dim=-1)
        self._apply_learning(
            self._l6_l4_synapse,
            pre_spikes=l6_combined,
            post_spikes=l4_spikes,
            learning_rate_override=pred_lr,
        )

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        """Map external synapse target population to layer-specific neuromodulator kwargs."""
        if synapse_id.target_population == CortexPopulation.L23_PYR:
            return {
                "dopamine": self._da_concentration_l23.mean().item(),
                "acetylcholine": self._ach_concentration_l23.mean().item(),
                "norepinephrine": self._ne_concentration_l23.mean().item(),
            }
        if synapse_id.target_population == CortexPopulation.L4_PYR:
            return {
                "dopamine": self._da_concentration_l4.mean().item(),
                "acetylcholine": self._ach_concentration_l4.mean().item(),
                "norepinephrine": self._ne_concentration_l4.mean().item(),
            }
        if synapse_id.target_population == CortexPopulation.L5_PYR:
            return {
                "dopamine": self._da_concentration_l5.mean().item(),
                "acetylcholine": self._ach_concentration_l5.mean().item(),
                "norepinephrine": self._ne_concentration_l5.mean().item(),
            }
        return {}

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Propagates dt update to neurons, STP components, and learning strategies.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)

        # Update neurons
        self.l23_neurons.update_temporal_parameters(dt_ms)
        self.l4_neurons.update_temporal_parameters(dt_ms)
        self.l5_neurons.update_temporal_parameters(dt_ms)
        self.l6a_neurons.update_temporal_parameters(dt_ms)
        self.l6b_neurons.update_temporal_parameters(dt_ms)

        # Update inhibitory networks
        self.l23_inhibitory.update_temporal_parameters(dt_ms)
        self.l4_inhibitory.update_temporal_parameters(dt_ms)
        self.l5_inhibitory.update_temporal_parameters(dt_ms)
        self.l6a_inhibitory.update_temporal_parameters(dt_ms)
        self.l6b_inhibitory.update_temporal_parameters(dt_ms)

        # Update learning strategies (CompositeStrategy propagates to sub-strategies)
        for composite in (self.composite_l23, self.composite_l4, self.composite_l5,
                          self.composite_l6a, self.composite_l6b):
            composite.update_temporal_parameters(dt_ms)
