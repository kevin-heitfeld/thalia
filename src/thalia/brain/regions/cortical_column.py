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

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from thalia import GlobalConfig
from thalia.brain.configs import CorticalColumnConfig
from thalia.brain.neurons import NeuronFactory
from thalia.brain.synapses import (
    NeuromodulatorReceptor,
    NMReceptorType,
    make_nm_receptor,
    WeightInitializer,
)
from thalia.brain.synapses.stp import (
    STPConfig,
    STPType,
    CORTICAL_FF_PRESET,
    CORTICAL_RECURRENT_PRESET,
)
from thalia.learning import (
    BCMConfig,
    BCMStrategy,
    STDPConfig,
    STDPStrategy,
    CompositeStrategy,
    PredictiveCodingConfig,
    PredictiveCodingStrategy,
)
from thalia.typing import (
    ConductanceTensor,
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
    from thalia.brain.neurons import ConductanceLIF, TwoCompartmentLIF


@register_region(
    "cortical_column",
    description="Multi-layer cortical microcircuit with L2/3/L4/L5/L6a/L6b structure",
    version="1.0",
    author="Thalia Project",
    config_class=CorticalColumnConfig,
)
class CorticalColumn(NeuralRegion[CorticalColumnConfig]):
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
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.DA_MESOCORTICAL,
        NeuromodulatorChannel.NE,
        NeuromodulatorChannel.ACH,
    ]

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: CorticalColumnConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize cortex."""
        super().__init__(config, population_sizes, region_name, device=device)

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
        self._init_layers(device)
        self._init_synaptic_weights(device)

        # Initialize stimulus gating (transient inhibition)
        self.stimulus_gating = StimulusGating(
            n_neurons=self.l4_pyr_size,
            threshold=config.ffi_threshold,
            max_inhibition=config.ffi_strength * 10.0,
            decay_rate=1.0 - (1.0 / config.ffi_tau),
            steepness=10.0,
            device=device,
        )

        # =====================================================================
        # NEUROMODULATOR RECEPTORS (DA, NE, ACh)
        # =====================================================================
        # Per-layer concentration slices are stored as registered buffers so
        # they participate in .to(device) / state_dict() automatically.
        # The layer sizes tuple is re-used by _process_neuromodulator() each step.
        self._layer_sizes: Tuple[int, ...] = (
            self.l23_pyr_size, self.l4_pyr_size, self.l5_pyr_size,
            self.l6a_pyr_size, self.l6b_pyr_size,
        )
        total_neurons = sum(self._layer_sizes)

        # DA: mesocortical VTA → deep layers (L5). D1-dominant in cortex.
        # D1 Gs/PKA cascade: τ_rise=500 ms, τ_decay=8000 ms (long-lasting WM gate).
        self.da_receptor = make_nm_receptor(
            NMReceptorType.DA_D1, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device,
            amplitude_scale=1.25,  # Sparse innervation; slight sensitivity boost
        )
        self._da_concentration_l23: torch.Tensor
        self._da_concentration_l4: torch.Tensor
        self._da_concentration_l5: torch.Tensor
        self._da_concentration_l6a: torch.Tensor
        self._da_concentration_l6b: torch.Tensor
        self._init_nm_buffers("_da_concentration", device)

        # NE: dense LC projection — all layers, α1-adrenergic (Gq).
        self.ne_receptor = make_nm_receptor(
            NMReceptorType.NE_ALPHA1, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device
        )
        self._ne_concentration_l23: torch.Tensor
        self._ne_concentration_l4: torch.Tensor
        self._ne_concentration_l5: torch.Tensor
        self._ne_concentration_l6a: torch.Tensor
        self._ne_concentration_l6b: torch.Tensor
        self._init_nm_buffers("_ne_concentration", device)

        # ACh: nucleus basalis nicotinic α4β2 (ionotropic; fast: τ_rise=3 ms, τ_decay=15 ms).
        self.ach_receptor = make_nm_receptor(
            NMReceptorType.ACH_NICOTINIC, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device,
            amplitude_scale=1.3,  # Preserve legacy ~0.2 effective amplitude
        )
        self._ach_concentration_l23: torch.Tensor
        self._ach_concentration_l4: torch.Tensor
        self._ach_concentration_l5: torch.Tensor
        self._ach_concentration_l6a: torch.Tensor
        self._ach_concentration_l6b: torch.Tensor
        self._init_nm_buffers("_ach_concentration", device)

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        # Layer 1 (above L2/3 in hierarchy; inhibitory by Dale's Law)
        self._register_neuron_population(CortexPopulation.L1_NGC, self.l1_ngc_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L23_PYR, self.l23_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CortexPopulation.L4_PYR, self.l4_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CortexPopulation.L5_PYR, self.l5_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CortexPopulation.L6A_PYR, self.l6a_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CortexPopulation.L6B_PYR, self.l6b_neurons, polarity=PopulationPolarity.EXCITATORY)

        # Prediction-error SST interneuron driven by L5; inhibits L4 dendrites
        self._register_neuron_population(CortexPopulation.L4_SST_PRED, self.l4_sst_pred_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L23_INHIBITORY_PV, self.l23_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L23_INHIBITORY_SST, self.l23_inhibitory.sst_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L23_INHIBITORY_VIP, self.l23_inhibitory.vip_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L23_INHIBITORY_NGC, self.l23_inhibitory.ngc_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L4_INHIBITORY_PV, self.l4_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L4_INHIBITORY_SST, self.l4_inhibitory.sst_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L4_INHIBITORY_VIP, self.l4_inhibitory.vip_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L4_INHIBITORY_NGC, self.l4_inhibitory.ngc_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L5_INHIBITORY_PV, self.l5_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L5_INHIBITORY_SST, self.l5_inhibitory.sst_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L5_INHIBITORY_VIP, self.l5_inhibitory.vip_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L5_INHIBITORY_NGC, self.l5_inhibitory.ngc_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L6A_INHIBITORY_PV, self.l6a_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L6A_INHIBITORY_SST, self.l6a_inhibitory.sst_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L6A_INHIBITORY_VIP, self.l6a_inhibitory.vip_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L6A_INHIBITORY_NGC, self.l6a_inhibitory.ngc_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(CortexPopulation.L6B_INHIBITORY_PV, self.l6b_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L6B_INHIBITORY_SST, self.l6b_inhibitory.sst_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L6B_INHIBITORY_VIP, self.l6b_inhibitory.vip_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CortexPopulation.L6B_INHIBITORY_NGC, self.l6b_inhibitory.ngc_neurons, polarity=PopulationPolarity.INHIBITORY)

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # NEUROMODULATOR HELPERS
    # =========================================================================

    def _init_nm_buffers(self, prefix: str, device: Union[str, torch.device]) -> None:
        """Register per-layer zero buffers for a neuromodulator concentration (e.g. ``'_da_concentration'``)."""
        for suffix, size in zip(("_l23", "_l4", "_l5", "_l6a", "_l6b"), self._layer_sizes):
            self.register_buffer(f"{prefix}{suffix}", torch.zeros(size, device=device))

    def _process_neuromodulator(
        self,
        receptor: NeuromodulatorReceptor,
        spikes: Optional[torch.Tensor],
        prefix: str,
        fractions: Tuple[float, float, float, float, float],
    ) -> None:
        """Update receptor dynamics and slice the result into per-layer buffers.

        Args:
            receptor: Neuromodulator receptor to update.
            spikes: Afferent spike tensor (or ``None`` if no input this step).
            prefix: Attribute-name prefix, e.g. ``'_da_concentration'``.
            fractions: Per-layer scale factors ``(l23, l4, l5, l6a, l6b)``.
        """
        full = receptor.update(spikes)
        offset = 0
        for suffix, size, frac in zip(("_l23", "_l4", "_l5", "_l6a", "_l6b"), self._layer_sizes, fractions):
            setattr(self, f"{prefix}{suffix}", full[offset: offset + size] * frac)
            offset += size

    def _neuromod_scalars(self, layer: str) -> Dict[str, float]:
        """Return ``{dopamine, acetylcholine, norepinephrine}`` scalars for *layer*.

        Args:
            layer: One of ``'l23'``, ``'l4'``, ``'l5'``, ``'l6a'``, ``'l6b'``.
        """
        return {
            "dopamine":       getattr(self, f"_da_concentration_{layer}").mean().item(),
            "acetylcholine":  getattr(self, f"_ach_concentration_{layer}").mean().item(),
            "norepinephrine": getattr(self, f"_ne_concentration_{layer}").mean().item(),
        }

    def _init_layers(self, device: Union[str, torch.device]) -> None:
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
        config = self.config

        # =====================================================================
        # LAYER-SPECIFIC HETEROGENEITY
        # =====================================================================
        self.l23_neurons = NeuronFactory.create_pyramidal_two_compartment(
            region_name=self.region_name,
            population_name=CortexPopulation.L23_PYR,
            n_neurons=self.l23_pyr_size,
            device=device,
            **config.population_overrides[CortexPopulation.L23_PYR]
        )
        self.l4_neurons = NeuronFactory.create_cortical_layer_neurons(
            region_name=self.region_name,
            population_name=CortexPopulation.L4_PYR,
            n_neurons=self.l4_pyr_size,
            device=device,
            **config.population_overrides[CortexPopulation.L4_PYR]
        )
        self.l5_neurons = NeuronFactory.create_pyramidal_two_compartment(
            region_name=self.region_name,
            population_name=CortexPopulation.L5_PYR,
            n_neurons=self.l5_pyr_size,
            device=device,
            **config.population_overrides[CortexPopulation.L5_PYR]
        )
        self.l6a_neurons = NeuronFactory.create_cortical_layer_neurons(
            region_name=self.region_name,
            population_name=CortexPopulation.L6A_PYR,
            n_neurons=self.l6a_pyr_size,
            device=device,
            **config.population_overrides[CortexPopulation.L6A_PYR]
        )
        self.l6b_neurons = NeuronFactory.create_cortical_layer_neurons(
            region_name=self.region_name,
            population_name=CortexPopulation.L6B_PYR,
            n_neurons=self.l6b_pyr_size,
            device=device,
            **config.population_overrides[CortexPopulation.L6B_PYR]
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
            pv_adapt_increment=0.0,
            dt_ms=config.dt_ms,
            device=device,
        )

        # L4 inhibitory network
        self.l4_inhibitory = CorticalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=CortexPopulation.L4_INHIBITORY,
            pyr_size=self.l4_pyr_size,
            total_inhib_fraction=0.25,
            pv_adapt_increment=0.10,
            dt_ms=config.dt_ms,
            device=device,
        )

        # L5 inhibitory network
        self.l5_inhibitory = CorticalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=CortexPopulation.L5_INHIBITORY,
            pyr_size=self.l5_pyr_size,
            total_inhib_fraction=0.25,
            pv_adapt_increment=0.0,
            dt_ms=config.dt_ms,
            device=device,
        )

        # L6a inhibitory network
        self.l6a_inhibitory = CorticalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=CortexPopulation.L6A_INHIBITORY,
            pyr_size=self.l6a_pyr_size,
            total_inhib_fraction=0.25,
            pv_adapt_increment=0.05,
            dt_ms=config.dt_ms,
            device=device,
        )

        # L6b inhibitory network
        self.l6b_inhibitory = CorticalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=CortexPopulation.L6B_INHIBITORY,
            pyr_size=self.l6b_pyr_size,
            total_inhib_fraction=0.25,
            pv_adapt_increment=0.0,
            dt_ms=config.dt_ms,
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
        # LAYER 1: NEUROGLIAFORM (NGC) CELLS
        # =====================================================================
        # Biology: Layer 1 is nearly acellular but contains a sparse population
        # of neurogliaform interneurons (~1% of cortical neurons) that form the
        # only true cellular residents of L1. They receive top-down axons from
        # higher cortical areas and L2/3 apical collaterals, and in turn inhibit
        # the apical tufts of L2/3 and L5 pyramidal cells via GABA_A.
        #
        # Key properties: high input resistance (slow, fire at low rates),
        # wide axonal spread (diffuse apical tuft coverage), and slow dynamics.
        #
        # References:
        #   - Jiang et al. 2013: L1 NGC characterization
        #   - Letzkus et al. 2011: Top-down disinhibitory circuit in L1
        #   - Wester & McBain 2014: L1 interneuron connectivity
        #
        # ~5% of L2/3 pyramidal count; tau_mem=30ms (high input resistance)
        self.l1_ngc_size: int = max(3, int(self.l23_pyr_size * 0.05))
        self.l1_ngc_neurons: ConductanceLIF = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=CortexPopulation.L1_NGC,
            n_neurons=self.l1_ngc_size,
            device=device,
            tau_mem=30.0,      # High input resistance → slow integration
            v_threshold=0.85,  # Slightly lower threshold (easily recruited by TD)
            adapt_increment=0.02,
            tau_adapt=300.0,   # Very slow spike-frequency adaptation
        )

        # =====================================================================
        # INTER-LAYER AXONAL DELAYS
        # =====================================================================
        # Create delay buffers for biological signal propagation within layers
        # L4→L2/3 delay: Short vertical projection (~2ms biologically)
        # L2/3→L5: Longer vertical projection (~2ms biologically)
        # L2/3→L6a/L6b: Within column (~2ms biologically)

        # Calculate delay steps
        self._l5_l4_delay_steps: int = int(config.l5_to_l4_delay_ms / config.dt_ms)
        self._l6_l4_delay_steps: int = int(config.l6_to_l4_delay_ms / config.dt_ms)
        self._l4_l23_delay_steps: int = int(config.l4_to_l23_delay_ms / config.dt_ms)
        self._l23_l23_delay_steps: int = int(config.l23_to_l23_delay_ms / config.dt_ms)
        self._l23_l5_delay_steps: int = int(config.l23_to_l5_delay_ms / config.dt_ms)
        self._l23_l6a_delay_steps: int = int(config.l23_to_l6a_delay_ms / config.dt_ms)
        self._l23_l6b_delay_steps: int = int(config.l23_to_l6b_delay_ms / config.dt_ms)

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
        # L1 NGC spike buffer (max_delay=1: accessed only for plasticity/diagnostics;
        # L1 → L2/3 apical inhibition is applied same-timestep, no axonal delay).
        self._l1_ngc_spike_buffer = CircularDelayBuffer(max_delay=1, size=self.l1_ngc_size, device=device, dtype=torch.bool)

        # =====================================================================
        # HOMEOSTATIC PLASTICITY (Intrinsic Excitability + Synaptic Scaling)
        # =====================================================================
        # Per-layer homeostatic state: firing-rate EMA + synaptic weight-scale
        # buffer.  Unified via NeuralRegion._register_homeostasis() so that
        # _update_homeostasis() / _apply_synaptic_scaling() require only the
        # population name, eliminating the fragile tensor-passing calling
        # convention.  Neurons are looked up lazily (registered below).
        self._register_homeostasis(CortexPopulation.L23_PYR, self.l23_pyr_size, target_firing_rate=0.003, device=device)  # 0–3 Hz
        self._register_homeostasis(CortexPopulation.L4_PYR,  self.l4_pyr_size,  target_firing_rate=0.005, device=device)  # 1–10 Hz
        self._register_homeostasis(CortexPopulation.L5_PYR,  self.l5_pyr_size,  target_firing_rate=0.008, device=device)  # 2–15 Hz
        self._register_homeostasis(CortexPopulation.L6A_PYR, self.l6a_pyr_size, target_firing_rate=0.004, device=device)  # 1–8 Hz
        self._register_homeostasis(CortexPopulation.L6B_PYR, self.l6b_pyr_size, target_firing_rate=0.004, device=device)  # 1–8 Hz
        self._register_homeostasis(CortexPopulation.L1_NGC,  self.l1_ngc_size,  target_firing_rate=0.002, device=device)  # ≤2 Hz (sparse)

    def _init_synaptic_weights(self, device: Union[str, torch.device]) -> None:
        """Initialize inter-layer weight matrices.

        Feedforward weights need to be strong enough to propagate sparse activity.
        With low initial activity, we use generous weight initialization to ensure
        signal propagation through the cortical layers.
        """
        # Expected number of active inputs based on sparse activity assumptions
        expected_active_l23 = max(1, int(self.l23_pyr_size * 0.10))
        expected_active_l4 = max(1, int(self.l4_pyr_size * 0.15))

        l23_std = 0.35 / expected_active_l23  # L2/3 downstream from L4
        l4_std = 0.5 / expected_active_l4    # L4 driving L2/3

        # L4 → L2/3: positive excitatory weights
        # STP: CORTICAL_FF_PRESET — moderate depression (Reyes & Sakmann 1999;
        # Thomson et al. 2002). Limits sustained runaway activation and
        # implements gain normalisation as sparsely coded L4 activity ascends.
        self._add_internal_connection(
            source_population=CortexPopulation.L4_PYR,
            target_population=CortexPopulation.L23_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l4_pyr_size,
                n_output=self.l23_pyr_size,
                connectivity=1.0,
                mean=0.0,
                std=l4_std,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=CORTICAL_FF_PRESET.configure(),
            learning_strategy=self.composite_l23,
        )

        # L2/3 → L5: positive excitatory weights
        # STP: CORTICAL_FF_PRESET — moderate depression. Thomson & Bannister
        # (2003) report L2/3→L5 can also be weakly facilitating; we use the
        # same depressing preset as L4→L23 to prevent cascade runaway.
        self._add_internal_connection(
            source_population=CortexPopulation.L23_PYR,
            target_population=CortexPopulation.L5_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l23_pyr_size,
                n_output=self.l5_pyr_size,
                connectivity=1.0,
                mean=0.0,
                std=l23_std,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=CORTICAL_FF_PRESET.configure(),
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
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=CORTICAL_FF_PRESET.configure(),
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
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=CORTICAL_FF_PRESET.configure(),
            learning_strategy=self.composite_l6b,
        )

        # =====================================================================
        # PREDICTIVE CODING: L5/L6 → L4 FEEDBACK
        # =====================================================================
        # Deep layers generate predictions that inhibit L4 when correct.
        # P1-03: Dale's Law compliant disynaptic pathway:
        #   L5_PYR (AMPA) → L4_SST_PRED → L4_PYR (GABA_A)
        # The SST_PRED interneuron relays the prediction without L5 emitting GABA_A.
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
                std=0.0005,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            # PYR→SOM is facilitating: low initial Pr, builds during L5 burst.
            # Reyes et al. (1998); Markram et al. (1998): E→SOM synapses are
            # the canonical EPSP-F type — enables prediction only during
            # sustained L5 activity, not transient spikes.
            stp_config=STPConfig.from_type(STPType.FACILITATING_MODERATE),
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
                std=0.001,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            # SST→PYR inhibition is depressing: strong initial IPSP that fades
            # with repetitive SST firing. Silberberg & Markram (2007): SST-type
            # inhibitory synapses depress at dendritic targets.
            stp_config=STPConfig.from_type(STPType.DEPRESSING),
            learning_strategy=PredictiveCodingStrategy(pred_cfg),
        )

        # L6 (combined) → L4: Inhibitory prediction (anti-Hebbian).
        # This weight is learned via PredictiveCodingStrategy and should start negligible.
        self._l6_l4_synapse = self._add_internal_connection(
            source_population=CortexPopulation.L6_PYR,
            target_population=CortexPopulation.L4_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l6a_pyr_size + self.l6b_pyr_size,
                n_output=self.l4_pyr_size,
                connectivity=1.0,
                mean=0.0,
                std=0.0003,  # Near-zero initial; learned via PredictiveCodingStrategy
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            # L6 feedback inhibition is depressing: high initial suppression
            # of L4 that self-limits with sustained L6 activity.
            # Prevents permanent L4 silencing by the predictive circuit.
            stp_config=STPConfig.from_type(STPType.DEPRESSING),
            learning_strategy=PredictiveCodingStrategy(pred_cfg),
        )

        # =====================================================================
        # L2/3 RECURRENT WEIGHTS
        # =====================================================================
        # Biology: L2/3 pyramidal neurons have extensive recurrent connections
        # critical for attractor dynamics, working memory, and pattern completion.
        # These are the most plastic connections in cortex.
        # STP: CORTICAL_RECURRENT_PRESET — strong facilitation (Markram et al.
        # 1998: the canonical EPSP-E facilitating synapse). Very low baseline
        # release probability builds up strongly during sustained activity,
        # implementing attractor dynamics and working memory maintenance.
        self._add_internal_connection(
            source_population=CortexPopulation.L23_PYR,
            target_population=CortexPopulation.L23_PYR,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.l23_pyr_size,
                n_output=self.l23_pyr_size,
                connectivity=self.config.l23_recurrent_connectivity,
                weight_scale=self.config.l23_recurrent_weight_scale,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=CORTICAL_RECURRENT_PRESET.configure(),
            learning_strategy=self.composite_l23,
        )

        # =====================================================================
        # INHIBITORY INTERNEURON CONNECTIONS (E→I, I→E, I→I) per layer
        # =====================================================================
        # All PV/SST/VIP/NGC weight matrices registered via _add_internal_connection
        # so they receive STP modulation, diagnostics tracking, and can accept
        # learning strategies.  The biological scale multipliers from the old
        # cortical_inhibitory_network.forward (PV ×1.0, SST ×0.3, VIP ×0.05,
        # NGC ×0.12) are ABSORBED into the weight magnitudes below.
        #
        # STP biology:
        #   Pyr→PV   CORTICAL_FF_PRESET     (depressing, Thomson et al. 2002)
        #   Pyr→SST  FACILITATING_MODERATE  (class E→SOM facilitating, Markram 1998)
        #   Pyr→VIP  CORTICAL_FF_PRESET     (depressing)
        #   Pyr→NGC  CORTICAL_FF_PRESET     (depressing)
        #   PV→Pyr   DEPRESSING             (fast fast-spiking IPSCs depress, Jonas 2004)
        #   SST→Pyr  DEPRESSING             (Silberberg & Markram, 2007)
        #   VIP→Pyr  None                   (minimal direct route, dynamic unimportant)
        #   NGC→Pyr  None                   (GABA_A volume transmission, no vesicle STP)
        #   I→I      CORTICAL_FF_PRESET     (interneuron→interneuron depressing)

        layer_defs = [
            # ( pyr_pop,                    pv_pop,                         sst_pop,                         vip_pop,                         ngc_pop,                         inhib_net,            pyr_size             )
            (CortexPopulation.L23_PYR,  CortexPopulation.L23_INHIBITORY_PV,  CortexPopulation.L23_INHIBITORY_SST,  CortexPopulation.L23_INHIBITORY_VIP,  CortexPopulation.L23_INHIBITORY_NGC,  self.l23_inhibitory,  self.l23_pyr_size),
            (CortexPopulation.L4_PYR,   CortexPopulation.L4_INHIBITORY_PV,   CortexPopulation.L4_INHIBITORY_SST,   CortexPopulation.L4_INHIBITORY_VIP,   CortexPopulation.L4_INHIBITORY_NGC,   self.l4_inhibitory,   self.l4_pyr_size),
            (CortexPopulation.L5_PYR,   CortexPopulation.L5_INHIBITORY_PV,   CortexPopulation.L5_INHIBITORY_SST,   CortexPopulation.L5_INHIBITORY_VIP,   CortexPopulation.L5_INHIBITORY_NGC,   self.l5_inhibitory,   self.l5_pyr_size),
            (CortexPopulation.L6A_PYR,  CortexPopulation.L6A_INHIBITORY_PV,  CortexPopulation.L6A_INHIBITORY_SST,  CortexPopulation.L6A_INHIBITORY_VIP,  CortexPopulation.L6A_INHIBITORY_NGC,  self.l6a_inhibitory,  self.l6a_pyr_size),
            (CortexPopulation.L6B_PYR,  CortexPopulation.L6B_INHIBITORY_PV,  CortexPopulation.L6B_INHIBITORY_SST,  CortexPopulation.L6B_INHIBITORY_VIP,  CortexPopulation.L6B_INHIBITORY_NGC,  self.l6b_inhibitory,  self.l6b_pyr_size),
        ]

        for pyr_pop, pv_pop, sst_pop, vip_pop, ngc_pop, inh, n_pyr in layer_defs:
            n_pv  = inh.pv_size
            n_sst = inh.sst_size
            n_vip = inh.vip_size
            n_ngc = inh.ngc_size

            # -- Weight scales -----------------------------------------------
            # E→I: each interneuron receives from a fraction of pyramidal cells.
            # Target ~30-50 Hz PV drive at expected 5-10 Hz pyramidal rate.
            # Formula: std = g_AMPA_needed / (n_pyr * conn * pyr_rate * tau_E * sqrt(2/pi))
            # For g_L=0.08-0.10 interneurons firing at 20-50 Hz:
            #   PV  (target 30-60 Hz, g_L≈0.10): std = 2.25 / n_pyr
            #   SST (target 10-25 Hz, g_L≈0.07): std = 2.0  / n_pyr
            #   VIP (target 10-30 Hz, g_L≈0.08): std = 0.60 / n_pyr  (kept)
            #   NGC (target  5-25 Hz, g_L≈0.08): std = 4.0  / n_pyr
            #
            # NGC requires the largest increase (was 0.30/n_pyr, all silent):
            # NGC have small local collateral connectivity (P≈0.2) and volume-
            # transmission output; they need stronger per-connection drive.
            #
            # Previous values (0.75, 0.45, 0.60, 0.30)/n_pyr were calibrated
            # for ~2 Hz pyramidal rate; at 5-10 Hz actual V_inf was 0.2-0.5×
            # threshold, leaving PV/SST underactive and NGC completely silent.
            ei_pv_std  = 2.25 / n_pyr   # Strong, reliable (P≈0.5)  — 3× increase
            ei_sst_std = 2.0  / n_pyr   # Facilitating (P≈0.3)      — 4.4× increase
            ei_vip_std = 0.60 / n_pyr   # Strong specific (P≈0.4)   — unchanged
            ei_ngc_std = 10.0 / n_pyr   # Weak collaterals (P≈0.2)  — 2.5× increase from 4.0 (run-03: still silent)

            # I→E: perisomatic PV gives strong inhibition ~2 mS/cm²/spike.
            # SST absorbed scale ×0.3:  0.002 × 0.3 = 0.0006
            # VIP absorbed scale ×0.05: 0.002 × 0.05 = 0.0001
            # NGC absorbed scale ×0.12: 0.002 × 0.12 = 0.00024
            ie_pv_std  = 0.002
            ie_sst_std = 0.0006
            ie_vip_std = 0.0001
            ie_ngc_std = 0.00024

            # I→I: moderate inhibition between interneuron types
            ii_std = 0.001

            # == E → I ========================================================
            # Pyr → PV (depressing, strong, reliable)
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pyr, n_output=n_pv,
                    connectivity=0.5, mean=0.0, std=ei_pv_std, device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=CORTICAL_FF_PRESET.configure(),
            )
            # Pyr → SST (facilitating — Markram 1998 EPSP-F, Reyes et al. 1998)
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=sst_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pyr, n_output=n_sst,
                    connectivity=0.3, mean=0.0, std=ei_sst_std, device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig.from_type(STPType.FACILITATING_MODERATE),
            )
            # Pyr → VIP (depressing)
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=vip_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pyr, n_output=n_vip,
                    connectivity=0.4, mean=0.0, std=ei_vip_std, device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=CORTICAL_FF_PRESET.configure(),
            )
            # Pyr → NGC (no STP: CORTICAL_FF_PRESET depressing STP at 2-6 Hz
            # depletes to ~35-60% efficiency; V_inf is already marginal so
            # any additional depression would silence NGC. Without STP, full
            # weight is delivered at each spike. NGC v_threshold=0.75 allows
            # firing once V_inf exceeds baseline. adapt_increment=0.03 +
            # tau_adapt=100ms then self-regulates NGC to 5-25 Hz.
            # Biology: NGC are late-spiking with minimal E→I depression
            # compared to PV/VIP pathways (Wozny & Williams 2011).)
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=ngc_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pyr, n_output=n_ngc,
                    connectivity=0.2, mean=0.0, std=ei_ngc_std, device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=None,  # No STP: even mild depression silences NGC at low (<5 Hz) pyr rates
            )

            # == I → E ========================================================
            # PV → Pyr GABA_A (perisomatic, fast, depressing — Jonas et al. 2004)
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pv, n_output=n_pyr,
                    connectivity=0.6, mean=0.0, std=ie_pv_std, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig.from_type(STPType.DEPRESSING),
            )
            # SST → Pyr GABA_A (dendritic, depressing — Silberberg & Markram 2007)
            self._add_internal_connection(
                source_population=sst_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_sst, n_output=n_pyr,
                    connectivity=0.4, mean=0.0, std=ie_sst_std, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig.from_type(STPType.DEPRESSING),
            )
            # VIP → Pyr GABA_A (minimal direct effect; VIP role is disinhibitory)
            self._add_internal_connection(
                source_population=vip_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_vip, n_output=n_pyr,
                    connectivity=0.15, mean=0.0, std=ie_vip_std, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=None,
            )
            # NGC → Pyr GABA_A (diffuse apical tuft; volume transmission, no vesicle STP)
            self._add_internal_connection(
                source_population=ngc_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_ngc, n_output=n_pyr,
                    connectivity=0.5, mean=0.0, std=ie_ngc_std, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=None,
            )

            # == I → I ========================================================
            # PV → PV (lateral inhibition via basket cell axon collaterals)
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_gaussian_no_autapses(
                    n_input=n_pv, n_output=n_pv,
                    connectivity=0.3, mean=0.0, std=ii_std, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=CORTICAL_FF_PRESET.configure(),
            )
            # PV → SST (suppresses SST → disinhibits Pyr apical tuft)
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=sst_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pv, n_output=n_sst,
                    connectivity=0.3, mean=0.0, std=ii_std, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=CORTICAL_FF_PRESET.configure(),
            )
            # SST → PV (weak; SST can briefly suppress PV allowing burst propagation)
            self._add_internal_connection(
                source_population=sst_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_sst, n_output=n_pv,
                    connectivity=0.2, mean=0.0, std=ii_std * 0.7, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=CORTICAL_RECURRENT_PRESET.configure(),
            )
            # VIP → PV (disinhibitory: VIP silence PV → releases pyramidal from peri-somatic inh)
            self._add_internal_connection(
                source_population=vip_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_vip, n_output=n_pv,
                    connectivity=0.6, mean=0.0, std=ii_std, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=CORTICAL_FF_PRESET.configure(),
            )
            # VIP → SST (strong disinhibition: VIP→SST fires → SST silenced → apical disinhibition)
            self._add_internal_connection(
                source_population=vip_pop,
                target_population=sst_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_vip, n_output=n_sst,
                    connectivity=0.7, mean=0.0, std=ii_std * 1.2, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=CORTICAL_FF_PRESET.configure(),
            )

        # =====================================================================
        # LAYER 1 NEUROGLIAFORM (NGC) CONNECTIONS
        # =====================================================================
        # L1 NGC cells occupy the only true L1 cell layer. They receive:
        #   (a) L2/3 apical collaterals via AMPA (recurrent; activates NGC during
        #       sustained L2/3 bursting to progressively suppress apical input)
        #   (b) Top-down afferents from higher areas via add_input_source()
        #       (registered externally by BrainBuilder targeting L1_NGC)
        #
        # The NGC output inhibits the apical tufts of L2/3 and L5 pyramidal
        # cells via GABA_A (physiologically a slow, diffuse GABA_A/B shunt).
        # No STP on NGC→Pyr: volume transmission bypasses vesicular depletion.
        #
        # References:
        #   Jiang et al. 2013 – NGC characterization (high-IR, late-spiking)
        #   Letzkus et al. 2011 – Top-down circuit via L1 interneurons
        #   Wester & McBain 2014 – L1 NGC→L2/3 apical inhibition

        # (a) L2/3 → L1 NGC: apical collateral (depressing; strong bursts recruit NGC)
        self._add_internal_connection(
            source_population=CortexPopulation.L23_PYR,
            target_population=CortexPopulation.L1_NGC,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l23_pyr_size,
                n_output=self.l1_ngc_size,
                connectivity=0.5,
                mean=0.0,
                std=0.5 / self.l23_pyr_size,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=CORTICAL_FF_PRESET.configure(),
        )

        # (b) L1 NGC → L2/3 apical tuft (GABA_A; wide, diffuse coverage)
        self._add_internal_connection(
            source_population=CortexPopulation.L1_NGC,
            target_population=CortexPopulation.L23_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l1_ngc_size,
                n_output=self.l23_pyr_size,
                connectivity=0.8,
                mean=0.0,
                std=0.0015,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=None,  # Volume transmission; no vesicular STP
        )

        # (c) L1 NGC → L5 apical tuft (GABA_A; L5 tufts reach L1)
        self._add_internal_connection(
            source_population=CortexPopulation.L1_NGC,
            target_population=CortexPopulation.L5_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l1_ngc_size,
                n_output=self.l5_pyr_size,
                connectivity=0.8,
                mean=0.0,
                std=0.0012,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=None,  # Volume transmission; no vesicular STP
        )

        # =====================================================================
        # CACHED SYNAPSE IDs (avoid per-timestep allocation in _step)
        # =====================================================================
        rn = self.region_name
        self._sid_l4_l23 = SynapseId(rn, CortexPopulation.L4_PYR,   rn, CortexPopulation.L23_PYR, receptor_type=ReceptorType.AMPA)
        self._sid_l23_l23 = SynapseId(rn, CortexPopulation.L23_PYR,  rn, CortexPopulation.L23_PYR, receptor_type=ReceptorType.AMPA)
        self._sid_l23_l5  = SynapseId(rn, CortexPopulation.L23_PYR,  rn, CortexPopulation.L5_PYR,  receptor_type=ReceptorType.AMPA)
        self._sid_l23_l6a = SynapseId(rn, CortexPopulation.L23_PYR,  rn, CortexPopulation.L6A_PYR, receptor_type=ReceptorType.AMPA)
        self._sid_l23_l6b = SynapseId(rn, CortexPopulation.L23_PYR,  rn, CortexPopulation.L6B_PYR, receptor_type=ReceptorType.AMPA)
        self._sid_l23_ngc = SynapseId(rn, CortexPopulation.L23_PYR,  rn, CortexPopulation.L1_NGC,  receptor_type=ReceptorType.AMPA)
        self._sid_ngc_l23 = SynapseId(rn, CortexPopulation.L1_NGC,   rn, CortexPopulation.L23_PYR, receptor_type=ReceptorType.GABA_A)
        self._sid_ngc_l5  = SynapseId(rn, CortexPopulation.L1_NGC,   rn, CortexPopulation.L5_PYR,  receptor_type=ReceptorType.GABA_A)
        self._sid_l6_l4   = SynapseId(rn, CortexPopulation.L6_PYR,   rn, CortexPopulation.L4_PYR,  receptor_type=ReceptorType.GABA_A)

        # =====================================================================
        # INHIBITORY POPULATION LOOKUP TABLE (eliminates identity chain in _compute_layer_inhibition)
        # =====================================================================
        # Maps inhibitory_network → (pyr_pop, pv_pop, sst_pop, vip_pop, ngc_pop)
        self._inhib_populations: Dict[CorticalInhibitoryNetwork, Tuple[PopulationName, PopulationName, PopulationName, PopulationName, PopulationName]] = {
            self.l23_inhibitory: (CortexPopulation.L23_PYR, CortexPopulation.L23_INHIBITORY_PV, CortexPopulation.L23_INHIBITORY_SST, CortexPopulation.L23_INHIBITORY_VIP, CortexPopulation.L23_INHIBITORY_NGC),
            self.l4_inhibitory:  (CortexPopulation.L4_PYR,  CortexPopulation.L4_INHIBITORY_PV,  CortexPopulation.L4_INHIBITORY_SST,  CortexPopulation.L4_INHIBITORY_VIP,  CortexPopulation.L4_INHIBITORY_NGC),
            self.l5_inhibitory:  (CortexPopulation.L5_PYR,  CortexPopulation.L5_INHIBITORY_PV,  CortexPopulation.L5_INHIBITORY_SST,  CortexPopulation.L5_INHIBITORY_VIP,  CortexPopulation.L5_INHIBITORY_NGC),
            self.l6a_inhibitory: (CortexPopulation.L6A_PYR, CortexPopulation.L6A_INHIBITORY_PV, CortexPopulation.L6A_INHIBITORY_SST, CortexPopulation.L6A_INHIBITORY_VIP, CortexPopulation.L6A_INHIBITORY_NGC),
            self.l6b_inhibitory: (CortexPopulation.L6B_PYR, CortexPopulation.L6B_INHIBITORY_PV, CortexPopulation.L6B_INHIBITORY_SST, CortexPopulation.L6B_INHIBITORY_VIP, CortexPopulation.L6B_INHIBITORY_NGC),
        }

    # =========================================================================
    # CORTICAL LAYER PROCESSING HELPERS
    # =========================================================================
    # These methods consolidate common operations across all cortical layers
    # to ensure biological consistency and reduce code duplication.

    def _run_cortical_inhibitory(
        self,
        inhib_net: CorticalInhibitoryNetwork,
        prev_pyr_spikes: torch.Tensor,
        pyr_pop: PopulationName,
        pv_pop: PopulationName,
        sst_pop: PopulationName,
        vip_pop: PopulationName,
        ngc_pop: PopulationName,
        feedforward_fsi_excitation: Optional[torch.Tensor] = None,
        long_range_excitation: Optional[torch.Tensor] = None,
        ach_spikes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute one inhibitory network step using registered STP-weighted connections.

        Mirrors `_run_hippocampal_inhibitory`.  All E→I / I→E / I→I conductances
        are computed from the weight matrices + STP modules registered via
        `_add_internal_connection` so they receive proper vesicle-pool depletion,
        diagnostics tracking, and can have learning strategies attached.

        The inhibitory network itself (``inhib_net``) is a pure neuron container
        that handles gap-junction coupling, long-range inputs, ACh receptors, and
        neuron integration — but NOT weight matrices.

        Args:
            inhib_net: Layer's CorticalInhibitoryNetwork (neuron container).
            prev_pyr_spikes: Previous-step pyramidal spikes [pyr_size] for E→I drive.
            pyr_pop / pv_pop / sst_pop / vip_pop / ngc_pop: Population names as
                registered in this NeuralRegion (matching _add_internal_connection keys).
            feedforward_fsi_excitation: Optional direct AMPA drive to PV [pv_size]
                (thalamic feedforward; L4 only).
            long_range_excitation: Optional top-down drive in pyr-space [pyr_size],
                routed by the inhibitory network to VIP (via w_lr_vip) and NGC (w_lr_ngc).
            ach_spikes: ACh spike tensor for nicotinic VIP + muscarinic NGC receptors.

        Returns:
            Dict with keys:
              total_inhibition, perisomatic_inhibition, dendritic_inhibition,
              pv_spikes, sst_spikes, vip_spikes, ngc_spikes,
              pv_membrane, sst_membrane, vip_membrane, ngc_membrane.
        """
        rn = self.region_name
        prev_pyr_f = prev_pyr_spikes.float()

        def _syn(src: "PopulationName", tgt: "PopulationName", rxn: ReceptorType) -> SynapseId:
            return SynapseId(rn, src, rn, tgt, receptor_type=rxn)

        def _matmul_stp(syn_id: SynapseId, pre_f: torch.Tensor) -> torch.Tensor:
            """STP-modulated weight × pre_float → conductance."""
            syn_info = self.get_synapse_info(syn_id)
            if syn_info.stp_module is not None:
                eff_w = syn_info.weights * syn_info.stp_module.forward(pre_f).T
            else:
                eff_w = syn_info.weights
            return torch.matmul(eff_w, pre_f)

        # ------------------------------------------------------------------
        # E → I  (Pyr → interneurons via registered STP weights)
        # ------------------------------------------------------------------
        pv_g_exc  = _matmul_stp(_syn(pyr_pop, pv_pop,  ReceptorType.AMPA), prev_pyr_f)
        sst_g_exc = _matmul_stp(_syn(pyr_pop, sst_pop, ReceptorType.AMPA), prev_pyr_f)
        vip_g_exc = _matmul_stp(_syn(pyr_pop, vip_pop, ReceptorType.AMPA), prev_pyr_f)
        ngc_g_exc = _matmul_stp(_syn(pyr_pop, ngc_pop, ReceptorType.AMPA), prev_pyr_f)

        # ------------------------------------------------------------------
        # I → I  (from previous-step inhibitory spikes, via registered weights)
        # ------------------------------------------------------------------
        prev_pv_f  = inhib_net._pv_spike_buffer.read(1).float()
        prev_sst_f = inhib_net._sst_spike_buffer.read(1).float()
        prev_vip_f = inhib_net._vip_spike_buffer.read(1).float()

        # PV receives inhibition from PV (lateral), SST (weak), VIP (strong disinhibitory),
        # SST receives from PV and VIP (disinhibitory).
        # (ngc is not a target of any registered I→I connections)
        pv_g_inh = (
            _matmul_stp(_syn(pv_pop,  pv_pop, ReceptorType.GABA_A), prev_pv_f)
            + _matmul_stp(_syn(sst_pop, pv_pop, ReceptorType.GABA_A), prev_sst_f)
            + _matmul_stp(_syn(vip_pop, pv_pop, ReceptorType.GABA_A), prev_vip_f)
        )
        sst_g_inh = (
            _matmul_stp(_syn(pv_pop,  sst_pop, ReceptorType.GABA_A), prev_pv_f)
            + _matmul_stp(_syn(vip_pop, sst_pop, ReceptorType.GABA_A), prev_vip_f)
        )
        vip_g_inh = torch.zeros(inhib_net.vip_size, device=self.device)
        ngc_g_inh = torch.zeros(inhib_net.ngc_size, device=self.device)

        # ------------------------------------------------------------------
        # Run interneurons (gap junction + long-range + ACh handled inside)
        # ------------------------------------------------------------------
        inhib_out = inhib_net.forward(
            pv_g_exc=pv_g_exc,
            pv_g_inh=pv_g_inh,
            sst_g_exc=sst_g_exc,
            sst_g_inh=sst_g_inh,
            vip_g_exc_from_pyr=vip_g_exc,
            vip_g_inh=vip_g_inh,
            ngc_g_exc_from_pyr=ngc_g_exc,
            ngc_g_inh=ngc_g_inh,
            feedforward_excitation=feedforward_fsi_excitation,
            long_range_excitation=long_range_excitation,
            ach_spikes=ach_spikes,
        )
        pv_spikes  = inhib_out["pv_spikes"]
        sst_spikes = inhib_out["sst_spikes"]
        vip_spikes = inhib_out["vip_spikes"]
        ngc_spikes = inhib_out["ngc_spikes"]

        # ------------------------------------------------------------------
        # I → E  (interneurons → pyramidal via registered STP weights)
        # The scale multipliers (PV ×1.0, SST ×0.3, VIP ×0.05, NGC ×0.12)
        # are absorbed into the weight std values in _init_synaptic_weights.
        # ------------------------------------------------------------------
        pv_f  = pv_spikes.float()
        sst_f = sst_spikes.float()
        vip_f = vip_spikes.float()
        ngc_f = ngc_spikes.float()

        perisomatic_inhibition = _matmul_stp(_syn(pv_pop,  pyr_pop, ReceptorType.GABA_A), pv_f)
        dendritic_inhibition   = _matmul_stp(_syn(sst_pop, pyr_pop, ReceptorType.GABA_A), sst_f)
        vip_to_pyr             = _matmul_stp(_syn(vip_pop, pyr_pop, ReceptorType.GABA_A), vip_f)
        ngc_to_pyr             = _matmul_stp(_syn(ngc_pop, pyr_pop, ReceptorType.GABA_A), ngc_f)

        total_inhibition = perisomatic_inhibition + dendritic_inhibition + vip_to_pyr + ngc_to_pyr

        return {
            "total_inhibition":       total_inhibition,
            "perisomatic_inhibition": perisomatic_inhibition,
            "dendritic_inhibition":   dendritic_inhibition,
            "pv_spikes":    pv_spikes,
            "sst_spikes":   sst_spikes,
            "vip_spikes":   vip_spikes,
            "ngc_spikes":   ngc_spikes,
            "pv_membrane":  inhib_out["pv_membrane"],
            "sst_membrane": inhib_out["sst_membrane"],
            "vip_membrane": inhib_out["vip_membrane"],
            "ngc_membrane": inhib_out["ngc_membrane"],
        }

    def _compute_layer_inhibition(
        self,
        prev_spikes: Optional[torch.Tensor],
        inhibitory_network: CorticalInhibitoryNetwork,
        ach_spikes: Optional[torch.Tensor],
        *,
        n_pyr: int,
        feedforward_fsi_excitation: Optional[torch.Tensor] = None,
        additional_inhibition: Optional[torch.Tensor] = None,
        long_range_excitation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Compute total inhibition for a cortical layer.

        Args:
            prev_spikes: Previous-step pyramidal spikes [pyr_size] for E→I drive.
                         Pass ``None`` on the first timestep.
            inhibitory_network: Layer's CorticalInhibitoryNetwork.
            ach_spikes: Raw NB ACh spikes for nicotinic VIP + muscarinic NGC.
            n_pyr: Pyramidal population size (used to build zero tensor on first step).
            feedforward_fsi_excitation: Direct AMPA to PV [pv_size] (L4 thalamic).
            additional_inhibition: Extra inhibition to add (e.g., predictive coding).
            long_range_excitation: Top-down input in pyr-space [pyr_size].

        Returns:
            (g_inh_total, inhibitory_output_dict or None)
        """
        if prev_spikes is not None:
            pyr_pop, pv_pop, sst_pop, vip_pop, ngc_pop = self._inhib_populations[inhibitory_network]
            inhib_output = self._run_cortical_inhibitory(
                inhib_net=inhibitory_network,
                prev_pyr_spikes=prev_spikes,
                pyr_pop=pyr_pop,
                pv_pop=pv_pop,
                sst_pop=sst_pop,
                vip_pop=vip_pop,
                ngc_pop=ngc_pop,
                feedforward_fsi_excitation=feedforward_fsi_excitation,
                long_range_excitation=long_range_excitation,
                ach_spikes=ach_spikes,
            )
            g_inh_total = inhib_output["total_inhibition"]
        else:
            # First timestep: no prior pyramidal activity
            g_inh_total = torch.zeros(n_pyr, device=self.device)
            inhib_output = None

        if additional_inhibition is not None:
            g_inh_total = g_inh_total + additional_inhibition

        return g_inh_total, inhib_output

    def _integrate_and_spike(
        self,
        g_exc: torch.Tensor,
        g_inh: torch.Tensor,
        neurons: ConductanceLIF,
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
        g_ampa, g_nmda = split_excitatory_conductance(g_exc, nmda_ratio=0.05)  # Reduced from 0.2: NMDA
        # Mg²⁺ positive feedback (unblocks at V>0.5, τ_NMDA=100ms) with g_L=0.05 caused
        # V_inf >> threshold for all cortical layers. 0.05 keeps NMDA contribution small.
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
        neurons: TwoCompartmentLIF,
        g_inh_apical: Optional[torch.Tensor] = None,
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
            g_inh_apical: Optional inhibitory (GABA_A) conductance for the apical
                          compartment (L1 NGC → apical tuft inhibition).
                          Pass ``None`` if no apical inhibition this timestep.

        Returns:
            (spikes, V_soma, V_dend)
        """
        from thalia.brain.neurons import TwoCompartmentLIF  # local import avoids circular
        assert isinstance(neurons, TwoCompartmentLIF), (
            f"_integrate_and_spike_two_compartment expects TwoCompartmentLIF, got {type(neurons).__name__}"
        )

        # Basal: feedforward / somatic inputs (5% NMDA — reduced from 20% to prevent
        # NMDA Mg²⁺ positive-feedback cascade; with g_L=0.05 even 0.007 noise→V_inf≫threshold)
        g_ampa_b, g_nmda_b = split_excitatory_conductance(g_exc_basal, nmda_ratio=0.05)

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
            g_gaba_a_apical=ConductanceTensor(g_inh_apical) if g_inh_apical is not None else None,
        )

        return spikes, V_soma, V_dend

    def _step_single_compartment_deep_layer(
        self,
        synapse_id_from_l23: SynapseId,
        l23_delayed: torch.Tensor,
        pyr_size: int,
        pyr_neurons: "ConductanceLIF",
        spike_buffer: CircularDelayBuffer,
        inhibitory_network: CorticalInhibitoryNetwork,
        ne_concentration: torch.Tensor,
        ach_spikes: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Shared forward step for single-compartment deep-layer pyramidal populations.

        Implements the common template for L6a and L6b (and any future deep layer):
        1. Integrate feedforward input from L2/3 (via provided SynapseId).
        2. Apply NE gain modulation.
        3. Compute causal inhibition.
        4. Integrate and spike.

        Returns:
            (spikes, g_exc, inhib_output)
        """
        g_exc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={synapse_id_from_l23: l23_delayed},
            n_neurons=pyr_size,
        ).g_ampa

        # Norepinephrine gain modulation
        ne_gain = compute_ne_gain(ne_concentration.mean().item())
        g_exc = g_exc * ne_gain

        g_inh, inhib_out = self._compute_layer_inhibition(
            prev_spikes=spike_buffer.read(1),
            inhibitory_network=inhibitory_network,
            ach_spikes=ach_spikes,
            n_pyr=pyr_size,
        )

        spikes, _ = self._integrate_and_spike(
            g_exc=g_exc,
            g_inh=g_inh,
            neurons=pyr_neurons,
        )
        return spikes, g_exc, inhib_out

    # =========================================================================
    # FORWARD PASS HELPERS (Layer-Level Submethods)
    # =========================================================================

    def _update_neuromodulators(self, neuromodulator_inputs: NeuromodulatorInput) -> Optional[torch.Tensor]:
        """Update DA/NE/ACh receptor dynamics and slice per-layer concentration buffers.

        Returns:
            Raw nucleus basalis ACh spike tensor (or ``None``), preserved for
            VIP nicotinic / NGC muscarinic receptors in each layer's inhibitory network.
        """
        nb_ach_spikes: Optional[torch.Tensor] = None
        if not GlobalConfig.NEUROMODULATION_DISABLED:
            self._process_neuromodulator(
                self.da_receptor,
                self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.DA_MESOCORTICAL),
                '_da_concentration',
                (self.config.da_l23_fraction, self.config.da_l4_fraction, self.config.da_l5_fraction,
                 self.config.da_l6a_fraction, self.config.da_l6b_fraction),
            )
            self._process_neuromodulator(
                self.ne_receptor,
                self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.NE),
                '_ne_concentration',
                (1.0, 1.0, 1.0, 1.0, 1.0),
            )
            nb_ach_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.ACH)
            self._process_neuromodulator(
                self.ach_receptor,
                nb_ach_spikes,
                '_ach_concentration',
                (1.0, 1.0, 1.0, 1.0, 1.0),
            )
        return nb_ach_spikes

    def _step_l4(
        self,
        synaptic_inputs: SynapticInput,
        nb_ach_spikes: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Process Layer 4 (feedforward input layer).

        Integrates bottom-up thalamic/sensory inputs, applies predictive coding
        inhibition from L5 (disynaptic via SST_PRED) and L6 (direct), runs NE
        gain, inhibitory network, and returns layer spikes.

        Args:
            synaptic_inputs: Full region synaptic input dict.
            nb_ach_spikes: Raw nucleus basalis ACh spikes for inhibitory network.

        Returns:
            (l4_spikes, l4_sst_pred_spikes, l4_inhib_output)
        """
        config = self.config

        # ---- Integrate bottom-up inputs ----
        l4_g_exc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l4_pyr_size,
            filter_by_target_population=CortexPopulation.L4_PYR,
        ).g_ampa

        # Thalamic afferents synapse directly onto PV cells for fast feedforward inhibition
        l4_g_exc_pv = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l4_inhibitory.pv_size,
            filter_by_target_population=CortexPopulation.L4_INHIBITORY_PV,
        ).g_ampa

        # ---- Predictive coding: L5/L6 → L4 inhibitory predictions ----
        # Deep layers predict what L4 should receive.
        # Good prediction → strong inhibition → L4 silent (no error)
        # Bad prediction → weak inhibition → L4 fires (error signal)
        l5_delayed = self._l5_spike_buffer.read(self._l5_l4_delay_steps)

        # Disynaptic pathway (Dale's Law compliant): L5_PYR → L4_SST_PRED → L4_PYR
        l5_sst_pred_conductance = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={self._l5_sst_pred_synapse: l5_delayed},
            n_neurons=self.l4_sst_pred_size,
        ).g_ampa
        sst_pred_g_ampa, sst_pred_g_nmda = split_excitatory_conductance(l5_sst_pred_conductance, nmda_ratio=0.1)
        l4_sst_pred_spikes, _ = self.l4_sst_pred_neurons.forward(
            g_ampa_input=ConductanceTensor(sst_pred_g_ampa),
            g_nmda_input=ConductanceTensor(sst_pred_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        l5_l4_inhibition = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={self._sst_pred_l4_synapse: l4_sst_pred_spikes},
            n_neurons=self.l4_pyr_size,
        ).g_gaba_a
        # Precision weighting: scale by L5 population activity (confidence)
        l5_activity = torch.mean(l5_delayed.float()).item()
        l5_precision = config.precision_min + (config.precision_max - config.precision_min) * l5_activity
        l5_l4_inhibition = l5_l4_inhibition * l5_precision

        # L6 (combined) → L4 prediction
        l6a_delayed = self._l6a_spike_buffer.read(self._l6_l4_delay_steps)
        l6b_delayed = self._l6b_spike_buffer.read(self._l6_l4_delay_steps)
        l6_delayed = torch.cat([l6a_delayed, l6b_delayed], dim=-1)
        l6_l4_inhibition = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={self._sid_l6_l4: l6_delayed},
            n_neurons=self.l4_pyr_size,
        ).g_gaba_a
        l6_activity = torch.mean(l6_delayed.float()).item()
        l6_precision = config.precision_min + (config.precision_max - config.precision_min) * l6_activity
        l6_l4_inhibition = l6_l4_inhibition * l6_precision

        l4_pred_inhibition = l5_l4_inhibition + l6_l4_inhibition

        # ---- NE gain (β-adrenergic increases L4 excitability) ----
        l4_g_exc = l4_g_exc * compute_ne_gain(self._ne_concentration_l4.mean().item())

        # ---- Inhibitory network (PV also receives direct thalamic drive) ----
        l4_g_inh, l4_inhib_output = self._compute_layer_inhibition(
            prev_spikes=self._l4_spike_buffer.read(1),
            inhibitory_network=self.l4_inhibitory,
            ach_spikes=nb_ach_spikes,
            n_pyr=self.l4_pyr_size,
            feedforward_fsi_excitation=l4_g_exc_pv,
            additional_inhibition=l4_pred_inhibition,
        )

        l4_spikes, _l4_membrane = self._integrate_and_spike(
            g_exc=l4_g_exc,
            g_inh=l4_g_inh,
            neurons=self.l4_neurons,
        )

        assert l4_spikes.shape == (self.l4_pyr_size,), (
            f"CorticalColumn: L4 spikes have shape {l4_spikes.shape} "
            f"but expected ({self.l4_pyr_size},). "
            f"Check input→L4 weights shape."
        )

        return l4_spikes, l4_sst_pred_spikes, l4_inhib_output

    def _step_l1_ngc(
        self,
        synaptic_inputs: SynapticInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process Layer 1 neurogliaform (NGC) cells.

        NGC cells receive:
          (a) L2/3 apical collaterals from the previous timestep
          (b) Top-down afferents from higher areas (targeting L1_NGC)
        Their spikes inhibit the apical tufts of L2/3 and L5 pyramidal
        cells in the same timestep (L1 axons are very short; no axonal delay).

        Reference: Letzkus et al. 2011 — top-down axons in L1 activate NGC
        cells within 1 ms, which then inhibit L2/3 pyramidal apical tufts.

        Args:
            synaptic_inputs: Full region synaptic input dict.

        Returns:
            (l1_ngc_spikes, l23_apical_inh, l5_apical_inh)
        """
        # (a) L2/3 → L1 NGC via apical collateral (previous timestep)
        l1_ngc_g_exc_from_l23 = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={self._sid_l23_ngc: self._l23_spike_buffer.read(1)},
            n_neurons=self.l1_ngc_size,
        ).g_ampa

        # (b) Top-down drive to L1 NGC from external regions
        l1_ngc_td = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l1_ngc_size,
            filter_by_target_population=CortexPopulation.L1_NGC,
        ).g_ampa

        l1_ngc_g_exc = F.relu(l1_ngc_g_exc_from_l23 + l1_ngc_td)

        # Single-compartment; no recurrent inhibitory drive
        l1_ngc_spikes, _l1_ngc_membrane = self._integrate_and_spike(
            g_exc=l1_ngc_g_exc,
            g_inh=torch.zeros(self.l1_ngc_size, device=self.device),
            neurons=self.l1_ngc_neurons,
        )

        # Apical GABA_A inhibition conductances for downstream pyramidal layers
        l23_apical_inh = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={self._sid_ngc_l23: l1_ngc_spikes},
            n_neurons=self.l23_pyr_size,
        ).g_gaba_a
        l5_apical_inh = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={self._sid_ngc_l5: l1_ngc_spikes},
            n_neurons=self.l5_pyr_size,
        ).g_gaba_a

        return l1_ngc_spikes, l23_apical_inh, l5_apical_inh

    def _step_l23(
        self,
        synaptic_inputs: SynapticInput,
        nb_ach_spikes: Optional[torch.Tensor],
        l23_apical_inh: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Process Layer 2/3 (cortico-cortical processing layer).

        Integrates feedforward input from L4 (via axonal delay), ACh-gated
        recurrent connections, and top-down feedback from higher areas.
        Uses two-compartment neurons with basal (FF + recurrent) and apical
        (top-down) routing.

        Args:
            synaptic_inputs: Full region synaptic input dict.
            nb_ach_spikes: Raw nucleus basalis ACh spikes for inhibitory network.
            l23_apical_inh: GABA_A conductance from L1 NGC onto L2/3 apical tufts.

        Returns:
            (l23_spikes, l23_inhib_output)
        """
        config = self.config

        # ---- L4→L2/3 feedforward (axonal delay) ----
        l23_ff = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={self._sid_l4_l23: self._l4_spike_buffer.read(self._l4_l23_delay_steps)},
            n_neurons=self.l23_pyr_size,
        ).g_ampa

        # ---- Stimulus gating (transient FFI, always enabled) ----
        # Biology: FFI is driven by thalamic inputs to L4, not all cortical inputs
        l4_source_activity = 0.0
        for synapse_id, source_spikes in synaptic_inputs.items():
            if synapse_id.target_population.startswith(CortexPopulation.L4):
                l4_source_activity += source_spikes.float().sum()

        if l4_source_activity > 0:
            # Normalize by l4_pyr_size to get average activity per neuron
            gating_input = torch.full((self.l4_pyr_size,), l4_source_activity / self.l4_pyr_size, device=self.device)
        else:
            gating_input = torch.zeros(self.l4_pyr_size, device=self.device)

        ffi = self.stimulus_gating.compute(gating_input, return_tensor=False)
        raw_ffi = ffi.item() if hasattr(ffi, "item") else float(ffi)
        effective_ffi_strength = min(1.0, raw_ffi / self.stimulus_gating.max_inhibition) * config.ffi_strength
        ffi_suppression = 1.0 - effective_ffi_strength

        # ---- ACh-gated recurrent suppression ----
        # High ACh (encoding): suppress recurrence to prevent interference
        # Low ACh (retrieval): enable recurrence for pattern completion
        ach_level = self._ach_concentration_l23.mean().item()
        ach_recurrent_modulation = compute_ach_recurrent_suppression(ach_level)

        l23_delayed = self._l23_spike_buffer.read(self._l23_l23_delay_steps)
        if l23_delayed is not None:
            l23_rec_raw = self._integrate_synaptic_inputs_at_dendrites(
                synaptic_inputs={self._sid_l23_l23: l23_delayed},
                n_neurons=self.l23_pyr_size,
            ).g_ampa
            l23_rec = l23_rec_raw * ffi_suppression * ach_recurrent_modulation
        else:
            l23_rec = torch.zeros_like(l23_ff)

        # ---- Top-down modulation to L2/3 (apical compartment) ----
        # L2/3 receives direct feedback from higher cortical areas without going
        # through L4, implementing predictive coding and attentional modulation.
        l23_td = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l23_pyr_size,
            filter_by_target_population=CortexPopulation.L23_PYR,
        ).g_ampa

        # ---- Two-compartment routing + NE gain ----
        # Basal: feedforward (L4→L2/3) + lateral recurrent (L2/3→L2/3)
        # Apical: top-down feedback from higher cortical areas
        ne_gain_l23 = compute_ne_gain(self._ne_concentration_l23.mean().item())
        l23_basal_input  = (l23_ff + l23_rec) * ne_gain_l23
        l23_apical_input = l23_td * ne_gain_l23

        # ---- Inhibitory network ----
        l23_g_inh, l23_inhib_output = self._compute_layer_inhibition(
            prev_spikes=self._l23_spike_buffer.read(1),
            inhibitory_network=self.l23_inhibitory,
            ach_spikes=nb_ach_spikes,
            n_pyr=self.l23_pyr_size,
            long_range_excitation=l23_td,  # top-down input gates VIP disinhibition
        )

        # Two-compartment integration; L1 NGC apical inhibition gates apical tuft
        l23_spikes, _l23_membrane, _l23_V_dend = self._integrate_and_spike_two_compartment(
            g_exc_basal=F.relu(l23_basal_input),
            g_inh_basal=l23_g_inh,
            g_exc_apical=F.relu(l23_apical_input),
            g_inh_apical=l23_apical_inh,
            neurons=self.l23_neurons,
        )

        assert l23_spikes.shape == (self.l23_pyr_size,), (
            f"CorticalColumn: L2/3 spikes have shape {l23_spikes.shape} "
            f"but expected ({self.l23_pyr_size},). "
            f"Check L4→L2/3 weights shape."
        )

        return l23_spikes, l23_inhib_output

    def _step_l5(
        self,
        synaptic_inputs: SynapticInput,
        nb_ach_spikes: Optional[torch.Tensor],
        l5_apical_inh: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Process Layer 5 (subcortical output layer).

        Integrates feedforward input from L2/3 (via axonal delay), top-down
        feedback, and inhibitory network. Uses two-compartment neurons with
        basal (L2/3 feedforward) and apical (top-down) routing.

        Args:
            synaptic_inputs: Full region synaptic input dict.
            nb_ach_spikes: Raw nucleus basalis ACh spikes for inhibitory network.
            l5_apical_inh: GABA_A conductance from L1 NGC onto L5 apical tufts.

        Returns:
            (l5_spikes, l5_inhib_output)
        """
        # ---- L2/3→L5 feedforward (axonal delay) ----
        l5_g_exc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={self._sid_l23_l5: self._l23_spike_buffer.read(self._l23_l5_delay_steps)},
            n_neurons=self.l5_pyr_size,
        ).g_ampa

        ne_gain_l5 = compute_ne_gain(self._ne_concentration_l5.mean().item())
        l5_g_exc = l5_g_exc * ne_gain_l5

        l5_g_inh, l5_inhib_output = self._compute_layer_inhibition(
            prev_spikes=self._l5_spike_buffer.read(1),
            inhibitory_network=self.l5_inhibitory,
            ach_spikes=nb_ach_spikes,
            n_pyr=self.l5_pyr_size,
        )

        # Top-down modulation to L5 apical tufts — mirrors the L2/3 apical mechanism.
        # Biology: L5 apical tufts in L1 receive top-down feedback from higher
        # cortical areas (PFC, association cortex, contralateral cortex).
        l5_td = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l5_pyr_size,
            filter_by_target_population=CortexPopulation.L5_PYR,
        ).g_ampa
        l5_apical_input = F.relu(l5_td * ne_gain_l5)

        # L2/3 → basal dendrites; top-down → apical tufts
        # L1 NGC apical inhibition gates the L5 apical tuft
        l5_spikes, _l5_membrane, _l5_V_dend = self._integrate_and_spike_two_compartment(
            g_exc_basal=l5_g_exc,
            g_inh_basal=l5_g_inh,
            g_exc_apical=l5_apical_input,
            g_inh_apical=l5_apical_inh,
            neurons=self.l5_neurons,
        )

        assert l5_spikes.shape == (self.l5_pyr_size,), (
            f"CorticalColumn: L5 spikes have shape {l5_spikes.shape} "
            f"but expected ({self.l5_pyr_size},). "
            f"Check L2/3→L5 weights shape."
        )

        return l5_spikes, l5_inhib_output

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @staticmethod
    def _inh_spikes(inhib_out: Optional[Dict[str, torch.Tensor]], key: str, n: int, device: torch.device) -> torch.Tensor:
        """Helper: extract spike tensor from inhibitory network output (None on first timestep)."""
        return inhib_out[key] if inhib_out is not None else torch.zeros(n, dtype=torch.bool, device=device)

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process one timestep through the full layered cortical circuit."""
        nb_ach_spikes = self._update_neuromodulators(neuromodulator_inputs)
        l4_spikes, l4_sst_pred_spikes, l4_inhib_output = self._step_l4(synaptic_inputs, nb_ach_spikes)
        l1_ngc_spikes, l23_apical_inh, l5_apical_inh = self._step_l1_ngc(synaptic_inputs)
        l23_spikes, l23_inhib_output = self._step_l23(synaptic_inputs, nb_ach_spikes, l23_apical_inh)
        l5_spikes, l5_inhib_output = self._step_l5(synaptic_inputs, nb_ach_spikes, l5_apical_inh)

        # =====================================================================
        # LAYER 6: CORTICOTHALAMIC FEEDBACK (L6a → TRN, L6b → relay)
        # =====================================================================
        # Both layers are single-compartment deep pyramidal cells and share
        # the same template: feedforward from L2/3 + NE gain + inhibition.
        l6a_spikes, _l6a_g_exc, l6a_inhib_output = self._step_single_compartment_deep_layer(
            synapse_id_from_l23=self._sid_l23_l6a,
            l23_delayed=self._l23_spike_buffer.read(self._l23_l6a_delay_steps),
            pyr_size=self.l6a_pyr_size,
            pyr_neurons=self.l6a_neurons,
            spike_buffer=self._l6a_spike_buffer,
            inhibitory_network=self.l6a_inhibitory,
            ne_concentration=self._ne_concentration_l6a,
            ach_spikes=nb_ach_spikes,
        )
        assert l6a_spikes.shape == (self.l6a_pyr_size,)

        l6b_spikes, _l6b_g_exc, l6b_inhib_output = self._step_single_compartment_deep_layer(
            synapse_id_from_l23=self._sid_l23_l6b,
            l23_delayed=self._l23_spike_buffer.read(self._l23_l6b_delay_steps),
            pyr_size=self.l6b_pyr_size,
            pyr_neurons=self.l6b_neurons,
            spike_buffer=self._l6b_spike_buffer,
            inhibitory_network=self.l6b_inhibitory,
            ne_concentration=self._ne_concentration_l6b,
            ach_spikes=nb_ach_spikes,
        )
        assert l6b_spikes.shape == (self.l6b_pyr_size,)

        # =====================================================================
        # POST-PROCESSING: HOMEOSTASIS AND PLASTICITY
        # =====================================================================

        region_outputs: RegionOutput = {
            # Layer 1 neurogliaform cells (apical tuft inhibition)
            CortexPopulation.L1_NGC:      l1_ngc_spikes,
            # Pyramidal populations
            CortexPopulation.L23_PYR:     l23_spikes,
            CortexPopulation.L4_PYR:      l4_spikes,
            CortexPopulation.L5_PYR:      l5_spikes,
            CortexPopulation.L6A_PYR:     l6a_spikes,
            CortexPopulation.L6B_PYR:     l6b_spikes,
            # L4 prediction-error SST interneuron
            CortexPopulation.L4_SST_PRED: l4_sst_pred_spikes,
            # L2/3 inhibitory populations
            CortexPopulation.L23_INHIBITORY_PV:  self._inh_spikes(l23_inhib_output, "pv_spikes",  self.l23_inhibitory.pv_size, self.device),
            CortexPopulation.L23_INHIBITORY_SST: self._inh_spikes(l23_inhib_output, "sst_spikes", self.l23_inhibitory.sst_size, self.device),
            CortexPopulation.L23_INHIBITORY_VIP: self._inh_spikes(l23_inhib_output, "vip_spikes", self.l23_inhibitory.vip_size, self.device),
            CortexPopulation.L23_INHIBITORY_NGC: self._inh_spikes(l23_inhib_output, "ngc_spikes", self.l23_inhibitory.ngc_size, self.device),
            # L4 inhibitory populations
            CortexPopulation.L4_INHIBITORY_PV:  self._inh_spikes(l4_inhib_output, "pv_spikes",  self.l4_inhibitory.pv_size, self.device),
            CortexPopulation.L4_INHIBITORY_SST: self._inh_spikes(l4_inhib_output, "sst_spikes", self.l4_inhibitory.sst_size, self.device),
            CortexPopulation.L4_INHIBITORY_VIP: self._inh_spikes(l4_inhib_output, "vip_spikes", self.l4_inhibitory.vip_size, self.device),
            CortexPopulation.L4_INHIBITORY_NGC: self._inh_spikes(l4_inhib_output, "ngc_spikes", self.l4_inhibitory.ngc_size, self.device),
            # L5 inhibitory populations
            CortexPopulation.L5_INHIBITORY_PV:  self._inh_spikes(l5_inhib_output, "pv_spikes",  self.l5_inhibitory.pv_size, self.device),
            CortexPopulation.L5_INHIBITORY_SST: self._inh_spikes(l5_inhib_output, "sst_spikes", self.l5_inhibitory.sst_size, self.device),
            CortexPopulation.L5_INHIBITORY_VIP: self._inh_spikes(l5_inhib_output, "vip_spikes", self.l5_inhibitory.vip_size, self.device),
            CortexPopulation.L5_INHIBITORY_NGC: self._inh_spikes(l5_inhib_output, "ngc_spikes", self.l5_inhibitory.ngc_size, self.device),
            # L6a inhibitory populations
            CortexPopulation.L6A_INHIBITORY_PV:  self._inh_spikes(l6a_inhib_output, "pv_spikes",  self.l6a_inhibitory.pv_size, self.device),
            CortexPopulation.L6A_INHIBITORY_SST: self._inh_spikes(l6a_inhib_output, "sst_spikes", self.l6a_inhibitory.sst_size, self.device),
            CortexPopulation.L6A_INHIBITORY_VIP: self._inh_spikes(l6a_inhib_output, "vip_spikes", self.l6a_inhibitory.vip_size, self.device),
            CortexPopulation.L6A_INHIBITORY_NGC: self._inh_spikes(l6a_inhib_output, "ngc_spikes", self.l6a_inhibitory.ngc_size, self.device),
            # L6b inhibitory populations
            CortexPopulation.L6B_INHIBITORY_PV:  self._inh_spikes(l6b_inhib_output, "pv_spikes",  self.l6b_inhibitory.pv_size, self.device),
            CortexPopulation.L6B_INHIBITORY_SST: self._inh_spikes(l6b_inhib_output, "sst_spikes", self.l6b_inhibitory.sst_size, self.device),
            CortexPopulation.L6B_INHIBITORY_VIP: self._inh_spikes(l6b_inhib_output, "vip_spikes", self.l6b_inhibitory.vip_size, self.device),
            CortexPopulation.L6B_INHIBITORY_NGC: self._inh_spikes(l6b_inhib_output, "ngc_spikes", self.l6b_inhibitory.ngc_size, self.device),
        }

        self._apply_all_population_homeostasis(region_outputs)
        self._apply_plasticity(synaptic_inputs, region_outputs)

        # =====================================================================
        # UPDATE STATE BUFFERS FOR NEXT TIMESTEP
        # =====================================================================
        self._l23_spike_buffer.write_and_advance(l23_spikes)
        self._l4_spike_buffer.write_and_advance(l4_spikes)
        self._l5_spike_buffer.write_and_advance(l5_spikes)
        self._l6a_spike_buffer.write_and_advance(l6a_spikes)
        self._l6b_spike_buffer.write_and_advance(l6b_spikes)
        self._l1_ngc_spike_buffer.write_and_advance(l1_ngc_spikes)

        return region_outputs

    def add_input_source(self, synapse_id: SynapseId, *args: Any, **kwargs: Any) -> None:
        """Extend base :meth:`add_input_source` to eagerly attach per-layer learning strategies.

        Pyramidal target populations receive a :class:`CompositeStrategy`
        (STDP + BCM) immediately at registration time, avoiding the per-timestep
        ``if strategy is None`` guard that previously lived in ``_apply_plasticity``.
        """
        super().add_input_source(synapse_id, *args, **kwargs)
        _pop_to_strategy = {
            CortexPopulation.L23_PYR: self.composite_l23,
            CortexPopulation.L4_PYR:  self.composite_l4,
            CortexPopulation.L5_PYR:  self.composite_l5,
        }
        strategy = _pop_to_strategy.get(synapse_id.target_population)
        if strategy is not None and self.get_learning_strategy(synapse_id) is None:
            self._add_learning_strategy(synapse_id, strategy, device=self.device)

    def _apply_plasticity(self, synaptic_inputs: SynapticInput, region_outputs: RegionOutput) -> None:
        """Apply continuous STDP+BCM learning with neuromodulator gating.

        Called automatically at each forward() timestep.  Neuromodulators gate
        *how much* weight change occurs from spike-timing events; they do not
        trigger learning by themselves.
        """
        if GlobalConfig.LEARNING_DISABLED:
            return

        config = self.config

        l23_spikes         = region_outputs[CortexPopulation.L23_PYR]
        l4_spikes          = region_outputs[CortexPopulation.L4_PYR]
        l5_spikes          = region_outputs[CortexPopulation.L5_PYR]
        l6a_spikes         = region_outputs[CortexPopulation.L6A_PYR]
        l6b_spikes         = region_outputs[CortexPopulation.L6B_PYR]
        l4_sst_pred_spikes = region_outputs[CortexPopulation.L4_SST_PRED]

        # Per-layer neuromodulator scalars (one dict per layer)
        nm_l23 = self._neuromod_scalars('l23')
        #nm_l4  = self._neuromod_scalars('l4')
        nm_l5  = self._neuromod_scalars('l5')
        nm_l6a = self._neuromod_scalars('l6a')
        nm_l6b = self._neuromod_scalars('l6b')

        # L4 → L2/3
        self._apply_learning(self._sid_l4_l23,   l4_spikes,  l23_spikes, **nm_l23)
        # L2/3 → L5
        self._apply_learning(self._sid_l23_l5,   l23_spikes, l5_spikes,  **nm_l5)
        # L2/3 → L6a
        self._apply_learning(self._sid_l23_l6a,  l23_spikes, l6a_spikes, **nm_l6a)
        # L2/3 → L6b
        self._apply_learning(self._sid_l23_l6b,  l23_spikes, l6b_spikes, **nm_l6b)

        # L2/3 recurrent (most plastic synapses in cortex)
        self._apply_learning(self._sid_l23_l23,  self._l23_spike_buffer.read(1), l23_spikes, **nm_l23)
        self.get_synaptic_weights(self._sid_l23_l23).data.fill_diagonal_(0.0)  # No autapses

        # Predictive coding: anti-Hebbian learning on inhibitory legs
        pred_lr = config.prediction_learning_rate * (1.0 + nm_l5["dopamine"])
        self._apply_learning(
            self._sst_pred_l4_synapse,
            pre_spikes=l4_sst_pred_spikes,
            post_spikes=l4_spikes,
            learning_rate_override=pred_lr,
        )
        self._apply_learning(
            self._l6_l4_synapse,
            pre_spikes=torch.cat([l6a_spikes, l6b_spikes], dim=-1),
            post_spikes=l4_spikes,
            learning_rate_override=pred_lr,
        )

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        """Map external synapse target population to layer-specific neuromodulator kwargs."""
        _pop_to_layer = {
            CortexPopulation.L23_PYR: 'l23',
            CortexPopulation.L4_PYR:  'l4',
            CortexPopulation.L5_PYR:  'l5',
        }
        layer = _pop_to_layer.get(synapse_id.target_population)
        return self._neuromod_scalars(layer) if layer is not None else {}

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Propagates dt update to neurons, STP components, and learning strategies.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)

        # Update learning strategies (CompositeStrategy propagates to sub-strategies)
        for composite in (self.composite_l23, self.composite_l4, self.composite_l5,
                          self.composite_l6a, self.composite_l6b):
            composite.update_temporal_parameters(dt_ms)
