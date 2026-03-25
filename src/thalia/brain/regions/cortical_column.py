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
    └───────────────┬───────────────────┘
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

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from thalia import GlobalConfig
from thalia.brain.configs import CorticalColumnConfig, CorticalPopulationConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    ConductanceLIF,
    TwoCompartmentLIF,
    build_conductance_lif_config,
    build_two_compartment_config,
    split_excitatory_conductance,
)
from thalia.brain.synapses import (
    NeuromodulatorReceptor,
    NMReceptorType,
    make_neuromodulator_receptor,
    STPConfig,
    WeightInitializer,
)
from thalia.brain.synapses.weight_init import ConductanceScaledSpec
from thalia.learning import (
    BCMConfig,
    BCMStrategy,
    InhibitorySTDPConfig,
    InhibitorySTDPStrategy,
    MetaplasticityConfig,
    MetaplasticityStrategy,
    STDPConfig,
    STDPStrategy,
    CompositeStrategy,
    PredictiveCodingConfig,
    PredictiveCodingStrategy,
)
from thalia.learning.strategies import LearningStrategy
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
)
from thalia.utils import cortical_inhibitory_fused as _ci_fused

from .cortical_inhibitory_network import CorticalInhibitoryNetwork
from .neural_region import NeuralRegion
from .population_names import CortexPopulation
from .region_registry import register_region
from .stimulus_gating import StimulusGating


@register_region(
    "cortical_column",
    description="Multi-layer cortical microcircuit with L2/3/L4/L5/L6a/L6b structure",
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
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        )

        # BCM config for homeostatic modulation
        # CRITICAL: BCM LR must be much slower than STDP to prevent runaway potentiation
        # Use 2% of STDP learning rate (50:1 ratio) to reduce BCM influence
        # Previous 10:1 ratio caused runaway depression overwhelming STDP
        bcm_cfg = BCMConfig(
            learning_rate=config.learning_rate * 0.02,  # 50x slower than STDP
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
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

        # Metaplasticity for L2/3 recurrent — the most plastic connections in
        # cortex, critical for attractor dynamics and working memory.  Wrapping
        # the shared composite_l23 is safe: MetaplasticityStrategy only stores
        # per-synapse rate/consolidation buffers; STDP/BCM traces inside the
        # composite use ensure_setup() and reinitialise naturally on shape change.
        meta_cfg = MetaplasticityConfig(
            tau_recovery_ms=5000.0,
            depression_strength=5.0,
            tau_consolidation_ms=300000.0,
            consolidation_sensitivity=0.1,
            rate_min=0.1,
        )
        self._meta_l23_recurrent = MetaplasticityStrategy(
            base_strategy=self.composite_l23,
            config=meta_cfg,
        )

        # Inhibitory STDP (Vogels et al. 2011) for I→E synapses.
        # PV→Pyr and SST→Pyr connections get per-layer iSTDP instances so
        # inhibition homeostatically tunes itself to balance excitation.
        istdp_cfg = InhibitorySTDPConfig(
            learning_rate=config.istdp_learning_rate,
            tau_istdp=config.istdp_tau_ms,
            alpha=config.istdp_alpha,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        )
        self.istdp_pv_l23:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_pv_l4:   InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_pv_l5:   InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_pv_l6a:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_pv_l6b:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_sst_l23: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_sst_l4:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_sst_l5:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_sst_l6a: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_sst_l6b: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)

        # iSTDP for NGC→Pyr and cross-layer SST→Pyr connections.
        # These I→E pathways need homeostatic plasticity so inhibition tracks
        # changing excitatory drive.
        self.istdp_ngc_l23:    InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_ngc_l5:     InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_l4sst_l23:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_l5sst_l23:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)

        # iSTDP for VIP→SST and VIP→PV disinhibitory connections.
        # VIP interneurons are the primary disinhibitory motif in cortex
        # (Pi et al. 2013; Lee et al. 2013).  VIP→SST gate strength is
        # ACh-modulated and must adapt to match evolving SST/PV inhibition
        # levels.  Homeostatic iSTDP keeps VIP disinhibition calibrated.
        self.istdp_vip_sst: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_vip_pv:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)

        # Hebbian STDP for E→I connections (Pyr→PV, Pyr→SST).
        # PV cells must track local pyramidal assemblies: when excitatory
        # representations change, E→PV weights should co-adapt so PV cells
        # are recruited by the new assembly (Kullmann & Lamsa 2007; Lu et al. 2007).
        # Conservative rate: E→I plasticity is slower than E→E to avoid
        # destabilising the inhibitory feedback loop.
        ei_stdp_cfg = STDPConfig(
            learning_rate=config.learning_rate * 0.15,  # 15% of E→E rate
            a_plus=config.a_plus * 0.5,
            a_minus=config.a_minus * 0.5,
            tau_plus=config.tau_plus_ms,
            tau_minus=config.tau_minus_ms,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        )
        self.ei_stdp_pv:  STDPStrategy = STDPStrategy(ei_stdp_cfg)
        self.ei_stdp_sst: STDPStrategy = STDPStrategy(ei_stdp_cfg)
        self.ei_stdp_ngc: STDPStrategy = STDPStrategy(ei_stdp_cfg)

        # Shared STDP strategy for external (inter-region) E→E inputs.
        # Thalamocortical (thal→L4) and corticocortical (ctx→L2/3, L5) inputs
        # undergo plasticity that shapes sensory representations and inter-area
        # routes.  Registered lazily per-synapse in apply_learning() via the
        # base-class dispatch + _get_learning_kwargs() hook.
        self._external_stdp_strategy = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate * 0.5,  # Conservative for afferent pathways
            a_plus=config.a_plus,
            a_minus=config.a_minus,
            tau_plus=config.tau_plus_ms,
            tau_minus=config.tau_minus_ms,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        ))

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
        self.da_receptor = make_neuromodulator_receptor(
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
        self.ne_receptor = make_neuromodulator_receptor(
            NMReceptorType.NE_ALPHA1, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device
        )
        self._ne_concentration_l23: torch.Tensor
        self._ne_concentration_l4: torch.Tensor
        self._ne_concentration_l5: torch.Tensor
        self._ne_concentration_l6a: torch.Tensor
        self._ne_concentration_l6b: torch.Tensor
        self._init_nm_buffers("_ne_concentration", device)

        # ACh: nucleus basalis nicotinic α4β2 (ionotropic; fast: τ_rise=3 ms, τ_decay=15 ms).
        self.ach_receptor = make_neuromodulator_receptor(
            NMReceptorType.ACH_NICOTINIC, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device,
            amplitude_scale=1.3,
        )
        self._ach_concentration_l23: torch.Tensor
        self._ach_concentration_l4: torch.Tensor
        self._ach_concentration_l5: torch.Tensor
        self._ach_concentration_l6a: torch.Tensor
        self._ach_concentration_l6b: torch.Tensor
        self._init_nm_buffers("_ach_concentration", device)

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # SYNAPTIC INPUT MANAGEMENT
    # =========================================================================

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
        """Extend base :meth:`add_input_source` to eagerly attach per-layer learning strategies.

        Pyramidal target populations receive a :class:`CompositeStrategy`
        (STDP + BCM) immediately at registration time, avoiding the per-timestep
        ``if strategy is None`` guard that previously lived in ``_apply_plasticity``.
        """
        super().add_input_source(
            synapse_id, n_input, connectivity, weight_scale,
            stp_config=stp_config, learning_strategy=learning_strategy, device=device,
        )
        _pop_to_strategy = {
            CortexPopulation.L23_PYR: self.composite_l23,
            CortexPopulation.L4_PYR:  self.composite_l4,
            CortexPopulation.L5_PYR:  self.composite_l5,
        }
        strategy = _pop_to_strategy.get(synapse_id.target_population)
        if strategy is not None and self.get_learning_strategy(synapse_id) is None:
            self._add_learning_strategy(synapse_id, strategy, device=self.device)

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
        (tau_mem_ms, v_threshold, adaptation) reflecting biological
        diversity of cortical cell types.
        """
        config = self.config

        # =====================================================================
        # LAYER-SPECIFIC HETEROGENEITY
        # =====================================================================
        l23_overrides: CorticalPopulationConfig = config.population_overrides[CortexPopulation.L23_PYR]
        l4_overrides: CorticalPopulationConfig = config.population_overrides[CortexPopulation.L4_PYR]
        l5_overrides: CorticalPopulationConfig = config.population_overrides[CortexPopulation.L5_PYR]
        l6a_overrides: CorticalPopulationConfig = config.population_overrides[CortexPopulation.L6A_PYR]
        l6b_overrides: CorticalPopulationConfig = config.population_overrides[CortexPopulation.L6B_PYR]

        self.l23_neurons: TwoCompartmentLIF
        self.l23_neurons = self._create_and_register_neuron_population(
            population_name=CortexPopulation.L23_PYR,
            n_neurons=self.l23_pyr_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=build_two_compartment_config(
                l23_overrides, self.l23_pyr_size, device,
                enable_nmda_plateau=config.l23_enable_nmda_plateau,
            ),
        )
        self.l4_neurons: ConductanceLIF
        self.l4_neurons = self._create_and_register_neuron_population(
            population_name=CortexPopulation.L4_PYR,
            n_neurons=self.l4_pyr_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=build_conductance_lif_config(l4_overrides, self.l4_pyr_size, device),
        )
        self.l5_neurons: TwoCompartmentLIF
        self.l5_neurons = self._create_and_register_neuron_population(
            population_name=CortexPopulation.L5_PYR,
            n_neurons=self.l5_pyr_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=build_two_compartment_config(l5_overrides, self.l5_pyr_size, device),
        )
        self.l6a_neurons: ConductanceLIF
        self.l6a_neurons = self._create_and_register_neuron_population(
            population_name=CortexPopulation.L6A_PYR,
            n_neurons=self.l6a_pyr_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=build_conductance_lif_config(l6a_overrides, self.l6a_pyr_size, device),
        )
        self.l6b_neurons: ConductanceLIF
        self.l6b_neurons = self._create_and_register_neuron_population(
            population_name=CortexPopulation.L6B_PYR,
            n_neurons=self.l6b_pyr_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=build_conductance_lif_config(l6b_overrides, self.l6b_pyr_size, device),
        )

        # =====================================================================
        # EXPLICIT INHIBITORY NETWORKS (PV, SST, VIP INTERNEURONS)
        # =====================================================================
        # Replace old FSI-only approach with explicit inhibitory networks containing
        # multiple cell types: PV (basket), SST (Martinotti), and VIP (disinhibitory)
        # Each layer gets its own inhibitory network with E→I, I→E, and I→I connectivity

        def _create_and_register_neurons(
            population_name: PopulationName,
            n_neurons: int,
            polarity: PopulationPolarity,
            config: ConductanceLIFConfig,
        ) -> ConductanceLIF:
            neurons = self._create_and_register_neuron_population(
                population_name,
                n_neurons,
                polarity,
                config,
            )
            assert isinstance(neurons, ConductanceLIF), "Expected ConductanceLIF neurons for excitatory populations"
            return neurons

        inhib_ovr = config.population_overrides  # region-specific cell-type overrides

        # L2/3 inhibitory network
        self.l23_inhibitory = CorticalInhibitoryNetwork(
            population_name=CortexPopulation.L23_INHIBITORY,
            pyr_size=self.l23_pyr_size,
            total_inhib_fraction=config.total_inhib_fraction,
            _create_and_register_neurons_fn=_create_and_register_neurons,
            dt_ms=config.dt_ms,
            device=device,
            population_overrides=inhib_ovr,
        )

        # L4 inhibitory network
        self.l4_inhibitory = CorticalInhibitoryNetwork(
            population_name=CortexPopulation.L4_INHIBITORY,
            pyr_size=self.l4_pyr_size,
            total_inhib_fraction=config.total_inhib_fraction,
            _create_and_register_neurons_fn=_create_and_register_neurons,
            dt_ms=config.dt_ms,
            device=device,
            population_overrides=inhib_ovr,
        )

        # L5 inhibitory network
        self.l5_inhibitory = CorticalInhibitoryNetwork(
            population_name=CortexPopulation.L5_INHIBITORY,
            pyr_size=self.l5_pyr_size,
            total_inhib_fraction=config.total_inhib_fraction,
            _create_and_register_neurons_fn=_create_and_register_neurons,
            dt_ms=config.dt_ms,
            device=device,
            population_overrides=inhib_ovr,
        )

        # L6a inhibitory network
        self.l6a_inhibitory = CorticalInhibitoryNetwork(
            population_name=CortexPopulation.L6A_INHIBITORY,
            pyr_size=self.l6a_pyr_size,
            total_inhib_fraction=config.total_inhib_fraction,
            _create_and_register_neurons_fn=_create_and_register_neurons,
            dt_ms=config.dt_ms,
            device=device,
            population_overrides=inhib_ovr,
        )

        # L6b inhibitory network
        self.l6b_inhibitory = CorticalInhibitoryNetwork(
            population_name=CortexPopulation.L6B_INHIBITORY,
            pyr_size=self.l6b_pyr_size,
            total_inhib_fraction=config.total_inhib_fraction,
            _create_and_register_neurons_fn=_create_and_register_neurons,
            dt_ms=config.dt_ms,
            device=device,
            population_overrides=inhib_ovr,
        )

        # =====================================================================
        # L4 SST PREDICTION-ERROR INTERNEURON (P1-03: disynaptic L5→L4 relay)
        # =====================================================================
        # Biology: L5 pyramidal cells drive SST+ Martinotti cells in L4 via AMPA.
        # These SST cells then inhibit L4 pyramidal dendrites (GABA_A), completing
        # the prediction pathway without violating Dale’s Law.
        # ~10% of L4 pyramidal count; fast Martinotti-like dynamics (tau_mem_ms 15ms).
        self.l4_sst_pred_size = max(5, int(self.l4_pyr_size * 0.10))
        self.l4_sst_pred_neurons: ConductanceLIF
        self.l4_sst_pred_neurons = self._create_and_register_neuron_population(
            population_name=CortexPopulation.L4_SST_PRED,
            n_neurons=self.l4_sst_pred_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=build_conductance_lif_config(
                CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.70, v_reset=0.0, adapt_increment=0.05, tau_adapt_ms=90.0, noise_std=0.08),
                self.l4_sst_pred_size, device,
            ),
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
        self.l1_ngc_size: int = max(3, int(self.l23_pyr_size * 0.05))
        self.l1_ngc_neurons: ConductanceLIF
        self.l1_ngc_neurons = self._create_and_register_neuron_population(
            population_name=CortexPopulation.L1_NGC,
            n_neurons=self.l1_ngc_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=build_conductance_lif_config(
                CorticalPopulationConfig(tau_mem_ms=30.0, v_threshold=0.65, v_reset=0.0, adapt_increment=0.02, tau_adapt_ms=300.0, noise_std=0.08),
                self.l1_ngc_size, device,
            ),
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
        l4_std = 0.7 / expected_active_l4    # L4 driving L2/3 — raised 0.5→1.2→0.7: 1.2 caused synchronized
                                                 # L4 volleys to propagate into L23 (cortex ρ=0.74-0.90).
                                                 # 0.7 is still 40% above original 0.5 to support laminar cascade.

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
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
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
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
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
            stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
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
            stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
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
                std=0.004,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
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
            stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=20.0),
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
            stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=20.0),
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
                connectivity=self.config.l23_recurrent_connectivity,
                weight_scale=self.config.l23_recurrent_weight_scale,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.25, tau_d=200.0, tau_f=300.0),  # Raised U=0.12→0.25: initial efficacy was too low for recurrent drive; tau_f 600→300ms for faster facilitation; tau_d 150→200ms for moderate depression
            learning_strategy=self._meta_l23_recurrent,
        )

        # =====================================================================
        # INHIBITORY INTERNEURON CONNECTIONS (E→I, I→E, I→I) per layer
        # =====================================================================
        # All PV/SST/VIP/NGC weight matrices registered via _add_internal_connection
        # so they receive STP modulation, diagnostics tracking, and can accept
        # learning strategies.

        layer_defs = [
            # ( pyr_pop, pv_pop, sst_pop, vip_pop, ngc_pop, inhib_net, pyr_size, istdp_pv, istdp_sst )
            (CortexPopulation.L23_PYR,  CortexPopulation.L23_INHIBITORY_PV,  CortexPopulation.L23_INHIBITORY_SST,  CortexPopulation.L23_INHIBITORY_VIP,  CortexPopulation.L23_INHIBITORY_NGC,  self.l23_inhibitory,  self.l23_pyr_size,  self.istdp_pv_l23,  self.istdp_sst_l23),
            (CortexPopulation.L4_PYR,   CortexPopulation.L4_INHIBITORY_PV,   CortexPopulation.L4_INHIBITORY_SST,   CortexPopulation.L4_INHIBITORY_VIP,   CortexPopulation.L4_INHIBITORY_NGC,   self.l4_inhibitory,   self.l4_pyr_size,   self.istdp_pv_l4,   self.istdp_sst_l4),
            (CortexPopulation.L5_PYR,   CortexPopulation.L5_INHIBITORY_PV,   CortexPopulation.L5_INHIBITORY_SST,   CortexPopulation.L5_INHIBITORY_VIP,   CortexPopulation.L5_INHIBITORY_NGC,   self.l5_inhibitory,   self.l5_pyr_size,   self.istdp_pv_l5,   self.istdp_sst_l5),
            (CortexPopulation.L6A_PYR,  CortexPopulation.L6A_INHIBITORY_PV,  CortexPopulation.L6A_INHIBITORY_SST,  CortexPopulation.L6A_INHIBITORY_VIP,  CortexPopulation.L6A_INHIBITORY_NGC,  self.l6a_inhibitory,  self.l6a_pyr_size,  self.istdp_pv_l6a,  self.istdp_sst_l6a),
            (CortexPopulation.L6B_PYR,  CortexPopulation.L6B_INHIBITORY_PV,  CortexPopulation.L6B_INHIBITORY_SST,  CortexPopulation.L6B_INHIBITORY_VIP,  CortexPopulation.L6B_INHIBITORY_NGC,  self.l6b_inhibitory,  self.l6b_pyr_size,  self.istdp_pv_l6b,  self.istdp_sst_l6b),
        ]

        for pyr_pop, pv_pop, sst_pop, vip_pop, ngc_pop, inh, n_pyr, istdp_pv, istdp_sst in layer_defs:
            n_pv  = inh.pv_size
            n_sst = inh.sst_size
            n_vip = inh.vip_size
            n_ngc = inh.ngc_size

            # -- Weight scales -----------------------------------------------
            # E→I: each interneuron receives from a fraction of pyramidal cells.
            # Target ~30-50 Hz PV drive at expected 5-10 Hz pyramidal rate.
            # Formula: std = g_AMPA_needed / (n_pyr * conn * pyr_rate * tau_E * sqrt(2/pi))
            # Cap at 0.25: 4σ = 1.0 = w_max — without the cap, small-layer populations
            # (PFC L4 n=80 → std=0.50; L6A/B n=60 → std=0.67) produce weights above w_max
            # that are silently truncated at every learning step from step 0 onwards.
            # For n_pyr < 160 the formula is dominated by the cap; total drive is lower for
            # genuinely small populations, which is biologically correct.
            ei_pv_std  = min(55.0 / n_pyr, 0.25)  # PV receives strongest unitary EPSPs from pyramidals (Holmgren et al. 2003)
            # L23 SST needs more Pyr drive (3.77 Hz at k=14; target 5-25 Hz).
            # k=20 gives ~5.4 Hz for sensory L23 (n_pyr=1000), but causes epileptiform
            # in small cortices (PFC n_pyr=360 → weight=0.056 uncapped, cap gives 0.030).
            # L4-L6B: cap at 0.070 prevents synchrony in small cortices without
            # meaningfully reducing sensory weights (sensory L4 n_pyr=800 → 0.0175 < 0.070).
            if pyr_pop == CortexPopulation.L23_PYR:
                ei_sst_std = min(20.0 / n_pyr, 0.030)
            else:
                ei_sst_std = min(14.0 / n_pyr, 0.070)  # Cap: assoc L4 (n=150, 0.093→0.070), PFC L4/L6A (n=80/60, 0.175/0.233→0.070)
            ei_vip_std = 1.0 / n_pyr   # Reduced 1.5→1.0: VIP populations (3-50 neurons) were
                                        # epileptiform in 9/25 layers due to shared pyramidal input.
                                        # Lower weight + reduced connectivity (0.4→0.25) below
                                        # decreases correlated activation within small VIP populations.
            ei_ngc_std = 18.0 / n_pyr  # Reduced 25→18: NGCs 30-43 Hz (target 5-30 Hz)

            # I→E: perisomatic PV gives strong inhibition ~2 mS/cm²/spike.
            ie_pv_std  = 0.002
            ie_sst_std = 0.0006
            ie_vip_std = 0.0001
            ie_ngc_std = 0.00024

            # I→I: moderate inhibition between interneuron types
            ii_std = 0.001

            # == E → I ========================================================
            # Pyr → PV (mildly depressing: Angulo et al 1999, Xiang et al 2002, Pyr→FS)
            # Hebbian STDP ensures PV cells track evolving pyramidal assemblies
            # (Kullmann & Lamsa 2007; Lu et al. 2007).
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pyr,
                    n_output=n_pv,
                    connectivity=0.5,
                    mean=0.0,
                    std=ei_pv_std,
                    device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig(U=0.25, tau_d=150.0, tau_f=20.0),
                learning_strategy=self.ei_stdp_pv,
            )
            # Pyr → SST (moderately facilitating — Markram 1998 EPSP-F, Reyes et al. 1998)
            # U raised 0.1→0.25: at Pyr=5 Hz, u_eff_initial=0.1 meant SST received only 10%
            # of nominal weight → chronic under-drive → SST at 1.88 Hz (target 5-25 Hz).
            # U=0.25 gives 25% initial efficacy while preserving facilitation at higher rates.
            # Hebbian STDP ensures SST cells learn to follow pyramidal populations.
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=sst_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pyr,
                    n_output=n_sst,
                    connectivity=0.3,
                    mean=0.0,
                    std=ei_sst_std,
                    device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig(U=0.25, tau_d=200.0, tau_f=150.0),
                learning_strategy=self.ei_stdp_sst,
            )
            # Pyr → VIP (depressing)
            # Hebbian STDP (E→I): VIP cells must learn which pyramidal
            # assemblies gate their disinhibitory output (Lee et al. 2013).
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=vip_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pyr,
                    n_output=n_vip,
                    connectivity=0.25,
                    mean=0.0,
                    std=ei_vip_std,
                    device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
                learning_strategy=self.ei_stdp_pv,  # Shared E→I STDP (same conservative rate)
            )
            # Pyr → NGC (facilitating — NGC integrate over long windows like SST)
            # Hebbian STDP (E→I): NGC must track pyramidal assemblies to
            # provide appropriate apical-tuft inhibition (like Pyr→PV/SST).
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=ngc_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pyr,
                    n_output=n_ngc,
                    connectivity=0.2,
                    mean=0.0,
                    std=ei_ngc_std,
                    device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig(U=0.10, tau_d=300.0, tau_f=300.0),
                learning_strategy=self.ei_stdp_ngc,
            )

            # == I → E ========================================================
            # PV → Pyr GABA_A (perisomatic, fast, moderately depressing — Kraushaar & Jonas 2000)
            # PV basket cells have reliable multi-vesicular release; U·τ_d must stay
            # low enough to maintain >10% efficacy at 40-80 Hz PV firing rates.
            # iSTDP (Vogels et al. 2011) homeostatically tunes inhibition strength.
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pv,
                    n_output=n_pyr,
                    connectivity=0.6,
                    mean=0.0,
                    std=ie_pv_std,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.25, tau_d=250.0, tau_f=20.0),
                learning_strategy=istdp_pv,
            )
            # SST → Pyr GABA_A (dendritic, moderately depressing — Silberberg & Markram 2007)
            # SST fires slower (~10-20 Hz) so tau_d can be higher than PV,
            # but must still maintain efficacy at sustained rates.
            # iSTDP (Vogels et al. 2011) homeostatically tunes inhibition strength.
            self._add_internal_connection(
                source_population=sst_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_sst,
                    n_output=n_pyr,
                    connectivity=0.4,
                    mean=0.0,
                    std=ie_sst_std,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.25, tau_d=400.0, tau_f=20.0),
                learning_strategy=istdp_sst,
            )
            # VIP → Pyr GABA_A (weak direct effect; VIP role is disinhibitory via VIP→SST)
            # Depressing; matches SST→Pyr pattern (Pfeffer et al. 2013).
            self._add_internal_connection(
                source_population=vip_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_vip,
                    n_output=n_pyr,
                    connectivity=0.15,
                    mean=0.0,
                    std=ie_vip_std,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.25, tau_d=400.0, tau_f=20.0),
            )
            # NGC → Pyr GABA_A (diffuse apical tuft; volume transmission, no vesicle STP)
            self._add_internal_connection(
                source_population=ngc_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_ngc,
                    n_output=n_pyr,
                    connectivity=0.5,
                    mean=0.0,
                    std=ie_ngc_std,
                    device=device,
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
                    n_input=n_pv,
                    n_output=n_pv,
                    connectivity=0.3,
                    mean=0.0,
                    std=ii_std,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.30, tau_d=300.0, tau_f=25.0),
            )
            # PV → SST (suppresses SST → disinhibits Pyr apical tuft)
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=sst_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_pv,
                    n_output=n_sst,
                    connectivity=0.3,
                    mean=0.0,
                    std=ii_std,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.30, tau_d=300.0, tau_f=25.0),
            )
            # SST → PV (weak; SST can briefly suppress PV allowing burst propagation)
            self._add_internal_connection(
                source_population=sst_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_sst,
                    n_output=n_pv,
                    connectivity=0.2,
                    mean=0.0,
                    std=ii_std * 0.7,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.12, tau_d=150.0, tau_f=600.0),
            )
            # VIP → PV (disinhibitory: VIP silence PV → releases pyramidal from peri-somatic inh)
            # iSTDP tunes VIP→PV strength to maintain target PV firing rate;
            # this gate adapts as PV drive changes during learning.
            self._add_internal_connection(
                source_population=vip_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_vip,
                    n_output=n_pv,
                    connectivity=0.6,
                    mean=0.0,
                    std=ii_std,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.35, tau_d=400.0, tau_f=25.0),
                learning_strategy=self.istdp_vip_pv,
            )
            # VIP → SST GABA_A (disinhibition: VIP→SST fires → SST silenced → apical disinhibition)
            # Trade-off: stronger VIP→SST reduces VIP-SST correlation but also suppresses SST.
            #   ii_std*16, conn=0.7 (total=11.2): SST=4.8 Hz, r=+0.83
            #   ii_std*24, conn=0.9 (total=21.6): SST=1.0 Hz, r=+0.80
            # Compromise: ii_std*18, conn=0.8 (total=14.4). True anticorrelation requires
            # top-down/ACh-driven VIP activation, which emerges during training.
            self._add_internal_connection(
                source_population=vip_pop,
                target_population=sst_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_vip,
                    n_output=n_sst,
                    connectivity=0.8,
                    mean=0.0,
                    std=ii_std * 18.0,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.45, tau_d=200.0, tau_f=25.0),
                learning_strategy=self.istdp_vip_sst,
            )
            # VIP → SST GABA_B (slow sustained inhibition for disinhibitory motif)
            # GABA_B tau ~150-300ms provides tonic suppression beyond fast GABA_A.
            self._add_internal_connection(
                source_population=vip_pop,
                target_population=sst_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_vip,
                    n_output=n_sst,
                    connectivity=0.35,
                    mean=0.0,
                    std=ii_std * 7.0,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_B,
                stp_config=STPConfig(U=0.35, tau_d=400.0, tau_f=25.0),
            )
            # SST → VIP GABA_A (reciprocal inhibition — Pfeffer et al. 2013)
            # Creates mutual inhibition between SST and VIP: when local Pyr drive
            # activates SST, SST suppresses VIP. When top-down or ACh signals activate
            # VIP independently, VIP suppresses SST. Winner-take-all → negative correlation.
            # Raised ii_std*5→ii_std*12→ii_std*16: sensory cortex VIP-SST correlation still
            # r=0.86 (association dropped to r=0.30). Sensory cortex has larger populations
            # with more shared noise; stronger SST→VIP inhibition needed for decorrelation.
            self._add_internal_connection(
                source_population=sst_pop,
                target_population=vip_pop,
                weights=WeightInitializer.sparse_gaussian(
                    n_input=n_sst,
                    n_output=n_vip,
                    connectivity=0.6,  # Raised 0.4→0.5→0.6: denser connectivity for reliable SST→VIP inhibition
                    mean=0.0,
                    std=ii_std * 10.0,  # Raised 5.0→12.0→16.0→10.0: SST→SST lateral inhibition now decorrelates SST, so less SST→VIP needed
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.20, tau_d=150.0, tau_f=300.0),  # Facilitating: SST builds up
                # VIP suppression via facilitation during sustained SST activity.
            )
            # SST → SST GABA_A (lateral inhibition among Martinotti cells)
            # Breaks synchrony: when SST cells fire in lockstep from shared PYR
            # input, mutual inhibition decorrelates them — an SST cell that fires
            # suppresses its neighbours, preventing population-wide co-activation.
            # Without this, increased thalamocortical drive triggers SST
            # epileptiform bursting (observed rounds 3, 5b).
            # Pfeffer et al. 2013, Jiang et al. 2015.
            self._add_internal_connection(
                source_population=sst_pop,
                target_population=sst_pop,
                weights=WeightInitializer.sparse_gaussian_no_autapses(
                    n_input=n_sst,
                    n_output=n_sst,
                    connectivity=0.15,
                    mean=0.0,
                    std=ii_std * 2.0,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.20, tau_d=150.0, tau_f=300.0),
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
        # Hebbian STDP (E→I): L1 NGC must track L2/3 assembly changes so that
        # top-down gating inhibition co-adapts with local representations.
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
            stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
            learning_strategy=self.ei_stdp_ngc,
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
            learning_strategy=self.istdp_ngc_l23,
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
            learning_strategy=self.istdp_ngc_l5,
        )

        # =====================================================================
        # CROSS-LAYER INHIBITORY CONNECTIONS
        # =====================================================================
        # Biology: inter-laminar inhibitory projections are critical for
        # coordinating activity across layers. Three key motifs:
        #
        # (1) L4 SST → L2/3 Pyr (ascending surround suppression)
        #     L4 Martinotti cells extend axons into L2/3 where they target
        #     pyramidal basal dendrites. When L4 is strongly activated, its
        #     SST cells suppress L2/3 pyramidal neurons that lack matching
        #     feedforward input — sharpening stimulus selectivity.
        #     (Xu et al. 2013; Muñoz et al. 2017)
        #
        # (2) L5 SST → L2/3 Pyr apical (predictive error gating)
        #     L5 SST cells project ascending dendrite-targeting inhibition
        #     onto L2/3 pyramidal apical tufts. When L5 is confidently driven
        #     by basal + apical coincidence, its SST cells gate L2/3 apical
        #     feedback — implementing a predictive error suppression signal.
        #     (Naka & Bhatt et al. 2019; Aru et al. 2020)
        #
        # (3) L2/3 PV ↔ L5 PV (gamma coherence across layers)
        #     PV basket cells in L2/3 and L5 make reciprocal GABAergic
        #     connections that synchronize gamma oscillations across the
        #     cortical column. This translaminar PV coupling is critical
        #     for coherent information routing.
        #     (Roopun et al. 2010; Cardin et al. 2009)

        # (d) L4 SST → L2/3 Pyr: ascending surround suppression
        self._add_internal_connection(
            source_population=CortexPopulation.L4_INHIBITORY_SST,
            target_population=CortexPopulation.L23_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l4_inhibitory.sst_size,
                n_output=self.l23_pyr_size,
                connectivity=0.3,
                mean=0.0,
                std=0.0004,  # Weak: cross-layer SST is modulatory, ~2/3 of within-layer SST std (0.0006)
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=400.0, tau_f=20.0),  # Matches within-layer SST→Pyr
            learning_strategy=self.istdp_l4sst_l23,
        )

        # (e) L5 SST → L2/3 Pyr apical: predictive error gating
        self._add_internal_connection(
            source_population=CortexPopulation.L5_INHIBITORY_SST,
            target_population=CortexPopulation.L23_PYR,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l5_inhibitory.sst_size,
                n_output=self.l23_pyr_size,
                connectivity=0.25,
                mean=0.0,
                std=0.0003,  # Weaker than L4 SST: L5→L2/3 is a modulatory gate, not primary inhibition
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=400.0, tau_f=20.0),
            learning_strategy=self.istdp_l5sst_l23,
        )

        # (f) L2/3 PV → L5 PV: descending gamma synchronization
        self._add_internal_connection(
            source_population=CortexPopulation.L23_INHIBITORY_PV,
            target_population=CortexPopulation.L5_INHIBITORY_PV,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l23_inhibitory.pv_size,
                n_output=self.l5_inhibitory.pv_size,
                connectivity=0.2,
                mean=0.0,
                std=0.0008,  # Moderate: PV→PV cross-layer slightly weaker than within-layer PV→PV (0.001)
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.30, tau_d=300.0, tau_f=25.0),  # Matches within-layer PV→PV
        )

        # (g) L5 PV → L2/3 PV: ascending gamma synchronization
        self._add_internal_connection(
            source_population=CortexPopulation.L5_INHIBITORY_PV,
            target_population=CortexPopulation.L23_INHIBITORY_PV,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.l5_inhibitory.pv_size,
                n_output=self.l23_inhibitory.pv_size,
                connectivity=0.2,
                mean=0.0,
                std=0.0008,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.30, tau_d=300.0, tau_f=25.0),
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
        # Cross-layer inhibitory SynapseIds
        self._sid_l4sst_l23  = SynapseId(rn, CortexPopulation.L4_INHIBITORY_SST,  rn, CortexPopulation.L23_PYR,           receptor_type=ReceptorType.GABA_A)
        self._sid_l5sst_l23  = SynapseId(rn, CortexPopulation.L5_INHIBITORY_SST,  rn, CortexPopulation.L23_PYR,           receptor_type=ReceptorType.GABA_A)
        self._sid_l23pv_l5pv = SynapseId(rn, CortexPopulation.L23_INHIBITORY_PV,  rn, CortexPopulation.L5_INHIBITORY_PV,  receptor_type=ReceptorType.GABA_A)
        self._sid_l5pv_l23pv = SynapseId(rn, CortexPopulation.L5_INHIBITORY_PV,   rn, CortexPopulation.L23_INHIBITORY_PV, receptor_type=ReceptorType.GABA_A)

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

        # Pre-cache weight matrices and SynapseId lists for fused C++ kernel.
        # Avoids dict lookups and SynapseId construction in the hot loop.
        self._fused_inhib_available = _ci_fused.is_available()
        self._fused_inhib_cache: Dict[CorticalInhibitoryNetwork, Tuple[List[SynapseId], List[torch.Tensor], List[SynapseId]]] = {}
        if self._fused_inhib_available:
            self._init_fused_inhibitory_cache()

    def _init_fused_inhibitory_cache(self) -> None:
        """Pre-cache weight matrices and SynapseId lists for each inhibitory network."""
        rn = self.region_name
        for inhib_net, (pyr_pop, pv_pop, sst_pop, vip_pop, ngc_pop) in self._inhib_populations.items():
            # Connection order must match cortical_inhibitory_kernel.cpp
            synapse_ids = [
                # E→I (0-3)
                SynapseId(rn, pyr_pop, rn, pv_pop,  receptor_type=ReceptorType.AMPA),
                SynapseId(rn, pyr_pop, rn, sst_pop, receptor_type=ReceptorType.AMPA),
                SynapseId(rn, pyr_pop, rn, vip_pop, receptor_type=ReceptorType.AMPA),
                SynapseId(rn, pyr_pop, rn, ngc_pop, receptor_type=ReceptorType.AMPA),
                # I→I (4-11)
                SynapseId(rn, pv_pop,  rn, pv_pop,  receptor_type=ReceptorType.GABA_A),
                SynapseId(rn, sst_pop, rn, pv_pop,  receptor_type=ReceptorType.GABA_A),
                SynapseId(rn, vip_pop, rn, pv_pop,  receptor_type=ReceptorType.GABA_A),
                SynapseId(rn, pv_pop,  rn, sst_pop, receptor_type=ReceptorType.GABA_A),
                SynapseId(rn, vip_pop, rn, sst_pop, receptor_type=ReceptorType.GABA_A),
                SynapseId(rn, vip_pop, rn, sst_pop, receptor_type=ReceptorType.GABA_B),
                SynapseId(rn, sst_pop, rn, vip_pop, receptor_type=ReceptorType.GABA_A),
                SynapseId(rn, sst_pop, rn, sst_pop, receptor_type=ReceptorType.GABA_A),
                # I→E (12-15)
                SynapseId(rn, pv_pop,  rn, pyr_pop, receptor_type=ReceptorType.GABA_A),
                SynapseId(rn, sst_pop, rn, pyr_pop, receptor_type=ReceptorType.GABA_A),
                SynapseId(rn, vip_pop, rn, pyr_pop, receptor_type=ReceptorType.GABA_A),
                SynapseId(rn, ngc_pop, rn, pyr_pop, receptor_type=ReceptorType.GABA_A),
            ]
            weights = [self.synaptic_weights[sid] for sid in synapse_ids]
            # STP SynapseIds: connections 0-14 have STP, connection 15 (NGC→Pyr) does not
            stp_sids = synapse_ids[:15]
            self._fused_inhib_cache[inhib_net] = (synapse_ids, weights, stp_sids)

    # =========================================================================
    # CORTICAL LAYER PROCESSING HELPERS
    # =========================================================================
    # These methods consolidate common operations across all cortical layers
    # to ensure biological consistency and reduce code duplication.

    def _run_cortical_inhibitory(
        self,
        inhib_net: CorticalInhibitoryNetwork,
        pyr_pop: PopulationName,
        pv_pop: PopulationName,
        sst_pop: PopulationName,
        vip_pop: PopulationName,
        ngc_pop: PopulationName,
        feedforward_fsi_excitation: Optional[torch.Tensor] = None,
        long_range_excitation: Optional[torch.Tensor] = None,
        ach_spikes: Optional[torch.Tensor] = None,
        vip_external_excitation: Optional[torch.Tensor] = None,
        pv_cross_layer_inhibition: Optional[torch.Tensor] = None,
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
            pyr_pop / pv_pop / sst_pop / vip_pop / ngc_pop: Population names as
                registered in this NeuralRegion (matching _add_internal_connection keys).
            feedforward_fsi_excitation: Optional direct AMPA drive to PV [pv_size]
                (thalamic feedforward; L4 only).
            long_range_excitation: Optional top-down drive in pyr-space [pyr_size],
                routed by the inhibitory network to VIP (via w_lr_vip) and NGC (w_lr_ngc).
            ach_spikes: ACh spike tensor for nicotinic VIP + muscarinic NGC receptors.
            vip_external_excitation: Optional direct AMPA drive to VIP [vip_size]
                from inter-region top-down connections targeting VIP populations.

        Returns:
            Dict with keys:
              total_inhibition, perisomatic_inhibition, dendritic_inhibition,
              pv_spikes, sst_spikes, vip_spikes, ngc_spikes,
              pv_membrane, sst_membrane, vip_membrane, ngc_membrane.
        """
        prev_pyr_spikes = self._prev_spikes(pyr_pop)
        prev_pv_spikes = self._prev_spikes(pv_pop)
        prev_sst_spikes = self._prev_spikes(sst_pop)
        prev_vip_spikes = self._prev_spikes(vip_pop)
        prev_ngc_spikes = self._prev_spikes(ngc_pop)

        # ---- Fused C++ fast path ----
        if self._fused_inhib_available:
            return self._run_cortical_inhibitory_fused(
                inhib_net, prev_pyr_spikes, prev_pv_spikes, prev_sst_spikes,
                prev_vip_spikes, prev_ngc_spikes,
                feedforward_fsi_excitation, long_range_excitation,
                ach_spikes, vip_external_excitation,
                pv_cross_layer_inhibition,
            )

        rn = self.region_name

        def _syn(src: PopulationName, tgt: PopulationName, rxn: ReceptorType) -> SynapseId:
            return SynapseId(rn, src, rn, tgt, receptor_type=rxn)

        # ------------------------------------------------------------------
        # E → I  (Pyr → interneurons via registered STP weights)
        # ------------------------------------------------------------------
        pv_g_exc  = self._integrate_single_synaptic_input(_syn(pyr_pop, pv_pop,  ReceptorType.AMPA), prev_pyr_spikes).g_ampa
        sst_g_exc = self._integrate_single_synaptic_input(_syn(pyr_pop, sst_pop, ReceptorType.AMPA), prev_pyr_spikes).g_ampa
        vip_g_exc = self._integrate_single_synaptic_input(_syn(pyr_pop, vip_pop, ReceptorType.AMPA), prev_pyr_spikes).g_ampa
        ngc_g_exc = self._integrate_single_synaptic_input(_syn(pyr_pop, ngc_pop, ReceptorType.AMPA), prev_pyr_spikes).g_ampa

        # ------------------------------------------------------------------
        # I → I  (from previous-step inhibitory spikes, via registered weights)
        # ------------------------------------------------------------------

        # PV receives inhibition from PV (lateral), SST (weak), VIP (strong disinhibitory).
        # SST receives from PV and VIP (disinhibitory).
        # VIP receives from SST (reciprocal inhibition — Pfeffer et al. 2013).
        # (ngc is not a target of any registered I→I connections)
        pv_g_gaba_a = (
            self._integrate_single_synaptic_input(_syn(pv_pop,  pv_pop,  ReceptorType.GABA_A), prev_pv_spikes).g_gaba_a
            + self._integrate_single_synaptic_input(_syn(sst_pop, pv_pop, ReceptorType.GABA_A), prev_sst_spikes).g_gaba_a
            + self._integrate_single_synaptic_input(_syn(vip_pop, pv_pop, ReceptorType.GABA_A), prev_vip_spikes).g_gaba_a
        )

        sst_g_gaba_a = (
            self._integrate_single_synaptic_input(_syn(pv_pop,  sst_pop, ReceptorType.GABA_A), prev_pv_spikes).g_gaba_a
            + self._integrate_single_synaptic_input(_syn(vip_pop, sst_pop, ReceptorType.GABA_A), prev_vip_spikes).g_gaba_a
            + self._integrate_single_synaptic_input(_syn(sst_pop, sst_pop, ReceptorType.GABA_A), prev_sst_spikes).g_gaba_a
        )
        sst_g_gaba_b = self._integrate_single_synaptic_input(
            _syn(vip_pop, sst_pop, ReceptorType.GABA_B), prev_vip_spikes
        ).g_gaba_b

        # SST → VIP reciprocal inhibition: local Pyr→SST drive suppresses VIP,
        # top-down/ACh→VIP drive suppresses SST. Winner-take-all dynamic.
        vip_g_gaba_a = self._integrate_single_synaptic_input(
            _syn(sst_pop, vip_pop, ReceptorType.GABA_A), prev_sst_spikes
        ).g_gaba_a

        # ------------------------------------------------------------------
        # Cross-layer PV inhibition (from other layer's PV cells)
        # ------------------------------------------------------------------
        if pv_cross_layer_inhibition is not None:
            pv_g_gaba_a = pv_g_gaba_a + pv_cross_layer_inhibition

        # ------------------------------------------------------------------
        # Run interneurons (gap junction + long-range + ACh handled inside)
        # ------------------------------------------------------------------
        inhib_out = inhib_net.forward(
            pv_g_exc=pv_g_exc,
            pv_g_gaba_a=pv_g_gaba_a,
            sst_g_exc=sst_g_exc,
            sst_g_gaba_a=sst_g_gaba_a,
            sst_g_gaba_b=sst_g_gaba_b,
            vip_g_exc_from_pyr=vip_g_exc,
            vip_g_gaba_a=vip_g_gaba_a,
            ngc_g_exc_from_pyr=ngc_g_exc,
            feedforward_excitation=feedforward_fsi_excitation,
            long_range_excitation=long_range_excitation,
            ach_spikes=ach_spikes,
            vip_external_excitation=vip_external_excitation,
        )
        pv_spikes  = inhib_out["pv_spikes"]
        sst_spikes = inhib_out["sst_spikes"]
        vip_spikes = inhib_out["vip_spikes"]
        ngc_spikes = inhib_out["ngc_spikes"]

        # ------------------------------------------------------------------
        # I → E  (previous-step inhibitory → pyramidal via registered STP weights)
        # Uses previous-step inhibitory spikes (1-step causal delay, biologically
        # correct: real I→E synaptic transmission takes 0.3-1ms).
        # ------------------------------------------------------------------
        perisomatic_inhibition = self._integrate_single_synaptic_input(_syn(pv_pop,  pyr_pop, ReceptorType.GABA_A), prev_pv_spikes).g_gaba_a
        dendritic_inhibition   = self._integrate_single_synaptic_input(_syn(sst_pop, pyr_pop, ReceptorType.GABA_A), prev_sst_spikes).g_gaba_a
        vip_to_pyr             = self._integrate_single_synaptic_input(_syn(vip_pop, pyr_pop, ReceptorType.GABA_A), prev_vip_spikes).g_gaba_a
        ngc_to_pyr             = self._integrate_single_synaptic_input(_syn(ngc_pop, pyr_pop, ReceptorType.GABA_A), prev_ngc_spikes).g_gaba_a

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

    def _run_cortical_inhibitory_fused(
        self,
        inhib_net: CorticalInhibitoryNetwork,
        prev_pyr_spikes: torch.Tensor,
        prev_pv_spikes: torch.Tensor,
        prev_sst_spikes: torch.Tensor,
        prev_vip_spikes: torch.Tensor,
        prev_ngc_spikes: torch.Tensor,
        feedforward_fsi_excitation: Optional[torch.Tensor],
        long_range_excitation: Optional[torch.Tensor],
        ach_spikes: Optional[torch.Tensor],
        vip_external_excitation: Optional[torch.Tensor],
        pv_cross_layer_inhibition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Fused C++ path: all 15 STP-modulated matmuls in one kernel call."""
        cache = self._fused_inhib_cache[inhib_net]
        _synapse_ids, weights, stp_sids = cache

        # Collect STP efficacy vectors (pre-computed in Phase 0 STPBatch)
        pre_eff = self._precomputed_stp_efficacy
        if pre_eff is not None:
            efficacies = [pre_eff[sid] for sid in stp_sids]
        else:
            # Fallback: compute STP efficacy inline (should not happen after Phase 0)
            stp_mods = self.stp_modules
            # Source mapping: connections 0-3 use pyr, 4 pv, 5 sst, 6 vip,
            # 7 pv, 8-9 vip, 10 sst, 11 sst, 12 pv, 13 sst, 14 vip
            src_map = [
                prev_pyr_spikes, prev_pyr_spikes, prev_pyr_spikes, prev_pyr_spikes, prev_pv_spikes,
                prev_sst_spikes, prev_vip_spikes, prev_pv_spikes,  prev_vip_spikes, prev_vip_spikes,
                prev_sst_spikes, prev_sst_spikes, prev_pv_spikes,  prev_sst_spikes, prev_vip_spikes,
            ]
            efficacies = [stp_mods[sid].forward(src_map[i]) for i, sid in enumerate(stp_sids)]

        # Single fused C++ call replaces 16 _integrate_single_synaptic_input calls
        results = _ci_fused.cortical_inhibitory_step(
            prev_pyr_spikes, prev_pv_spikes, prev_sst_spikes, prev_vip_spikes, prev_ngc_spikes, weights, efficacies
        )

        # Unpack: [0-3] E→I exc, [4-7] I→I inh, [8-11] I→E inh, [12] total
        pv_g_exc, sst_g_exc, vip_g_exc, ngc_g_exc = results[0], results[1], results[2], results[3]
        pv_g_gaba_a, sst_g_gaba_a, sst_g_gaba_b, vip_g_gaba_a = results[4], results[5], results[6], results[7]

        # Cross-layer PV inhibition (computed outside the fused kernel)
        if pv_cross_layer_inhibition is not None:
            pv_g_gaba_a = pv_g_gaba_a + pv_cross_layer_inhibition

        # Run interneurons (gap junction + long-range + ACh handled inside)
        inhib_out = inhib_net.forward(
            pv_g_exc=pv_g_exc,
            pv_g_gaba_a=pv_g_gaba_a,
            sst_g_exc=sst_g_exc,
            sst_g_gaba_a=sst_g_gaba_a,
            sst_g_gaba_b=sst_g_gaba_b,
            vip_g_exc_from_pyr=vip_g_exc,
            vip_g_gaba_a=vip_g_gaba_a,
            ngc_g_exc_from_pyr=ngc_g_exc,
            feedforward_excitation=feedforward_fsi_excitation,
            long_range_excitation=long_range_excitation,
            ach_spikes=ach_spikes,
            vip_external_excitation=vip_external_excitation,
        )

        perisomatic_inhibition = results[8]
        dendritic_inhibition = results[9]
        total_inhibition = results[12]

        return {
            "total_inhibition":       total_inhibition,
            "perisomatic_inhibition": perisomatic_inhibition,
            "dendritic_inhibition":   dendritic_inhibition,
            "pv_spikes":    inhib_out["pv_spikes"],
            "sst_spikes":   inhib_out["sst_spikes"],
            "vip_spikes":   inhib_out["vip_spikes"],
            "ngc_spikes":   inhib_out["ngc_spikes"],
            "pv_membrane":  inhib_out["pv_membrane"],
            "sst_membrane": inhib_out["sst_membrane"],
            "vip_membrane": inhib_out["vip_membrane"],
            "ngc_membrane": inhib_out["ngc_membrane"],
        }

    def _compute_layer_inhibition(
        self,
        inhibitory_network: CorticalInhibitoryNetwork,
        ach_spikes: Optional[torch.Tensor],
        *,
        feedforward_fsi_excitation: Optional[torch.Tensor] = None,
        additional_inhibition: Optional[torch.Tensor] = None,
        long_range_excitation: Optional[torch.Tensor] = None,
        vip_external_excitation: Optional[torch.Tensor] = None,
        pv_cross_layer_inhibition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Compute total inhibition for a cortical layer.

        Args:
            inhibitory_network: Layer's CorticalInhibitoryNetwork.
            ach_spikes: Raw NB ACh spikes for nicotinic VIP + muscarinic NGC.
            feedforward_fsi_excitation: Direct AMPA to PV [pv_size] (L4 thalamic).
            additional_inhibition: Extra inhibition to add (e.g., predictive coding).
            long_range_excitation: Top-down input in pyr-space [pyr_size].
            vip_external_excitation: Direct AMPA to VIP [vip_size] from inter-region
                top-down connections targeting VIP populations (Pi et al. 2013).
            pv_cross_layer_inhibition: GABA_A from cross-layer PV cells [pv_size]
                for translaminar gamma coherence.

        Returns:
            (g_inh_total, inhibitory_output_dict or None)
        """
        pyr_pop, pv_pop, sst_pop, vip_pop, ngc_pop = self._inhib_populations[inhibitory_network]

        inhib_output = self._run_cortical_inhibitory(
            inhib_net=inhibitory_network,
            pyr_pop=pyr_pop,
            pv_pop=pv_pop,
            sst_pop=sst_pop,
            vip_pop=vip_pop,
            ngc_pop=ngc_pop,
            feedforward_fsi_excitation=feedforward_fsi_excitation,
            long_range_excitation=long_range_excitation,
            ach_spikes=ach_spikes,
            vip_external_excitation=vip_external_excitation,
            pv_cross_layer_inhibition=pv_cross_layer_inhibition,
        )
        g_inh_total = inhib_output["total_inhibition"]

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
        g_ampa, g_nmda = split_excitatory_conductance(g_exc, nmda_ratio=0.05)
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
        pyr_neurons: ConductanceLIF,
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
            inhibitory_network=inhibitory_network,
            ach_spikes=ach_spikes,
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
        if not self.config.neuromodulation_disabled:
            self._process_neuromodulator(
                self.da_receptor,
                self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.DA_MESOCORTICAL),
                '_da_concentration',
                self.config.da_fractions.as_tuple(),
            )
            self._process_neuromodulator(
                self.ne_receptor,
                self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.NE),
                '_ne_concentration',
                self.config.ne_fractions.as_tuple(),
            )
            nb_ach_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.ACH)
            self._process_neuromodulator(
                self.ach_receptor,
                nb_ach_spikes,
                '_ach_concentration',
                self.config.ach_fractions.as_tuple(),
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
        l5_delayed = self._l5_spike_buffer.read(self._l5_l4_delay_steps).float()

        # Disynaptic pathway (Dale's Law compliant): L5_PYR → L4_SST_PRED → L4_PYR
        l5_sst_pred_conductance = self._integrate_single_synaptic_input(self._l5_sst_pred_synapse, l5_delayed).g_ampa
        sst_pred_g_ampa, sst_pred_g_nmda = split_excitatory_conductance(l5_sst_pred_conductance, nmda_ratio=0.1)
        l4_sst_pred_spikes, _ = self.l4_sst_pred_neurons.forward(
            g_ampa_input=ConductanceTensor(sst_pred_g_ampa),
            g_nmda_input=ConductanceTensor(sst_pred_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        # Use PREVIOUS-STEP SST_PRED spikes for L4 pyramidal inhibition (1-step causal delay)
        prev_sst_pred = self._prev_spikes(CortexPopulation.L4_SST_PRED)
        l5_l4_inhibition = self._integrate_single_synaptic_input(self._sst_pred_l4_synapse, prev_sst_pred).g_gaba_a
        # Precision weighting: scale by L5 population activity (confidence)
        l5_activity = torch.mean(l5_delayed).item()
        l5_precision = config.precision_min + (config.precision_max - config.precision_min) * l5_activity
        l5_l4_inhibition = l5_l4_inhibition * l5_precision

        # L6 (combined) → L4 prediction
        l6a_delayed = self._l6a_spike_buffer.read(self._l6_l4_delay_steps)
        l6b_delayed = self._l6b_spike_buffer.read(self._l6_l4_delay_steps)
        l6_delayed = torch.cat([l6a_delayed, l6b_delayed], dim=-1).float()
        l6_l4_inhibition = self._integrate_single_synaptic_input(self._sid_l6_l4, l6_delayed).g_gaba_a
        l6_activity = torch.mean(l6_delayed).item()
        l6_precision = config.precision_min + (config.precision_max - config.precision_min) * l6_activity
        l6_l4_inhibition = l6_l4_inhibition * l6_precision

        l4_pred_inhibition = l5_l4_inhibition + l6_l4_inhibition

        # ---- NE gain (β-adrenergic increases L4 excitability) ----
        l4_g_exc = l4_g_exc * compute_ne_gain(self._ne_concentration_l4.mean().item())

        # ---- Inhibitory network (PV also receives direct thalamic drive) ----
        l4_g_inh, l4_inhib_output = self._compute_layer_inhibition(
            inhibitory_network=self.l4_inhibitory,
            ach_spikes=nb_ach_spikes,
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
        Apical GABA inhibition for L2/3 and L5 uses PREVIOUS-STEP NGC spikes
        (1-step causal delay).

        Args:
            synaptic_inputs: Full region synaptic input dict.

        Returns:
            (l1_ngc_spikes, l23_apical_inh, l5_apical_inh)
        """
        # (a) L2/3 → L1 NGC via apical collateral (previous timestep)
        prev_l23 = self._prev_spikes(CortexPopulation.L23_PYR)
        l1_ngc_g_exc_from_l23 = self._integrate_single_synaptic_input(self._sid_l23_ngc, prev_l23).g_ampa

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

        # Apical GABA_A inhibition from PREVIOUS-STEP NGC spikes (1-step causal delay)
        prev_ngc = self._prev_spikes(CortexPopulation.L1_NGC)
        l23_apical_inh = self._integrate_single_synaptic_input(self._sid_ngc_l23, prev_ngc).g_gaba_a
        l5_apical_inh = self._integrate_single_synaptic_input(self._sid_ngc_l5, prev_ngc).g_gaba_a

        return l1_ngc_spikes, l23_apical_inh, l5_apical_inh

    def _step_l23(
        self,
        synaptic_inputs: SynapticInput,
        nb_ach_spikes: Optional[torch.Tensor],
        l23_apical_inh: torch.Tensor,
        l4_sst_cross_layer_inh: torch.Tensor,
        l5_sst_cross_layer_apical_inh: torch.Tensor,
        l23_pv_cross_layer_inh: Optional[torch.Tensor],
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
            l4_sst_cross_layer_inh: GABA_A from L4 SST onto L2/3 basal dendrites
                (ascending surround suppression).
            l5_sst_cross_layer_apical_inh: GABA_A from L5 SST onto L2/3 apical tufts
                (predictive error gating from deep layers).
            l23_pv_cross_layer_inh: GABA_A from L5 PV onto L2/3 PV
                (ascending gamma synchronization).

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
                l4_source_activity += source_spikes.sum()

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
        # Extract direct top-down drive to L2/3 VIP from inter-region connections
        # (Pi et al. 2013: feedback preferentially targets VIP for disinhibition)
        l23_vip_external = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l23_inhibitory.vip_size,
            filter_by_target_population=CortexPopulation.L23_INHIBITORY_VIP,
        ).g_ampa

        l23_g_inh, l23_inhib_output = self._compute_layer_inhibition(
            inhibitory_network=self.l23_inhibitory,
            ach_spikes=nb_ach_spikes,
            long_range_excitation=l23_td,  # top-down input gates VIP disinhibition
            vip_external_excitation=l23_vip_external,
            pv_cross_layer_inhibition=l23_pv_cross_layer_inh,
        )

        # Cross-layer SST inhibition adds to basal inhibition (L4 SST surround suppression)
        l23_g_inh = l23_g_inh + l4_sst_cross_layer_inh

        # Two-compartment integration; L1 NGC + L5 SST apical inhibition gates apical tuft
        l23_spikes, _l23_membrane, _l23_V_dend = self._integrate_and_spike_two_compartment(
            g_exc_basal=F.relu(l23_basal_input),
            g_inh_basal=l23_g_inh,
            g_exc_apical=F.relu(l23_apical_input),
            g_inh_apical=l23_apical_inh + l5_sst_cross_layer_apical_inh,
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
        l5_pv_cross_layer_inh: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Process Layer 5 (subcortical output layer).

        Integrates feedforward input from L2/3 (via axonal delay), top-down
        feedback, and inhibitory network. Uses two-compartment neurons with
        basal (L2/3 feedforward) and apical (top-down) routing.

        Args:
            synaptic_inputs: Full region synaptic input dict.
            nb_ach_spikes: Raw nucleus basalis ACh spikes for inhibitory network.
            l5_apical_inh: GABA_A conductance from L1 NGC onto L5 apical tufts.
            l5_pv_cross_layer_inh: GABA_A from L2/3 PV onto L5 PV
                (descending gamma synchronization).

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

        # Top-down modulation to L5 apical tufts — mirrors the L2/3 apical mechanism.
        # Biology: L5 apical tufts in L1 receive top-down feedback from higher
        # cortical areas (PFC, association cortex, contralateral cortex).
        l5_td = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l5_pyr_size,
            filter_by_target_population=CortexPopulation.L5_PYR,
        ).g_ampa

        # Extract direct top-down drive to L5 VIP from inter-region connections
        l5_vip_external = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.l5_inhibitory.vip_size,
            filter_by_target_population=CortexPopulation.L5_INHIBITORY_VIP,
        ).g_ampa

        l5_g_inh, l5_inhib_output = self._compute_layer_inhibition(
            inhibitory_network=self.l5_inhibitory,
            ach_spikes=nb_ach_spikes,
            long_range_excitation=l5_td,  # top-down also modulates L5 VIP/NGC
            vip_external_excitation=l5_vip_external,
            pv_cross_layer_inhibition=l5_pv_cross_layer_inh,
        )

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

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Process one timestep through the full layered cortical circuit."""
        # =====================================================================
        # NEUROMODULATOR UPDATE: Update DA/NE/ACh receptor dynamics and slice per-layer concentration buffers.
        # =====================================================================
        nb_ach_spikes = self._update_neuromodulators(neuromodulator_inputs)

        # Cache nn.Module submodules once — avoids __getattr__ per access
        _l23_inhib = self.l23_inhibitory
        _l4_inhib  = self.l4_inhibitory
        _l5_inhib  = self.l5_inhibitory
        _l6a_inhib = self.l6a_inhibitory
        _l6b_inhib = self.l6b_inhibitory
        _device    = self.device

        # =====================================================================
        # LAYERS 1-6: Process each layer in order, passing synaptic inputs and neuromodulator signals.
        # =====================================================================
        l4_spikes, l4_sst_pred_spikes, l4_inhib_output = self._step_l4(synaptic_inputs, nb_ach_spikes)
        l1_ngc_spikes, l23_apical_inh, l5_apical_inh = self._step_l1_ngc(synaptic_inputs)

        # =====================================================================
        # CROSS-LAYER INHIBITION: compute inter-laminar inhibitory drive
        # from PREVIOUS-STEP spikes (1-step causal delay).
        # =====================================================================
        prev_l4_sst = self._prev_spikes(CortexPopulation.L4_INHIBITORY_SST)
        prev_l5_sst = self._prev_spikes(CortexPopulation.L5_INHIBITORY_SST)
        prev_l23_pv = self._prev_spikes(CortexPopulation.L23_INHIBITORY_PV)
        prev_l5_pv = self._prev_spikes(CortexPopulation.L5_INHIBITORY_PV)

        l4_sst_cross_inh = self._integrate_single_synaptic_input(self._sid_l4sst_l23, prev_l4_sst).g_gaba_a
        l5_sst_cross_apical_inh = self._integrate_single_synaptic_input(self._sid_l5sst_l23, prev_l5_sst).g_gaba_a
        l23pv_to_l5pv = self._integrate_single_synaptic_input(self._sid_l23pv_l5pv, prev_l23_pv).g_gaba_a
        l5pv_to_l23pv = self._integrate_single_synaptic_input(self._sid_l5pv_l23pv, prev_l5_pv).g_gaba_a

        l23_spikes, l23_inhib_output = self._step_l23(
            synaptic_inputs, nb_ach_spikes, l23_apical_inh,
            l4_sst_cross_inh, l5_sst_cross_apical_inh, l5pv_to_l23pv,
        )
        l5_spikes, l5_inhib_output = self._step_l5(
            synaptic_inputs, nb_ach_spikes, l5_apical_inh, l23pv_to_l5pv,
        )

        # =====================================================================
        # LAYER 6: CORTICOTHALAMIC FEEDBACK (L6a → TRN, L6b → relay)
        # =====================================================================
        l6a_spikes, _l6a_g_exc, l6a_inhib_output = self._step_single_compartment_deep_layer(
            synapse_id_from_l23=self._sid_l23_l6a,
            l23_delayed=self._l23_spike_buffer.read(self._l23_l6a_delay_steps),
            pyr_size=self.l6a_pyr_size,
            pyr_neurons=self.l6a_neurons,
            inhibitory_network=_l6a_inhib,
            ne_concentration=self._ne_concentration_l6a,
            ach_spikes=nb_ach_spikes,
        )
        assert l6a_spikes.shape == (self.l6a_pyr_size,)

        l6b_spikes, _l6b_g_exc, l6b_inhib_output = self._step_single_compartment_deep_layer(
            synapse_id_from_l23=self._sid_l23_l6b,
            l23_delayed=self._l23_spike_buffer.read(self._l23_l6b_delay_steps),
            pyr_size=self.l6b_pyr_size,
            pyr_neurons=self.l6b_neurons,
            inhibitory_network=_l6b_inhib,
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
            CortexPopulation.L23_INHIBITORY_PV:  self._inh_spikes(l23_inhib_output, "pv_spikes",  _l23_inhib.pv_size, _device),
            CortexPopulation.L23_INHIBITORY_SST: self._inh_spikes(l23_inhib_output, "sst_spikes", _l23_inhib.sst_size, _device),
            CortexPopulation.L23_INHIBITORY_VIP: self._inh_spikes(l23_inhib_output, "vip_spikes", _l23_inhib.vip_size, _device),
            CortexPopulation.L23_INHIBITORY_NGC: self._inh_spikes(l23_inhib_output, "ngc_spikes", _l23_inhib.ngc_size, _device),
            # L4 inhibitory populations
            CortexPopulation.L4_INHIBITORY_PV:  self._inh_spikes(l4_inhib_output, "pv_spikes",  _l4_inhib.pv_size, _device),
            CortexPopulation.L4_INHIBITORY_SST: self._inh_spikes(l4_inhib_output, "sst_spikes", _l4_inhib.sst_size, _device),
            CortexPopulation.L4_INHIBITORY_VIP: self._inh_spikes(l4_inhib_output, "vip_spikes", _l4_inhib.vip_size, _device),
            CortexPopulation.L4_INHIBITORY_NGC: self._inh_spikes(l4_inhib_output, "ngc_spikes", _l4_inhib.ngc_size, _device),
            # L5 inhibitory populations
            CortexPopulation.L5_INHIBITORY_PV:  self._inh_spikes(l5_inhib_output, "pv_spikes",  _l5_inhib.pv_size, _device),
            CortexPopulation.L5_INHIBITORY_SST: self._inh_spikes(l5_inhib_output, "sst_spikes", _l5_inhib.sst_size, _device),
            CortexPopulation.L5_INHIBITORY_VIP: self._inh_spikes(l5_inhib_output, "vip_spikes", _l5_inhib.vip_size, _device),
            CortexPopulation.L5_INHIBITORY_NGC: self._inh_spikes(l5_inhib_output, "ngc_spikes", _l5_inhib.ngc_size, _device),
            # L6a inhibitory populations
            CortexPopulation.L6A_INHIBITORY_PV:  self._inh_spikes(l6a_inhib_output, "pv_spikes",  _l6a_inhib.pv_size, _device),
            CortexPopulation.L6A_INHIBITORY_SST: self._inh_spikes(l6a_inhib_output, "sst_spikes", _l6a_inhib.sst_size, _device),
            CortexPopulation.L6A_INHIBITORY_VIP: self._inh_spikes(l6a_inhib_output, "vip_spikes", _l6a_inhib.vip_size, _device),
            CortexPopulation.L6A_INHIBITORY_NGC: self._inh_spikes(l6a_inhib_output, "ngc_spikes", _l6a_inhib.ngc_size, _device),
            # L6b inhibitory populations
            CortexPopulation.L6B_INHIBITORY_PV:  self._inh_spikes(l6b_inhib_output, "pv_spikes",  _l6b_inhib.pv_size, _device),
            CortexPopulation.L6B_INHIBITORY_SST: self._inh_spikes(l6b_inhib_output, "sst_spikes", _l6b_inhib.sst_size, _device),
            CortexPopulation.L6B_INHIBITORY_VIP: self._inh_spikes(l6b_inhib_output, "vip_spikes", _l6b_inhib.vip_size, _device),
            CortexPopulation.L6B_INHIBITORY_NGC: self._inh_spikes(l6b_inhib_output, "ngc_spikes", _l6b_inhib.ngc_size, _device),
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

        return region_outputs

    def _apply_plasticity(
        self,
        synaptic_inputs: SynapticInput,
        region_outputs: RegionOutput,
    ) -> None:
        """Apply continuous STDP+BCM learning with neuromodulator gating.

        Called automatically at each forward() timestep.  Neuromodulators gate
        *how much* weight change occurs from spike-timing events; they do not
        trigger learning by themselves.
        """
        config = self.config

        if config.learning_disabled:
            return

        # Lazy-register external (inter-region) E→E input learning.
        # The base class apply_learning() dispatches these after _step()
        # completes; we just need to ensure a strategy is registered.
        # Only excitatory (AMPA/NMDA) afferents get Hebbian STDP; inhibitory
        # inputs are skipped (excitatory STDP is inappropriate for GABA synapses).
        device = self.device
        for synapse_id in list(synaptic_inputs.keys()):
            if self.get_learning_strategy(synapse_id) is None:
                if synapse_id.receptor_type.is_excitatory:
                    self._add_learning_strategy(
                        synapse_id, self._external_stdp_strategy, device=device,
                    )

        l23_spikes         = region_outputs[CortexPopulation.L23_PYR]
        l4_spikes          = region_outputs[CortexPopulation.L4_PYR]
        l5_spikes          = region_outputs[CortexPopulation.L5_PYR]
        l6a_spikes         = region_outputs[CortexPopulation.L6A_PYR]
        l6b_spikes         = region_outputs[CortexPopulation.L6B_PYR]
        l4_sst_pred_spikes = region_outputs[CortexPopulation.L4_SST_PRED]

        # Per-layer neuromodulator scalars (one dict per layer)
        nm_l23 = self._neuromod_scalars('l23')
        # nm_l4  = self._neuromod_scalars('l4')
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
        prev_l23 = self._prev_spikes(CortexPopulation.L23_PYR)
        self._apply_learning(self._sid_l23_l23, prev_l23, l23_spikes, **nm_l23)
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

        # NGC → Pyr: iSTDP homeostatically tunes apical tuft inhibition
        l1_ngc_spikes = region_outputs[CortexPopulation.L1_NGC]
        self._apply_learning(self._sid_ngc_l23, l1_ngc_spikes, l23_spikes)
        self._apply_learning(self._sid_ngc_l5,  l1_ngc_spikes, l5_spikes)

        # Cross-layer SST → L2/3 Pyr: iSTDP tunes surround suppression and error gating
        l4_sst_spikes = region_outputs[CortexPopulation.L4_INHIBITORY_SST]
        l5_sst_spikes = region_outputs[CortexPopulation.L5_INHIBITORY_SST]
        self._apply_learning(self._sid_l4sst_l23, l4_sst_spikes, l23_spikes)
        self._apply_learning(self._sid_l5sst_l23, l5_sst_spikes, l23_spikes)

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        """Map external synapse target population to layer-specific neuromodulator kwargs."""
        _pop_to_layer = {
            CortexPopulation.L23_PYR: 'l23',
            CortexPopulation.L4_PYR:  'l4',
            CortexPopulation.L5_PYR:  'l5',
        }
        layer = _pop_to_layer.get(synapse_id.target_population)
        return self._neuromod_scalars(layer) if layer is not None else {}
