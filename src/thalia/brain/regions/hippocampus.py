"""
Trisynaptic Hippocampus - Biologically-Accurate DG→CA3→CA1 Episodic Memory Circuit.

This implements the classic hippocampal trisynaptic circuit for episodic memory:
- **Dentate Gyrus (DG)**: Pattern SEPARATION via sparse coding (~2-5% active)
- **CA3**: Pattern COMPLETION via recurrent connections (autoassociative memory)
- **CA1**: Output/comparison layer detecting match vs mismatch

**Key Biological Features**:
===========================
1. **THETA MODULATION** (6-10 Hz oscillations):
   - Theta trough (0-Ï€): Encoding phase (CA3 learning enabled)
   - Theta peak (Ï€-2Ï€): Retrieval phase (comparison active)
   - Phase separation prevents interference between encoding and retrieval

2. **FEEDFORWARD INHIBITION**:
   - Stimulus onset triggers transient inhibition
   - Naturally clears residual activity
   - Fast-spiking interneuron-like dynamics

3. **CONTINUOUS DYNAMICS**:
   - Everything flows naturally
   - Membrane potentials decay via LIF dynamics
   - Theta phase advances continuously
   - Smooth transitions between encoding and retrieval phases
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from thalia import GlobalConfig
from thalia.brain.configs import HippocampusConfig, HippocampalPopulationConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    ConductanceLIF,
    TwoCompartmentLIF,
    build_conductance_lif_config,
    build_two_compartment_config,
    split_excitatory_conductance,
)
from thalia.brain.synapses import (
    NMReceptorType,
    STPConfig,
    WeightInitializer,
    make_neuromodulator_receptor,
)
from thalia.learning import (
    InhibitorySTDPConfig,
    InhibitorySTDPStrategy,
    MetaplasticityConfig,
    MetaplasticityStrategy,
    STDPConfig,
    STDPStrategy,
    TagAndCaptureConfig,
    TagAndCaptureStrategy,
    ThreeFactorConfig,
    ThreeFactorStrategy,
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
    clamp_weights,
    compute_ach_recurrent_suppression,
    compute_ne_gain,
    decay_tensor,
)

from .hippocampus_inhibitory_network import HippocampalInhibitoryNetwork
from .neural_region import NeuralRegion
from .population_names import HippocampusPopulation, MedialSeptumPopulation
from .region_registry import register_region
from .stimulus_gating import StimulusGating


@register_region(
    "hippocampus",
    aliases=["trisynaptic", "trisynaptic_hippocampus"],
    description="DG→CA3→CA1 trisynaptic circuit with theta-modulated encoding/retrieval and episodic memory",
)
class Hippocampus(NeuralRegion[HippocampusConfig]):
    """Biologically-accurate hippocampus with DG→CA3→CA1 trisynaptic circuit."""

    # Mesolimbic DA (VTA → hippocampus) modulates Schaffer collateral LTP and replay.
    # NE from LC modulates novelty-driven encoding and theta power.
    # ACh from nucleus basalis gates encoding vs. retrieval modes (nicotinic receptors).
    # 'ach_septal': slow muscarinic (M1) modulation from medial septum cholinergic neurons.
    # 'ach_septal': slow muscarinic (M1) modulation from medial septum cholinergic neurons.
    # '5ht': 5-HT2C receptors on CA3 pyramidals suppress recurrent NMDA drive under
    #         aversive/stress states (Bhagya et al. 2022; Barnes & Sharp 1999).
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.DA_MESOLIMBIC,
        NeuromodulatorChannel.NE,
        NeuromodulatorChannel.ACH,
        NeuromodulatorChannel.ACH_SEPTAL,
        NeuromodulatorChannel.SHT,
    ]

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: HippocampusConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize trisynaptic hippocampus."""
        super().__init__(config, population_sizes, region_name, device=device)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.dg_size = population_sizes[HippocampusPopulation.DG]
        self.ca3_size = population_sizes[HippocampusPopulation.CA3]
        self.ca2_size = population_sizes[HippocampusPopulation.CA2]
        self.ca1_size = population_sizes[HippocampusPopulation.CA1]

        # =====================================================================
        # INTERNAL STATE VARIABLES (for dynamics and plasticity)
        # =====================================================================
        # Encoding/retrieval modulation state
        # _prev_encoding_mod: OLM-based fallback (t-1 OLM firing → t encoding mode)
        # _ms_theta_trace: fast exponential trace of MS ACH→CA1 spikes (τ=30ms)
        #   used as primary encoding_mod driver when MS is connected.
        self._prev_encoding_mod: float = 0.5  # OLM-based fallback; balanced default
        self._ms_theta_trace: float = 0.0     # MS ACH trace; 0 until MS is wired

        # NMDA trace for temporal integration (slow kinetics); None until first forward step
        self.nmda_trace: Optional[torch.Tensor]
        self.register_buffer("nmda_trace", None)

        # Spontaneous replay (sharp-wave ripple) detection.
        # ripple_detected: True on every timestep within the SWR window (not just onset).
        # ripple_duration_steps: countdown in timesteps; refreshed on each high-rate CA3 burst.
        self.ripple_detected: bool = False
        self.ripple_duration_steps: int = 0
        # CA3 attractor pattern captured at ripple onset — re-injected each timestep of the
        # window to sustain multi-timestep replay.
        self._swr_ca3_pattern: Optional[torch.Tensor]
        self.register_buffer("_swr_ca3_pattern", None)

        # CA1 prediction signal: match vs. mismatch between CA3 stored pattern and EC input
        # match_level: fraction of CA1 neurons co-driven by BOTH CA3 Schaffer AND EC direct
        #   → recognition/familiarity signal (suppresses VTA novelty response)
        # mismatch_level: EC input not predicted by CA3
        #   → novelty/prediction-error signal; propagates via Subiculum → VTA DA burst
        self.ca1_match_level: float = 0.0
        self.ca1_mismatch_level: float = 0.0

        # =====================================================================
        # HIPPOCAMPAL EXCITATORY NEURONS (LIF with adaptation for sparse coding)
        # =====================================================================
        dg_overrides: HippocampalPopulationConfig = config.population_overrides[HippocampusPopulation.DG]
        ca3_overrides: HippocampalPopulationConfig = config.population_overrides[HippocampusPopulation.CA3]
        ca2_overrides: HippocampalPopulationConfig = config.population_overrides[HippocampusPopulation.CA2]
        ca1_overrides: HippocampalPopulationConfig = config.population_overrides[HippocampusPopulation.CA1]

        # Create LIF neurons for each layer using factory functions
        # DG: Sparse coding requires high threshold
        self.dg_neurons: ConductanceLIF
        self.dg_neurons = self._create_and_register_neuron_population(
            population_name=HippocampusPopulation.DG,
            n_neurons=self.dg_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=build_conductance_lif_config(dg_overrides, self.dg_size, device, dendrite_cv=0.25),
        )

        # CA3 — Autoassociative memory: recurrent excitation, pattern completion.
        # Also gets spike-frequency adaptation to prevent frozen attractors
        # Two-compartment: basal dendrites receive DG mossy fibers + recurrents;
        # apical compartment is available for future EC direct-path feedback.
        self.ca3_neurons: TwoCompartmentLIF
        self.ca3_neurons = self._create_and_register_neuron_population(
            population_name=HippocampusPopulation.CA3,
            n_neurons=self.ca3_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=build_two_compartment_config(
                ca3_overrides, self.ca3_size, device,
                g_nmda_max=0.5,  # CA3 autoassociative recurrents risk NMDA accumulation during SWR
                g_Ca_spike=0.38,
            ),
        )

        # CA2: Social memory and temporal context - moderate threshold for selectivity
        self.ca2_neurons: ConductanceLIF
        self.ca2_neurons = self._create_and_register_neuron_population(
            population_name=HippocampusPopulation.CA2,
            n_neurons=self.ca2_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=build_conductance_lif_config(ca2_overrides, self.ca2_size, device, dendrite_cv=0.25),
        )

        # CA1: Output layer – two-compartment pyramidal (CA3 Schaffer collateral basal, EC direct apical)
        self.ca1_neurons: TwoCompartmentLIF
        self.ca1_neurons = self._create_and_register_neuron_population(
            population_name=HippocampusPopulation.CA1,
            n_neurons=self.ca1_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=build_two_compartment_config(ca1_overrides, self.ca1_size, device),
        )

        # =====================================================================
        # HIPPOCAMPAL INHIBITORY NETWORKS (with OLM cells for emergent theta)
        # =====================================================================
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

        # DG inhibitory network: Minimal inhibition for pattern separation
        self.dg_inhibitory = HippocampalInhibitoryNetwork(
            population_name=HippocampusPopulation.DG_INHIBITORY,
            pyr_size=self.dg_size,
            total_inhib_fraction=dg_overrides.total_inhib_fraction,
            v_threshold_olm=dg_overrides.v_threshold_olm,
            v_threshold_bistratified=dg_overrides.v_threshold_bistratified,
            _create_and_register_neurons_fn=_create_and_register_neurons,
            dt_ms=config.dt_ms,
            device=device,
        )

        # CA3 inhibitory network: Moderate inhibition for pattern completion stability
        self.ca3_inhibitory = HippocampalInhibitoryNetwork(
            population_name=HippocampusPopulation.CA3_INHIBITORY,
            pyr_size=self.ca3_size,
            total_inhib_fraction=ca3_overrides.total_inhib_fraction,
            v_threshold_olm=ca3_overrides.v_threshold_olm,
            v_threshold_bistratified=ca3_overrides.v_threshold_bistratified,
            _create_and_register_neurons_fn=_create_and_register_neurons,
            dt_ms=config.dt_ms,
            device=device,
        )

        # CA2 inhibitory network: Social/temporal context processing
        self.ca2_inhibitory = HippocampalInhibitoryNetwork(
            population_name=HippocampusPopulation.CA2_INHIBITORY,
            pyr_size=self.ca2_size,
            total_inhib_fraction=ca2_overrides.total_inhib_fraction,
            v_threshold_olm=ca2_overrides.v_threshold_olm,
            v_threshold_bistratified=ca2_overrides.v_threshold_bistratified,
            _create_and_register_neurons_fn=_create_and_register_neurons,
            dt_ms=config.dt_ms,
            device=device,
        )

        # CA1 inhibitory network: PV, OLM, Bistratified cells
        # OLM cells phase-lock to septal GABA → emergent encoding/retrieval
        self.ca1_inhibitory = HippocampalInhibitoryNetwork(
            population_name=HippocampusPopulation.CA1_INHIBITORY,
            pyr_size=self.ca1_size,
            total_inhib_fraction=ca1_overrides.total_inhib_fraction,
            v_threshold_olm=ca1_overrides.v_threshold_olm,
            v_threshold_bistratified=ca1_overrides.v_threshold_bistratified,
            _create_and_register_neurons_fn=_create_and_register_neurons,
            dt_ms=config.dt_ms,
            device=device,
        )

        # Stimulus gating module (transient inhibition at stimulus changes)
        self.stimulus_gating = StimulusGating(
            n_neurons=self.dg_size,
            threshold=config.ffi_threshold,
            max_inhibition=config.ffi_strength * 10.0,  # Scale to appropriate range
            decay_rate=1.0 - (1.0 / config.ffi_tau),    # Convert tau to rate
            steepness=10.0,
            device=device,
        )

        # Inhibitory STDP (Vogels et al. 2011) for PV→Pyr and OLM→Pyr connections.
        # Per-subfield strategy instances so each subregion tunes independently.
        istdp_cfg = InhibitorySTDPConfig(
            learning_rate=config.istdp_learning_rate,
            tau_istdp=config.istdp_tau_ms,
            alpha=config.istdp_alpha,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        )
        self.istdp_pv_dg:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_pv_ca3: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_pv_ca2: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_pv_ca1: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_olm_dg:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_olm_ca3: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_olm_ca2: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_olm_ca1: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_bist_dg:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_bist_ca3: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_bist_ca2: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_bist_ca1: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)

        # Hebbian STDP for E→I connections (Pyr→PV, Pyr→OLM, Pyr→Bistratified).
        # PV interneurons must track local pyramidal assemblies; as new place
        # fields form, E→PV weights co-adapt so PV provides matched inhibition
        # (Kullmann & Lamsa 2007; Lu et al. 2007).  Conservative rate to avoid
        # destabilising the inhibitory loop.
        ei_stdp_cfg = STDPConfig(
            learning_rate=config.learning_rate * 0.05,  # 5% of E→E rate (hippocampal E→E is fast)
            a_plus=0.005,
            a_minus=0.0025,
            tau_plus=20.0,
            tau_minus=20.0,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        )
        self.ei_stdp: STDPStrategy = STDPStrategy(ei_stdp_cfg)

        # =====================================================================
        # GAP JUNCTIONS (Electrical Synapses) - Config Setup
        # =====================================================================
        # Override weights with trisynaptic circuit weights
        self._init_circuit_weights(device)

        # =====================================================================
        # INTER-LAYER AXONAL DELAYS (using CircularDelayBuffer utility)
        # =====================================================================
        # Create delay buffers for biological signal propagation within circuit
        # DG→CA3 delay: Mossy fiber transmission (~3ms biologically)
        # CA3→CA2 delay: Short proximity-based delay (~2ms biologically)
        # CA2→CA1 delay: Short proximity-based delay (~2ms biologically)
        # CA3→CA1 delay: Schaffer collateral transmission (~3ms biologically, direct bypass)

        # Initialize CircularDelayBuffer for each pathway
        dg_ca3_delay_steps = int(config.dg_to_ca3_delay_ms / config.dt_ms)
        ca3_ca3_delay_steps = int(config.ca3_to_ca3_delay_ms / config.dt_ms)
        ca3_ca2_delay_steps = int(config.ca3_to_ca2_delay_ms / config.dt_ms)
        ca2_ca1_delay_steps = int(config.ca2_to_ca1_delay_ms / config.dt_ms)
        ca2_ca3_delay_steps = int(config.ca2_to_ca3_delay_ms / config.dt_ms)
        ca3_ca1_delay_steps = int(config.ca3_to_ca1_delay_ms / config.dt_ms)

        # Store delay steps for conditional checks
        self._dg_ca3_delay_steps = dg_ca3_delay_steps
        self._ca3_ca3_delay_steps = ca3_ca3_delay_steps
        self._ca3_ca2_delay_steps = ca3_ca2_delay_steps
        self._ca2_ca1_delay_steps = ca2_ca1_delay_steps
        self._ca2_ca3_delay_steps = ca2_ca3_delay_steps
        self._ca3_ca1_delay_steps = ca3_ca1_delay_steps

        self._dg_ca3_buffer = CircularDelayBuffer(
            max_delay=dg_ca3_delay_steps,
            size=self.dg_size,
            device=device,
            dtype=torch.bool,
        )
        self._ca3_ca3_buffer = CircularDelayBuffer(
            max_delay=ca3_ca3_delay_steps,
            size=self.ca3_size,
            device=device,
            dtype=torch.bool,
        )
        self._ca3_ca2_buffer = CircularDelayBuffer(
            max_delay=ca3_ca2_delay_steps,
            size=self.ca3_size,
            device=device,
            dtype=torch.bool,
        )
        self._ca3_ca1_buffer = CircularDelayBuffer(
            max_delay=ca3_ca1_delay_steps,
            size=self.ca3_size,
            device=device,
            dtype=torch.bool,
        )
        self._ca2_ca1_buffer = CircularDelayBuffer(
            max_delay=ca2_ca1_delay_steps,
            size=self.ca2_size,
            device=device,
            dtype=torch.bool,
        )
        self._ca2_ca3_buffer = CircularDelayBuffer(
            max_delay=ca2_ca3_delay_steps,
            size=self.ca2_size,
            device=device,
            dtype=torch.bool,
        )

        # =====================================================================
        # DOPAMINE RECEPTOR (minimal 10% VTA projection)
        # =====================================================================
        # Hippocampus receives minimal DA innervation for novelty/salience modulation
        # Primarily affects CA1 output and CA3 consolidation
        # Biological: VTA DA enhances LTP in novelty-detecting neurons
        total_neurons = self.dg_size + self.ca3_size + self.ca2_size + self.ca1_size
        self.da_receptor = make_neuromodulator_receptor(
            NMReceptorType.DA_D1, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device,
            amplitude_scale=6.0,
        )
        # Per-subregion DA concentration buffers (0.5 = tonic baseline)
        self._da_concentration_dg = torch.full((self.dg_size,), 0.5, device=device)
        self._da_concentration_ca3 = torch.full((self.ca3_size,), 0.5, device=device)
        self._da_concentration_ca2 = torch.full((self.ca2_size,), 0.5, device=device)
        self._da_concentration_ca1 = torch.full((self.ca1_size,), 0.5, device=device)

        # =====================================================================
        # NOREPINEPHRINE RECEPTOR (LC projection for arousal/novelty)
        # =====================================================================
        # Hippocampus receives dense NE innervation from LC
        # NE modulates novelty detection and arousal-dependent memory formation
        # Î±1-adrenergic (Gq): arousal / novelty modulation, Ï„_rise=10 ms, Ï„_decay=150 ms.
        self.ne_receptor = make_neuromodulator_receptor(
            NMReceptorType.NE_ALPHA1, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device
        )
        # Per-subregion NE concentration buffers
        self._ne_concentration_dg = torch.zeros(self.dg_size, device=device)
        self._ne_concentration_ca3 = torch.zeros(self.ca3_size, device=device)
        self._ne_concentration_ca2 = torch.zeros(self.ca2_size, device=device)
        self._ne_concentration_ca1 = torch.zeros(self.ca1_size, device=device)

        # =====================================================================
        # ACETYLCHOLINE RECEPTOR (NB projection for encoding/retrieval)
        # =====================================================================
        # Hippocampus receives strong ACh innervation from nucleus basalis
        # ACh controls encoding vs retrieval modes (Hasselmo 1999):
        # - High ACh → encoding mode (suppress recurrence, enhance feedforward)
        # - Low ACh → retrieval mode (enable pattern completion, consolidation)
        # NB projects via nicotinic nAChR (ionotropic, fast: Ï„_rise=3 ms, Ï„_decay=15 ms).
        # Controls encoding vs retrieval mode (Hasselmo 1999).
        self.ach_receptor = make_neuromodulator_receptor(
            NMReceptorType.ACH_NICOTINIC, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device,
            amplitude_scale=1.3,
        )
        # Per-subregion ACh concentration buffers
        self._ach_concentration_dg = torch.zeros(self.dg_size, device=device)
        self._ach_concentration_ca3 = torch.zeros(self.ca3_size, device=device)
        self._ach_concentration_ca2 = torch.zeros(self.ca2_size, device=device)
        self._ach_concentration_ca1 = torch.zeros(self.ca1_size, device=device)

        # =====================================================================
        # SEPTAL ACETYLCHOLINE RECEPTOR (muscarinic M1, from medial septum)
        # =====================================================================
        # Septal ACh projects to CA1 via volume transmission, acting on M1-type
        # muscarinic receptors. Effects:
        #   - Enhances NMDA NR2B subunit conductance (slower, CaÂ²âº-permeable)
        #   - Modulates Schaffer collateral LTP threshold (encoding gate)
        # Kinetics: Ï„_rise=20ms, Ï„_decay=300ms (much slower than nicotinic NB ACh)
        # Muscarinic M1 (Gq → PLC/IP3): slow cascade, Ï„_rise=100 ms, Ï„_decay=1500 ms.
        # Phase-locked to theta; boosts NMDA NR2B insertion in CA1 at encoding peaks.
        self.ach_septal_receptor = make_neuromodulator_receptor(
            NMReceptorType.ACH_MUSCARINIC_M1, n_receptors=self.ca1_size, dt_ms=self.dt_ms, device=device
        )
        self._ach_septal_concentration_ca1 = torch.zeros(self.ca1_size, device=device)

        # =====================================================================
        # SEROTONIN RECEPTOR (5-HT2C on CA3 pyramidals – from DRN)
        # =====================================================================
        # Biology: 5-HT2C receptors on CA3 somata suppress recurrent NMDA
        # conductance, attenuating pattern completion and Schaffer collateral
        # LTP during aversive / high-serotonin states (Barnes & Sharp 1999).
        # Kinetics: tau_rise ~8 ms, tau_decay ~100 ms (SERT reuptake).
        # 5-HT2C (Gq → PLC): fast metabotropic, Ï„_rise=8 ms, Ï„_decay=100 ms.
        self.sht_receptor_ca3 = make_neuromodulator_receptor(
            NMReceptorType.SHT_2C, n_receptors=self.ca3_size, dt_ms=self.dt_ms, device=device
        )
        self._sht_concentration_ca3: torch.Tensor
        self.register_buffer("_sht_concentration_ca3", torch.zeros(self.ca3_size, device=device))

        # =====================================================================
        # LEARNING STRATEGY (Tag-and-Capture wrapping Three-Factor Learning)
        # =====================================================================
        # TagAndCaptureStrategy wraps ThreeFactorStrategy:
        # - compute_update: applies three-factor rule (eligibility Ã— DA Ã— lr)
        #   AND updates the tag matrix as a side effect (no extra calls needed).
        # - consolidate(): DA-gated capture that permanently strengthens tagged
        #   synapses when dopamine is elevated.
        # - tags tensor: readable for spontaneous replay prioritisation.
        tag_and_capture = TagAndCaptureStrategy(
            base_strategy=ThreeFactorStrategy(ThreeFactorConfig(
                learning_rate=0.001,  # Conservative rate for stable learning
                eligibility_tau=100.0,  # Eligibility trace decay (ms)
                modulator_tau=50.0,  # Modulator (dopamine) decay (ms)
            )),
            config=TagAndCaptureConfig(
                tag_decay=0.95,        # ~20 step tag lifetime at 1 ms dt
                tag_threshold=0.1,     # Minimum post-spike to create tag
                consolidation_lr_scale=0.5,  # Half of base lr for capture
                consolidation_da_threshold=0.1,
            ),
        )
        # Wrap with metaplasticity: per-synapse plasticity rate modulation.
        # Recently modified CA3 recurrent synapses become temporarily refractory;
        # frequently modified synapses consolidate and resist overwriting.
        # Abraham & Bear 1996 — prevents catastrophic forgetting of stored patterns.
        self._tag_and_capture_strategy = MetaplasticityStrategy(
            base_strategy=tag_and_capture,
            config=MetaplasticityConfig(
                tau_recovery_ms=5000.0,       # CaMKII refractory (~5s)
                depression_strength=5.0,       # |Δw|=0.01 → 0.05 rate drop
                tau_consolidation_ms=300000.0, # Protein synthesis (~5 min)
                consolidation_sensitivity=0.1, # Cumulative |Δw|=0.1 → 37% rate_rest
                rate_min=0.1,                  # 10% plasticity floor
            ),
        )
        # Eagerly allocate metaplasticity + tags buffers (CA3 recurrent: square matrix)
        self._tag_and_capture_strategy.setup(
            n_pre=self.ca3_size,
            n_post=self.ca3_size,
            device=torch.device(device),
        )

        # =====================================================================
        # CA3→CA1 SCHAFFER COLLATERAL STRATEGY (Tag-and-Capture + Metaplasticity)
        # =====================================================================
        # Classic NMDA-dependent LTP site (Bliss & Lømo 1973). Three-factor rule
        # with dopamine gating mirrors the D1/D5 receptor-dependent late-LTP at
        # CA3→CA1 (Frey & Morris 1997). Conservative learning rate because this
        # feedforward path should relay CA3 patterns without overwriting them.
        ca3_ca1_tac = TagAndCaptureStrategy(
            base_strategy=ThreeFactorStrategy(ThreeFactorConfig(
                learning_rate=0.0005,   # 0.5× CA3 recurrent (feedforward, more conservative)
                eligibility_tau=80.0,   # Slightly shorter trace (feedforward timing)
                modulator_tau=50.0,     # Same DA integration window
            )),
            config=TagAndCaptureConfig(
                tag_decay=0.95,
                tag_threshold=0.1,
                consolidation_lr_scale=0.3,  # Less capture than recurrent
                consolidation_da_threshold=0.1,
            ),
        )
        self._ca3_ca1_strategy = MetaplasticityStrategy(
            base_strategy=ca3_ca1_tac,
            config=MetaplasticityConfig(
                tau_recovery_ms=5000.0,
                depression_strength=5.0,
                tau_consolidation_ms=300000.0,
                consolidation_sensitivity=0.1,
                rate_min=0.1,
            ),
        )
        self._ca3_ca1_strategy.setup(
            n_pre=self.ca3_size,
            n_post=self.ca1_size,
            device=torch.device(device),
        )

        # =====================================================================
        # PATHWAY STDP STRATEGIES (replace inline Hebbian outer-product rules)
        # =====================================================================
        # Trace-based STDP gives timing-dependent LTP/LTD rather than
        # instantaneous co-firing, which is more biologically accurate
        # (Bi & Poo 1998) while preserving the DA-gated, encoding-mod-gated
        # learning dynamics of the original inline rules.

        _w_min = config.synaptic_scaling.w_min
        _w_max = config.synaptic_scaling.w_max

        # DG→CA3 mossy fiber: "detonator synapses" with rapid, powerful LTP.
        # 3× base learning rate for one-shot pattern indexing.
        self._dg_ca3_stdp = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate * 3.0,
            a_plus=0.01, a_minus=0.003,
            tau_plus=10.0, tau_minus=10.0,  # Short timing window (mossy synapses)
            w_min=_w_min, w_max=_w_max,
        ))

        # CA3→CA2: RGS14-suppressed plasticity (10× lower than CA3→CA3).
        self._ca3_ca2_stdp = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate * 0.1,
            a_plus=0.005, a_minus=0.002,
            tau_plus=20.0, tau_minus=20.0,
            w_min=_w_min, w_max=_w_max,
        ))

        # CA2→CA1: Standard Hebbian for temporal / social context binding.
        self._ca2_ca1_stdp = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate,
            a_plus=0.005, a_minus=0.002,
            tau_plus=20.0, tau_minus=20.0,
            w_min=_w_min, w_max=_w_max,
        ))

        # CA2→CA3: Modulatory back-projection (0.5× base rate).
        self._ca2_ca3_stdp = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate * 0.5,
            a_plus=0.005, a_minus=0.002,
            tau_plus=20.0, tau_minus=20.0,
            w_min=_w_min, w_max=_w_max,
        ))

        # Shared STDP for external EC→HPC inputs (all subfields).
        # Lazily registered per-synapse in _apply_plasticity().
        self._external_stdp_strategy = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate * 0.3,
            a_plus=0.005, a_minus=0.002,
            tau_plus=20.0, tau_minus=20.0,
            w_min=_w_min, w_max=_w_max,
        ))

        # Homeostatic iSTDP for external inhibitory inputs (e.g. MS GABA→CA3,
        # MS GABA→OLM).  Excitatory STDP is biologically inappropriate for
        # GABAergic synapses; iSTDP (Vogels et al. 2011) maintains E/I balance.
        self._external_istdp_strategy = InhibitorySTDPStrategy(InhibitorySTDPConfig(
            learning_rate=config.istdp_learning_rate * 0.3,
            tau_istdp=config.istdp_tau_ms,
            alpha=config.istdp_alpha,
            w_min=_w_min, w_max=_w_max,
        ))

        # Encoding modifier cached for _get_learning_kwargs (updated each step)
        self._current_encoding_mod: float = 0.0

        # Register pathway STDP strategies for existing internal connections.
        # Connections were already created by _init_circuit_weights() above;
        # _add_learning_strategy() links a strategy to an existing synapse_id.
        rn = self.region_name
        _dev = torch.device(device)

        self._sid_dg_ca3  = SynapseId(rn, HippocampusPopulation.DG,  rn, HippocampusPopulation.CA3, ReceptorType.AMPA)
        self._sid_ca3_ca2 = SynapseId(rn, HippocampusPopulation.CA3, rn, HippocampusPopulation.CA2, ReceptorType.AMPA)
        self._sid_ca3_ca1 = SynapseId(rn, HippocampusPopulation.CA3, rn, HippocampusPopulation.CA1, ReceptorType.AMPA)
        self._sid_ca2_ca1 = SynapseId(rn, HippocampusPopulation.CA2, rn, HippocampusPopulation.CA1, ReceptorType.AMPA)
        self._sid_ca2_ca3 = SynapseId(rn, HippocampusPopulation.CA2, rn, HippocampusPopulation.CA3, ReceptorType.AMPA)

        self._add_learning_strategy(self._sid_dg_ca3,  self._dg_ca3_stdp,  self.dg_size,  self.ca3_size, device=_dev)
        self._add_learning_strategy(self._sid_ca3_ca2, self._ca3_ca2_stdp, self.ca3_size, self.ca2_size, device=_dev)
        self._add_learning_strategy(self._sid_ca2_ca1, self._ca2_ca1_stdp, self.ca2_size, self.ca1_size, device=_dev)
        self._add_learning_strategy(self._sid_ca2_ca3, self._ca2_ca3_stdp, self.ca2_size, self.ca3_size, device=_dev)
        self._manually_stepped_stp = {
            self._sid_dg_ca3,
            self._sid_ca3_ca2,
            self._sid_ca3_ca1,
            self._sid_ca2_ca1,
        }

        # Ensure all tensors are on the correct device
        self.to(device)

    def _init_circuit_weights(self, device: Union[str, torch.device]) -> None:
        """Initialize internal circuit weights.

        Internal circuit weights (DG→CA3, CA3→CA1, etc.) are initialized here.
        """
        # =====================================================================
        # INTERNAL WEIGHTS
        # =====================================================================
        # DG → CA3: Random but less sparse (mossy fibers)
        # Biology: "Detonator synapses" with powerful one-shot transmission.
        # U=0.15: mossy fibers are facilitating but start from moderate release
        # probability (not near-zero like U=0.01 which causes chronic depletion
        # x·u ≈ 0.013 — well below the functional threshold).  tau_f=100ms keeps
        # facilitation within the DG burst window; tau_d=200ms sets recovery.
        self._add_internal_connection(
            source_population=HippocampusPopulation.DG,
            target_population=HippocampusPopulation.CA3,
            weights=WeightInitializer.sparse_random(
                n_input=self.dg_size,
                n_output=self.ca3_size,
                connectivity=0.15,
                weight_scale=0.003,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=100.0),
        )

        # CA3 → CA3 RECURRENT: Autoassociative memory weights
        # Learning: One-shot Hebbian with fast/slow traces and heterosynaptic LTD
        # Weight reduced from 0.0007: CA3 burst-silence SFA=5.89 driven by
        # recurrent excitation + Ca²⁺ dendritic spike positive feedback loop.
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3,
            target_population=HippocampusPopulation.CA3,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.ca3_size,
                n_output=self.ca3_size,
                connectivity=0.05,
                weight_scale=0.0004,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            # U 0.50→0.35, tau_d 500→400, tau_f 30→100: CA3 recurrents had
            # extreme depression killing autoassociative dynamics. Moderate
            # facilitation supports pattern completion.
            stp_config=STPConfig(U=0.35, tau_d=400.0, tau_f=100.0),
        )

        # =====================================================================
        # CA2 PATHWAYS: Social memory and temporal context
        # =====================================================================
        # CA3 → CA2: Weak plasticity (stability mechanism)
        # CA2 is resistant to CA3 pattern completion interference
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3,
            target_population=HippocampusPopulation.CA2,
            weights=WeightInitializer.sparse_random(
                n_input=self.ca3_size,
                n_output=self.ca2_size,
                connectivity=0.3,
                weight_scale=0.010,  # Raised 0.002→0.003→0.005→0.010: CA2 still at 1.3 Hz (target 3.0 Hz);
                                     # gain converged at 0.99. At CA3=1.0 Hz with conn=0.3 and STP,
                                     # effective drive per timestep is negligible. Need 2× more.
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.25, tau_d=200.0, tau_f=200.0),  # U 0.35→0.25, tau_d 300→200:
            # Less depression at low CA3 rates; x·u at 1 Hz is now ~0.22 vs 0.30 — maintains
            # more synaptic efficacy.
        )

        # CA2 → CA1: Output to decision layer
        # Provides temporal/social context to CA1 processing
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA2,
            target_population=HippocampusPopulation.CA1,
            weights=WeightInitializer.sparse_random(
                n_input=self.ca2_size,
                n_output=self.ca1_size,
                connectivity=0.2,
                weight_scale=0.008,  # Raised 0.0005→0.002→0.004→0.008: CA2→CA1 at 0.004 still gave
                                     # CA1=1.2 Hz. CA2 at 1.3 Hz provides negligible drive — need 2×.
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.25, tau_d=300.0, tau_f=200.0),  # U 0.35→0.25: less depletion at 1.3 Hz
        )

        # CA2 → CA3: Modulatory back-projection (temporal context stabilization)
        # Biology: CA2 pyramidal cells send weak back-projections to CA3 via local
        # axon collaterals.  These inputs modulate CA3 attractor dynamics by
        # providing temporal/social context without overriding pattern completion.
        # Kohara et al. 2014: CA2→CA3 projections are excitatory but sparse.
        # Unlike the strong CA3 recurrents, this pathway is facilitating — sustained
        # CA2 activity gradually primes CA3 representations.
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA2,
            target_population=HippocampusPopulation.CA3,
            weights=WeightInitializer.sparse_random(
                n_input=self.ca2_size,
                n_output=self.ca3_size,
                connectivity=0.10,
                weight_scale=0.003,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.20, tau_d=200.0, tau_f=300.0),
        )

        # CA3 → CA1: Feedforward (retrieved memory)
        # This is the DIRECT bypass pathway (Schaffer collaterals)
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3,
            target_population=HippocampusPopulation.CA1,
            weights=WeightInitializer.sparse_random(
                n_input=self.ca3_size,
                n_output=self.ca1_size,
                connectivity=0.30,  # Raised 0.15→0.20→0.25→0.30: maximize Schaffer collateral convergence
                weight_scale=0.010, # Raised 0.0015→0.003→0.005→0.010: CA1 at 1.2 Hz (target 3.0 Hz),
                                    # gain converged at 0.99. Previous 2× increases had no effect —
                                    # CA3 at 1.0 Hz with sparse connectivity needs very strong weights.
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.25, tau_d=300.0, tau_f=200.0),  # U 0.35→0.25, tau_d 500→300:
            # At CA3=2 Hz, U=0.5/tau_d=700 gives extreme depletion. Lower U + faster recovery
            # maintains ~60% efficacy, providing usable Schaffer collateral drive to CA1.
        )

        # =====================================================================
        # HIPPOCAMPAL INHIBITORY NETWORK SYNAPTIC CONNECTIONS
        # =====================================================================
        # All E→I, I→E, I→I weight matrices are registered here so they
        # participate in the standard STP, diagnostic, and learning-strategy
        # pipeline.  Weight values match those previously hard-coded inside
        # HippocampalInhibitoryNetwork (now a pure neuron container).
        # =====================================================================

        # Pre-compute Pyr→PV weight maximums (biologically-constrained conductance budget)
        # DG uses rate=0.003 (higher than CA3/CA1's 0.002) to reduce per-synapse w_max:
        # with 500 granule cells converging on 35 PV (14:1 ratio), even moderate
        # per-synapse weights accumulate massive excitatory drive.  Rate 0.003 gives
        # w_max ≈ 0.012 vs 0.019 at 0.002, targeting ~30-40 Hz PV firing.
        # CA2 uses rate=0.004 because its tiny population (75 pyr → 3 PV) produces
        # extreme per-synapse w_max; at 0.002 w_max=0.124 drove PV to 94 Hz.
        _pv_w_max_dg = 2.0 * 0.02317 / (0.003 * 5.0 * self.dg_size * 0.5)
        _pv_w_max_ca3 = 2.0 * 0.02317 / (0.002 * 5.0 * self.ca3_size * 0.5)
        _pv_w_max_ca2 = 2.0 * 0.02317 / (0.004 * 5.0 * self.ca2_size * 0.5)
        _pv_w_max_ca1 = 2.0 * 0.02317 / (0.002 * 5.0 * self.ca1_size * 0.5)

        for subregion, pyr_pop, pv_pop, olm_pop, bist_pop, pyr_size, pv_w_max_ei, pv_ie, pv_ie_wmax, pv_pv_wmax, pv_stp_tau_d, istdp_pv, istdp_olm, istdp_bist in [
            (
                "DG",
                HippocampusPopulation.DG,
                HippocampusPopulation.DG_INHIBITORY_PV,
                HippocampusPopulation.DG_INHIBITORY_OLM,
                HippocampusPopulation.DG_INHIBITORY_BISTRATIFIED,
                self.dg_size,
                _pv_w_max_dg,
                0.5,   # PV→Pyr connectivity  (0.3→0.5: DG PV at 65 Hz with STP depleted;
                       # inhibitory coverage 5× weaker than CA3/CA1 — raise to match)
                0.010, # PV→Pyr GABA_A w_max  (0.005→0.010: same rationale as connectivity)
                0.004, # PV→PV w_max  (0.0005→0.004: DG gets 2× more excitatory drive per PV
                       # neuron than CA3 due to larger granule cell population; standard 0.0005
                       # depletes under runaway, leaving PV→PV effectively silent)
                100.0, # PV→ STP τd (ms)  — DG PV fires at high rates due to massive
                       # granule cell convergence (500→35); standard 250ms causes
                       # catastrophic STP depletion (eff=0.054) above ~30 Hz.
                       # PV basket perisomatic synapses recover fast (Kraushaar & Jonas
                       # 2000); 100ms keeps eff>0.10 up to ~40 Hz.
                self.istdp_pv_dg,
                self.istdp_olm_dg,
                self.istdp_bist_dg,
            ),
            (
                "CA3",
                HippocampusPopulation.CA3,
                HippocampusPopulation.CA3_INHIBITORY_PV,
                HippocampusPopulation.CA3_INHIBITORY_OLM,
                HippocampusPopulation.CA3_INHIBITORY_BISTRATIFIED,
                self.ca3_size,
                _pv_w_max_ca3,
                0.7,   # PV→Pyr connectivity (strong: controls runaway recurrence)
                0.015, # PV→Pyr GABA_A w_max
                0.0005, # PV→PV w_max
                250.0, # PV→ STP τd (standard)
                self.istdp_pv_ca3,
                self.istdp_olm_ca3,
                self.istdp_bist_ca3,
            ),
            (
                "CA2",
                HippocampusPopulation.CA2,
                HippocampusPopulation.CA2_INHIBITORY_PV,
                HippocampusPopulation.CA2_INHIBITORY_OLM,
                HippocampusPopulation.CA2_INHIBITORY_BISTRATIFIED,
                self.ca2_size,
                _pv_w_max_ca2,
                0.5,   # PV→Pyr connectivity
                0.012, # PV→Pyr GABA_A w_max
                0.004, # PV→PV w_max  (raised 0.0005→0.004: CA2 PV at 91 Hz; PV→PV
                       # mutual inhibition was 8× weaker than DG, leaving CA2 PV
                       # uncontrolled under high excitatory drive)
                250.0, # PV→ STP τd (standard)
                self.istdp_pv_ca2,
                self.istdp_olm_ca2,
                self.istdp_bist_ca2,
            ),
            (
                "CA1",
                HippocampusPopulation.CA1,
                HippocampusPopulation.CA1_INHIBITORY_PV,
                HippocampusPopulation.CA1_INHIBITORY_OLM,
                HippocampusPopulation.CA1_INHIBITORY_BISTRATIFIED,
                self.ca1_size,
                _pv_w_max_ca1,
                0.5,   # PV→Pyr connectivity (supports theta modulations)
                0.015, # PV→Pyr GABA_A w_max
                0.0005, # PV→PV w_max
                250.0, # PV→ STP τd (standard)
                self.istdp_pv_ca1,
                self.istdp_olm_ca1,
                self.istdp_bist_ca1,
            ),
        ]:
            # Retrieve interneuron sizes from the corresponding inhibitory network
            inhib_net = getattr(self, f"{subregion.lower()}_inhibitory")
            n_pv = inhib_net.n_pv
            n_olm = inhib_net.n_olm
            n_bist = inhib_net.n_bistratified

            # ------------------------------------------------------------------
            # E → I  (Pyramidal → Interneurons)
            # ------------------------------------------------------------------
            # Pyr→PV: Hebbian STDP so PV tracks evolving place cell assemblies.
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=pyr_size,
                    n_output=n_pv,
                    connectivity=0.5,
                    w_min=0.0,
                    w_max=pv_w_max_ei,
                    device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig(U=0.55, tau_d=500.0, tau_f=15.0),
                learning_strategy=self.ei_stdp,
            )
            # Pyr→OLM: Hebbian STDP so OLM cells learn dendritic gating.
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=olm_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=pyr_size,
                    n_output=n_olm,
                    connectivity=0.3,
                    w_min=0.0,
                    w_max=0.06,
                    device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
                learning_strategy=self.ei_stdp,
            )
            # Pyr→Bistratified: Hebbian STDP for feedforward inhibition matching.
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=bist_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=pyr_size,
                    n_output=n_bist,
                    connectivity=0.35,
                    w_min=0.0,
                    w_max=0.025,
                    device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig(U=0.55, tau_d=500.0, tau_f=15.0),
                learning_strategy=self.ei_stdp,
            )

            # ------------------------------------------------------------------
            # I → E  (Interneurons → Pyramidal)
            # ------------------------------------------------------------------
            # PV→Pyr GABA_A: iSTDP tunes perisomatic inhibition strength.
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=n_pv,
                    n_output=pyr_size,
                    connectivity=pv_ie,
                    w_min=0.0,
                    w_max=pv_ie_wmax,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.25, tau_d=pv_stp_tau_d, tau_f=15.0),
                learning_strategy=istdp_pv,
            )
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=n_pv,
                    n_output=pyr_size,
                    connectivity=pv_ie,
                    w_min=0.0,
                    w_max=pv_ie_wmax * 0.15,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_B,
                stp_config=None,  # GABA_B recruits via spill-over; vesicular STP does not apply
            )
            # OLM→Pyr GABA_A: iSTDP tunes dendritic inhibition strength.
            self._add_internal_connection(
                source_population=olm_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=n_olm,
                    n_output=pyr_size,
                    connectivity=0.5,
                    w_min=0.0,
                    w_max=0.001,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
                learning_strategy=istdp_olm,
            )
            self._add_internal_connection(
                source_population=bist_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=n_bist,
                    n_output=pyr_size,
                    connectivity=0.55,
                    w_min=0.0,
                    w_max=0.002,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.25, tau_d=250.0, tau_f=15.0),
                learning_strategy=istdp_bist,
            )

            # ------------------------------------------------------------------
            # I → I  (Lateral inhibition within inhibitory network)
            # ------------------------------------------------------------------
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_uniform_no_autapses(
                    n_input=n_pv,
                    n_output=n_pv,
                    connectivity=0.6,  # Raised 0.3→0.6: with as few as 3 PV neurons
                                       # (CA2), 0.3 on a 3×3 no-autapse matrix gave
                                       # ~12% chance of all-zero weights; 0.6 brings
                                       # P(all-zero) < 0.5%.
                    w_min=0.0,
                    w_max=pv_pv_wmax,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.25, tau_d=pv_stp_tau_d, tau_f=15.0),
            )
            self._add_internal_connection(
                source_population=olm_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=n_olm,
                    n_output=n_pv,
                    connectivity=0.5,  # Raised 0.2→0.5: with as few as 3 OLM and 3 PV neurons
                                       # (CA2 at 15% interneuron fraction), 0.2 gave a ~13% chance
                                       # of a fully-zero weight matrix; 0.5 keeps expected density
                                       # ≥1 connection even for the smallest plausible sub-circuit.
                    w_min=0.0,
                    w_max=0.0004,
                    device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
            )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _run_hippocampal_inhibitory(
        self,
        inhib_net: HippocampalInhibitoryNetwork,
        prev_pyr_spikes: torch.Tensor,
        prev_pv_spikes: torch.Tensor,
        prev_olm_spikes: torch.Tensor,
        prev_bist_spikes: torch.Tensor,
        pyr_pop: str,
        pv_pop: str,
        olm_pop: str,
        bist_pop: str,
        synaptic_inputs: SynapticInput,
    ) -> Dict[str, torch.Tensor]:
        """Compute one inhibitory network step using registered STP-weighted connections.

        External feedforward excitation to PV and septal GABA inhibition to OLM are extracted
        from ``synaptic_inputs`` and summed with local drive.

        Args:
            inhib_net: The inhibitory network to run (neuron container + gap junctions).
            prev_pyr_spikes: Previous-step pyramidal spikes [pyr_size] for E→I drive.
            prev_pv_spikes: Previous-step PV spikes [n_pv] for I→I and I→E.
            prev_olm_spikes: Previous-step OLM spikes [n_olm] for I→I and I→E.
            prev_bist_spikes: Previous-step bistratified spikes [n_bist] for I→E.
            pyr_pop / pv_pop / olm_pop / bist_pop: Population names registered in this region.
            synaptic_inputs: Full inter-region synaptic inputs (from ``_step``).
                External inputs targeting PV and OLM populations are extracted and integrated
                (feedforward inhibition + septal theta).

        Returns:
            dict with keys: perisomatic_gaba_a, perisomatic_gaba_b, dendritic, olm_dendritic,
                            pv_spikes, olm_spikes, bistratified_spikes
        """
        rn = self.region_name

        def _syn(src_pop: str, tgt_pop: str, rxn: ReceptorType) -> SynapseId:
            return SynapseId(rn, src_pop, rn, tgt_pop, receptor_type=rxn)

        # ------------------------------------------------------------------
        # E → I  (Pyr → interneurons, STP-modulated)
        # ------------------------------------------------------------------
        syn_pyr_pv = _syn(pyr_pop, pv_pop, ReceptorType.AMPA)
        syn_pyr_olm = _syn(pyr_pop, olm_pop, ReceptorType.AMPA)
        syn_pyr_bist = _syn(pyr_pop, bist_pop, ReceptorType.AMPA)

        pv_g_exc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={syn_pyr_pv: prev_pyr_spikes},
            n_neurons=inhib_net.n_pv,
        ).g_ampa
        olm_g_exc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={syn_pyr_olm: prev_pyr_spikes},
            n_neurons=inhib_net.n_olm,
        ).g_ampa
        bist_g_exc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={syn_pyr_bist: prev_pyr_spikes},
            n_neurons=inhib_net.n_bistratified,
        ).g_ampa

        # ------------------------------------------------------------------
        # External feedforward excitation → PV  (e.g. EC_II → DG_PV)
        # ------------------------------------------------------------------
        # Biology: perforant path axons bifurcate onto both principal cells and
        # local PV basket cells.  PV fires ~1-2 ms before granule/pyramidal
        # cells respond, enforcing a narrow excitability window (feedforward
        # inhibition).  This sharpens pattern separation and prevents runaway.
        ext_pv = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, n_neurons=inhib_net.n_pv,
            filter_by_target_population=pv_pop,
        )
        pv_g_exc = pv_g_exc + ext_pv.g_ampa

        # ------------------------------------------------------------------
        # I → I  (PV→PV lateral + OLM→PV, from prev-step inhibitory spikes)
        # ------------------------------------------------------------------
        syn_pv_pv = _syn(pv_pop, pv_pop, ReceptorType.GABA_A)
        syn_olm_pv = _syn(olm_pop, pv_pop, ReceptorType.GABA_A)

        pv_g_inh = (
            self._integrate_synaptic_inputs_at_dendrites(
                synaptic_inputs={syn_pv_pv: prev_pv_spikes},
                n_neurons=inhib_net.n_pv,
            ).g_gaba_a
            + self._integrate_synaptic_inputs_at_dendrites(
                synaptic_inputs={syn_olm_pv: prev_olm_spikes},
                n_neurons=inhib_net.n_pv,
            ).g_gaba_a
        )

        # ------------------------------------------------------------------
        # Septal GABA → OLM  (external inter-region input for theta phase-locking)
        # ------------------------------------------------------------------
        # Biology: MS GABAergic neurons innervate hippocampal OLM cells at theta
        # peaks, causing rebound bursting at theta troughs.  This creates emergent
        # encoding/retrieval separation gated by the septal theta pacemaker.
        ext_olm = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, n_neurons=inhib_net.n_olm,
            filter_by_target_population=olm_pop,
        )
        olm_g_inh = ext_olm.g_gaba_a
        bist_g_inh = torch.zeros(inhib_net.n_bistratified, device=self.device)

        # ------------------------------------------------------------------
        # Run interneurons
        # ------------------------------------------------------------------
        inhib_out = inhib_net.forward(
            pv_g_exc=pv_g_exc,
            pv_g_inh=pv_g_inh,
            olm_g_exc=olm_g_exc,
            olm_g_inh=olm_g_inh,
            bistratified_g_exc=bist_g_exc,
            bistratified_g_inh=bist_g_inh,
        )
        pv_spikes = inhib_out["pv_spikes"]
        olm_spikes = inhib_out["olm_spikes"]
        bist_spikes = inhib_out["bistratified_spikes"]

        # ------------------------------------------------------------------
        # I → E  (prev-step interneurons → pyramidal, STP-modulated)
        # ------------------------------------------------------------------
        syn_pv_pyr_a = _syn(pv_pop, pyr_pop, ReceptorType.GABA_A)
        syn_pv_pyr_b = _syn(pv_pop, pyr_pop, ReceptorType.GABA_B)
        syn_olm_pyr = _syn(olm_pop, pyr_pop, ReceptorType.GABA_A)
        syn_bist_pyr = _syn(bist_pop, pyr_pop, ReceptorType.GABA_A)

        perisomatic_gaba_a = self._integrate_single_synaptic_input(syn_pv_pyr_a, prev_pv_spikes).g_gaba_a
        perisomatic_gaba_b = self._integrate_single_synaptic_input(syn_pv_pyr_b, prev_pv_spikes).g_gaba_b
        olm_dendritic = self._integrate_single_synaptic_input(syn_olm_pyr, prev_olm_spikes).g_gaba_a
        bist_dendritic = self._integrate_single_synaptic_input(syn_bist_pyr, prev_bist_spikes).g_gaba_a

        dendritic = olm_dendritic + bist_dendritic

        return {
            "perisomatic_gaba_a": perisomatic_gaba_a,
            "perisomatic_gaba_b": perisomatic_gaba_b,
            "dendritic": dendritic,
            "olm_dendritic": olm_dendritic,
            "pv_spikes": pv_spikes,
            "olm_spikes": olm_spikes,
            "bistratified_spikes": bist_spikes,
        }

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Process input spikes through the DG→CA3→CA2→CA1 trisynaptic circuit.

        The full forward pass is delegated to eight focused phase methods:

        1. :meth:`_process_neuromodulators` – update concentration buffers
        2. :meth:`_extract_circuit_inputs`  – integrate synaptic inputs + compute gating
        3. :meth:`_step_dg`                 – pattern separation (Dentate Gyrus)
        4. :meth:`_step_ca3`                – pattern completion + recurrence (CA3)
        5. :meth:`_step_ca2`                – temporal / social context (CA2)
        6. :meth:`_step_ca1`                – memory output layer (CA1)
        7. :meth:`_update_match_mismatch`   – familiarity / prediction-error signals
        8. :meth:`_apply_plasticity`        – Hebbian + three-factor weight updates
        9. :meth:`_update_spike_buffers`    – advance all delay buffers

        **Sharp-wave ripple (SWR) replay path:** When CA3 population rate exceeds
        ``config.ripple_threshold``, a replay window opens (``ripple_detected=True``,
        length ``config.ripple_duration_max_ms``).  The CA3 attractor pattern captured
        at onset is re-injected each timestep (``config.ripple_replay_injection``),
        sustaining CA1 activation via boosted Schaffer collaterals
        (``config.ripple_boost_factor``).  The replayed CA1 spikes propagate
        downstream through Subiculum → EC_V → cortex, implementing offline
        memory consolidation (Buzsaki 1989; Wilson & McNaughton 1994).
        """
        self._process_neuromodulators(neuromodulator_inputs)

        dg_input, ca3_input, ca2_input, ca1_input, ffi_factor, encoding_mod, retrieval_mod = (
            self._extract_circuit_inputs(synaptic_inputs)
        )

        dg_spikes,  dg_inhib_output  = self._step_dg(dg_input, ffi_factor, synaptic_inputs)
        ca3_spikes, ca3_inhib_output = self._step_ca3(ca3_input, synaptic_inputs)
        ca2_spikes, ca2_inhib_output = self._step_ca2(ca2_input, synaptic_inputs)
        ca1_spikes, ca1_inhib_output, ca1_basal_g_exc, ca1_apical_g_exc = (
            self._step_ca1(ca1_input, ffi_factor, encoding_mod, retrieval_mod, synaptic_inputs)
        )

        self._update_match_mismatch(ca1_basal_g_exc, ca1_apical_g_exc, retrieval_mod)

        if not self.config.learning_disabled:
            self._apply_plasticity(
                synaptic_inputs, dg_spikes, ca3_spikes, ca2_spikes, ca1_spikes, encoding_mod
            )

        region_outputs: RegionOutput = {
            HippocampusPopulation.DG:  dg_spikes,
            HippocampusPopulation.CA3: ca3_spikes,
            HippocampusPopulation.CA2: ca2_spikes,
            HippocampusPopulation.CA1: ca1_spikes,
            HippocampusPopulation.DG_INHIBITORY_PV:            dg_inhib_output["pv_spikes"],
            HippocampusPopulation.DG_INHIBITORY_OLM:           dg_inhib_output["olm_spikes"],
            HippocampusPopulation.DG_INHIBITORY_BISTRATIFIED:  dg_inhib_output["bistratified_spikes"],
            HippocampusPopulation.CA3_INHIBITORY_PV:           ca3_inhib_output["pv_spikes"],
            HippocampusPopulation.CA3_INHIBITORY_OLM:          ca3_inhib_output["olm_spikes"],
            HippocampusPopulation.CA3_INHIBITORY_BISTRATIFIED: ca3_inhib_output["bistratified_spikes"],
            HippocampusPopulation.CA2_INHIBITORY_PV:           ca2_inhib_output["pv_spikes"],
            HippocampusPopulation.CA2_INHIBITORY_OLM:          ca2_inhib_output["olm_spikes"],
            HippocampusPopulation.CA2_INHIBITORY_BISTRATIFIED: ca2_inhib_output["bistratified_spikes"],
            HippocampusPopulation.CA1_INHIBITORY_PV:           ca1_inhib_output["pv_spikes"],
            HippocampusPopulation.CA1_INHIBITORY_OLM:          ca1_inhib_output["olm_spikes"],
            HippocampusPopulation.CA1_INHIBITORY_BISTRATIFIED: ca1_inhib_output["bistratified_spikes"],
        }

        self._apply_all_population_homeostasis(region_outputs)
        self._update_spike_buffers(dg_spikes, ca3_spikes, ca2_spikes, ca1_spikes)

        return region_outputs

    # =========================================================================
    # PHASE METHODS (decomposed from _step)
    # =========================================================================

    def _process_neuromodulators(self, neuromodulator_inputs: NeuromodulatorInput) -> None:
        """Update all neuromodulator concentration buffers from incoming spikes."""
        if self.config.neuromodulation_disabled:
            return

        # DOPAMINE (from VTA mesolimbic pathway)
        # Hippocampus receives ~10% DA innervation for novelty / salience gating.
        vta_da_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.DA_MESOLIMBIC)
        da_concentration_full = self.da_receptor.update(vta_da_spikes)
        # Phasic DA (spikes) adds to tonic baseline (0.5).
        self._da_concentration_dg  = 0.5 + da_concentration_full[: self.dg_size] * 1.0
        self._da_concentration_ca3 = 0.5 + da_concentration_full[self.dg_size : self.dg_size + self.ca3_size] * 1.0
        self._da_concentration_ca2 = 0.5 + da_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size] * 1.0
        self._da_concentration_ca1 = 0.5 + da_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :] * 1.0

        # NOREPINEPHRINE (from Locus Coeruleus) – arousal / novelty gain
        lc_ne_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.NE)
        ne_concentration_full = self.ne_receptor.update(lc_ne_spikes)
        self._ne_concentration_dg  = ne_concentration_full[: self.dg_size]
        self._ne_concentration_ca3 = ne_concentration_full[self.dg_size : self.dg_size + self.ca3_size]
        self._ne_concentration_ca2 = ne_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size]
        self._ne_concentration_ca1 = ne_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :]

        # ACETYLCHOLINE – nicotinic nAChR (from Nucleus Basalis)
        # High ACh → encoding mode (suppresses CA3 recurrence)
        nb_ach_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.ACH)
        ach_concentration_full = self.ach_receptor.update(nb_ach_spikes)
        self._ach_concentration_dg  = ach_concentration_full[: self.dg_size]
        self._ach_concentration_ca3 = ach_concentration_full[self.dg_size : self.dg_size + self.ca3_size]
        self._ach_concentration_ca2 = ach_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size]
        self._ach_concentration_ca1 = ach_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :]

        # ACETYLCHOLINE – muscarinic M1 mAChR (from Medial Septum)
        # Slow kinetics (Ï„_decay=300 ms) gate NMDA NR2B insertion in CA1.
        septal_ach_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.ACH_SEPTAL)
        self._ach_septal_concentration_ca1 = self.ach_septal_receptor.update(septal_ach_spikes)

        # SEROTONIN – 5-HT2C (from Dorsal Raphe Nucleus, targets CA3)
        # High 5-HT: suppresses CA3 recurrent NMDA and Schaffer LTP.
        sht_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.SHT)
        self._sht_concentration_ca3 = self.sht_receptor_ca3.update(sht_spikes)

    def _extract_circuit_inputs(
        self, synaptic_inputs: SynapticInput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        """Integrate all multi-source synaptic inputs and compute gating signals.

        Returns:
            ``(dg_input, ca3_input, ca2_input, ca1_input,
               ffi_factor, encoding_mod, retrieval_mod)``
        """
        dg_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, n_neurons=self.dg_size,
            filter_by_target_population=HippocampusPopulation.DG,
        ).g_ampa

        # EC_II perforant path (SLM) targets the apical compartment of CA3 pyramidal cells.
        # _step_ca3 routes this to g_ampa_apical; DG mossy fibers go to basal.
        ca3_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, n_neurons=self.ca3_size,
            filter_by_target_population=HippocampusPopulation.CA3,
        ).g_ampa

        ca2_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, n_neurons=self.ca2_size,
            filter_by_target_population=HippocampusPopulation.CA2,
        ).g_ampa

        ca1_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, n_neurons=self.ca1_size,
            filter_by_target_population=HippocampusPopulation.CA1,
        ).g_ampa

        # Encoding/retrieval modulation: MS ACH theta pacemaker (primary) or OLM fallback.
        #
        # When the medial septum is wired (MS ACH→CA1 AMPA synapse present), derive
        # encoding_mod from a fast (τ=30ms) exponential trace of MS ACH spike activity.
        # High MS ACH firing → encoding phase dominant; quiet phase → retrieval dominant.
        # When MS is absent, fall back to the OLM-based _prev_encoding_mod (causal,
        # computed at the end of the previous _step_ca1 call).
        _ms_ach_spike_rate: float = -1.0  # sentinel < 0 means "no MS ACH synapse found"
        for _sid in synaptic_inputs:
            if (
                _sid.source_population == MedialSeptumPopulation.ACH
                and _sid.target_population == HippocampusPopulation.CA1
                and _sid.receptor_type == ReceptorType.AMPA
            ):
                _ms_ach_spike_rate = synaptic_inputs[_sid].mean().item()
                break

        if _ms_ach_spike_rate >= 0.0:
            # MS ACH drives encoding phase: update fast exponential trace (τ=30ms)
            # and derive encoding_mod via sigmoid with 50% crossover at trace≈0.08.
            # At tonic 8 Hz (spike_frac≈0.008/ms): trace_ss ≈ 0.24 → encoding_mod ≈ 0.96.
            # During theta trough (MS ACH silent): trace decays → encoding_mod → 0.17.
            _ms_decay = math.exp(-self.dt_ms / 30.0)
            self._ms_theta_trace = self._ms_theta_trace * _ms_decay + _ms_ach_spike_rate
            encoding_mod = 1.0 / (1.0 + math.exp(-(self._ms_theta_trace - 0.08) * 20.0))
            retrieval_mod = 1.0 - encoding_mod
        else:
            # OLM-based fallback (computed at end of previous _step_ca1; causal)
            encoding_mod = self._prev_encoding_mod
            retrieval_mod = 1.0 - self._prev_encoding_mod

        # Stimulus-onset feedforward inhibition (FFI)
        ffi = self.stimulus_gating.compute(dg_input, return_tensor=False)
        raw_ffi   = ffi.item() if hasattr(ffi, "item") else float(ffi)
        ffi_strength = min(1.0, raw_ffi / self.stimulus_gating.max_inhibition)
        ffi_factor   = 1.0 - ffi_strength * self.config.ffi_strength

        return dg_input, ca3_input, ca2_input, ca1_input, ffi_factor, encoding_mod, retrieval_mod

    def _step_dg(
        self,
        dg_input: torch.Tensor,
        ffi_factor: float,
        synaptic_inputs: SynapticInput,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process one timestep of the Dentate Gyrus (pattern separation).

        Returns:
            ``(dg_spikes, dg_inhib_output)``
        """
        # Apply FFI gating then clamp to non-negative conductance
        dg_g_exc = F.relu(dg_input * ffi_factor)

        dg_inhib_output = self._run_hippocampal_inhibitory(
            inhib_net=self.dg_inhibitory,
            prev_pyr_spikes=self._prev_spikes(HippocampusPopulation.DG),
            prev_pv_spikes=self._prev_spikes(HippocampusPopulation.DG_INHIBITORY_PV),
            prev_olm_spikes=self._prev_spikes(HippocampusPopulation.DG_INHIBITORY_OLM),
            prev_bist_spikes=self._prev_spikes(HippocampusPopulation.DG_INHIBITORY_BISTRATIFIED),
            pyr_pop=HippocampusPopulation.DG,
            pv_pop=HippocampusPopulation.DG_INHIBITORY_PV,
            olm_pop=HippocampusPopulation.DG_INHIBITORY_OLM,
            bist_pop=HippocampusPopulation.DG_INHIBITORY_BISTRATIFIED,
            synaptic_inputs=synaptic_inputs,
        )
        dg_perisomatic_inhib = dg_inhib_output["perisomatic_gaba_a"]

        dg_g_ampa, dg_g_nmda = split_excitatory_conductance(dg_g_exc, nmda_ratio=0.05)

        dg_spikes, _ = self.dg_neurons.forward(
            g_ampa_input=ConductanceTensor(dg_g_ampa),
            g_nmda_input=ConductanceTensor(dg_g_nmda),
            g_gaba_a_input=ConductanceTensor(dg_perisomatic_inhib),
            g_gaba_b_input=None,
        )
        return dg_spikes, dg_inhib_output

    def _step_ca3(
        self,
        ca3_input: torch.Tensor,
        synaptic_inputs: SynapticInput,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process one timestep of CA3 (pattern completion + recurrence).

        Reads ``_dg_ca3_buffer``, ``_ca3_ca3_buffer``, and ``_ca2_ca3_buffer`` internally.
        Sets ``self.ripple_detected``, ``self.ripple_duration_steps``, and ``self._swr_ca3_pattern``.

        Returns:
            ``(ca3_spikes, ca3_inhib_output)``
        """
        config = self.config

        dg_ca3_synapse = SynapseId(
            self.region_name, HippocampusPopulation.DG,
            self.region_name, HippocampusPopulation.CA3,
            receptor_type=ReceptorType.AMPA,
        )
        ca3_ca3_synapse = SynapseId(
            self.region_name, HippocampusPopulation.CA3,
            self.region_name, HippocampusPopulation.CA3,
            receptor_type=ReceptorType.AMPA,
        )
        ca2_ca3_synapse = SynapseId(
            self.region_name, HippocampusPopulation.CA2,
            self.region_name, HippocampusPopulation.CA3,
            receptor_type=ReceptorType.AMPA,
        )
        dg_ca3_weights = self.get_synaptic_weights(dg_ca3_synapse)
        ca3_ca3_weights = self.get_synaptic_weights(ca3_ca3_synapse)
        ca2_ca3_weights = self.get_synaptic_weights(ca2_ca3_synapse)

        # DG mossy fiber input (with axonal delay + STP facilitation) → basal dendrites
        # Biology: mossy fibers synapse on proximal apical/basal (stratum lucidum) of CA3.
        dg_spikes_delayed = self._dg_ca3_buffer.read(self._dg_ca3_delay_steps).float()
        stp_efficacy = self.stp_modules[dg_ca3_synapse].forward(dg_spikes_delayed)
        ca3_from_dg = torch.matmul(dg_ca3_weights, stp_efficacy * dg_spikes_delayed)

        # CA3 recurrent input (ACh-gated, delayed) → basal dendrites
        # High ACh (encoding) suppresses recurrence; low ACh (retrieval) releases it.
        ach_level = self._ach_concentration_ca3.mean().item()
        ach_recurrent_modulation = compute_ach_recurrent_suppression(ach_level)
        ca3_delayed   = self._ca3_ca3_buffer.read(self._ca3_ca3_buffer.max_delay).float()
        ca3_rec_raw   = torch.matmul(ca3_ca3_weights, ca3_delayed)
        ca3_rec       = ca3_rec_raw * ach_recurrent_modulation

        # CA2 → CA3 modulatory back-projection (delayed + STP facilitating)
        # Biology: CA2 provides temporal/social context via weak back-projections.
        # Kohara et al. 2014: these inputs modulate CA3 attractor dynamics without
        # overriding pattern completion.
        ca2_spikes_for_ca3 = self._ca2_ca3_buffer.read(self._ca2_ca3_delay_steps).float()
        stp_efficacy_ca2 = self.stp_modules[ca2_ca3_synapse].forward(ca2_spikes_for_ca3)
        ca3_from_ca2 = torch.matmul(ca2_ca3_weights, stp_efficacy_ca2 * ca2_spikes_for_ca3)

        # NE gain modulation (β-adrenergic: high NE → more responsive)
        ne_level = self._ne_concentration_ca3.mean().item()
        ne_gain  = compute_ne_gain(ne_level)
        # Basal: DG mossy fibers + CA3 recurrents + CA2 back-projection
        ca3_basal_input = (ca3_from_dg + ca3_rec + ca3_from_ca2) * ne_gain
        # Apical: EC_II perforant path (SLM) — same pattern as CA1's EC_III→apical routing
        ca3_apical_input = ca3_input * ne_gain

        # SWR replay injection into basal (sustains attractor across ripple window)
        if self.ripple_detected and self._swr_ca3_pattern is not None:
            ca3_basal_input = (
                ca3_basal_input
                + self._swr_ca3_pattern.float() * config.ripple_replay_injection
            )

        # CA3 inhibitory network (PV, OLM, Bistratified; OLM phase-locks to septal theta)
        ca3_inhib_output = self._run_hippocampal_inhibitory(
            inhib_net=self.ca3_inhibitory,
            prev_pyr_spikes=self._prev_spikes(HippocampusPopulation.CA3),
            prev_pv_spikes=self._prev_spikes(HippocampusPopulation.CA3_INHIBITORY_PV),
            prev_olm_spikes=self._prev_spikes(HippocampusPopulation.CA3_INHIBITORY_OLM),
            prev_bist_spikes=self._prev_spikes(HippocampusPopulation.CA3_INHIBITORY_BISTRATIFIED),
            pyr_pop=HippocampusPopulation.CA3,
            pv_pop=HippocampusPopulation.CA3_INHIBITORY_PV,
            olm_pop=HippocampusPopulation.CA3_INHIBITORY_OLM,
            bist_pop=HippocampusPopulation.CA3_INHIBITORY_BISTRATIFIED,
            synaptic_inputs=synaptic_inputs,
        )
        ca3_perisomatic_inhib = ca3_inhib_output["perisomatic_gaba_a"]

        ca3_g_basal = F.relu(ca3_basal_input)
        ca3_g_apical = F.relu(ca3_apical_input)

        # Conductance validation: biological range [0, 5] relative to g_L = 1
        if ca3_g_basal.max() > 10.0:
            print(f"⚠️  WARNING: CA3 g_basal={ca3_g_basal.max():.2f} exceeds biological range [0, 5]!")
            print(f"  ca3_from_dg={ca3_from_dg.mean():.4f}, ca3_rec={ca3_rec.mean():.4f}")
            ca3_g_basal = torch.clamp(ca3_g_basal, 0.0, 5.0)

        # Burst-risk self-inhibition: Only apply when V_mem > 80% of threshold
        v_threshold  = self.ca3_neurons.v_threshold.mean().item()
        v_normalized = torch.clamp(self.ca3_neurons.V_soma / v_threshold, 0.0, 1.0)
        burst_risk   = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)

        ca3_g_inh = (
            F.relu(ca3_perisomatic_inhib)
            + config.tonic_inhibition
            + burst_risk * 0.1
        )

        ca3_g_ampa_basal, ca3_g_nmda_basal = split_excitatory_conductance(ca3_g_basal, nmda_ratio=0.2)
        ca3_g_ampa_apical, ca3_g_nmda_apical = split_excitatory_conductance(ca3_g_apical, nmda_ratio=0.2)

        # 5-HT2C: high serotonin suppresses NMDA NR2B-mediated currents on CA3 spines
        sht_level = self._sht_concentration_ca3.mean().item()
        if sht_level > 0.0:
            ca3_g_nmda_basal  = ca3_g_nmda_basal  * (1.0 - 0.5 * sht_level)
            ca3_g_nmda_apical = ca3_g_nmda_apical * (1.0 - 0.5 * sht_level)

        ca3_spikes, _, _ = self.ca3_neurons.forward(
            g_ampa_basal=ConductanceTensor(ca3_g_ampa_basal),
            g_nmda_basal=ConductanceTensor(ca3_g_nmda_basal),
            g_gaba_a_basal=ConductanceTensor(ca3_g_inh),
            g_gaba_b_basal=ConductanceTensor(ca3_inhib_output["perisomatic_gaba_b"]),
            g_ampa_apical=ConductanceTensor(ca3_g_ampa_apical),
            g_nmda_apical=ConductanceTensor(ca3_g_nmda_apical),
            g_gaba_a_apical=None,
        )

        # SWR state: duration-gated detection (replaces single-timestep bool).
        # Biology: SWRs last 50-150 ms; the window sustains replay drive across
        # the full event including low-rate inter-cycle timesteps.
        ca3_pop_rate  = ca3_spikes.float().mean().item()
        swr_max_steps = max(1, round(config.ripple_duration_max_ms / config.dt_ms))
        if ca3_pop_rate > config.ripple_threshold:
            if not self.ripple_detected:
                # Onset: capture the active CA3 attractor pattern for replay injection.
                self._swr_ca3_pattern = ca3_spikes.detach().clone()
            self.ripple_duration_steps = swr_max_steps   # extend/refresh window
            self.ripple_detected = True
        else:
            if self.ripple_duration_steps > 1:
                self.ripple_duration_steps -= 1
                self.ripple_detected = True   # still within the ripple envelope
            else:
                self.ripple_duration_steps = 0
                self.ripple_detected = False  # window expired

        return ca3_spikes, ca3_inhib_output

    def _step_ca2(
        self,
        ca2_input: torch.Tensor,
        synaptic_inputs: SynapticInput,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process one timestep of CA2 (social memory / temporal context).

        Reads ``_ca3_ca2_buffer`` internally.

        Args:
            ca2_input: External excitatory input (EC_II perforant path).

        Returns:
            ``(ca2_spikes, ca2_inhib_output)``
        """
        config = self.config

        ca3_ca2_synapse = SynapseId(
            self.region_name, HippocampusPopulation.CA3,
            self.region_name, HippocampusPopulation.CA2,
            receptor_type=ReceptorType.AMPA,
        )
        ca3_ca2_weights = self.get_synaptic_weights(ca3_ca2_synapse)

        # CA3→CA2 with axonal delay + STP (depressing – stability mechanism)
        ca3_spikes_for_ca2 = self._ca3_ca2_buffer.read(self._ca3_ca2_delay_steps).float()
        stp_efficacy = self.stp_modules[ca3_ca2_synapse].forward(ca3_spikes_for_ca2)
        ca2_from_ca3 = torch.matmul(ca3_ca2_weights, stp_efficacy * ca3_spikes_for_ca2)

        ca2_g_exc = F.relu(ca2_from_ca3 + ca2_input)

        ca2_inhib_output = self._run_hippocampal_inhibitory(
            inhib_net=self.ca2_inhibitory,
            prev_pyr_spikes=self._prev_spikes(HippocampusPopulation.CA2),
            prev_pv_spikes=self._prev_spikes(HippocampusPopulation.CA2_INHIBITORY_PV),
            prev_olm_spikes=self._prev_spikes(HippocampusPopulation.CA2_INHIBITORY_OLM),
            prev_bist_spikes=self._prev_spikes(HippocampusPopulation.CA2_INHIBITORY_BISTRATIFIED),
            pyr_pop=HippocampusPopulation.CA2,
            pv_pop=HippocampusPopulation.CA2_INHIBITORY_PV,
            olm_pop=HippocampusPopulation.CA2_INHIBITORY_OLM,
            bist_pop=HippocampusPopulation.CA2_INHIBITORY_BISTRATIFIED,
            synaptic_inputs=synaptic_inputs,
        )
        ca2_perisomatic_inhib = ca2_inhib_output["perisomatic_gaba_a"]

        # Burst-risk self-inhibition
        v_threshold  = self.ca2_neurons.v_threshold.mean().item()
        v_normalized = torch.clamp(self.ca2_neurons.V_soma / v_threshold, 0.0, 1.0)
        burst_risk   = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)
        ca2_g_inh = (
            F.relu(config.tonic_inhibition + ca2_perisomatic_inhib)
            + burst_risk * 0.1
        )

        ca2_g_ampa, ca2_g_nmda = split_excitatory_conductance(ca2_g_exc, nmda_ratio=0.05)

        ca2_spikes, _ = self.ca2_neurons.forward(
            g_ampa_input=ConductanceTensor(ca2_g_ampa),
            g_nmda_input=ConductanceTensor(ca2_g_nmda),
            g_gaba_a_input=ConductanceTensor(ca2_g_inh),
            g_gaba_b_input=ConductanceTensor(ca2_inhib_output["perisomatic_gaba_b"]),
        )
        return ca2_spikes, ca2_inhib_output

    def _step_ca1(
        self,
        ca1_input: torch.Tensor,
        ffi_factor: float,
        encoding_mod: float,
        retrieval_mod: float,
        synaptic_inputs: SynapticInput,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Process one timestep of CA1 (memory output / EC comparison layer).

        Reads ``_ca3_ca1_buffer`` and ``_ca2_ca1_buffer`` internally.
        Sets ``self._prev_encoding_mod`` (OLM-based fallback for when MS is absent)
        and ``self.nmda_trace``.

        Returns:
            ``(ca1_spikes, ca1_inhib_output, ca1_basal_g_exc, ca1_apical_g_exc)``
        """
        config = self.config

        ca3_ca1_synapse = SynapseId(
            self.region_name, HippocampusPopulation.CA3,
            self.region_name, HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        )
        ca2_ca1_synapse = SynapseId(
            self.region_name, HippocampusPopulation.CA2,
            self.region_name, HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        )
        ca3_ca1_weights = self.get_synaptic_weights(ca3_ca1_synapse)
        ca2_ca1_weights = self.get_synaptic_weights(ca2_ca1_synapse)

        # CA3→CA1 Schaffer collateral (delayed + STP depressing)
        ca3_spikes_delayed = self._ca3_ca1_buffer.read(self._ca3_ca1_delay_steps).float()
        stp_efficacy = self.stp_modules[ca3_ca1_synapse].forward(ca3_spikes_delayed)
        ca1_from_ca3 = torch.matmul(ca3_ca1_weights, stp_efficacy * ca3_spikes_delayed)

        # SWR replay boost (offline memory consolidation via sharp-wave ripples)
        if self.ripple_detected:
            ca1_from_ca3 = ca1_from_ca3 * config.ripple_boost_factor

        # FFI gating clears stale activity on stimulus change
        ca1_from_ec  = ca1_input   * ffi_factor
        ca1_from_ca3 = ca1_from_ca3 * ffi_factor

        # NMDA trace: tracks CA3-induced depolarisation for MgÂ²âº block removal
        if self.nmda_trace is not None:
            nmda_decay = decay_tensor(config.dt_ms, config.nmda_tau, device=self.nmda_trace.device)
            self.nmda_trace = self.nmda_trace * nmda_decay + ca1_from_ca3 * (1.0 - nmda_decay)
        else:
            self.nmda_trace = ca1_from_ca3

        # NMDA gating: stronger during retrieval (theta peak)
        mg_block_removal  = torch.sigmoid(
            (self.nmda_trace - config.nmda_threshold) * config.nmda_steepness
        ) * retrieval_mod
        nmda_conductance  = ca1_from_ec * mg_block_removal

        # Muscarinic NR2B enhancement (septal ACh boosts CaÂ²âº influx during encoding)
        ach_septal_mean  = self._ach_septal_concentration_ca1.mean().item()
        nmda_conductance = nmda_conductance * (1.0 + ach_septal_mean * 0.5)

        ampa_conductance  = ca1_from_ec * config.ampa_ratio
        ca3_contribution  = ca1_from_ca3 * (0.5 + 0.5 * encoding_mod)

        # CA2→CA1 (delayed + STP facilitating – temporal sequences)
        ca2_spikes_delayed = self._ca2_ca1_buffer.read(self._ca2_ca1_delay_steps).float()
        stp_efficacy = self.stp_modules[ca2_ca1_synapse].forward(ca2_spikes_delayed)
        ca1_from_ca2 = torch.matmul(ca2_ca1_weights, stp_efficacy * ca2_spikes_delayed) * ffi_factor

        # CA1 inhibitory network (OLM creates emergent encoding/retrieval separation)
        ca1_inhib_output = self._run_hippocampal_inhibitory(
            inhib_net=self.ca1_inhibitory,
            prev_pyr_spikes=self._prev_spikes(HippocampusPopulation.CA1),
            prev_pv_spikes=self._prev_spikes(HippocampusPopulation.CA1_INHIBITORY_PV),
            prev_olm_spikes=self._prev_spikes(HippocampusPopulation.CA1_INHIBITORY_OLM),
            prev_bist_spikes=self._prev_spikes(HippocampusPopulation.CA1_INHIBITORY_BISTRATIFIED),
            pyr_pop=HippocampusPopulation.CA1,
            pv_pop=HippocampusPopulation.CA1_INHIBITORY_PV,
            olm_pop=HippocampusPopulation.CA1_INHIBITORY_OLM,
            bist_pop=HippocampusPopulation.CA1_INHIBITORY_BISTRATIFIED,
            synaptic_inputs=synaptic_inputs,
        )
        ca1_perisomatic_inhib = ca1_inhib_output["perisomatic_gaba_a"]  # PV basket cells
        ca1_dendritic_inhib   = ca1_inhib_output["dendritic"]           # OLM + bistratified
        ca1_olm_inhib         = ca1_inhib_output["olm_dendritic"]       # OLM only

        # Update OLM-based encoding_mod fallback (used only when MS ACH is not wired).
        # Baseline 0.3 ensures tonic encoding drive during early training / standalone use.
        olm_firing_rate = ca1_olm_inhib.mean().item()
        self._prev_encoding_mod = torch.clamp(
            torch.tensor(0.3 + olm_firing_rate * 2.0), 0.0, 1.0
        ).item()

        # -----------------------------------------------------------------------
        # Two-compartment CA1:
        #   Basal (proximal)  – CA3 Schaffer collateral + CA2 temporal context
        #   Apical (distal)   – EC direct perforant path (retrieval route)
        # -----------------------------------------------------------------------
        ca1_basal_g_exc  = ca3_contribution + ca1_from_ca2
        ca1_apical_g_exc = ampa_conductance  + nmda_conductance

        ca1_basal_g_ampa,  ca1_basal_g_nmda  = split_excitatory_conductance(ca1_basal_g_exc,  nmda_ratio=0.2)
        ca1_apical_g_ampa, ca1_apical_g_nmda = split_excitatory_conductance(ca1_apical_g_exc, nmda_ratio=0.3)

        # Perisomatic inhibition (PV basket cells + population feedback + tonic ambient)
        prev_ca1_spikes    = self._prev_spikes(HippocampusPopulation.CA1)
        ca1_feedback_inhib = prev_ca1_spikes.mean() * 0.1

        v_threshold  = self.ca1_neurons.v_threshold.mean().item()
        v_normalized = torch.clamp(self.ca1_neurons.V_soma / v_threshold, 0.0, 1.0)
        burst_risk   = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)
        ca1_g_inh = (
            F.relu(config.tonic_inhibition + ca1_perisomatic_inhib + ca1_feedback_inhib)
            + burst_risk * 0.1
        )

        # OLM dendritic inhibition targets apical tuft (biologically correct)
        ca1_spikes, _ca1_membrane, _ca1_V_dend = self.ca1_neurons.forward(
            g_ampa_basal=ConductanceTensor(ca1_basal_g_ampa),
            g_nmda_basal=ConductanceTensor(ca1_basal_g_nmda),
            g_gaba_a_basal=ConductanceTensor(ca1_g_inh),
            g_gaba_b_basal=ConductanceTensor(ca1_inhib_output["perisomatic_gaba_b"]),
            g_ampa_apical=ConductanceTensor(ca1_apical_g_ampa),
            g_nmda_apical=ConductanceTensor(ca1_apical_g_nmda),
            g_gaba_a_apical=ConductanceTensor(F.relu(ca1_dendritic_inhib)),
        )

        return ca1_spikes, ca1_inhib_output, ca1_basal_g_exc, ca1_apical_g_exc

    def _update_match_mismatch(
        self,
        ca1_basal_g_exc: torch.Tensor,
        ca1_apical_g_exc: torch.Tensor,
        retrieval_mod: float,
    ) -> None:
        """Compare CA3 Schaffer drive vs EC direct drive to produce match/mismatch.

        Updates ``self.ca1_match_level`` and ``self.ca1_mismatch_level``.

        Ref: Kumaran & Maguire 2007; Lisman & Grace 2005 (Hippocampal-VTA loop).
        """
        _basal_peak  = ca1_basal_g_exc.max().item()
        _apical_peak = ca1_apical_g_exc.max().item()
        if _basal_peak > 1e-8 and _apical_peak > 1e-8:
            _basal_norm  = ca1_basal_g_exc  / _basal_peak
            _apical_norm = ca1_apical_g_exc / _apical_peak
            # Phase-gate: comparison is genuine only during retrieval mode.
            # During encoding, apical > basal just means "new input", not mismatch.
            self.ca1_match_level    = (torch.minimum(_basal_norm, _apical_norm).mean() * retrieval_mod).item()
            self.ca1_mismatch_level = (F.relu(_apical_norm - _basal_norm).mean() * retrieval_mod).item()
        elif _apical_peak > 1e-8:
            # EC active, CA3 silent → pure novelty (gated by retrieval mode)
            self.ca1_match_level    = 0.0
            self.ca1_mismatch_level = retrieval_mod
        else:
            self.ca1_match_level    = 0.0
            self.ca1_mismatch_level = 0.0

    def _apply_plasticity(
        self,
        synaptic_inputs: SynapticInput,
        dg_spikes: torch.Tensor,
        ca3_spikes: torch.Tensor,
        ca2_spikes: torch.Tensor,
        ca1_spikes: torch.Tensor,
        encoding_mod: float,
    ) -> None:
        """Apply all Hebbian and three-factor plasticity rules for the current timestep.

        Must be called BEFORE :meth:`_update_spike_buffers` so that delay buffers
        still hold the pre-update spike trains needed for learning.
        """
        config = self.config

        ca3_spikes_float = ca3_spikes.float()

        # Cache encoding_mod for _get_learning_kwargs (used by apply_learning)
        self._current_encoding_mod = encoding_mod

        # Synaptic handles
        ca3_ca3_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3, self.region_name, HippocampusPopulation.CA3, receptor_type=ReceptorType.AMPA)
        ca3_ca3_weights = self.get_synaptic_weights(ca3_ca3_synapse)

        # Re-read delay buffers (not yet advanced for this timestep)
        dg_spikes_delayed  = self._dg_ca3_buffer.read(self._dg_ca3_delay_steps)
        ca3_spikes_for_ca2 = self._ca3_ca2_buffer.read(self._ca3_ca2_delay_steps)
        ca2_spikes_delayed = self._ca2_ca1_buffer.read(self._ca2_ca1_delay_steps)
        ca2_spikes_for_ca3 = self._ca2_ca3_buffer.read(self._ca2_ca3_delay_steps)

        # Per-subregion DA levels
        ca3_da_level = self._da_concentration_ca3.mean().item()
        ca1_da_level = self._da_concentration_ca1.mean().item()

        # -----------------------------------------------------------------------
        # CA3→CA3 Hebbian (instantaneous delta + heterosynaptic LTD)
        # Kept inline: heterosynaptic LTD + diagonal zeroing are too specialized
        # for the generic STDPStrategy framework.
        # -----------------------------------------------------------------------
        if ca3_spikes.any():
            ca3_da_gain  = 0.2 + 1.8 * ca3_da_level  # Range: [0.2, 2.0]
            effective_lr = config.learning_rate * encoding_mod * ca3_da_gain
            dW = torch.outer(ca3_spikes_float, ca3_spikes_float)

            if config.heterosynaptic_ratio > 0:
                inactive_ca3_post = torch.logical_not(ca3_spikes).float()
                hetero_dW = -config.heterosynaptic_ratio * torch.outer(ca3_spikes_float, inactive_ca3_post)
                dW = dW + hetero_dW

            ca3_syn_mask = (ca3_ca3_weights.data > 0).float()
            ca3_ca3_weights.data += dW * effective_lr * ca3_syn_mask
            ca3_ca3_weights.data.fill_diagonal_(0.0)  # No self-connections
            clamp_weights(ca3_ca3_weights.data, config.synaptic_scaling.w_min, config.synaptic_scaling.w_max)

        # -----------------------------------------------------------------------
        # DG→CA3 Mossy fiber (one-shot binding via trace-based STDP)
        # "Detonator synapses" – 3× learning rate baked into strategy config.
        # DA modulation passed via dopamine kwarg to STDPStrategy.
        # -----------------------------------------------------------------------
        self._apply_learning(
            self._sid_dg_ca3, dg_spikes_delayed, ca3_spikes,
            dopamine=ca3_da_level * encoding_mod,
        )

        # -----------------------------------------------------------------------
        # CA3→CA2 suppressed STDP (RGS14-mediated CaMKII block)
        # 0.1× learning rate baked into strategy config.
        # -----------------------------------------------------------------------
        ca2_da_level = self._da_concentration_ca2.mean().item()
        self._apply_learning(
            self._sid_ca3_ca2, ca3_spikes_for_ca2, ca2_spikes,
            dopamine=ca2_da_level * encoding_mod,
        )

        # -----------------------------------------------------------------------
        # CA2→CA1 STDP (temporal / social context binding)
        # -----------------------------------------------------------------------
        self._apply_learning(
            self._sid_ca2_ca1, ca2_spikes_delayed, ca1_spikes,
            dopamine=encoding_mod,  # Encoding-gated only (no DA scaling per original)
        )

        # -----------------------------------------------------------------------
        # CA2→CA3 modulatory STDP (temporal context stabilization)
        # -----------------------------------------------------------------------
        self._apply_learning(
            self._sid_ca2_ca3, ca2_spikes_for_ca3, ca3_spikes,
            dopamine=ca3_da_level * encoding_mod,
        )

        # -----------------------------------------------------------------------
        # External EC→HPC input learning (via framework dispatch)
        # Strategies are lazily registered here; base-class apply_learning()
        # dispatches them after _step() completes.
        # -----------------------------------------------------------------------
        if not self.config.learning_disabled:
            device = self.device

            for synapse_id in list(synaptic_inputs.keys()):
                if self.get_learning_strategy(synapse_id) is None:
                    if synapse_id.receptor_type.is_inhibitory:
                        self._add_learning_strategy(
                            synapse_id, self._external_istdp_strategy, device=device,
                        )
                    else:
                        self._add_learning_strategy(
                            synapse_id, self._external_stdp_strategy, device=device,
                        )

            if encoding_mod > 0.5:
                # CA3 recurrent three-factor learning (tag-and-capture)
                # dW = eligibility_trace * dopamine * learning_rate
                ca3_delayed_for_3f = self._ca3_ca3_buffer.read(self._ca3_ca3_delay_steps)
                da_ca3_deviation   = ca3_da_level - 0.5  # Deviation from tonic baseline

                if self.get_learning_strategy(ca3_ca3_synapse) is None:
                    self._add_learning_strategy(ca3_ca3_synapse, self._tag_and_capture_strategy, device=device)
                self._apply_learning(
                    ca3_ca3_synapse, ca3_delayed_for_3f, ca3_spikes,
                    modulator=da_ca3_deviation,
                )
                self.get_synaptic_weights(ca3_ca3_synapse).data.fill_diagonal_(0.0)

                # -----------------------------------------------------------------------
                # CA3→CA1 Schaffer collateral three-factor learning (tag-and-capture)
                # Classic NMDA-dependent LTP site; DA modulation from CA1 D1/D5
                # receptors gates late-LTP (Frey & Morris 1997).
                # -----------------------------------------------------------------------
                ca3_ca1_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3, self.region_name, HippocampusPopulation.CA1, receptor_type=ReceptorType.AMPA)
                ca3_delayed_for_ca1 = self._ca3_ca1_buffer.read(self._ca3_ca1_delay_steps)
                da_ca1_deviation    = ca1_da_level - 0.5  # CA1-local DA deviation

                if self.get_learning_strategy(ca3_ca1_synapse) is None:
                    self._add_learning_strategy(ca3_ca1_synapse, self._ca3_ca1_strategy, device=device)
                self._apply_learning(
                    ca3_ca1_synapse, ca3_delayed_for_ca1, ca1_spikes,
                    modulator=da_ca1_deviation,
                )

        # -----------------------------------------------------------------------
        # DA-gated consolidation (tag-and-capture) for CA3→CA3 and CA3→CA1
        # -----------------------------------------------------------------------
        if ca3_da_level > 0.1:
            tac = self._tag_and_capture_strategy.base_strategy
            assert isinstance(tac, TagAndCaptureStrategy)
            ca3_ca3_weights.data = tac.consolidate(ca3_da_level, ca3_ca3_weights)
            clamp_weights(ca3_ca3_weights.data, config.synaptic_scaling.w_min, config.synaptic_scaling.w_max)

        if ca1_da_level > 0.1:
            ca3_ca1_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3, self.region_name, HippocampusPopulation.CA1, receptor_type=ReceptorType.AMPA)
            ca3_ca1_weights = self.get_synaptic_weights(ca3_ca1_synapse)
            tac_ca1 = self._ca3_ca1_strategy.base_strategy
            assert isinstance(tac_ca1, TagAndCaptureStrategy)
            ca3_ca1_weights.data = tac_ca1.consolidate(ca1_da_level, ca3_ca1_weights)
            clamp_weights(ca3_ca1_weights.data, config.synaptic_scaling.w_min, config.synaptic_scaling.w_max)

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        """Supply encoding-gated DA modulation for external EC→HPC synapses.

        Called by the base class :meth:`apply_learning` after ``_step()``
        completes.  encoding_mod is cached each step so that EC→HPC
        learning is theta-gated (active during encoding, suppressed during
        retrieval).
        """
        # Average DA across subfields for external inputs
        da_mean = (
            self._da_concentration_dg.mean().item()
            + self._da_concentration_ca3.mean().item()
            + self._da_concentration_ca2.mean().item()
            + self._da_concentration_ca1.mean().item()
        ) / 4.0
        # Encoding gate: encode → full DA; retrieve → zero DA → near-zero LTP
        return {"dopamine": da_mean * self._current_encoding_mod}

    def _update_spike_buffers(
        self,
        dg_spikes:  torch.Tensor,
        ca3_spikes: torch.Tensor,
        ca2_spikes: torch.Tensor,
        ca1_spikes: torch.Tensor,
    ) -> None:
        """Advance all inter-layer delay buffers.

        Must be called AFTER :meth:`_apply_plasticity` so that plasticity can
        still read the pre-update spike trains.
        """
        # Inter-layer axonal delay buffers
        self._dg_ca3_buffer.write_and_advance(dg_spikes)
        self._ca3_ca3_buffer.write_and_advance(ca3_spikes)
        self._ca3_ca2_buffer.write_and_advance(ca3_spikes)
        self._ca3_ca1_buffer.write_and_advance(ca3_spikes)
        self._ca2_ca1_buffer.write_and_advance(ca2_spikes)
        self._ca2_ca3_buffer.write_and_advance(ca2_spikes)
