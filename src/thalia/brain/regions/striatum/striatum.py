"""
Striatum - Reinforcement Learning with Three-Factor Rule

The striatum (part of basal ganglia) learns through dopamine-modulated
plasticity, implementing the classic three-factor learning rule for
reinforcement learning.

Key Features:
=============
1. THREE-FACTOR LEARNING: Δw = eligibility × dopamine
   - Pre-post activity creates eligibility traces
   - Eligibility alone does NOT cause plasticity
   - Dopamine arriving later converts eligibility to weight change
   - DA burst → LTP, DA dip → LTD, No DA → no learning

2. DOPAMINE as REWARD PREDICTION ERROR:
   - Burst: "Better than expected" → reinforce recent actions
   - Dip: "Worse than expected" → weaken recent actions
   - Baseline: "As expected" → maintain current policy

3. LONG ELIGIBILITY TRACES:
   - Biological tau: 500-2000ms (Yagishita et al., 2014)
   - Allows credit assignment for delayed rewards
   - Synaptic tag persists until dopamine arrives

4. **ACTION SELECTION** (Winner-Take-All):
   - Lateral inhibition creates competition between action neurons
   - Winning action's synapses become eligible for learning
   - Dopamine retroactively credits (burst) or blames (dip) the winner
   - Losers' eligibility decays without reinforcement

**Biological Basis**:
====================
- **Medium Spiny Neurons (MSNs)**: 95% of striatal neurons
- **D1-MSNs (direct pathway)**: Express D1 receptors, DA → LTP → "Go" signal
- **D2-MSNs (indirect pathway)**: Express D2 receptors, DA → LTD → "No-Go" signal
- **Opponent Processing**: D1 promotes, D2 suppresses actions
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from thalia.brain.configs import StriatumConfig
from thalia.brain.regions.population_names import StriatumPopulation
from thalia.components import (
    ConductanceLIF,
    GapJunctionConfig,
    GapJunctionCoupling,
    NeuronFactory,
    NeuronType,
    STPConfig,
    WeightInitializer,
    NeuromodulatorReceptor,
)
from thalia.components.synapses.stp import (
    CORTICOSTRIATAL_PRESET,
    SCHAFFER_COLLATERAL_PRESET,
    THALAMO_STRIATAL_PRESET,
)
from thalia.typing import (
    NeuromodulatorInput,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.units import ConductanceTensor
from thalia.learning import (
    D1STDPConfig,
    D2STDPConfig,
    D1STDPStrategy,
    D2STDPStrategy,
)
from thalia.utils import (
    CircularDelayBuffer,
    compute_da_gain,
    compute_ne_gain,
)

from ..neural_region import NeuralRegion
from ..region_registry import register_region


@register_region(
    "striatum",
    aliases=["basal_ganglia"],
    description="Reinforcement learning via dopamine-modulated three-factor rule with D1/D2 opponent pathways",
    version="1.0",
    author="Thalia Project",
    config_class=StriatumConfig,
)
class Striatum(NeuralRegion[StriatumConfig]):
    """Striatal region with three-factor reinforcement learning.

    Implements dopamine-modulated learning:
    - Eligibility traces tag recently active synapses
    - Dopamine signal converts eligibility to plasticity
    - No learning without dopamine (unlike Hebbian)
    - Synaptic weights stored per-source in synaptic_weights dict
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: StriatumConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize Striatum with D1/D2 opponent pathways."""
        super().__init__(config, population_sizes, region_name)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.d1_size = population_sizes[StriatumPopulation.D1.value]
        self.d2_size = population_sizes[StriatumPopulation.D2.value]
        self.n_actions = population_sizes["n_actions"]
        self.neurons_per_action = population_sizes["neurons_per_action"]

        total_msn_neurons = self.d1_size + self.d2_size

        # =====================================================================
        # MULTI-SOURCE ELIGIBILITY TRACES
        # =====================================================================
        # Per-synapse D1STDPStrategy / D2STDPStrategy instances are lazily created
        # in _apply_msn_learning when a new synapse_id is first encountered.
        # Each strategy stores its own fast_trace + slow_trace buffers as
        # nn.Module buffers, replacing the previous SynapseIdBufferDict approach.
        # Source-specific eligibility tau configuration (optional overrides for future use)
        self._source_eligibility_tau: Dict[str, float] = {}

        # =====================================================================
        # EXPLORATION (UCB + Adaptive Exploration)
        # =====================================================================
        # Adaptive exploration tracking
        self._recent_rewards: List[float] = []
        self._recent_accuracy = 0.0
        self.tonic_dopamine = self.config.tonic_dopamine

        # =====================================================================
        # D1/D2 PATHWAYS - Separate MSN Populations
        # =====================================================================
        # Create D1 and D2 pathways with their own neuron populations
        self.d1_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=StriatumPopulation.D1.value,
            neuron_type=NeuronType.MSN_D1,
            n_neurons=self.d1_size,
            device=self.device,
        )
        self.d2_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=StriatumPopulation.D2.value,
            neuron_type=NeuronType.MSN_D2,
            n_neurons=self.d2_size,
            device=self.device,
        )

        # =====================================================================
        # FSI (FAST-SPIKING INTERNEURONS) - Parvalbumin+ Interneurons
        # =====================================================================
        # FSI are ~2% of striatal neurons, provide feedforward inhibition
        # Critical for action selection timing (Koós & Tepper 1999)
        # Gap junction networks enable ultra-fast synchronization (<0.1ms)

        self.fsi_size = max(1, int(total_msn_neurons * 0.02))  # FSI as fraction of total striatal neurons (2%), minimum 1 neuron
        self.fsi_neurons: ConductanceLIF = NeuronFactory.create_fast_spiking_neurons(
            region_name=self.region_name,
            population_name=StriatumPopulation.FSI.value,
            n_neurons=self.fsi_size,
            device=self.device,
        )
        self.gap_junctions_fsi: Optional[GapJunctionCoupling] = None  # Will be initialized after weights

        # =====================================================================
        # MSN→FSI CONNECTIONS (Excitatory Feedback, Koos & Tepper 1999)
        # =====================================================================
        # Biology: MSNs excite FSI via glutamatergic collaterals (~30% connectivity)
        # This creates the feedback loop needed for winner-take-all dynamics:
        # - Winning action's MSNs fire more → excite FSI more
        # - FSI depolarizes → voltage-dependent Ca²⁺ channels open
        # - More GABA release → stronger inhibition of losing actions
        #
        # Implementation: Sparse connectivity from both D1 and D2 MSNs to FSI
        # Shape: [fsi_size, d1_size+d2_size] - FSI receive from all MSNs

        # D1 MSNs → FSI (excitatory, ~30% connectivity)
        # CONDUCTANCE-BASED: Weak MSN→FSI connections
        self._add_internal_connection(
            source_population=StriatumPopulation.D1.value,
            target_population=StriatumPopulation.FSI.value,
            weights=WeightInitializer.sparse_random(
                n_input=self.d1_size,
                n_output=self.fsi_size,
                connectivity=0.3,
                weight_scale=0.0005,
                device=self.device,
            ),
            stp_config=None,
            is_inhibitory=False,
        )

        # D2 MSNs → FSI (excitatory, ~30% connectivity)
        # CONDUCTANCE-BASED: Weak MSN→FSI connections (matches D1)
        self._add_internal_connection(
            source_population=StriatumPopulation.D2.value,
            target_population=StriatumPopulation.FSI.value,
            weights=WeightInitializer.sparse_random(
                n_input=self.d2_size,
                n_output=self.fsi_size,
                connectivity=0.3,
                weight_scale=0.0005,
                device=self.device,
            ),
            stp_config=None,
            is_inhibitory=False,
        )

        # =====================================================================
        # FSI→MSN CONNECTIONS (Per-Neuron, Moyer 2014)
        # =====================================================================
        # Biology: Each MSN receives ~116 feedforward connections from ~18 FSIs
        # FSI inputs are 4-10× STRONGER than MSN lateral inputs
        #
        # Implementation: Sparse connectivity matrix from FSI → MSNs
        # Shape: [msn_size, fsi_size] - which FSI neurons connect to which MSNs
        # NOT a global broadcast - each MSN gets different FSI subset

        # FSI → D1 connections
        # ~15% connectivity to match ~18 FSI per MSN (18/120 ≈ 0.15)
        # CONDUCTANCE-BASED: Strong inhibition (4-10x stronger than MSN lateral)
        self._add_internal_connection(
            source_population=StriatumPopulation.FSI.value,
            target_population=StriatumPopulation.D1.value,
            weights=WeightInitializer.sparse_random(
                n_input=self.fsi_size,
                n_output=self.d1_size,
                connectivity=0.15,
                weight_scale=0.001,
                device=self.device,
            ),
            stp_config=None,
            is_inhibitory=True,
        )

        # FSI → D2 connections (same structure)
        # CONDUCTANCE-BASED: Strong inhibition (matches D1)
        self._add_internal_connection(
            source_population=StriatumPopulation.FSI.value,
            target_population=StriatumPopulation.D2.value,
            weights=WeightInitializer.sparse_random(
                n_input=self.fsi_size,
                n_output=self.d2_size,
                connectivity=0.15,
                weight_scale=0.001,
                device=self.device,
            ),
            stp_config=None,
            is_inhibitory=True,
        )

        # =====================================================================
        # MSN→MSN LATERAL INHIBITION (INTERNAL - like CA3→CA3 recurrence)
        # =====================================================================
        # D1 → D1: Lateral inhibition for action selection
        # MSN→MSN GABAergic collaterals create winner-take-all dynamics
        # Distance: ~100-300μm (local), unmyelinated → 1-2ms delay (handled in forward)
        # Enables action-specific competition
        self._add_internal_connection(
            source_population=StriatumPopulation.D1.value,
            target_population=StriatumPopulation.D1.value,
            # TODO: Zero self-connections (no autapses)
            weights=WeightInitializer.sparse_random(
                n_input=self.d1_size,
                n_output=self.d1_size,
                connectivity=0.4,
                weight_scale=0.0005,
                device=self.device,
            ),
            stp_config=None,
            is_inhibitory=True,
        )

        # D2 → D2: Lateral inhibition for NoGo pathway
        # Similar MSN→MSN collaterals in indirect pathway
        self._add_internal_connection(
            source_population=StriatumPopulation.D2.value,
            target_population=StriatumPopulation.D2.value,
            # TODO: Zero self-connections (no autapses)
            weights=WeightInitializer.sparse_random(
                n_input=self.d2_size,
                n_output=self.d2_size,
                connectivity=0.4,
                weight_scale=0.0005,
                device=self.device,
            ),
            stp_config=None,
            is_inhibitory=True,
        )

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY (Adaptive Gain)
        # =====================================================================
        # EMA tracking of firing rates (per pathway)
        self.register_buffer("d1_firing_rate", torch.zeros(self.d1_size, device=self.device))
        self.register_buffer("d2_firing_rate", torch.zeros(self.d2_size, device=self.device))

        # =====================================================================
        # D1/D2 PATHWAY DELAY BUFFERS (Temporal Competition)
        # =====================================================================
        # Implement biologically-accurate transmission delays for opponent pathways:
        # - D1 direct pathway: ~15ms (Striatum → GPi/SNr → Thalamus)
        # - D2 indirect pathway: ~25ms (Striatum → GPe → STN → GPi/SNr → Thalamus)
        # D1 arrives ~10ms before D2, creating temporal competition window.

        # Calculate delay steps from millisecond delays
        self._d1_delay_steps = int(self.config.d1_to_output_delay_ms / self.config.dt_ms)
        self._d2_delay_steps = int(self.config.d2_to_output_delay_ms / self.config.dt_ms)

        self._d1_spike_buffer = CircularDelayBuffer(max_delay=self._d1_delay_steps, size=self.d1_size, device=self.device, dtype=torch.bool)
        self._d2_spike_buffer = CircularDelayBuffer(max_delay=self._d2_delay_steps, size=self.d2_size, device=self.device, dtype=torch.bool)

        # =====================================================================
        # DOPAMINE RECEPTORS (Spiking DA from VTA)
        # =====================================================================
        # Convert VTA dopamine neuron spikes to synaptic concentration.
        # Biology:
        # - D1 receptors: Gs-coupled → increase cAMP → facilitate LTP
        # - D2 receptors: Gi-coupled → decrease cAMP → facilitate LTD
        # - DA rise time: ~10-20 ms (fast release)
        # - DA decay time: ~200 ms (slow DAT reuptake)
        # Both pathways receive same DA spikes, but receptors have opposite effects.

        # Create D1 and D2 DA receptors (both receive same VTA spikes)
        self.da_receptor_d1 = NeuromodulatorReceptor(
            n_receptors=self.d1_size,
            tau_rise_ms=10.0,  # Fast release (ms)
            tau_decay_ms=200.0,  # Slow reuptake via DAT (ms)
            spike_amplitude=0.15,  # Moderate amplitude for summation
            device=self.device,
        )
        self.da_receptor_d2 = NeuromodulatorReceptor(
            n_receptors=self.d2_size,
            tau_rise_ms=10.0,
            tau_decay_ms=200.0,
            spike_amplitude=0.15,
            device=self.device,
        )

        # DA concentration state (updated each timestep from VTA spikes)
        self._da_concentration_d1 = torch.zeros(self.d1_size, device=self.device)
        self._da_concentration_d2 = torch.zeros(self.d2_size, device=self.device)

        # =====================================================================
        # NOREPINEPHRINE RECEPTORS (Spiking NE from LC)
        # =====================================================================
        # Convert LC norepinephrine neuron spikes to synaptic concentration.
        # Biology:
        # - NE modulates gain: high NE → increased excitability
        # - NE promotes exploration: high NE → more random action selection
        # - NE rise time: ~8 ms (fast release)
        # - NE decay time: ~150 ms (NET reuptake)
        # Both D1 and D2 MSNs receive NE modulation.

        self.ne_receptor_d1 = NeuromodulatorReceptor(
            n_receptors=self.d1_size,
            tau_rise_ms=8.0,  # Fast release (ms)
            tau_decay_ms=150.0,  # NET reuptake (ms)
            spike_amplitude=0.12,  # Moderate amplitude
            device=self.device,
        )
        self.ne_receptor_d2 = NeuromodulatorReceptor(
            n_receptors=self.d2_size,
            tau_rise_ms=8.0,
            tau_decay_ms=150.0,
            spike_amplitude=0.12,
            device=self.device,
        )

        # NE concentration state (updated from LC spikes)
        self._ne_concentration_d1 = torch.zeros(self.d1_size, device=self.device)
        self._ne_concentration_d2 = torch.zeros(self.d2_size, device=self.device)

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(StriatumPopulation.D1.value, self.d1_neurons)
        self._register_neuron_population(StriatumPopulation.D2.value, self.d2_neurons)
        self._register_neuron_population(StriatumPopulation.FSI.value, self.fsi_neurons)

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

    # =========================================================================
    # SYNAPTIC WEIGHT MANAGEMENT
    # =========================================================================

    def add_input_source(
        self,
        synapse_id: SynapseId,
        n_input: int,
        connectivity: float,
        weight_scale: float,
        *,
        stp_config: Optional[STPConfig | List[STPConfig]] = None,
    ) -> None:
        """Add synaptic weights for a new input source with automatic D1/D2 pathway weight creation.

        This is the primary method for connecting input sources to the striatum.
        It creates BOTH D1 and D2 pathway weights for the given source.

        Args:
            synapse_id: Key identifying the input source (e.g., "cortex:l5")
            n_input: Input size from this source
            connectivity: Connection probability (0.0 = no connections, 1.0 = fully connected)
            weight_scale: Initial weight scale multiplier
        """
        # NOTE: We create separate D1 and D2 weights for each input.
        # This is because both pathways receive the same input but may have different
        # synaptic strengths and STP dynamics.
        # Biology: Corticostriatal synapses onto D1 and D2 MSNs can have different properties,
        # so we maintain separate weight matrices for each pathway.
        # The synapse_id's target_population is used to identify the source, but we create
        # separate keys for D1 and D2 pathways.
        # The target_population in the synapse_id is expected to be "d1" for this method,
        # but we will create both "d1" and "d2" keys internally.

        assert synapse_id.target_population == StriatumPopulation.D1.value, (
            f"SynapseId target_population must be '{StriatumPopulation.D1.value}' for add_input_source. "
            f"Received target_population='{synapse_id.target_population}'"
        )

        d1_synapse = SynapseId(
            source_region=synapse_id.source_region,
            source_population=synapse_id.source_population,
            target_region=self.region_name,
            target_population=StriatumPopulation.D1.value,
        )
        d2_synapse = SynapseId(
            source_region=synapse_id.source_region,
            source_population=synapse_id.source_population,
            target_region=self.region_name,
            target_population=StriatumPopulation.D2.value,
        )
        fsi_synapse = SynapseId(
            source_region=synapse_id.source_region,
            source_population=synapse_id.source_population,
            target_region=self.region_name,
            target_population=StriatumPopulation.FSI.value,
        )

        # =====================================================================
        # CREATE STP MODULES FOR SOURCE-PATHWAY
        # =====================================================================
        if stp_config is None:
            # Determine STP type based on source name
            if synapse_id.source_region == "cortex":
                stp_preset = CORTICOSTRIATAL_PRESET
            elif synapse_id.source_region == "thalamus":
                stp_preset = THALAMO_STRIATAL_PRESET
            elif synapse_id.source_region == "hippocampus":
                stp_preset = SCHAFFER_COLLATERAL_PRESET
            else:
                stp_preset = CORTICOSTRIATAL_PRESET  # Default to cortical

            d1_d2_stp_config = stp_preset.configure()
        else:
            # If stp_config is provided, use it for both D1 and D2 pathways of this source.
            d1_d2_stp_config = stp_config

        super().add_input_source(d1_synapse, n_input, connectivity, weight_scale, stp_config=d1_d2_stp_config)
        super().add_input_source(d2_synapse, n_input, connectivity, weight_scale, stp_config=d1_d2_stp_config)
        super().add_input_source(fsi_synapse, n_input, 0.5, weight_scale, stp_config=None)

        # Create gap junction coupling (if this is first FSI source)
        if self.gap_junctions_fsi is None:
            fsi_weights = self.get_synaptic_weights(fsi_synapse)
            gap_config_fsi = GapJunctionConfig(
                coupling_strength=self.config.gap_junction_strength,
                connectivity_threshold=self.config.gap_junction_threshold,
                max_neighbors=self.config.gap_junction_max_neighbors,
            )
            # Use first FSI source weights for gap junction neighborhood computation
            self.gap_junctions_fsi = GapJunctionCoupling(
                n_neurons=self.fsi_size,
                afferent_weights=fsi_weights,
                config=gap_config_fsi,
                device=self.device,
            )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process input and select action using separate D1/D2 populations.

        Args:
            synaptic_inputs: Point-to-point synaptic connections from cortex, hippocampus, thalamus
            neuromodulator_inputs: Broadcast neuromodulatory signals (DA, NE, ACh)
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        cfg = self.config

        if cfg.enable_neuromodulation:
            # =====================================================================
            # DOPAMINE RECEPTOR UPDATE (VTA Spikes → Concentration)
            # =====================================================================
            # Convert spiking DA from VTA to synaptic concentration for learning.
            vta_da_spikes = neuromodulator_inputs.get('da', None)
            # Update D1 and D2 receptors (both receive same VTA spikes)
            self._da_concentration_d1 = self.da_receptor_d1.update(vta_da_spikes)
            self._da_concentration_d2 = self.da_receptor_d2.update(vta_da_spikes)

            # =====================================================================
            # NOREPINEPHRINE RECEPTOR UPDATE (LC Spikes → Concentration)
            # =====================================================================
            # Convert spiking NE from LC to synaptic concentration for gain modulation.
            # NE increases excitability and promotes exploration.
            lc_ne_spikes = neuromodulator_inputs.get('ne', None)
            # Update D1 and D2 receptors (both receive same LC spikes)
            self._ne_concentration_d1 = self.ne_receptor_d1.update(lc_ne_spikes)
            self._ne_concentration_d2 = self.ne_receptor_d2.update(lc_ne_spikes)
        else:
            # Neuromodulation disabled: keep baseline concentrations
            pass

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Each source (cortex:l5, hippocampus, thalamus) has separate weights
        # for D1 and D2 pathways. Filter inputs by target population.
        # Biology: D1 and D2 MSNs are distinct neurons with independent synaptic
        # weights, so integration must remain separate per pathway.
        d1_conductance = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, self.d1_size, filter_by_target_population=StriatumPopulation.D1.value
        )
        d2_conductance = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, self.d2_size, filter_by_target_population=StriatumPopulation.D2.value
        )

        # Compute excitation from D1 and D2 MSNs to FSI
        # Use previous timestep's spikes for causal feedback (delay=1)
        prev_d1_spikes = self._d1_spike_buffer.read(1).float()
        prev_d2_spikes = self._d2_spike_buffer.read(1).float()

        # =====================================================================
        # FSI (FAST-SPIKING INTERNEURONS) - Feedforward Inhibition
        # =====================================================================
        # FSI process inputs in parallel with MSNs but with:
        # 1. Gap junction coupling for synchronization (<0.1ms)
        # 2. Feedforward inhibition to MSNs (sharpens action timing)
        # Biology: FSI are parvalbumin+ interneurons (~2% of striatum)
        if self.fsi_size > 0:
            fsi_conductance = self._integrate_synaptic_inputs_at_dendrites(
                synaptic_inputs, self.fsi_size, filter_by_target_population=StriatumPopulation.FSI.value
            )

            # Apply gap junction coupling
            if (self.gap_junctions_fsi is not None and self.fsi_neurons.membrane is not None):
                gap_conductance, _gap_reversal = self.gap_junctions_fsi.forward(self.fsi_neurons.membrane)
                fsi_conductance = fsi_conductance + gap_conductance

            # =====================================================================
            # MSN→FSI EXCITATION (Winner-Take-All Feedback)
            # =====================================================================
            # Biology: MSN collaterals excite FSI via glutamatergic synapses
            # This creates positive feedback for winner-take-all:
            # - Winning action's MSNs fire more → excite FSI more
            # - FSI depolarizes → voltage-dependent GABA release increases
            # - Increased GABA → suppresses losing action more
            # - Gap widens → runaway to winner-take-all state
            #
            # CRITICAL: Use PREVIOUS timestep's MSN activity (causal)
            # FSI response from t-1 MSN activity influences t MSN spikes

            d1_fsi_synapse = SynapseId(
                source_region=self.region_name,
                source_population=StriatumPopulation.D1.value,
                target_region=self.region_name,
                target_population=StriatumPopulation.FSI.value,
            )
            d2_fsi_synapse = SynapseId(
                source_region=self.region_name,
                source_population=StriatumPopulation.D2.value,
                target_region=self.region_name,
                target_population=StriatumPopulation.FSI.value,
            )

            d1_fsi_conductance = self.get_synaptic_weights(d1_fsi_synapse) @ prev_d1_spikes
            d2_fsi_conductance = self.get_synaptic_weights(d2_fsi_synapse) @ prev_d2_spikes

            # Add MSN excitation to FSI conductance
            fsi_conductance = fsi_conductance + d1_fsi_conductance + d2_fsi_conductance

            # Update FSI neurons (fast kinetics, tau_mem ~5ms)
            # Split excitatory conductance into AMPA (fast) and NMDA (slow)
            fsi_g_ampa, fsi_g_nmda = self._split_excitatory_conductance(fsi_conductance)

            fsi_spikes, fsi_membrane = self.fsi_neurons.forward(
                g_ampa_input=ConductanceTensor(fsi_g_ampa),
                g_gaba_a_input=None,
                g_nmda_input=ConductanceTensor(fsi_g_nmda),
            )
            fsi_spikes_float = fsi_spikes.float()

            # =====================================================================
            # PER-NEURON FSI→MSN INHIBITION WITH VOLTAGE-DEPENDENT GABA RELEASE
            # =====================================================================
            # Biology: Each MSN receives ~116 feedforward connections from ~18 FSIs
            # FSI inputs are 4-10× STRONGER than MSN lateral inputs
            # CRITICAL: GABA release is voltage-dependent (Ca²⁺ channel dynamics)!
            #
            # Mechanism:
            # 1. FSI spikes create baseline inhibition via fsi_to_msn_weights
            # 2. FSI membrane voltage modulates release strength (Ca²⁺-dependent)
            # 3. Depolarized FSI (recent high activity) → more GABA release
            # 4. Hyperpolarized FSI (recent low activity) → less GABA release
            #
            # This creates BISTABILITY:
            # - Competitive state (~-60 mV): weak GABA, balanced competition
            # - Winner-take-all state (~-50 mV): strong GABA, losers suppressed

            # Compute voltage-dependent inhibition scaling factor
            # fsi_membrane is [fsi_size] tensor of membrane voltages
            # Returns [fsi_size] tensor of scaling factors (0.1 to 0.8)
            inhibition_scale = self._fsi_membrane_to_inhibition_strength(fsi_membrane)

            # Average scaling across FSI population (they're synchronized via gap junctions)
            # This gives single scaling factor for whole network
            avg_inhibition_scale = inhibition_scale.mean()

            # FSI → D1+D2 inhibition
            fsi_d1_inhib_synapse = SynapseId(
                source_region=self.region_name,
                source_population=StriatumPopulation.FSI.value,
                target_region=self.region_name,
                target_population=StriatumPopulation.D1.value,
                is_inhibitory=True,
            )
            fsi_d2_inhib_synapse = SynapseId(
                source_region=self.region_name,
                source_population=StriatumPopulation.FSI.value,
                target_region=self.region_name,
                target_population=StriatumPopulation.D2.value,
                is_inhibitory=True,
            )
            fsi_d1_inhib_weights = self.get_synaptic_weights(fsi_d1_inhib_synapse)
            fsi_d2_inhib_weights = self.get_synaptic_weights(fsi_d2_inhib_synapse)

            fsi_inhibition_d1 = (fsi_d1_inhib_weights @ fsi_spikes_float) * avg_inhibition_scale
            fsi_inhibition_d2 = (fsi_d2_inhib_weights @ fsi_spikes_float) * avg_inhibition_scale
        else:
            # FSI disabled → no FSI inhibition
            fsi_inhibition_d1 = torch.zeros(self.d1_size, device=self.device)
            fsi_inhibition_d2 = torch.zeros(self.d2_size, device=self.device)

        # =====================================================================
        # MSN→MSN LATERAL INHIBITION
        # =====================================================================
        # Biology: GABAergic collaterals, local (~100-300μm), unmyelinated → 1-2ms
        # Mechanism: GABAergic collaterals with action-specific spatial organization
        # Creates action competition: neurons of one action inhibit neurons of other actions
        #
        # NOTE: MSN lateral inhibition is NOT ACh-modulated (unlike cortex/hippocampus):
        # - Striatal ACh comes from local cholinergic interneurons (ChIs), not nucleus basalis
        # - ChI-ACh primarily modulates corticostriatal INPUT (M1/M4 receptors on dendrites)
        # - MSN-MSN GABAergic collaterals lack strong cholinergic modulation
        # - Dopamine is the primary modulator of MSN lateral inhibition dynamics

        # D1/D2 lateral inhibition: D1/D2 MSNs inhibit each other (winner-take-all)
        d1_d1_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=StriatumPopulation.D1.value,
            target_region=self.region_name,
            target_population=StriatumPopulation.D1.value,
            is_inhibitory=True,
        )
        d2_d2_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=StriatumPopulation.D2.value,
            target_region=self.region_name,
            target_population=StriatumPopulation.D2.value,
            is_inhibitory=True,
        )
        d1_d1_inhib_weights = self.get_synaptic_weights(d1_d1_inhib_synapse)
        d2_d2_inhib_weights = self.get_synaptic_weights(d2_d2_inhib_synapse)
        d1_d1_inhibition = d1_d1_inhib_weights @ prev_d1_spikes
        d2_d2_inhibition = d2_d2_inhib_weights @ prev_d2_spikes

        # =====================================================================
        # D1/D2 NEURON ACTIVATION with Modulation
        # =====================================================================
        # Apply all modulation (theta, dopamine, NE, PFC, homeostasis) to currents
        # before neuron execution

        # Theta modulation emerges from hippocampal-cortical projections to striatum
        # D1/D2 balance determined by dopamine, inputs, and circuit dynamics
        # (no explicit encoding/retrieval phase modulation)

        # Dopamine gain modulation (per-neuron from receptors)
        # D1: DA increases excitability (Gs-coupled)
        # D2: DA decreases excitability (Gi-coupled) - inverted gain
        d1_da_gain = compute_da_gain(self._da_concentration_d1, da_factor=0.3)
        d2_da_gain = compute_da_gain(self._da_concentration_d2, da_factor=-0.2)

        # NE gain modulation (average across neurons)
        d1_ne_gain = compute_ne_gain(self._ne_concentration_d1.mean().item())
        d2_ne_gain = compute_ne_gain(self._ne_concentration_d2.mean().item())

        d1_conductance = d1_conductance * d1_da_gain * d1_ne_gain
        d2_conductance = d2_conductance * d2_da_gain * d2_ne_gain

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY
        # =====================================================================
        # Add baseline noise (spontaneous miniature EPSPs)
        if cfg.baseline_noise_conductance_enabled:
            d1_noise = torch.randn_like(d1_conductance) * 0.007
            d2_noise = torch.randn_like(d2_conductance) * 0.007
        else:
            d1_noise = torch.zeros_like(d1_conductance)
            d2_noise = torch.zeros_like(d2_conductance)

        d1_conductance = (d1_conductance + d1_noise).clamp(min=0)
        d2_conductance = (d2_conductance + d2_noise).clamp(min=0)

        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        d1_g_ampa, d1_g_nmda = self._split_excitatory_conductance(d1_conductance)
        d2_g_ampa, d2_g_nmda = self._split_excitatory_conductance(d2_conductance)

        # CONDUCTANCE-BASED: Inhibition goes to g_gaba_a_input, NOT mixed with excitation
        # Combine all inhibitory sources (FSI + MSN lateral) - all are POSITIVE conductances
        d1_inhibition = (fsi_inhibition_d1 + d1_d1_inhibition).clamp(min=0)
        d2_inhibition = (fsi_inhibition_d2 + d2_d2_inhibition).clamp(min=0)

        # Execute D1 and D2 MSN populations
        d1_spikes, _ = self.d1_neurons.forward(
            g_ampa_input=ConductanceTensor(d1_g_ampa),
            g_gaba_a_input=ConductanceTensor(d1_inhibition),
            g_nmda_input=ConductanceTensor(d1_g_nmda),
        )
        d2_spikes, _ = self.d2_neurons.forward(
            g_ampa_input=ConductanceTensor(d2_g_ampa),
            g_gaba_a_input=ConductanceTensor(d2_inhibition),
            g_nmda_input=ConductanceTensor(d2_g_nmda),
        )

        # =====================================================================
        # HOMEOSTATIC GAIN UPDATE (After Spiking)
        # =====================================================================
        # Update firing rate EMA and adapt gains to maintain target rates
        # BIOLOGICAL STRATEGY: Regulate COMBINED D1+D2 rate, not independently
        # - Allows natural D1/D2 balance to emerge from competition
        # - Prevents asymmetric gain drift that causes weight divergence
        # - If total rate too high: reduce both gains proportionally
        # - If total rate too low: increase both gains proportionally

        # Update D1/D2 firing rates (EMA)
        self.d1_firing_rate.data.mul_(1.0 - self._firing_rate_alpha).add_(self._firing_rate_alpha * d1_spikes.float())
        self.d2_firing_rate.data.mul_(1.0 - self._firing_rate_alpha).add_(self._firing_rate_alpha * d2_spikes.float())

        # Compute COMBINED firing rate (D1 + D2 together)
        # Biology: Striatum as a whole should maintain sparse coding
        combined_rate = (self.d1_firing_rate.mean() + self.d2_firing_rate.mean()) / 2.0

        # Rate error for combined population (positive = underactive)
        combined_rate_error = cfg.target_firing_rate - combined_rate

        # INTRINSIC EXCITABILITY: Modulate leak conductance for BOTH pathways
        # Inverse relationship: underactive → lower g_L
        g_L_update = -cfg.gain_learning_rate * combined_rate_error
        self.d1_neurons.g_L_scale.data.add_(g_L_update).clamp_(min=0.1, max=2.0)
        self.d2_neurons.g_L_scale.data.add_(g_L_update).clamp_(min=0.1, max=2.0)

        # Adaptive threshold update (complementary to g_L modulation)
        # Also use combined error to maintain balance
        # Adjust thresholds based on combined activity, not independently
        # Lower threshold when underactive, raise when overactive
        threshold_update = -cfg.threshold_learning_rate * combined_rate_error
        self.d1_neurons.adjust_thresholds(threshold_update, cfg.threshold_min, cfg.threshold_max)
        self.d2_neurons.adjust_thresholds(threshold_update, cfg.threshold_min, cfg.threshold_max)

        # =====================================================================
        # LEARNING: UPDATE ELIGIBILITY TRACES AND APPLY DOPAMINE-MODULATED PLASTICITY
        # =====================================================================
        self._apply_msn_learning(synaptic_inputs, d1_spikes, d2_spikes)

        region_outputs: RegionOutput = {
            StriatumPopulation.D1.value: d1_spikes,
            StriatumPopulation.D2.value: d2_spikes,
        }

        # =====================================================================
        # UPDATE STATE BUFFERS FOR NEXT TIMESTEP
        # =====================================================================
        # Write current spikes to all state buffers
        # Next forward pass will read these with delay=1 for previous timestep
        self._d1_spike_buffer.write_and_advance(d1_spikes)
        self._d2_spike_buffer.write_and_advance(d2_spikes)

        return self._post_forward(region_outputs)

    def _apply_msn_learning(
        self,
        synaptic_inputs: SynapticInput,
        d1_spikes: torch.Tensor,
        d2_spikes: torch.Tensor,
    ) -> None:
        """Apply D1/D2 three-factor learning via registered D1/D2STDPStrategy instances.

        Each synapse_id targeting a D1 or D2 population gets a lazily-created
        :class:`~thalia.learning.D1STDPStrategy` or :class:`~thalia.learning.D2STDPStrategy`
        registered in ``self._learning_strategies``.  The strategy owns the multi-scale
        eligibility trace buffers, replacing the previous SynapseIdBufferDict approach.

        Three-factor update rule (D1):  Δw = (fast + α·slow) × DA × lr
        D2 path is identical but with dopamine signal inverted (Gi-coupled receptor).
        """
        # Use per-neuron DA concentration from receptors (not scalar broadcast)
        d1_da = self._da_concentration_d1   # [d1_size]
        d2_da = self._da_concentration_d2   # [d2_size]

        for synapse_id, source_spikes in synaptic_inputs.items():
            if not self.has_synaptic_weights(synapse_id):
                continue

            if synapse_id.target_population == StriatumPopulation.D1.value:
                if self.get_learning_strategy(synapse_id) is None:
                    self._register_msn_strategy(synapse_id, d1_pathway=True)
                self.apply_learning(
                    synapse_id,
                    pre_spikes=source_spikes,
                    post_spikes=d1_spikes,
                    dopamine=d1_da,
                )

            elif synapse_id.target_population == StriatumPopulation.D2.value:
                if self.get_learning_strategy(synapse_id) is None:
                    self._register_msn_strategy(synapse_id, d1_pathway=False)
                self.apply_learning(
                    synapse_id,
                    pre_spikes=source_spikes,
                    post_spikes=d2_spikes,
                    dopamine=d2_da,
                )

    def _register_msn_strategy(self, synapse_id: SynapseId, *, d1_pathway: bool) -> None:
        """Lazily create and register a D1 or D2 learning strategy for *synapse_id*.

        Called the first time a synapse_id for a D1 or D2 population is encountered
        during learning.  The strategy is added to ``self._learning_strategies``
        (a :class:`SynapseIdModuleDict`) so it is tracked as a proper ``nn.Module``
        submodule and participates in ``.to(device)`` and ``state_dict()``.
        """
        cfg = self.config
        base_cfg = D1STDPConfig(
            learning_rate=cfg.learning_rate,
            fast_eligibility_tau_ms=cfg.fast_eligibility_tau_ms,
            slow_eligibility_tau_ms=cfg.slow_eligibility_tau_ms,
            eligibility_consolidation_rate=cfg.eligibility_consolidation_rate,
            slow_trace_weight=cfg.slow_trace_weight,
        )
        if d1_pathway:
            strategy: D1STDPStrategy = D1STDPStrategy(base_cfg)
        else:
            strategy = D2STDPStrategy(D2STDPConfig(
                learning_rate=cfg.learning_rate,
                fast_eligibility_tau_ms=cfg.fast_eligibility_tau_ms,
                slow_eligibility_tau_ms=cfg.slow_eligibility_tau_ms,
                eligibility_consolidation_rate=cfg.eligibility_consolidation_rate,
                slow_trace_weight=cfg.slow_trace_weight,
            ))
        # Register without eager setup — ensure_setup() is called on first compute_update()
        self.add_learning_strategy(synapse_id, strategy)

    def _get_source_eligibility_tau(self, source_name: str) -> float:
        """Get source-specific eligibility trace tau.

        Different input sources have different temporal dynamics:
        - Cortical inputs: Long traces (1000ms) for sustained context
        - Hippocampal inputs: Fast traces (300ms) for episodic snapshots
        - Thalamic inputs: Intermediate (500ms) for phasic signals

        Args:
            source_name: Source identifier (e.g., "cortex:l5", "hippocampus")

        Returns:
            Eligibility tau in milliseconds
        """
        # Check if custom tau is configured
        if hasattr(self, "_source_eligibility_tau") and source_name in self._source_eligibility_tau:
            tau_value = self._source_eligibility_tau[source_name]
            # Ensure it's a float (handle tensor or other numeric types)
            if isinstance(tau_value, torch.Tensor):
                return float(tau_value.item())
            return float(tau_value)

        # Apply biological defaults based on source type
        if "cortex" in source_name:
            return 1000.0  # Standard corticostriatal (long traces)
        elif "hippocampus" in source_name or "hippoc" in source_name:
            return 300.0  # Fast episodic context
        elif "thalamus" in source_name or "thal" in source_name:
            return 500.0  # Intermediate phasic signals
        else:
            # Default to config value
            return (
                self.config.eligibility_tau_ms
                if self.config.eligibility_tau_ms is not None
                else 1000.0
            )

    # =========================================================================
    # MULTI-SOURCE SYNAPTIC INTEGRATION
    # =========================================================================

    def _fsi_membrane_to_inhibition_strength(self, fsi_membrane_v: torch.Tensor) -> torch.Tensor:
        """Convert FSI membrane potential to inhibition scaling factor.

        Biology: GABA release is voltage-dependent!
        - Calcium influx increases with depolarization
        - More Ca²⁺ → more vesicle fusion → more GABA release
        - This creates nonlinear relationship between membrane voltage and inhibition

        NO RATE COMPUTATION - the membrane potential itself carries temporal history:
        - Membrane time constant (tau_mem ~5ms) naturally integrates recent spikes
        - Depolarized membrane = recent high activity
        - Hyperpolarized membrane = recent low activity

        This voltage-dependent release creates BISTABILITY:
        - Below ~-58 mV: weak GABA release, competitive dynamics
        - Above ~-52 mV: strong GABA release, winner-take-all dynamics

        Implementation:
        - Baseline (0.1): Minimum inhibition at rest (~-65 mV)
        - Maximum (0.8): Maximum inhibition near threshold (~-45 mV)
        - Inflection (-55 mV): Where transition happens (FSI saturation)
        - Steepness (3.0 mV): How sharp the transition is

        Args:
            fsi_membrane_v: FSI membrane potentials [n_fsi] in mV

        Returns:
            Per-FSI inhibition scaling factors [n_fsi] (0.1 to 0.8)
        """
        baseline = 0.1  # Minimum inhibition (at rest, ~-65 mV)
        maximum = 0.8  # Maximum inhibition (near threshold, ~-45 mV)
        inflection = -55.0  # Voltage where transition happens (mV)
        steepness = 3.0  # How sharp the transition is (mV)

        # Sigmoid based on membrane voltage (voltage-dependent GABA release)
        # σ(v) = baseline + (maximum - baseline) / (1 + exp(-(v - inflection) / steepness))
        sigmoid = torch.sigmoid((fsi_membrane_v - inflection) / steepness)
        return baseline + (maximum - baseline) * sigmoid

    # =========================================================================
    # ACTION SELECTION HELPERS
    # =========================================================================

    def update_performance(self, reward: float) -> None:
        """Update performance history for adaptive exploration.

        This should be called ONCE per trial by the brain_system after
        reward is received. Not called inside forward() because forward()
        runs multiple times per timestep.

        Args:
            reward: Reward received for the selected action
        """
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > self.config.performance_window:
            self._recent_rewards.pop(0)

        # Update running accuracy
        window = min(len(self._recent_rewards), self.config.performance_window)
        if window > 0:
            correct_count = sum(1 for r in self._recent_rewards[-window:] if r > 0)
            self._recent_accuracy = correct_count / window

        # Adjust tonic dopamine based on performance
        # Poor performance → higher tonic DA → more exploration
        # Good performance → lower tonic DA → more exploitation
        if self._recent_accuracy < 0.5:
            self.tonic_dopamine = self.config.max_tonic_dopamine
        elif self._recent_accuracy > 0.8:
            self.tonic_dopamine = self.config.min_tonic_dopamine
        else:
            # Linear interpolation
            self.tonic_dopamine = (
                self.config.min_tonic_dopamine
                + (self.config.max_tonic_dopamine - self.config.min_tonic_dopamine)
                * (0.8 - self._recent_accuracy)
                / 0.3
            )

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Propagates dt update to neurons, STP components, and learning strategies.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)
        self.d1_neurons.update_temporal_parameters(dt_ms)
        self.d2_neurons.update_temporal_parameters(dt_ms)
        self.fsi_neurons.update_temporal_parameters(dt_ms)
