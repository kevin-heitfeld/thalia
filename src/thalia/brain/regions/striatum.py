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

import math
from typing import Any, ClassVar, Dict, List, Optional

import torch

from thalia import GlobalConfig
from thalia.brain.configs import StriatumConfig
from thalia.components import (
    ConductanceLIF,
    ConductanceLIFConfig,
    GapJunctionConfig,
    GapJunctionCoupling,
    NeuronFactory,
    NeuronType,
    WeightInitializer,
    NeuromodulatorReceptor,
)
from thalia.components.synapses.stp import FSI_MSN_PRESET, MSN_LATERAL_PRESET
from thalia.learning import (
    D1D2STDPConfig,
    D1STDPStrategy,
    D2STDPStrategy,
)
from thalia.typing import (
    ConductanceTensor,
    GapJunctionReversal,
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
    compute_da_gain,
    compute_ne_gain,
    split_excitatory_conductance,
)

from .neural_region import NeuralRegion
from .population_names import StriatumPopulation
from .region_registry import register_region


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

    # Mesolimbic DA (VTA → ventral striatum) drives D1/D2 opponent learning.
    # NE from LC modulates the gain of burst responses and threshold adaptation.
    neuromodulator_subscriptions: ClassVar[List[str]] = ['da_mesolimbic', 'ne']

    # Striatum publishes local ACh from TANs (cholinergic interneurons) on a
    # dedicated 'ach_striatal' channel so downstream circuits (e.g., SNc, VTA DA
    # terminals) can detect striatal ACh tone independently from cortical NB ACh.
    neuromodulator_outputs: ClassVar[Dict[str, str]] = {
        'ach_striatal': StriatumPopulation.TAN,
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: StriatumConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize Striatum with D1/D2 opponent pathways."""
        super().__init__(config, population_sizes, region_name)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.d1_size = population_sizes[StriatumPopulation.D1]
        self.d2_size = population_sizes[StriatumPopulation.D2]
        self.n_actions = population_sizes["n_actions"]
        self.neurons_per_action = population_sizes["neurons_per_action"]

        total_msn_neurons = self.d1_size + self.d2_size

        # =====================================================================
        # MULTI-SOURCE ELIGIBILITY TRACES
        # =====================================================================
        # Per-synapse D1STDPStrategy / D2STDPStrategy instances are lazily created
        # in forward when a new synapse_id is first encountered.
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
            population_name=StriatumPopulation.D1,
            neuron_type=NeuronType.MSN_D1,
            n_neurons=self.d1_size,
            device=self.device,
        )
        self.d2_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=StriatumPopulation.D2,
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

        # FSI: 2% of MSNs (Koós & Tepper 1999), but clamped to a minimum for
        # simulation stability — at <15 neurons gap-junction topology collapses
        # and feedforward inhibition becomes all-or-nothing.
        self.fsi_size = max(self.config.fsi_min_neurons, int(total_msn_neurons * 0.02))
        self.fsi_neurons: ConductanceLIF = NeuronFactory.create_fast_spiking_neurons(
            region_name=self.region_name,
            population_name=StriatumPopulation.FSI,
            n_neurons=self.fsi_size,
            device=self.device,
            # v_threshold restored to 1.0 (standard): threshold=1.3 was a stopgap for
            # the old NMDA-driven 406 Hz saturation. NMDA is now disabled (nmda_ratio=0.0
            # in forward pass), so 1.3 is too high and prevents any firing on cortical input.
            # With 1.0 and 20× afferent weights, FSI fires at ~80-120 Hz on thalamic drive
            # (fast-spiking tau_mem=5ms means FSI always fire fast once above threshold).
            v_threshold=1.0,
        )
        self.gap_junctions_fsi: Optional[GapJunctionCoupling] = None  # Lazily initialized on first forward pass

        # =====================================================================
        # REGISTER FOUNDATIONAL POPULATIONS
        # MUST happen before _add_internal_connection so the Dale's Law polarity
        # checks in _add_internal_connection have the correct registered polarity.
        # =====================================================================
        self._register_neuron_population(StriatumPopulation.D1, self.d1_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(StriatumPopulation.D2, self.d2_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(StriatumPopulation.FSI, self.fsi_neurons, polarity=PopulationPolarity.INHIBITORY)

        # =====================================================================
        # MSN→FSI CONNECTIONS (Inhibitory Feedback via GABAergic Collaterals)
        # =====================================================================
        # Biology: MSNs are purely GABAergic neurons (Dale's Law) and therefore
        # form INHIBITORY synapses via GABA_A receptors on all targets, including
        # FSI.  The comment previously claiming "glutamatergic collaterals" was
        # biologically incorrect.  MSN axon collaterals release GABA, not glutamate.
        # This lateral GABA inhibition contributes to winner-take-all dynamics:
        # - Winning action's MSNs fire more → increase GABA release onto rival MSNs
        # - FSI also receive inhibitory input (matching Tunstall et al. 2002)
        #
        # Implementation: Sparse connectivity from both D1 and D2 MSNs to FSI
        # Shape: [fsi_size, d1_size+d2_size] - FSI receive from all MSNs

        # D1 MSNs → FSI (inhibitory GABAergic collaterals, ~30% connectivity)
        # CONDUCTANCE-BASED: Weak MSN→FSI GABAergic connections (Dale's Law)
        self._add_internal_connection(
            source_population=StriatumPopulation.D1,
            target_population=StriatumPopulation.FSI,
            weights=WeightInitializer.sparse_random(
                n_input=self.d1_size,
                n_output=self.fsi_size,
                connectivity=0.3,
                weight_scale=0.0005,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=None,
        )

        # D2 MSNs → FSI (inhibitory GABAergic collaterals, ~30% connectivity)
        # CONDUCTANCE-BASED: Weak MSN→FSI GABAergic connections (matches D1, Dale's Law)
        self._add_internal_connection(
            source_population=StriatumPopulation.D2,
            target_population=StriatumPopulation.FSI,
            weights=WeightInitializer.sparse_random(
                n_input=self.d2_size,
                n_output=self.fsi_size,
                connectivity=0.3,
                weight_scale=0.0005,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=None,
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
        # weight_scale raised from 0.001 → 0.005 → 0.015: diagnostics showed E/I = 12.0 for
        # striatum with D1/D2 firing at 17 Hz; heavier transient IPSP needed to gate bursts.
        # STP: FSI_MSN_PRESET — strong depression (Planert et al. 2010).
        # Large initial IPSP fades rapidly; feedforward inhibition is transient.
        self._add_internal_connection(
            source_population=StriatumPopulation.FSI,
            target_population=StriatumPopulation.D1,
            weights=WeightInitializer.sparse_random(
                n_input=self.fsi_size,
                n_output=self.d1_size,
                connectivity=0.15,
                weight_scale=0.015,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=FSI_MSN_PRESET.configure(),
        )

        # FSI → D2 connections (same structure)
        # CONDUCTANCE-BASED: Strong inhibition (matches D1)
        # STP: FSI_MSN_PRESET — same strong depression as FSI→D1.
        self._add_internal_connection(
            source_population=StriatumPopulation.FSI,
            target_population=StriatumPopulation.D2,
            weights=WeightInitializer.sparse_random(
                n_input=self.fsi_size,
                n_output=self.d2_size,
                connectivity=0.15,
                weight_scale=0.015,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=FSI_MSN_PRESET.configure(),
        )

        # =====================================================================
        # MSN→MSN LATERAL INHIBITION
        # =====================================================================
        # D1 → D1: Lateral inhibition for action selection
        # MSN→MSN GABAergic collaterals create winner-take-all dynamics
        # Distance: ~100-300μm (local), unmyelinated → 1-2ms delay
        # Enables action-specific competition.
        self._add_internal_connection(
            source_population=StriatumPopulation.D1,
            target_population=StriatumPopulation.D1,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.d1_size,
                n_output=self.d1_size,
                connectivity=0.4,
                weight_scale=0.001,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=MSN_LATERAL_PRESET.configure(),
        )

        # D2 → D2: Lateral inhibition for NoGo pathway
        # Similar MSN→MSN collaterals in indirect pathway
        self._add_internal_connection(
            source_population=StriatumPopulation.D2,
            target_population=StriatumPopulation.D2,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.d2_size,
                n_output=self.d2_size,
                connectivity=0.4,
                weight_scale=0.001,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=MSN_LATERAL_PRESET.configure(),
        )

        # =====================================================================
        # MSN CROSS-PATHWAY LATERAL INHIBITION (Go/NoGo Competition)
        # =====================================================================
        # Biology: D1 and D2 MSNs inhibit each other via GABAergic axon collaterals,
        # creating the opponent Go/NoGo competition that underlies action selection.
        # Cross-pathway connectivity is sparser (~10%) than within-pathway (~40%)
        # reflecting the greater anatomical distance between D1/D2 MSN soma clusters.
        # Reference: Taverna et al. (2008) J. Neurosci.; Planert et al. (2010)
        #
        # D1 (Go) → D2 (NoGo): when Go pathway is activated it actively suppresses NoGo
        self._add_internal_connection(
            source_population=StriatumPopulation.D1,
            target_population=StriatumPopulation.D2,
            weights=WeightInitializer.sparse_random(
                n_input=self.d1_size,
                n_output=self.d2_size,
                connectivity=0.1,
                weight_scale=0.0005,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=MSN_LATERAL_PRESET.configure(),
        )

        # D2 (NoGo) → D1 (Go): when NoGo pathway is activated it suppresses Go
        self._add_internal_connection(
            source_population=StriatumPopulation.D2,
            target_population=StriatumPopulation.D1,
            weights=WeightInitializer.sparse_random(
                n_input=self.d2_size,
                n_output=self.d1_size,
                connectivity=0.1,
                weight_scale=0.0005,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=MSN_LATERAL_PRESET.configure(),
        )

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY (Adaptive Gain)
        # =====================================================================
        # EMA tracking of firing rates (per pathway).
        # Buffer names produced by _register_homeostasis are d1_firing_rate /
        # d2_firing_rate, matching the direct accesses in forward().
        # Striatum uses a custom joint D1+D2 update rather than _update_homeostasis.
        self._register_homeostasis(StriatumPopulation.D1, self.d1_size, target_firing_rate=0.008)
        self._register_homeostasis(StriatumPopulation.D2, self.d2_size, target_firing_rate=0.008)

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
        self._da_concentration_d1: torch.Tensor
        self._da_concentration_d2: torch.Tensor
        self.register_buffer("_da_concentration_d1", torch.zeros(self.d1_size, device=self.device))
        self.register_buffer("_da_concentration_d2", torch.zeros(self.d2_size, device=self.device))

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
        self._ne_concentration_d1: torch.Tensor
        self._ne_concentration_d2: torch.Tensor
        self.register_buffer("_ne_concentration_d1", torch.zeros(self.d1_size, device=self.device))
        self.register_buffer("_ne_concentration_d2", torch.zeros(self.d2_size, device=self.device))

        # =====================================================================
        # TAN (TONICALLY ACTIVE NEURONS) - Cholinergic Interneurons
        # =====================================================================
        # TANs are cholinergic interneurons (~1% of striatum), tonically active at ~5 Hz.
        # They pause briefly in response to salient stimuli (cue onset, reward) then burst.
        # TAN ACh inhibits MSNs via M2 muscarinic receptors (shunting inhibition on dendrites).
        # Key role: modulate corticostriatal plasticity window and action gating.
        #
        # NEURON TYPE: TANs use slow conductance params (NOT FAST_SPIKING).
        # FSI neurons (g_L=0.10, tau_mem=8ms) cannot fire below ~55 Hz above threshold.
        # TANs are large cholinergic neurons: tau_mem~80ms, g_L=0.04, tau_E=10ms.
        self.tan_size = max(1, int(total_msn_neurons * 0.01))  # ~1% of MSNs, minimum 1
        self.tan_neurons = ConductanceLIF(
            n_neurons=self.tan_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=StriatumPopulation.TAN,
                tau_mem=80.0,    # Long integration window (cholinergic neurons: ~80-100ms)
                g_L=0.04,        # Low leak for sustained depolarisation
                tau_E=10.0,      # Slow AMPA kinetics matching cortical/thalamic cholinergic drive
                tau_I=20.0,      # Slow GABA-A kinetics (pause inhibition)
                v_threshold=1.2, # Raised (1.0→1.2): thalamic overflow was driving 28 Hz; needs higher bar
                v_reset=0.0,
                tau_ref=5.0,     # Longer refractory for slow cholinergic neurons
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                noise_std=0.005, # Reduced (0.01→0.005) to lower noise-driven jitter
                adapt_increment=0.20,  # Added adaptation to resist thalamic overflow (28 Hz → target 5-10 Hz)
                tau_adapt=200.0,       # Slow AHP matching cholinergic neuron biophysics
            ),
            device=self.device,
        )
        # Tonic baseline drive for autonomous 5-10 Hz pacemaking.
        # TANs fire tonically without external drive (intrinsic pacemaking via I_h etc.),
        # approximated here as a constant excitatory conductance baseline.
        self._tan_baseline = torch.full(
            (self.tan_size,), self.config.tan_baseline_drive, device=self.device
        )

        # Tonic baseline drive for FSI; provides minimal sub-threshold depolarisation so
        # cortical/thalamic bursts can push FSI above threshold. AMPA-only (NMDA disabled).
        self._fsi_baseline = torch.full(
            (self.fsi_size,), self.config.fsi_baseline_drive, device=self.device
        )

        # TAN → D1 inhibition (M2 receptor-mediated, widespread cholinergic inhibition)
        self._add_internal_connection(
            source_population=StriatumPopulation.TAN,
            target_population=StriatumPopulation.D1,
            weights=WeightInitializer.sparse_random(
                n_input=self.tan_size,
                n_output=self.d1_size,
                connectivity=0.5,
                weight_scale=0.0005,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=None,
        )

        # TAN → D2 inhibition (M2 receptor-mediated, cholinergic gating of NoGo pathway)
        self._add_internal_connection(
            source_population=StriatumPopulation.TAN,
            target_population=StriatumPopulation.D2,
            weights=WeightInitializer.sparse_random(
                n_input=self.tan_size,
                n_output=self.d2_size,
                connectivity=0.5,
                weight_scale=0.0005,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=None,
        )

        # Register TAN population after neuron creation and its internal connections.
        # (D1, D2, FSI were registered earlier, before their first _add_internal_connection)
        self._register_neuron_population(StriatumPopulation.TAN, self.tan_neurons, polarity=PopulationPolarity.ANY)

        # =====================================================================
        # TAN ACh CONCENTRATION TRACKING (muscarinic timescale)
        # =====================================================================
        # TANs fire tonically at ~5 Hz, releasing ACh that tonically suppresses
        # corticostriatal LTP via M1/M4 MSN receptors.  During the TAN pause the
        # ACh concentration drops (tau_decay ~300 ms) and the plasticity window
        # opens in synchrony with the arriving DA burst.
        self.tan_ach_receptor = NeuromodulatorReceptor(
            n_receptors=self.tan_size,
            tau_rise_ms=5.0,
            tau_decay_ms=300.0,
            spike_amplitude=0.5,
            device=self.device,
        )
        self.register_buffer("_tan_ach_concentration", torch.zeros(self.tan_size, device=self.device))
        # Scalar inhibitory trace driving the TAN pause; decays with tau = 300 ms.
        self.register_buffer("_tan_pause_trace", torch.zeros(1, device=self.device))
        # Pre-compute per-step decay factor (updated by update_temporal_parameters).
        self._tan_pause_decay: float = math.exp(-GlobalConfig.DEFAULT_DT_MS / 300.0)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    # =========================================================================
    # FSI GAP JUNCTION INITIALIZATION
    # =========================================================================

    def _init_gap_junctions_fsi(self) -> None:
        """Lazily initialize FSI gap junctions from the first registered FSI weight matrix.

        Called on the first forward pass once FSI synaptic weights have been registered
        by brain_builder.connect_to_striatum().  Gap junction topology is derived from
        the afferent weight matrix so that anatomically nearby FSIs (similar input
        fingerprints) become electrically coupled neighbours.
        """
        fsi_weights = None
        for synapse_id, weights in self.synaptic_weights.items():
            if synapse_id.target_population == StriatumPopulation.FSI:
                fsi_weights = weights
                break

        if fsi_weights is None:
            return  # No FSI sources registered; gap junctions remain disabled

        gap_config_fsi = GapJunctionConfig(
            coupling_strength=self.config.gap_junction_strength,
            connectivity_threshold=self.config.gap_junction_threshold,
            max_neighbors=self.config.gap_junction_max_neighbors,
        )
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

        if not GlobalConfig.NEUROMODULATION_DISABLED:
            # =====================================================================
            # DOPAMINE RECEPTOR UPDATE (VTA Spikes → Concentration)
            # =====================================================================
            # Convert spiking DA from VTA to synaptic concentration for learning.
            vta_da_spikes = self._extract_neuromodulator(neuromodulator_inputs, 'da_mesolimbic')
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

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Each source (cortex:l5, hippocampus, thalamus) has separate weights
        # for D1 and D2 pathways. Filter inputs by target population.
        # Biology: D1 and D2 MSNs are distinct neurons with independent synaptic
        # weights, so integration must remain separate per pathway.
        d1_conductance = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, self.d1_size, filter_by_target_population=StriatumPopulation.D1
        ).g_ampa
        d2_conductance = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, self.d2_size, filter_by_target_population=StriatumPopulation.D2
        ).g_ampa

        # =====================================================================
        # FSI (FAST-SPIKING INTERNEURONS) - Feedforward Inhibition
        # =====================================================================
        # FSI process inputs in parallel with MSNs but with:
        # 1. Gap junction coupling for synchronization (<0.1ms)
        # 2. Feedforward inhibition to MSNs (sharpens action timing)
        # Biology: FSI are parvalbumin+ interneurons (~2% of striatum)

        # Lazy-initialize gap junctions on first forward pass (after weights have been registered)
        if self.gap_junctions_fsi is None:
            self._init_gap_junctions_fsi()

        fsi_conductance = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, self.fsi_size, filter_by_target_population=StriatumPopulation.FSI
        ).g_ampa

        # Apply gap junction coupling
        # IMPORTANT: gap_conductance is g_gap_total (row-sum of coupling matrix) and
        # gap_reversal is the voltage-weighted average of neighbours.  These must be
        # passed to ConductanceLIF.forward() as g_gap_input / E_gap_reversal so the
        # neuron model computes the correct current I_gap = g_gap × (E_gap - V).
        # Adding gap_conductance to the AMPA channel (old bug) treated it as a
        # constant excitatory drive with E_E=3.0 reversal, creating ~1.05 artificial
        # excitation that saturated FSI at 400+ Hz.
        fsi_gap_conductance: Optional[torch.Tensor] = None
        fsi_gap_reversal: Optional[GapJunctionReversal] = None
        if self.gap_junctions_fsi is not None:
            fsi_gap_conductance, fsi_gap_reversal = self.gap_junctions_fsi.forward(self.fsi_neurons.V_soma)

        # =====================================================================
        # MSN→FSI INHIBITION (GABAergic Lateral Collaterals)
        # =====================================================================
        # Biology: MSN axon collaterals release GABA onto FSI (Dale's Law —
        # MSNs are purely GABAergic).  Contrary to earlier comments, this
        # connection is INHIBITORY (GABA_A) not excitatory.  The net effect
        # on winner-take-all dynamics is via disinhibition: when the winning
        # action's MSNs fire, they suppress FSI activity, reducing
        # feedforward inhibition on themselves (a relief-from-inhibition
        # mechanism rather than the direct excitation previously described).
        #
        # CRITICAL: Use PREVIOUS timestep's MSN activity (causal)
        # FSI response from t-1 MSN activity influences t MSN spikes

        d1_fsi_synapse = SynapseId(
            source_region=self.region_name,
            source_population=StriatumPopulation.D1,
            target_region=self.region_name,
            target_population=StriatumPopulation.FSI,
            receptor_type=ReceptorType.GABA_A,
        )
        d2_fsi_synapse = SynapseId(
            source_region=self.region_name,
            source_population=StriatumPopulation.D2,
            target_region=self.region_name,
            target_population=StriatumPopulation.FSI,
            receptor_type=ReceptorType.GABA_A,
        )

        # Add MSN→FSI inhibition (causal: t-1 MSN spikes via spike buffers)
        # Route to a SEPARATE inhibitory conductance accumulator so it
        # can be passed to g_gaba_a_input (not mixed with excitatory AMPA).
        fsi_gaba_a_conductance = self._integrate_synaptic_inputs_at_dendrites(
            {
                d1_fsi_synapse: self._d1_spike_buffer.read(1),
                d2_fsi_synapse: self._d2_spike_buffer.read(1),
            },
            n_neurons=self.fsi_size,
        ).g_gaba_a

        # Update FSI neurons (fast kinetics, tau_mem ~5ms)
        # FSI are PV+ fast-spiking: AMPA-dominated, no meaningful NMDA contribution.
        # nmda_ratio=0.2 with tau_nmda=100ms gives g_NMDA_ss = input × 100 — a runaway
        # attractor that saturates FSI at 400+ Hz.  Use purely AMPA drive (nmda_ratio=0.0).
        fsi_g_ampa, fsi_g_nmda = split_excitatory_conductance(fsi_conductance, nmda_ratio=0.0)
        fsi_g_ampa = fsi_g_ampa + self._fsi_baseline  # Tonic sub-threshold seed
        fsi_spikes, fsi_membrane = self.fsi_neurons.forward(
            g_ampa_input=ConductanceTensor(fsi_g_ampa),
            g_nmda_input=ConductanceTensor(fsi_g_nmda),
            g_gaba_a_input=ConductanceTensor(fsi_gaba_a_conductance),
            g_gaba_b_input=None,
            g_gap_input=fsi_gap_conductance,
            E_gap_reversal=fsi_gap_reversal,
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
            source_population=StriatumPopulation.FSI,
            target_region=self.region_name,
            target_population=StriatumPopulation.D1,
            receptor_type=ReceptorType.GABA_A,
        )
        fsi_d2_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=StriatumPopulation.FSI,
            target_region=self.region_name,
            target_population=StriatumPopulation.D2,
            receptor_type=ReceptorType.GABA_A,
        )
        fsi_d1_inhib_weights = self.get_synaptic_weights(fsi_d1_inhib_synapse)
        fsi_d2_inhib_weights = self.get_synaptic_weights(fsi_d2_inhib_synapse)

        fsi_inhibition_d1 = (fsi_d1_inhib_weights @ fsi_spikes_float) * avg_inhibition_scale
        fsi_inhibition_d2 = (fsi_d2_inhib_weights @ fsi_spikes_float) * avg_inhibition_scale

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
            source_population=StriatumPopulation.D1,
            target_region=self.region_name,
            target_population=StriatumPopulation.D1,
            receptor_type=ReceptorType.GABA_A,
        )
        d2_d2_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=StriatumPopulation.D2,
            target_region=self.region_name,
            target_population=StriatumPopulation.D2,
            receptor_type=ReceptorType.GABA_A,
        )
        d1_d1_inhibition = self._integrate_synaptic_inputs_at_dendrites(
            {d1_d1_inhib_synapse: self._d1_spike_buffer.read(1)},
            n_neurons=self.d1_size,
        ).g_gaba_a
        d2_d2_inhibition = self._integrate_synaptic_inputs_at_dendrites(
            {d2_d2_inhib_synapse: self._d2_spike_buffer.read(1)},
            n_neurons=self.d2_size,
        ).g_gaba_a

        # =====================================================================
        # D1 ↔ D2 CROSS-PATHWAY LATERAL INHIBITION (Go/NoGo Competition)
        # =====================================================================
        # Biology: D1 and D2 MSNs mutually inhibit each other via sparse GABAergic
        # collaterals, creating the opponent process that drives action selection.
        # This is the key circuit mechanism for Go/NoGo gating (Taverna et al. 2008).
        d1_d2_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=StriatumPopulation.D1,
            target_region=self.region_name,
            target_population=StriatumPopulation.D2,
            receptor_type=ReceptorType.GABA_A,
        )
        d2_d1_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=StriatumPopulation.D2,
            target_region=self.region_name,
            target_population=StriatumPopulation.D1,
            receptor_type=ReceptorType.GABA_A,
        )
        # D1 spikes → inhibit D2 (Go suppresses NoGo)
        d1_d2_inhibition = self._integrate_synaptic_inputs_at_dendrites(
            {d1_d2_inhib_synapse: self._d1_spike_buffer.read(1)},
            n_neurons=self.d2_size,
        ).g_gaba_a
        # D2 spikes → inhibit D1 (NoGo suppresses Go)
        d2_d1_inhibition = self._integrate_synaptic_inputs_at_dendrites(
            {d2_d1_inhib_synapse: self._d2_spike_buffer.read(1)},
            n_neurons=self.d1_size,
        ).g_gaba_a

        # =====================================================================
        # TAN (TONICALLY ACTIVE NEURONS) - Cholinergic Inhibition of MSNs
        # =====================================================================
        # Biology: TANs receive cortical and thalamic drive, fire tonically ~5 Hz.
        # TAN ACh → M2 receptors on MSN dendrites → shunting inhibition.
        # Pause-burst response: TANs pause at CS onset → disinhibit MSNs (enable learning);
        # then burst → inhibit MSNs (terminate plasticity window).
        tan_conductance = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, self.tan_size, filter_by_target_population=StriatumPopulation.TAN
        ).g_ampa

        # S3-4 — TAN PAUSE DETECTION
        # Biology: Coincident corticostriatal + thalamostriatal bursts trigger a
        # ~300 ms silence in TAN firing, mediated by mAChR autoreceptors (M2/M4)
        # and GABAergic input from the same afferent burst (Aosaki et al. 1994).
        # Approximation: detect when mean TAN afferent conductance exceeds the
        # burst threshold; drive a slow inhibitory trace (tau = 300 ms) that
        # adds g_gaba_a to TANs and suppresses tonic firing during the pause.
        tan_burst = (tan_conductance.mean() > cfg.tan_pause_threshold).float()
        self._tan_pause_trace = (
            self._tan_pause_trace * self._tan_pause_decay
            + tan_burst * (1.0 - self._tan_pause_decay)
        )
        # Expand scalar pause trace to a per-neuron inhibitory conductance tensor
        tan_g_pause = self._tan_pause_trace.expand(self.tan_size) * cfg.tan_pause_strength

        # Add tonic baseline + synaptic AMPA (NMDA omitted: 100ms tau creates bistability
        # at sub-threshold V due to Mg2+ block; AMPA-only stable for slow pacemaking)
        tan_total_exc = self._tan_baseline.clone() + tan_conductance
        tan_spikes, _ = self.tan_neurons.forward(
            g_ampa_input=ConductanceTensor(tan_total_exc),
            g_nmda_input=None,
            g_gaba_a_input=ConductanceTensor(tan_g_pause),  # pause-driven inhibition
            g_gaba_b_input=None,
        )

        # S3-3 / S3-5 — Track TAN ACh concentration for plasticity gating.
        # High TAN firing → high [ACh] → M1/M4 suppresses corticostriatal LTP.
        # TAN pause → [ACh] drops (tau ~300 ms) → plasticity window opens.
        self._tan_ach_concentration = self.tan_ach_receptor.update(tan_spikes)

        # TAN → D1 inhibition (M2-mediated cholinergic shunting)
        tan_d1_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=StriatumPopulation.TAN,
            target_region=self.region_name,
            target_population=StriatumPopulation.D1,
            receptor_type=ReceptorType.GABA_A,
        )
        # TAN → D2 inhibition
        tan_d2_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=StriatumPopulation.TAN,
            target_region=self.region_name,
            target_population=StriatumPopulation.D2,
            receptor_type=ReceptorType.GABA_A,
        )
        tan_d1_weights = self.get_synaptic_weights(tan_d1_inhib_synapse)
        tan_d2_weights = self.get_synaptic_weights(tan_d2_inhib_synapse)
        tan_inhibition_d1 = (tan_d1_weights @ tan_spikes.float()).clamp(min=0)
        tan_inhibition_d2 = (tan_d2_weights @ tan_spikes.float()).clamp(min=0)

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
        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        # nmda_ratio reduced 0.2→0.10: high NMDA was accumulating (tau=100ms) and driving MSN bistability issues
        d1_g_ampa, d1_g_nmda = split_excitatory_conductance(d1_conductance, nmda_ratio=0.10)
        d2_g_ampa, d2_g_nmda = split_excitatory_conductance(d2_conductance, nmda_ratio=0.10)

        # CONDUCTANCE-BASED: Inhibition goes to g_gaba_a_input, NOT mixed with excitation
        # Combine all inhibitory sources (FSI + within-pathway + cross-pathway + TAN)
        # All are POSITIVE conductances
        d1_inhibition = (fsi_inhibition_d1 + d1_d1_inhibition + d2_d1_inhibition + tan_inhibition_d1).clamp(min=0)
        d2_inhibition = (fsi_inhibition_d2 + d2_d2_inhibition + d1_d2_inhibition + tan_inhibition_d2).clamp(min=0)

        # Execute D1 and D2 MSN populations
        d1_spikes, _ = self.d1_neurons.forward(
            g_ampa_input=ConductanceTensor(d1_g_ampa),
            g_nmda_input=ConductanceTensor(d1_g_nmda),
            g_gaba_a_input=ConductanceTensor(d1_inhibition),
            g_gaba_b_input=None,
        )
        d2_spikes, _ = self.d2_neurons.forward(
            g_ampa_input=ConductanceTensor(d2_g_ampa),
            g_nmda_input=ConductanceTensor(d2_g_nmda),
            g_gaba_a_input=ConductanceTensor(d2_inhibition),
            g_gaba_b_input=None,
        )

        if not GlobalConfig.HOMEOSTASIS_DISABLED:
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
            combined_rate_error = self._get_target_firing_rate(StriatumPopulation.D1) - combined_rate

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
        region_outputs: RegionOutput = {
            StriatumPopulation.D1: d1_spikes,
            StriatumPopulation.D2: d2_spikes,
            StriatumPopulation.TAN: tan_spikes,  # Include TAN spikes so NeuromodulatorHub can broadcast 'ach_striatal'.
            StriatumPopulation.FSI: fsi_spikes,  # Include FSI spikes for diagnostics and downstream tracts.
        }

        if not GlobalConfig.LEARNING_DISABLED:
            # Lazily register strategies for D1/D2 targets before dispatching
            for synapse_id in synaptic_inputs:
                if self.get_learning_strategy(synapse_id) is None:
                    strategy_class = None
                    if synapse_id.target_population == StriatumPopulation.D1:
                        strategy_class = D1STDPStrategy
                    elif synapse_id.target_population == StriatumPopulation.D2:
                        strategy_class = D2STDPStrategy

                    if strategy_class is not None:
                        strategy = strategy_class(D1D2STDPConfig(
                            learning_rate=cfg.learning_rate,
                            fast_eligibility_tau_ms=cfg.fast_eligibility_tau_ms,
                            slow_eligibility_tau_ms=cfg.slow_eligibility_tau_ms,
                            eligibility_consolidation_rate=cfg.eligibility_consolidation_rate,
                            slow_trace_weight=cfg.slow_trace_weight,
                        ))
                        self._add_learning_strategy(synapse_id, strategy)

        # =====================================================================
        # UPDATE STATE BUFFERS FOR NEXT TIMESTEP
        # =====================================================================
        # Write current spikes to all state buffers
        # Next forward pass will read these with delay=1 for previous timestep
        self._d1_spike_buffer.write_and_advance(d1_spikes)
        self._d2_spike_buffer.write_and_advance(d2_spikes)

        return self._post_forward(region_outputs)

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        d1_da = self._da_concentration_d1.mean().item()
        d2_da = self._da_concentration_d2.mean().item()
        tan_gate = 1.0 - self._tan_ach_concentration.mean().item()
        if synapse_id.target_population == StriatumPopulation.D1:
            return {"dopamine": d1_da, "acetylcholine": tan_gate}
        if synapse_id.target_population == StriatumPopulation.D2:
            return {"dopamine": d2_da, "acetylcholine": tan_gate}
        return {}

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
