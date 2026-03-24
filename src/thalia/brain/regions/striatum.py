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

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import StriatumConfig
from thalia.brain.gap_junctions import (
    GapJunctionConfig,
    GapJunctionCoupling,
)
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    heterogeneous_dendrite_coupling,
    heterogeneous_noise_std,
    heterogeneous_tau_adapt,
    heterogeneous_tau_mem,
    heterogeneous_v_reset,
    heterogeneous_v_threshold,
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    split_excitatory_conductance,
)
from thalia.brain.synapses import (
    NeuromodulatorReceptor,
    STPConfig,
    WeightInitializer,
)
from thalia.learning import (
    D1D2STDPConfig,
    D1STDPStrategy,
    D2STDPStrategy,
    InhibitorySTDPConfig,
    InhibitorySTDPStrategy,
    MetaplasticityConfig,
    MetaplasticityStrategy,
)
from thalia.typing import (
    ConductanceTensor,
    GapJunctionReversal,
    NeuromodulatorInput,
    NeuromodulatorChannel,
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
    compute_da_gain,
    compute_ne_gain,
    decay_float,
)

from .neural_region import NeuralRegion
from .population_names import StriatumPopulation
from .region_registry import register_region

if TYPE_CHECKING:
    from thalia.brain.neurons import ConductanceLIF


@register_region(
    "striatum",
    aliases=["basal_ganglia"],
    description="Reinforcement learning via dopamine-modulated three-factor rule with D1/D2 opponent pathways",
)
class Striatum(NeuralRegion[StriatumConfig]):
    """Striatal region with three-factor reinforcement learning.

    Implements dopamine-modulated learning:
    - Eligibility traces tag recently active synapses
    - Dopamine signal converts eligibility to plasticity
    - No learning without dopamine (unlike Hebbian)
    - Synaptic weights stored per-source in synaptic_weights dict
    """

    # Mesolimbic DA (VTA → ventral striatum) and nigrostriatal DA (SNc → dorsal
    # striatum) jointly drive D1/D2 opponent learning.  Both converge on the same
    # D1/D2 receptor populations — biology has one striatal DA tone.
    # NE from LC modulates the gain of burst responses and threshold adaptation.
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.DA_MESOLIMBIC,
        NeuromodulatorChannel.DA_NIGROSTRIATAL,
        NeuromodulatorChannel.NE,
        NeuromodulatorChannel.SHT,
        NeuromodulatorChannel.ACH,
    ]

    # Striatum publishes local ACh from TANs (cholinergic interneurons) on a
    # dedicated 'ach_striatal' channel so downstream circuits (e.g., SNc, VTA DA
    # terminals) can detect striatal ACh tone independently from cortical NB ACh.
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorChannel, PopulationName]] = {
        NeuromodulatorChannel.ACH_STRIATAL: StriatumPopulation.TAN,
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: StriatumConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize Striatum with D1/D2 opponent pathways."""
        super().__init__(config, population_sizes, region_name, device=device)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.d1_size = population_sizes[StriatumPopulation.D1]
        self.d2_size = population_sizes[StriatumPopulation.D2]

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
        self.d1_neurons: ConductanceLIF
        self.d1_neurons = self._create_and_register_neuron_population(
            population_name=StriatumPopulation.D1,
            n_neurons=self.d1_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(20.0, self.d1_size, device),
                v_reset=heterogeneous_v_reset(-0.10, self.d1_size, device),
                v_threshold=heterogeneous_v_threshold(1.02, self.d1_size, device),
                tau_ref=2.0,
                g_L=heterogeneous_g_L(0.05, self.d1_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.025, self.d1_size, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(200.0, self.d1_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.18, self.d1_size, device),
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.d1_size, device, cv=0.25),
            ),
        )

        self.d2_neurons: ConductanceLIF
        self.d2_neurons = self._create_and_register_neuron_population(
            population_name=StriatumPopulation.D2,
            n_neurons=self.d2_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(20.0, self.d2_size, device),
                v_reset=heterogeneous_v_reset(-0.10, self.d2_size, device),
                v_threshold=heterogeneous_v_threshold(1.02, self.d2_size, device),
                tau_ref=2.0,
                g_L=heterogeneous_g_L(0.05, self.d2_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.025, self.d2_size, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(200.0, self.d2_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.24, self.d2_size, device),
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.d2_size, device, cv=0.25),
            ),
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
        self.fsi_neurons: ConductanceLIF
        self.fsi_neurons = self._create_and_register_neuron_population(
            population_name=StriatumPopulation.FSI,
            n_neurons=self.fsi_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(8.0, self.fsi_size, device=device, cv=0.10),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(1.15, self.fsi_size, device, cv=0.06),
                tau_ref=2.5,
                g_L=heterogeneous_g_L(0.10, self.fsi_size, device, cv=0.08),
                E_E=3.0,
                E_I=-0.5,
                tau_E=3.0,
                tau_I=3.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, self.fsi_size, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=50.0,
                adapt_increment=0.0,  # PV/FSI are non-adapting (Kv3 channels)
                E_adapt=-0.5,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.fsi_size, device, cv=0.20),
            ),
        )

        # Lazily initialized on first forward pass
        self.gap_junctions_fsi: Optional[GapJunctionCoupling] = None

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
        # Depressing — same STP as MSN lateral collaterals (D1↔D2).
        self._add_internal_connection(
            source_population=StriatumPopulation.D1,
            target_population=StriatumPopulation.FSI,
            weights=WeightInitializer.sparse_random(
                n_input=self.d1_size,
                n_output=self.fsi_size,
                connectivity=0.3,
                weight_scale=0.0005,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.35, tau_d=600.0, tau_f=20.0),
        )

        # D2 MSNs → FSI (inhibitory GABAergic collaterals, ~30% connectivity)
        # CONDUCTANCE-BASED: Weak MSN→FSI GABAergic connections (matches D1, Dale's Law)
        # Depressing — same STP as MSN lateral collaterals (D1↔D2).
        self._add_internal_connection(
            source_population=StriatumPopulation.D2,
            target_population=StriatumPopulation.FSI,
            weights=WeightInitializer.sparse_random(
                n_input=self.d2_size,
                n_output=self.fsi_size,
                connectivity=0.3,
                weight_scale=0.0005,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.35, tau_d=600.0, tau_f=20.0),
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

        # Inhibitory STDP for FSI→MSN synapses (Vogels et al. 2011)
        istdp_cfg = InhibitorySTDPConfig(
            learning_rate=config.istdp_learning_rate,
            tau_istdp=config.istdp_tau_ms,
            alpha=config.istdp_alpha,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        )
        self.istdp_fsi_d1: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_fsi_d2: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)

        # FSI → D1 connections
        # Raised connectivity 0.15→0.30: each MSN should receive from more FSIs
        # to provide stronger decorrelation. At 15% with N_FSI=6-10, many MSNs
        # received input from 0-1 FSI — insufficient for population-level decorrelation.
        # 30% ensures each MSN receives from ~3-5 FSIs (Koós & Tepper 1999: ~18 FSI per MSN).
        self._add_internal_connection(
            source_population=StriatumPopulation.FSI,
            target_population=StriatumPopulation.D1,
            weights=WeightInitializer.sparse_random(
                n_input=self.fsi_size,
                n_output=self.d1_size,
                connectivity=0.35,  # Raised 0.15→0.30→0.35: asymmetric with FSI→D2 (0.25).
                                     # More FSI inhibition on D1 (Go) pathway creates bias toward NoGo
                                     # at rest — biologically correct (D2/indirect activity dominates
                                     # in absence of DA-gated D1 facilitation). Asymmetry also decorrelates
                                     # D1 and D2 firing patterns (different FSI subsets).
                weight_scale=0.010,  # Reduced 0.03→0.015→0.010: FSI at 52 Hz with 0.015 still created
                                     # too-strong GABA blanket → D1 at 0.6 Hz (target 2.0 Hz) with gain
                                     # plateau reached. Further halving allows D1 to fire while
                                     # maintaining feedforward inhibition structure.
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=250.0, tau_f=15.0),
            learning_strategy=self.istdp_fsi_d1,
        )

        # FSI → D2 connections (same structure)
        self._add_internal_connection(
            source_population=StriatumPopulation.FSI,
            target_population=StriatumPopulation.D2,
            weights=WeightInitializer.sparse_random(
                n_input=self.fsi_size,
                n_output=self.d2_size,
                connectivity=0.25,   # Raised 0.15→0.30→0.25: asymmetric with FSI→D1 (0.35).
                                     # Less FSI inhibition on D2 (NoGo) pathway creates Go/NoGo imbalance
                                     # that improves D1/D2 decorrelation (CI was 0.62).
                weight_scale=0.010,  # Reduced 0.03→0.015→0.010: matched to FSI→D1 reduction
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=250.0, tau_f=15.0),
            learning_strategy=self.istdp_fsi_d2,
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
                weight_scale=0.012,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.35, tau_d=600.0, tau_f=20.0),
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
                weight_scale=0.012,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.35, tau_d=600.0, tau_f=20.0),
        )

        # =====================================================================
        # MSN CROSS-PATHWAY LATERAL INHIBITION (Go/NoGo Competition)
        # =====================================================================
        # Biology: D1 and D2 MSNs inhibit each other via GABAergic axon collaterals,
        # creating the opponent Go/NoGo competition that underlies action selection.
        # Cross-pathway connectivity is sparser (~10%) than within-pathway (~40%)
        # reflecting the greater anatomical distance between D1/D2 MSN soma clusters.

        # D1 (Go) → D2 (NoGo): when Go pathway is activated it actively suppresses NoGo
        # Weight raised 0.08→0.15: CI=0.46 still above target <0.3. Primary cause is the
        # global alpha oscillation driving both D1 and D2 synchronously from cortex. Alpha
        # fix (BG/thalamus) should reduce CI substantially; stronger cross-pathway inhibition
        # provides additional decorrelation. STP: U=0.35→0.25/tau_d=600→300: less depletion
        # at 3 Hz MSN rate → more sustained cross-pathway inhibition.
        self._add_internal_connection(
            source_population=StriatumPopulation.D1,
            target_population=StriatumPopulation.D2,
            weights=WeightInitializer.sparse_random(
                n_input=self.d1_size,
                n_output=self.d2_size,
                connectivity=0.45,
                weight_scale=0.10,  # Reduced 0.15→0.10: at MSN rates 0.4-0.8 Hz, very strong cross-pathway
                                    # inhibition (0.15) couldn't decorrelate because shared cortical input
                                    # dominates. Moderate weight preserves Go/NoGo competition.
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=300.0, tau_f=20.0),
        )

        # D2 (NoGo) → D1 (Go): when NoGo pathway is activated it suppresses Go
        self._add_internal_connection(
            source_population=StriatumPopulation.D2,
            target_population=StriatumPopulation.D1,
            weights=WeightInitializer.sparse_random(
                n_input=self.d2_size,
                n_output=self.d1_size,
                connectivity=0.45,
                weight_scale=0.10,  # Reduced 0.15→0.10: matches D1→D2 (see above)
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=300.0, tau_f=20.0),
        )

        # =====================================================================
        # DOPAMINE RECEPTORS (Spiking DA from VTA)
        # =====================================================================
        # Convert dopamine neuron spikes to synaptic concentration.
        # Biology:
        # - D1 receptors: Gs-coupled → increase cAMP → facilitate LTP
        # - D2 receptors: Gi-coupled → decrease cAMP → facilitate LTD
        # - DA rise time: ~10-20 ms (fast release)
        # - DA decay time: ~200 ms (slow DAT reuptake)
        # Both pathways receive same DA spikes, but receptors have opposite effects.
        # VTA (mesolimbic) and SNc (nigrostriatal) both converge on D1/D2 MSNs;
        # tracked separately so we can sum them into a unified DA tone.

        # DA D1/D2 (VTA + SNc), NE α1 (LC), 5-HT2A (DRN), ACh M1 (NB) on D1/D2 MSNs
        self._init_receptors_from_config(device)

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
        self.tan_neurons: ConductanceLIF
        self.tan_neurons = self._create_and_register_neuron_population(
            population_name=StriatumPopulation.TAN,
            n_neurons=self.tan_size,
            polarity=PopulationPolarity.ANY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(80.0, self.tan_size, device, cv=0.20),
                v_reset=0.0,
                # Swept via auto_calibrate: optimal at v_threshold=1.70 with
                # baseline_drive=0.005 gives 7.4 Hz in isolated region test.
                # Sensitivity: -33 Hz per unit v_threshold.
                v_threshold=heterogeneous_v_threshold(1.70, self.tan_size, device, cv=0.15, clamp_fraction=0.25),
                tau_ref=5.0,
                g_L=heterogeneous_g_L(0.04, self.tan_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=10.0,
                tau_I=20.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.006, self.tan_size, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=200.0,
                adapt_increment=0.0,  # TAN pause response is synaptically driven, not intrinsic adaptation
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.tan_size, device, cv=0.25),
            ),
        )

        # Tonic baseline drive for autonomous 5-10 Hz pacemaking.
        # TANs fire tonically without external drive (intrinsic pacemaking via I_h etc.),
        # approximated here as a constant excitatory conductance baseline.
        self._tan_baseline: torch.Tensor
        self.register_buffer("_tan_baseline", torch.full((self.tan_size,), self.config.tan_baseline_drive, device=device))

        # Tonic baseline drive for FSI; provides minimal sub-threshold depolarisation so
        # cortical/thalamic bursts can push FSI above threshold. AMPA-only (NMDA disabled).
        self._fsi_baseline: torch.Tensor
        self.register_buffer("_fsi_baseline", torch.full((self.fsi_size,), self.config.fsi_baseline_drive, device=device))

        # TAN → D1 inhibition (M2 receptor-mediated, widespread cholinergic inhibition)
        self._add_internal_connection(
            source_population=StriatumPopulation.TAN,
            target_population=StriatumPopulation.D1,
            weights=WeightInitializer.sparse_random(
                n_input=self.tan_size,
                n_output=self.d1_size,
                connectivity=0.5,
                weight_scale=0.0005,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=400.0, tau_f=20.0),
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
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=400.0, tau_f=20.0),
        )

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
            dt_ms=self.dt_ms,
            device=device,
        )
        self.register_buffer("_tan_ach_concentration", torch.zeros(self.tan_size, device=device))
        # Scalar inhibitory trace driving the TAN pause; decays with tau = 300 ms.
        self.register_buffer("_tan_pause_trace", torch.zeros(1, device=device))
        # Pre-compute per-step decay factor (updated by update_temporal_parameters).
        self._tan_pause_decay: float = decay_float(GlobalConfig.DEFAULT_DT_MS, 300.0)

        # Metaplasticity config for D1/D2 corticostriatal synapses.
        # Habitual behaviours resist change; striatal action-value memories
        # consolidate through repeated reinforcement (Yin & Knowlton 2006).
        self._meta_config = MetaplasticityConfig(
            tau_recovery_ms=5000.0,
            depression_strength=5.0,
            tau_consolidation_ms=300000.0,
            consolidation_sensitivity=0.1,
            rate_min=0.1,
        )

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # FSI GAP JUNCTION INITIALIZATION
    # =========================================================================

    def _init_gap_junctions_fsi(self, device: Union[str, torch.device]) -> None:
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
            coupling_strength=self.config.gap_junctions.coupling_strength,
            connectivity_threshold=self.config.gap_junctions.connectivity_threshold,
            max_neighbors=self.config.gap_junctions.max_neighbors,
        )
        self.gap_junctions_fsi = GapJunctionCoupling(
            n_neurons=self.fsi_size,
            afferent_weights=fsi_weights,
            config=gap_config_fsi,
            device=device,
        )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Process input and select action using separate D1/D2 populations."""
        device = self.device
        config = self.config

        # =================================================================
        # NEUROMODULATOR RECEPTOR UPDATES
        # =================================================================
        self._update_receptors(neuromodulator_inputs)

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
            self._init_gap_junctions_fsi(device=device)

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

        # Add MSN→FSI inhibition
        # Route to a SEPARATE inhibitory conductance accumulator so it
        # can be passed to g_gaba_a_input (not mixed with excitatory AMPA).
        prev_d1 = self._prev_spikes(StriatumPopulation.D1)
        prev_d2 = self._prev_spikes(StriatumPopulation.D2)
        fsi_gaba_a_conductance = self._integrate_synaptic_inputs_at_dendrites(
            {
                d1_fsi_synapse: prev_d1,
                d2_fsi_synapse: prev_d2,
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
        d1_d1_inhibition = self._integrate_single_synaptic_input(d1_d1_inhib_synapse, prev_d1).g_gaba_a
        d2_d2_inhibition = self._integrate_single_synaptic_input(d2_d2_inhib_synapse, prev_d2).g_gaba_a

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
        d1_d2_inhibition = self._integrate_single_synaptic_input(d1_d2_inhib_synapse, prev_d1).g_gaba_a
        # D2 spikes → inhibit D1 (NoGo suppresses Go)
        d2_d1_inhibition = self._integrate_single_synaptic_input(d2_d1_inhib_synapse, prev_d2).g_gaba_a

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
        tan_burst = (tan_conductance.mean() > config.tan_pause_threshold).float()
        self._tan_pause_trace = (
            self._tan_pause_trace * self._tan_pause_decay
            + tan_burst * (1.0 - self._tan_pause_decay)
        )
        # Expand scalar pause trace to a per-neuron inhibitory conductance tensor
        tan_g_pause = self._tan_pause_trace.expand(self.tan_size) * config.tan_pause_strength

        # Add tonic baseline + synaptic AMPA (NMDA omitted: 100ms tau creates bistability
        # at sub-threshold V due to Mg2+ block; AMPA-only stable for slow pacemaking)
        tan_total_exc = self._tan_baseline.clone() + tan_conductance
        # DA-mediated TAN pause via D2 autoreceptors (Straub et al. 2014;
        # Aosaki et al. 1994). Phasic DA burst activates striatal D2Rs on TANs,
        # coupling to GIRK channels → slow K⁺ outward current, approximated here as
        # a GABA_B-like conductance proportional to excess DA above threshold.
        # This creates the coincidence gate: DA burst + TAN pause occur simultaneously,
        # opening the plasticity window for corticostriatal LTP.
        tan_g_gaba_b: Optional[torch.Tensor] = None
        if not GlobalConfig.NEUROMODULATION_DISABLED:
            da_mean = (self._da_mesolimbic_d2 + self._da_nigrostriatal_d2).mean().item()
            excess_da = max(0.0, da_mean - config.tan_da2_threshold)
            if excess_da > 0.0:
                tan_g_gaba_b = torch.full(
                    (self.tan_size,), excess_da * config.tan_da2_pause_strength, device=device
                )

        tan_spikes, _ = self.tan_neurons.forward(
            g_ampa_input=ConductanceTensor(tan_total_exc),
            g_nmda_input=None,
            g_gaba_a_input=ConductanceTensor(tan_g_pause),  # pause-driven inhibition
            g_gaba_b_input=ConductanceTensor(tan_g_gaba_b) if tan_g_gaba_b is not None else None,
        )
        tan_spikes_float = tan_spikes.float()

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
        tan_inhibition_d1 = (tan_d1_weights @ tan_spikes_float).clamp(min=0)
        tan_inhibition_d2 = (tan_d2_weights @ tan_spikes_float).clamp(min=0)

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
        # Combined mesolimbic (VTA) + nigrostriatal (SNc) DA tone:
        d1_da_total = self._da_mesolimbic_d1 + self._da_nigrostriatal_d1
        d2_da_total = self._da_mesolimbic_d2 + self._da_nigrostriatal_d2
        d1_da_gain = compute_da_gain(d1_da_total, da_factor=0.3)
        d2_da_gain = compute_da_gain(d2_da_total, da_factor=-0.2)

        # NE gain modulation (average across neurons)
        d1_ne_gain = compute_ne_gain(self._ne_concentration_d1.mean().item())
        d2_ne_gain = compute_ne_gain(self._ne_concentration_d2.mean().item())

        # NB ACh M1 gain modulation (attention-driven excitability boost)
        # High NB ACh → enhanced corticostriatal transmission (+20% at max ACh)
        d1_ach_gain = 1.0 + 0.2 * self._nb_ach_concentration_d1.mean().item()
        d2_ach_gain = 1.0 + 0.2 * self._nb_ach_concentration_d2.mean().item()

        d1_conductance = d1_conductance * d1_da_gain * d1_ne_gain * d1_ach_gain
        d2_conductance = d2_conductance * d2_da_gain * d2_ne_gain * d2_ach_gain

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
            g_L_update = -config.homeostatic_gain.lr_per_ms * combined_rate_error
            self.d1_neurons.g_L_scale.data.add_(g_L_update).clamp_(min=0.1, max=2.0)
            self.d2_neurons.g_L_scale.data.add_(g_L_update).clamp_(min=0.1, max=2.0)

            # Adaptive threshold update (complementary to g_L modulation)
            # Also use combined error to maintain balance
            # Adjust thresholds based on combined activity, not independently
            # Lower threshold when underactive, raise when overactive
            threshold_update = -config.homeostatic_threshold.lr_per_ms * combined_rate_error
            self.d1_neurons.adjust_thresholds(threshold_update, config.homeostatic_threshold.threshold_min, config.homeostatic_threshold.threshold_max)
            self.d2_neurons.adjust_thresholds(threshold_update, config.homeostatic_threshold.threshold_min, config.homeostatic_threshold.threshold_max)

            # SYNAPTIC SCALING: Multiplicative upscaling of afferent weights targeting
            # chronically silent MSN populations (Turrigiano & Nelson 2004).
            self._synaptic_scaling_step += 1
            if self._synaptic_scaling_step >= config.synaptic_scaling.interval_steps:
                self._synaptic_scaling_step = 0
                self._apply_synaptic_scaling(StriatumPopulation.D1)
                self._apply_synaptic_scaling(StriatumPopulation.D2)

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
            # Lazily register strategies for D1/D2 targets before dispatching.
            # Each synapse gets its own MetaplasticityStrategy so that per-synapse
            # consolidation/rate buffers match the connection shape.
            for synapse_id in synaptic_inputs:
                if self.get_learning_strategy(synapse_id) is None:
                    strategy_class = None
                    if synapse_id.target_population == StriatumPopulation.D1:
                        strategy_class = D1STDPStrategy
                    elif synapse_id.target_population == StriatumPopulation.D2:
                        strategy_class = D2STDPStrategy

                    if strategy_class is not None:
                        base = strategy_class(D1D2STDPConfig(
                            learning_rate=config.learning_rate,
                            fast_eligibility_tau_ms=config.fast_eligibility_tau_ms,
                            slow_eligibility_tau_ms=config.slow_eligibility_tau_ms,
                            eligibility_consolidation_rate=config.eligibility_consolidation_rate,
                            slow_trace_weight=config.slow_trace_weight,
                        ))
                        strategy = MetaplasticityStrategy(
                            base_strategy=base,
                            config=self._meta_config,
                        )
                        self._add_learning_strategy(synapse_id, strategy, device=device)

        return region_outputs

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        d1_da = (self._da_mesolimbic_d1 + self._da_nigrostriatal_d1).mean().item()
        d2_da = (self._da_mesolimbic_d2 + self._da_nigrostriatal_d2).mean().item()
        tan_gate = 1.0 - self._tan_ach_concentration.mean().item()
        if synapse_id.target_population == StriatumPopulation.D1:
            # 5-HT2A: high serotonin attenuates DA-gated D1 plasticity (patience effect).
            # Implementation: reduce the effective dopamine signal seen by the learning rule.
            sht_d1 = self._sht_concentration_d1.mean().item()
            patience_gate = 1.0 - 0.5 * sht_d1  # range [0.5, 1.0]
            return {"dopamine": d1_da * patience_gate, "acetylcholine": tan_gate}
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
