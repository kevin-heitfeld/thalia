"""Cortical circuit connections preset.

Wires the three major cortical circuit groups:

Thalamocortical
    Bidirectional thalamus ↔ sensory cortex loop, plus higher-order
    thalamocortical projections to association and prefrontal cortex.

Prefrontal
    PFC inputs from association cortex, striatum, and hippocampus;
    PFC outputs to cortex and hippocampus for executive control.

Corticocortical
    Canonical feedforward / feedback hierarchy between sensory and
    association cortex (Felleman & Van Essen 1991), plus top-down
    VIP disinhibition motif (Pi et al. 2013).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thalia.brain.regions.population_names import (
    CortexPopulation,
    HippocampusPopulation,
    StriatumPopulation,
    SubiculumPopulation,
    ThalamusPopulation,
)
from thalia.brain.synapses import ConductanceScaledSpec, STPConfig
from thalia.typing import ReceptorType, SynapseId

if TYPE_CHECKING:
    from thalia.brain.brain_builder import BrainBuilder


# =============================================================================
# Thalamocortical circuit: thalamus ↔ cortex_sensory
# =============================================================================

def connect_thalamocortical(builder: BrainBuilder) -> None:
    """Wire the three thalamocortical loops (sensory, association, mediodorsal).

    Each thalamic nucleus has a dedicated bidirectional loop with its cortical
    target, following the canonical architecture (Sherman & Guillery 2006):

    Sensory thalamus (VPL/VPM) ↔ Sensory cortex
        First-order relay: ascending sensory streams → L4, with L6 feedback.

    Association thalamus (Pulvinar) ↔ Association cortex
        Higher-order relay: L5 driver input relayed to L4 of other areas,
        supporting cortico-thalamo-cortical communication.

    Mediodorsal thalamus (MD) ↔ Prefrontal cortex
        Mnemonic relay: PFC ↔ MD reciprocal loop for working memory
        maintenance and memory-guided planning.
    """
    # =========================================================================
    # SENSORY THALAMUS (VPL/VPM) ↔ SENSORY CORTEX
    # =========================================================================

    # Sensory thalamus → L4 Pyramidal: Main thalamocortical drive
    # Distance: ~2-3cm, conduction velocity: ~10-20 m/s → 2-3ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus_sensory",
            source_population=ThalamusPopulation.RELAY,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.5,
        connectivity=0.5,  # Reduced 0.7→0.5: thalamic burst synchrony drives cortex_sensory:l4_inhibitory_sst epileptiform (5.3%); lower connectivity reduces burst-correlated L4 PYR co-activation without changing per-neuron total drive (ConductanceScaledSpec compensates weight)
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=30.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.70,
            fraction_of_drive=0.85,
            inhibitory_load=0.35,  # Strong PV feedforward + SST feedback on L4 PYR.
        ),
        stp_config=STPConfig(U=0.25, tau_d=300.0, tau_f=20.0),  # U 0.30→0.20→0.25, tau_d 400→250→300:
        # U=0.20/tau_d=250 transmitted too much synchronized thalamic burst input to L4.
        # U=0.25, tau_d=300: moderate depression filters burst synchrony while maintaining
        # adequate drive. x_ss at 24 Hz: 1/(1+0.25*0.024*300)=0.36, x·u=0.09 (Gil et al. 1999).
    )

    # Sensory thalamus → L4 PV: Feedforward inhibition drive
    # PV cells have lower thresholds; thalamus provides 35% of PV drive.
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus_sensory",
            source_population=ThalamusPopulation.RELAY,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L4_INHIBITORY_PV,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.5,
        connectivity=0.7,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=20.0,
            target_g_L=0.10,
            target_tau_E_ms=3.0,
            target_v_inf=0.95,
            fraction_of_drive=0.35,
        ),
        stp_config=STPConfig(U=0.20, tau_d=200.0, tau_f=20.0),
    )

    # Sensory L6a → Sensory thalamus TRN: Inhibitory attention modulation (type-I, slow)
    # L6a→TRN: ~10ms (selectively gates thalamic relay for selective attention).
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L6A_PYR,
            target_region="thalamus_sensory",
            target_population=ThalamusPopulation.TRN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=10.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.10,
            target_tau_E_ms=4.0,
            target_v_inf=1.0,
            fraction_of_drive=0.16,  # 0.13→0.22→0.16: 0.22 drove TRN to 9.7 Hz with epileptiform
                                     # bursting (5.3%) and E/I worsened to 5.6. Moderate value balances
                                     # corticothalamic TRN drive without triggering rebound cascade.
        ),
        stp_config=STPConfig(U=0.4, tau_d=700.0, tau_f=30.0),
    )

    # Sensory L6b → Sensory thalamus RELAY: Excitatory precision feedback (type-II, fast)
    # L6b→Relay: ~5ms; precision-enhancing corticothalamic feedback.
    # Fraction raised 0.30→0.50: with cortex hypoactive (L6B firing ~2 Hz vs designed 5 Hz),
    # the relay only received 0.30×0.40=12% of needed drive. Raising to 0.50 doubles the
    # corticothalamic contribution, helping bootstrap the thalamocortical loop.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L6B_PYR,
            target_region="thalamus_sensory",
            target_population=ThalamusPopulation.RELAY,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.08,
            target_tau_E_ms=5.0,
            target_v_inf=0.60,  # Lowered 0.85→0.60: match external sensory target.
            fraction_of_drive=0.50,  # corticothalamic bootstrap: dominant when cortex active.
        ),
        stp_config=STPConfig(U=0.08, tau_d=150.0, tau_f=800.0),
    )

    # =========================================================================
    # ASSOCIATION THALAMUS (PULVINAR) ↔ ASSOCIATION CORTEX
    # =========================================================================
    # The pulvinar is a higher-order thalamic nucleus that relays cortical L5
    # driver inputs to L4 of target areas (Sherman & Guillery 2002). This
    # enables cortico-thalamo-cortical communication: association L5 →
    # pulvinar → association L4 establishes a transthalamic feedforward pathway
    # complementing direct corticocortical connections.

    # Association thalamus RELAY → Association cortex L4 PYR
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus_association",
            source_population=ThalamusPopulation.RELAY,
            target_region="cortex_association",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        connectivity=0.35,  # Raised 0.30→0.35: L4 assoc only getting drive from relay
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=30.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.70,
            fraction_of_drive=0.55,
            inhibitory_load=0.30,  # Moderate PV/SST feedback on association L4 PYR.
        ),
        stp_config=STPConfig(U=0.25, tau_d=300.0, tau_f=20.0),  # U 0.30→0.25, tau_d 400→300:
        # Less depression so steady-state transmission stays higher under tonic relay firing.
    )

    # Association L6A → Association thalamus TRN: corticothalamic feedback (type-I)
    # Each thalamic nucleus has its own TRN sector receiving feedback from its
    # cortical partner (Crabtree & Isaac 2002).
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L6A_PYR,
            target_region="thalamus_association",
            target_population=ThalamusPopulation.TRN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=10.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.10,
            target_tau_E_ms=4.0,
            target_v_inf=1.0,
            fraction_of_drive=0.16,
        ),
        stp_config=STPConfig(U=0.4, tau_d=700.0, tau_f=30.0),
    )

    # Association L6B → Association thalamus RELAY: corticothalamic feedback (type-II)
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L6B_PYR,
            target_region="thalamus_association",
            target_population=ThalamusPopulation.RELAY,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.08,
            target_tau_E_ms=5.0,
            target_v_inf=0.60,
            fraction_of_drive=0.50,
        ),
        stp_config=STPConfig(U=0.08, tau_d=150.0, tau_f=800.0),
    )

    # =========================================================================
    # MEDIODORSAL THALAMUS (MD) ↔ PREFRONTAL CORTEX
    # =========================================================================
    # The mediodorsal nucleus is the primary thalamic relay for PFC (Giguere &
    # Goldman-Rakic 1988). The PFC ↔ MD reciprocal loop sustains working memory
    # through reverberatory activity (Bolkan et al. 2017): PFC L5 drives MD,
    # MD relays back to PFC L4, and PFC L6 provides modulatory feedback.

    # MD thalamus RELAY → PFC L4 PYR
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus_md",
            source_population=ThalamusPopulation.RELAY,
            target_region="prefrontal_cortex",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        connectivity=0.30,  # Raised 0.25→0.30: PFC L4 pool is small (80 neurons);
                             # more connectivity helps overcome sparse sampling.
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=30.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.70,
            fraction_of_drive=0.45,
            inhibitory_load=0.30,  # Moderate PV/SST feedback on PFC L4 PYR.
        ),
        stp_config=STPConfig(U=0.25, tau_d=300.0, tau_f=20.0),  # Matched to assoc thalamocortical
    )

    # PFC L6A → MD thalamus TRN: corticothalamic feedback (type-I)
    # PFC L6 projects to the MD-associated TRN sector for WM gating
    # (Zikopoulos & Barbas 2006).
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L6A_PYR,
            target_region="thalamus_md",
            target_population=ThalamusPopulation.TRN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=10.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.10,
            target_tau_E_ms=4.0,
            target_v_inf=1.0,
            fraction_of_drive=0.16,
        ),
        stp_config=STPConfig(U=0.4, tau_d=700.0, tau_f=30.0),
    )

    # PFC L6B → MD thalamus RELAY: corticothalamic feedback (type-II)
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L6B_PYR,
            target_region="thalamus_md",
            target_population=ThalamusPopulation.RELAY,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.08,
            target_tau_E_ms=5.0,
            target_v_inf=0.60,
            fraction_of_drive=0.50,
        ),
        stp_config=STPConfig(U=0.08, tau_d=150.0, tau_f=800.0),
    )


# =============================================================================
# Prefrontal cortex circuit: PFC ↔ hippocampus, PFC → cortex, striatum → PFC
# =============================================================================

def connect_prefrontal(builder: BrainBuilder) -> None:
    """Wire prefrontal cortex into the rest of the brain.

    * Association cortex → PFC: multi-modal input to executive control.
    * Striatum D1 → PFC: basal ganglia gating of working memory (via thalamus).
    * PFC ↔ Hippocampus: memory-guided decision making.
    * PFC → Sensory cortex L2/3: top-down attentional modulation.
    * CA1 → PFC L5 apical: hippocampal context to PFC output-layer apical tufts.
    * PFC L23 → Sensory L5 apical: top-down deep-layer FB to sensory output layer.
    * Sub → Thalamus RELAY: memory-guided thalamic re-activation (planning).
    """
    # Association → PFC: higher-level representations drive executive control
    # Distance: ~5-10cm → 10-15ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L23_PYR,
            target_region="prefrontal_cortex",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.02,
            target_tau_E_ms=10.0,
            target_v_inf=1.05,
            fraction_of_drive=0.50,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Striatum D1 → PFC: BG gating of working memory (via MD/VA thalamic relay)
    # Total delay: striatum→thalamus→PFC relay → 15-20ms.
    builder.connect(
        synapse_id=SynapseId(
            source_region="striatum",
            source_population=StriatumPopulation.D1,
            target_region="prefrontal_cortex",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=17.5,
        connectivity=0.6,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=2.0,
            target_g_L=0.02,
            target_tau_E_ms=10.0,
            target_v_inf=1.05,
            fraction_of_drive=0.20,
        ),
        stp_config=STPConfig(U=0.5, tau_d=800.0, tau_f=20.0),
    )

    # PFC → Hippocampus CA1: top-down memory retrieval and schema application
    # Distance: ~5-7cm → 12-18ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="hippocampus",
            target_population=HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=15.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.15,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Subiculum → PFC: memory-guided decision making
    # Subiculum is the primary hippocampal output relay; CA1→Sub burst-to-regular
    # conversion is wired inside the MTL preset.  Sub projects to PFC for
    # memory-guided executive control (Barbas & Blatt 1995; Jay & Witter 1991).
    # Distance: ~5-7cm → 10-15ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="subiculum",
            source_population=SubiculumPopulation.PRINCIPAL,
            target_region="prefrontal_cortex",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.08,  # g_L_d(0.03) + g_c(0.05): dendritic coupling is effective leak.
            target_tau_E_ms=5.0,
            target_v_inf=1.6,
            fraction_of_drive=0.30,  # Primary top-down input to PFC L23 apical.
            stp_utilization_factor=0.28,
        ),
        stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
    )

    # Subiculum → PFC L5: hippocampal context to PFC output-layer apical tufts.
    # Sub projects to BOTH L2/3 and L5/6 of PFC via the fornix.
    # The L5 target provides apical dendritic input, gating subcortical output via
    # coincidence detection in the two-compartment model.
    # PFC is the apex of the cortical hierarchy — hippocampal formation is the only
    # major source of top-down context for PFC L5 apical compartment.
    builder.connect(
        synapse_id=SynapseId(
            source_region="subiculum",
            source_population=SubiculumPopulation.PRINCIPAL,
            target_region="prefrontal_cortex",
            target_population=CortexPopulation.L5_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.08,  # g_L_d(0.03) + g_c(0.05): dendritic coupling is effective leak.
            target_tau_E_ms=5.0,
            target_v_inf=1.6,
            fraction_of_drive=0.75,  # Only major source of top-down context for PFC L5 apical.
            stp_utilization_factor=0.28,
        ),
        stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
    )

    # PFC → Sensory cortex L2/3: top-down attention and cognitive control
    # Corticocortical feedback targets superficial layers (L2/3), bypassing thalamic input.
    # Distance: ~5-8cm → 10-15ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.08,  # g_L_d(0.03) + g_c(0.05): apical compartment effective leak.
            target_tau_E_ms=5.0,
            target_v_inf=1.6,
            fraction_of_drive=0.15,  # Modulatory feedback; must not dominate L4 feedforward.
            stp_utilization_factor=0.16,
        ),
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # PFC → Sensory cortex L5 apical: deep-layer top-down context to sensory output cells.
    # PFC gates subcortical output via coincidence detection in the two-compartment model.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L5_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.08,  # g_L_d(0.03) + g_c(0.05): apical compartment effective leak.
            target_tau_E_ms=5.0,
            target_v_inf=1.6,
            fraction_of_drive=0.75,  # Dominant top-down pathway to sensory L5 apical.
            stp_utilization_factor=0.16,
        ),
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # Subiculum → MD Thalamus RELAY: memory-guided thalamic re-activation
    # Subiculum projects to the mediodorsal (MD) and reuniens (Re) thalamic
    # nuclei (Wouterlood et al. 1990; Vertes 2006). This pathway enables
    # hippocampal memory retrieval to re-activate the PFC ↔ MD loop,
    # supporting memory-guided planning and spatial navigation. The reuniens
    # nucleus is a critical hub in the HPC→thalamus→PFC circuit for working
    # memory consolidation (Dolleman-van der Weel et al. 2019).
    # Distance: ~3-5cm (hippocampal formation → thalamus) → 5-8ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="subiculum",
            source_population=SubiculumPopulation.PRINCIPAL,
            target_region="thalamus_md",
            target_population=ThalamusPopulation.RELAY,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        connectivity=0.20,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.10,
            target_tau_E_ms=4.0,
            target_v_inf=1.0,
            fraction_of_drive=0.12,  # Modulatory: hippocampal re-activation, not primary thalamic drive
        ),
        # Facilitating: sustained hippocampal replay builds thalamic activation.
        stp_config=STPConfig(U=0.20, tau_d=200.0, tau_f=400.0),
    )


# =============================================================================
# Corticocortical connections: two-column hierarchy + association column inputs
# =============================================================================

def connect_corticocortical(builder: BrainBuilder) -> None:
    """Wire the inter-column corticocortical hierarchy.

    Implements the canonical predictive-coding FF/FB architecture
    (Felleman & Van Essen 1991; Bastos et al. 2012):

    * Sensory L2/3 → Association L4: feedforward (FF) percept transfer.
    * Association L6B → Sensory L2/3: feedback (FB) prediction.
    * Hippocampus CA1 → Association L2/3: episodic content to context.
    * PFC → Association L2/3: top-down executive modulation.
    * PFC → Association L5 apical: deep-layer top-down to assoc output neurons.
    * Association L6A → Thalamus TRN: corticothalamic attention gating.
    """
    # Sensory L2/3 → Association L4: feedforward percept transfer
    # Supragranular pyramidals project to granular layer of the next-higher area.
    # Distance: ~2-3cm, well-myelinated → 5ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_association",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=2.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.85,
            fraction_of_drive=0.30,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Association L6B → Sensory L2/3: top-down prediction feedback
    # Deep-layer → superficial-layer of lower area (canonical FB pathway).
    # Carries predictions; suppresses expected patterns (predictive coding).
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L6B_PYR,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.03,
            target_tau_E_ms=5.0,
            target_v_inf=1.6,
            fraction_of_drive=0.30,  # Reduced 0.70→0.30: Association L6B→Sensory L2/3 top-down drive
                                     # bypasses L4 → reversed laminar cascade. Reduce to let L4 feedforward
                                     # dominate L2/3 activation timing.
            stp_utilization_factor=0.25,
        ),
        stp_config=STPConfig(U=0.08, tau_d=150.0, tau_f=800.0),
    )

    # Subiculum → Association L2/3: retrieved episodic content to context
    # Subicular output relays hippocampal retrieval to association cortex
    # for integration with ongoing percepts.
    builder.connect(
        synapse_id=SynapseId(
            source_region="subiculum",
            source_population=SubiculumPopulation.PRINCIPAL,
            target_region="cortex_association",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.03,
            target_tau_E_ms=5.0,
            target_v_inf=1.6,
            fraction_of_drive=0.50,
            stp_utilization_factor=0.28,
        ),
        stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
    )

    # PFC → Association L2/3: top-down executive modulation of higher representations
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_association",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.03,
            target_tau_E_ms=5.0,
            target_v_inf=1.6,
            fraction_of_drive=0.30,  # Reduced 0.50→0.30: top-down L2/3 drive caused reversed laminar
                                     # cascade in association cortex (L2/3 at 0.0ms vs L4 at 1.8ms).
            stp_utilization_factor=0.16,
        ),
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # PFC → Association L5 apical: deep-layer top-down context to assoc output neurons
    # Complements the L2/3 target: while PFC→L2/3 suppresses prediction errors,
    # PFC→L5 apical gates what association-cortex outputs reach subcortical targets
    # (striatum, thalamus).  L5 apical coincidence detection (Larkum 2013) requires
    # both L2/3 basal drive AND apical top-down input to produce burst firing.
    # fraction_of_drive raised 0.90→2.0: apical was silent (see sensory L5 comment).
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_association",
            target_population=CortexPopulation.L5_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.03,
            target_tau_E_ms=5.0,
            target_v_inf=1.6,
            fraction_of_drive=2.0,
            stp_utilization_factor=0.16,
        ),
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # Association L6A → Sensory thalamus TRN: corticothalamic attention control
    # Association cortex gates sensory thalamic relay to shape L4 input in sensory column.
    # Higher-order cortex modulating first-order relay via TRN (Crabtree 2018).
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L6A_PYR,
            target_region="thalamus_sensory",
            target_population=ThalamusPopulation.TRN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.10,
            target_tau_E_ms=4.0,
            target_v_inf=1.0,
            fraction_of_drive=0.16,  # 0.13→0.22→0.16: matches sensory L6A→TRN reduction.
        ),
        stp_config=STPConfig(U=0.4, tau_d=700.0, tau_f=30.0),
    )

    # =========================================================================
    # TOP-DOWN → VIP CONNECTIONS (disinhibitory motif — Pi et al. 2013)
    # =========================================================================
    # Feedback from higher cortical areas preferentially targets VIP interneurons
    # in L1/L2/3 (Lee et al. 2013, Zhang et al. 2014). These connections provide
    # VIP with independent top-down drive that doesn't correlate with local
    # pyramidal activity, enabling the VIP→SST disinhibitory motif.
    #
    # When higher areas activate VIP directly → VIP suppresses SST → SST releases
    # pyramidal apical dendrites → enhanced responses to matching bottom-up input.
    # This creates the expected NEGATIVE VIP-SST correlation.

    # Association L6B → Sensory L2/3 VIP: feedback activates disinhibitory circuit
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L6B_PYR,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L23_INHIBITORY_VIP,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.2,
            fraction_of_drive=0.20,  # Reduced 0.50→0.20: shared cortical oscillatory input co-drove VIP
                                     # and SST → positive r=0.84 correlation. VIP disinhibition should
                                     # emerge from ACh and sparse top-down drive, not tonic cortical input.
            stp_utilization_factor=0.25,
        ),
        stp_config=STPConfig(U=0.08, tau_d=150.0, tau_f=800.0),
    )

    # PFC L2/3 → Sensory L2/3 VIP: prefrontal attention activates VIP disinhibition
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L23_INHIBITORY_VIP,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.2,
            fraction_of_drive=0.25,  # Reduced 0.60→0.25: same rationale as Association→VIP above.
                                     # Shared cortical oscillations create VIP-SST co-activation.
                                     # VIP should be primarily noise + ACh driven pre-training.
            stp_utilization_factor=0.16,
        ),
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # PFC L2/3 → Association L2/3 VIP: executive control over association disinhibition
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_association",
            target_population=CortexPopulation.L23_INHIBITORY_VIP,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.2,
            fraction_of_drive=0.40,
            stp_utilization_factor=0.16,
        ),
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # PFC L2/3 → Sensory L5 VIP: deep-layer disinhibition for output gating
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L5_INHIBITORY_VIP,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.2,
            fraction_of_drive=0.40,
            stp_utilization_factor=0.16,
        ),
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # PFC L2/3 → Association L5 VIP: deep-layer disinhibition for output gating
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_association",
            target_population=CortexPopulation.L5_INHIBITORY_VIP,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.2,
            fraction_of_drive=0.40,
            stp_utilization_factor=0.16,
        ),
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )
