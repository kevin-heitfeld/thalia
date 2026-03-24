"""Amygdala circuit connections preset.

Wires the BLA (fear input / extinction) and CeA (fear output) circuits:

BLA inputs
    * Sensory cortex L5 (CS slow, detailed pathway).
    * Thalamus RELAY (CS fast, thalamo-amygdalar shortcut).
    * Subiculum (contextual fear / extinction renewal).
    * PFC L5 → BLA SOM (top-down extinction regulation).

BLA → Hippocampus
    * BLA → CA1 (emotional memory enhancement).
    * BLA → CA3 (emotional context for pattern completion).

Hippocampus → BLA
    * CA1 → BLA PRINCIPAL (contextual-emotional binding).

BLA → sensory cortex
    * BLA PRINCIPAL → sensory L1_NGC (fear-gated sensory gain).

BLA → CeA
    * BLA PRINCIPAL → CeL (core fear signal via lateral gate).
    * BLA PRINCIPAL → CeM (direct bypass for rapid output).

CeA outputs
    * CeM → LC NE (fear-driven norepinephrine arousal).
    * CeM → LHb (aversive prediction error signal).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thalia.brain.regions.population_names import (
    BLAPopulation,
    CeAPopulation,
    CortexPopulation,
    HippocampusPopulation,
    LHbPopulation,
    LocusCoeruleusPopulation,
    SubiculumPopulation,
    ThalamusPopulation,
)
from thalia.brain.synapses import ConductanceScaledSpec, STPConfig
from thalia.typing import ReceptorType, SynapseId

if TYPE_CHECKING:
    from thalia.brain.brain_builder import BrainBuilder


def connect_amygdala(builder: BrainBuilder) -> None:
    """Wire the amygdala circuits.

    BLA inputs: sensory cortex (CS slow), thalamus (CS fast), hippocampus (context),
                PFC (top-down extinction regulation).
    CA1 → BLA: contextual-emotional binding (reciprocal of BLA→CA1).
    BLA → sensory cortex: fear-gated sensory gain modulation.
    BLA → CeA: transmit conditioned fear signal to output nucleus.
    CeA outputs: LC (arousal) and LHb (aversive RPE).
    """
    # Sensory cortex L5 → BLA PRINCIPAL: CS representation (slow, detailed pathway)
    # Auditory/somatosensory cortex provides the conditioned stimulus (CS) signal.
    # Distance: ~3-5cm → 6-10ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        connectivity=0.15,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=6.0,
            target_g_L=0.05,
            target_tau_E_ms=7.0,
            target_v_inf=1.05,
            fraction_of_drive=0.35,
        ),
        # Depressing long-range cortical projection.
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Sensory thalamus RELAY → BLA PRINCIPAL: fast CS pathway (thalamo-amygdalar shortcut)
    # Direct thalamic relay bypasses cortex (~12ms faster than cortical path).
    # Enables rapid fear conditioning before full cortical elaboration of CS.
    # Distance: ~2-3cm → ~5ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus_sensory",
            source_population=ThalamusPopulation.RELAY,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=30.0,
            target_g_L=0.05,
            target_tau_E_ms=7.0,
            target_v_inf=1.05,
            fraction_of_drive=0.40,
        ),
        # Moderately depressing thalamocortical relay.
        stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=20.0),
    )

    # Subiculum → BLA PRINCIPAL: contextual fear / extinction renewal
    # Subiculum encodes spatial/temporal context; gates fear recall based on place-memory.
    # Distance: ~1-2cm (directly adjacent structures) → 3-5ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="subiculum",
            source_population=SubiculumPopulation.PRINCIPAL,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=7.0,
            target_v_inf=1.05,
            fraction_of_drive=0.15,
        ),
        # Facilitating — hippocampal→amygdala gates memory-driven salience.
        stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=300.0),
    )

    # PFC → BLA SOM: top-down extinction regulation
    # Infralimbic PFC → BLA SOM interneurons inhibit principal neurons → extinction.
    # Distance: ~4-6cm → 6-10ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L5_PYR,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.SOM,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.06,
            target_tau_E_ms=8.0,
            target_v_inf=1.05,
            fraction_of_drive=1.0,
        ),
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # BLA PRINCIPAL → Hippocampus CA1: emotional memory enhancement
    # BLA projects to CA1 stratum lacunosum-moleculare and stratum oriens
    # via the amygdalohippocampal pathway (Pikkarainen et al. 1999;
    # McGaugh 2004; Roozendaal & McGaugh 2011).  Emotional arousal (BLA
    # activation) enhances synaptic plasticity at CA1 synapses, biasing
    # memory encoding toward emotionally salient events.
    # Distance: ~1-2cm (adjacent medial temporal lobe structures) → 3-5ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="hippocampus",
            target_population=HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        connectivity=0.15,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.10,  # Modulatory: enhances ongoing plasticity, not primary drive
        ),
        # Facilitating: emotional salience builds with sustained BLA activation.
        stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=300.0),
    )

    # BLA PRINCIPAL → Hippocampus CA3: emotional context for pattern completion
    # BLA also projects to CA3, enhancing recurrent attractor dynamics for
    # emotionally tagged memories (Huff et al. 2016).  Weaker than CA1 pathway.
    # Distance: ~1-2cm → 4-6ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="hippocampus",
            target_population=HippocampusPopulation.CA3,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        connectivity=0.10,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.06,  # Weaker than CA1: modulatory enhancement
        ),
        stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=300.0),
    )

    # =========================================================================
    # Hippocampus → BLA: contextual-emotional binding
    # =========================================================================

    # Hippocampus CA1 → BLA PRINCIPAL: contextual memory drives fear reactivation
    # CA1 place cells and time cells project back to BLA, providing the contextual
    # representation that reactivates fear memories in the appropriate context
    # (Maren & Fanselow 1995; Xu et al. 2016). This reciprocal pathway (reverse
    # of BLA→CA1) is essential for contextual fear conditioning and renewal.
    # Distance: ~1-2cm (adjacent medial temporal lobe structures) → 3-5ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        connectivity=0.15,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=7.0,
            target_v_inf=1.05,
            fraction_of_drive=0.10,  # Modulatory: context gates fear, not primary drive
        ),
        # Facilitating: sustained contextual signals build BLA re-activation.
        stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=300.0),
    )

    # =========================================================================
    # BLA → sensory cortex: fear-gated sensory gain modulation
    # =========================================================================

    # BLA PRINCIPAL → Sensory L1_NGC: amygdalar fear-driven sensory gain
    # BLA projects to superficial layers of sensory cortex, targeting L1
    # neurogliaform (NGC) inhibitory interneurons (Amaral & Price 1984;
    # McDonald 1998). NGC activation provides slow GABA_B inhibition onto
    # pyramidal apical dendrites, sharpening receptive fields under threat.
    # This enables fear-gated sensory attention without requiring direct
    # excitation of pyramidal neurons.
    # Distance: ~3-5cm (amygdala → cortex) → 6-8ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L1_NGC,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=7.0,
        connectivity=0.15,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.12,  # Modest: fear modulates but doesn't dominate sensory processing
        ),
        stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=300.0),
    )

    # BLA PRINCIPAL → CeA LATERAL: core fear signal transmission
    # LA/BA principal neurons project to CeL, driving fear-ON cells.
    # Distance: ~0.5-1cm (within amygdaloid complex) → 2-3ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="central_amygdala",
            target_population=CeAPopulation.LATERAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.06,
            target_tau_E_ms=6.0,
            target_v_inf=1.05,
            fraction_of_drive=0.60,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )

    # BLA PRINCIPAL → CeA MEDIAL: direct projection (bypasses CeL gating)
    # Some BLA principal neurons project directly to CeM for rapid fear output.
    # Distance: ~0.5-1cm → 2-3ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="central_amygdala",
            target_population=CeAPopulation.MEDIAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=6.0,
            target_v_inf=0.90,
            fraction_of_drive=0.45,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )

    # CeA MEDIAL → LC NE: fear-driven norepinephrine arousal
    # CeM activates LC during fear, driving NE release and sympathetic arousal.
    # Distance: ~3-5cm (amygdala → pons) → 5-10ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="central_amygdala",
            source_population=CeAPopulation.MEDIAL,
            target_region="locus_coeruleus",
            target_population=LocusCoeruleusPopulation.NE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=7.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.056,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.08,  # Reduced 0.10→0.08: LC:NE=5.64 Hz (target ≤5), continue reducing
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )

    # CeA MEDIAL → LHb: aversive prediction error signal
    # CeM output encodes expected punishment; drives LHb for negative RPE.
    # LHb will then activate RMTg → DA pause in VTA.
    # Distance: ~3-4cm → 5-8ms delay.
    # fraction_of_drive 0.25→0.40→0.30→0.26: steep threshold; 0.27+0.26=0.53 total.
    builder.connect(
        synapse_id=SynapseId(
            source_region="central_amygdala",
            source_population=CeAPopulation.MEDIAL,
            target_region="lateral_habenula",
            target_population=LHbPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.08,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.26,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )
