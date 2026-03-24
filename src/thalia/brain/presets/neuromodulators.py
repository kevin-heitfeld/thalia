"""Neuromodulatory system connections preset.

Wires inputs to the three spike-based neuromodulator centres, plus
spike-based outgoing projections from LC and NB to cortex:

* **Locus Coeruleus (NE)**: PFC uncertainty + hippocampal novelty + fear (CeA).
* **Nucleus Basalis (ACh)**: PFC prediction error + BLA emotional salience.
* **Dorsal Raphe Nucleus (5-HT)**: LHb punishment signal suppresses serotonin.
* **VTA DA direct projections**: Glutamate co-release to HPC CA1, BLA, and striatum.
* **LC NE → cortex**: Noradrenergic gain modulation of cortical output layers.
* **NB ACh → cortex**: Cholinergic attentional disinhibition via VIP interneurons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thalia.brain.regions.population_names import (
    BLAPopulation,
    CortexPopulation,
    DRNPopulation,
    HippocampusPopulation,
    LHbPopulation,
    LocusCoeruleusPopulation,
    NucleusBasalisPopulation,
    VTAPopulation,
)
from thalia.brain.synapses import ConductanceScaledSpec, STPConfig
from thalia.typing import ReceptorType, SynapseId

if TYPE_CHECKING:
    from thalia.brain.brain_builder import BrainBuilder


def connect_neuromodulators(builder: BrainBuilder) -> None:
    """Wire inputs into the three spike-based neuromodulator systems, plus
    outgoing spike-based projections from LC and NB to cortex.

    * Locus Coeruleus (NE): PFC uncertainty + hippocampal novelty + fear.
    * Nucleus Basalis (ACh): PFC prediction error + BLA emotional salience.
    * Dorsal Raphe (5-HT): LHb punishment signal suppresses serotonin.
    * LC NE → cortex L5: noradrenergic gain modulation.
    * NB ACh → cortex VIP: cholinergic attentional disinhibition.
    """
    # --- Locus Coeruleus -------------------------------------------------------

    # PFC → LC NE: prefrontal variance signals uncertainty
    # High PFC activity variance → high LC firing → NE release.
    # Distance: ~3-5cm → 5-8ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L5_PYR,
            target_region="locus_coeruleus",
            target_population=LocusCoeruleusPopulation.NE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.056,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.06,  # Reduced 0.08→0.06: LC:NE=5.64 Hz (target ≤5), PFC input
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Hippocampus CA1 → LC NE: novelty detection drives arousal
    # CA1 output variance indicates contextual novelty.
    # Distance: ~4-6cm → 8-12ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="locus_coeruleus",
            target_population=LocusCoeruleusPopulation.NE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=10.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.056,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.04,  # Reduced 0.06→0.04: LC:NE=5.64 Hz (target ≤5), CA1 novelty input
        ),
        stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
    )

    # --- Nucleus Basalis -------------------------------------------------------

    # PFC → NB ACH: prefrontal activity changes signal prediction errors
    # Unexpected events drive encoding mode in cortex/hippocampus.
    # Distance: ~3-5cm → 5-8ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L5_PYR,
            target_region="nucleus_basalis",
            target_population=NucleusBasalisPopulation.ACH,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.083,
            target_tau_E_ms=4.0,
            target_v_inf=0.95,
            fraction_of_drive=0.25,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # BLA PRINCIPAL → NB ACH: emotional salience drives ACh encoding-mode bursts
    # BLA principal neurons respond to unexpected / aversive stimuli (US, threat).
    # High BLA activity → strong prediction error → NB bursts ACh.
    # Distance: ~2-4cm → 5-8ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="nucleus_basalis",
            target_population=NucleusBasalisPopulation.ACH,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.083,
            target_tau_E_ms=4.0,
            target_v_inf=0.95,
            fraction_of_drive=0.28,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )

    # --- VTA Dopamine Direct Projections ---------------------------------------

    # VTA DA_MESOLIMBIC → Hippocampus CA1: novelty-gated memory encoding
    # VTA DA neurons project to CA1 stratum lacunosum-moleculare via the
    # mesolimbic pathway (Gasbarri et al. 1994; Lisman & Grace 2005).
    # Besides volume-transmission DA (already handled by neuromodulator broadcast),
    # DA terminals co-release glutamate (Hnasko et al. 2010), providing a fast
    # excitatory signal that gates plasticity windows for memory encoding.
    # Distance: ~4-6cm (midbrain → hippocampus) → 6-10ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="vta",
            source_population=VTAPopulation.DA_MESOLIMBIC,
            target_region="hippocampus",
            target_population=HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        connectivity=0.15,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.08,  # Modest: glutamate co-release is secondary to DA modulation
        ),
        stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=600.0),
    )

    # VTA DA_MESOLIMBIC → BLA PRINCIPAL: emotional saliency gating
    # VTA DA projections to BLA gate fear conditioning and appetitive learning
    # (Bissière et al. 2003; Fadok et al. 2009).  DA release at BLA enhances
    # plasticity at CS→BLA synapses during salient events.
    # Distance: ~3-5cm (midbrain → amygdala) → 5-8ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="vta",
            source_population=VTAPopulation.DA_MESOLIMBIC,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        connectivity=0.15,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=7.0,
            target_v_inf=1.05,
            fraction_of_drive=0.08,  # Modest: DA tone modulation is the primary effect
        ),
        stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=600.0),
    )

    # VTA DA_MESOLIMBIC → Striatum: mesolimbic reward pathway (glutamate co-release)
    # VTA mesolimbic DA neurons project to ventral striatum (nucleus accumbens).
    # Besides volume-transmission DA (handled by DA_MESOLIMBIC neuromodulator
    # channel), DA neurons co-release glutamate at striatal synapses
    # (Hnasko et al. 2010; Tritsch et al. 2012).  This provides a fast
    # excitatory coincidence signal with DA release, enabling reward-driven
    # goal learning and incentive motivation.
    # Distance: ~3-4cm (midbrain → ventral striatum), myelinated → 4-6ms delay.
    # DA neurons fire tonically at ~4-6 Hz; facilitating STP amplifies phasic bursts.
    builder.connect_to_striatum(
        source_region="vta",
        source_population=VTAPopulation.DA_MESOLIMBIC,
        axonal_delay_ms=6.0,
        connectivity=0.20,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.08,  # Modest: glutamate co-release is secondary to DA modulation
        ),
        stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=600.0),
    )

    # --- Dorsal Raphe Nucleus --------------------------------------------------

    # LHb → DRN SEROTONIN: punishment / negative RPE → 5-HT pause
    # LHb principal (glutamatergic) projects heavily to DRN.
    # High LHb activity → 5-HT pause (via local GABA interneurons in DRN).
    # Distance: ~1-2cm (adjacent midbrain) → 2-4ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="lateral_habenula",
            source_population=LHbPopulation.PRINCIPAL,
            target_region="dorsal_raphe",
            target_population=DRNPopulation.SEROTONIN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        connectivity=0.5,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.067,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.22,  # Reduced 0.25→0.22: DR serotonin=3.05 Hz (target ≤3), marginal overshoot
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )

    # PFC L5 → DRN SEROTONIN: top-down excitatory control of serotonin tone
    # PFC glutamatergic L5 neurons project to DRN 5-HT neurons, supporting
    # serotonin tone when executive control is engaged.  Active PFC → high
    # 5-HT → patience, impulse control, and extinction gating.
    # When PFC is hypoactive (loss of control), serotonin drops → impulsivity.
    # Biology: Celada et al. 2001; Hajós et al. 2007; Vertes 2004.
    # Distance: ~4-6cm (frontal cortex → midbrain) → 6-10ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L5_PYR,
            target_region="dorsal_raphe",
            target_population=DRNPopulation.SEROTONIN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.067,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.12,  # Moderate: top-down modulation, not primary drive
        ),
        stp_config=STPConfig(U=0.30, tau_d=500.0, tau_f=200.0),
    )

    for cortex_region in ("cortex_sensory", "cortex_association", "prefrontal_cortex"):
        # --- LC NE → Cortex (outgoing) --------------------------------------------
        # LC NE neurons project broadly to cortex, with strongest innervation of
        # L5 output neurons (Berridge & Waterhouse 2003; Aston-Jones & Cohen 2005).
        # NE enhances signal-to-noise ratio via gain modulation: spike-based
        # glutamate co-release from NE terminals provides fast excitatory drive
        # on top of the slower volume-transmission NE modulation (already handled
        # by neuromodulator broadcast). This pathway targets L5_PYR across all
        # three cortical regions (sensory, association, PFC).
        # Distance: ~4-7cm (brainstem → cortex) → 6-8ms delay.

        builder.connect(
            synapse_id=SynapseId(
                source_region="locus_coeruleus",
                source_population=LocusCoeruleusPopulation.NE,
                target_region=cortex_region,
                target_population=CortexPopulation.L5_PYR,
                receptor_type=ReceptorType.AMPA,
            ),
            axonal_delay_ms=7.0,
            connectivity=0.30,
            weight_scale=ConductanceScaledSpec(
                source_rate_hz=3.0,
                target_g_L=0.05,
                target_tau_E_ms=5.0,
                target_v_inf=1.05,
                fraction_of_drive=0.08,  # Modest: gain modulation, not primary drive
            ),
            # Facilitating: sustained arousal builds cortical gain over time.
            stp_config=STPConfig(U=0.20, tau_d=500.0, tau_f=300.0),
        )

        # --- NB ACh → Cortex (outgoing) -------------------------------------------
        # Nucleus basalis cholinergic neurons project diffusely to cortex, with
        # preferential targeting of VIP interneurons in L1/L2/3 (Letzkus et al.
        # 2011; Alitto & Dan 2013). ACh activates VIP → VIP inhibits SST →
        # SST releases pyramidal apical dendrites → enhanced sensory responses
        # (disinhibitory motif, Pi et al. 2013). This provides the spike-based
        # fast component of cholinergic attention, complementing the slower
        # volume-transmission ACh modulation.
        # Distance: ~3-6cm (basal forebrain → cortex) → 5-8ms delay.

        builder.connect(
            synapse_id=SynapseId(
                source_region="nucleus_basalis",
                source_population=NucleusBasalisPopulation.ACH,
                target_region=cortex_region,
                target_population=CortexPopulation.L23_INHIBITORY_VIP,
                receptor_type=ReceptorType.AMPA,
            ),
            axonal_delay_ms=6.0,
            connectivity=0.25,
            weight_scale=ConductanceScaledSpec(
                source_rate_hz=5.0,
                target_g_L=0.05,
                target_tau_E_ms=5.0,
                target_v_inf=1.2,
                fraction_of_drive=0.15,  # Modest: ACh primes VIP disinhibition circuit
            ),
            # Facilitating: attentional signals build with sustained ACh drive.
            stp_config=STPConfig(U=0.20, tau_d=300.0, tau_f=400.0),
        )
