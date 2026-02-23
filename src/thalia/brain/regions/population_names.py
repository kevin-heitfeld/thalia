"""Population names for different brain regions."""

from enum import StrEnum


class ExternalPopulation(StrEnum):
    """Special population names for external inputs that don't belong to any specific region."""

    REWARD = "reward"
    SENSORY = "sensory"


class BLAPopulation(StrEnum):
    """Basolateral amygdala population names.

    Principal neurons (glutamatergic) are the primary computation units.
    PV interneurons provide fast feedforward inhibition for fear gating.
    SOM interneurons provide slow dendritic inhibition for extinction.
    """

    PRINCIPAL = "principal"    # Glutamatergic principal (fear/extinction engrams)
    PV = "pv"                  # Parvalbumin interneurons (feedforward gating)
    SOM = "som"                # Somatostatin interneurons (dendritic inhibition)


class CeAPopulation(StrEnum):
    """Central amygdala population names.

    CeA is the output nucleus: integrates BLA signals and drives
    downstream fear responses via hypothalamus, brainstem, LC, and LHb.
    """

    LATERAL = "lateral"        # CeL: integrates BLA, contains On/Off cells
    MEDIAL = "medial"          # CeM: final output → hypothalamus / LC / LHb


class CerebellumPopulation(StrEnum):
    """Cerebellar population names."""

    DCN = "dcn"
    GRANULE = "granule"
    INFERIOR_OLIVE = "inferior_olive"
    MOSSY = "mossy"
    PARALLEL_FIBERS = "parallel_fibers"
    PURKINJE = "purkinje"


class CortexPopulation(StrEnum):
    """Cortical population names."""

    L23 = "l23"
    L23_PYR = "l23_pyr"
    L23_INHIBITORY = "l23_inhibitory"
    L23_INHIBITORY_PV = "l23_inhibitory_pv"
    L23_INHIBITORY_SST = "l23_inhibitory_sst"
    L23_INHIBITORY_VIP = "l23_inhibitory_vip"

    L4 = "l4"
    L4_PYR = "l4_pyr"
    L4_INHIBITORY = "l4_inhibitory"
    L4_INHIBITORY_PV = "l4_inhibitory_pv"
    L4_INHIBITORY_SST = "l4_inhibitory_sst"
    L4_INHIBITORY_VIP = "l4_inhibitory_vip"
    L4_SST_PRED = "l4_sst_pred"  # Dedicated prediction-error SST interneuron (L5→disynaptic→L4)

    L5 = "l5"
    L5_PYR = "l5_pyr"
    L5_INHIBITORY = "l5_inhibitory"
    L5_INHIBITORY_PV = "l5_inhibitory_pv"
    L5_INHIBITORY_SST = "l5_inhibitory_sst"
    L5_INHIBITORY_VIP = "l5_inhibitory_vip"

    L6 = "l6"
    L6_PYR = "l6_pyr"
    L6_INHIBITORY = "l6_inhibitory"
    L6_INHIBITORY_PV = "l6_inhibitory_pv"
    L6_INHIBITORY_SST = "l6_inhibitory_sst"
    L6_INHIBITORY_VIP = "l6_inhibitory_vip"

    L6A = "l6a"
    L6A_PYR = "l6a_pyr"
    L6A_INHIBITORY = "l6a_inhibitory"
    L6A_INHIBITORY_PV = "l6a_inhibitory_pv"
    L6A_INHIBITORY_SST = "l6a_inhibitory_sst"
    L6A_INHIBITORY_VIP = "l6a_inhibitory_vip"

    L6B = "l6b"
    L6B_PYR = "l6b_pyr"
    L6B_INHIBITORY = "l6b_inhibitory"
    L6B_INHIBITORY_PV = "l6b_inhibitory_pv"
    L6B_INHIBITORY_SST = "l6b_inhibitory_sst"
    L6B_INHIBITORY_VIP = "l6b_inhibitory_vip"


class GPePopulation(StrEnum):
    """Globus pallidus externa population names."""

    ARKYPALLIDAL = "arkypallidal"  # Projects back to striatum (global suppression)
    PROTOTYPIC = "prototypic"    # Projects to STN / SNr (canonical indirect pathway)


class HippocampusPopulation(StrEnum):
    """Hippocampal population names."""

    DG = "dg"
    DG_INHIBITORY = "dg_inhibitory"
    DG_INHIBITORY_PV = "dg_inhibitory_pv"
    DG_INHIBITORY_OLM = "dg_inhibitory_olm"
    DG_INHIBITORY_BISTRATIFIED = "dg_inhibitory_bistratified"

    CA3 = "ca3"
    CA3_INHIBITORY = "ca3_inhibitory"
    CA3_INHIBITORY_PV = "ca3_inhibitory_pv"
    CA3_INHIBITORY_OLM = "ca3_inhibitory_olm"
    CA3_INHIBITORY_BISTRATIFIED = "ca3_inhibitory_bistratified"

    CA2 = "ca2"
    CA2_INHIBITORY = "ca2_inhibitory"
    CA2_INHIBITORY_PV = "ca2_inhibitory_pv"
    CA2_INHIBITORY_OLM = "ca2_inhibitory_olm"
    CA2_INHIBITORY_BISTRATIFIED = "ca2_inhibitory_bistratified"

    CA1 = "ca1"
    CA1_INHIBITORY = "ca1_inhibitory"
    CA1_INHIBITORY_PV = "ca1_inhibitory_pv"
    CA1_INHIBITORY_OLM = "ca1_inhibitory_olm"
    CA1_INHIBITORY_BISTRATIFIED = "ca1_inhibitory_bistratified"


class LHbPopulation(StrEnum):
    """Lateral habenula population names."""

    PRINCIPAL = "principal"


class LocusCoeruleusPopulation(StrEnum):
    """Locus coeruleus population names."""

    NE = "ne"
    GABA = "gaba"


class MedialSeptumPopulation(StrEnum):
    """Medial septum population names."""

    ACH = "ach"
    GABA = "gaba"


class NucleusBasalisPopulation(StrEnum):
    """Nucleus basalis population names."""

    ACH = "ach"
    GABA = "gaba"


class PrefrontalPopulation(StrEnum):
    """Prefrontal population names."""

    EXECUTIVE = "executive"


class RMTgPopulation(StrEnum):
    """Rostromedial tegmental nucleus (tail of VTA / anti-reward centre) population names."""

    GABA = "gaba"


class SNcPopulation(StrEnum):
    """Substantia nigra pars compacta population names."""

    DA = "da"
    GABA = "gaba"


class STNPopulation(StrEnum):
    """Subthalamic nucleus population names."""

    STN = "stn"


class StriatumPopulation(StrEnum):
    """Striatal population names."""

    D1 = "d1"
    D2 = "d2"
    FSI = "fsi"
    TAN = "tan"   # Tonically Active Neurons (cholinergic interneurons)


class SubstantiaNigraPopulation(StrEnum):
    """Substantia nigra population names."""

    VTA_FEEDBACK = "vta_feedback"


class ThalamusPopulation(StrEnum):
    """Thalamic population names."""

    RELAY = "relay"
    TRN = "trn"


class VTAPopulation(StrEnum):
    """Ventral tegmental area population names.

    DA_MESOLIMBIC: Mesolimbic DA neurons (55% of VTA DA) — project to ventral striatum,
        hippocampus, amygdala. Have D2 somatodendritic autoreceptors. Encode reward RPE.
    DA_MESOCORTICAL: Mesocortical DA neurons (35% of VTA DA) — project to PFC. Lack D2
        autoreceptors; higher baseline firing (~8 Hz). Encode cognitive salience/arousal.
    DA: Legacy alias kept for synaptic input targeting (e.g. RMTg→VTA connections that
        should inhibit both sub-populations). Regions declaring neuromodulator_subscriptions
        should use the specific sub-population keys.
    """

    DA_MESOLIMBIC = "da_mesolimbic"
    DA_MESOCORTICAL = "da_mesocortical"
    DA = "da"  # Legacy — maps to combined DA input; prefer sub-population keys
    GABA = "gaba"
