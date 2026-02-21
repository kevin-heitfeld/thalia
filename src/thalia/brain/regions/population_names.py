
from enum import Enum


class CerebellumPopulation(Enum):
    """Cerebellar population names."""

    GRANULE = "granule"
    PURKINJE = "purkinje"
    DCN = "dcn"
    MOSSY = "mossy"
    PARALLEL_FIBERS = "parallel_fibers"
    INFERIOR_OLIVE = "inferior_olive"


class CortexPopulation(Enum):
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


class HippocampusPopulation(Enum):
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


class LocusCoeruleusPopulation(Enum):
    """Locus coeruleus population names."""

    NE = "ne"
    GABA = "gaba"


class MedialSeptumPopulation(Enum):
    """Medial septum population names."""

    ACH = "ach"
    GABA = "gaba"


class NucleusBasalisPopulation(Enum):
    """Nucleus basalis population names."""

    ACH = "ach"
    GABA = "gaba"


class PrefrontalPopulation(Enum):
    """Prefrontal population names."""

    EXECUTIVE = "executive"


class RewardEncoderPopulation(Enum):
    """Reward encoder population names."""

    REWARD_SIGNAL = "reward_signal"


class SubstantiaNigraPopulation(Enum):
    """Substantia nigra population names."""

    VTA_FEEDBACK = "vta_feedback"


class StriatumPopulation(Enum):
    """Striatal population names."""

    D1 = "d1"
    D2 = "d2"
    FSI = "fsi"


class ThalamusPopulation(Enum):
    """Thalamic population names."""

    RELAY = "relay"
    TRN = "trn"


class VTAPopulation(Enum):
    """Ventral tegmental area population names."""

    DA = "da"
    GABA = "gaba"
