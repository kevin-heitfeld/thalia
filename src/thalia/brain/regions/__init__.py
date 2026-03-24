"""
Brain Region Modules for Thalia

Each brain region has specialized circuitry optimized for different learning tasks.
This module provides biologically-accurate implementations of these regions.
"""

from __future__ import annotations

from thalia.brain.configs.basal_ganglia import BGPopulationConfig

from .neural_region import NeuralRegion, InternalConnectionSpec
from .neuromodulator_source_region import NeuromodulatorSourceRegion
from .basal_ganglia_output_nucleus import BasalGangliaOutputNucleus
from .globus_pallidus_externa import GlobusPollidusExterna
from .basolateral_amygdala import BasolateralAmygdala
from .central_amygdala import CentralAmygdala
from .cerebellum import Cerebellum
from .cortical_column import CorticalColumn
from .dorsal_raphe import DorsalRapheNucleus
from .entorhinal_cortex import EntorhinalCortex
from .hippocampus import Hippocampus
from .lateral_habenula import LateralHabenula
from .locus_coeruleus import LocusCoeruleus
from .medial_septum import MedialSeptum
from .nucleus_basalis import NucleusBasalis
from .prefrontal_cortex import PrefrontalCortex
from .rostromedial_tegmentum import RostromedialTegmentum
from .striatum import Striatum
from .subiculum import Subiculum
from .substantia_nigra_compacta import SubstantiaNigraCompacta
from .subthalamic_nucleus import SubthalamicNucleus
from .thalamus import Thalamus
from .vta import VTA

from .region_registry import NeuralRegionRegistry, register_region
from .stimulus_gating import StimulusGating

__all__ = [
    # Base Neural Region
    "NeuralRegion",
    "NeuromodulatorSourceRegion",
    "InternalConnectionSpec",
    "BasalGangliaOutputNucleus",
    "GlobusPollidusExterna",
    "BGPopulationConfig",
    # Regions
    "BasolateralAmygdala",
    "CentralAmygdala",
    "Cerebellum",
    "CorticalColumn",
    "DorsalRapheNucleus",
    "EntorhinalCortex",
    "Hippocampus",
    "LateralHabenula",
    "LocusCoeruleus",
    "MedialSeptum",
    "NucleusBasalis",
    "PrefrontalCortex",
    "RostromedialTegmentum",
    "Striatum",
    "Subiculum",
    "SubstantiaNigraCompacta",
    "SubthalamicNucleus",
    "Thalamus",
    "VTA",
    # Neural Region Registry
    "NeuralRegionRegistry",
    "register_region",
    # Stimulus Gating
    "StimulusGating",
]
