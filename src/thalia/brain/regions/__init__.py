"""
Brain Region Modules for Thalia

Each brain region has specialized circuitry optimized for different learning tasks.
This module provides biologically-accurate implementations of these regions.

Overview of Regions:
====================

CEREBELLUM
----------
Function: Precise timing, motor control, supervised sequence learning
Learning: Supervised error-corrective (delta rule via climbing fibers)
Error signal: Climbing fibers from inferior olive
Use when: You have explicit target outputs and want to learn input→output mappings

CORTEX
----------------------------------
Function: Feature extraction, pattern recognition, unsupervised clustering
Learning: Unsupervised Hebbian + BCM homeostasis
Neuromodulation: ACh (attention/gating), NE (arousal)
Use when: You want the network to discover structure in data without labels

HIPPOCAMPUS
-----------
Function: Episodic memory, sequence encoding, spatial navigation
Learning: One-shot Hebbian with theta-phase timing
Neuromodulation: ACh (novelty), theta rhythm (temporal organization)
Use when: You need to rapidly memorize sequences or episodes

PREFRONTAL CORTEX
-----------------
Function: Working memory, rule learning, executive control
Learning: Gated Hebbian with dopamine modulation
Neuromodulation: Dopamine (working memory gating), NE (flexibility)
Use when: You need to maintain and manipulate information over time

STRIATUM
--------------------------------
Function: Reinforcement learning, action selection, habit formation
Learning: Three-factor rule (eligibility × dopamine)
Neuromodulation: Dopamine (reward prediction error)
Use when: You have reward/punishment signals but no explicit target outputs

MEDIAL SEPTUM
-------------
Function: Theta pacemaker for hippocampal circuits
Learning: None (pacemaker dynamics)
Neuromodulation: ACh (frequency), NE (amplitude), DA (frequency)
Use when: You need emergent theta rhythm for hippocampal encoding/retrieval

THALAMUS
--------
Function: Sensory relay and preprocessing
Learning: Minimal, primarily relay function
Use when: You need to model sensory input pathways
"""

from __future__ import annotations

from .basolateral_amygdala import BasolateralAmygdala
from .central_amygdala import CentralAmygdala
from .cerebellum import Cerebellum
from .cortical_column import CorticalColumn
from .globus_pallidus_externa import GlobusPallidusExterna
from .hippocampus import Hippocampus
from .lateral_habenula import LateralHabenula
from .locus_coeruleus import LocusCoeruleus
from .medial_septum import MedialSeptum
from .nucleus_basalis import NucleusBasalis
from .prefrontal import Prefrontal
from .rostromedial_tegmentum import RostromedialTegmentum
from .striatum import Striatum
from .substantia_nigra import SubstantiaNigra
from .substantia_nigra_compacta import SubstantiaNigraCompacta
from .subthalamic_nucleus import SubthalamicNucleus
from .thalamus import Thalamus
from .vta import VTA

from .neural_region import NeuralRegion
from .region_registry import NeuralRegionRegistry, register_region
from .stimulus_gating import StimulusGating

__all__ = [
    # Base Neural Region
    "NeuralRegion",
    # Regions
    "BasolateralAmygdala",
    "CentralAmygdala",
    "Cerebellum",
    "CorticalColumn",
    "GlobusPallidusExterna",
    "Hippocampus",
    "LateralHabenula",
    "LocusCoeruleus",
    "MedialSeptum",
    "NucleusBasalis",
    "Prefrontal",
    "RostromedialTegmentum",
    "Striatum",
    "SubstantiaNigra",
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
