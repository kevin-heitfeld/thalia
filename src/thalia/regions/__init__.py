"""
Brain Region Modules for Thalia

Each brain region has specialized circuitry optimized for different learning tasks.
This module provides biologically-accurate implementations of these regions.

Overview of Regions:
====================

CORTEX (sensory/association areas)
----------------------------------
Learning: Unsupervised Hebbian + BCM homeostasis
Neuromodulation: ACh (attention/gating), NE (arousal)
Function: Feature extraction, pattern recognition, unsupervised clustering
Use when: You want the network to discover structure in data without labels

CEREBELLUM
----------
Learning: Supervised error-corrective (delta rule via climbing fibers)
Error signal: Climbing fibers from inferior olive
Function: Precise timing, motor control, supervised sequence learning
Use when: You have explicit target outputs and want to learn input→output mappings

STRIATUM (part of basal ganglia)
--------------------------------
Learning: Three-factor rule (eligibility × dopamine)
Neuromodulation: Dopamine (reward prediction error)
Function: Reinforcement learning, action selection, habit formation
Use when: You have reward/punishment signals but no explicit target outputs

HIPPOCAMPUS
-----------
Learning: One-shot Hebbian with theta-phase timing
Neuromodulation: ACh (novelty), theta rhythm (temporal organization)
Function: Episodic memory, sequence encoding, spatial navigation
Use when: You need to rapidly memorize sequences or episodes

PREFRONTAL CORTEX
-----------------
Learning: Gated Hebbian with dopamine modulation
Neuromodulation: Dopamine (working memory gating), NE (flexibility)
Function: Working memory, rule learning, executive control
Use when: You need to maintain and manipulate information over time

Example Usage:
==============

```python
from thalia.regions import Cortex, Cerebellum, Striatum

# Unsupervised feature learning
cortex = Cortex(n_input=20, n_output=10)
cortex.learn(input_pattern)  # Discovers structure via Hebbian learning

# Supervised sequence learning
cerebellum = Cerebellum(n_input=20, n_output=10)
cerebellum.learn(input_pattern, target_output)  # Error-corrective learning

# Reinforcement learning
striatum = Striatum(n_input=20, n_output=10)
striatum.learn(input_pattern, action, reward)  # Three-factor learning
```

Biological References:
======================
- Cortex: Bienenstock, Cooper & Munro (1982) - BCM theory
- Cerebellum: Marr (1969), Albus (1971) - Cerebellar learning
- Striatum: Schultz et al. (1997) - Dopamine and reward prediction error
- Hippocampus: O'Keefe & Nadel (1978), Buzsáki (2002) - Theta and memory
"""

from thalia.regions.base import BrainRegion, LearningRule, RegionConfig, RegionState
from thalia.regions.cortex import Cortex, CortexConfig
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.hippocampus import Hippocampus, HippocampusConfig
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig

__all__ = [
    # Base classes
    "BrainRegion",
    "LearningRule",
    "RegionConfig",
    "RegionState",
    # Regions
    "Cortex",
    "CortexConfig",
    "Cerebellum",
    "CerebellumConfig",
    "Striatum",
    "StriatumConfig",
    "Hippocampus",
    "HippocampusConfig",
    "Prefrontal",
    "PrefrontalConfig",
]
