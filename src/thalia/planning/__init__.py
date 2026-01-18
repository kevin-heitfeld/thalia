"""
Model-Based Planning System for Thalia

⚠️ IMPORTANT DESIGN PRINCIPLE ⚠️
==================================

Model-based planning is NOT a separate module - it **emerges** from coordination
between existing brain regions using their native learning mechanisms:

**Correct Architecture:**
    - PFC: Maintains simulated states in working memory (already has this!)
    - Hippocampus: Predicts outcomes via episodic memory pattern completion
    - Cortex: Provides state representations via predictive coding
    - Striatum: Evaluates simulated states using learned values
    - Cerebellum: Provides sensorimotor forward models

**Why this matters:**
    - NO BACKPROP: All learning is local (Hebbian, STDP, error-corrective)
    - NO SEPARATE MODULES: Planning emerges from region coordination
    - BIOLOGICALLY PLAUSIBLE: Matches neuroscience (no "world model" brain area)
    - INTEGRATED: Uses existing representations and learning rules

**Delayed Gratification Enhancements**

This module provides **coordination utilities** for mental simulation, NOT
separate world models or learning systems.

Components:
    - Mental simulation coordinator (orchestrates PFC + Hippocampus + Striatum)
    - Rollout utilities (tree search coordination)
    - Replay prioritization (for hippocampal replay)

Biological Inspiration:
    - Hippocampal replay (Foster & Wilson 2006)
    - PFC working memory (Miller & Cohen 2001)
    - VTA prioritization (Mattar & Daw 2018)
    - Model-based/model-free arbitration (Doll et al. 2015)

Example:
    ```python
    from thalia.core.dynamic_brain import DynamicBrain

    brain = BrainBuilder.preset("default", brain_config)

    # Mental simulation uses EXISTING regions (no separate world model!)
    # PFC holds simulated state, hippocampus predicts next, striatum evaluates
    rollout = brain.simulate_rollout(
        current_state=state,
        actions=[0, 1, 2],  # Actions to simulate
        depth=5,  # Steps ahead
    )

    # Choose best action from simulation
    best_action = rollout.best_action
    ```

Author: Thalia Project
Date: December 10, 2025
"""

from __future__ import annotations

# Mental Simulation Coordinator
from thalia.planning.coordinator import (
    MentalSimulationCoordinator,
    Rollout,
    SimulationConfig,
)

# Dyna Background Planning
from thalia.planning.dyna import (
    DynaConfig,
    DynaPlanner,
)

__all__ = [
    # Mental Simulation Coordinator
    "MentalSimulationCoordinator",
    "SimulationConfig",
    "Rollout",
    # Dyna Background Planning
    "DynaPlanner",
    "DynaConfig",
]
