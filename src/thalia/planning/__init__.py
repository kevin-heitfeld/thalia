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

**Phase 2 of Delayed Gratification Enhancements**

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
    from thalia.core.brain import EventDrivenBrain

    brain = EventDrivenBrain(config)

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
Phase: 2 - Model-Based Planning (via region coordination)
"""

from __future__ import annotations

# Phase 2 Week 1-2: Mental Simulation Coordinator (IMPLEMENTED)
from thalia.planning.coordinator import (
    MentalSimulationCoordinator,
    SimulationConfig,
    Rollout,
)

# Phase 2 Week 3: Dyna Background Planning (IMPLEMENTED)
from thalia.planning.dyna import (
    DynaPlanner,
    DynaConfig,
)

__all__ = [
    # Phase 2 Week 1-2 - Mental Simulation Coordinator (IMPLEMENTED)
    "MentalSimulationCoordinator",
    "SimulationConfig",
    "Rollout",
    
    # Phase 2 Week 3 - Dyna Integration (IMPLEMENTED)
    "DynaPlanner",
    "DynaConfig",
]

# Version and status
__version__ = "0.3.0"
__status__ = "Phase 2 - Week 1-3 Complete (Coordinator + DynaPlanner implemented)"
__next_step__ = "Brain integration: update select_action() and BrainConfig"
