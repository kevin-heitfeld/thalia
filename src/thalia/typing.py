"""
Type Aliases for Thalia

This module defines type aliases used throughout the Thalia codebase for
clearer type hints and better IDE support.

All type aliases are organized by category and should be imported from this
module rather than defining them inline.

Example:
    from thalia.typing import ComponentGraph, SourceOutputs, StateDict

Author: Thalia Project
Date: December 21, 2025
"""

from typing import Dict, List, Tuple, Optional, Any
import torch

# Re-export for convenience when using TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thalia.regions.base import NeuralRegion
    from thalia.learning.strategies.base import LearningStrategy

# ============================================================================
# Component Organization
# ============================================================================

ComponentGraph = Dict[str, "NeuralRegion"]
"""Maps component names to component instances.

Example:
    components: ComponentGraph = {
        "cortex": cortex_region,
        "hippocampus": hippocampus_region,
    }
"""

ConnectionGraph = Dict[Tuple[str, str], "NeuralRegion"]
"""Maps (source, target) pairs to pathway instances.

Example:
    connections: ConnectionGraph = {
        ("thalamus", "cortex"): thalamic_pathway,
        ("cortex", "striatum"): corticostriatal_pathway,
    }
"""

TopologyGraph = Dict[str, List[str]]
"""Maps source region names to lists of target region names.

Example:
    topology: TopologyGraph = {
        "thalamus": ["cortex", "striatum"],
        "cortex": ["striatum", "hippocampus"],
    }
"""

# ============================================================================
# Multi-Source Pathways
# ============================================================================

SourceOutputs = Dict[str, torch.Tensor]
"""Maps source names to their output spike tensors.

Used for multi-source pathways where multiple regions project to one target.

Example:
    source_outputs: SourceOutputs = {
        "cortex": cortex_spikes,
        "hippocampus": hippocampus_spikes,
    }
"""

InputSizes = Dict[str, int]
"""Maps source names to their input sizes.

Used by multi-source pathways to track input dimensions per source.

Example:
    input_sizes: InputSizes = {
        "cortex": 100,
        "hippocampus": 64,
    }
"""

SynapticWeights = Dict[str, torch.Tensor]
"""Maps source names to their synaptic weight matrices.

Weights are stored at target dendrites, organized by source region.

Example:
    synaptic_weights: SynapticWeights = {
        "cortex": torch.randn(n_output, 100),
        "hippocampus": torch.randn(n_output, 64),
    }
"""

LearningStrategies = Dict[str, "LearningStrategy"]
"""Maps source names to their learning strategies.

Each source can have its own learning rule (STDP, BCM, Hebbian, etc.)

Example:
    strategies: LearningStrategies = {
        "cortex": stdp_strategy,
        "hippocampus": hebbian_strategy,
    }
"""

# ============================================================================
# Port-Based Routing
# ============================================================================

SourcePort = Optional[str]
"""Optional source port identifier for layer-specific outputs.

Examples: 'l23', 'l5', 'l4', 'ca1', 'ca3', None
"""

TargetPort = Optional[str]
"""Optional target port identifier for input types.

Examples: 'feedforward', 'top_down', 'ec_l3', 'pfc_modulation', None
"""

# ============================================================================
# State Management
# ============================================================================

StateDict = Dict[str, torch.Tensor]
"""Component state for checkpointing.

Contains all tensors needed to restore component state.

Example:
    state: StateDict = {
        "membrane_voltage": v_mem,
        "synaptic_traces": traces,
        "eligibility": eligibility,
    }
"""

CheckpointMetadata = Dict[str, Any]
"""Training progress and stage information.

Example:
    metadata: CheckpointMetadata = {
        "stage": 2,
        "epoch": 150,
        "global_step": 45000,
        "timestamp": "2025-12-21T08:00:00Z",
    }
"""

# ============================================================================
# Diagnostics
# ============================================================================

DiagnosticsDict = Dict[str, Any]
"""Component health and performance metrics.

Example:
    diagnostics: DiagnosticsDict = {
        "firing_rate": 5.2,
        "weight_mean": 0.35,
        "weight_std": 0.12,
        "sparsity": 0.15,
    }
"""

# ============================================================================
# Configuration (dataclasses, not aliases)
# ============================================================================

# Note: ComponentSpec and ConnectionSpec are actual dataclasses,
# not type aliases. They are defined in thalia.core.brain_builder
# and thalia.config respectively.

# ============================================================================
# Neuromodulation
# ============================================================================

NeuromodulatorLevels = Dict[str, float]
"""Maps neuromodulator names to their current levels (0.0-1.0).

Example:
    levels: NeuromodulatorLevels = {
        "dopamine": 0.8,
        "acetylcholine": 0.6,
        "norepinephrine": 0.5,
    }
"""

# ============================================================================
# Batch Processing
# ============================================================================

BatchData = Dict[str, torch.Tensor]
"""Batch of training/inference data.

Example:
    batch: BatchData = {
        "input": input_spikes,    # (batch_size, timesteps, input_dim)
        "target": target_labels,   # (batch_size,)
        "mask": attention_mask,    # (batch_size, timesteps)
    }
"""

# ============================================================================
# Exported Symbols
# ============================================================================

__all__ = [
    # Component Organization
    "ComponentGraph",
    "ConnectionGraph",
    "TopologyGraph",
    # Multi-Source Pathways
    "SourceOutputs",
    "InputSizes",
    "SynapticWeights",
    "LearningStrategies",
    # Port-Based Routing
    "SourcePort",
    "TargetPort",
    # State Management
    "StateDict",
    "CheckpointMetadata",
    # Diagnostics
    "DiagnosticsDict",
    # Neuromodulation
    "NeuromodulatorLevels",
    # Batch Processing
    "BatchData",
    # Re-exports
    "TYPE_CHECKING",
]
