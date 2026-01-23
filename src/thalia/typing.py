# pyright: strict
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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypedDict

import torch

if TYPE_CHECKING:
    from thalia.core.neural_region import NeuralRegion
    from thalia.learning.rules.strategies import LearningStrategy

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

SourceSpec = Tuple[str, Optional[str]]
"""Specification for a source component with optional port.

Tuple of (region_name, port) where port identifies layer-specific outputs.

Example:
    source_spec: SourceSpec = ("cortex", "l23")  # Layer 2/3 output
    source_spec: SourceSpec = ("hippocampus", None)  # Default output
"""

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

# Structured diagnostic types (Architecture Review 2025-12-22, Tier 1.5)
# =========================================================================


class BaseDiagnostics(TypedDict, total=False):
    """Base diagnostic fields shared by all regions.

    All fields are optional (total=False) to allow partial diagnostics.
    Regions should return as many of these as applicable.

    Fields:
        firing_rate: Mean firing rate across all neurons (spikes/timestep)
        spike_count: Total number of spikes in current timestep
        membrane_mean: Mean membrane potential (normalized units)
        membrane_std: Standard deviation of membrane potential
        weight_mean: Mean synaptic weight value
        weight_std: Standard deviation of synaptic weights
        sparsity: Fraction of active neurons (0.0-1.0)
        health_score: Overall health metric (0.0-1.0, 1.0=healthy)
        has_nans: Whether NaN values were detected
        has_infs: Whether Inf values were detected
        silent_fraction: Fraction of neurons with zero activity
        saturated_fraction: Fraction of neurons at max firing rate
    """

    firing_rate: float
    spike_count: int
    membrane_mean: float
    membrane_std: float
    weight_mean: float
    weight_std: float
    sparsity: float
    health_score: float
    has_nans: bool
    has_infs: bool
    silent_fraction: float
    saturated_fraction: float


class LayeredCortexDiagnostics(BaseDiagnostics):
    """Extended diagnostics for LayeredCortex region.

    Includes layer-specific metrics (L4, L2/3, L5, L6).

    Fields:
        l4_firing_rate: Layer 4 firing rate
        l23_firing_rate: Layer 2/3 firing rate
        l5_firing_rate: Layer 5 firing rate
        l6_firing_rate: Layer 6 firing rate (if present)
        l4_to_l23_strength: Effective synaptic strength L4→L2/3
        l23_recurrence: L2/3 recurrent connection strength
        l23_to_l5_strength: Effective synaptic strength L2/3→L5
        feedforward_balance: Ratio of feedforward to feedback drive
        gamma_power: Gamma oscillation power (20-80 Hz)
        theta_gamma_coupling: Theta-gamma cross-frequency coupling
    """

    l4_firing_rate: float
    l23_firing_rate: float
    l5_firing_rate: float
    l6_firing_rate: float
    l4_to_l23_strength: float
    l23_recurrence: float
    l23_to_l5_strength: float
    feedforward_balance: float
    gamma_power: float
    theta_gamma_coupling: float


class StriatumDiagnostics(BaseDiagnostics):
    """Extended diagnostics for Striatum region.

    Includes D1/D2 pathway metrics and action selection statistics.

    Fields:
        d1_firing_rate: D1 pathway firing rate (Go signal)
        d2_firing_rate: D2 pathway firing rate (No-Go signal)
        d1_d2_balance: Ratio of D1 to D2 activity
        selected_action: Currently selected action index
        action_confidence: Confidence in selected action (0.0-1.0)
        exploration_rate: Current exploration probability
        eligibility_mean: Mean eligibility trace value
        eligibility_std: Standard deviation of eligibility traces
        dopamine_level: Current dopamine concentration (0.0-1.0)
        q_value_mean: Mean estimated Q-value across actions
        q_value_std: Standard deviation of Q-values
        winning_vote_margin: Margin between winner and runner-up
    """

    d1_firing_rate: float
    d2_firing_rate: float
    d1_d2_balance: float
    selected_action: int
    action_confidence: float
    exploration_rate: float
    eligibility_mean: float
    eligibility_std: float
    dopamine_level: float
    q_value_mean: float
    q_value_std: float
    winning_vote_margin: float


class HippocampusDiagnostics(BaseDiagnostics):
    """Extended diagnostics for TrisynapticHippocampus region.

    Includes DG/CA3/CA1 subregion metrics and memory statistics.

    Fields:
        dg_firing_rate: Dentate gyrus firing rate
        ca3_firing_rate: CA3 firing rate
        ca1_firing_rate: CA1 firing rate
        dg_sparsity: DG sparsity (pattern separation quality)
        ca3_recurrence: CA3 recurrent connection strength
        ca1_mismatch: CA1 prediction error (novelty detection)
        theta_phase: Current theta oscillation phase (0-2π)
        encoding_mode: Whether in encoding (True) or retrieval (False)
        memory_capacity_used: Fraction of memory capacity used (0.0-1.0)
        last_retrieval_similarity: Similarity of last retrieval (0.0-1.0)
        acetylcholine_level: Current ACh concentration (0.0-1.0)
    """

    dg_firing_rate: float
    ca3_firing_rate: float
    ca1_firing_rate: float
    dg_sparsity: float
    ca3_recurrence: float
    ca1_mismatch: float
    theta_phase: float
    encoding_mode: bool
    memory_capacity_used: float
    last_retrieval_similarity: float
    acetylcholine_level: float


class PrefrontalDiagnostics(BaseDiagnostics):
    """Extended diagnostics for Prefrontal Cortex region.

    Includes working memory and goal hierarchy metrics.

    Fields:
        working_memory_load: Number of items in working memory
        working_memory_capacity: Maximum working memory capacity
        gate_open: Whether input gate is open (dopamine-gated)
        maintenance_current: Current maintaining working memory
        distractor_resistance: Resistance to distractors (0.0-1.0)
        active_goal_level: Hierarchy level of active goal (0=lowest)
        n_active_goals: Number of simultaneously active goals
        goal_conflict: Degree of conflict between active goals
    """

    working_memory_load: int
    working_memory_capacity: int
    gate_open: bool
    maintenance_current: float
    distractor_resistance: float
    active_goal_level: int
    n_active_goals: int
    goal_conflict: float


class ThalamicRelayDiagnostics(BaseDiagnostics):
    """Extended diagnostics for Thalamic Relay region.

    Includes relay and TRN (reticular nucleus) metrics.

    Fields:
        relay_firing_rate: Relay neuron firing rate
        trn_firing_rate: TRN neuron firing rate
        gating_strength: Effective gating from TRN (0.0-1.0)
        attention_modulation: Attention-based gain modulation
        burst_mode: Whether in burst mode (True) or tonic mode (False)
        relay_gain: Current relay gain factor
        feedback_inhibition: Strength of L6 feedback to TRN
    """

    relay_firing_rate: float
    trn_firing_rate: float
    gating_strength: float
    attention_modulation: float
    burst_mode: bool
    relay_gain: float
    feedback_inhibition: float


class CerebellumDiagnostics(BaseDiagnostics):
    """Extended diagnostics for Cerebellum region.

    Includes Purkinje cell and climbing fiber error metrics.

    Fields:
        purkinje_firing_rate: Purkinje cell firing rate
        granule_firing_rate: Granule cell firing rate
        climbing_fiber_error: Current error signal from climbing fibers
        parallel_fiber_strength: Parallel fiber synaptic strength
        learning_rate_modulated: Current effective learning rate
        error_corrective_magnitude: Magnitude of error-driven updates
    """

    purkinje_firing_rate: float
    granule_firing_rate: float
    climbing_fiber_error: float
    parallel_fiber_strength: float
    learning_rate_modulated: float
    error_corrective_magnitude: float


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
    "SourceSpec",
    "SourcePort",
    "TargetPort",
    # State Management
    "StateDict",
    "CheckpointMetadata",
    # Diagnostics
    # DiagnosticsDict removed - use TypedDict subclasses below instead
    "BaseDiagnostics",
    "LayeredCortexDiagnostics",
    "StriatumDiagnostics",
    "HippocampusDiagnostics",
    "PrefrontalDiagnostics",
    "ThalamicRelayDiagnostics",
    "CerebellumDiagnostics",
    # Neuromodulation
    "NeuromodulatorLevels",
    # Batch Processing
    "BatchData",
    # Re-exports
    "TYPE_CHECKING",
]
