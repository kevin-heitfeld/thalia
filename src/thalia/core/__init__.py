"""
Core components: neurons, synapses, layers, and networks.
"""

from thalia.core.neuron import LIFNeuron, LIFConfig, ConductanceLIF, ConductanceLIFConfig
from thalia.core.dendritic import (
    DendriticBranch,
    DendriticBranchConfig,
    DendriticNeuron,
    DendriticNeuronConfig,
    compute_branch_selectivity,
    create_clustered_input,
    create_scattered_input,
)
from thalia.core.stp import (
    ShortTermPlasticity,
    STPConfig,
    STPType,
    STPSynapse,
)
from thalia.core.diagnostics import (
    DiagnosticLevel,
    DiagnosticsConfig,
    DiagnosticsManager,
    StriatumDiagnostics,
    HippocampusDiagnostics,
    BrainSystemDiagnostics,
)
from thalia.core.event_system import (
    # Event types
    Event,
    EventType,
    SpikePayload,
    ThetaPayload,
    DopaminePayload,
    # Theta oscillations (canonical implementation)
    ThetaGenerator,
    ThetaState,  # Alias for ThetaGenerator
    ThetaConfig,
    TrialPhase,
    # Event scheduling
    EventScheduler,
    Connection,
    get_axonal_delay,
    AXONAL_DELAYS,
)
from thalia.core.predictive_coding import (
    PredictiveCodingLayer,
    PredictiveCodingConfig,
    PredictiveCodingState,
    HierarchicalPredictiveCoding,
    ErrorType,
)
from thalia.core.scalable_attention import (
    ScalableSpikingAttention,
    ScalableAttentionConfig,
    AttentionType,
    CoincidenceAttention,
    WinnerTakeAllAttention,
    GammaPhaseAttention,
    MultiScaleSpikingAttention,
)

__all__ = [
    "LIFNeuron",
    "LIFConfig",
    "ConductanceLIF",
    "ConductanceLIFConfig",
    # Dendritic computation
    "DendriticBranch",
    "DendriticBranchConfig",
    "DendriticNeuron",
    "DendriticNeuronConfig",
    "compute_branch_selectivity",
    "create_clustered_input",
    "create_scattered_input",
    # Short-term plasticity
    "ShortTermPlasticity",
    "STPConfig",
    "STPType",
    "STPSynapse",
    # Diagnostics
    "DiagnosticLevel",
    "DiagnosticsConfig",
    "DiagnosticsManager",
    "StriatumDiagnostics",
    "HippocampusDiagnostics",
    "BrainSystemDiagnostics",
    # Event system
    "Event",
    "EventType",
    "SpikePayload",
    "ThetaPayload",
    "DopaminePayload",
    # Theta oscillations
    "ThetaGenerator",
    "ThetaState",
    "ThetaConfig",
    "TrialPhase",
    # Event scheduling
    "EventScheduler",
    "Connection",
    "get_axonal_delay",
    "AXONAL_DELAYS",
    # Predictive Coding
    "PredictiveCodingLayer",
    "PredictiveCodingConfig",
    "PredictiveCodingState",
    "HierarchicalPredictiveCoding",
    "ErrorType",
    # Scalable Spiking Attention
    "ScalableSpikingAttention",
    "ScalableAttentionConfig",
    "AttentionType",
    "CoincidenceAttention",
    "WinnerTakeAllAttention",
    "GammaPhaseAttention",
    "MultiScaleSpikingAttention",
]
