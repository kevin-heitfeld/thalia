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
from thalia.core.mixins import (
    DeviceMixin,
    ResettableMixin,
    ConfigurableMixin,
    DiagnosticCollectorMixin,
)
from thalia.core.spike_coding import (
    CodingStrategy,
    SpikeCodingConfig,
    SpikeEncoder,
    SpikeDecoder,
    RateEncoder,
    RateDecoder,
    compute_spike_similarity,
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
    DopaminePayload,
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
# NOTE: scalable_attention module removed - attention emerges from brain mechanisms
# - Coincidence detection: STDP, gamma synchrony, theta-gamma coupling
# - Winner-take-all: Lateral inhibition, striatum action selection, PFC gating
# - Phase binding: Gamma oscillations, cross-modal binding pathways
from thalia.core.utils import (
    ensure_batch_dim,
    ensure_batch_dims,
    remove_batch_dim,
    clamp_weights,
    apply_soft_bounds,
    cosine_similarity_safe,
    zeros_like_config,
    ones_like_config,
)
from thalia.core.protocols import (
    Resettable,
    Learnable,
    Forwardable,
    Diagnosable,
    WeightContainer,
    Configurable,
    BrainRegionProtocol,
)
from thalia.core.pathway_protocol import (
    NeuralPathway,
    BaseNeuralPathway,
    Pathway,
)
from thalia.core.diagnostics_mixin import DiagnosticsMixin
from thalia.core.traces import (
    SpikeTrace,
    PairedTraces,
    TraceConfig,
    compute_stdp_update,
    create_trace,
    update_trace,
    compute_decay,
)
from thalia.core.weight_init import (
    InitStrategy,
    WeightInitializer,
)

__all__ = [
    "LIFNeuron",
    "LIFConfig",
    "ConductanceLIF",
    "ConductanceLIFConfig",
    # Mixins
    "DeviceMixin",
    "ResettableMixin",
    "ConfigurableMixin",
    "DiagnosticCollectorMixin",
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
    "DopaminePayload",
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
    # NOTE: Scalable attention removed - emerges from gamma/STDP/lateral-inhibition
    # Utilities
    "ensure_batch_dim",
    "ensure_batch_dims",
    "remove_batch_dim",
    "clamp_weights",
    "apply_soft_bounds",
    "cosine_similarity_safe",
    "zeros_like_config",
    "ones_like_config",
    # Protocols
    "Resettable",
    "Learnable",
    "Forwardable",
    "Diagnosable",
    "WeightContainer",
    "Configurable",
    "BrainRegionProtocol",
    # Pathway protocols
    "NeuralPathway",
    "BaseNeuralPathway",
    "Pathway",
    # Diagnostic mixin
    "DiagnosticsMixin",
    # Spike Coding
    "CodingStrategy",
    "SpikeCodingConfig",
    "SpikeEncoder",
    "SpikeDecoder",
    "RateEncoder",
    "RateDecoder",
    "compute_spike_similarity",
    # Spike Traces
    "SpikeTrace",
    "PairedTraces",
    "TraceConfig",
    "compute_stdp_update",
    "create_trace",
    "update_trace",
    "compute_decay",
    # Weight Initialization
    "InitStrategy",
    "WeightInitializer",
]
