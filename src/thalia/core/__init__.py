"""
Core components: neurons, synapses, layers, and networks.
"""

from thalia.core.neuron import LIFNeuron, LIFConfig, ConductanceLIF, ConductanceLIFConfig
# Note: BaseNeuronConfig is in thalia.config.neuron_config, not imported here to avoid circular import
from thalia.core.neuron_constants import (
    # Membrane time constants
    TAU_MEM_STANDARD, TAU_MEM_FAST, TAU_MEM_SLOW,
    # Synaptic time constants
    TAU_SYN_EXCITATORY, TAU_SYN_INHIBITORY, TAU_SYN_NMDA,
    # Voltage parameters
    V_THRESHOLD_STANDARD, V_RESET_STANDARD, V_REST_STANDARD,
    # Reversal potentials
    E_LEAK, E_EXCITATORY, E_INHIBITORY,
    # Conductances
    G_LEAK_STANDARD, G_LEAK_FAST, G_LEAK_SLOW,
    # Presets
    STANDARD_PYRAMIDAL, FAST_SPIKING_INTERNEURON,
)
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
from thalia.core.stp_presets import (
    STP_PRESETS,
    STPPreset,
    get_stp_config,
    list_presets,
)
from thalia.core.component_registry import (
    ComponentRegistry,
    register_region,
    register_pathway,
    register_module,
)
from thalia.core.component_protocol import (
    BrainComponent,
    BrainComponentBase,
    BrainComponentMixin,
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
from thalia.core.eligibility_utils import (
    EligibilityTraceManager,
    STDPConfig,
)
from thalia.core.base_manager import (
    BaseManager,
    ManagerContext,
)
from thalia.core.diagnostics import (
    DiagnosticLevel,
    DiagnosticsConfig,
    DiagnosticsManager,
    StriatumDiagnostics,
    HippocampusDiagnostics,
    BrainSystemDiagnostics,
)
# Event system moved to thalia.events package (see thalia.events for imports)
# from thalia.events import (
#     Event, EventType, SpikePayload, EventScheduler,
#     Connection, get_axonal_delay, AXONAL_DELAYS,
#     ParallelExecutor, EventDrivenCortex, etc.
# )
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
    clamp_weights,
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
    # Neuron models
    "LIFNeuron",
    "LIFConfig",
    "ConductanceLIF",
    "ConductanceLIFConfig",
    # Note: BaseNeuronConfig not exported to avoid circular import
    # Neuron constants
    "TAU_MEM_STANDARD",
    "TAU_MEM_FAST",
    "TAU_MEM_SLOW",
    "TAU_SYN_EXCITATORY",
    "TAU_SYN_INHIBITORY",
    "TAU_SYN_NMDA",
    "V_THRESHOLD_STANDARD",
    "V_RESET_STANDARD",
    "V_REST_STANDARD",
    "E_LEAK",
    "E_EXCITATORY",
    "E_INHIBITORY",
    "G_LEAK_STANDARD",
    "G_LEAK_FAST",
    "G_LEAK_SLOW",
    "STANDARD_PYRAMIDAL",
    "FAST_SPIKING_INTERNEURON",
    # Dendritic processing
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
    # STP presets
    "STP_PRESETS",
    "STPPreset",
    "get_stp_config",
    "list_presets",
    # Component Registry
    "ComponentRegistry",
    "register_region",
    "register_pathway",
    "register_module",
    # Component Protocol
    "BrainComponent",
    "BrainComponentBase",
    "BrainComponentMixin",
    # Diagnostics
    "DiagnosticLevel",
    "DiagnosticsConfig",
    "DiagnosticsManager",
    "StriatumDiagnostics",
    "HippocampusDiagnostics",
    "BrainSystemDiagnostics",
    # Event system moved to thalia.events package
    # (Event, EventType, SpikePayload, EventScheduler, Connection, etc.)
    # Predictive Coding
    "PredictiveCodingLayer",
    "PredictiveCodingConfig",
    "PredictiveCodingState",
    "HierarchicalPredictiveCoding",
    "ErrorType",
    # NOTE: Scalable attention removed - emerges from gamma/STDP/lateral-inhibition
    # Utilities
    "clamp_weights",
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
    # Note: BaseNeuralPathway was consolidated into NeuralComponent (ADR-008)
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
    # Eligibility Traces
    "EligibilityTraceManager",
    "STDPConfig",
    # Manager Base Classes
    "BaseManager",
    "ManagerContext",
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

# Note: EventDrivenBrain has circular import issues and should be imported directly:
# from thalia.core.brain import EventDrivenBrain
