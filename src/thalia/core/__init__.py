"""
Core infrastructure: brain coordinator, errors, diagnostics, and protocols.

After ADR-012 directory restructuring, most components have been moved to specialized modules.
This module now contains only the true core infrastructure.

For other components, import from their actual locations:
- Neurons: thalia.components.neurons
- Synapses: thalia.components.synapses
- Learning: thalia.learning
- Pathways: thalia.pathways
- Neuromodulation: thalia.neuromodulation
"""

# Core infrastructure only
from thalia.core.errors import (
    ThaliaError,
    ComponentError,
    ConfigurationError,
    BiologicalPlausibilityError,
    CheckpointError,
    IntegrationError,
    validate_spike_tensor,
    validate_device_consistency,
    validate_weight_matrix,
    validate_positive,
    validate_probability,
    validate_temporal_causality,
)
from thalia.core.diagnostics import (
    DiagnosticLevel,
    DiagnosticsConfig,
    DiagnosticsManager,
    StriatumDiagnostics,
    HippocampusDiagnostics,
    BrainSystemDiagnostics,
)
from thalia.core.protocols.neural import (
    Resettable,
    Learnable,
    Forwardable,
    Diagnosable,
    WeightContainer,
    Configurable,
    NeuralComponentProtocol,
)
from thalia.core.protocols.component import (
    BrainComponent,
    BrainComponentBase,
    BrainComponentMixin,
)
from thalia.core.pathway_state import (
    PathwayState,
    AxonalProjectionState,
)
from thalia.core.region_state import (
    RegionState,
    BaseRegionState,
    save_region_state,
    load_region_state,
    transfer_state,
    get_state_version,
    validate_state_protocol,
)

__all__ = [
    # Error handling
    "ThaliaError",
    "ComponentError",
    "ConfigurationError",
    "BiologicalPlausibilityError",
    "CheckpointError",
    "IntegrationError",
    "validate_spike_tensor",
    "validate_device_consistency",
    "validate_weight_matrix",
    "validate_positive",
    "validate_probability",
    "validate_temporal_causality",
    # Diagnostics
    "DiagnosticLevel",
    "DiagnosticsConfig",
    "DiagnosticsManager",
    "StriatumDiagnostics",
    "HippocampusDiagnostics",
    "BrainSystemDiagnostics",
    # Neural Protocols
    "Resettable",
    "Learnable",
    "Forwardable",
    "Diagnosable",
    "WeightContainer",
    "Configurable",
    "NeuralComponentProtocol",
    # Component Protocols
    "BrainComponent",
    "BrainComponentBase",
    "BrainComponentMixin",
    # State Management (Pathways)
    "PathwayState",
    "AxonalProjectionState",
    # State Management (Regions)
    "RegionState",
    "BaseRegionState",
    "save_region_state",
    "load_region_state",
    "transfer_state",
    "get_state_version",
    "validate_state_protocol",
]
