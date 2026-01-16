"""Component and Connection Specifications.

Data classes for representing brain component and connection specifications
used by BrainBuilder and DynamicBrain.

Author: Thalia Project
Date: January 11, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ComponentSpec:
    """Specification for a brain component (region, pathway, or module).

    Attributes:
        name: Instance name (e.g., "my_cortex", "visual_input")
        component_type: Type of component ("region", "pathway", "module")
        registry_name: Component type in registry (e.g., "layered_cortex")
        config_params: Configuration parameters dict
        instance: Instantiated component (set after build())
    """

    name: str
    component_type: str
    registry_name: str
    config_params: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[Any] = None


@dataclass
class ConnectionSpec:
    """Specification for a connection between two components.

    Attributes:
        source: Source component name
        target: Target component name
        pathway_type: Pathway registry name (e.g., "axonal_projection")
        source_port: Output port on source (e.g., 'l23', 'l5')
        target_port: Input port on target (e.g., 'feedforward', 'top_down')
        config_params: Pathway configuration parameters
        instance: Instantiated pathway (set after build())
    """

    source: str
    target: str
    pathway_type: str = "axonal_projection"
    source_port: Optional[str] = None
    target_port: Optional[str] = None
    config_params: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[Any] = None
