"""
Cerebellum Submodule - Enhanced Microcircuit Components.

This submodule provides detailed cerebellar microcircuit components:

1. **GranuleCellLayer**: Sparse expansion and pattern separation
2. **EnhancedPurkinjeCell**: Dendritic computation with complex/simple spikes
3. **DeepCerebellarNuclei**: Final output integration

These components can be used to build biologically-detailed cerebellar circuits
with proper granule→Purkinje→DCN information flow.

Author: Thalia Project
Date: December 17, 2025
"""

from __future__ import annotations


from thalia.regions.cerebellum.granule_layer import GranuleCellLayer
from thalia.regions.cerebellum.purkinje_cell import EnhancedPurkinjeCell
from thalia.regions.cerebellum.deep_nuclei import DeepCerebellarNuclei
from thalia.regions.cerebellum.cerebellum import Cerebellum, CerebellumConfig, CerebellumState

__all__ = [
    "GranuleCellLayer",
    "EnhancedPurkinjeCell",
    "DeepCerebellarNuclei",
    "Cerebellum",
    "CerebellumConfig",
    "CerebellumState",
]
