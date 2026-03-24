"""Built-in brain preset architectures."""

from .amygdala import connect_amygdala
from .basal_ganglia import add_basal_ganglia_circuit
from .cerebellum import connect_cerebellum
from .cortical import connect_corticocortical, connect_prefrontal, connect_thalamocortical
from .medial_temporal_lobe import add_medial_temporal_lobe_circuit
from .neuromodulators import connect_neuromodulators

__all__ = [
    "add_basal_ganglia_circuit",
    "add_medial_temporal_lobe_circuit",
    "connect_amygdala",
    "connect_cerebellum",
    "connect_corticocortical",
    "connect_neuromodulators",
    "connect_prefrontal",
    "connect_thalamocortical",
]
