"""Structural protocol for Brain-like objects used by the diagnostics system.

Defines ``BrainLike`` — the minimal interface that :class:`DiagnosticsRecorder`
and :class:`PopulationIndex` require.  Both ``Brain`` and test doubles (e.g.
``RegionTestRunner._FakeBrain``) should conform to this protocol so that
type-checking catches incompatibilities at authoring time rather than at
runtime.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from thalia.typing import RegionName, SynapseId


@runtime_checkable
class BrainLike(Protocol):
    """Minimal Brain interface required by the diagnostics subsystem.

    Attributes expected:
        dt_ms:          Simulation timestep in milliseconds.
        regions:        Mapping of region names to region objects (each exposing
                        ``neuron_populations``, ``stp_modules``, ``named_modules()``,
                        ``neuromodulator_outputs``, ``_population_polarities``,
                        ``_homeostasis``, ``synaptic_weights``, ``get_learning_strategy()``).
        axonal_tracts:  Mapping of :class:`SynapseId` to tract objects (each
                        exposing ``spec.delay_ms``).  May be empty.
    """

    dt_ms: float
    regions: Dict[RegionName, Any]
    axonal_tracts: Dict[SynapseId, Any]
