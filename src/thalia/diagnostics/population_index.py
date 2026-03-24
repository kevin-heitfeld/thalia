"""Immutable population / region / tract index built from a Brain.

Pure metadata — no simulation state, no mutable buffers.  Built once in
:class:`DiagnosticsRecorder.__init__` and shared read-only by all other
recording components.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from thalia.brain.synapses import NeuromodulatorReceptor
from thalia.learning.strategies import BCMStrategy
from thalia.typing import PopulationName, RegionName, SynapseId

from .brain_protocol import BrainLike
from .diagnostics_config import DiagnosticsConfig


class PopulationIndex:
    """Immutable index metadata extracted from a :class:`Brain`.

    Every field is set once during construction and should be treated as
    read-only.  The :class:`DiagnosticsRecorder`, :class:`BufferManager`,
    and snapshot-building functions all share a single instance.
    """

    __slots__ = (
        "pop_keys",
        "pop_index",
        "n_pops",
        "pop_sizes",
        "region_keys",
        "region_index",
        "n_regions",
        "region_pop_indices",
        "tract_keys",
        "tract_index",
        "n_tracts",
        "stp_keys",
        "stp_configs",
        "n_stp",
        "nm_receptor_keys",
        "n_nm_receptors",
        "nm_source_pop_keys",
        "pop_polarities",
        "tract_delay_ms",
        "homeostasis_target_hz",
        "v_sample_idx",
        "c_sample_idx",
        "learning_keys",
        "learning_strategy_types",
        "bcm_keys",
        "n_learning",
        "n_bcm",
    )

    def __init__(self, brain: BrainLike, config: DiagnosticsConfig) -> None:
        # ── Populations ──────────────────────────────────────────────────
        self.pop_keys: List[Tuple[RegionName, PopulationName]] = []
        for region_name, region in brain.regions.items():
            for pop_name in region.neuron_populations.keys():
                self.pop_keys.append((region_name, pop_name))

        self.pop_index: Dict[Tuple[RegionName, PopulationName], int] = {
            key: idx for idx, key in enumerate(self.pop_keys)
        }
        self.n_pops: int = len(self.pop_keys)

        self.pop_sizes: np.ndarray = np.zeros(self.n_pops, dtype=np.int32)
        for idx, (rn, pn) in enumerate(self.pop_keys):
            self.pop_sizes[idx] = brain.regions[rn].neuron_populations[pn].n_neurons

        # ── Regions ──────────────────────────────────────────────────────
        self.region_keys: List[RegionName] = list(brain.regions.keys())
        self.region_index: Dict[RegionName, int] = {
            k: i for i, k in enumerate(self.region_keys)
        }
        self.n_regions: int = len(self.region_keys)

        self.region_pop_indices: Dict[RegionName, List[int]] = defaultdict(list)
        for idx, (rn, _) in enumerate(self.pop_keys):
            self.region_pop_indices[rn].append(idx)

        # ── Axonal tracts ────────────────────────────────────────────────
        self.tract_keys: List[SynapseId] = list(brain.axonal_tracts.keys())
        self.tract_index: Dict[SynapseId, int] = {
            key: idx for idx, key in enumerate(self.tract_keys)
        }
        self.n_tracts: int = len(self.tract_keys)

        # ── STP modules ─────────────────────────────────────────────────
        self.stp_keys: List[Tuple[RegionName, SynapseId]] = []
        self.stp_configs: List[Tuple[float, float, float]] = []
        for rn, region in brain.regions.items():
            for syn_id, stp_mod in region.stp_modules.items():
                self.stp_keys.append((rn, syn_id))
                cfg = stp_mod.config
                self.stp_configs.append(
                    (float(cfg.U), float(cfg.tau_d), float(cfg.tau_f))
                )
        self.n_stp: int = len(self.stp_keys)

        # ── Neuromodulator receptors ─────────────────────────────────────
        self.nm_receptor_keys: List[Tuple[RegionName, str]] = []
        for rn, region in brain.regions.items():
            for mod_name, mod in region.named_modules():
                if isinstance(mod, NeuromodulatorReceptor) and mod_name:
                    self.nm_receptor_keys.append((rn, mod_name))
        self.n_nm_receptors: int = len(self.nm_receptor_keys)

        self.nm_source_pop_keys: List[Tuple[RegionName, PopulationName]] = []
        for rn, region in brain.regions.items():
            nm_outputs = getattr(region, "neuromodulator_outputs", None)
            if nm_outputs:
                for pop_name in nm_outputs.values():
                    self.nm_source_pop_keys.append((rn, str(pop_name)))

        # ── Static brain metadata ────────────────────────────────────────
        self.pop_polarities: Dict[Tuple[RegionName, PopulationName], str] = {}
        for rn, region in brain.regions.items():
            pols = region._population_polarities
            for pn in region.neuron_populations.keys():
                pol = pols.get(pn)
                self.pop_polarities[(rn, pn)] = str(pol) if pol is not None else "any"

        self.tract_delay_ms: List[float] = [
            brain.axonal_tracts[sid].spec.delay_ms for sid in self.tract_keys
        ]

        self.homeostasis_target_hz: Dict[Tuple[RegionName, PopulationName], float] = {}
        for rn, region in brain.regions.items():
            homeostasis = region._homeostasis
            for pn in region.neuron_populations.keys():
                hs = homeostasis.get(pn, None)
                if hs is not None and hs.target_firing_rate > 0:
                    self.homeostasis_target_hz[(rn, pn)] = (
                        hs.target_firing_rate * 1000.0
                    )

        # ── Neuron sample indices (seed=42 for reproducibility) ──────────
        rng = np.random.default_rng(seed=42)
        V = config.voltage_sample_size
        C = config.conductance_sample_size
        self.v_sample_idx: List[np.ndarray] = []
        self.c_sample_idx: List[np.ndarray] = []
        for size in self.pop_sizes:
            n_v = min(V, int(size))
            self.v_sample_idx.append(
                rng.choice(int(size), size=n_v, replace=False)
                if n_v > 0
                else np.array([], dtype=int)
            )
            n_c = min(C, int(size))
            self.c_sample_idx.append(
                rng.choice(int(size), size=n_c, replace=False)
                if n_c > 0
                else np.array([], dtype=int)
            )

        # ── Learning strategy index ──────────────────────────────────────
        self.learning_keys: List[Tuple[str, SynapseId]] = []
        self.learning_strategy_types: List[str] = []
        self.bcm_keys: List[int] = []
        for rn, region in brain.regions.items():
            for syn_id in region.synaptic_weights.keys():
                strategy = region.get_learning_strategy(syn_id)
                if strategy is not None:
                    idx = len(self.learning_keys)
                    self.learning_keys.append((rn, syn_id))
                    self.learning_strategy_types.append(type(strategy).__name__)
                    if isinstance(strategy, BCMStrategy):
                        self.bcm_keys.append(idx)
        self.n_learning: int = len(self.learning_keys)
        self.n_bcm: int = len(self.bcm_keys)
