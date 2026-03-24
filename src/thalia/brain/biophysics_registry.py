"""Read-only biophysics registry built by introspecting a constructed Brain.

Provides a queryable snapshot of all neuron population parameters across the
entire brain, enabling cross-region comparison, diagnostic output, and
consistency checking — without touching existing config infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class PopulationBiophysics:
    """Immutable biophysical snapshot for one neuron population."""

    region: str
    population: str
    n_neurons: int
    neuron_type: str  # "ConductanceLIF" | "TwoCompartmentLIF" | "AcetylcholineNeuron" | ...

    # --- Membrane parameters (mean values; heterogeneous across neurons) ---
    tau_mem_ms: float
    v_threshold: float
    v_reset: float
    g_L: float
    noise_std: float

    # --- Adaptation ---
    adapt_increment: float
    tau_adapt_ms: float

    # --- Reversal potentials (scalar, shared by all neurons in population) ---
    E_E: float
    E_I: float

    # --- Synaptic time constants ---
    tau_E: float
    tau_I: float
    tau_nmda: float
    tau_ref: float

    # --- NMDA ---
    mg_conc: float

    # --- Dendritic coupling (single-compartment estimate) ---
    dendrite_coupling_scale: float

    # --- Optional: I_h pacemaker ---
    enable_ih: bool = False
    g_h_max: Optional[float] = None

    # --- Optional: T-type Ca²⁺ channels ---
    enable_t_channels: bool = False
    g_T: Optional[float] = None

    # --- Optional: Two-compartment parameters ---
    g_c: Optional[float] = None
    C_d: Optional[float] = None
    g_L_d: Optional[float] = None
    bap_amplitude: Optional[float] = None
    theta_Ca: Optional[float] = None
    g_Ca_spike: Optional[float] = None
    tau_Ca_ms: Optional[float] = None


def _tensor_mean(value: Union[float, torch.Tensor]) -> float:
    """Extract scalar mean from a scalar or per-neuron tensor."""
    if isinstance(value, torch.Tensor):
        return value.mean().item()
    return float(value)


def _cmp_gt(val: Any, target: Any) -> bool:
    return val is not None and val > target

def _cmp_lt(val: Any, target: Any) -> bool:
    return val is not None and val < target

def _cmp_gte(val: Any, target: Any) -> bool:
    return val is not None and val >= target

def _cmp_lte(val: Any, target: Any) -> bool:
    return val is not None and val <= target


def _extract_population(
    region_name: str,
    population_name: str,
    neurons: Any,
) -> PopulationBiophysics:
    """Extract a PopulationBiophysics snapshot from a live neuron module."""
    from .neurons import TwoCompartmentLIF

    config = neurons.config
    neuron_type = type(neurons).__name__

    # Per-neuron tensor buffers → mean
    tau_mem_ms = _tensor_mean(neurons.tau_mem_ms)
    v_threshold = _tensor_mean(neurons.v_threshold)
    v_reset = _tensor_mean(neurons.v_reset)
    g_L = _tensor_mean(neurons.g_L)
    noise_std = _tensor_mean(getattr(neurons, "noise_std", config.noise_std))
    adapt_increment = _tensor_mean(neurons.adapt_increment)
    tau_adapt_ms = _tensor_mean(getattr(neurons, "tau_adapt_ms", config.tau_adapt_ms))
    dendrite_coupling_scale = _tensor_mean(
        getattr(neurons, "dendrite_coupling_scale", config.dendrite_coupling_scale)
    )

    # Scalar config values
    kwargs: Dict[str, Any] = dict(
        region=region_name,
        population=population_name,
        n_neurons=neurons.n_neurons,
        neuron_type=neuron_type,
        tau_mem_ms=tau_mem_ms,
        v_threshold=v_threshold,
        v_reset=v_reset,
        g_L=g_L,
        noise_std=noise_std,
        adapt_increment=adapt_increment,
        tau_adapt_ms=tau_adapt_ms,
        E_E=float(config.E_E),
        E_I=float(config.E_I),
        tau_E=float(config.tau_E),
        tau_I=float(config.tau_I),
        tau_nmda=float(config.tau_nmda),
        tau_ref=float(config.tau_ref),
        mg_conc=float(config.mg_conc),
        dendrite_coupling_scale=dendrite_coupling_scale,
        enable_ih=bool(config.enable_ih),
        g_h_max=float(config.g_h_max) if config.enable_ih else None,
        enable_t_channels=bool(config.enable_t_channels),
        g_T=float(config.g_T) if config.enable_t_channels else None,
    )

    # Two-compartment extras
    if isinstance(neurons, TwoCompartmentLIF):
        kwargs.update(
            g_c=float(config.g_c),
            C_d=float(config.C_d),
            g_L_d=float(config.g_L_d),
            bap_amplitude=float(config.bap_amplitude),
            theta_Ca=float(config.theta_Ca),
            g_Ca_spike=float(config.g_Ca_spike),
            tau_Ca_ms=float(config.tau_Ca_ms),
        )

    return PopulationBiophysics(**kwargs)


class BiophysicsRegistry:
    """Queryable, read-only registry of all neuron population parameters.

    Built once at brain construction time by introspecting the live
    ``brain.regions`` → ``region.neuron_populations`` hierarchy.

    Example usage::

        registry = brain.biophysics
        # Compare a parameter across all populations
        for key, val in registry.compare("tau_mem_ms").items():
            print(f"{key}: {val:.1f} ms")

        # Filter populations
        fast = registry.populations_with(tau_mem_ms_lt=12.0)
    """

    def __init__(self, entries: List[PopulationBiophysics]) -> None:
        self._entries: Tuple[PopulationBiophysics, ...] = tuple(entries)
        self._index: Dict[Tuple[str, str], PopulationBiophysics] = {
            (e.region, e.population): e for e in entries
        }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_brain(cls, brain: Any) -> BiophysicsRegistry:
        """Introspect a constructed Brain and build the registry."""
        entries: List[PopulationBiophysics] = []
        for region_name, region in brain.regions.items():
            for pop_name, neurons in region.neuron_populations.items():
                entries.append(_extract_population(region_name, pop_name, neurons))
        return cls(entries)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def all_populations(self) -> List[PopulationBiophysics]:
        """Return all population snapshots."""
        return list(self._entries)

    def get(self, region: str, population: str) -> PopulationBiophysics:
        """Get biophysical snapshot for a specific (region, population) pair.

        Raises:
            KeyError: If the (region, population) pair is not found.
        """
        return self._index[(region, population)]

    def compare(self, param: str) -> Dict[Tuple[str, str], float]:
        """Compare a single numeric parameter across all populations.

        Args:
            param: Field name on :class:`PopulationBiophysics` (e.g. ``"tau_mem_ms"``).

        Returns:
            Dict mapping ``(region, population)`` to the parameter's value.

        Raises:
            AttributeError: If *param* is not a field on ``PopulationBiophysics``.
        """
        result: Dict[Tuple[str, str], float] = {}
        for e in self._entries:
            val = getattr(e, param)
            if val is not None:
                result[(e.region, e.population)] = float(val)
        return result

    def populations_with(self, **criteria: Any) -> List[PopulationBiophysics]:
        """Filter populations by parameter ranges or exact values.

        Supported suffixes:
            - ``_gt``:  greater-than  (e.g. ``tau_mem_ms_gt=15.0``)
            - ``_lt``:  less-than     (e.g. ``tau_mem_ms_lt=25.0``)
            - ``_gte``: greater-or-equal
            - ``_lte``: less-or-equal
            - (none):   exact match   (e.g. ``neuron_type="TwoCompartmentLIF"``)
        """
        _SUFFIXES = {"_gte": 4, "_lte": 4, "_gt": 3, "_lt": 3}

        results: List[PopulationBiophysics] = list(self._entries)
        for key, target in criteria.items():
            matched_suffix = False
            for suffix, length in _SUFFIXES.items():
                if key.endswith(suffix):
                    param = key[:-length]
                    if suffix == "_gt":
                        results = [e for e in results if _cmp_gt(getattr(e, param, None), target)]
                    elif suffix == "_lt":
                        results = [e for e in results if _cmp_lt(getattr(e, param, None), target)]
                    elif suffix == "_gte":
                        results = [e for e in results if _cmp_gte(getattr(e, param, None), target)]
                    elif suffix == "_lte":
                        results = [e for e in results if _cmp_lte(getattr(e, param, None), target)]
                    matched_suffix = True
                    break
            if not matched_suffix:
                results = [e for e in results if getattr(e, key, None) == target]
        return results

    @property
    def total_neurons(self) -> int:
        """Total neuron count across all populations."""
        return sum(e.n_neurons for e in self._entries)

    def summary_table(self) -> str:
        """Return a formatted text summary of all populations.

        Useful for diagnostics and logging.
        """
        lines = [
            f"{'Region':<24} {'Population':<20} {'Type':<24} {'N':>6} "
            f"{'τ_mem':>7} {'V_thr':>7} {'g_L':>7} {'adapt':>7} {'noise':>7}"
        ]
        lines.append("-" * len(lines[0]))
        for e in self._entries:
            lines.append(
                f"{e.region:<24} {e.population:<20} {e.neuron_type:<24} {e.n_neurons:>6} "
                f"{e.tau_mem_ms:>7.1f} {e.v_threshold:>7.3f} {e.g_L:>7.4f} "
                f"{e.adapt_increment:>7.4f} {e.noise_std:>7.4f}"
            )
        lines.append(f"\nTotal populations: {len(self._entries)}  |  Total neurons: {self.total_neurons}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"BiophysicsRegistry({len(self._entries)} populations, {self.total_neurons} neurons)"
