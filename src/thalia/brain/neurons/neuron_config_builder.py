"""Builder functions for constructing neuron configs from population configs.

Provides :func:`build_conductance_lif_config` and :func:`build_two_compartment_config`
that take a :class:`~thalia.brain.configs.neural_region.NeuralPopulationConfig` (the single
source of truth for per-population biophysical center values) and expand them into
fully-constructed :class:`ConductanceLIFConfig` or :class:`TwoCompartmentLIFConfig`
with automatic per-neuron heterogeneity via the ``heterogeneous_*()`` factory functions.

This eliminates the boilerplate where every region manually calls 6+ heterogeneous_*()
functions and hard-codes the same universal synaptic defaults (E_E=3.0, tau_nmda=100.0, etc.).
"""

from __future__ import annotations

from typing import Any, Union

import torch

from thalia.brain.configs.neural_region import NeuralPopulationConfig

from .conductance_lif_neuron import (
    ConductanceLIFConfig,
    heterogeneous_adapt_increment,
    heterogeneous_dendrite_coupling,
    heterogeneous_g_L,
    heterogeneous_noise_std,
    heterogeneous_tau_adapt,
    heterogeneous_tau_mem,
    heterogeneous_v_reset,
    heterogeneous_v_threshold,
)
from .two_compartment_lif_neuron import TwoCompartmentLIFConfig


def _build_common_lif_fields(
    pop_config: NeuralPopulationConfig,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    tau_ref: float,
    g_L: float,
    tau_E: float,
    tau_I: float,
    dendrite_coupling: float | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the shared field dict for ConductanceLIF/TwoCompartmentLIF configs.

    The 6 center values from *pop_config* (tau_mem_ms, v_threshold, v_reset,
    noise_std, adapt_increment, tau_adapt_ms) are expanded to per-neuron
    heterogeneous tensors.  Remaining ConductanceLIFConfig parameters use
    biologically standard defaults that can be overridden via *kwargs*.

    CV overrides for the heterogeneous functions can be passed as keyword
    arguments (``tau_mem_cv``, ``v_threshold_cv``, ``g_L_cv``, ``dendrite_cv``).
    """
    tau_mem_cv = kwargs.pop("tau_mem_cv", 0.20)
    v_threshold_cv = kwargs.pop("v_threshold_cv", 0.30)
    g_L_cv = kwargs.pop("g_L_cv", 0.30)
    dendrite_cv = kwargs.pop("dendrite_cv", 0.20)

    has_adapt = pop_config.adapt_increment > 0

    fields: dict[str, Any] = dict(
        # Per-population center values → per-neuron heterogeneous tensors
        tau_mem_ms=heterogeneous_tau_mem(pop_config.tau_mem_ms, n_neurons, device, cv=tau_mem_cv),
        v_threshold=heterogeneous_v_threshold(pop_config.v_threshold, n_neurons, device, cv=v_threshold_cv),
        v_reset=heterogeneous_v_reset(pop_config.v_reset, n_neurons, device),
        noise_std=heterogeneous_noise_std(pop_config.noise_std, n_neurons, device),
        adapt_increment=(
            heterogeneous_adapt_increment(pop_config.adapt_increment, n_neurons, device)
            if has_adapt
            else 0.0
        ),
        tau_adapt_ms=(
            heterogeneous_tau_adapt(pop_config.tau_adapt_ms, n_neurons, device)
            if has_adapt
            else pop_config.tau_adapt_ms
        ),
        # Synaptic / biophysical parameters with universal defaults
        tau_ref=tau_ref,
        g_L=heterogeneous_g_L(g_L, n_neurons, device, cv=g_L_cv),
        E_E=kwargs.pop("E_E", 3.0),
        E_I=kwargs.pop("E_I", -0.5),
        tau_E=tau_E,
        tau_I=tau_I,
        tau_nmda=kwargs.pop("tau_nmda", 100.0),
        E_nmda=kwargs.pop("E_nmda", 3.0),
        tau_GABA_B=kwargs.pop("tau_GABA_B", 400.0),
        E_GABA_B=kwargs.pop("E_GABA_B", -0.8),
        noise_tau_ms=kwargs.pop("noise_tau_ms", 3.0),
        E_adapt=kwargs.pop("E_adapt", -0.5),
    )

    if dendrite_coupling is not None:
        fields["dendrite_coupling_scale"] = heterogeneous_dendrite_coupling(
            dendrite_coupling, n_neurons, device, cv=dendrite_cv,
        )

    # Pass through remaining kwargs (enable_ih, g_h_max, enable_t_channels, g_T,
    # mg_conc, g_nmda_max, etc.) directly to the config constructor.
    fields.update(kwargs)
    return fields


def build_conductance_lif_config(
    pop_config: NeuralPopulationConfig,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    tau_ref: float = 5.0,
    g_L: float = 0.05,
    tau_E: float = 5.0,
    tau_I: float = 10.0,
    dendrite_coupling: float = 0.2,
    **kwargs: Any,
) -> ConductanceLIFConfig:
    """Build a :class:`ConductanceLIFConfig` from a :class:`NeuralPopulationConfig`.

    The 6 center values in *pop_config* (tau_mem_ms, v_threshold, v_reset,
    noise_std, adapt_increment, tau_adapt_ms) are automatically expanded to
    per-neuron heterogeneous tensors.  All other parameters use biologically
    standard defaults and can be overridden via keyword arguments.

    Supported CV overrides: ``tau_mem_cv``, ``v_threshold_cv``, ``g_L_cv``,
    ``dendrite_cv``.  Any extra kwargs are forwarded to :class:`ConductanceLIFConfig`
    (e.g. ``enable_ih``, ``g_h_max``, ``g_nmda_max``).
    """
    fields = _build_common_lif_fields(
        pop_config, n_neurons, device,
        tau_ref=tau_ref, g_L=g_L, tau_E=tau_E, tau_I=tau_I,
        dendrite_coupling=dendrite_coupling, **kwargs,
    )
    return ConductanceLIFConfig(**fields)


def build_two_compartment_config(
    pop_config: NeuralPopulationConfig,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    tau_ref: float = 5.0,
    g_L: float = 0.05,
    tau_E: float = 5.0,
    tau_I: float = 10.0,
    # Two-compartment specific
    g_c: float = 0.05,
    C_d: float = 0.5,
    g_L_d: float = 0.03,
    bap_amplitude: float = 0.3,
    theta_Ca: float = 2.0,
    g_Ca_spike: float = 0.30,
    tau_Ca_ms: float = 20.0,
    **kwargs: Any,
) -> TwoCompartmentLIFConfig:
    """Build a :class:`TwoCompartmentLIFConfig` from a :class:`NeuralPopulationConfig`.

    Same as :func:`build_conductance_lif_config` but returns a two-compartment
    config with soma-dendrite coupling, BAP, and Ca²⁺ spike parameters.
    ``dendrite_coupling_scale`` is left at the default (TwoCompartmentLIF uses
    ``g_c`` for soma-dendrite coupling instead).
    """
    fields = _build_common_lif_fields(
        pop_config, n_neurons, device,
        tau_ref=tau_ref, g_L=g_L, tau_E=tau_E, tau_I=tau_I,
        dendrite_coupling=None,  # TwoCompartment uses g_c instead
        **kwargs,
    )
    fields.update(
        g_c=g_c, C_d=C_d, g_L_d=g_L_d,
        bap_amplitude=bap_amplitude, theta_Ca=theta_Ca,
        g_Ca_spike=g_Ca_spike, tau_Ca_ms=tau_Ca_ms,
    )
    return TwoCompartmentLIFConfig(**fields)
