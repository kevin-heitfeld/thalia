"""
Utility functions for neuron models, such as splitting excitatory conductance into AMPA and NMDA components.
"""

import torch


def split_excitatory_conductance(g_exc_total: torch.Tensor, nmda_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Split excitatory conductance into AMPA (fast) and NMDA (slow) components.

    Biology: Excitatory synapses contain both AMPA and NMDA receptors.
    AMPA provides fast transmission (tau~5ms), NMDA provides slow temporal
    integration (tau~100ms) with coincidence-detection via Mg²⁺ voltage-gate.

    Args:
        g_exc_total: Total excitatory conductance to split
        nmda_ratio: Fraction of total conductance that is NMDA

    Returns:
        g_ampa: Fast AMPA conductance (80% of total)
        g_nmda: Slow NMDA conductance (20% of total, voltage-gated downstream)
    """
    ampa_ratio = 1.0 - nmda_ratio
    g_ampa = g_exc_total * ampa_ratio
    g_nmda = g_exc_total * nmda_ratio
    return g_ampa, g_nmda
