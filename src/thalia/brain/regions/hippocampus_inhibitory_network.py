"""
Hippocampal Inhibitory Network with OLM Cells.

The hippocampus has distinct inhibitory interneuron populations compared to cortex.
This module implements the key types for CA1/CA3:

Hippocampal Interneuron Types:
===============================

1. **PV+ Basket Cells (35% of inhibitory neurons)**
   - Perisomatic inhibition (soma/axon initial segment)
   - Fast-spiking (tau_mem ~7ms)
   - Phase-lock to theta peaks
   - Generate gamma oscillations

2. **OLM Cells (Oriens-Lacunosum Moleculare, 30% of inhibitory)**
   - SST+ interneurons in stratum oriens
   - Dendrit targeting (apical dendrites in lacunosum-moleculare)
   - Regular-spiking (tau_mem ~25ms)
   - **Phase-lock to theta troughs** (critical for encoding/retrieval)
   - Strong adaptation creates burst-pause dynamics

3. **Bistratified Cells (20% of inhibitory)**
   - Target proximal and distal dendrites
   - Phase-lock opposite to basket cells
   - Medium dynamics (tau_mem ~12ms)

4. **Other interneurons (15%)**
   - Simplified representation

Key Circuit for Theta Modulation:
==================================

Septal GABA → OLM cells (inhibition at theta peaks)
                 ↓
            OLM rebounds at theta troughs (rebound bursting)
                 ↓
            OLM → CA1 apical dendrites (suppression)
                 ↓
            Blocks EC→CA1 retrieval during encoding

This creates **emergent encoding/retrieval separation** without hardcoded modulation!
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union, cast

import torch
import torch.nn as nn

from thalia.brain.neurons import (
    ConductanceLIFConfig,
    ConductanceLIF,
    heterogeneous_dendrite_coupling,
    heterogeneous_noise_std,
    heterogeneous_tau_adapt,
    heterogeneous_tau_mem,
    heterogeneous_v_reset,
    heterogeneous_v_threshold,
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    split_excitatory_conductance,
)
from thalia.brain.regions.population_names import HippocampusPopulation
from thalia.brain.gap_junctions import GapJunctionCoupling
from thalia.brain.synapses import WeightInitializer
from thalia.typing import ConductanceTensor, GapJunctionReversal, PopulationName, PopulationPolarity


class HippocampalInhibitoryNetwork(nn.Module):
    """Hippocampal inhibitory network with OLM cells for theta modulation.

    Key differences from cortical inhibitory network:
    - OLM cells (not SST Martinotti)
    - No VIP cells (less disinhibition in hippocampus)
    - Bistratified cells (unique to hippocampus)
    - Septal input integration (for OLM phase-locking)
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        population_name: PopulationName,
        pyr_size: int,
        total_inhib_fraction: float,
        v_threshold_olm: float,
        v_threshold_bistratified: float,
        _create_and_register_neurons_fn: Callable[[], ConductanceLIF],
        dt_ms: float,
        device: Union[str, torch.device],
    ):
        """Initialize hippocampal inhibitory network.

        Args:
            population_name: Population identifier (e.g., "ca1", "ca1_inhibitory")
            pyr_size: Number of pyramidal neurons
            total_inhib_fraction: Fraction of pyramidal count
            v_threshold_olm: Spike threshold for OLM cells.
            v_threshold_bistratified: Spike threshold for bistratified cells.
            _create_and_register_neurons_fn: Function to create and register neurons in parent NeuralRegion
            dt_ms: Timestep in milliseconds
            device: Torch device
        """
        super().__init__()

        self.pyr_size = pyr_size
        self.dt_ms = dt_ms

        # Total inhibitory count
        total_inhib = int(pyr_size * total_inhib_fraction)

        # =====================================================================
        # CELL TYPE PROPORTIONS (hippocampal interneurons)
        # =====================================================================
        pv_frac = 0.35        # PV+ basket cells
        olm_frac = 0.30       # OLM cells (SST+)
        bistratified_frac = 0.20  # Bistratified cells
        # other_frac = 0.15     # Other interneurons (simplified)

        self.n_pv = int(total_inhib * pv_frac)
        self.n_olm = int(total_inhib * olm_frac)
        self.n_bistratified = int(total_inhib * bistratified_frac)
        self.n_other = total_inhib - self.n_pv - self.n_olm - self.n_bistratified

        # =====================================================================
        # CREATE NEURON POPULATIONS
        # =====================================================================

        pv_pop_name = cast(HippocampusPopulation, f"{population_name}_pv")
        olm_pop_name = cast(HippocampusPopulation, f"{population_name}_olm")
        bistratified_pop_name = cast(HippocampusPopulation, f"{population_name}_bistratified")

        # PV basket cells: Fast-spiking perisomatic inhibition
        # Threshold history: 0.9→0.7→0.8→1.0.  At 0.8, DG_PV hit 66 Hz and CA2_PV
        # hit 94 Hz (both far above 5-20 Hz target).  Raising to 1.0 requires 25%
        # more excitatory drive to fire, which combined with reduced E→PV w_max for
        # DG and CA2 should bring PV rates into the 20-40 Hz range.
        self.pv_neurons: ConductanceLIF = _create_and_register_neurons_fn(
            population_name=pv_pop_name,
            n_neurons=self.n_pv,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(7.0, self.n_pv, device, cv=0.10),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(1.0, self.n_pv, device, cv=0.06),  # Raised 0.8→1.0
                tau_ref=5.0,
                g_L=heterogeneous_g_L(0.05, self.n_pv, device, cv=0.08),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, self.n_pv, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=50.0,
                adapt_increment=0.0,  # PV/FSI are non-adapting (Kv3 channels)
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.n_pv, device, cv=0.20),
            ),
        )

        # OLM cells: CRITICAL for theta phase-locking
        # High adaptation creates burst-pause dynamics for theta rhythm
        self.olm_neurons: ConductanceLIF = _create_and_register_neurons_fn(
            population_name=olm_pop_name,
            n_neurons=self.n_olm,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(25.0, self.n_olm, device),
                v_reset=heterogeneous_v_reset(-0.05, self.n_olm, device),
                v_threshold=heterogeneous_v_threshold(v_threshold_olm, self.n_olm, device),
                tau_ref=5.0,
                g_L=heterogeneous_g_L(0.05, self.n_olm, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.07, self.n_olm, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(3000.0, self.n_olm, device),
                adapt_increment=heterogeneous_adapt_increment(0.04, self.n_olm, device),
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.n_olm, device, cv=0.20),
            ),
        )

        # Bistratified cells: Dendritic targeting
        self.bistratified_neurons: ConductanceLIF = _create_and_register_neurons_fn(
            population_name=bistratified_pop_name,
            n_neurons=self.n_bistratified,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(12.0, self.n_bistratified, device),
                v_reset=heterogeneous_v_reset(-0.05, self.n_bistratified, device),
                v_threshold=heterogeneous_v_threshold(v_threshold_bistratified, self.n_bistratified, device),
                tau_ref=5.0,
                g_L=heterogeneous_g_L(0.05, self.n_bistratified, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.06, self.n_bistratified, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(3000.0, self.n_bistratified, device),
                adapt_increment=heterogeneous_adapt_increment(0.02, self.n_bistratified, device),
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.n_bistratified, device, cv=0.20),
            ),
        )

        # =====================================================================
        # GAP JUNCTIONS (PV cells only — electrical coupling, not synaptic)
        # =====================================================================
        # Gap junctions are electrical (V-difference coupling), not synaptic, so
        # they are not registered via _add_internal_connection().  Computed as
        # proper (g_gap, E_gap) and passed to the neuron's g_gap_input channel.
        pv_gap_matrix = WeightInitializer.sparse_uniform_no_autapses(
            n_input=self.n_pv,
            n_output=self.n_pv,
            connectivity=0.5,
            w_min=0.0,
            w_max=0.0003,
            device=device,
        )
        # Make symmetric (gap junctions are bidirectional)
        pv_gap_matrix.data = (pv_gap_matrix.data + pv_gap_matrix.data.T) * 0.5
        self.pv_gap_junctions = GapJunctionCoupling.from_coupling_matrix(pv_gap_matrix)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(
        self,
        pv_g_exc: torch.Tensor,
        pv_g_inh: torch.Tensor,
        olm_g_exc: torch.Tensor,
        olm_g_inh: torch.Tensor,
        bistratified_g_exc: torch.Tensor,
        bistratified_g_inh: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run hippocampal interneuron populations given pre-computed synaptic conductances.

        All synaptic conductances (E→I, I→E, I→I) are computed by the parent
        NeuralRegion using STP-modulated weight matrices registered via
        _add_internal_connection().  This method only handles:
          - Gap junction coupling (electrical, V-difference based; PV cells only)
          - AMPA/NMDA splitting per cell type
          - Neuron integration

        Args:
            pv_g_exc: Total excitatory conductance to PV cells [n_pv].
                      Sum of STP-weighted Pyr→PV and (optionally) external drive.
            pv_g_inh: GABA_A conductance to PV cells [n_pv].
                      Sum of PV→PV lateral + OLM→PV (both STP-weighted in parent).
            olm_g_exc: Total excitatory conductance to OLM cells [n_olm].
            olm_g_inh: GABA_A conductance to OLM cells [n_olm].
                       Primarily from septal GABA (computed in parent).
            bistratified_g_exc: Total excitatory conductance to bistratified cells [n_bistratified].
            bistratified_g_inh: GABA_A conductance to bistratified cells [n_bistratified].

        Returns:
            Dict with pv_spikes, olm_spikes, bistratified_spikes.
            I→E inhibitory conductances are computed by the parent from these spikes
            using the registered pv_to_pyr / olm_to_pyr / bistratified_to_pyr weights.
        """
        # =====================================================================
        # GAP JUNCTION COUPLING (PV cells — bidirectional electrical synapse)
        # =====================================================================
        # Biology: PV basket cells are coupled via connexin-36 gap junctions,
        # enabling ultra-fast synchronization for coherent gamma oscillations.
        # IMPORTANT: Only allow coupling with active pyramidal drive to prevent
        # spontaneous PV entrainment from gap junctions alone.
        pv_g_gap: Optional[ConductanceTensor] = None
        pv_E_gap: Optional[GapJunctionReversal] = None
        pyr_drive_strength = pv_g_exc.mean().item()
        if pyr_drive_strength > 0.01:
            gap_junction_gain = min(1.0, pyr_drive_strength / 0.1)
            g_gap_total, E_gap = self.pv_gap_junctions.forward(self.pv_neurons.V_soma)
            pv_g_gap = ConductanceTensor(g_gap_total * gap_junction_gain)
            pv_E_gap = GapJunctionReversal(E_gap)

        # =====================================================================
        # AMPA / NMDA SPLIT
        # =====================================================================
        # PV basket cells: significant NMDA (30%) — important for burst firing.
        # OLM and bistratified: AMPA-dominated (5% NMDA) — fire only on genuine
        # pyramidal bursts; higher NMDA ratio caused noise-driven spontaneous firing.
        pv_g_ampa, pv_g_nmda = split_excitatory_conductance(pv_g_exc, nmda_ratio=0.3)
        olm_g_ampa, olm_g_nmda = split_excitatory_conductance(olm_g_exc, nmda_ratio=0.05)
        bist_g_ampa, bist_g_nmda = split_excitatory_conductance(bistratified_g_exc, nmda_ratio=0.05)

        # =====================================================================
        # RUN INTERNEURONS
        # =====================================================================
        pv_spikes, _ = self.pv_neurons.forward(
            g_ampa_input=ConductanceTensor(pv_g_ampa),
            g_nmda_input=ConductanceTensor(pv_g_nmda),
            g_gaba_a_input=ConductanceTensor(pv_g_inh),
            g_gaba_b_input=None,
            g_gap_input=pv_g_gap,
            E_gap_reversal=pv_E_gap,
        )
        olm_spikes, _ = self.olm_neurons.forward(
            g_ampa_input=ConductanceTensor(olm_g_ampa),
            g_nmda_input=ConductanceTensor(olm_g_nmda),
            g_gaba_a_input=ConductanceTensor(olm_g_inh),
            g_gaba_b_input=None,
        )
        bistratified_spikes, _ = self.bistratified_neurons.forward(
            g_ampa_input=ConductanceTensor(bist_g_ampa),
            g_nmda_input=ConductanceTensor(bist_g_nmda),
            g_gaba_a_input=ConductanceTensor(bistratified_g_inh),
            g_gaba_b_input=None,
        )

        return {
            "pv_spikes": pv_spikes,
            "olm_spikes": olm_spikes,
            "bistratified_spikes": bistratified_spikes,
        }
