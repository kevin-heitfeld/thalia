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

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from thalia.brain.neurons import ConductanceLIFConfig, ConductanceLIF
from thalia.brain.synapses import WeightInitializer
from thalia.typing import ConductanceTensor, PopulationName, RegionName
from thalia.utils import split_excitatory_conductance


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
        region_name: RegionName,
        population_name: PopulationName,
        pyr_size: int,
        total_inhib_fraction: float,
        v_threshold_olm: float,
        v_threshold_bistratified: float,
        dt_ms: float,
        device: Union[str, torch.device],
    ):
        """Initialize hippocampal inhibitory network.

        Args:
            region_name: Region identifier (e.g., "hippocampus")
            population_name: Population identifier (e.g., "ca1", "ca1_inhibitory")
            pyr_size: Number of pyramidal neurons
            total_inhib_fraction: Fraction of pyramidal count
            v_threshold_olm: Spike threshold for OLM cells.
            v_threshold_bistratified: Spike threshold for bistratified cells.
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

        # PV basket cells: Fast-spiking perisomatic inhibition
        # Normal fast-spiking threshold (0.9) — PV cells should fire in response
        # to pyramidal activity to maintain E/I balance and generate gamma oscillations.
        self.pv_neurons = ConductanceLIF(
            n_neurons=self.n_pv,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name=f"{population_name}_pv",
                tau_mem=7.0,    # Fast-spiking
                v_threshold=0.9,  # Normal fast-spiking threshold
                v_reset=0.0,
                tau_adapt=50.0,     # Weak adaptation
                adapt_increment=0.05,
            ),
            device=device,
        )

        # OLM cells: CRITICAL for theta phase-locking
        # High adaptation creates burst-pause dynamics for theta rhythm
        self.olm_neurons = ConductanceLIF(
            n_neurons=self.n_olm,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name=f"{population_name}_olm",
                tau_mem=25.0,   # Slow (regular-spiking)
                v_threshold=v_threshold_olm,
                v_reset=0.0,
                tau_adapt=100.0,    # STRONG adaptation (creates bursting)
                adapt_increment=0.20,  # Large increment (burst termination)
                noise_std=0.020,
            ),
            device=device,
        )

        # Bistratified cells: Dendritic targeting
        self.bistratified_neurons = ConductanceLIF(
            n_neurons=self.n_bistratified,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name=f"{population_name}_bistratified",
                tau_mem=12.0,   # Medium
                v_threshold=v_threshold_bistratified,
                v_reset=0.0,
                tau_adapt=60.0,
                adapt_increment=0.08,
                noise_std=0.020,
            ),
            device=device,
        )

        # =====================================================================
        # GAP JUNCTIONS (PV cells only — electrical coupling, not synaptic)
        # =====================================================================
        # All synaptic weight matrices (E→I, I→E, I→I) have been moved to the
        # parent NeuralRegion via _add_internal_connection() so they participate
        # in the standard STP, diagnostic, and learning-strategy pipeline.
        # Gap junctions are electrical (V-difference coupling), not synaptic, so
        # they remain here along with the membrane-potential state they need.
        self.pv_gap_junctions = WeightInitializer.sparse_uniform_no_autapses(
            n_input=self.n_pv,
            n_output=self.n_pv,
            connectivity=0.5,
            w_min=0.0,
            w_max=0.0003,
            device=device,
        )
        # Make symmetric (gap junctions are bidirectional)
        self.pv_gap_junctions.data = (self.pv_gap_junctions.data + self.pv_gap_junctions.data.T) * 0.5

        # Membrane-potential state for gap junction coupling (V-difference, not spike-based)
        self._prev_pv_v_soma: Optional[torch.Tensor] = None

        # =====================================================================
        # SEPTAL INPUT WEIGHTS (medial septum → OLM)
        # =====================================================================
        # Septal GABAergic neurons inhibit OLM cells at theta peaks → rebound at troughs.
        # This is an EXTERNAL INPUT transform (not an internal registered connection):
        # the parent NeuralRegion computes olm_g_inh from this weight and the septal
        # spike tensor, then passes it via the forward() signature.
        # Biology: ~100 pacemaker GABA neurons in medial septum project broadly to all
        # hippocampal OLM cells with full connectivity, driving theta phase-locking.
        self.septal_to_olm = nn.Parameter(WeightInitializer.sparse_random(
            n_input=100,   # Assumed septal GABA population size (pacemaker cells)
            n_output=self.n_olm,
            connectivity=1.0,  # All-to-all: septal GABA projects broadly
            weight_scale=0.5,
            device=device,
        ))

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
        if self._prev_pv_v_soma is not None:
            pyr_drive_strength = pv_g_exc.mean().item()
            if pyr_drive_strength > 0.01:
                gap_junction_gain = min(1.0, pyr_drive_strength / 0.1)
                pv_gap_current = self._prev_pv_v_soma @ self.pv_gap_junctions.T
                pv_g_exc = pv_g_exc + pv_gap_current * 0.05 * gap_junction_gain

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

        # Update gap junction state (membrane potential from this step)
        self._prev_pv_v_soma = self.pv_neurons.V_soma.clone()

        return {
            "pv_spikes": pv_spikes,
            "olm_spikes": olm_spikes,
            "bistratified_spikes": bistratified_spikes,
        }
