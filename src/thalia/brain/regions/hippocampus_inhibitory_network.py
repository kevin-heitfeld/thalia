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

from typing import Dict, Optional

import torch
import torch.nn as nn

from thalia.components import ConductanceLIFConfig, ConductanceLIF, WeightInitializer
from thalia.typing import ConductanceTensor, PopulationName, RegionName
from thalia.utils import split_excitatory_conductance

from .population_names import HippocampusPopulation


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
        dt_ms: float,
        device: str,
    ):
        """Initialize hippocampal inhibitory network.

        Args:
            region_name: Region identifier (e.g., "hippocampus")
            population_name: Population identifier (e.g., "ca1", "ca1_inhibitory")
            pyr_size: Number of pyramidal neurons
            total_inhib_fraction: Fraction of pyramidal count
            dt_ms: Timestep in milliseconds
            device: Torch device
        """
        super().__init__()

        self.region_name = region_name
        self.pyr_size = pyr_size
        self.dt_ms = dt_ms
        self.device = torch.device(device)

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
        # CRITICAL: PV neurons should ONLY fire when driven by pyramidal cells
        self.pv_neurons = ConductanceLIF(
            n_neurons=self.n_pv,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name=f"{population_name}_pv",
                device=device,
                tau_mem=7.0,    # Fast-spiking
                v_threshold=2.0,  # HIGH threshold: Prevents spontaneous firing
                v_reset=0.0,
                tau_adapt=50.0,     # Weak adaptation
                adapt_increment=0.05,
            ),
        )

        # OLM cells: CRITICAL for theta phase-locking
        # High adaptation creates burst-pause dynamics for theta rhythm
        self.olm_neurons = ConductanceLIF(
            n_neurons=self.n_olm,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name=f"{population_name}_olm",
                device=device,
                tau_mem=25.0,   # Slow (regular-spiking)
                v_threshold=1.1,  # Higher threshold (needs strong drive)
                v_reset=0.0,
                tau_adapt=100.0,    # STRONG adaptation (creates bursting)
                adapt_increment=0.20,  # Large increment (burst termination)
            ),
        )

        # Bistratified cells: Dendritic targeting
        self.bistratified_neurons = ConductanceLIF(
            n_neurons=self.n_bistratified,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name=f"{population_name}_bistratified",
                device=device,
                tau_mem=12.0,   # Medium
                v_threshold=0.9,
                v_reset=0.0,
                tau_adapt=60.0,
                adapt_increment=0.08,
            ),
        )

        # =====================================================================
        # E → I CONNECTIVITY (Pyramidal → Interneurons)
        # =====================================================================

        # Pyramidal → PV (strong, reliable)
        # CONDUCTANCE-BASED: Single Pyr→PV synapse ~0.5-1nS, normalized by g_L (10nS) = 0.05-0.1
        self.pyr_to_pv = WeightInitializer.sparse_uniform(
            n_input=pyr_size,
            n_output=self.n_pv,
            connectivity=0.5,
            w_min=0.0,
            w_max=0.0012,
            device=self.device,
        )

        # Pyramidal → OLM (moderate)
        # CONDUCTANCE-BASED: OLM cells need moderate drive for theta phase-locking
        self.pyr_to_olm = WeightInitializer.sparse_uniform(
            n_input=pyr_size,
            n_output=self.n_olm,
            connectivity=0.3,
            w_min=0.0,
            w_max=0.0006,
            device=self.device,
        )

        # Pyramidal → Bistratified (moderate)
        # CONDUCTANCE-BASED: Bistratified cells provide dendritic inhibition
        self.pyr_to_bistratified = WeightInitializer.sparse_uniform(
            n_input=pyr_size,
            n_output=self.n_bistratified,
            connectivity=0.35,
            w_min=0.0,
            w_max=0.0007,
            device=self.device,
        )

        # =====================================================================
        # I → E CONNECTIVITY (Interneurons → Pyramidal)
        # =====================================================================

        # PV → Pyramidal (perisomatic inhibition)
        if population_name == HippocampusPopulation.DG_INHIBITORY:
            # DG: Light inhibition for sparse coding (pattern separation)
            pv_connectivity = 0.3
            pv_w_min = 0.0
            pv_w_max = 0.001
        elif population_name == HippocampusPopulation.CA3_INHIBITORY:
            # CA3: STRONG inhibition to control runaway recurrent activity
            pv_connectivity = 0.7  # High connectivity for effective inhibition
            pv_w_min = 0.0
            pv_w_max = 0.003
        elif population_name == HippocampusPopulation.CA2_INHIBITORY:
            # CA2: Stronger inhibition for precise timing control (encoding/retrieval separation)
            pv_connectivity = 0.5
            pv_w_min = 0.0
            pv_w_max = 0.003
        elif population_name == HippocampusPopulation.CA1_INHIBITORY:
            # CA1: Strong inhibition to control excitability and support theta modulations
            pv_connectivity = 0.5
            pv_w_min = 0.0
            pv_w_max = 0.003
        else:
            raise ValueError(
                f"Unknown hippocampal population: {population_name}. "
                f"Expected: {', '.join(HippocampusPopulation)}."
            )

        self.pv_to_pyr = WeightInitializer.sparse_uniform(
            n_input=self.n_pv,
            n_output=pyr_size,
            connectivity=pv_connectivity,
            w_min=pv_w_min,
            w_max=pv_w_max,
            device=self.device,
        )

        # PV → Pyramidal GABA_B (slow metabotropic K⁺ channel)
        # Activated only during high-frequency PV bursts (sufficient [GABA] to recruit
        # metabotropic receptors). Weight ~15% of GABA_A: slower onset, ~400ms decay.
        # Provides the inter-burst brake for SWR termination.
        self.pv_to_pyr_gaba_b = WeightInitializer.sparse_uniform(
            n_input=self.n_pv,
            n_output=pyr_size,
            connectivity=pv_connectivity,
            w_min=0.0,
            w_max=pv_w_max * 0.15,
            device=self.device,
        )

        # OLM → Pyramidal (dendritic inhibition)
        # This is the KEY pathway for encoding/retrieval separation!
        self.olm_to_pyr = WeightInitializer.sparse_uniform(
            n_input=self.n_olm,
            n_output=pyr_size,
            connectivity=0.5,
            w_min=0.0,
            w_max=0.001,
            device=self.device,
        )

        # Bistratified → Pyramidal (dendritic inhibition)
        self.bistratified_to_pyr = WeightInitializer.sparse_uniform(
            n_input=self.n_bistratified,
            n_output=pyr_size,
            connectivity=0.55,
            w_min=0.0,
            w_max=0.002,
            device=self.device,
        )

        # =====================================================================
        # I → I CONNECTIVITY (Lateral inhibition)
        # =====================================================================

        # PV → PV (lateral)
        self.pv_to_pv = WeightInitializer.sparse_uniform_no_autapses(
            n_input=self.n_pv,
            n_output=self.n_pv,
            connectivity=0.3,
            w_min=0.0,
            w_max=0.0005,
            device=self.device,
        )

        # OLM → PV (weak)
        self.olm_to_pv = WeightInitializer.sparse_uniform(
            n_input=self.n_olm,
            n_output=self.n_pv,
            connectivity=0.2,
            w_min=0.0,
            w_max=0.0004,
            device=self.device,
        )

        # =====================================================================
        # GAP JUNCTIONS (PV cells only)
        # =====================================================================
        self.pv_gap_junctions = WeightInitializer.sparse_uniform_no_autapses(
            n_input=self.n_pv,
            n_output=self.n_pv,
            connectivity=0.5,
            w_min=0.0,
            w_max=0.0003,
            device=self.device,
        )
        # Make symmetric (gap junctions are bidirectional)
        self.pv_gap_junctions.data = (self.pv_gap_junctions.data + self.pv_gap_junctions.data.T) * 0.5

        # =====================================================================
        # SEPTAL INPUT WEIGHTS (medial septum → OLM)
        # =====================================================================
        # Septal GABAergic neurons inhibit OLM cells at theta peaks
        # → OLM cells rebound at theta troughs
        self.septal_to_olm = WeightInitializer.sparse_random(
            n_input=100,  # Assumed septal GABA population size
            n_output=self.n_olm,
            connectivity=1.0,  # Fully connected
            weight_scale=0.5,
            device=self.device,
        )

        self._prev_pv_spikes: Optional[torch.Tensor] = None
        self._prev_olm_spikes: Optional[torch.Tensor] = None
        self._prev_pv_v_mem: Optional[torch.Tensor] = None

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(
        self,
        pyr_spikes: torch.Tensor,
        septal_gaba: Optional[torch.Tensor],
        external_exc: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run hippocampal inhibitory network.

        Args:
            pyr_spikes: Pyramidal neuron spikes [pyr_size]
            septal_gaba: Septal GABAergic input [n_septal_gaba]
            external_exc: External excitation to interneurons [varies]

        Returns:
            Dictionary with:
                - perisomatic: Perisomatic inhibition (PV cells)
                - dendritic: Dendritic inhibition (OLM + bistratified)
                - olm_dendritic: OLM-specific dendritic inhibition
                - pv_spikes, olm_spikes, bistratified_spikes
        """
        # Assert 1D inputs (ADR-005: No Batch Dimension)
        assert pyr_spikes.dim() == 1, (
            f"{self.__class__.__name__}: pyr_spikes must be 1D [pyr_size], "
            f"got shape {pyr_spikes.shape}."
        )
        assert pyr_spikes.shape[0] == self.pyr_size, (
            f"{self.__class__.__name__}: pyr_spikes has {pyr_spikes.shape[0]} neurons "
            f"but expected {self.pyr_size}."
        )

        if septal_gaba is not None and septal_gaba.numel() > 0:
            assert septal_gaba.dim() == 1, (
                f"{self.__class__.__name__}: septal_gaba must be 1D, "
                f"got shape {septal_gaba.shape}."
            )

        # =====================================================================
        # E → I EXCITATION
        # =====================================================================
        pyr_spikes_float = pyr_spikes.float()
        pv_exc = self.pyr_to_pv @ pyr_spikes_float
        olm_exc = self.pyr_to_olm @ pyr_spikes_float
        bistratified_exc = self.pyr_to_bistratified @ pyr_spikes_float

        # Store original pyramidal drive before lateral inhibition modifies it
        pv_exc_from_pyramidal = pv_exc.clone()

        # =====================================================================
        # EXTERNAL EXCITATION (e.g., from EC or other regions)
        # =====================================================================
        # Interneurons can receive direct excitation from external sources
        # This is important for proper network dynamics
        if external_exc is not None and external_exc.numel() > 0:
            # Scale external excitation appropriately for each cell type
            # PV cells: moderate sensitivity to external drive
            # OLM cells: lower sensitivity (primarily local circuit driven)
            # Bistratified: moderate sensitivity
            ext_exc_mean = external_exc.mean()
            pv_exc = pv_exc + ext_exc_mean * 0.3  # 30% weight
            olm_exc = olm_exc + ext_exc_mean * 0.1  # 10% weight
            bistratified_exc = bistratified_exc + ext_exc_mean * 0.2  # 20% weight

        # =====================================================================
        # SEPTAL INPUT TO OLM (key for theta phase-locking!)
        # =====================================================================
        # Septal GABA inhibits OLM at theta peaks by providing true GABAergic
        # hyperpolarization — NOT by reducing excitatory drive.
        # Only proper GABA_A conductance (passed to the neuron as g_inh) can
        # push V_mem below rest, which is the prerequisite for any I_h-like
        # rebound after the septal burst ends.
        olm_g_inh = torch.zeros(self.n_olm, device=self.device)
        if septal_gaba is not None and septal_gaba.numel() > 0:
            # Compute synaptic inhibitory conductance from septal spikes → OLM
            olm_g_inh = self.septal_to_olm @ septal_gaba.float()

        # =====================================================================
        # I → I LATERAL INHIBITION
        # =====================================================================

        # Get previous spikes (or zeros if first step)
        if self._prev_pv_spikes is not None:
            assert self._prev_olm_spikes is not None
            # Correct: weight_matrix @ spike_vector gives [n_post] output
            # Weight matrices are [n_post, n_pre], spike vectors are [n_pre]
            pv_lateral = self.pv_to_pv @ self._prev_pv_spikes.float()
            olm_to_pv_inhib = self.olm_to_pv @ self._prev_olm_spikes.float()
        else:
            pv_lateral = torch.zeros(self.n_pv, device=self.device)
            olm_to_pv_inhib = torch.zeros(self.n_pv, device=self.device)

        pv_exc = pv_exc - pv_lateral * 0.5 - olm_to_pv_inhib * 0.3

        # =====================================================================
        # GAP JUNCTION COUPLING (PV cells)
        # =====================================================================
        # BIOLOGICAL FIX: Gap junctions should NOT create spontaneous PV activity!
        # Only allow gap junction coupling when there is actual pyramidal drive
        # Otherwise, gap junctions create positive feedback loops that sustain
        # PV firing even with zero pyramidal input (biologically impossible)

        if self._prev_pv_v_mem is not None:
            # Measure pyramidal drive strength (use original drive before lateral inhibition)
            pyr_drive_strength = pv_exc_from_pyramidal.mean().item()

            # Only enable gap junctions when pyramidal cells are actually driving PV cells
            # If pyr_drive < 0.01 (essentially silent), disable gap junctions completely
            if pyr_drive_strength > 0.01:
                # Scale gap junction strength proportional to pyramidal drive
                # This prevents spontaneous synchronization
                gap_junction_gain = min(1.0, pyr_drive_strength / 0.1)
                pv_gap_current = (self._prev_pv_v_mem @ self.pv_gap_junctions.T)
                pv_exc = pv_exc + pv_gap_current * 0.05 * gap_junction_gain

        # =====================================================================
        # RUN INTERNEURONS
        # =====================================================================

        # Split excitatory conductance: 70% AMPA (fast), 30% NMDA (slow)
        pv_g_ampa, pv_g_nmda = split_excitatory_conductance(pv_exc, nmda_ratio=0.3)
        olm_g_ampa, olm_g_nmda = split_excitatory_conductance(olm_exc, nmda_ratio=0.3)
        bistratified_g_ampa, bistratified_g_nmda = split_excitatory_conductance(bistratified_exc, nmda_ratio=0.3)

        pv_spikes, _ = self.pv_neurons.forward(
            g_ampa_input=ConductanceTensor(pv_g_ampa),
            g_nmda_input=ConductanceTensor(pv_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )
        olm_spikes, _ = self.olm_neurons.forward(
            g_ampa_input=ConductanceTensor(olm_g_ampa),
            g_nmda_input=ConductanceTensor(olm_g_nmda),
            g_gaba_a_input=ConductanceTensor(olm_g_inh),  # Septal GABA as true GABA_A conductance
            g_gaba_b_input=None,
        )
        bistratified_spikes, _ = self.bistratified_neurons.forward(
            g_ampa_input=ConductanceTensor(bistratified_g_ampa),
            g_nmda_input=ConductanceTensor(bistratified_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        # =====================================================================
        # I → E INHIBITION TO PYRAMIDAL
        # =====================================================================

        # Perisomatic inhibition (PV basket cells)
        # Weight matrices are [n_post, n_pre], need to transpose for [n_pre] @ [n_pre, n_post]
        perisomatic_inhib = self.pv_to_pyr @ pv_spikes.float()

        # Slow perisomatic GABA_B (same PV basket cells, metabotropic K⁺)
        perisomatic_gaba_b = self.pv_to_pyr_gaba_b @ pv_spikes.float()

        # Dendritic inhibition (OLM + bistratified)
        olm_dendritic_inhib = self.olm_to_pyr @ olm_spikes.float()
        bistratified_dendritic_inhib = self.bistratified_to_pyr @ bistratified_spikes.float()

        total_dendritic_inhib = olm_dendritic_inhib + bistratified_dendritic_inhib

        # =====================================================================
        # STORE STATE FOR NEXT TIMESTEP
        # =====================================================================

        self._prev_pv_spikes = pv_spikes
        self._prev_olm_spikes = olm_spikes
        self._prev_pv_v_mem = self.pv_neurons.membrane if self.pv_neurons.membrane is not None else torch.zeros(self.n_pv, device=self.device)

        # =====================================================================
        # RETURN STRUCTURED OUTPUT
        # =====================================================================

        return {
            "perisomatic": perisomatic_inhib,          # PV → soma (GABA_A, fast)
            "perisomatic_gaba_b": perisomatic_gaba_b,  # PV → soma (GABA_B, slow brake)
            "dendritic": total_dendritic_inhib,        # OLM + bistratified → dendrites
            "olm_dendritic": olm_dendritic_inhib,      # OLM only (for theta tracking)
            "pv_spikes": pv_spikes,
            "olm_spikes": olm_spikes,
            "bistratified_spikes": bistratified_spikes,
        }

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters for all inhibitory neuron populations."""
        self.dt_ms = dt_ms
        self.pv_neurons.update_temporal_parameters(dt_ms)
        self.olm_neurons.update_temporal_parameters(dt_ms)
        self.bistratified_neurons.update_temporal_parameters(dt_ms)
