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

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from thalia.components import WeightInitializer, ConductanceLIF, ConductanceLIFConfig
from thalia.constants import DEFAULT_DT_MS


@dataclass
class HippocampalInhibitoryPopulation:
    """Hippocampal inhibitory neuron population.

    Attributes:
        name: Population identifier ("pv", "olm", "bistratified")
        size: Number of neurons
        neurons: ConductanceLIF neuron model
        cell_type: Biological name
        tau_mem: Membrane time constant (ms)
        targets_dendrites: Whether this population targets dendrites (not soma)
    """
    name: str
    size: int
    neurons: nn.Module
    cell_type: str
    tau_mem: float
    targets_dendrites: bool


class HippocampalInhibitoryNetwork(nn.Module):
    """Hippocampal inhibitory network with OLM cells for theta modulation.

    Key differences from cortical inhibitory network:
    - OLM cells (not SST Martinotti)
    - No VIP cells (less disinhibition in hippocampus)
    - Bistratified cells (unique to hippocampus)
    - Septal input integration (for OLM phase-locking)
    """

    def __init__(
        self,
        region_name: str,
        pyr_size: int,
        total_inhib_fraction: float = 0.25,
        device: str = "cpu",
        dt_ms: float = DEFAULT_DT_MS,
    ):
        """Initialize hippocampal inhibitory network.

        Args:
            region_name: Region identifier (e.g., "ca1", "ca3")
            pyr_size: Number of pyramidal neurons
            total_inhib_fraction: Fraction of pyramidal count
            device: Torch device
            dt_ms: Timestep in milliseconds
        """
        super().__init__()

        self.region_name = region_name
        self.pyr_size = pyr_size
        self.device = torch.device(device)
        self.dt_ms = dt_ms

        # Total inhibitory count
        total_inhib = int(pyr_size * total_inhib_fraction)

        # =====================================================================
        # CELL TYPE PROPORTIONS (hippocampal interneurons)
        # =====================================================================
        pv_frac = 0.35        # PV+ basket cells
        olm_frac = 0.30       # OLM cells (SST+)
        bistratified_frac = 0.20  # Bistratified cells
        other_frac = 0.15     # Other interneurons (simplified)

        self.n_pv = int(total_inhib * pv_frac)
        self.n_olm = int(total_inhib * olm_frac)
        self.n_bistratified = int(total_inhib * bistratified_frac)
        self.n_other = total_inhib - self.n_pv - self.n_olm - self.n_bistratified

        # =====================================================================
        # CREATE NEURON POPULATIONS
        # =====================================================================

        # PV basket cells: Fast-spiking perisomatic inhibition
        pv_config = ConductanceLIFConfig(
            tau_mem=7.0,    # Fast-spiking
            v_threshold=1.0,
            v_reset=0.0,
            tau_adapt=50.0,     # Weak adaptation
            adapt_increment=0.05,
        )
        self.pv_neurons = ConductanceLIF(
            n_neurons=self.n_pv,
            config=pv_config,
            device=device,
        )

        # OLM cells: CRITICAL for theta phase-locking
        # High adaptation creates burst-pause dynamics for theta rhythm
        olm_config = ConductanceLIFConfig(
            tau_mem=25.0,   # Slow (regular-spiking)
            v_threshold=1.1,  # Higher threshold (needs strong drive)
            v_reset=0.0,
            tau_adapt=100.0,    # STRONG adaptation (creates bursting)
            adapt_increment=0.20,  # Large increment (burst termination)
        )
        self.olm_neurons = ConductanceLIF(
            n_neurons=self.n_olm,
            config=olm_config,
            device=device,
        )

        # Bistratified cells: Dendritic targeting
        bistratified_config = ConductanceLIFConfig(
            tau_mem=12.0,   # Medium
            v_threshold=0.9,
            v_reset=0.0,
            tau_adapt=60.0,
            adapt_increment=0.08,
        )
        self.bistratified_neurons = ConductanceLIF(
            n_neurons=self.n_bistratified,
            config=bistratified_config,
            device=device,
        )

        # Store populations
        self.populations = [
            HippocampalInhibitoryPopulation(
                name="pv",
                size=self.n_pv,
                neurons=self.pv_neurons,
                cell_type="PV+ Basket",
                tau_mem=7.0,
                targets_dendrites=False,
            ),
            HippocampalInhibitoryPopulation(
                name="olm",
                size=self.n_olm,
                neurons=self.olm_neurons,
                cell_type="OLM (SST+)",
                tau_mem=25.0,
                targets_dendrites=True,  # Apical dendrites
            ),
            HippocampalInhibitoryPopulation(
                name="bistratified",
                size=self.n_bistratified,
                neurons=self.bistratified_neurons,
                cell_type="Bistratified",
                tau_mem=12.0,
                targets_dendrites=True,  # Both proximal and distal dendrites
            ),
        ]

        # =====================================================================
        # E → I CONNECTIVITY (Pyramidal → Interneurons)
        # =====================================================================

        # Pyramidal → PV (strong, reliable)
        self.pyr_to_pv = nn.Parameter(
            WeightInitializer.sparse_uniform(
                n_pre=pyr_size,
                n_post=self.n_pv,
                prob=0.5,  # High connection probability
                w_min=0.0,
                w_max=0.8,
                device=self.device,
            )
        )

        # Pyramidal → OLM (moderate)
        self.pyr_to_olm = nn.Parameter(
            WeightInitializer.sparse_uniform(
                n_pre=pyr_size,
                n_post=self.n_olm,
                prob=0.3,
                w_min=0.0,
                w_max=0.6,
                device=self.device,
            )
        )

        # Pyramidal → Bistratified (moderate)
        self.pyr_to_bistratified = nn.Parameter(
            WeightInitializer.sparse_uniform(
                n_pre=pyr_size,
                n_post=self.n_bistratified,
                prob=0.35,
                w_min=0.0,
                w_max=0.7,
                device=self.device,
            )
        )

        # =====================================================================
        # I → E CONNECTIVITY (Interneurons → Pyramidal)
        # =====================================================================

        # PV → Pyramidal (strong perisomatic inhibition)
        self.pv_to_pyr = nn.Parameter(
            WeightInitializer.sparse_uniform(
                n_pre=self.n_pv,
                n_post=pyr_size,
                prob=0.6,
                w_min=0.0,
                w_max=1.0,  # Strong inhibition
                device=self.device,
            )
        )

        # OLM → Pyramidal (dendritic inhibition)
        # This is the KEY pathway for encoding/retrieval separation!
        self.olm_to_pyr = nn.Parameter(
            WeightInitializer.sparse_uniform(
                n_pre=self.n_olm,
                n_post=pyr_size,
                prob=0.4,
                w_min=0.0,
                w_max=0.9,  # Strong dendritic suppression
                device=self.device,
            )
        )

        # Bistratified → Pyramidal (dendritic inhibition)
        self.bistratified_to_pyr = nn.Parameter(
            WeightInitializer.sparse_uniform(
                n_pre=self.n_bistratified,
                n_post=pyr_size,
                prob=0.45,
                w_min=0.0,
                w_max=0.8,
                device=self.device,
            )
        )

        # =====================================================================
        # I → I CONNECTIVITY (Lateral inhibition)
        # =====================================================================

        # PV → PV (lateral)
        self.pv_to_pv = nn.Parameter(
            WeightInitializer.sparse_uniform(
                n_pre=self.n_pv,
                n_post=self.n_pv,
                prob=0.3,
                w_min=0.0,
                w_max=0.5,
                device=self.device,
            )
        )
        # Zero self-connections
        with torch.no_grad():
            self.pv_to_pv.fill_diagonal_(0.0)

        # OLM → PV (weak)
        self.olm_to_pv = nn.Parameter(
            WeightInitializer.sparse_uniform(
                n_pre=self.n_olm,
                n_post=self.n_pv,
                prob=0.2,
                w_min=0.0,
                w_max=0.4,
                device=self.device,
            )
        )

        # =====================================================================
        # GAP JUNCTIONS (PV cells only)
        # =====================================================================

        self.pv_gap_junctions = nn.Parameter(
            WeightInitializer.sparse_uniform(
                n_pre=self.n_pv,
                n_post=self.n_pv,
                prob=0.5,  # Dense coupling
                w_min=0.0,
                w_max=0.3,  # Moderate strength
                device=self.device,
            )
        )
        with torch.no_grad():
            self.pv_gap_junctions.fill_diagonal_(0.0)
            # Make symmetric (gap junctions are bidirectional)
            self.pv_gap_junctions.data = (
                self.pv_gap_junctions.data + self.pv_gap_junctions.data.T
            ) / 2.0

        # =====================================================================
        # SEPTAL INPUT WEIGHTS (medial septum → OLM)
        # =====================================================================
        # Septal GABAergic neurons inhibit OLM cells at theta peaks
        # → OLM cells rebound at theta troughs
        # Default size assumes ~100 septal GABA neurons
        self.septal_to_olm = nn.Parameter(
            torch.randn(self.n_olm, 100, device=self.device) * 0.5 / np.sqrt(100)
        )

    def forward(
        self,
        pyr_spikes: torch.Tensor,
        septal_gaba: Optional[torch.Tensor] = None,
        external_exc: Optional[torch.Tensor] = None,
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
            f"HippocampalInhibitoryNetwork: pyr_spikes must be 1D [pyr_size], "
            f"got shape {pyr_spikes.shape}. Source: {self.region_name}"
        )
        assert pyr_spikes.shape[0] == self.pyr_size, (
            f"HippocampalInhibitoryNetwork: pyr_spikes has {pyr_spikes.shape[0]} neurons "
            f"but expected {self.pyr_size} for region {self.region_name}"
        )

        if septal_gaba is not None and septal_gaba.numel() > 0:
            assert septal_gaba.dim() == 1, (
                f"HippocampalInhibitoryNetwork: septal_gaba must be 1D, "
                f"got shape {septal_gaba.shape} for region {self.region_name}"
            )

        # =====================================================================
        # E → I EXCITATION
        # =====================================================================
        # Weight matrices are [n_post, n_pre], need to transpose for [n_pre] @ [n_pre, n_post]
        pv_exc = pyr_spikes.float() @ self.pyr_to_pv.T
        olm_exc = pyr_spikes.float() @ self.pyr_to_olm.T
        bistratified_exc = pyr_spikes.float() @ self.pyr_to_bistratified.T

        # =====================================================================
        # SEPTAL INPUT TO OLM (key for theta phase-locking!)
        # =====================================================================

        if septal_gaba is not None:
            # Septal GABA inhibits OLM at theta peaks
            # OLM rebounds at theta troughs (rebound bursting)
            if septal_gaba.numel() > 0:
                # Resize if needed
                if septal_gaba.size(0) != 100:
                    # Pad or truncate
                    if septal_gaba.size(0) < 100:
                        septal_gaba = torch.cat([
                            septal_gaba,
                            torch.zeros(100 - septal_gaba.size(0), device=self.device)
                        ])
                    else:
                        septal_gaba = septal_gaba[:100]

                septal_inhib = septal_gaba.float() @ self.septal_to_olm.T
                olm_exc = olm_exc - septal_inhib * 2.0  # Strong inhibition
            else:
                septal_inhib = torch.zeros(self.n_olm, device=self.device)
        else:
            septal_inhib = torch.zeros(self.n_olm, device=self.device)

        # =====================================================================
        # I → I LATERAL INHIBITION
        # =====================================================================

        # Get previous spikes (or zeros if first step)
        if hasattr(self, "_prev_pv_spikes"):
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

        if hasattr(self, "_prev_pv_v_mem"):
            # Voltage coupling (not spike-based)
            pv_gap_current = (self._prev_pv_v_mem @ self.pv_gap_junctions.T)
            pv_exc = pv_exc + pv_gap_current * 0.2
        else:
            pv_gap_current = torch.zeros(self.n_pv, device=self.device)

        # =====================================================================
        # RUN INTERNEURONS
        # =====================================================================

        pv_spikes, _ = self.pv_neurons(pv_exc)
        olm_spikes, _ = self.olm_neurons(olm_exc)
        bistratified_spikes, _ = self.bistratified_neurons(bistratified_exc)

        # =====================================================================
        # I → E INHIBITION TO PYRAMIDAL
        # =====================================================================

        # Perisomatic inhibition (PV basket cells)
        # Weight matrices are [n_post, n_pre], need to transpose for [n_pre] @ [n_pre, n_post]
        perisomatic_inhib = pv_spikes.float() @ self.pv_to_pyr.T

        # Dendritic inhibition (OLM + bistratified)
        olm_dendritic_inhib = olm_spikes.float() @ self.olm_to_pyr.T
        bistratified_dendritic_inhib = bistratified_spikes.float() @ self.bistratified_to_pyr.T

        total_dendritic_inhib = olm_dendritic_inhib + bistratified_dendritic_inhib

        # =====================================================================
        # STORE STATE FOR NEXT TIMESTEP
        # =====================================================================

        # Assert outputs are 1D before storing
        assert pv_spikes.dim() == 1, f"pv_spikes must be 1D, got {pv_spikes.shape}"
        assert olm_spikes.dim() == 1, f"olm_spikes must be 1D, got {olm_spikes.shape}"

        self._prev_pv_spikes = pv_spikes
        self._prev_olm_spikes = olm_spikes
        self._prev_pv_v_mem = self.pv_neurons.membrane if self.pv_neurons.membrane is not None else torch.zeros(self.n_pv, device=self.device)

        # =====================================================================
        # RETURN STRUCTURED OUTPUT
        # =====================================================================

        return {
            "perisomatic": perisomatic_inhib,      # PV → soma
            "dendritic": total_dendritic_inhib,    # OLM + bistratified → dendrites
            "olm_dendritic": olm_dendritic_inhib,  # OLM only (for theta tracking)
            "pv_spikes": pv_spikes,
            "olm_spikes": olm_spikes,
            "bistratified_spikes": bistratified_spikes,
        }
