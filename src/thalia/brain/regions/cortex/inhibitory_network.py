"""
Explicit Inhibitory Network for Cortex.

This module implements biologically-accurate inhibitory interneuron populations
and their connectivity patterns. Real cortex has ~20% inhibitory neurons with
specialized subtypes that provide different forms of inhibition.

Inhibitory Cell Types (based on cortical interneuron literature):
==================================================================

1. **PV+ Basket Cells (40% of inhibitory neurons)**
   - Fast-spiking interneurons (FSI)
   - Perisomatic inhibition (target cell bodies)
   - Generate gamma oscillations (40-80Hz)
   - Fast dynamics: tau_mem ~5ms
   - Connected via gap junctions
   - Lateral inhibition for winner-take-all

2. **SST+ Martinotti Cells (30% of inhibitory neurons)**
   - Regular-spiking interneurons
   - Dendritic inhibition (target apical dendrites)
   - Modulate top-down feedback
   - Slower dynamics: tau_mem ~15ms
   - No gap junctions
   - Widespread lateral inhibition

3. **VIP+ Interneurons (20% of inhibitory neurons)**
   - Disinhibitory neurons (inhibit other inhibitory neurons)
   - Target SST and PV cells
   - Modulated by acetylcholine and attention
   - Medium dynamics: tau_mem ~10ms
   - Enable selective disinhibition

4. **5-HT3aR+ Interneurons (10% of inhibitory neurons)**
   - Mixed properties
   - Simplified in this implementation

Connectivity Patterns (based on Pfeffer et al. 2013, Pi et al. 2013):
=====================================================================

Excitatory → Inhibitory:
- Pyramidal → PV: Strong, reliable (P=0.5)
- Pyramidal → SST: Moderate (P=0.3)
- Pyramidal → VIP: Strong, specific (P=0.4)

Inhibitory → Excitatory:
- PV → Pyramidal: Strong, perisomatic (P=0.6)
- SST → Pyramidal: Moderate, dendritic (P=0.4)
- VIP → Pyramidal: Weak/absent (P=0.05)

Inhibitory → Inhibitory:
- PV → PV: Weak lateral (P=0.3)
- PV → SST: Moderate (P=0.3)
- SST → PV: Weak (P=0.2)
- VIP → PV: Strong (P=0.6)
- VIP → SST: Strong (P=0.7)

Gap Junctions:
- PV ↔ PV: Dense (P=0.5)
- Others: Absent

References:
===========
- Pfeffer et al. (2013) "Inhibition of inhibition in visual cortex"
- Pi et al. (2013) "Cortical interneurons that specialize in disinhibitory control"
- Tremblay et al. (2016) "GABAergic Interneurons in the Neocortex"
- Kepecs & Fishell (2014) "Interneuron cell types are fit to function"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from thalia.components import WeightInitializer, NeuronFactory
from thalia.constants import DEFAULT_DT_MS


@dataclass
class InhibitoryPopulation:
    """Population of inhibitory neurons with specific properties.

    Attributes:
        name: Population identifier (e.g., "pv", "sst", "vip")
        size: Number of neurons
        neurons: ConductanceLIF neuron model
        cell_type: Biological cell type name
        tau_mem: Membrane time constant (ms)
        has_gap_junctions: Whether this population has electrical coupling
    """
    name: str
    size: int
    neurons: nn.Module
    cell_type: str
    tau_mem: float
    has_gap_junctions: bool


class InhibitoryNetwork(nn.Module):
    """Explicit inhibitory network with multiple cell types.

    This module manages inhibitory interneurons and their connectivity for
    one cortical layer. It implements:
    - Multiple inhibitory cell types (PV, SST, VIP)
    - E→I feedforward excitation
    - I→E feedback inhibition
    - I→I mutual inhibition and disinhibition
    - Gap junction coupling (PV cells only)
    """

    def __init__(
        self,
        layer_name: str,
        pyr_size: int,
        total_inhib_fraction: float = 0.25,
        device: str = "cpu",
        dt_ms: float = DEFAULT_DT_MS,
    ):
        """Initialize inhibitory network for one cortical layer.

        Args:
            layer_name: Layer identifier (e.g., "l23", "l4", "l5")
            pyr_size: Number of pyramidal neurons in this layer
            total_inhib_fraction: Fraction of pyramidal count (0.25 = 20% of total)
            device: Torch device
            dt_ms: Simulation timestep in milliseconds
        """
        super().__init__()

        self.layer_name = layer_name
        self.pyr_size = pyr_size
        self.device = torch.device(device)
        self.dt_ms = dt_ms

        # Total inhibitory neurons (25% of pyramidal = 20% of total)
        total_inhib = max(int(pyr_size * total_inhib_fraction), 10)

        # Divide into subtypes (based on cortical distributions)
        self.pv_size = max(int(total_inhib * 0.40), 4)  # 40% - basket cells
        self.sst_size = max(int(total_inhib * 0.30), 3)  # 30% - Martinotti
        self.vip_size = max(int(total_inhib * 0.20), 2)  # 20% - disinhibitory
        self.other_size = total_inhib - (self.pv_size + self.sst_size + self.vip_size)

        # Create inhibitory populations
        self._create_populations()

        # Create connectivity matrices
        self._create_connectivity()

    def _create_populations(self) -> None:
        """Create inhibitory neuron populations with cell-type-specific properties."""

        # PV+ Basket Cells (fast-spiking)
        pv_neurons = NeuronFactory.create_cortical_layer_neurons(
            self.pv_size,
            self.layer_name,
            self.device,
            tau_mem=5.0,  # Very fast
            v_threshold=0.8,  # Lower threshold (more excitable)
            adapt_increment=0.0,  # No adaptation (sustain high rates)
        )
        self.pv_population = InhibitoryPopulation(
            name="pv",
            size=self.pv_size,
            neurons=pv_neurons,
            cell_type="PV+ Basket Cell",
            tau_mem=5.0,
            has_gap_junctions=True,
        )

        # SST+ Martinotti Cells (regular-spiking)
        sst_neurons = NeuronFactory.create_cortical_layer_neurons(
            self.sst_size,
            self.layer_name,
            self.device,
            tau_mem=15.0,  # Slower
            v_threshold=1.0,  # Normal threshold
            adapt_increment=0.05,  # Moderate adaptation
        )
        self.sst_population = InhibitoryPopulation(
            name="sst",
            size=self.sst_size,
            neurons=sst_neurons,
            cell_type="SST+ Martinotti Cell",
            tau_mem=15.0,
            has_gap_junctions=False,
        )

        # VIP+ Interneurons (disinhibitory)
        vip_neurons = NeuronFactory.create_cortical_layer_neurons(
            self.vip_size,
            self.layer_name,
            self.device,
            tau_mem=10.0,  # Medium
            v_threshold=0.9,  # Slightly lower
            adapt_increment=0.02,  # Little adaptation
        )
        self.vip_population = InhibitoryPopulation(
            name="vip",
            size=self.vip_size,
            neurons=vip_neurons,
            cell_type="VIP+ Interneuron",
            tau_mem=10.0,
            has_gap_junctions=False,
        )

    def _create_connectivity(self) -> None:
        """Create E→I, I→E, and I→I connectivity matrices.

        Implements biologically-realistic connection probabilities and strengths
        based on anatomical and physiological data.
        """
        device = self.device

        # =====================================================================
        # EXCITATORY → INHIBITORY (E→I)
        # =====================================================================
        # Pyramidal neurons drive inhibitory neurons

        # Pyr → PV (strong, reliable - drives fast inhibition)
        pyr_pv_prob = 0.5
        pyr_pv_strength = 1.2
        self.w_pyr_pv = nn.Parameter(
            torch.abs(WeightInitializer.sparse_gaussian(
                n_output=self.pv_size,
                n_input=self.pyr_size,
                sparsity=1.0 - pyr_pv_prob,
                mean=pyr_pv_strength,
                std=0.3,
                device=device,
            ))
        )

        # Pyr → SST (moderate - drives dendritic inhibition)
        pyr_sst_prob = 0.3
        pyr_sst_strength = 0.8
        self.w_pyr_sst = nn.Parameter(
            torch.abs(WeightInitializer.sparse_gaussian(
                n_output=self.sst_size,
                n_input=self.pyr_size,
                sparsity=1.0 - pyr_sst_prob,
                mean=pyr_sst_strength,
                std=0.3,
                device=device,
            ))
        )

        # Pyr → VIP (strong, specific - drives disinhibition)
        pyr_vip_prob = 0.4
        pyr_vip_strength = 1.0
        self.w_pyr_vip = nn.Parameter(
            torch.abs(WeightInitializer.sparse_gaussian(
                n_output=self.vip_size,
                n_input=self.pyr_size,
                sparsity=1.0 - pyr_vip_prob,
                mean=pyr_vip_strength,
                std=0.3,
                device=device,
            ))
        )

        # =====================================================================
        # INHIBITORY → EXCITATORY (I→E)
        # =====================================================================
        # Inhibitory neurons suppress pyramidal neurons

        # PV → Pyr (perisomatic inhibition - gamma gating)
        # Reduced 10x to prevent suppression of activity
        pv_pyr_prob = 0.6
        pv_pyr_strength = 0.15  # Was 1.5 - reduced 10x
        self.w_pv_pyr = nn.Parameter(
            torch.abs(WeightInitializer.sparse_gaussian(
                n_output=self.pyr_size,
                n_input=self.pv_size,
                sparsity=1.0 - pv_pyr_prob,
                mean=pv_pyr_strength,
                std=0.04,  # Proportionally reduced std
                device=device,
            ))
        )

        # SST → Pyr (dendritic inhibition - feedback modulation)
        # Reduced 10x to prevent suppression of activity
        sst_pyr_prob = 0.4
        sst_pyr_strength = 0.10  # Was 1.0 - reduced 10x
        self.w_sst_pyr = nn.Parameter(
            torch.abs(WeightInitializer.sparse_gaussian(
                n_output=self.pyr_size,
                n_input=self.sst_size,
                sparsity=1.0 - sst_pyr_prob,
                mean=sst_pyr_strength,
                std=0.03,  # Proportionally reduced std
                device=device,
            ))
        )

        # VIP → Pyr (very weak/absent)
        vip_pyr_prob = 0.05
        vip_pyr_strength = 0.2
        self.w_vip_pyr = nn.Parameter(
            torch.abs(WeightInitializer.sparse_gaussian(
                n_output=self.pyr_size,
                n_input=self.vip_size,
                sparsity=1.0 - vip_pyr_prob,
                mean=vip_pyr_strength,
                std=0.1,
                device=device,
            ))
        )

        # =====================================================================
        # INHIBITORY → INHIBITORY (I→I)
        # =====================================================================
        # Competition and disinhibition

        # PV → PV (weak lateral inhibition)
        pv_pv_prob = 0.3
        pv_pv_strength = 0.5
        w_pv_pv = torch.abs(WeightInitializer.sparse_gaussian(
            n_output=self.pv_size,
            n_input=self.pv_size,
            sparsity=1.0 - pv_pv_prob,
            mean=pv_pv_strength,
            std=0.2,
            device=device,
        ))
        w_pv_pv.fill_diagonal_(0.0)  # No self-connections
        self.w_pv_pv = nn.Parameter(w_pv_pv)

        # PV → SST (moderate inhibition)
        pv_sst_prob = 0.3
        pv_sst_strength = 0.6
        self.w_pv_sst = nn.Parameter(
            torch.abs(WeightInitializer.sparse_gaussian(
                n_output=self.sst_size,
                n_input=self.pv_size,
                sparsity=1.0 - pv_sst_prob,
                mean=pv_sst_strength,
                std=0.2,
                device=device,
            ))
        )

        # SST → PV (weak inhibition)
        sst_pv_prob = 0.2
        sst_pv_strength = 0.4
        self.w_sst_pv = nn.Parameter(
            torch.abs(WeightInitializer.sparse_gaussian(
                n_output=self.pv_size,
                n_input=self.sst_size,
                sparsity=1.0 - sst_pv_prob,
                mean=sst_pv_strength,
                std=0.2,
                device=device,
            ))
        )

        # VIP → PV (strong disinhibition)
        vip_pv_prob = 0.6
        vip_pv_strength = 1.2
        self.w_vip_pv = nn.Parameter(
            torch.abs(WeightInitializer.sparse_gaussian(
                n_output=self.pv_size,
                n_input=self.vip_size,
                sparsity=1.0 - vip_pv_prob,
                mean=vip_pv_strength,
                std=0.3,
                device=device,
            ))
        )

        # VIP → SST (strong disinhibition - primary VIP target)
        vip_sst_prob = 0.7
        vip_sst_strength = 1.5
        self.w_vip_sst = nn.Parameter(
            torch.abs(WeightInitializer.sparse_gaussian(
                n_output=self.sst_size,
                n_input=self.vip_size,
                sparsity=1.0 - vip_sst_prob,
                mean=vip_sst_strength,
                std=0.3,
                device=device,
            ))
        )

        # =====================================================================
        # GAP JUNCTIONS (PV cells only)
        # =====================================================================
        # Electrical coupling for synchronization
        gap_prob = 0.5
        gap_strength = 0.4
        w_pv_gap = torch.abs(WeightInitializer.sparse_gaussian(
            n_output=self.pv_size,
            n_input=self.pv_size,
            sparsity=1.0 - gap_prob,
            mean=gap_strength,
            std=0.15,
            device=device,
        ))
        w_pv_gap.fill_diagonal_(0.0)  # No self-coupling
        self.w_pv_gap = nn.Parameter(w_pv_gap)

    def forward(
        self,
        pyr_spikes: torch.Tensor,
        pyr_membrane: torch.Tensor,
        external_excitation: torch.Tensor,
        acetylcholine: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Run inhibitory network for one timestep.

        Args:
            pyr_spikes: Pyramidal neuron spikes [pyr_size], bool
            pyr_membrane: Pyramidal membrane potentials [pyr_size], float
            external_excitation: External excitation to layer [pyr_size], float
            acetylcholine: ACh level (0-1), modulates VIP activation

        Returns:
            Dictionary with:
                - total_inhibition: Combined inhibitory conductance to pyramidal [pyr_size]
                - pv_spikes: PV cell spikes [pv_size], bool
                - sst_spikes: SST cell spikes [sst_size], bool
                - vip_spikes: VIP cell spikes [vip_size], bool
                - perisomatic_inhibition: From PV cells [pyr_size]
                - dendritic_inhibition: From SST cells [pyr_size]
        """
        device = self.device
        pyr_spikes_float = pyr_spikes.float()

        # =====================================================================
        # COMPUTE EXCITATION TO INHIBITORY POPULATIONS
        # =====================================================================

        # PV cells: Driven by pyramidal activity + external input
        pv_exc_from_pyr = torch.matmul(self.w_pyr_pv, pyr_spikes_float)
        pv_exc_external = torch.matmul(
            self.w_pyr_pv,
            external_excitation / (1.0 + external_excitation.sum() * 0.01)
        )
        pv_exc = pv_exc_from_pyr + pv_exc_external * 0.3  # Partial external drive

        # SST cells: Driven by pyramidal activity
        sst_exc_from_pyr = torch.matmul(self.w_pyr_sst, pyr_spikes_float)
        sst_exc = sst_exc_from_pyr

        # VIP cells: Driven by pyramidal activity + acetylcholine modulation
        # ACh enhances VIP activity → disinhibition during attention/encoding
        vip_exc_from_pyr = torch.matmul(self.w_pyr_vip, pyr_spikes_float)
        ach_boost = 1.0 + acetylcholine * 0.5  # Up to 1.5x with high ACh
        vip_exc = vip_exc_from_pyr * ach_boost

        # =====================================================================
        # COMPUTE INHIBITION TO INHIBITORY POPULATIONS (I→I)
        # =====================================================================

        # Get previous spikes (for I→I connections)
        # Initialize if first timestep
        if not hasattr(self, '_prev_pv_spikes'):
            self._prev_pv_spikes = torch.zeros(self.pv_size, device=device, dtype=torch.bool)
            self._prev_sst_spikes = torch.zeros(self.sst_size, device=device, dtype=torch.bool)
            self._prev_vip_spikes = torch.zeros(self.vip_size, device=device, dtype=torch.bool)

        prev_pv = self._prev_pv_spikes.float()
        prev_sst = self._prev_sst_spikes.float()
        prev_vip = self._prev_vip_spikes.float()

        # PV inhibition: from PV, SST, and VIP (disinhibition)
        pv_inh_from_pv = torch.matmul(self.w_pv_pv, prev_pv)
        pv_inh_from_sst = torch.matmul(self.w_sst_pv, prev_sst)
        pv_inh_from_vip = torch.matmul(self.w_vip_pv, prev_vip)  # Disinhibition
        pv_inh = pv_inh_from_pv + pv_inh_from_sst + pv_inh_from_vip

        # SST inhibition: from PV and VIP (disinhibition)
        sst_inh_from_pv = torch.matmul(self.w_pv_sst, prev_pv)
        sst_inh_from_vip = torch.matmul(self.w_vip_sst, prev_vip)  # Disinhibition
        sst_inh = sst_inh_from_pv + sst_inh_from_vip

        # VIP inhibition: minimal (VIP cells are not strongly targeted)
        vip_inh = torch.zeros(self.vip_size, device=device)

        # =====================================================================
        # GAP JUNCTION COUPLING (PV cells only)
        # =====================================================================
        # Electrical coupling based on membrane potential similarity
        # Gap junctions conduct when neighbors are at similar potentials
        if hasattr(self, '_prev_pv_membrane'):
            # Compute coupling current proportional to voltage difference
            pv_gap_coupling = torch.matmul(
                self.w_pv_gap,
                self._prev_pv_membrane
            ) - self._prev_pv_membrane * self.w_pv_gap.sum(dim=1)

            # Add to excitation (gap junctions are bidirectional)
            pv_exc = pv_exc + pv_gap_coupling * 0.3  # Scale coupling effect

        # =====================================================================
        # RUN INHIBITORY NEURONS
        # =====================================================================

        # PV cells (fast-spiking)
        pv_spikes, pv_membrane = self.pv_population.neurons(pv_exc, pv_inh)

        # SST cells (regular-spiking)
        sst_spikes, sst_membrane = self.sst_population.neurons(sst_exc, sst_inh)

        # VIP cells (disinhibitory)
        vip_spikes, vip_membrane = self.vip_population.neurons(vip_exc, vip_inh)

        # Store for next timestep
        self._prev_pv_spikes = pv_spikes
        self._prev_sst_spikes = sst_spikes
        self._prev_vip_spikes = vip_spikes
        self._prev_pv_membrane = pv_membrane

        # =====================================================================
        # COMPUTE INHIBITION TO PYRAMIDAL NEURONS
        # =====================================================================

        # PV → Pyr (perisomatic inhibition - fast, strong)
        perisomatic_inhibition = torch.matmul(self.w_pv_pyr, pv_spikes.float())

        # SST → Pyr (dendritic inhibition - slower, modulatory)
        dendritic_inhibition = torch.matmul(self.w_sst_pyr, sst_spikes.float())

        # VIP → Pyr (minimal direct effect)
        vip_to_pyr = torch.matmul(self.w_vip_pyr, vip_spikes.float())

        # Total inhibition (weighted combination)
        # PV provides strongest, fastest inhibition (gamma gating)
        # SST provides slower, dendritic modulation
        total_inhibition = (
            perisomatic_inhibition * 1.5 +  # Strong, fast
            dendritic_inhibition * 0.8 +    # Moderate, slow
            vip_to_pyr * 0.2                # Weak
        )

        return {
            "total_inhibition": total_inhibition,
            "pv_spikes": pv_spikes,
            "sst_spikes": sst_spikes,
            "vip_spikes": vip_spikes,
            "perisomatic_inhibition": perisomatic_inhibition,
            "dendritic_inhibition": dendritic_inhibition,
            "pv_membrane": pv_membrane,
            "sst_membrane": sst_membrane,
            "vip_membrane": vip_membrane,
        }

    def get_total_size(self) -> int:
        """Get total number of inhibitory neurons."""
        return self.pv_size + self.sst_size + self.vip_size

    def get_all_spikes(self) -> torch.Tensor:
        """Get concatenated spikes from all inhibitory populations.

        Returns:
            Tensor [total_inhib_size] with spikes from PV, SST, VIP
        """
        if not hasattr(self, '_prev_pv_spikes'):
            # Not yet initialized
            return torch.zeros(
                self.get_total_size(),
                device=self.device,
                dtype=torch.bool
            )

        return torch.cat([
            self._prev_pv_spikes,
            self._prev_sst_spikes,
            self._prev_vip_spikes,
        ])
