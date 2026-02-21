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
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from thalia.components import NeuronFactory, WeightInitializer
from thalia.constants import DEFAULT_DT_MS
from thalia.typing import PopulationName, RegionName
from thalia.units import ConductanceTensor
from thalia.utils import CircularDelayBuffer


class CorticalInhibitoryNetwork(nn.Module):
    """Explicit inhibitory network with multiple cell types.

    This module manages inhibitory interneurons and their connectivity for
    one cortical layer. It implements:
    - Multiple inhibitory cell types (PV, SST, VIP)
    - E→I feedforward excitation
    - I→E feedback inhibition
    - I→I mutual inhibition and disinhibition
    - Gap junction coupling (PV cells only)
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
        dt_ms: float = DEFAULT_DT_MS,
        device: str = "cpu",
    ):
        """Initialize inhibitory network for one cortical layer.

        Args:
            region_name: Region name (e.g., 'cortex') for RNG seeding
            population_name: Layer name (e.g., 'L23', 'L4') for RNG seeding
            pyr_size: Number of pyramidal neurons in this layer
            total_inhib_fraction: Fraction of pyramidal count (e.g. 0.25 = 20% of total)
            device: Torch device
            dt_ms: Simulation timestep in milliseconds
        """
        super().__init__()

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

        # PV+ Basket Cells (fast-spiking)
        # Use fast_spiking factory (already has tau_mem=8ms, heterogeneous)
        # Override for slightly faster dynamics (5ms mean) and no adaptation
        self.pv_neurons = NeuronFactory.create_fast_spiking_neurons(
            region_name=region_name,
            population_name=f"{population_name}_pv",
            n_neurons=self.pv_size,
            device=self.device,
            tau_mem=5.0,
            v_threshold=0.9,
            adapt_increment=0.0,
        )

        # SST+ Martinotti Cells (regular-spiking)
        # Use pyramidal factory as base (regular-spiking, similar dynamics)
        self.sst_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=region_name,
            population_name=f"{population_name}_sst",
            n_neurons=self.sst_size,
            device=self.device,
            tau_mem=15.0,
            v_threshold=1.0,
            adapt_increment=0.05,
            tau_adapt=90.0,
        )

        # VIP+ Interneurons (disinhibitory)
        # Use pyramidal factory as base (regular-spiking)
        self.vip_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=region_name,
            population_name=f"{population_name}_vip",
            n_neurons=self.vip_size,
            device=self.device,
            tau_mem=10.0,
            v_threshold=0.9,
            adapt_increment=0.02,
            tau_adapt=70.0,
        )

        # =====================================================================
        # EXCITATORY → INHIBITORY (E→I)
        # =====================================================================
        # Pyr → PV (moderate - drives fast inhibition without over-suppression)
        self.w_pyr_pv = WeightInitializer.sparse_gaussian(
            n_input=self.pyr_size,
            n_output=self.pv_size,
            connectivity=0.5,
            mean=0.008,
            std=0.003,
            device=device,
        )

        # Pyr → SST (weak to moderate - drives dendritic inhibition)
        self.w_pyr_sst = WeightInitializer.sparse_gaussian(
            n_input=self.pyr_size,
            n_output=self.sst_size,
            connectivity=0.3,
            mean=0.005,
            std=0.003,
            device=device,
        )

        # Pyr → VIP (strong, specific - drives disinhibition)
        self.w_pyr_vip = WeightInitializer.sparse_gaussian(
            n_input=self.pyr_size,
            n_output=self.vip_size,
            connectivity=0.4,
            mean=0.015,
            std=0.003,
            device=device,
        )

        # =====================================================================
        # INHIBITORY → EXCITATORY (I→E)
        # =====================================================================
        # PV → Pyr (perisomatic inhibition - gamma gating)
        self.w_pv_pyr = WeightInitializer.sparse_gaussian(
            n_input=self.pv_size,
            n_output=self.pyr_size,
            connectivity=0.6,
            mean=0.003,
            std=0.001,
            device=device,
        )

        # SST → Pyr (dendritic inhibition - feedback modulation)
        self.w_sst_pyr = WeightInitializer.sparse_gaussian(
            n_input=self.sst_size,
            n_output=self.pyr_size,
            connectivity=0.4,
            mean=0.001,
            std=0.004,
            device=device,
        )

        # VIP → Pyr (very weak/absent)
        self.w_vip_pyr = WeightInitializer.sparse_gaussian(
            n_input=self.vip_size,
            n_output=self.pyr_size,
            connectivity=0.05,
            mean=0.002,
            std=0.001,
            device=device,
        )

        # =====================================================================
        # INHIBITORY → INHIBITORY (I→I)
        # =====================================================================
        # PV → PV (weak lateral inhibition)
        self.w_pv_pv = WeightInitializer.sparse_gaussian(
            n_input=self.pv_size,
            n_output=self.pv_size,
            connectivity=0.3,
            mean=0.005,
            std=0.002,
            device=device,
        )
        self.w_pv_pv.fill_diagonal_(0.0)  # No self-connections

        # PV → SST (moderate inhibition)
        self.w_pv_sst = WeightInitializer.sparse_gaussian(
            n_input=self.pv_size,
            n_output=self.sst_size,
            connectivity=0.3,
            mean=0.006,
            std=0.002,
            device=device,
        )

        # SST → PV (weak inhibition)
        self.w_sst_pv = WeightInitializer.sparse_gaussian(
            n_input=self.sst_size,
            n_output=self.pv_size,
            connectivity=0.2,
            mean=0.004,
            std=0.002,
            device=device,
        )

        # VIP → PV (strong disinhibition - critical for preventing runaway inhibition)
        self.w_vip_pv = WeightInitializer.sparse_gaussian(
            n_input=self.vip_size,
            n_output=self.pv_size,
            connectivity=0.6,
            mean=0.0030,
            std=0.003,
            device=device,
        )

        # VIP → SST (strong disinhibition - primary VIP target)
        self.w_vip_sst = WeightInitializer.sparse_gaussian(
            n_input=self.vip_size,
            n_output=self.sst_size,
            connectivity=0.7,
            mean=0.0015,
            std=0.003,
            device=device,
        )

        # =====================================================================
        # GAP JUNCTIONS (PV cells only)
        # =====================================================================
        self.w_pv_gap = WeightInitializer.sparse_gaussian(
            n_input=self.pv_size,
            n_output=self.pv_size,
            connectivity=0.5,
            mean=0.004,
            std=0.001,
            device=device,
        )
        self.w_pv_gap.fill_diagonal_(0.0)  # No self-coupling

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def __call__(self, *args, **kwds):
        assert False, f"{self.__class__.__name__} instances should not be called directly. Use forward() instead."
        return super().__call__(*args, **kwds)

    def forward(
        self,
        pyr_spikes: torch.Tensor,
        pyr_membrane: torch.Tensor,
        external_excitation: torch.Tensor,
        acetylcholine: float,
        feedforward_excitation: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run inhibitory network for one timestep.

        Args:
            pyr_spikes: Pyramidal neuron spikes [pyr_size], bool
            pyr_membrane: Pyramidal membrane potentials [pyr_size], float
            external_excitation: External excitation to layer [pyr_size], float
            acetylcholine: ACh level (0-1), modulates VIP activation
            feedforward_excitation: Direct excitation to PV cells [pv_size] for feedforward inhibition

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

        # PV cells: Driven by pyramidal activity + external input + feedforward
        pv_exc_from_pyr = torch.matmul(self.w_pyr_pv, pyr_spikes_float)

        # Add feedforward excitation directly to PV cells (bypasses pyramidal)
        # Biology: Thalamic afferents drive PV cells directly for fast feedforward inhibition
        if feedforward_excitation is not None:
            assert feedforward_excitation.size(0) == self.pv_size, "Feedforward excitation size must match PV population size"
            # Use ONLY feedforward drive (thalamic), not external_excitation
            # external_excitation already contains thalamic input, so we'd double-count
            pv_exc = pv_exc_from_pyr + feedforward_excitation  # Direct thalamic drive only
        else:
            # Fallback: use external excitation if no feedforward provided (old behavior)
            pv_exc_external = torch.matmul(
                self.w_pyr_pv,
                external_excitation / (1.0 + external_excitation.sum() * 0.01)
            )
            pv_exc = pv_exc_from_pyr + pv_exc_external

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

        # Get previous spikes (for I→I connections) using CircularDelayBuffer
        # Initialize buffers if first timestep
        if not hasattr(self, '_pv_spike_buffer'):
            self._pv_spike_buffer = CircularDelayBuffer(max_delay=1, size=self.pv_size, dtype=torch.bool, device=device)
            self._sst_spike_buffer = CircularDelayBuffer(max_delay=1, size=self.sst_size, dtype=torch.bool, device=device)
            self._vip_spike_buffer = CircularDelayBuffer(max_delay=1, size=self.vip_size, dtype=torch.bool, device=device)
            self._pv_membrane_buffer = CircularDelayBuffer(max_delay=1, size=self.pv_size, dtype=torch.float32, device=device)

        prev_pv = self._pv_spike_buffer.read(delay=1).float()
        prev_sst = self._sst_spike_buffer.read(delay=1).float()
        prev_vip = self._vip_spike_buffer.read(delay=1).float()

        # PV inhibition: from PV, SST, and VIP (disinhibition)
        pv_inh_from_pv = torch.matmul(self.w_pv_pv, prev_pv)
        pv_inh_from_sst = torch.matmul(self.w_sst_pv, prev_sst)
        pv_inh_from_vip = torch.matmul(self.w_vip_pv, prev_vip)
        pv_inh = pv_inh_from_pv + pv_inh_from_sst + pv_inh_from_vip

        # SST inhibition: from PV and VIP (disinhibition)
        sst_inh_from_pv = torch.matmul(self.w_pv_sst, prev_pv)
        sst_inh_from_vip = torch.matmul(self.w_vip_sst, prev_vip)
        sst_inh = sst_inh_from_pv + sst_inh_from_vip

        # VIP inhibition: minimal (VIP cells are not strongly targeted)
        vip_inh = torch.zeros(self.vip_size, device=device)

        # =====================================================================
        # GAP JUNCTION COUPLING (PV cells only)
        # =====================================================================
        # Electrical coupling based on membrane potential similarity
        # Gap junctions conduct when neighbors are at similar potentials
        prev_pv_membrane = self._pv_membrane_buffer.read(delay=1)
        # Compute coupling current proportional to voltage difference
        pv_gap_coupling = torch.matmul(
            self.w_pv_gap,
            prev_pv_membrane
        ) - prev_pv_membrane * self.w_pv_gap.sum(dim=1)

        # Add to excitation (gap junctions are bidirectional)
        pv_exc = pv_exc + pv_gap_coupling * 0.3  # Scale coupling effect

        # =====================================================================
        # RUN INHIBITORY NEURONS
        # =====================================================================
        # Split excitatory conductance: 70% AMPA (fast), 30% NMDA (slow)

        # PV cells (fast-spiking)
        pv_g_ampa, pv_g_nmda = pv_exc * 0.7, pv_exc * 0.3
        pv_spikes, pv_membrane = self.pv_neurons.forward(
            g_ampa_input=ConductanceTensor(pv_g_ampa),
            g_gaba_a_input=ConductanceTensor(pv_inh),
            g_nmda_input=ConductanceTensor(pv_g_nmda),
        )

        # SST cells (regular-spiking)
        sst_g_ampa, sst_g_nmda = sst_exc * 0.7, sst_exc * 0.3
        sst_spikes, sst_membrane = self.sst_neurons.forward(
            g_ampa_input=ConductanceTensor(sst_g_ampa),
            g_gaba_a_input=ConductanceTensor(sst_inh),
            g_nmda_input=ConductanceTensor(sst_g_nmda),
        )

        # VIP cells (disinhibitory)
        vip_g_ampa, vip_g_nmda = vip_exc * 0.7, vip_exc * 0.3
        vip_spikes, vip_membrane = self.vip_neurons.forward(
            g_ampa_input=ConductanceTensor(vip_g_ampa),
            g_gaba_a_input=ConductanceTensor(vip_inh),
            g_nmda_input=ConductanceTensor(vip_g_nmda),
        )

        # Store for next timestep using CircularDelayBuffer
        self._pv_spike_buffer.write(pv_spikes)
        self._sst_spike_buffer.write(sst_spikes)
        self._vip_spike_buffer.write(vip_spikes)
        self._pv_membrane_buffer.write(pv_membrane)

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
        # NOTE: These are ADDITIONAL to baseline inhibition (50% in cortex.py L4)
        # With threshold 2.0, PV needs to be strong to counteract drive
        total_inhibition = (
            perisomatic_inhibition * 1.0 +  # Strong PV inhibition (threshold 2.0 compensates)
            dendritic_inhibition * 0.3 +    # SST provides modulatory inhibition
            vip_to_pyr * 0.05               # VIP has minimal direct effect
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

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters for all inhibitory neuron populations."""
        self.dt_ms = dt_ms
        self.pv_neurons.update_temporal_parameters(dt_ms)
        self.sst_neurons.update_temporal_parameters(dt_ms)
        self.vip_neurons.update_temporal_parameters(dt_ms)
