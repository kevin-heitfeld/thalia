"""
Explicit Inhibitory Network for CorticalColumn.

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

4. **L1 Neurogliaform (NGC) Cells (10% of inhibitory neurons)**
   - Layer 1 interneurons with extremely dense local axonal clouds ("cottonball" morphology)
   - Diffuse, widespread GABA_A + GABA_B inhibition of L2/3 apical tuft dendrites
   - Primary recipients of long-range top-down feedback axons (terminate in L1)
   - Modulated by acetylcholine (M1 muscarinic — METABOTROPIC, slow: tau ~200ms)
   - Slow dynamics: tau_mem ~15ms, moderate adaptation
   - Gate whether top-down feedback reaches L2/3 dendrites

Connectivity Patterns:
======================================================================================

Excitatory → Inhibitory:
- Pyramidal → PV: Strong, reliable (P=0.5)
- Pyramidal → SST: Moderate (P=0.3)
- Pyramidal → VIP: Strong, specific (P=0.4)
- Pyramidal → NGC: Weak, local collaterals (P=0.2)
- Long-range → NGC: Moderate, via L1 feedback axons (P=0.3)

Inhibitory → Excitatory:
- PV → Pyramidal: Strong, perisomatic (P=0.6)
- SST → Pyramidal: Moderate, dendritic (P=0.4)
- VIP → Pyramidal: Weak/absent (P=0.05)
- NGC → Pyramidal: Small, diffuse GABA_A on apical tuft (P=0.5)

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

from typing import Callable, Dict, Optional, Union, cast

import torch
import torch.nn as nn

from thalia.brain.configs.cortical_column import CorticalPopulationConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    ConductanceLIF,
    split_excitatory_conductance,
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
)
from thalia.brain.regions.population_names import CortexPopulation
from thalia.brain.synapses import WeightInitializer, NMReceptorType, make_neuromodulator_receptor
from thalia.typing import ConductanceTensor, PopulationPolarity
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

    # Default electrophysiology parameters per cell type.
    _PV_DEFAULTS = CorticalPopulationConfig(tau_mem_ms=5.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.10, tau_adapt=100.0)
    _SST_DEFAULTS = CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.10, tau_adapt=90.0)
    _VIP_DEFAULTS = CorticalPopulationConfig(tau_mem_ms=10.0, v_threshold=0.95, v_reset=0.0, adapt_increment=0.12, tau_adapt=70.0)
    _NGC_DEFAULTS = CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.30, tau_adapt=100.0)

    def __init__(
        self,
        population_name: CortexPopulation,
        pyr_size: int,
        total_inhib_fraction: float,
        _create_and_register_neurons_fn: Callable[[], ConductanceLIF],
        dt_ms: float,
        device: Union[str, torch.device],
        population_overrides: Dict[CortexPopulation, CorticalPopulationConfig],
    ):
        """Initialize inhibitory network for one cortical layer.

        Args:
            population_name: Layer name (e.g., 'L23', 'L4') for RNG seeding
            pyr_size: Number of pyramidal neurons in this layer
            total_inhib_fraction: Fraction of pyramidal count (e.g. 0.25 = 20% of total)
            dt_ms: Simulation timestep in milliseconds
            device: Torch device
            population_overrides: Per-cell-type parameter overrides.
        """
        super().__init__()

        self.pyr_size = pyr_size
        self.dt_ms = dt_ms

        pv_pop_name = cast(CortexPopulation, f"{population_name}_pv")
        sst_pop_name = cast(CortexPopulation, f"{population_name}_sst")
        vip_pop_name = cast(CortexPopulation, f"{population_name}_vip")
        ngc_pop_name = cast(CortexPopulation, f"{population_name}_ngc")

        pv_overrides = population_overrides.get(pv_pop_name, self._PV_DEFAULTS)
        sst_overrides = population_overrides.get(sst_pop_name, self._SST_DEFAULTS)
        vip_overrides = population_overrides.get(vip_pop_name, self._VIP_DEFAULTS)
        ngc_overrides = population_overrides.get(ngc_pop_name, self._NGC_DEFAULTS)

        # Total inhibitory neurons (25% of pyramidal = 20% of total)
        total_inhib = max(int(pyr_size * total_inhib_fraction), 10)

        # Divide into subtypes (based on cortical distributions)
        self.pv_size  = max(int(total_inhib * 0.40), 4)  # 40% - basket cells
        self.sst_size = max(int(total_inhib * 0.30), 3)  # 30% - Martinotti
        self.vip_size = max(int(total_inhib * 0.20), 2)  # 20% - disinhibitory
        self.ngc_size = max(int(total_inhib * 0.10), 2)  # 10% - L1 neurogliaform
        self.other_size = total_inhib - (self.pv_size + self.sst_size + self.vip_size + self.ngc_size)

        # PV+ Basket Cells (fast-spiking)
        self.pv_neurons: ConductanceLIF = _create_and_register_neurons_fn(
            population_name=pv_pop_name,
            n_neurons=self.pv_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                v_reset=pv_overrides.v_reset,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=3.0,
                tau_I=3.0,
                tau_ref=2.5,
                g_L=heterogeneous_g_L(0.10, self.pv_size, device, cv=0.08),
                tau_mem_ms=heterogeneous_tau_mem(pv_overrides.tau_mem_ms, self.pv_size, device, cv=0.10),
                v_threshold=heterogeneous_v_threshold(pv_overrides.v_threshold, self.pv_size, device, cv=0.06),
                adapt_increment=heterogeneous_adapt_increment(pv_overrides.adapt_increment, self.pv_size, device),
            ),
        )

        # SST+ Martinotti Cells (regular-spiking)
        self.sst_neurons: ConductanceLIF = _create_and_register_neurons_fn(
            population_name=sst_pop_name,
            n_neurons=self.sst_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                g_L=heterogeneous_g_L(0.05, self.sst_size, device),
                tau_E=5.0,
                tau_I=10.0,
                v_reset=sst_overrides.v_reset,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_mem_ms=heterogeneous_tau_mem(sst_overrides.tau_mem_ms, self.sst_size, device),
                v_threshold=heterogeneous_v_threshold(sst_overrides.v_threshold, self.sst_size, device),
                adapt_increment=heterogeneous_adapt_increment(sst_overrides.adapt_increment, self.sst_size, device),
                tau_adapt=sst_overrides.tau_adapt,
            ),
        )

        # VIP+ Interneurons (disinhibitory)
        self.vip_neurons: ConductanceLIF = _create_and_register_neurons_fn(
            population_name=vip_pop_name,
            n_neurons=self.vip_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                g_L=heterogeneous_g_L(0.05, self.vip_size, device),
                tau_E=5.0,
                tau_I=10.0,
                v_reset=vip_overrides.v_reset,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_mem_ms=heterogeneous_tau_mem(vip_overrides.tau_mem_ms, self.vip_size, device),
                v_threshold=heterogeneous_v_threshold(vip_overrides.v_threshold, self.vip_size, device),
                adapt_increment=heterogeneous_adapt_increment(vip_overrides.adapt_increment, self.vip_size, device),
                tau_adapt=vip_overrides.tau_adapt,
            ),
        )

        # L1 Neurogliaform (NGC) cells
        self.ngc_neurons: ConductanceLIF = _create_and_register_neurons_fn(
            population_name=ngc_pop_name,
            n_neurons=self.ngc_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                g_L=heterogeneous_g_L(0.05, self.ngc_size, device),
                tau_E=5.0,
                tau_I=10.0,
                v_reset=ngc_overrides.v_reset,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_mem_ms=heterogeneous_tau_mem(ngc_overrides.tau_mem_ms, self.ngc_size, device),
                v_threshold=heterogeneous_v_threshold(ngc_overrides.v_threshold, self.ngc_size, device),
                adapt_increment=heterogeneous_adapt_increment(ngc_overrides.adapt_increment, self.ngc_size, device),
                tau_adapt=ngc_overrides.tau_adapt,
            ),
        )

        # =====================================================================
        # SYNAPTIC WEIGHT MATRICES (E→I, I→E, I→I)
        # =====================================================================
        # All E→I, I→E, and I→I weight matrices for PV/SST/VIP/NGC populations
        # have been moved to the parent CorticalColumn via _add_internal_connection()
        # so they participate in the standard STP, diagnostic, and learning pipeline.
        #
        # This class is a NEURON CONTAINER: it holds neurons, gap junctions,
        # long-range input transforms (w_lr_vip, w_lr_ngc), ACh receptors, and
        # the PV membrane-voltage buffer needed for gap junction coupling.
        # I→I spike state buffers are owned by CorticalColumn (matching the
        # hippocampal pattern) to eliminate cross-module private access.

        # =====================================================================
        # VIP LONG-RANGE INPUT MATRIX (external input transform, not registered)
        # =====================================================================
        # Long-range top-down projections → VIP disinhibitory gate.
        # Not registered via _add_internal_connection because the source is an
        # external population (PFC/association cortex), not a local population.
        self.w_lr_vip = WeightInitializer.sparse_gaussian(
            n_input=self.pyr_size,
            n_output=self.vip_size,
            connectivity=0.4,
            mean=0.001,
            std=0.0005,
            device=device,
        )

        # =====================================================================
        # NICOTINIC ACh RECEPTOR ON VIP (alpha4beta2 subtype)
        # =====================================================================
        # Nicotinic α4β2 on VIP (ionotropic; τ_rise=3 ms, τ_decay=15 ms).
        self.ach_nicotinic_vip = make_neuromodulator_receptor(
            NMReceptorType.ACH_NICOTINIC, n_receptors=self.vip_size, dt_ms=dt_ms, device=device
        )

        # =====================================================================
        # NGC LONG-RANGE INPUT MATRIX (external input transform, not registered)
        # =====================================================================
        self.w_lr_ngc = WeightInitializer.sparse_gaussian(
            n_input=self.pyr_size,
            n_output=self.ngc_size,
            connectivity=0.3,
            mean=0.0008,
            std=0.0004,
            device=device,
        )

        # =====================================================================
        # MUSCARINIC ACh RECEPTOR ON NGC (M1 subtype, metabotropic)
        # =====================================================================
        # Muscarinic M1 on NGC (Gq → PLC/IP3): τ_rise=100 ms, τ_decay=1500 ms.
        # Very slow metabotropic cascade: sustained suppression of NGC axon-initial-segment
        # inhibition during attentional states (Bhagya et al. 2002).
        self.ach_muscarinic_ngc = make_neuromodulator_receptor(
            NMReceptorType.ACH_MUSCARINIC_M1, n_receptors=self.ngc_size, dt_ms=dt_ms, device=device
        )

        # =====================================================================
        # GAP JUNCTIONS (PV cells only — electrical coupling, not synaptic)
        # =====================================================================
        self.w_pv_gap = WeightInitializer.sparse_gaussian_no_autapses(
            n_input=self.pv_size,
            n_output=self.pv_size,
            connectivity=0.5,
            mean=0.004,
            std=0.001,
            device=device,
        )

        # =====================================================================
        # MEMBRANE STATE BUFFER (PV gap junction coupling)
        # =====================================================================
        # PV membrane voltage from the previous timestep is needed by the gap
        # junction computation inside forward().  I→I spike state buffers live
        # in CorticalColumn to avoid cross-module private access.
        self._pv_membrane_buffer = CircularDelayBuffer(max_delay=1, size=self.pv_size, dtype=torch.float32, device=device)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(
        self,
        pv_g_exc: torch.Tensor,
        pv_g_inh: torch.Tensor,
        sst_g_exc: torch.Tensor,
        sst_g_inh: torch.Tensor,
        vip_g_exc_from_pyr: torch.Tensor,
        vip_g_inh: torch.Tensor,
        ngc_g_exc_from_pyr: torch.Tensor,
        ngc_g_inh: torch.Tensor,
        feedforward_excitation: Optional[torch.Tensor] = None,
        long_range_excitation: Optional[torch.Tensor] = None,
        ach_spikes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run inhibitory interneuron populations given pre-computed synaptic conductances.

        All standard population→population synaptic conductances (E→I, I→E, I→I) are
        computed by the parent CorticalColumn using STP-modulated weight matrices
        registered via _add_internal_connection().  This method handles:
          - Long-range top-down inputs routed to VIP/NGC (via w_lr_vip / w_lr_ngc)
          - ACh receptor modulation (nicotinic VIP, muscarinic NGC)
          - Optional thalamic feedforward drive directly to PV
          - Gap junction coupling (electrical, PV cells only)
          - Neuron integration (all four cell types)

        Args:
            pv_g_exc: AMPA conductance to PV from Pyr (STP-matmul in parent) [pv_size].
            pv_g_inh: GABA_A conductance to PV from I→I (parent: PV→PV + SST→PV + VIP→PV).
            sst_g_exc: AMPA conductance to SST from Pyr [sst_size].
            sst_g_inh: GABA_A to SST (parent: PV→SST + VIP→SST) [sst_size].
            vip_g_exc_from_pyr: AMPA from Pyr to VIP (parent) [vip_size]. Long-range and
                ACh drives are ADDED HERE (they stay in this class, not registered).
            vip_g_inh: GABA_A to VIP (currently zero — VIP is not targeted) [vip_size].
            ngc_g_exc_from_pyr: AMPA from Pyr to NGC (parent) [ngc_size].
            ngc_g_inh: GABA_A to NGC (currently zero — NGC targets, not targeted) [ngc_size].
            feedforward_excitation: Optional direct AMPA drive to PV cells [pv_size].
                Used by L4 where thalamic afferents bypass pyramidal cells.
            long_range_excitation: Optional top-down drive in pyr-space [pyr_size].
                Routed to VIP via w_lr_vip and to NGC via w_lr_ngc.
            ach_spikes: Optional NB ACh spikes for nicotinic VIP + muscarinic NGC.

        Returns:
            Dict with pv_spikes, sst_spikes, vip_spikes, ngc_spikes, pv_membrane,
            sst_membrane, vip_membrane, ngc_membrane.
            I→E inhibitory conductances are computed by the parent from these spikes
            using the registered pv→pyr / sst→pyr / vip→pyr / ngc→pyr weights.
        """
        device = pv_g_exc.device

        # ------------------------------------------------------------------
        # L4 thalamic feedforward direct drive to PV
        # ------------------------------------------------------------------
        if feedforward_excitation is not None:
            assert feedforward_excitation.size(0) == self.pv_size
            pv_g_exc = pv_g_exc + feedforward_excitation

        # ------------------------------------------------------------------
        # Long-range top-down → VIP and NGC
        # ------------------------------------------------------------------
        if long_range_excitation is not None:
            pv_g_exc_vip = torch.matmul(self.w_lr_vip, long_range_excitation)
            ngc_lr_drive = torch.matmul(self.w_lr_ngc, long_range_excitation)
        else:
            pv_g_exc_vip = torch.zeros(self.vip_size, device=device)
            ngc_lr_drive = torch.zeros(self.ngc_size, device=device)

        # ACh nicotinic on VIP (ionotropic, fast)
        ach_nicotinic_drive = self.ach_nicotinic_vip.update(ach_spikes)
        vip_g_exc = vip_g_exc_from_pyr + pv_g_exc_vip + ach_nicotinic_drive * 0.25

        # ACh muscarinic on NGC (metabotropic, slow)
        ach_muscarinic_drive = self.ach_muscarinic_ngc.update(ach_spikes)
        ngc_g_exc = ngc_g_exc_from_pyr + ngc_lr_drive + ach_muscarinic_drive * 0.20

        # ------------------------------------------------------------------
        # Gap junction coupling (PV cells — bidirectional electrical synapse)
        # ------------------------------------------------------------------
        prev_pv_mem = self._pv_membrane_buffer.read(1)
        pv_gap_coupling = torch.matmul(self.w_pv_gap, prev_pv_mem) - prev_pv_mem * self.w_pv_gap.sum(dim=1)
        pv_g_exc = pv_g_exc + pv_gap_coupling * 0.3

        # ------------------------------------------------------------------
        # AMPA/NMDA split (AMPA-only for cortical interneurons; nmda_ratio=0.0)
        # ------------------------------------------------------------------
        # NMDA runaway issue: nmda_ratio=0.3 with tau_nmda=100ms creates
        # g_NMDA_ss that saturates PV/SST/VIP at 240-400 Hz. AMPA-only is correct.
        pv_g_ampa, pv_g_nmda = split_excitatory_conductance(pv_g_exc, nmda_ratio=0.0)
        sst_g_ampa, sst_g_nmda = split_excitatory_conductance(sst_g_exc, nmda_ratio=0.0)
        vip_g_ampa, vip_g_nmda = split_excitatory_conductance(vip_g_exc, nmda_ratio=0.0)
        ngc_g_ampa, ngc_g_nmda = split_excitatory_conductance(ngc_g_exc, nmda_ratio=0.0)

        # ------------------------------------------------------------------
        # Run interneurons
        # ------------------------------------------------------------------
        pv_spikes, pv_membrane = self.pv_neurons.forward(
            g_ampa_input=ConductanceTensor(pv_g_ampa),
            g_nmda_input=ConductanceTensor(pv_g_nmda),
            g_gaba_a_input=ConductanceTensor(pv_g_inh),
            g_gaba_b_input=None,
        )
        sst_spikes, sst_membrane = self.sst_neurons.forward(
            g_ampa_input=ConductanceTensor(sst_g_ampa),
            g_nmda_input=ConductanceTensor(sst_g_nmda),
            g_gaba_a_input=ConductanceTensor(sst_g_inh),
            g_gaba_b_input=None,
        )
        vip_spikes, vip_membrane = self.vip_neurons.forward(
            g_ampa_input=ConductanceTensor(vip_g_ampa),
            g_nmda_input=ConductanceTensor(vip_g_nmda),
            g_gaba_a_input=ConductanceTensor(vip_g_inh),
            g_gaba_b_input=None,
        )
        ngc_spikes, ngc_membrane = self.ngc_neurons.forward(
            g_ampa_input=ConductanceTensor(ngc_g_ampa),
            g_nmda_input=ConductanceTensor(ngc_g_nmda),
            g_gaba_a_input=ConductanceTensor(ngc_g_inh),
            g_gaba_b_input=None,
        )

        # ------------------------------------------------------------------
        # Advance PV membrane buffer (gap junction state)
        # ------------------------------------------------------------------
        self._pv_membrane_buffer.write_and_advance(pv_membrane)

        return {
            "pv_spikes": pv_spikes,
            "sst_spikes": sst_spikes,
            "vip_spikes": vip_spikes,
            "ngc_spikes": ngc_spikes,
            "pv_membrane": pv_membrane,
            "sst_membrane": sst_membrane,
            "vip_membrane": vip_membrane,
            "ngc_membrane": ngc_membrane,
        }
