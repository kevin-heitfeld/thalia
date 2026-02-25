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

from typing import Dict, Optional

import torch
import torch.nn as nn

from thalia import GlobalConfig
from thalia.components import NeuronFactory, NeuromodulatorReceptor, WeightInitializer
from thalia.typing import ConductanceTensor, PopulationName, RegionName
from thalia.utils import CircularDelayBuffer, split_excitatory_conductance


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
        expected_pyr_rate_hz: float = 3.0,
        pv_adapt_increment: float = 0.0,
        dt_ms: float = GlobalConfig.DEFAULT_DT_MS,
        device: str = "cpu",
    ):
        """Initialize inhibitory network for one cortical layer.

        Args:
            region_name: Region name (e.g., 'cortex') for RNG seeding
            population_name: Layer name (e.g., 'L23', 'L4') for RNG seeding
            pyr_size: Number of pyramidal neurons in this layer
            total_inhib_fraction: Fraction of pyramidal count (e.g. 0.25 = 20% of total)
            expected_pyr_rate_hz: Expected steady-state pyramidal firing rate (Hz).
                Used to calibrate E→I weights so PV and SST neurons fire in their
                biological range under realistic cortical activity.  Pass the
                layer-specific rate observed (or expected) from diagnostics.
            pv_adapt_increment: Spike-frequency adaptation increment for PV neurons.
                Default 0.0 (non-adapting fast-spiking, biologically canonical).
                L4 passes ~0.10 to prevent runaway firing when L4_pyr drive exceeds
                the w_pyr_pv design rate—a pragmatic rate-limiting brake.
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
        self.pv_size  = max(int(total_inhib * 0.40), 4)  # 40% - basket cells
        self.sst_size = max(int(total_inhib * 0.30), 3)  # 30% - Martinotti
        self.vip_size = max(int(total_inhib * 0.20), 2)  # 20% - disinhibitory
        self.ngc_size = max(int(total_inhib * 0.10), 2)  # 10% - L1 neurogliaform
        self.other_size = total_inhib - (self.pv_size + self.sst_size + self.vip_size + self.ngc_size)

        # PV+ Basket Cells (fast-spiking)
        # Use fast_spiking factory (already has tau_mem=8ms, heterogeneous)
        # Override for slightly faster dynamics (5ms mean).
        # pv_adapt_increment is 0.0 by default (non-adapting fast-spiking), but L4
        # passes a non-zero value to prevent overdrive when L4_pyr fires at 1-2 Hz
        # with w_pyr_pv calibrated at a lower expected rate.
        self.pv_neurons = NeuronFactory.create_fast_spiking_neurons(
            region_name=region_name,
            population_name=f"{population_name}_pv",
            n_neurons=self.pv_size,
            device=self.device,
            tau_mem=5.0,
            v_threshold=0.9,
            adapt_increment=pv_adapt_increment,
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

        # L1 Neurogliaform (NGC) cells
        # Slow, regular-spiking with moderate adaptation.
        # tau_mem=15ms matches real NGC (Jiang et al. 2015 Table S3: ~12-18ms).
        # v_threshold=1.0 keeps them below spontaneous threshold; they fire only
        # when recruited by genuine top-down or local pyramidal drive.
        self.ngc_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=region_name,
            population_name=f"{population_name}_ngc",
            n_neurons=self.ngc_size,
            device=self.device,
            tau_mem=15.0,
            v_threshold=1.0,
            adapt_increment=0.03,
            tau_adapt=100.0,
        )

        # =====================================================================
        # EXCITATORY → INHIBITORY (E→I)
        # =====================================================================
        # FAN-IN NORMALISATION
        # Different cortical layers have very different pyramidal population sizes.
        # Without normalisation large layers (L23, L6b) drive PV/VIP far harder than small ones.
        #
        # The biologically meaningful quantity is the TOTAL DRIVE each inhibitory
        # neuron receives, summed across all its inputs:
        #   total_drive = pyr_size × connectivity × mean_weight
        # This should be constant regardless of layer size.  Rearranging:
        #   mean_weight = total_drive / (pyr_size × connectivity)

        # Pyr → PV: target total weight per PV cell.
        _g_L_pv    = 0.10   # fast-spiking factory
        _tau_E_pv  = 3.0    # fast-spiking factory AMPA time constant
        _E_E       = 3.0    # normalised reversal potential
        _v_inf_pv  = 0.95   # slightly above threshold 0.9
        _g_E_pv_needed = _v_inf_pv * _g_L_pv / (_E_E - _v_inf_pv)
        _rate_spk_ms   = max(expected_pyr_rate_hz, 0.1) / 1000.0    # avoid division by zero
        _PV_TOTAL  = _g_E_pv_needed / (_rate_spk_ms * _tau_E_pv)
        _pv_connectivity = 0.5
        _pv_mean   = _PV_TOTAL / (self.pyr_size * _pv_connectivity)
        self.w_pyr_pv = WeightInitializer.sparse_gaussian(
            n_input=self.pyr_size,
            n_output=self.pv_size,
            connectivity=_pv_connectivity,
            mean=_pv_mean,
            std=_pv_mean * 0.375,
            device=device,
        )

        # Pyr → SST: target total weight per SST cell.
        _g_L_sst    = 0.05   # pyramidal factory
        _tau_E_sst  = 5.0    # pyramidal factory AMPA time constant
        _v_inf_sst  = 1.05   # slightly above threshold 1.0
        _g_E_sst_needed = _v_inf_sst * _g_L_sst / (_E_E - _v_inf_sst)
        _SST_TOTAL = _g_E_sst_needed / (_rate_spk_ms * _tau_E_sst)
        _sst_connectivity = 0.3
        _sst_mean  = _SST_TOTAL / (self.pyr_size * _sst_connectivity)
        self.w_pyr_sst = WeightInitializer.sparse_gaussian(
            n_input=self.pyr_size,
            n_output=self.sst_size,
            connectivity=_sst_connectivity,
            mean=_sst_mean,
            std=_sst_mean * 0.6,
            device=device,
        )

        # Pyr → VIP: target total weight per VIP cell ≈ 0.9 (AMPA-only)
        _VIP_TOTAL = 0.9
        _vip_connectivity = 0.4
        _vip_mean  = _VIP_TOTAL / (self.pyr_size * _vip_connectivity)
        self.w_pyr_vip = WeightInitializer.sparse_gaussian(
            n_input=self.pyr_size,
            n_output=self.vip_size,
            connectivity=_vip_connectivity,
            mean=_vip_mean,
            std=_vip_mean * 0.2,
            device=device,
        )

        # Pyr → NGC: weak local collateral excitation (P=0.2)
        # NGC cells receive recurrent excitation from L2/3 pyramidal collaterals.
        # Drive is intentionally weak — NGC fires only when both local activity AND
        # top-down input coincide (coincidence-detection role).
        _ngc_connectivity = 0.2
        _ngc_mean = 0.4 / (self.pyr_size * _ngc_connectivity)  # target total drive ≈ 0.4
        self.w_pyr_ngc = WeightInitializer.sparse_gaussian(
            n_input=self.pyr_size,
            n_output=self.ngc_size,
            connectivity=_ngc_connectivity,
            mean=_ngc_mean,
            std=_ngc_mean * 0.3,
            device=device,
        )

        # =====================================================================
        # INHIBITORY → EXCITATORY (I→E)
        # =====================================================================
        # PV → Pyr (perisomatic inhibition - gamma gating)
        # Reverted 0.006 → 0.003: the 0.006 overcorrection combined with L4 v_threshold=1.8
        # made L4_pyr completely un-fireable. With corrected L4 threshold=1.1 and PV at
        # 30-60 Hz, mean=0.003 provides adequate perisomatic inhibition for gamma gating.
        self.w_pv_pyr = WeightInitializer.sparse_gaussian(
            n_input=self.pv_size,
            n_output=self.pyr_size,
            connectivity=0.6,
            mean=0.003,
            std=0.002,
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

        # NGC → Pyr: diffuse weak GABA_A on apical tuft dendrites (P=0.5)
        # Biology: NGC axons spread like a "cottonball" covering ~50% of local
        # pyramidal apical tufts with weak GABA_A conductances (Jiang et al. 2015).
        # This gates the DEGREE to which top-down feedback modulates the pyramidal
        # apical compartment — not a strong driver of somatic inhibition.
        self.w_ngc_pyr = WeightInitializer.sparse_gaussian(
            n_input=self.ngc_size,
            n_output=self.pyr_size,
            connectivity=0.5,
            mean=0.0008,
            std=0.0004,
            device=device,
        )

        # =====================================================================
        # INHIBITORY → INHIBITORY (I→I)
        # =====================================================================
        # PV → PV (weak lateral inhibition)
        self.w_pv_pv = WeightInitializer.sparse_gaussian_no_autapses(
            n_input=self.pv_size,
            n_output=self.pv_size,
            connectivity=0.3,
            mean=0.005,
            std=0.002,
            device=device,
        )

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

        # VIP → SST (strong disinhibition - primary VIP target, ~80% per Pfeffer et al. 2013)
        self.w_vip_sst = WeightInitializer.sparse_gaussian(
            n_input=self.vip_size,
            n_output=self.sst_size,
            connectivity=0.8,
            mean=0.0015,
            std=0.003,
            device=device,
        )

        # =====================================================================
        # VIP LONG-RANGE INPUT MATRIX
        # =====================================================================
        # Biology: VIP cells are the primary cortical recipients of top-down
        # long-range corticocortical projections (from PFC, association cortex).
        # This weight matrix projects from the layer's pyramidal population size
        # (since long-range input size matches that of the local excitatory signal)
        # into VIP space, implementing the attentional disinhibitory gate:
        #   top-down input → VIP fires → SST suppressed → apical disinhibition
        # Reference: Zhang et al. 2014, Pi et al. 2013
        # Weight reduced from 0.012 → 0.004 → 0.001: the previous values drove non-L4
        # VIP into saturation at 107-186 Hz. At mean=0.001 long-range inputs produce
        # moderate VIP activity in the 2-20 Hz biological range during sustained input.
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
        # Biology: VIP interneurons are selectively excited by ACh via nicotinic
        # alpha4beta2 receptors (Pi et al. 2013; Arroyo et al. 2012).
        # These are IONOTROPIC (fast): tau_rise ~2ms, tau_decay ~20ms,
        # contrasting with muscarinic metabotropic effects on pyramidal cells.
        # NB → ACh spikes → nicotinic depolarisation of VIP → VIP→SST disinhibition
        self.ach_nicotinic_vip = NeuromodulatorReceptor(
            n_receptors=self.vip_size,
            tau_rise_ms=2.0,
            tau_decay_ms=20.0,
            spike_amplitude=0.15,
            device=device,
        )

        # =====================================================================
        # NGC LONG-RANGE INPUT MATRIX
        # =====================================================================
        # Biology: L1 NGC cells are the primary cortical recipients of long-range
        # top-down axons whose terminals arborise in L1 (Cauller 1995; Felleman &
        # Van Essen 1991).  These are the axons from PFC, association cortex, and
        # higher-order thalamus that deliver predictive / attentional signals.
        #
        # Circuit: top-down AMPA → NGC excitation → NGC → GABA_A on pyramidal
        #   apical tuft → controlled gating of whether feedback reaches the
        #   pyramidal soma via the apical Ca²⁺ spike.  High ACh (M1 muscarinic)
        #   boosts NGC recruitment → tighter feedback gating during attention.
        #
        # Weight uses the same pyr-space indexing as w_lr_vip because long-range
        # axons innervate both L1 (NGC) and L2/3 (VIP) proportionally.
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
        # Biology: L1 NGC cells express M1 muscarinic ACh receptors (Abs et al.
        # 2018).  Unlike the nicotinic alpha4beta2 on VIP cells (fast, ionotropic),
        # muscarinic M1 is METABOTROPIC: tau_rise ~15ms, tau_decay ~200ms.
        # The slow decay sustains NGC activation for hundreds of milliseconds after
        # a cholinergic volley — matching the sustained attention window.
        # Net effect: ACh burst → slow NGC depolarisation → NGC fires with delay
        #   → GABA_A gate on apical tuft opens → feedback gating adjusts.
        self.ach_muscarinic_ngc = NeuromodulatorReceptor(
            n_receptors=self.ngc_size,
            tau_rise_ms=15.0,
            tau_decay_ms=200.0,
            spike_amplitude=0.10,
            device=device,
        )

        # =====================================================================
        # GAP JUNCTIONS (PV cells only)
        # =====================================================================
        self.w_pv_gap = WeightInitializer.sparse_gaussian_no_autapses(
            n_input=self.pv_size,
            n_output=self.pv_size,
            connectivity=0.5,
            mean=0.004,
            std=0.001,
            device=device,
        )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(
        self,
        pyr_spikes: torch.Tensor,
        pyr_membrane: torch.Tensor,
        external_excitation: torch.Tensor,
        feedforward_excitation: Optional[torch.Tensor] = None,
        long_range_excitation: Optional[torch.Tensor] = None,
        ach_spikes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run inhibitory network for one timestep.

        Args:
            pyr_spikes: Pyramidal neuron spikes [pyr_size], bool
            pyr_membrane: Pyramidal membrane potentials [pyr_size], float
            external_excitation: External excitation to layer [pyr_size], float
            feedforward_excitation: Direct excitation to PV cells [pv_size] for feedforward
                inhibition (e.g., thalamic afferents driving PV directly).
            long_range_excitation: Top-down / long-range corticocortical input projected
                into this layer's pyramidal space [pyr_size].  Routed to:
                (1) VIP cells via ``w_lr_vip`` — disinhibitory attention gate, and
                (2) NGC cells via ``w_lr_ngc`` — L1 feedback filter on apical tuft.
            ach_spikes: Raw ACh spike tensor from the nucleus basalis (or None if
                neuromodulation is disabled).  Processed by the nicotinic alpha4beta2
                receptor on VIP cells, producing fast (tau ~20 ms) ionotropic excitation.

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
            # No thalamic feedforward available (layers other than L4):
            # PV cells are driven exclusively by local pyramidal recurrent activity.
            # Do NOT route external_excitation through w_pyr_pv — that matrix maps
            # pyr_size → pv_size, but external_excitation is a conductance vector
            # (not spikes), so multiplying it produces an always-on, unbounded drive
            # that saturates PV cells at ~400 Hz.
            pv_exc = pv_exc_from_pyr

        # SST cells: Driven by pyramidal activity
        sst_exc_from_pyr = torch.matmul(self.w_pyr_sst, pyr_spikes_float)
        sst_exc = sst_exc_from_pyr

        # VIP cells: Driven by pyramidal activity + long-range inputs + nicotinic ACh
        vip_exc_from_pyr = torch.matmul(self.w_pyr_vip, pyr_spikes_float)

        # S2-3 — Long-range (top-down) inputs preferentially routed to VIP cells.
        # Biology: VIP interneurons are primary recipients of long-range projections
        # from PFC and association cortex; firing VIP releases SST inhibition from
        # pyramidal apical dendrites (disinhibitory attention gate, Zhang et al. 2014).
        if long_range_excitation is not None:
            vip_lr_drive = torch.matmul(self.w_lr_vip, long_range_excitation)
        else:
            vip_lr_drive = torch.zeros(self.vip_size, device=device)

        # S2-4 — Nicotinic ACh (alpha4beta2) excitation of VIP cells.
        # Biology: VIP interneurons express nicotinic receptors whose ionotropic
        # kinetics (tau_rise=2ms, tau_decay=20ms) are much faster than the muscarinic
        # M1/M4 receptors on pyramidal cells.  This means ACh bursts from NB rapidly
        # gate attention via VIP before slower pyramidal modulation takes effect.
        ach_nicotinic_drive = self.ach_nicotinic_vip.update(ach_spikes)  # [vip_size]
        vip_exc = vip_exc_from_pyr + vip_lr_drive + ach_nicotinic_drive * 0.25

        # NGC cells: driven by pyramidal collaterals + long-range top-down + muscarinic ACh
        # Biology: NGC fires on coincidence of (a) local L2/3 recurrent activity and
        # (b) top-down input arriving in L1.  When both are present, NGC provides a
        # controlled GABA_A gate on the pyramidal apical tuft.
        ngc_exc_from_pyr = torch.matmul(self.w_pyr_ngc, pyr_spikes_float)
        if long_range_excitation is not None:
            ngc_lr_drive = torch.matmul(self.w_lr_ngc, long_range_excitation)
        else:
            ngc_lr_drive = torch.zeros(self.ngc_size, device=device)
        ach_muscarinic_drive = self.ach_muscarinic_ngc.update(ach_spikes)  # [ngc_size]
        ngc_exc = ngc_exc_from_pyr + ngc_lr_drive + ach_muscarinic_drive * 0.20

        # =====================================================================
        # COMPUTE INHIBITION TO INHIBITORY POPULATIONS (I→I)
        # =====================================================================

        # Get previous spikes (for I→I connections) using CircularDelayBuffer
        # Initialize buffers if first timestep
        if not hasattr(self, '_pv_spike_buffer'):
            self._pv_spike_buffer = CircularDelayBuffer(max_delay=1, size=self.pv_size, dtype=torch.bool, device=device)
            self._sst_spike_buffer = CircularDelayBuffer(max_delay=1, size=self.sst_size, dtype=torch.bool, device=device)
            self._vip_spike_buffer = CircularDelayBuffer(max_delay=1, size=self.vip_size, dtype=torch.bool, device=device)
            self._ngc_spike_buffer = CircularDelayBuffer(max_delay=1, size=self.ngc_size, dtype=torch.bool, device=device)
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

        # NGC inhibition: NGC cells are not meaningfully targeted by other inhibitory types
        ngc_inh = torch.zeros(self.ngc_size, device=device)

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

        # Cortical interneurons (PV, SST, VIP) are AMPA-dominated: they express NMDA receptors
        # at much lower density than pyramidal cells and their thin dendrites show minimal
        # voltage-dependent NMDA unblocking.  Using nmda_ratio=0.3 with tau_nmda=100ms creates
        # g_NMDA_ss = input_per_step × 100, a runaway NMDA attractor that saturates all three
        # cell types at 240-400 Hz regardless of pyramidal firing rate.  Setting nmda_ratio=0.0
        # removes this artefact; AMPA alone provides the biologically correct drive.

        # PV cells (fast-spiking, AMPA only)
        pv_g_ampa, pv_g_nmda = split_excitatory_conductance(pv_exc, nmda_ratio=0.0)
        pv_spikes, pv_membrane = self.pv_neurons.forward(
            g_ampa_input=ConductanceTensor(pv_g_ampa),
            g_nmda_input=ConductanceTensor(pv_g_nmda),
            g_gaba_a_input=ConductanceTensor(pv_inh),
            g_gaba_b_input=None,
        )

        # SST cells (regular-spiking, AMPA only)
        sst_g_ampa, sst_g_nmda = split_excitatory_conductance(sst_exc, nmda_ratio=0.0)
        sst_spikes, sst_membrane = self.sst_neurons.forward(
            g_ampa_input=ConductanceTensor(sst_g_ampa),
            g_nmda_input=ConductanceTensor(sst_g_nmda),
            g_gaba_a_input=ConductanceTensor(sst_inh),
            g_gaba_b_input=None,
        )

        # VIP cells (disinhibitory, AMPA only)
        vip_g_ampa, vip_g_nmda = split_excitatory_conductance(vip_exc, nmda_ratio=0.0)
        vip_spikes, vip_membrane = self.vip_neurons.forward(
            g_ampa_input=ConductanceTensor(vip_g_ampa),
            g_nmda_input=ConductanceTensor(vip_g_nmda),
            g_gaba_a_input=ConductanceTensor(vip_inh),
            g_gaba_b_input=None,
        )

        # NGC cells (slow, regular-spiking, AMPA only)
        # Uses AMPA-only drive (same reasoning as other cortical interneurons: no NMDA
        # runaway attractor).  Slow tau_mem and adaptation produce sparse, sustained firing.
        ngc_g_ampa, ngc_g_nmda = split_excitatory_conductance(ngc_exc, nmda_ratio=0.0)
        ngc_spikes, ngc_membrane = self.ngc_neurons.forward(
            g_ampa_input=ConductanceTensor(ngc_g_ampa),
            g_nmda_input=ConductanceTensor(ngc_g_nmda),
            g_gaba_a_input=ConductanceTensor(ngc_inh),
            g_gaba_b_input=None,
        )

        # Store for next timestep using CircularDelayBuffer
        self._pv_spike_buffer.write(pv_spikes)
        self._sst_spike_buffer.write(sst_spikes)
        self._vip_spike_buffer.write(vip_spikes)
        self._ngc_spike_buffer.write(ngc_spikes)
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

        # NGC → Pyr (diffuse weak GABA_A on apical tuft)
        # Biology: NGC axons arborise widely across L1 providing a gentle, spatially
        # diffuse inhibitory gate on pyramidal apical tufts (Jiang et al. 2015).
        # Weight 0.12 keeps the effect sub-threshold on its own; it only tips the balance
        # when combined with SST dendritic inhibition during high attention states.
        ngc_to_pyr = torch.matmul(self.w_ngc_pyr, ngc_spikes.float())

        # Total inhibition (weighted combination)
        # PV provides strongest, fastest inhibition (gamma gating)
        # SST provides slower, dendritic modulation
        # NGC provides diffuse L1 apical gating (feedback filter)
        # NOTE: These are ADDITIONAL to baseline inhibition (50% in cortex.py L4)
        # With threshold 2.0, PV needs to be strong to counteract drive
        total_inhibition = (
            perisomatic_inhibition * 1.0 +  # Strong PV inhibition (threshold 2.0 compensates)
            dendritic_inhibition * 0.3 +    # SST provides modulatory inhibition
            vip_to_pyr * 0.05 +             # VIP has minimal direct effect
            ngc_to_pyr * 0.12               # NGC: diffuse apical gating (weak but widespread)
        )

        return {
            "total_inhibition": total_inhibition,
            "pv_spikes": pv_spikes,
            "sst_spikes": sst_spikes,
            "vip_spikes": vip_spikes,
            "ngc_spikes": ngc_spikes,
            "perisomatic_inhibition": perisomatic_inhibition,
            "dendritic_inhibition": dendritic_inhibition,
            "pv_membrane": pv_membrane,
            "sst_membrane": sst_membrane,
            "vip_membrane": vip_membrane,
            "ngc_membrane": ngc_membrane,
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
        self.ngc_neurons.update_temporal_parameters(dt_ms)
