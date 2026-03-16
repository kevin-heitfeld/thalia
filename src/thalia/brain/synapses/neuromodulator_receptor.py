"""Neuromodulator Receptor System for Converting Spikes to Synaptic Concentration.

This module provides biologically-accurate conversion of neuromodulator neuron
spikes (DA, NE, ACh) into synaptic concentrations at target receptors, with
realistic release, diffusion, and reuptake dynamics.

Biological Background:
======================
Neuromodulators (dopamine, norepinephrine, acetylcholine) operate via:
1. **Volume transmission**: Diffuse through extracellular space (1-10 μm radius)
2. **Slow timescales**: Persist for 50-200+ ms (vs. 1-5 ms for fast synapses)
3. **G-protein coupled receptors**: Metabotropic (not ionotropic)
4. **Reuptake mechanisms**: Transporters (DAT, NET) or enzymatic degradation (AChE)

Dynamics:
=========
- **Release**: Fast (5-20 ms) concentration increase per presynaptic spike
- **Diffusion**: Spatial spread from release sites (volume transmission)
- **Binding**: Rapid receptor binding once present
- **Clearance**: Slow (50-200 ms) via reuptake or degradation

This provides the bridge between spiking neuromodulator neurons (VTA, LC, NB)
and the concentration-based three-factor learning rules in target regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import torch
import torch.nn as nn

from thalia import GlobalConfig
from thalia.utils import decay_float


class NeuromodulatorReceptor(nn.Module):
    """Convert neuromodulator spikes to synaptic concentration.

    Implements biologically-realistic dynamics:
    - Fast release on spike arrival (5-20 ms rise time)
    - Slow reuptake/degradation (50-200+ ms decay time)
    - Volume transmission (spatial averaging of spikes)
    - Physiological bounds [0, 1]

    This unified class handles DA (D1/D2), NE (α/β), and ACh (nicotinic/muscarinic)
    receptors with appropriate time constants.

    Attributes:
        concentration: Current receptor activation level [n_receptors] in [0, 1]
        rising: Fast rising component (models rapid release)
    """

    def __init__(
        self,
        n_receptors: int,
        tau_rise_ms: float,
        tau_decay_ms: float,
        spike_amplitude: float = 0.1,
        dt_ms: float = GlobalConfig.DEFAULT_DT_MS,
        device: Optional[torch.device] = None,
    ):
        """Initialize neuromodulator receptor system.

        Args:
            n_receptors: Number of postsynaptic receptor sites
            tau_rise_ms: Rise time constant in ms (fast release dynamics)
                      Typical: 5-20 ms
            tau_decay_ms: Decay time constant in ms (slow reuptake/degradation)
                       DA: ~200 ms, NE: ~150 ms, ACh: ~50 ms
            spike_amplitude: Concentration increase per presynaptic spike
                            Typically 0.1-0.2 to allow summation
            dt_ms: Simulation timestep in milliseconds.  Decay constants and
                   transfer rate are scaled by this value so dynamics are
                   independent of the chosen integration step.
            device: PyTorch device for tensor allocation
        """
        super().__init__()
        self.n_receptors = n_receptors
        self.tau_rise_ms = tau_rise_ms
        self.tau_decay_ms = tau_decay_ms
        self.spike_amplitude = spike_amplitude
        self.dt_ms = dt_ms

        # Synaptic concentration state [0, 1] — registered as buffers so
        # .to(device), state_dict(), and load_state_dict() all work correctly.
        self.concentration: torch.Tensor
        self.rising: torch.Tensor
        self.register_buffer("concentration", torch.zeros(n_receptors))
        self.register_buffer("rising", torch.zeros(n_receptors))

        # Decay factors scaled by dt_ms so dynamics are timestep-independent.
        # Using exp(-dt_ms / tau) gives the one-step decay factor for Euler integration.
        self.alpha_rise = decay_float(dt_ms, tau_rise_ms)
        self.alpha_decay = decay_float(dt_ms, tau_decay_ms)

        # Transfer rate: fraction of the rising pool moved to the concentration
        # pool each step.  Must equal (1 - alpha_decay) so that at steady state
        # concentration_ss == rising_ss.  Using any larger value (e.g. 0.5 * dt)
        # inflates the denominator ratio by ~4000× for slow receptors like DA_D1
        # (tau_decay=8000 ms), saturating concentration to 1.0 for any firing rate.
        self.transfer_rate: float = 1.0 - self.alpha_decay

        if device is not None:
            self.to(device)

    def update(self, neuromod_spikes: Optional[torch.Tensor]) -> torch.Tensor:
        """Update concentration from incoming neuromodulator spikes.

        Implements two-stage dynamics:
        1. Fast rising phase (tau_rise_ms): Spike → rapid concentration increase
        2. Slow decay phase (tau_decay_ms): Exponential clearance

        Args:
            neuromod_spikes: Presynaptic neuromodulator spikes [n_source]
                           Bool or float tensor. If None, just decay.

        Returns:
            Current concentration [n_receptors] in range [0, 1]
        """
        # Decay existing components
        self.rising.mul_(self.alpha_rise)
        self.concentration.mul_(self.alpha_decay)

        # No input → just decay
        if neuromod_spikes is None or neuromod_spikes.sum() == 0:
            return self.concentration

        # Convert to float and ensure correct device
        device = self.concentration.device
        spikes_float = neuromod_spikes.float().to(device)

        # Project spikes to receptor size if needed (volume transmission)
        if spikes_float.shape[0] != self.n_receptors:
            # Spatial averaging: models diffusion from release sites
            spike_rate = spikes_float.mean()
            spikes_float = torch.full((self.n_receptors,), spike_rate, device=device)

        # Fast release on spike arrival
        self.rising.add_(spikes_float, alpha=self.spike_amplitude)

        # Transfer from rising pool to concentration pool
        # (Models: release → diffusion → receptor binding)
        # transfer_rate is pre-scaled by dt_ms (set in __init__).
        self.concentration.add_(self.rising, alpha=self.transfer_rate)

        # Enforce physiological bounds
        self.concentration.clamp_(0.0, 1.0)

        return self.concentration

    def get_mean_concentration(self) -> float:
        """Get spatial average concentration.

        Useful for monitoring global neuromodulation levels
        and for regions that use global (not local) modulation.

        Returns:
            Mean concentration across all receptors
        """
        return self.concentration.mean().item()


# =============================================================================
# SPECIALIZED RECEPTOR CONSTRUCTORS
# =============================================================================


def create_dopamine_receptors(
    n_receptors: int,
    dt_ms: float = GlobalConfig.DEFAULT_DT_MS,
    device: Optional[torch.device] = None,
) -> NeuromodulatorReceptor:
    """Create D1/D2 dopamine receptors with biologically-accurate dynamics.

    Dopamine receptors have:
    - Moderate rise time (~10 ms)
    - Very slow decay (~200 ms) due to DAT reuptake
    - Moderate amplitude for stable three-factor learning

    Used by: Striatum, Prefrontal cortex, Hippocampus, Motor cortex

    Args:
        n_receptors: Number of receptor sites (typically matches neuron population)
        dt_ms: Simulation timestep in milliseconds.
        device: PyTorch device

    Returns:
        Configured NeuromodulatorReceptor for dopamine
    """
    return NeuromodulatorReceptor(
        n_receptors=n_receptors,
        tau_rise_ms=10.0,  # DA release dynamics
        tau_decay_ms=200.0,  # DAT reuptake (very slow)
        spike_amplitude=0.15,
        dt_ms=dt_ms,
        device=device,
    )


def create_norepinephrine_receptors(
    n_receptors: int,
    dt_ms: float = GlobalConfig.DEFAULT_DT_MS,
    device: Optional[torch.device] = None,
) -> NeuromodulatorReceptor:
    """Create α/β adrenergic (norepinephrine) receptors.

    Norepinephrine receptors have:
    - Fast rise time (~8 ms)
    - Moderate decay (~150 ms) via NET reuptake
    - Lower amplitude (more transient modulation)

    Used by: All cortical regions, Hippocampus, Cerebellum, Thalamus

    Args:
        n_receptors: Number of receptor sites
        dt_ms: Simulation timestep in milliseconds.
        device: PyTorch device

    Returns:
        Configured NeuromodulatorReceptor for norepinephrine
    """
    return NeuromodulatorReceptor(
        n_receptors=n_receptors,
        tau_rise_ms=8.0,
        tau_decay_ms=150.0,  # NET reuptake (moderate)
        spike_amplitude=0.12,
        dt_ms=dt_ms,
        device=device,
    )


def create_acetylcholine_receptors(
    n_receptors: int,
    dt_ms: float = GlobalConfig.DEFAULT_DT_MS,
    device: Optional[torch.device] = None,
) -> NeuromodulatorReceptor:
    """Create nicotinic/muscarinic (acetylcholine) receptors.

    Acetylcholine receptors have:
    - Very fast rise time (~5 ms)
    - Fast decay (~50 ms) via AChE enzymatic degradation
    - High amplitude for rapid encoding/retrieval switching

    Args:
        n_receptors: Number of receptor sites
        dt_ms: Simulation timestep in milliseconds.
        device: PyTorch device

    Returns:
        Configured NeuromodulatorReceptor for acetylcholine
    """
    return NeuromodulatorReceptor(
        n_receptors=n_receptors,
        tau_rise_ms=5.0,
        tau_decay_ms=50.0,  # AChE hydrolysis (very fast)
        spike_amplitude=0.2,
        dt_ms=dt_ms,
        device=device,
    )


# ============================================================================
# DATA CONTAINER
# ============================================================================


@dataclass(frozen=True)
class ReceptorKinetics:
    """Immutable kinetics specification for one receptor subtype.

    Attributes:
        tau_rise_ms: Rise time constant (ms) — models the lag from spike
            arrival to peak effector activation (e.g. PKA phosphorylation).
        tau_decay_ms: Decay time constant (ms) — models reuptake / enzyme
            degradation returning the effector to baseline.
        spike_amplitude: Fractional concentration increase per presynaptic
            spike (0–1 units).  Calibrated so a ~5 Hz tonic rate saturates
            to roughly 0.3–0.5 in the decay timescale.
    """
    tau_rise_ms: float
    tau_decay_ms: float
    spike_amplitude: float


# ============================================================================
# RECEPTOR SUBTYPE ENUM
# ============================================================================


class NMReceptorType(StrEnum):
    """Neuromodulator receptor subtypes with distinct downstream cascades.

    Naming follows the pharmacological convention:
    *transmitter*_*receptor_subtype* (or *ion_channel* for ionotropic).
    """
    # Dopaminergic
    DA_D1 = "da_d1"  # Gs → cAMP → PKA (very slow, long-lasting)
    DA_D2 = "da_d2"  # Gi → GIRK / ↓cAMP (fast, transient)

    # Noradrenergic
    NE_ALPHA1 = "ne_alpha1"  # α1-adrenergic, Gq (moderate)
    NE_BETA   = "ne_beta"    # β-adrenergic, Gs → cAMP (slow cAMP cascade)

    # Cholinergic
    ACH_NICOTINIC     = "ach_nicotinic"      # nAChR, ionotropic (fast)
    ACH_MUSCARINIC_M1 = "ach_muscarinic_m1"  # M1, Gq → PLC/IP3 (very slow)
    ACH_MUSCARINIC_M2 = "ach_muscarinic_m2"  # M2, Gi → GIRK (moderate)

    # Serotonergic
    SHT_1A = "5ht_1a"  # Gi → GIRK (slow)
    SHT_2A = "5ht_2a"  # Gq → PLC (fast)
    SHT_2C = "5ht_2c"  # Gq → PLC (fast, same cascade as 2A)


# ============================================================================
# CANONICAL KINETICS TABLE
# ============================================================================


CANONICAL_KINETICS: dict[NMReceptorType, ReceptorKinetics] = {
    # ------------------------------------------------------------------
    # Dopamine
    # ------------------------------------------------------------------
    # D1: Gs → adenylyl cyclase → ↑cAMP → PKA activation.
    # Onset: ~500 ms (PKA activation peak after a burst).
    # Decay: ~8 s (phosphodiesterase cleaves cAMP; DARPP-32 dephosphorylation).
    NMReceptorType.DA_D1: ReceptorKinetics(
        tau_rise_ms=500.0,
        tau_decay_ms=8000.0,
        spike_amplitude=0.08,
    ),
    # D2: Gi → ↓cAMP + GIRK channel opening.
    # Onset: ~80 ms (GIRK opens rapidly once Gβγ is released).
    # Decay: ~500 ms (GIRK closes after GTP hydrolysis / DAT clearance).
    NMReceptorType.DA_D2: ReceptorKinetics(
        tau_rise_ms=80.0,
        tau_decay_ms=500.0,
        spike_amplitude=0.12,
    ),
    # ------------------------------------------------------------------
    # Norepinephrine
    # ------------------------------------------------------------------
    # α1-adrenergic: Gq → PLC → DAG/IP3 → PKC.
    # Moderate kinetics: onset ~8–15 ms (fast Gq activation), decay ~150 ms.
    NMReceptorType.NE_ALPHA1: ReceptorKinetics(
        tau_rise_ms=10.0,
        tau_decay_ms=150.0,
        spike_amplitude=0.12,
    ),
    # β-adrenergic: Gs → cAMP → PKA (same cascade as D1, but faster PDE clearance).
    # Onset: ~80 ms.  Decay: ~1000 ms.
    NMReceptorType.NE_BETA: ReceptorKinetics(
        tau_rise_ms=80.0,
        tau_decay_ms=1000.0,
        spike_amplitude=0.10,
    ),
    # ------------------------------------------------------------------
    # Acetylcholine
    # ------------------------------------------------------------------
    # Nicotinic (α4β2 / α7): ionotropic — channel opens in <5 ms, desensitises
    # within ~15 ms.
    NMReceptorType.ACH_NICOTINIC: ReceptorKinetics(
        tau_rise_ms=3.0,
        tau_decay_ms=15.0,
        spike_amplitude=0.15,
    ),
    # Muscarinic M1: Gq → PLC → IP3/DAG.
    # Onset: ~100 ms (PLC activation lag), Decay: ~1500 ms (IP3-R & PKC decay).
    NMReceptorType.ACH_MUSCARINIC_M1: ReceptorKinetics(
        tau_rise_ms=100.0,
        tau_decay_ms=1500.0,
        spike_amplitude=0.10,
    ),
    # Muscarinic M2: Gi → GIRK + ↓ACh release (autoreceptor / Golgi cells).
    # Moderate: onset ~50 ms, decay ~600 ms.
    NMReceptorType.ACH_MUSCARINIC_M2: ReceptorKinetics(
        tau_rise_ms=50.0,
        tau_decay_ms=600.0,
        spike_amplitude=0.10,
    ),
    # ------------------------------------------------------------------
    # Serotonin
    # ------------------------------------------------------------------
    # 5-HT1A: Gi → GIRK (hyperpolarisation, gates fear extinction / DA pause).
    # Onset: ~10 ms (GIRK).  Decay: ~500 ms (SERT reuptake + GTP hydrolysis).
    NMReceptorType.SHT_1A: ReceptorKinetics(
        tau_rise_ms=10.0,
        tau_decay_ms=500.0,
        spike_amplitude=0.15,
    ),
    # 5-HT2A / 5-HT2C: Gq → PLC (attention gain, patience gating, striatal attenuation).
    # Fast Gq cascade: onset ~8 ms, decay ~100 ms (SERT reuptake).
    NMReceptorType.SHT_2A: ReceptorKinetics(
        tau_rise_ms=8.0,
        tau_decay_ms=100.0,
        spike_amplitude=0.12,
    ),
    NMReceptorType.SHT_2C: ReceptorKinetics(
        tau_rise_ms=8.0,
        tau_decay_ms=100.0,
        spike_amplitude=0.12,
    ),
}


# ============================================================================
# FACTORY HELPER
# ============================================================================

def make_neuromodulator_receptor(
    subtype: NMReceptorType,
    n_receptors: int,
    dt_ms: float = GlobalConfig.DEFAULT_DT_MS,
    device: Optional[torch.device] = None,
    *,
    amplitude_scale: float = 1.0,
) -> NeuromodulatorReceptor:
    """Create a ``NeuromodulatorReceptor`` with canonical kinetics for *subtype*.

    Args:
        subtype: Receptor subtype key from :class:`NMReceptorType`.
        n_receptors: Number of postsynaptic receptor sites.
        dt_ms: Simulation timestep (ms).
        device: PyTorch device.
        amplitude_scale: Optional multiplicative override on ``spike_amplitude``
            for regions with atypical innervation density.

    Returns:
        Configured :class:`NeuromodulatorReceptor` instance.
    """
    kinetics = CANONICAL_KINETICS[subtype]
    return NeuromodulatorReceptor(
        n_receptors=n_receptors,
        tau_rise_ms=kinetics.tau_rise_ms,
        tau_decay_ms=kinetics.tau_decay_ms,
        spike_amplitude=kinetics.spike_amplitude * amplitude_scale,
        dt_ms=dt_ms,
        device=device,
    )
