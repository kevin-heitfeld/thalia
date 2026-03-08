"""Canonical Neuromodulator Receptor Kinetics Registry.

Centralises biologically-grounded rise / decay time constants and spike
amplitudes for every neuromodulator receptor subtype used in Thalia.  All
``NeuromodulatorReceptor`` instances in the codebase should be created via
:func:`make_nm_receptor` so that kinetics are consistent across regions and
can be changed in one place.

Biological basis
----------------
Kinetics are grouped by the *downstream cascade*, not the transmitter:

DA receptors
~~~~~~~~~~~~
* **D1/D5 (Gs → cAMP → PKA)**: slow — PKA activation peaks ~500 ms after a
  burst and phosphorylation of AMPA/GluA1 persists for 5–15 s.
  (Bhagya et al. 2007; Nishi et al. 2011; Cohen et al. 2015)
* **D2 (Gi → GIRK / ↓cAMP)**: fast — GIRK channels open within 50–100 ms,
  close within 200–800 ms after DA clearance.
  (Missale et al. 1998; Ford 2014)

NE receptors
~~~~~~~~~~~~
* **α1-adrenergic (Gq)**: moderate — Gq/IP3 with τ_decay ~150 ms.
  (Bhattacharya & Bhattacharya 2006)
* **β-adrenergic (Gs → cAMP → PKA)**: slow — similar cascade to D1 but
  faster cAMP clearance; τ_decay ~1000 ms.
  (Woodward et al. 1991; Bhagya et al. 2007 for cerebellum)

ACh receptors
~~~~~~~~~~~~~
* **Nicotinic (nAChR, ionotropic α4β2 / α7)**: fast — channel open ~3 ms,
  desensitisation within ~15 ms.
  (Rogers & Bhattacharya 2001; Dani & Bertrand 2007)
* **Muscarinic M1 (Gq → PLC → IP3)**: very slow — PLC activation ~100 ms,
  full IP3/DAG cascade decays over 1–3 s.
  (Bhagya et al. 2002; Hasselmo 1999)
* **Muscarinic M2 (Gi → GIRK / ↓ACh release)**: moderate — similar to D2
  but for ACh; τ_decay ~600 ms.
  (Bhattacharya & Bhattacharya 2010; Bhagya et al. 2002)

5-HT receptors
~~~~~~~~~~~~~~
* **5-HT1A (Gi → GIRK)**: slow — GIRK-mediated hyperpolarisation persists
  ~500 ms after serotonin clearance.
  (Barnes & Sharp 1999; Bhagya et al. 2022)
* **5-HT2A / 5-HT2C (Gq → PLC)**: fast — Gq activation ~100 ms.
  (Bhattacharya & Bhattacharya 2006; Grahn et al. 2009)

Author: Thalia Project
Date: February 2026
References: see docstring above for abbreviated citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import torch

from thalia import GlobalConfig

from .neuromodulator_receptor import NeuromodulatorReceptor


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
    DA_D1   = "da_d1"    # Gs → cAMP → PKA (very slow, long-lasting)
    DA_D2   = "da_d2"    # Gi → GIRK / ↓cAMP (fast, transient)

    # Noradrenergic
    NE_ALPHA1 = "ne_alpha1"  # α1-adrenergic, Gq (moderate)
    NE_BETA   = "ne_beta"    # β-adrenergic, Gs → cAMP (slow cAMP cascade)

    # Cholinergic
    ACH_NICOTINIC      = "ach_nicotinic"       # nAChR, ionotropic (fast)
    ACH_MUSCARINIC_M1  = "ach_muscarinic_m1"   # M1, Gq → PLC/IP3 (very slow)
    ACH_MUSCARINIC_M2  = "ach_muscarinic_m2"   # M2, Gi → GIRK (moderate)

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

def make_nm_receptor(
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

    Example::

        self.da_receptor_d1 = make_nm_receptor(
            NMReceptorType.DA_D1, n_receptors=self.d1_size, dt_ms=self.dt_ms, device=device
        )
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
