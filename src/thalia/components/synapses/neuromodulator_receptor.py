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

Author: Thalia Project
Date: February 2026
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class NeuromodulatorReceptor:
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
            device: PyTorch device for tensor allocation
        """
        self.n_receptors = n_receptors
        self.tau_rise_ms = tau_rise_ms
        self.tau_decay_ms = tau_decay_ms
        self.spike_amplitude = spike_amplitude
        self.device = device or torch.device("cpu")

        # Synaptic concentration state [0, 1]
        self.concentration = torch.zeros(n_receptors, device=self.device)

        # Rising phase component (fast release dynamics)
        self.rising = torch.zeros(n_receptors, device=self.device)

        # Decay factors (precomputed for efficiency)
        self.alpha_rise = math.exp(-1.0 / tau_rise_ms)
        self.alpha_decay = math.exp(-1.0 / tau_decay_ms)

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
        self.rising *= self.alpha_rise
        self.concentration *= self.alpha_decay

        # No input → just decay
        if neuromod_spikes is None or neuromod_spikes.sum() == 0:
            return self.concentration

        # Convert to float and ensure correct device
        spikes_float = neuromod_spikes.float().to(self.device)

        # Project spikes to receptor size if needed (volume transmission)
        if spikes_float.shape[0] != self.n_receptors:
            # Spatial averaging: models diffusion from release sites
            spike_rate = spikes_float.mean()
            spikes_float = torch.full(
                (self.n_receptors,), spike_rate, device=self.device
            )

        # Fast release on spike arrival
        self.rising += spikes_float * self.spike_amplitude

        # Transfer from rising pool to concentration pool
        # (Models: release → diffusion → receptor binding)
        transfer_fraction = 0.5  # Partial transfer each timestep
        self.concentration += self.rising * transfer_fraction

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
    n_receptors: int, device: Optional[torch.device] = None
) -> NeuromodulatorReceptor:
    """Create D1/D2 dopamine receptors with biologically-accurate dynamics.

    Dopamine receptors have:
    - Moderate rise time (~10 ms)
    - Very slow decay (~200 ms) due to DAT reuptake
    - Moderate amplitude for stable three-factor learning

    Used by: Striatum, Prefrontal cortex, Hippocampus, Motor cortex

    Args:
        n_receptors: Number of receptor sites (typically matches neuron population)
        device: PyTorch device

    Returns:
        Configured NeuromodulatorReceptor for dopamine
    """
    return NeuromodulatorReceptor(
        n_receptors=n_receptors,
        tau_rise_ms=10.0,  # DA release dynamics
        tau_decay_ms=200.0,  # DAT reuptake (very slow)
        spike_amplitude=0.15,
        device=device,
    )


def create_norepinephrine_receptors(
    n_receptors: int, device: Optional[torch.device] = None
) -> NeuromodulatorReceptor:
    """Create α/β adrenergic (norepinephrine) receptors.

    Norepinephrine receptors have:
    - Fast rise time (~8 ms)
    - Moderate decay (~150 ms) via NET reuptake
    - Lower amplitude (more transient modulation)

    Used by: All cortical regions, Hippocampus, Cerebellum, Thalamus

    Args:
        n_receptors: Number of receptor sites
        device: PyTorch device

    Returns:
        Configured NeuromodulatorReceptor for norepinephrine
    """
    return NeuromodulatorReceptor(
        n_receptors=n_receptors,
        tau_rise_ms=8.0,
        tau_decay_ms=150.0,  # NET reuptake (moderate)
        spike_amplitude=0.12,
        device=device,
    )


def create_acetylcholine_receptors(
    n_receptors: int, device: Optional[torch.device] = None
) -> NeuromodulatorReceptor:
    """Create nicotinic/muscarinic (acetylcholine) receptors.

    Acetylcholine receptors have:
    - Very fast rise time (~5 ms)
    - Fast decay (~50 ms) via AChE enzymatic degradation
    - High amplitude for rapid encoding/retrieval switching

    Used by: Cortex (especially sensory), Hippocampus, Striatum

    Args:
        n_receptors: Number of receptor sites
        device: PyTorch device

    Returns:
        Configured NeuromodulatorReceptor for acetylcholine
    """
    return NeuromodulatorReceptor(
        n_receptors=n_receptors,
        tau_rise_ms=5.0,
        tau_decay_ms=50.0,  # AChE hydrolysis (very fast)
        spike_amplitude=0.2,
        device=device,
    )
