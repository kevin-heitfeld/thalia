"""
Oscillator utility functions for theta-gamma-alpha phase computations.

This module provides centralized implementations of oscillator-based
modulation patterns used across brain regions, eliminating code duplication
and ensuring consistent biological behavior.

Functions:
- compute_theta_encoding_retrieval(): Theta-phase encoding/retrieval modulation
- compute_ach_recurrent_suppression(): ACh-mediated recurrent suppression
- compute_gamma_phase_gate(): Gamma-phase attention gating

Author: Thalia Project
Date: December 21, 2025 (Architecture Review Tier 1.2)
"""

from __future__ import annotations

import math

from thalia.regulation.oscillator_constants import (
    THETA_ENCODING_PHASE_SCALE,
    THETA_RETRIEVAL_PHASE_SCALE,
    ACH_RECURRENT_SUPPRESSION,
    ACH_THRESHOLD_FOR_SUPPRESSION,
    GAMMA_ATTENTION_THRESHOLD,
    THETA_GAMMA_PHASE_DIFF_SIGMA,
)


def compute_theta_encoding_retrieval(theta_phase: float) -> tuple[float, float]:
    """Compute theta-phase encoding/retrieval modulation.

    Encoding peaks at theta peak (phase=0), retrieval at trough (phase=π).
    This pattern is used throughout hippocampus, cortex, and other regions
    to coordinate encoding/retrieval dynamics.

    Args:
        theta_phase: Theta phase in radians [0, 2π]

    Returns:
        (encoding_mod, retrieval_mod): Both in range [0.0, 1.0]
        - encoding_mod: Peaks at theta peak (phase=0), minimizes at trough
        - retrieval_mod: Peaks at theta trough (phase=π), minimizes at peak

    Biological Basis:
        Theta rhythm coordinates encoding and retrieval in hippocampus and cortex:
        - **Theta Peak (0°)**: Strong DG→CA3 (pattern separation), weak EC→CA1
        - **Theta Trough (180°)**: Strong EC→CA1 (pattern completion), weak DG→CA3

        This segregation prevents interference between encoding and retrieval.

    References:
        - Hasselmo et al. (2002): Theta rhythm and encoding/retrieval dynamics
        - Buzsáki & Draguhn (2004): Neuronal oscillations in cortical networks
        - Colgin (2016): Rhythms of the hippocampal network

    Examples:
        >>> # At theta peak (encoding mode)
        >>> enc, ret = compute_theta_encoding_retrieval(0.0)
        >>> assert enc == 1.0 and ret == 0.0

        >>> # At theta trough (retrieval mode)
        >>> enc, ret = compute_theta_encoding_retrieval(math.pi)
        >>> assert enc == 0.0 and ret == 1.0

        >>> # At rising phase (balanced)
        >>> enc, ret = compute_theta_encoding_retrieval(math.pi / 2)
        >>> assert abs(enc - 0.5) < 0.01 and abs(ret - 0.5) < 0.01
    """
    encoding_mod = THETA_ENCODING_PHASE_SCALE * (1.0 + math.cos(theta_phase))
    retrieval_mod = THETA_RETRIEVAL_PHASE_SCALE * (1.0 - math.cos(theta_phase))
    return encoding_mod, retrieval_mod


def compute_ach_recurrent_suppression(ach_level: float) -> float:
    """Compute ACh-mediated suppression of recurrent connections.

    High ACh suppresses recurrence to prioritize new encoding over retrieval.
    This is a key mechanism for mode switching between encoding and retrieval.

    Args:
        ach_level: Acetylcholine level [0.0, 1.0]

    Returns:
        Multiplicative gain [0.3, 1.0] for recurrent weights
        - 1.0: No suppression (low ACh, retrieval mode)
        - 0.3: Maximum suppression (high ACh, encoding mode)

    Biological Basis:
        Acetylcholine modulates the balance between afferent and recurrent input:
        - **Low ACh (<0.5)**: Full recurrence for memory retrieval
        - **High ACh (>0.5)**: Suppressed recurrence for new encoding

        This prevents interference from stored patterns during encoding.

    References:
        - Hasselmo & McGaughy (2004): Acetylcholine and learning
        - Hasselmo (2006): The role of acetylcholine in learning and memory
        - Douchamps et al. (2013): Evidence for encoding vs. retrieval modes

    Examples:
        >>> # Low ACh: Full recurrence (retrieval mode)
        >>> gain = compute_ach_recurrent_suppression(0.0)
        >>> assert gain == 1.0

        >>> # Medium ACh: Partial suppression
        >>> gain = compute_ach_recurrent_suppression(0.5)
        >>> assert gain == 1.0

        >>> # High ACh: Strong suppression (encoding mode)
        >>> gain = compute_ach_recurrent_suppression(1.0)
        >>> assert gain == 0.3
    """
    if ach_level <= ACH_THRESHOLD_FOR_SUPPRESSION:
        return 1.0

    # Linear suppression above threshold
    suppression_factor = (
        (ach_level - ACH_THRESHOLD_FOR_SUPPRESSION) /
        (1.0 - ACH_THRESHOLD_FOR_SUPPRESSION)
    )
    return 1.0 - ACH_RECURRENT_SUPPRESSION * suppression_factor


def compute_gamma_phase_gate(
    gamma_phase: float,
    threshold: float = GAMMA_ATTENTION_THRESHOLD,
) -> float:
    """Compute gamma-phase attention gating.

    Gamma rhythm creates temporal windows for selective processing.
    Neurons only respond to inputs arriving during specific gamma phases.

    Args:
        gamma_phase: Gamma phase in radians [0, 2π]
        threshold: Gating threshold [0.0, 1.0] (default from constants)

    Returns:
        Gating factor [0.0, 1.0]:
        - 1.0 at gamma peak (optimal processing window)
        - 0.0 at gamma trough (suppressed processing)

    Biological Basis:
        Gamma oscillations create temporal windows for binding and attention:
        - **Gamma Peak**: Excitatory window for processing selected inputs
        - **Gamma Trough**: Inhibitory suppression of unselected inputs

        This implements attentional selection at millisecond timescales.

    References:
        - Fries (2009): Neuronal gamma-band synchronization as a fundamental
          process in cortical computation
        - Buzsáki & Wang (2012): Mechanisms of gamma oscillations
        - Ni et al. (2016): Gamma-rhythmic gain modulation

    Examples:
        >>> # At gamma peak (full processing)
        >>> gate = compute_gamma_phase_gate(0.0)
        >>> assert gate == 1.0

        >>> # At gamma trough (suppressed)
        >>> gate = compute_gamma_phase_gate(math.pi)
        >>> assert gate == 0.0
    """
    # Cosine gating: 1.0 at peak (phase=0), 0.0 at trough (phase=π)
    gate_value = 0.5 * (1.0 + math.cos(gamma_phase))

    # Apply threshold (below threshold → 0, above threshold → linear)
    if gate_value < threshold:
        return 0.0
    else:
        return (gate_value - threshold) / (1.0 - threshold)


def compute_theta_gamma_coupling_gate(
    theta_phase: float,
    gamma_phase: float,
    sigma: float = THETA_GAMMA_PHASE_DIFF_SIGMA,
) -> float:
    """Compute theta-gamma cross-frequency coupling gating.

    Gamma amplitude is modulated by theta phase, creating nested oscillations.
    This implements phase-amplitude coupling observed in hippocampus.

    Args:
        theta_phase: Theta phase in radians [0, 2π]
        gamma_phase: Gamma phase in radians [0, 2π]
        sigma: Gaussian width for phase-difference gating

    Returns:
        Coupling strength [0.0, 1.0]
        - 1.0: Phases aligned (strong gamma during theta peak)
        - ~0: Phases misaligned (weak gamma during theta trough)

    Biological Basis:
        Theta-gamma coupling coordinates multi-timescale processing:
        - **Theta**: Organizes sequences and encoding/retrieval cycles
        - **Gamma**: Binds features within each theta cycle
        - **Coupling**: Gamma amplitude peaks at specific theta phases

    References:
        - Lisman & Jensen (2013): The theta-gamma neural code
        - Tort et al. (2009): Measuring phase-amplitude coupling
        - Colgin (2015): Theta-gamma coupling in memory

    Examples:
        >>> # Aligned phases (strong coupling)
        >>> coupling = compute_theta_gamma_coupling_gate(0.0, 0.0)
        >>> assert coupling == 1.0

        >>> # Misaligned phases (weak coupling)
        >>> coupling = compute_theta_gamma_coupling_gate(0.0, math.pi)
        >>> assert coupling < 0.1
    """
    # Phase difference
    phase_diff = ((gamma_phase - theta_phase + math.pi) % (2 * math.pi)) - math.pi

    # Gaussian gating centered at zero phase difference
    gating = math.exp(-(phase_diff ** 2) / (2 * sigma ** 2))

    return gating


def compute_oscillator_modulated_gain(
    base_gain: float,
    modulation_range: float,
    modulation_value: float,
) -> float:
    """Generic function for oscillator-modulated gain computation.

    Many regions use the pattern: gain = base + range * modulation
    This utility standardizes that computation.

    Args:
        base_gain: Minimum gain (at modulation=0)
        modulation_range: Range of modulation (max - min)
        modulation_value: Modulation factor [0.0, 1.0]

    Returns:
        Modulated gain = base_gain + modulation_range * modulation_value

    Examples:
        >>> # DG→CA3 gating during encoding
        >>> from thalia.regulation.oscillator_constants import (
        ...     DG_CA3_GATE_MIN, DG_CA3_GATE_RANGE
        ... )
        >>> enc_mod = 1.0  # Full encoding
        >>> gate = compute_oscillator_modulated_gain(
        ...     DG_CA3_GATE_MIN, DG_CA3_GATE_RANGE, enc_mod
        ... )
        >>> assert gate == 1.0  # Full strength during encoding

        >>> # During retrieval
        >>> enc_mod = 0.0
        >>> gate = compute_oscillator_modulated_gain(
        ...     DG_CA3_GATE_MIN, DG_CA3_GATE_RANGE, enc_mod
        ... )
        >>> assert gate == 0.1  # Minimal strength during retrieval
    """
    return base_gain + modulation_range * modulation_value


def compute_learning_rate_modulation(
    base_lr: float,
    gamma_modulation: float,
    scale: float = 0.5,
) -> float:
    """Compute gamma-modulated learning rate.

    Learning rate is modulated by gamma phase to concentrate plasticity
    during optimal processing windows.

    Args:
        base_lr: Base learning rate
        gamma_modulation: Gamma phase modulation [0.0, 1.0]
        scale: Modulation scale (default: 0.5 for 50-100% range)

    Returns:
        Modulated learning rate = base_lr * (scale + scale * gamma_mod)
        Range: [base_lr * scale, base_lr * (2 * scale)]
        With default scale=0.5: [50%, 100%] of base_lr

    Biological Basis:
        Plasticity is enhanced during gamma peaks when neurons are most
        responsive to input, implementing a form of attentional gating.

    Examples:
        >>> # At gamma peak (full learning)
        >>> lr = compute_learning_rate_modulation(0.01, gamma_modulation=1.0)
        >>> assert lr == 0.01  # 100% of base

        >>> # At gamma trough (reduced learning)
        >>> lr = compute_learning_rate_modulation(0.01, gamma_modulation=0.0)
        >>> assert lr == 0.005  # 50% of base
    """
    return base_lr * (scale + scale * gamma_modulation)


__all__ = [
    "compute_theta_encoding_retrieval",
    "compute_ach_recurrent_suppression",
    "compute_gamma_phase_gate",
    "compute_theta_gamma_coupling_gate",
    "compute_oscillator_modulated_gain",
    "compute_learning_rate_modulation",
]
