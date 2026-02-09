"""Utility functions for oscillation modulation in Thalia's brain model."""

from __future__ import annotations


def compute_ach_recurrent_suppression(
    ach_level: float,
    ach_threshold: float = 0.5,
) -> float:
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
    """
    if ach_level <= ach_threshold:
        return 1.0

    # Linear suppression above threshold
    suppression_factor = (ach_level - ach_threshold) / (1.0 - ach_threshold)
    return 1.0 - 0.7 * suppression_factor


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
    """
    return base_lr * (scale + scale * gamma_modulation)
