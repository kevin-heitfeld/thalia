"""Neuromodulation utilities for Thalia."""

from __future__ import annotations

import torch


def compute_ach_recurrent_suppression(
    ach_level: float,
    ach_threshold: float = 0.5,
) -> float:
    """Compute ACh-mediated suppression of recurrent connections."""
    # NOTE: These assertions ensure that the ACh level and threshold are within biologically plausible ranges (0 to 1).
    # The function implements a simple linear suppression mechanism where ACh levels below the threshold do not suppress
    # recurrent connections, while levels above the threshold increasingly suppress them up to a maximum of 70% suppression
    # at ACh level 1.0. This reflects the role of acetylcholine in modulating cortical circuits, where it can suppress
    # recurrent activity to enhance feedforward processing and reduce interference from previous inputs, particularly
    # during attention-demanding tasks.
    assert 0.0 <= ach_level <= 1.0, "ACh level must be between 0 and 1"
    assert 0.0 <= ach_threshold <= 1.0, "ACh threshold must be between 0 and 1"

    if ach_level <= ach_threshold:
        return 1.0

    # Linear suppression above threshold
    suppression_factor = (ach_level - ach_threshold) / (1.0 - ach_threshold)
    return 1.0 - 0.7 * suppression_factor


def compute_da_gain(
    da_level: float | torch.Tensor,
    da_factor: float,
    da_baseline: float = 0.5,
) -> float | torch.Tensor:
    """Compute dopamine modulation factor from DA level."""
    # NOTE: These assertions ensure that the dopamine level is within a biologically plausible range (0 to 1),
    # and that the baseline is non-negative. The linear modulation allows for both increases and decreases
    # in gain depending on the sign of da_factor, with the modulation centered around the specified baseline level.
    # This reflects the complex role of dopamine in modulating neural excitability, where it can enhance
    # or suppress activity based on receptor type and current DA levels.
    if isinstance(da_level, torch.Tensor):
        assert torch.all((0.0 <= da_level) & (da_level <= 1.0)), "DA level tensor must have values between 0 and 1"
    else:
        assert 0.0 <= da_level <= 1.0, "DA level must be between 0 and 1"
    assert da_baseline >= 0, "DA baseline must be non-negative"

    # Linear modulation around baseline, scaled by da_factor
    da_gain = 1.0 + da_factor * (da_level - da_baseline)  # Modulation centered around baseline

    # Ensure gain is non-negative
    if isinstance(da_gain, torch.Tensor):
        da_gain = da_gain.clamp(min=0.0)
    else:
        da_gain = max(0.0, da_gain)

    return da_gain


def compute_ne_gain(
    ne_level: float,
    ne_gain_min: float = 1.0,
    ne_gain_max: float = 1.5,
) -> float:
    """Compute norepinephrine gain modulation from NE level."""
    # NOTE: These assertions ensure that the gain modulation is biologically plausible,
    # with NE levels between 0 and 1, and gain values that are positive and within a reasonable range.
    # The linear interpolation allows for a smooth increase in gain as NE levels rise,
    # reflecting the modulatory role of norepinephrine in enhancing neural responsiveness
    # under conditions of arousal or attention.
    assert 0.0 <= ne_level <= 1.0, "NE level must be between 0 and 1"
    assert ne_gain_min > 0, "NE gain minimum must be positive"
    assert ne_gain_max >= ne_gain_min, "NE gain maximum must be >= minimum"

    # Linear gain modulation from 1.0 to ne_gain_max as NE level increases
    return ne_gain_min + (ne_gain_max - ne_gain_min) * ne_level
