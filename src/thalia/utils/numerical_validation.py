"""Utility functions for numerical validation."""

from __future__ import annotations

import math
from typing import Optional


# Module-level flag for numerical validation (mutable by design)
# ruff: noqa: N816 (allow lowercase module-level variable)
_enable_numerical_validation = True
"""Global flag to enable/disable numerical validation.

Set to False only for production/benchmarking after thorough testing.
"""


def set_numerical_validation(enabled: bool) -> None:
    """Enable or disable numerical validation globally.

    Args:
        enabled: True to enable validation, False to disable
    """
    global _enable_numerical_validation  # noqa: PLW0603
    _enable_numerical_validation = enabled


def validate_finite(
    value: float,
    name: str,
    valid_range: Optional[tuple[float, float]] = None,
) -> None:
    """Validate that a numerical value is finite and optionally in range.

    Args:
        value: Value to validate
        name: Parameter name for error message
        valid_range: Optional (min, max) tuple for range checking

    Raises:
        ValueError: If value is NaN, Inf, or out of range
    """
    if not _enable_numerical_validation:
        return

    # Check for NaN
    if math.isnan(value):
        raise ValueError(
            f"Invalid {name}: NaN is not a valid value. "
            f"This usually indicates a numerical instability upstream."
        )

    # Check for Inf
    if math.isinf(value):
        raise ValueError(
            f"Invalid {name}: Inf is not a valid value. "
            f"This usually indicates a numerical overflow upstream."
        )

    # Optional range check
    if valid_range is not None:
        min_val, max_val = valid_range
        if value < min_val or value > max_val:
            raise ValueError(
                f"Invalid {name}: {value:.4f} is outside valid range "
                f"[{min_val}, {max_val}]."
            )
