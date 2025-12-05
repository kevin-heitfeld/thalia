"""
Learning rules: STDP, homeostatic mechanisms, and reward-modulated learning.
"""

from thalia.learning.bcm import (
    BCMRule,
    BCMConfig,
)
from thalia.learning.unified_homeostasis import (
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
    StriatumHomeostasis,
)

__all__ = [
    # BCM (Bienenstock-Cooper-Munro)
    "BCMRule",
    "BCMConfig",
    # Unified Homeostasis (constraint-based)
    "UnifiedHomeostasis",
    "UnifiedHomeostasisConfig",
    "StriatumHomeostasis",
]
