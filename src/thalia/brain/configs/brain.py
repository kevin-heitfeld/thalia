"""BrainConfig: Global configuration for entire brain instance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from thalia import GlobalConfig


@dataclass
class BrainConfig:
    """Complete brain configuration for a single brain instance.

    Each brain instance is fully self-contained with
    its own device, dt, oscillator frequencies, etc.

    This enables:
    - Multiple independent brains with different devices (GPU vs CPU)
    - Different temporal resolutions per brain (dt_ms)
    - Different oscillator frequencies per brain
    """

    device: str = "cpu"  # Device to run on: 'cpu', 'cuda', 'cuda:0', etc.
    seed: Optional[int] = None  # Random seed for reproducibility. None = no seeding.

    # =========================================================================
    # TIMING
    # =========================================================================
    dt_ms: float = GlobalConfig.DEFAULT_DT_MS
    """Simulation timestep in milliseconds. Smaller = more precise but slower.

    **CRITICAL**: This is the single source of truth for temporal resolution.
    All decay factors, and delays derive from this value.

    Can be changed adaptively during simulation via brain.set_timestep(new_dt).
    Typical values:
    - 1.0ms: Standard biological timescale (Brian2, most research)
    - 0.1ms: High-resolution for detailed temporal dynamics
    - 10ms: Fast replay for memory consolidation
    """

    # =========================================================================
    # VALIDATION AND SUMMARY
    # =========================================================================

    def summary(self) -> str:
        """Return formatted summary of brain configuration."""
        lines = [
            "=== Brain Configuration ===",
            f"  Device: {self.device}",
            f"  Data type: {self.dtype}",
            f"  Timestep: {self.dt_ms} ms",
            "",
        ]
        return "\n".join(lines)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        # Timing validation
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms must be positive, got {self.dt_ms}")
