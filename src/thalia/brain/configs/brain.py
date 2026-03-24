"""BrainConfig: Global configuration for entire brain instance."""

from __future__ import annotations

from dataclasses import dataclass

from thalia import GlobalConfig


@dataclass
class BrainConfig:
    """Complete brain configuration for a single brain instance.

    Each brain instance is fully self-contained with
    its own dt, oscillator frequencies, etc.

    This enables:
    - Different temporal resolutions per brain (dt_ms)
    - Different oscillator frequencies per brain
    """

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
    # PARALLELISM
    # =========================================================================
    execute_regions_in_parallel: bool = False
    """Execute brain regions in parallel using a thread pool during Brain.forward().

    Currently OFF by default: most Phase 1 work is GIL-held Python (tiny tensor
    dispatch, dict lookups, attribute access).  Threading overhead exceeds the
    benefit from overlapping the small fraction of GIL-free C++ kernel calls.
    Re-enable once cortical inhibitory networks are fused into C++ (~0.8 s of
    torch.mv moved out of Python dispatch → meaningful GIL-free overlap).
    """

    parallel_regions_max_workers: int = 4
    """Maximum number of worker threads for parallel region execution."""

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Timing validation
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms must be positive, got {self.dt_ms}")
        # Parallelism validation
        if self.parallel_regions_max_workers < 1:
            raise ValueError(f"parallel_regions_max_workers must be at least 1, got {self.parallel_regions_max_workers}")
