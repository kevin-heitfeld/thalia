"""Fused C++ kernel for ConductanceLIF neuron step.

Provides a drop-in replacement for the inner loop of ConductanceLIF.forward(),
fusing ~40 individual PyTorch tensor operations into a single parallelised C++ loop.

Loaded lazily via torch.utils.cpp_extension.load() with the same pattern as rng.py.
Falls back gracefully to the Python implementation if the C++ extension cannot be compiled.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.utils.cpp_extension import load as _cpp_load

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native C++ kernel (compiled once, cached)
# ---------------------------------------------------------------------------
_clif_ext: Optional[ModuleType] = None
_clif_ext_loaded: bool = False


def _get_clif_ext() -> Optional[ModuleType]:
    """Lazy-load the compiled ConductanceLIF C++ extension. Returns None on failure."""
    global _clif_ext, _clif_ext_loaded
    if _clif_ext_loaded:
        return _clif_ext
    _clif_ext_loaded = True
    try:
        # Ensure the venv Scripts directory is on PATH so torch can find ninja.exe
        venv_scripts = str(Path(sys.executable).parent)
        if venv_scripts not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

        src = Path(__file__).parent / "conductance_lif_kernel.cpp"
        _src_hash = hashlib.md5(src.read_bytes()).hexdigest()[:8]
        build = Path(__file__).parent / "_build" / f"_clif_{_src_hash}"
        build.mkdir(parents=True, exist_ok=True)

        # On Windows, vcvarsall may set INCLUDE/LIB with ucrt paths but omit
        # um/shared/winrt (headers) and um/x64 (libs like kernel32.lib).
        extra_include_paths: list[str] = []
        extra_ldflags: list[str] = []
        if os.name == "nt":
            include_env = os.environ.get("INCLUDE", "")
            lib_env = os.environ.get("LIB", "")
            sdk_ver_dir: Path | None = None
            for entry in include_env.split(os.pathsep):
                p = Path(entry)
                if p.name.lower() == "ucrt" and p.parent.exists():
                    sdk_ver_dir = p.parent
                    break
            if sdk_ver_dir is not None:
                for subdir in ("um", "shared", "winrt"):
                    candidate = sdk_ver_dir / subdir
                    if candidate.is_dir():
                        s = str(candidate)
                        if s not in include_env:
                            extra_include_paths.append(s)
                sdk_root = sdk_ver_dir.parent.parent
                sdk_version = sdk_ver_dir.name
                for arch in ("x64", "x86"):
                    lib_candidate = sdk_root / "lib" / sdk_version / "um" / arch
                    if lib_candidate.is_dir():
                        s = str(lib_candidate)
                        if s not in lib_env:
                            extra_ldflags.append(f"/LIBPATH:{s}")
                        break

        _clif_ext = _cpp_load(
            name="conductance_lif_cpu",
            sources=[str(src)],
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
            build_directory=str(build),
            verbose=False,
        )
        logger.debug("ConductanceLIF C++ kernel loaded")
    except Exception as exc:
        logger.warning("Could not load ConductanceLIF C++ kernel, falling back to Python: %s", exc)
        _clif_ext = None
    return _clif_ext


def is_available() -> bool:
    """Check if the C++ ConductanceLIF kernel is available."""
    return _get_clif_ext() is not None


def conductance_lif_step(
    # State tensors (in/out)
    V_soma: torch.Tensor,
    g_E: torch.Tensor,
    g_I: torch.Tensor,
    g_nmda: torch.Tensor,
    g_GABA_B: torch.Tensor,
    g_adapt: torch.Tensor,
    ou_noise: torch.Tensor,
    refractory: torch.Tensor,
    # Synaptic inputs
    g_ampa_input: torch.Tensor,
    g_nmda_input: torch.Tensor,
    g_gaba_a_input: torch.Tensor,
    g_gaba_b_input: torch.Tensor,
    # Per-neuron parameters
    g_E_decay: torch.Tensor,
    g_I_decay: torch.Tensor,
    g_nmda_decay: torch.Tensor,
    g_GABA_B_decay: torch.Tensor,
    adapt_decay: torch.Tensor,
    V_soma_decay: torch.Tensor,
    g_L: torch.Tensor,
    g_L_scale: torch.Tensor,
    v_threshold: torch.Tensor,
    adapt_increment: torch.Tensor,
    tau_ref_per_neuron: torch.Tensor,
    # Scalar-or-per-neuron constants
    v_reset: torch.Tensor,
    E_E: torch.Tensor,
    E_I: torch.Tensor,
    E_nmda: torch.Tensor,
    E_GABA_B: torch.Tensor,
    E_adapt: torch.Tensor,
    dendrite_coupling_scale: torch.Tensor,
    nmda_a: torch.Tensor,
    nmda_b: torch.Tensor,
    g_nmda_max: torch.Tensor,
    dt_ms: float,
    # Noise parameters
    enable_noise: bool,
    neuron_seeds: torch.Tensor,
    rng_timestep: int,
    ou_decay: torch.Tensor,
    ou_std: torch.Tensor,
    # Gap junctions
    has_gap_junctions: bool,
    g_gap_input: torch.Tensor,
    E_gap_reversal: torch.Tensor,
    # T-channels
    enable_t_channels: bool,
    h_T: torch.Tensor,
    h_T_decay: torch.Tensor,
    g_T: torch.Tensor,
    E_Ca: torch.Tensor,
    V_half_h_T: torch.Tensor,
    k_h_T: torch.Tensor,
    # I_h (HCN)
    enable_ih: bool,
    h_gate: torch.Tensor,
    h_decay: torch.Tensor,
    g_h_max: torch.Tensor,
    E_h: torch.Tensor,
    V_half_h: torch.Tensor,
    k_h: torch.Tensor,
) -> torch.Tensor:
    """Call the fused C++ ConductanceLIF step. Returns bool spike tensor [N].

    All state tensors are modified in-place. Returns None if C++ kernel
    is not available (caller should fall back to Python implementation).
    """
    ext = _get_clif_ext()
    if ext is None:
        raise RuntimeError("ConductanceLIF C++ kernel not available")
    return ext.conductance_lif_step_cpp(
        V_soma, g_E, g_I, g_nmda, g_GABA_B, g_adapt, ou_noise, refractory,
        g_ampa_input, g_nmda_input, g_gaba_a_input, g_gaba_b_input,
        g_E_decay, g_I_decay, g_nmda_decay, g_GABA_B_decay,
        adapt_decay, V_soma_decay, g_L, g_L_scale, v_threshold,
        adapt_increment, tau_ref_per_neuron,
        v_reset, E_E, E_I, E_nmda, E_GABA_B, E_adapt,
        dendrite_coupling_scale, nmda_a, nmda_b, g_nmda_max, dt_ms,
        enable_noise, neuron_seeds, rng_timestep, ou_decay, ou_std,
        has_gap_junctions, g_gap_input, E_gap_reversal,
        enable_t_channels, h_T, h_T_decay, g_T, E_Ca, V_half_h_T, k_h_T,
        enable_ih, h_gate, h_decay, g_h_max, E_h, V_half_h, k_h,
    )
