"""Fused C++ kernel for TwoCompartmentLIF neuron step.

Provides a drop-in replacement for the inner loop of TwoCompartmentLIF.forward(),
fusing ~60 individual PyTorch tensor operations into a single parallelised C++ loop.

Loaded lazily via torch.utils.cpp_extension.load() with the same pattern as
conductance_lif_fused.py and rng.py.
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
_tclif_ext: Optional[ModuleType] = None
_tclif_ext_loaded: bool = False


def _get_tclif_ext() -> Optional[ModuleType]:
    """Lazy-load the compiled TwoCompartmentLIF C++ extension. Returns None on failure."""
    global _tclif_ext, _tclif_ext_loaded
    if _tclif_ext_loaded:
        return _tclif_ext
    _tclif_ext_loaded = True
    try:
        # Ensure the venv Scripts directory is on PATH so torch can find ninja.exe
        venv_scripts = str(Path(sys.executable).parent)
        if venv_scripts not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

        src = Path(__file__).parent / "two_compartment_lif_kernel.cpp"
        _src_hash = hashlib.md5(src.read_bytes()).hexdigest()[:8]
        build = Path(__file__).parent / "_build" / f"_tclif_{_src_hash}"
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

        _tclif_ext = _cpp_load(
            name="two_compartment_lif_cpu",
            sources=[str(src)],
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
            build_directory=str(build),
            verbose=False,
        )
        logger.debug("TwoCompartmentLIF C++ kernel loaded")
    except Exception as exc:
        logger.warning("Could not load TwoCompartmentLIF C++ kernel, falling back to Python: %s", exc)
        _tclif_ext = None
    return _tclif_ext


def is_available() -> bool:
    """Check if the C++ TwoCompartmentLIF kernel is available."""
    return _get_tclif_ext() is not None


def two_compartment_lif_step(
    # Somatic state (in/out)
    V_soma: torch.Tensor,
    g_E_basal: torch.Tensor,
    g_I_basal: torch.Tensor,
    g_GABA_B_basal: torch.Tensor,
    g_nmda_basal: torch.Tensor,
    g_adapt: torch.Tensor,
    # Dendritic state (in/out)
    V_dend: torch.Tensor,
    g_E_apical: torch.Tensor,
    g_I_apical: torch.Tensor,
    g_nmda_apical: torch.Tensor,
    g_Ca: torch.Tensor,
    g_plateau: torch.Tensor,
    # Noise state
    ou_noise: torch.Tensor,
    refractory: torch.Tensor,
    # Basal synaptic inputs
    g_ampa_basal_in: torch.Tensor,
    g_nmda_basal_in: torch.Tensor,
    g_gaba_a_basal_in: torch.Tensor,
    g_gaba_b_basal_in: torch.Tensor,
    # Apical synaptic inputs
    g_ampa_apical_in: torch.Tensor,
    g_nmda_apical_in: torch.Tensor,
    g_gaba_a_apical_in: torch.Tensor,
    # Per-neuron decay factors
    g_E_decay: torch.Tensor,
    g_I_decay: torch.Tensor,
    g_nmda_decay: torch.Tensor,
    g_GABA_B_decay: torch.Tensor,
    g_Ca_decay: torch.Tensor,
    g_plateau_decay: torch.Tensor,
    adapt_decay: torch.Tensor,
    V_soma_decay: torch.Tensor,
    V_dend_decay: torch.Tensor,
    # Per-neuron parameters
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
    E_Ca: torch.Tensor,
    nmda_a: torch.Tensor,
    nmda_b: torch.Tensor,
    g_nmda_max: torch.Tensor,
    dt_ms: float,
    # Two-compartment coupling parameters
    g_c: float,
    g_L_dend: float,
    bap_amplitude: float,
    theta_Ca: float,
    g_Ca_spike: float,
    # NMDA plateau
    enable_nmda_plateau: bool,
    nmda_plateau_threshold: float,
    v_dend_plateau_threshold: float,
    g_nmda_plateau: float,
    # Noise
    enable_noise: bool,
    neuron_seeds: torch.Tensor,
    rng_timestep: int,
    ou_decay: torch.Tensor,
    ou_std: torch.Tensor,
    # Gap junctions
    has_gap_junctions: bool,
    g_gap_input: torch.Tensor,
    E_gap_reversal: torch.Tensor,
) -> torch.Tensor:
    """Call the fused C++ TwoCompartmentLIF step. Returns bool spike tensor [N].

    All state tensors are modified in-place.
    """
    ext = _get_tclif_ext()
    if ext is None:
        raise RuntimeError("TwoCompartmentLIF C++ kernel not available")
    return ext.two_compartment_lif_step_cpp(
        V_soma, g_E_basal, g_I_basal, g_GABA_B_basal, g_nmda_basal, g_adapt,
        V_dend, g_E_apical, g_I_apical, g_nmda_apical, g_Ca, g_plateau,
        ou_noise, refractory,
        g_ampa_basal_in, g_nmda_basal_in, g_gaba_a_basal_in, g_gaba_b_basal_in,
        g_ampa_apical_in, g_nmda_apical_in, g_gaba_a_apical_in,
        g_E_decay, g_I_decay, g_nmda_decay, g_GABA_B_decay, g_Ca_decay,
        g_plateau_decay,
        adapt_decay, V_soma_decay, V_dend_decay,
        g_L, g_L_scale, v_threshold, adapt_increment, tau_ref_per_neuron,
        v_reset, E_E, E_I, E_nmda, E_GABA_B, E_adapt, E_Ca,
        nmda_a, nmda_b, g_nmda_max, dt_ms,
        g_c, g_L_dend, bap_amplitude, theta_Ca, g_Ca_spike,
        enable_nmda_plateau, nmda_plateau_threshold, v_dend_plateau_threshold, g_nmda_plateau,
        enable_noise, neuron_seeds, rng_timestep, ou_decay, ou_std,
        has_gap_junctions, g_gap_input, E_gap_reversal,
    )
