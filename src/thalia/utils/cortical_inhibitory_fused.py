"""Fused C++ kernel for cortical inhibitory network synaptic integration.

Loaded lazily via torch.utils.cpp_extension.load() following the same
pattern as stp_fused.py.  Falls back gracefully when the C++ extension
cannot be compiled.
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
_ci_ext: Optional[ModuleType] = None
_ci_ext_loaded: bool = False


def _get_ci_ext() -> Optional[ModuleType]:
    """Lazy-load the compiled cortical inhibitory C++ extension."""
    global _ci_ext, _ci_ext_loaded
    if _ci_ext_loaded:
        return _ci_ext
    _ci_ext_loaded = True
    try:
        venv_scripts = str(Path(sys.executable).parent)
        if venv_scripts not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

        src = Path(__file__).parent / "cortical_inhibitory_kernel.cpp"
        _src_hash = hashlib.md5(src.read_bytes()).hexdigest()[:8]
        build = Path(__file__).parent / "_build" / f"_ci_{_src_hash}"
        build.mkdir(parents=True, exist_ok=True)

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

        _ci_ext = _cpp_load(
            name="cortical_inhibitory_cpu",
            sources=[str(src)],
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
            build_directory=str(build),
            verbose=False,
        )
        logger.debug("Cortical inhibitory C++ kernel loaded")
    except Exception as exc:
        logger.warning(
            "Could not load cortical inhibitory C++ kernel, falling back to Python: %s", exc
        )
        _ci_ext = None
    return _ci_ext


def is_available() -> bool:
    """Check if the C++ cortical inhibitory kernel is available."""
    return _get_ci_ext() is not None


def cortical_inhibitory_step(
    pyr_f: torch.Tensor,
    pv_f: torch.Tensor,
    sst_f: torch.Tensor,
    vip_f: torch.Tensor,
    ngc_f: torch.Tensor,
    weights: list[torch.Tensor],
    efficacies: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Fused synaptic integration for one cortical inhibitory network layer.

    Returns list of 13 tensors (see cortical_inhibitory_kernel.cpp for ordering).
    Expects 16 weight matrices and 15 STP efficacy vectors.
    Raises RuntimeError if the C++ extension is not available.
    """
    ext = _get_ci_ext()
    if ext is None:
        raise RuntimeError("Cortical inhibitory C++ kernel not available")
    return ext.cortical_inhibitory_step_cpp(
        pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies
    )
