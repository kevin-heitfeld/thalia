"""Fused C++ kernel for batched STP (Short-Term Plasticity) step.

Loaded lazily via torch.utils.cpp_extension.load() following the same
pattern as conductance_lif_fused.py.  Falls back gracefully when the C++
extension cannot be compiled.
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
_stp_ext: Optional[ModuleType] = None
_stp_ext_loaded: bool = False


def _get_stp_ext() -> Optional[ModuleType]:
    """Lazy-load the compiled STP C++ extension.  Returns None on failure."""
    global _stp_ext, _stp_ext_loaded
    if _stp_ext_loaded:
        return _stp_ext
    _stp_ext_loaded = True
    try:
        venv_scripts = str(Path(sys.executable).parent)
        if venv_scripts not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

        src = Path(__file__).parent / "stp_kernel.cpp"
        _src_hash = hashlib.md5(src.read_bytes()).hexdigest()[:8]
        build = Path(__file__).parent / "_build" / f"_stp_{_src_hash}"
        build.mkdir(parents=True, exist_ok=True)

        # Windows SDK header / lib fixup (same logic as conductance_lif_fused.py)
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

        _stp_ext = _cpp_load(
            name="stp_cpu",
            sources=[str(src)],
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
            build_directory=str(build),
            verbose=False,
        )
        logger.debug("STP C++ kernel loaded")
    except Exception as exc:
        logger.warning("Could not load STP C++ kernel, falling back to Python: %s", exc)
        _stp_ext = None
    return _stp_ext


def is_available() -> bool:
    """Check if the C++ STP kernel is available."""
    return _get_stp_ext() is not None


def stp_step(
    u: torch.Tensor,
    x: torch.Tensor,
    U: torch.Tensor,
    decay_d: torch.Tensor,
    decay_f: torch.Tensor,
    recovery_d: torch.Tensor,
    recovery_f: torch.Tensor,
    pre_spikes: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Call the fused C++ STP step.  Returns efficacy [N].

    All state tensors (u, x) are modified in-place.

    Raises:
        RuntimeError: If the C++ kernel is not available.
    """
    ext = _get_stp_ext()
    if ext is None:
        raise RuntimeError("STP C++ kernel not available")
    return ext.stp_step_cpp(u, x, U, decay_d, decay_f, recovery_d, recovery_f, pre_spikes, N)
