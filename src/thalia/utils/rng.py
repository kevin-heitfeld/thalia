"""Random number generation utilities for Thalia, including Philox-based RNG for reproducibility and GPU efficiency."""

import hashlib
import logging
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.utils.cpp_extension import load as _cpp_load

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native C++ kernel (compiled once, cached in src/thalia/utils/_philox_build/)
# ---------------------------------------------------------------------------
_philox_ext: Optional[ModuleType | str] = None
_philox_ext_loaded: bool = False


def _get_philox_ext() -> Optional[ModuleType | str]:
    """Lazy-load the compiled Philox C++ extension.  Returns None on failure."""
    global _philox_ext, _philox_ext_loaded
    if _philox_ext_loaded:
        return _philox_ext
    _philox_ext_loaded = True
    try:
        # Ensure the venv Scripts/bin directory is on PATH so torch can find ninja.exe
        venv_scripts = str(Path(sys.executable).parent)
        if venv_scripts not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

        src = Path(__file__).parent / "philox_cpu_kernel.cpp"
        _src_hash = hashlib.md5(src.read_bytes()).hexdigest()[:8]
        build = Path(__file__).parent / "_build" / f"_philox_{_src_hash}"
        build.mkdir(parents=True, exist_ok=True)

        # On Windows, vcvarsall may set INCLUDE/LIB with ucrt paths but omit
        # um/shared/winrt (headers) and um/x64 (libs like kernel32.lib).
        # Detect the SDK base from the ucrt entry and patch both.
        extra_include_paths: list[str] = []
        extra_ldflags: list[str] = []
        if os.name == "nt":
            include_env = os.environ.get("INCLUDE", "")
            lib_env = os.environ.get("LIB", "")
            sdk_ver_dir: Path | None = None
            for entry in include_env.split(os.pathsep):
                p = Path(entry)
                if p.name.lower() == "ucrt" and p.parent.exists():
                    sdk_ver_dir = p.parent  # e.g. ...Windows Kits\10\include\<version>
                    break
            if sdk_ver_dir is not None:
                # Include: sdk_ver_dir/um, sdk_ver_dir/shared, sdk_ver_dir/winrt
                for subdir in ("um", "shared", "winrt"):
                    candidate = sdk_ver_dir / subdir
                    if candidate.is_dir():
                        s = str(candidate)
                        if s not in include_env:
                            extra_include_paths.append(s)
                # Lib: structure is ...Windows Kits\10\lib\<version>\um\x64
                # sdk_ver_dir is under include\<version>, so sdk root is 2 levels up
                sdk_root = sdk_ver_dir.parent.parent
                sdk_version = sdk_ver_dir.name
                for arch in ("x64", "x86"):
                    lib_candidate = sdk_root / "lib" / sdk_version / "um" / arch
                    if lib_candidate.is_dir():
                        s = str(lib_candidate)
                        if s not in lib_env:
                            extra_ldflags.append(f"/LIBPATH:{s}")
                        break

        _philox_ext = _cpp_load(
            name="philox_cpu",
            sources=[str(src)],
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
            build_directory=str(build),
            verbose=False,
        )
        logger.debug("Philox C++ kernel loaded")
    except Exception as exc:
        logger.warning("Could not load Philox C++ kernel, falling back to TorchScript: %s", exc)
        _philox_ext = None
    return _philox_ext


# ---------------------------------------------------------------------------
# TorchScript fallback (used when C++ kernel is unavailable)
# ---------------------------------------------------------------------------

@torch.jit.script
def _philox_uniform_jit(counters: torch.Tensor) -> torch.Tensor:
    W0, W1 = 0x9E3779B9, 0xBB67AE85
    x = counters.clone().to(torch.int64)
    for r in range(10):
        lo = x & 0xffffffff
        hi = (x >> 32) & 0xffffffff
        lo = (lo * W0) & 0xffffffff
        hi = (hi * W1) & 0xffffffff
        x = ((hi << 32) | lo) ^ (W0 * (r + 1))
    u = ((x & 0xffffffff) + 1).float() / (2**32 + 2)
    return u


@torch.jit.script
def _philox_gaussian_jit(counters: torch.Tensor) -> torch.Tensor:
    u1 = _philox_uniform_jit(counters)
    u2 = _philox_uniform_jit(counters + 1)
    # Box-Muller transform to get Gaussian(0,1) from uniform(0,1)
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# Public API — dispatches to C kernel when available, TorchScript otherwise
# ---------------------------------------------------------------------------

def philox_uniform(counters: torch.Tensor) -> torch.Tensor:
    """Philox 2x32-10 → uniform (0, 1).  Uses native C++ kernel when available."""
    ext = _get_philox_ext()
    if ext is not None and counters.is_cpu:
        return ext.philox_uniform_cpp(counters)
    return _philox_uniform_jit(counters)


def philox_gaussian(counters: torch.Tensor) -> torch.Tensor:
    """Per-neuron independent Gaussian(0,1) noise via Philox + Box-Muller.  Uses native C++ kernel when available."""
    ext = _get_philox_ext()
    if ext is not None and counters.is_cpu:
        return ext.philox_gaussian_cpp(counters)
    return _philox_gaussian_jit(counters)
