"""Save and restore full brain state during training.

``brain.state_dict()`` captures nn.Parameters and registered buffers (synaptic
weights, learning traces, neuromodulator receptor state, axonal delay buffers)
but MISSES dynamic state stored on plain Python objects:

- ``brain._neuron_batch`` (ConductanceLIFBatch): membrane voltages, conductances,
  adaptation, refractory counters, OU noise, T-current gate state
- ``brain._stp_batch`` (STPBatch): facilitation (u) and depression (x) variables

This module captures both, along with training metadata and RNG state.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from thalia.brain.brain import Brain

# Dynamic state tensor names on ConductanceLIFBatch
_NEURON_BATCH_STATE_KEYS = [
    "V_soma", "g_E", "g_I", "g_nmda", "g_GABA_B",
    "g_adapt", "ou_noise", "refractory", "h_gate", "h_T",
]

# Dynamic state tensor names on STPBatch
_STP_BATCH_STATE_KEYS = ["u", "x", "pre_spikes"]


def save_checkpoint(
    brain: Brain,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a complete brain checkpoint to *path*.

    Args:
        brain: Brain instance to snapshot.
        path: File path for the checkpoint (``.pt`` recommended).
        metadata: Optional dict of training info (trial index, results, etc.).

    Returns:
        The resolved Path the checkpoint was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. nn.Module state (weights, learning buffers, neuromodulator state, ...)
    module_state = brain.state_dict()

    # 2. Neuron batch dynamic state
    nb = brain._neuron_batch
    neuron_state = {k: getattr(nb, k).clone() for k in _NEURON_BATCH_STATE_KEYS}

    # 3. STP batch dynamic state
    sb = brain._stp_batch
    stp_state = {k: getattr(sb, k).clone() for k in _STP_BATCH_STATE_KEYS}

    # 4. RNG state for reproducibility
    rng_state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }

    checkpoint = {
        "module_state": module_state,
        "neuron_state": neuron_state,
        "stp_state": stp_state,
        "rng_state": rng_state,
        "metadata": metadata or {},
    }

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    brain: Brain,
    path: str | Path,
) -> dict[str, Any]:
    """Restore brain state from a checkpoint file.

    Args:
        brain: Brain instance to restore into.  Must have identical architecture
            to the one that was saved (same regions, populations, connections).
        path: Path to the checkpoint file.

    Returns:
        The metadata dict that was stored in the checkpoint.
    """
    checkpoint = torch.load(path, weights_only=False)

    # 1. Restore nn.Module state
    brain.load_state_dict(checkpoint["module_state"])

    # 2. Restore neuron batch dynamic state
    nb = brain._neuron_batch
    for k, v in checkpoint["neuron_state"].items():
        getattr(nb, k).copy_(v)

    # 3. Restore STP batch dynamic state
    sb = brain._stp_batch
    for k, v in checkpoint["stp_state"].items():
        getattr(sb, k).copy_(v)

    # 4. Restore RNG state
    rng = checkpoint["rng_state"]
    torch.set_rng_state(rng["torch"])
    random.setstate(rng["python"])

    return checkpoint["metadata"]
