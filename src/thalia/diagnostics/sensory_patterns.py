"""Sensory input pattern generators for diagnostic runs and training loops.

Each pattern is a callable ``(brain: Brain, t: int) -> Optional[SynapticInput]``
that produces a :data:`~thalia.typing.SynapticInput` dict (or ``None``) suitable
for passing directly to :meth:`~thalia.brain.Brain.forward`.

Use :func:`make_sensory_input` to dispatch by name, or consult
:data:`SENSORY_PATTERNS` for the full list of available pattern names.

Usage in a training loop::

    from thalia.diagnostics import make_sensory_input

    for t in range(n_steps):
        inputs = make_sensory_input(brain, t, pattern="random")
        outputs = brain.forward(inputs)
        recorder.record(t, outputs)
"""

from __future__ import annotations

import math
import threading
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Optional

import torch

from thalia.brain.regions.population_names import ThalamusPopulation
from thalia.typing import SynapseId, SynapticInput

if TYPE_CHECKING:
    from thalia.brain import Brain


# Emitted at most once per process to avoid per-timestep noise.
# _WARN_LOCK protects the flag to prevent races when two Brain instances run
# concurrently in different threads.
_WARN_LOCK: threading.Lock = threading.Lock()
_WARNED_NO_EXTERNAL_INPUT: bool = False


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _get_relay_context(
    brain: "Brain",
) -> "Optional[tuple]":
    """Return ``(thalamus, relay_size, device)`` for relay spike construction.

    Returns ``None`` when no thalamus region is registered, allowing callers to
    bail early before allocating any tensors.
    """
    thalamus = brain.get_region_by_name("thalamus")
    if thalamus is None:
        return None
    return thalamus, thalamus.get_population_size(ThalamusPopulation.RELAY), brain.device


# =============================================================================
# PATTERN FUNCTIONS
# =============================================================================


def _sensory_none(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """No external input — spontaneous activity only."""
    return None


def _sensory_background(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """Low-rate (≈2 Hz) independent Poisson noise into every registered
    external-input synapse across all brain regions (E6 pattern)."""
    result: SynapticInput = {}
    device = brain.device
    for region in brain.regions.values():
        for synapse_id in region.synaptic_weights.keys():
            if synapse_id.source_region != "external":
                continue
            try:
                n = region.get_population_size(synapse_id.target_population)
            except (KeyError, ValueError, AttributeError):
                continue
            # ≈2 Hz at dt=1 ms: 0.2 % of neurons active per step
            result[synapse_id] = torch.rand(n, device=device) < 0.002
    if not result:
        global _WARNED_NO_EXTERNAL_INPUT  # pylint: disable=global-statement
        with _WARN_LOCK:
            if not _WARNED_NO_EXTERNAL_INPUT:
                _WARNED_NO_EXTERNAL_INPUT = True  # type: ignore[assignment]
                warnings.warn(
                    "background pattern: no external-input SynapseIds found beyond thalamus. "
                    "Add background_drive hooks to BrainBuilder for full multi-region coverage.",
                    stacklevel=3,
                )
        return None


def _sensory_random(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """Sparse Poisson: ~3 % of relay neurons, each with 20 % spike probability."""
    ctx = _get_relay_context(brain)
    if ctx is None:
        return None
    thalamus, relay_size, device = ctx
    n_active = max(1, int(relay_size * 0.03))
    active_idx = torch.randperm(relay_size, device=device)[:n_active]
    mask = torch.rand(n_active, device=device) < 0.20
    spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)
    spikes[active_idx] = mask
    return {SynapseId.external_sensory_to_thalamus_relay(thalamus.region_name): spikes}


def _sensory_rhythmic(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """Theta rhythm (8 Hz / 125 ms period): active for the first 20 % of each cycle."""
    ctx = _get_relay_context(brain)
    if ctx is None:
        return None
    thalamus, relay_size, device = ctx
    period_ms = 125.0
    phase = (t * brain.dt_ms % period_ms) / period_ms
    spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)
    if phase < 0.20:
        n_active = max(1, int(relay_size * 0.10))
        spikes[torch.randperm(relay_size, device=device)[:n_active]] = True
    return {SynapseId.external_sensory_to_thalamus_relay(thalamus.region_name): spikes}


def _sensory_burst(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """Single 50 ms synchronous burst (30 % of relay) at t = 100 ms.

    Models a transient dense thalamic volley (e.g. startle response).
    """
    ctx = _get_relay_context(brain)
    if ctx is None:
        return None
    thalamus, relay_size, device = ctx
    burst_start = int(100.0 / brain.dt_ms)
    burst_end   = int(150.0 / brain.dt_ms)
    spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)
    if burst_start <= t < burst_end:
        n_active = max(1, int(relay_size * 0.30))
        spikes[torch.randperm(relay_size, device=device)[:n_active]] = True
    return {SynapseId.external_sensory_to_thalamus_relay(thalamus.region_name): spikes}


def _sensory_sustained_burst(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """Repeating 50-ms-on / 450-ms-off burst cycle (30 % of relay active).

    Tests how the network recovers from repeated synchronous volleys.
    """
    ctx = _get_relay_context(brain)
    if ctx is None:
        return None
    thalamus, relay_size, device = ctx
    cycle_ms = 500.0
    burst_ms = 50.0
    phase_ms = (t * brain.dt_ms) % cycle_ms
    spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)
    if phase_ms < burst_ms:
        n_active = max(1, int(relay_size * 0.30))
        spikes[torch.randperm(relay_size, device=device)[:n_active]] = True
    return {SynapseId.external_sensory_to_thalamus_relay(thalamus.region_name): spikes}


def _sensory_gamma(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """Sinusoidal 40 Hz drive to thalamus relay.

    Carrier probability oscillates between 5 % and 15 % per step,
    testing the thalamocortical gamma relay chain and cortical gamma
    amplification.  At dt=1 ms the phase advances ~0.25 rad per step.
    """
    ctx = _get_relay_context(brain)
    if ctx is None:
        return None
    thalamus, relay_size, device = ctx
    phase = 2.0 * math.pi * 40.0 * (t * brain.dt_ms / 1000.0)
    p_spike = 0.05 + 0.10 * (1.0 + math.sin(phase)) / 2.0  # 5–15 % per step
    spikes = torch.rand(relay_size, device=device) < p_spike
    return {SynapseId.external_sensory_to_thalamus_relay(thalamus.region_name): spikes}


def _sensory_correlated_background(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """Low-rate background with a shared common-input driver (correlated noise).

    Half the relay neurons receive an independent Poisson spike at ≈2 Hz;
    the other half share a single common Poisson process at ≈4 Hz.  This
    injects synchrony into half the thalamocortical projection so that
    downstream analyses can distinguish common-input synchrony from local
    recurrent synchrony.
    """
    ctx = _get_relay_context(brain)
    if ctx is None:
        return None
    thalamus, relay_size, device = ctx
    spikes = torch.rand(relay_size, device=device) < 0.002  # ≈2 Hz independent
    if torch.rand(1, device=device).item() < 0.004:  # ≈4 Hz common event
        spikes[relay_size // 2:] = True
    return {SynapseId.external_sensory_to_thalamus_relay(thalamus.region_name): spikes}


def _sensory_ramp(brain: "Brain", t: int, n_timesteps: int) -> Optional[SynapticInput]:
    """Linearly ramping input from 0 to 30 % relay activation over *n_timesteps* steps.

    Tests rate coding, neural gain, and whether gain modulation
    (e.g. via VIP/SOM interneurons) compresses or expands the dynamic range
    as the drive increases.
    """
    if n_timesteps <= 0:
        raise ValueError("n_timesteps must be positive for ramp pattern")
    ctx = _get_relay_context(brain)
    if ctx is None:
        return None
    thalamus, relay_size, device = ctx
    p_spike = min(0.30, t / n_timesteps * 0.30)
    spikes = torch.rand(relay_size, device=device) < p_spike
    return {SynapseId.external_sensory_to_thalamus_relay(thalamus.region_name): spikes}


def _sensory_slow_wave(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """Slow-wave / burst-suppression pattern for testing up/down state dynamics.

    The cycle is **600 ms** long, split into:

    * **Up-state (0–300 ms):** 50 % of relay neurons are randomly selected and
      each fires with probability 0.04 per step (≈40 Hz mean rate within the
      active pool), producing a synchronous thalamic burst.  The 40 Hz carrier
      within the burst mimics the spindle-range activity seen during slow-wave
      sleep (Steriade et al. 1993 *J Neurosci*).
    * **Down-state (300–600 ms):** complete silence — no thalamic drive.

    This is the input regime that should trigger ``voltage_bimodality``,
    where cortical membrane potentials form a bimodal distribution with peaks
    corresponding to the up and down states (Compte et al. 2003 *J Neurophysiol*).
    It can also be used to test slow oscillation-dependent memory consolidation
    (Diekelmann & Born 2010).
    """
    ctx = _get_relay_context(brain)
    if ctx is None:
        return None
    thalamus, relay_size, device = ctx
    cycle_ms = 600.0
    up_ms    = 300.0
    phase_ms = (t * brain.dt_ms) % cycle_ms
    spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)
    if phase_ms < up_ms:
        # Select 50 % of relay neurons as the active pool for this up-state.
        # Using a deterministic half-split (first half of a per-cycle shuffle)
        # keeps the active pool consistent within a single up-state cycle.
        cycle_idx = int(t * brain.dt_ms // cycle_ms)
        gen = torch.Generator(device=device)
        gen.manual_seed(cycle_idx)
        perm = torch.randperm(relay_size, generator=gen, device=device)
        active = perm[:relay_size // 2]
        # Each active neuron fires at ≈40 Hz (0.04 probability at dt=1 ms)
        spikes[active] = torch.rand(len(active), device=device) < 0.04
    return {SynapseId.external_sensory_to_thalamus_relay(thalamus.region_name): spikes}


def _sensory_hippocampal_theta(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """Theta-frequency (8 Hz) drive to medial septum GABA and ACh populations.

    Activates the septohippocampal theta generator directly, bypassing thalamus.
    The active window covers the descending phase (0–40 % of the 125 ms cycle),
    matching the burst timing of GABAergic medial-septum neurons that pace
    hippocampal theta (Freund & Antal 1988 *Nature*).

    Returns ``None`` when no medial-septum external-input synapses are registered,
    allowing the pattern to be used safely in thalamus-only brains.
    """
    result: SynapticInput = {}
    device = brain.device
    phase = (t * brain.dt_ms % 125.0) / 125.0  # 8 Hz; 125 ms period
    active = phase < 0.40  # drive on descending phase
    for region in brain.regions.values():
        if "medial_septum" not in region.region_name.lower():
            continue
        for synapse_id in region.synaptic_weights.keys():
            if synapse_id.source_region != "external":
                continue
            pn = synapse_id.target_population.lower()
            if pn not in ("ach", "gaba"):
                continue
            try:
                n = region.get_population_size(synapse_id.target_population)
            except (KeyError, ValueError, AttributeError):
                continue
            spikes = torch.zeros(n, dtype=torch.bool, device=device)
            if active:
                n_active = max(1, int(n * 0.30))
                spikes[torch.randperm(n, device=device)[:n_active]] = True
            result[synapse_id] = spikes
    return result if result else None


def _sensory_direct_cortical(brain: "Brain", t: int) -> Optional[SynapticInput]:
    """Sparse Poisson drive (≈5 Hz) to cortical L4 external-input synapses.

    Fires 0.5 % of each L4 external-input synapse population per step,
    bypassing thalamus entirely.  Targets all regions whose names contain
    ``cortex``, ``prefrontal``, or ``entorhinal``; within those regions only
    synapses whose target population name starts with ``l4`` are driven.

    Returns ``None`` when no matching L4 external-input synapses are registered.
    """
    result: SynapticInput = {}
    device = brain.device
    cortical_tags = ("cortex", "prefrontal", "entorhinal")
    for region in brain.regions.values():
        rn = region.region_name.lower()
        if not any(tag in rn for tag in cortical_tags):
            continue
        for synapse_id in region.synaptic_weights.keys():
            if synapse_id.source_region != "external":
                continue
            if not synapse_id.target_population.lower().startswith("l4"):
                continue
            try:
                n = region.get_population_size(synapse_id.target_population)
            except (KeyError, ValueError, AttributeError):
                continue
            # ≈5 Hz at dt=1 ms: 0.5 % per step
            result[synapse_id] = torch.rand(n, device=device) < 0.005
    return result if result else None


def _ramp_sentinel(brain: "Brain", t: int) -> Optional[SynapticInput]:  # noqa: ARG001
    """Registry stub for the ramp pattern.

    ``_sensory_ramp`` requires a third argument (``n_timesteps``) that is not
    part of the standard ``Callable[[Brain, int], Optional[SynapticInput]]``
    contract.  This 2-arg sentinel satisfies the registry type and produces a
    clear error if someone bypasses :func:`make_sensory_input` and calls the
    registered function directly.
    """
    raise TypeError(
        "'ramp' pattern requires n_timesteps; dispatch via make_sensory_input() "
        "which injects the required parameter."
    )


# =============================================================================
# REGISTRY & PUBLIC DISPATCH
# =============================================================================

#: Ordered mapping of pattern name → generator function.
#: Add new patterns here — no changes to :func:`make_sensory_input` required.
SENSORY_PATTERNS: Dict[str, Callable[["Brain", int], Optional[SynapticInput]]] = {
    "none":                    _sensory_none,
    "background":              _sensory_background,
    "burst":                   _sensory_burst,
    "correlated_background":   _sensory_correlated_background,
    "direct_cortical":         _sensory_direct_cortical,
    "gamma":                   _sensory_gamma,
    "hippocampal_theta":        _sensory_hippocampal_theta,
    "ramp":                    _ramp_sentinel,
    "random":                  _sensory_random,
    "rhythmic":                _sensory_rhythmic,
    "slow_wave":               _sensory_slow_wave,
    "sustained_burst":         _sensory_sustained_burst,
}

#: Patterns that correspond to a **waking / alert** physiological state.
#: Health checks that depend on arousal context (e.g. thalamic burst-mode,
#: beta-band gating) should test ``pattern in WAKING_PATTERNS``.
WAKING_PATTERNS: frozenset[str] = frozenset({
    "random", "rhythmic", "burst", "sustained_burst",
    "gamma", "correlated_background", "ramp",
    "hippocampal_theta", "direct_cortical",
})

#: Patterns that simulate **NREM sleep / slow-wave** dynamics.
SLEEP_PATTERNS: frozenset[str] = frozenset({"slow_wave"})

#: Patterns that provide **neutral / spontaneous** drive with no strong
#: physiological-state implication (low-rate background or no input).
NEUTRAL_PATTERNS: frozenset[str] = frozenset({"none", "background"})


def make_sensory_input(
    brain: "Brain",
    timestep: int,
    pattern: str,
    *,
    n_timesteps: Optional[int] = None,
) -> Optional[SynapticInput]:
    """Dispatch to the appropriate per-pattern sensory-input generator.

    Parameters
    ----------
    brain:
        The live ``Brain`` instance.
    timestep:
        Current integer timestep index.
    pattern:
        Name of the input pattern.  Must be a key in :data:`SENSORY_PATTERNS`.
    n_timesteps:
        Total number of simulation timesteps in the current run.  Only used by
        the ``'ramp'`` pattern, which scales the ramp slope so that 30 % relay
        activation is reached exactly at timestep ``n_timesteps - 1``.

    Available patterns:

    * ``none``                  – spontaneous activity only
    * ``background``            – ≈2 Hz Poisson to every external-input synapse
    * ``burst``                 – single 50 ms burst at t=100 ms (30 % relay)
    * ``correlated_background`` – half relay receives shared common Poisson driver
    * ``direct_cortical``       – ≈5 Hz sparse Poisson to cortical L4 external synapses (thalamus-bypassing)
    * ``gamma``                 – sinusoidal 40 Hz drive (5–15 % relay per step)
    * ``hippocampal_theta``     – 8 Hz descending-phase drive to medial septum GABA/ACh (septohippocampal theta)
    * ``ramp``                  – linearly ramping relay activation (0→30 % over the full run)
    * ``random``                – sparse Poisson to thalamus relay (3 %, 20 %)
    * ``rhythmic``              – 8 Hz theta burst to thalamus relay
    * ``slow_wave``             – 300 ms up-state (50 % relay, 40 Hz) / 300 ms silence; tests voltage bimodality
    * ``sustained_burst``       – repeating 50 ms on / 450 ms off cycle

    Raises
    ------
    ValueError
        If *pattern* is not a key in :data:`SENSORY_PATTERNS`.
    """
    fn = SENSORY_PATTERNS.get(pattern)
    if fn is None:
        raise ValueError(
            f"Unknown input pattern: {pattern!r}. "
            f"Available patterns: {list(SENSORY_PATTERNS)}"
        )
    if pattern == "ramp":
        if n_timesteps is None:
            raise ValueError("n_timesteps must be provided for 'ramp' pattern")
        return _sensory_ramp(brain, timestep, n_timesteps)
    return fn(brain, timestep)
