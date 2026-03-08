"""NeuromodulatorHub: Centralised neuromodulator broadcast management for Brain.

This module owns the logic that was previously inline in ``Brain.forward()``:
collecting spiking neuromodulator outputs from the previous timestep and broadcasting
them as a flat ``NeuromodulatorInput`` dict to all regions.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from thalia.typing import BrainOutput, NeuromodulatorChannel, NeuromodulatorInput, PopulationName, RegionName

from .regions import NeuralRegion
from .configs import NeuralRegionConfig


class NeuromodulatorHub(nn.Module):
    """Collect, accumulate, and broadcast neuromodulator spike signals.

    ``NeuromodulatorHub`` is owned by ``Brain`` and is responsible for
    building the ``NeuromodulatorInput`` dict that is passed to every region's
    ``forward()`` at each timestep.

    It inspects each region for a ``neuromodulator_outputs`` ClassVar (a
    ``Dict[channel_key, population_name]`` mapping) and, after each timestep,
    extracts the corresponding population outputs from ``BrainOutput`` to populate
    the channel dict.

    Args:
        regions: The live ``{region_name: NeuralRegion}`` dict owned by
            ``Brain``.  A reference (not a copy) is stored so that
            regions added later are automatically visible.
    """

    def __init__(self, regions: Dict[RegionName, NeuralRegion[NeuralRegionConfig]]) -> None:
        super().__init__()
        # Store a live reference — Brain's nn.ModuleDict is the authority.
        self._regions = regions

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def build(self, last_brain_output: Optional[BrainOutput]) -> NeuromodulatorInput:
        """Build the ``NeuromodulatorInput`` dict from the previous timestep's output.

        This is a pure construction step — no side effects, no state mutation.
        The one-timestep lag is intentional (clock-driven simulation, ADR-003).

        Channels whose source region produced no spikes this step are published as a
        **zero tensor** (same shape and device as the source population), never as
        ``None``.  ``NeuromodulatorReceptor.update()`` short-circuits on ``sum()==0``
        so the computational cost is identical, but callers are guaranteed to receive
        a ``torch.Tensor`` for every channel key and need no ``is not None`` guards.

        Args:
            last_brain_output: ``BrainOutput`` from the **previous** timestep,
                or ``None`` on the very first step.

        Returns:
            ``NeuromodulatorInput`` mapping channel keys to spike tensors.
            Every key declared by any region's ``neuromodulator_outputs`` is present.
        """
        signals: NeuromodulatorInput = {}
        # Channels not yet filled with actual spikes: maps channel_key → (region_name, pop_name)
        # so we can emit zero tensors after the loop without blocking later active publishers.
        _zero_pending: dict[str, tuple[str, str]] = {}

        for region_name, region in self._regions.items():
            nm_outputs: Optional[Dict[NeuromodulatorChannel, PopulationName]] = getattr(region, 'neuromodulator_outputs', None)
            if nm_outputs is None:
                continue

            for channel_key, pop_name in nm_outputs.items():
                raw: Optional[torch.Tensor] = None
                if last_brain_output is not None:
                    region_out = last_brain_output.get(region_name, {})
                    # PopulationName may be a StrEnum; str() normalises to its value.
                    raw = region_out.get(str(pop_name))  # type: ignore[arg-type]

                if raw is not None:
                    existing = signals.get(channel_key)
                    if existing is not None:
                        # Two regions publish to the same channel.  Use additive combination
                        # clamped to 1.0: biologically, DA (or any volume-transmitted signal)
                        # released from two independent terminal fields adds in the extracellular
                        # space.  torch.maximum would undercount when both sources fire
                        # simultaneously, and is incorrect for non-binary rate signals.
                        # Shapes MUST match: a NeuromodulatorReceptor operates on a fixed-size
                        # tensor that maps 1-to-1 to the target population.  Different-size
                        # publishers need separate channel keys (preferred architecture: one
                        # unique channel per subpopulation, subscribe to all relevant channels
                        # in the receiving receptor).
                        if existing.shape != raw.shape:
                            raise ValueError(
                                f"NeuromodulatorHub: cannot combine signals for channel "
                                f"'{channel_key}' from two regions with different population "
                                f"sizes ({tuple(existing.shape)} vs {tuple(raw.shape)}).  "
                                f"Use distinct channel keys per subpopulation and subscribe "
                                f"to each separately in the receiving NeuromodulatorReceptor."
                            )
                        signals[channel_key] = torch.clamp(existing + raw, max=1.0)
                    else:
                        signals[channel_key] = raw.float()
                    # This channel now has real spikes — no zero-fill needed.
                    _zero_pending.pop(channel_key, None)
                elif channel_key not in signals and channel_key not in _zero_pending:
                    # No spikes yet from any publisher for this channel.  Record
                    # source info so we can emit a zero tensor after the loop.
                    # Using a deferred approach prevents a silent early region from
                    # blocking a later active region from writing real spikes.
                    _zero_pending[channel_key] = (region_name, str(pop_name))

        # Emit zero tensors for channels where no region produced spikes this step.
        for channel_key, (region_name, pop_name) in _zero_pending.items():
            try:
                src_region = self._regions[region_name]
                n_source: int = src_region.get_population_size(pop_name)
                first_buf = next(src_region.buffers(), None)
                src_device = first_buf.device if first_buf is not None else torch.device('cpu')
            except (KeyError, ValueError):
                n_source = 1
                src_device = torch.device('cpu')
            signals[channel_key] = torch.zeros(n_source, dtype=torch.float32, device=src_device)

        return signals

    # ------------------------------------------------------------------
    # INTROSPECTION HELPERS
    # ------------------------------------------------------------------

    def registered_channels(self) -> list[str]:
        """Return a sorted list of all channel keys declared by registered regions."""
        channels: set[str] = set()
        for region in self._regions.values():
            nm_outputs: Optional[Dict[NeuromodulatorChannel, PopulationName]] = getattr(region, 'neuromodulator_outputs', None)
            if nm_outputs:
                channels.update(nm_outputs.keys())
        return sorted(channels)

    def source_regions_for_channel(self, channel_key: str) -> list[str]:
        """Return the names of all regions that publish *channel_key*."""
        sources: list[str] = []
        for region_name, region in self._regions.items():
            nm_outputs: Optional[Dict[NeuromodulatorChannel, PopulationName]] = getattr(region, 'neuromodulator_outputs', None)
            if nm_outputs and channel_key in nm_outputs:
                sources.append(region_name)
        return sources

    def validate(self) -> None:
        """Validate neuromodulator subscriptions against published channels.

        For every region that declares ``neuromodulator_subscriptions``, checks that
        each subscribed channel key is published by at least one other region's
        ``neuromodulator_outputs``.  Raises ``ValueError`` listing all mismatches so
        the entire set of problems is visible in a single build-time error.

        Call this from ``BrainBuilder.build()`` after all regions are instantiated.

        Raises:
            ValueError: If any subscription references an unpublished channel.
        """
        published: set[str] = set(self.registered_channels())

        errors: list[str] = []
        for region_name, region in self._regions.items():
            subscriptions: list[NeuromodulatorChannel] = getattr(region, 'neuromodulator_subscriptions', [])
            for channel_key in subscriptions:
                if channel_key not in published:
                    errors.append(
                        f"  Region '{region_name}' subscribes to channel '{channel_key}' "
                        f"but no region publishes it. "
                        f"Published channels: {sorted(published) or '(none)'}"
                    )

        if errors:
            raise ValueError(
                "NeuromodulatorHub.validate() failed — subscription/publication mismatches:\n"
                + "\n".join(errors)
            )
