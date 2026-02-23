"""NeuromodulatorHub: Centralised neuromodulator broadcast management for DynamicBrain.

This module owns the logic that was previously inline in ``DynamicBrain.forward()``:
collecting spiking neuromodulator outputs from the previous timestep and broadcasting
them as a flat ``NeuromodulatorInput`` dict to all regions.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from thalia.typing import BrainOutput, NeuromodulatorInput, RegionName

from .regions import NeuralRegion
from .configs import NeuralRegionConfig


class NeuromodulatorHub(nn.Module):
    """Collect, accumulate, and broadcast neuromodulator spike signals.

    ``NeuromodulatorHub`` is owned by ``DynamicBrain`` and is responsible for
    building the ``NeuromodulatorInput`` dict that is passed to every region's
    ``forward()`` at each timestep.

    It inspects each region for a ``neuromodulator_outputs`` ClassVar (a
    ``Dict[channel_key, population_name]`` mapping) and, after each timestep,
    extracts the corresponding population outputs from ``BrainOutput`` to populate
    the channel dict.

    Args:
        regions: The live ``{region_name: NeuralRegion}`` dict owned by
            ``DynamicBrain``.  A reference (not a copy) is stored so that
            regions added later are automatically visible.
    """

    def __init__(self, regions: Dict[RegionName, NeuralRegion[NeuralRegionConfig]]) -> None:
        super().__init__()
        # Store a live reference — DynamicBrain's nn.ModuleDict is the authority.
        self._regions = regions

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def build(self, last_brain_output: Optional[BrainOutput]) -> NeuromodulatorInput:
        """Build the ``NeuromodulatorInput`` dict from the previous timestep's output.

        This is a pure construction step — no side effects, no state mutation.
        The one-timestep lag is intentional (clock-driven simulation, ADR-003).

        Args:
            last_brain_output: ``BrainOutput`` from the **previous** timestep,
                or ``None`` on the very first step.

        Returns:
            ``NeuromodulatorInput`` mapping channel keys to spike tensors (or
            ``None`` for channels whose source region has not fired yet).
        """
        signals: NeuromodulatorInput = {}

        for region_name, region in self._regions.items():
            nm_outputs: Optional[Dict[str, str]] = getattr(region, 'neuromodulator_outputs', None)
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
                    # Logical OR accumulates spikes from multiple source regions
                    # (e.g., VTA + SNc both contributing to a 'da' flood channel).
                    signals[channel_key] = (existing | raw) if existing is not None else raw
                elif channel_key not in signals:
                    # Publish the channel as None so regions know it exists but
                    # there was no activity this step (receptor dynamics return
                    # decayed baseline instead of crashing on a missing key).
                    signals[channel_key] = None

        return signals

    # ------------------------------------------------------------------
    # INTROSPECTION HELPERS
    # ------------------------------------------------------------------

    def registered_channels(self) -> list[str]:
        """Return a sorted list of all channel keys declared by registered regions."""
        channels: set[str] = set()
        for region in self._regions.values():
            nm_outputs: Optional[Dict[str, str]] = getattr(region, 'neuromodulator_outputs', None)
            if nm_outputs:
                channels.update(nm_outputs.keys())
        return sorted(channels)

    def source_regions_for_channel(self, channel_key: str) -> list[str]:
        """Return the names of all regions that publish *channel_key*."""
        sources: list[str] = []
        for region_name, region in self._regions.items():
            nm_outputs: Optional[Dict[str, str]] = getattr(region, 'neuromodulator_outputs', None)
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
            subscriptions: list[str] = getattr(region, 'neuromodulator_subscriptions', [])
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
