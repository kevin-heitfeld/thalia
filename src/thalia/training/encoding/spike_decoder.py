"""Decode brain output spikes into behavioural readout counts."""

from __future__ import annotations

from thalia.typing import BrainOutput, RegionName, PopulationName


def count_spikes(
    outputs: BrainOutput,
    region: RegionName,
    population: PopulationName,
    start_neuron: int,
    end_neuron: int,
) -> int:
    """Count spikes in a neuron slice of a single-timestep output.

    Args:
        outputs: Brain output from one ``brain.forward()`` call.
        region: Region name to read from.
        population: Population name within the region.
        start_neuron: First neuron index (inclusive).
        end_neuron: Last neuron index (exclusive).

    Returns:
        Number of neurons that spiked in ``[start_neuron, end_neuron)``.
    """
    spikes = outputs[region][population]
    return int(spikes[start_neuron:end_neuron].sum().item())


class ReadoutGroup:
    """A named slice of a population used as a behavioural readout channel.

    Attributes:
        name: Label for this readout group (used as key in spike_counts).
        region: Brain region containing the readout neurons.
        population: Population within the region.
        start_neuron: First neuron index (inclusive).
        end_neuron: Last neuron index (exclusive).
    """

    __slots__ = ("name", "region", "population", "start_neuron", "end_neuron")

    def __init__(
        self,
        name: str,
        region: RegionName,
        population: PopulationName,
        start_neuron: int,
        end_neuron: int,
    ) -> None:
        self.name = name
        self.region = region
        self.population = population
        self.start_neuron = start_neuron
        self.end_neuron = end_neuron

    def count(self, outputs: BrainOutput) -> int:
        """Count spikes in this readout group for one timestep."""
        return count_spikes(
            outputs, self.region, self.population,
            self.start_neuron, self.end_neuron,
        )
