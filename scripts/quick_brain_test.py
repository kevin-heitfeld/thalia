"""Quick test script to create a brain and perform a forward pass with random input, printing outputs and synaptic weight stats for verification."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Dict, Optional

import torch

from thalia import GlobalConfig
from thalia.brain import BrainBuilder
from thalia.brain.regions.population_names import ExternalPopulation
from thalia.learning.strategies import LearningStrategy
from thalia.typing import SynapseId

if TYPE_CHECKING:
    from thalia.brain import Brain
    from thalia.typing import BrainOutput, RegionSizes, SynapticInput


# NOTE: Enable line buffering for real-time output
sys.stdout.reconfigure(line_buffering=True)


if __name__ == "__main__":
    print("GlobalConfig:")
    for field in GlobalConfig.__dataclass_fields__.values():
        value = getattr(GlobalConfig, field.name)
        print(f"    {field.name:25s}: {value}")

    # Override default population sizes for testing
    external_reward_size: int = 36
    external_sensory_size: int = 42
    default_overrides: Dict[str, RegionSizes] = {
        "population_sizes": {
            SynapseId._EXTERNAL_REGION_NAME: {
                ExternalPopulation.REWARD: external_reward_size,
                ExternalPopulation.SENSORY: external_sensory_size,
            },
        },
    }

    # Create brain with default preset
    print("\nCreating brain...")
    start_time = time.time()
    brain: Brain = BrainBuilder.preset("default", **default_overrides)
    print(f"Brain created in {time.time() - start_time:.2f}s")

    print("=== Brain Regions ===")
    for region_name, region in brain.regions.items():
        print(f"  - {region_name} ({region.__class__.__name__})")
        # print("    - Synaptic weights:")
        # for synapse_id, weights in region.synaptic_weights.items():
        #     mean_weight = weights.mean()
        #     min_weight = weights.min()
        #     max_weight = weights.max()
        #     stp: Optional[ShortTermPlasticity] = region.get_stp_module(synapse_id)
        #     if stp is not None:
        #         stp_info = f"(STP: U={stp.config.U}, tau_d={stp.config.tau_d}ms, tau_f={stp.config.tau_f}ms)"
        #     else:
        #         stp_info = "(No STP)"
        #     print(
        #         f"  {synapse_id} - "
        #         f"Mean: {mean_weight:.4f}, Min: {min_weight:.4f}, Max: {max_weight:.4f}"
        #         f" {stp_info}"
        #     )

    # Perform a forward pass through the brain
    print("\n=== Forward Pass ===")
    print("Preparing input...")
    sensory_input = torch.randn(external_sensory_size) < 0.5  # Random binary spikes for testing
    brain_input: SynapticInput = {SynapseId.external_sensory_to_thalamus_relay("thalamus_sensory"): sensory_input}
    print("Running forward pass...")
    start_time = time.time()
    brain_output: BrainOutput = brain.forward(brain_input)
    print(f"Forward pass completed in {time.time() - start_time:.2f}s")

    # Print learning strategies after forward pass to verify lazy registration of external input learning strategies
    print("\n=== Learning strategies ===")
    for region_name, region in brain.regions.items():
        print(f"  - {region_name} ({region.__class__.__name__})")
        for synapse_id in region.synaptic_weights.keys():
            strategy: Optional[LearningStrategy] = region.get_learning_strategy(synapse_id)
            print(f"      {synapse_id}: {strategy.__class__.__name__ if strategy is not None else 'None'}")

    # Print output shapes and number of active spikes for each output port
    # print("\n=== Output ===")
    # for region_name, region_output in brain_output.items():
    #     print(f"{region_name}:")
    #     for population_name, spikes in region_output.items():
    #         n_active = spikes.sum().item()
    #         print(f"  {population_name}: {spikes.shape} ({n_active} active spikes)")
