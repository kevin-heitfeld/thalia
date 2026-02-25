"""Quick test script for the Brain class to verify basic functionality and output shapes."""

import sys
import time
from typing import Optional, cast

import torch

from thalia.brain import BrainBuilder, DynamicBrain
from thalia.brain.regions.population_names import ExternalPopulation, ThalamusPopulation
from thalia.components import ShortTermPlasticity, STPConfig
from thalia.typing import BrainOutput, SynapseId, SynapticInput


# NOTE: Enable line buffering for real-time output
sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    # Override default population sizes for testing
    external_reward_size: int = 10
    external_sensory_size: int = 42
    default_overrides = {
        "population_sizes": {
            SynapseId._EXTERNAL_REGION_NAME: {
                ExternalPopulation.REWARD: external_reward_size,
                ExternalPopulation.SENSORY: external_sensory_size,
            },
        },
    }

    # Create brain with default preset
    print("Creating brain...")
    start_time = time.time()
    # TODO: brain: DynamicBrain = BrainBuilder.preset("default", overrides=default_overrides)
    brain: DynamicBrain = BrainBuilder.preset("default")
    print(f"Brain created in {time.time() - start_time:.2f}s")

    # TODO
    # {
    thalamus = brain.get_region_by_name("thalamus")
    assert thalamus is not None, "Thalamus region not found in brain"
    external_sensory_size = thalamus.get_population_size(ThalamusPopulation.RELAY)
    # }

    print("=== Brain Regions ===")
    for region_name, region in brain.regions.items():
        print(f"  - {region_name} ({region.__class__.__name__})")
        print("    - Synaptic weights:")
        for synapse_id, weights in region.synaptic_weights.items():
            mean_weight = weights.mean()
            min_weight = weights.min()
            max_weight = weights.max()
            stp: Optional[ShortTermPlasticity] = cast(Optional[ShortTermPlasticity], region.stp_modules.get(synapse_id, None))
            if stp is not None:
                stp_config: STPConfig = stp.config
                stp_info = f"(STP: U={stp_config.U}, tau_d={stp_config.tau_d}ms, tau_f={stp_config.tau_f}ms)"
            else:
                stp_info = "(No STP)"
            print(
                f"  {synapse_id} - "
                f"Mean: {mean_weight:.4f}, Min: {min_weight:.4f}, Max: {max_weight:.4f}"
                f" {stp_info}"
            )

    # Perform a forward pass through the brain
    print("\n=== Forward Pass ===")
    print("Preparing input...")
    sensory_input = torch.randn(external_sensory_size) < 0.5  # Random binary spikes for testing
    brain_input: SynapticInput = {SynapseId.external_sensory_to_thalamus_relay("thalamus"): sensory_input}
    print("Running forward pass...")
    start_time = time.time()
    brain_output: BrainOutput = brain.forward(brain_input)
    print(f"Forward pass completed in {time.time() - start_time:.2f}s")

    # Print output shapes and number of active spikes for each output port
    print("\n=== Output ===")
    for region_name, region_output in brain_output.items():
        print(f"{region_name}:")
        for population_name, spikes in region_output.items():
            n_active = spikes.sum().item()
            print(f"  {population_name}: {spikes.shape} ({n_active} active spikes)")


if __name__ == "__main__":
    main()
