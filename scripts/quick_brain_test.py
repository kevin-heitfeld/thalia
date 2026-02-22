"""Quick test script for the Brain class to verify basic functionality and output shapes."""

import sys
import time
from typing import Optional

import torch

from thalia.brain import BrainBuilder, DynamicBrain, Thalamus
from thalia.components import ShortTermPlasticity, STPConfig
from thalia.typing import BrainOutput, SynapseId, SynapticInput


# NOTE: Enable line buffering for real-time output
sys.stdout.reconfigure(line_buffering=True)


# Create brain with default preset
print("Creating brain...")
start = time.time()
brain: DynamicBrain = BrainBuilder.preset("default")
print(f"Brain created in {time.time() - start:.2f}s")

print("=== Brain Regions ===")
for region_name, region in brain.regions.items():
    print(f"  - {region_name}")
    print("    - Synaptic weights:")
    for synapse_id, weights in region._synaptic_weights.items():
        print(f"      - Name: {synapse_id}, weights shape: {weights.shape}")

for region_name, region in brain.regions.items():
    print(f"  - {region_name}")
    print("    - Synaptic weights:")
    for synapse_id, weights in region._synaptic_weights.items():
        mean_weight = weights.mean()
        min_weight = weights.min()
        max_weight = weights.max()
        stp: Optional[ShortTermPlasticity] = region.stp_modules.get(synapse_id, None)
        if stp is not None:
            stp_config: STPConfig = stp.config
            stp_info = f"(STP: U={stp_config.U}, tau_d={stp_config.tau_d}ms, tau_f={stp_config.tau_f}ms)"
        else:
            stp_info = "(No STP)"
        print(
            f"  {region.__class__.__name__}:{synapse_id} - "
            f"Mean: {mean_weight:.4f}, Min: {min_weight:.4f}, Max: {max_weight:.4f}"
            f" {stp_info}"
        )

thalamus = brain.get_first_region_of_type(Thalamus)
assert thalamus is not None, "Thalamus region not found in the brain."

# Perform a forward pass through the brain
print("\n=== Forward Pass ===")
print("Preparing input...")
sensory_input = torch.randn(thalamus.relay_size) < 0.5  # Random binary spikes for testing
brain_input: SynapticInput = {SynapseId.external_sensory_to_thalamus_relay(): sensory_input}
print("Running forward pass...")
start = time.time()
brain_output: BrainOutput = brain.forward(brain_input)
print(f"Forward pass completed in {time.time() - start:.2f}s")

# Print output shapes and number of active spikes for each output port
print("\n=== Output ===")
for region_name, region_output in brain_output.items():
    print(f"{region_name}:")
    for population_name, spikes in region_output.items():
        n_active = spikes.sum().item()
        print(f"  {population_name}: {spikes.shape} ({n_active} active spikes)")
