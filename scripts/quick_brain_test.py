"""Quick test script for the Brain class to verify basic functionality and output shapes."""

import sys
import time

import torch

from thalia.brain import BrainBuilder, Thalamus
from thalia.typing import BrainOutput, SynapseId, SynapticInput


# NOTE: Enable line buffering for real-time output
sys.stdout.reconfigure(line_buffering=True)


# Create brain with default preset
print("Creating brain...")
start = time.time()
brain = BrainBuilder.preset("default")
print(f"Brain created in {time.time() - start:.2f}s")

print("=== Brain Regions ===")
for region_name, region in brain.regions.items():
    print(f"  - {region_name}")
    print("    - Synaptic weights:")
    for synapse_id, weights in region._synaptic_weights.items():
        print(f"      - Name: {synapse_id}, weights shape: {weights.shape}")

thalamus = brain.get_first_region_of_type(Thalamus)
assert thalamus is not None, "Thalamus region not found in the brain."

# Perform a forward pass through the brain
print("\n=== Forward Pass ===")
print("Preparing input...")
sensory_input = torch.randn(thalamus.relay_size) < 0.5  # Random binary spikes for testing
brain_input: SynapticInput = {
    SynapseId.external_sensory_to_thalamus_relay(): sensory_input,
}
print("Running forward pass...")
start = time.time()
brain_output: BrainOutput = brain.forward(brain_input)
print(f"Forward pass completed in {time.time() - start:.2f}s")

# Print output shapes and number of active spikes for each output port
print("\n=== Output ===")
for region_name, region_output in brain_output.items():
    print(f"{region_name}:")
    for port_name, spikes in region_output.items():
        n_active = spikes.sum().item()
        print(f"  {port_name}: {spikes.shape} ({n_active} active spikes)")
