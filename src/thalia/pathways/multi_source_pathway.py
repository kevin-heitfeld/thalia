"""
MultiSourcePathway: Pathway that integrates inputs from multiple source regions.

This pathway handles the biological reality that most brain regions receive
convergent inputs from multiple sources. Instead of using adapters to concatenate
inputs, the pathway itself manages multiple input streams and their integration.

Key Design Principles:
======================
1. ONE PATHWAY PER FUNCTIONAL CONNECTION: The corticostriatal pathway includes
   synaptic integration at striatal dendrites, not just spike transmission.

2. PATHWAY OWNS THE WEIGHTS: The weight matrix maps combined inputs to target,
   matching biological reality where synapses are part of the pathway.

3. NO ADAPTER COMPLEXITY: Pathway handles buffering and integration internally,
   eliminating the need for separate adapter logic.

4. CLEAN GROWTH: When a source region grows, only this pathway needs to update
   its input dimension for that specific source.

Example Usage:
==============
    # Create multi-source pathway for striatum
    # Receives from cortex L5, hippocampus, and PFC
    sources = [
        ("cortex", "l5"),      # Cortex layer 5
        ("hippocampus", None), # Full hippocampus output
        ("pfc", None),         # Full PFC output
    ]

    pathway = MultiSourcePathway(
        sources=sources,
        target="striatum",
        config=PathwayConfig(n_output=90, ...),
    )

    # Forward pass with dict of source outputs
    output_spikes = pathway({
        "cortex": cortex_spikes,
        "hippocampus": hippo_spikes,
        "pfc": pfc_spikes,
    })

    # Growth: only this pathway updates when cortex grows
    pathway.grow_source("cortex", new_size=300)
"""

from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn

from thalia.pathways.spiking_pathway import SpikingPathway
from thalia.core.base.component_config import PathwayConfig
from thalia.managers.component_registry import register_pathway
from thalia.components.synapses.weight_init import WeightInitializer


@register_pathway(
    "multi_source",
    description="Pathway integrating inputs from multiple source regions",
    version="1.0",
    author="Thalia Project",
    config_class=PathwayConfig,
)
class MultiSourcePathway(SpikingPathway):
    """Pathway that receives and integrates inputs from multiple sources.

    This pathway extends SpikingPathway to handle multiple input streams,
    each potentially from different ports of different regions. It manages:
    - Individual size tracking per source
    - Input concatenation in consistent order
    - Growth when any source expands
    - Port extraction from source outputs

    Attributes:
        sources: List of (region_name, port) tuples defining input sources
        input_sizes: Dict mapping source names to their current sizes
        weights: Combined weight matrix [n_output, total_input_size]
    """

    def __init__(
        self,
        sources: List[Tuple[str, Optional[str]]],
        target: str,
        config: PathwayConfig,
    ):
        """Initialize multi-source pathway.

        Args:
            sources: List of (region_name, port) tuples, e.g.,
                     [("cortex", "l5"), ("hippocampus", None), ("pfc", None)]
            target: Target region name (for identification/logging)
            config: Pathway configuration with n_output set
                   (n_input will be computed from sources)
        """
        self.sources = sources
        self.target = target
        self.input_sizes: Dict[str, int] = {}

        # Calculate total input size from sources
        # NOTE: Sizes must be provided during initialization or set before first forward
        total_input = 0
        for source_name, port in sources:
            # For now, we'll initialize with equal sizes and let grow_source fix it
            # This will be set properly during brain construction
            initial_size = config.n_input // len(sources) if hasattr(config, 'n_input') else 64
            self.input_sizes[source_name] = initial_size
            total_input += initial_size

        # Override config n_input with computed total
        config.n_input = total_input

        # Initialize parent SpikingPathway with combined input size
        super().__init__(config)

    def set_source_size(self, source_name: str, size: int) -> None:
        """Set the size for a specific input source.

        This should be called during brain construction to set correct sizes
        before the first forward pass.

        Args:
            source_name: Name of the source region
            size: Output size from that source (after port extraction if applicable)
        """
        if source_name not in self.input_sizes:
            raise ValueError(
                f"Unknown source '{source_name}'. Valid sources: {list(self.input_sizes.keys())}"
            )

        old_size = self.input_sizes[source_name]
        if old_size == size:
            return  # No change needed

        # Update size tracking
        self.input_sizes[source_name] = size

        # Rebuild weight matrix with new total input size
        self._resize_weights()

    def forward(
        self,
        source_outputs: Dict[str, torch.Tensor],
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Process inputs from multiple sources.

        Args:
            source_outputs: Dict mapping source names to their output tensors
                           Shape: {source_name: [source_size]}
            dt: Timestep in milliseconds

        Returns:
            Output spikes [n_output]

        Note:
            Missing sources are filled with zero tensors (biologically plausible -
            no spikes from that source yet). This prevents deadlocks in recurrent
            networks where multiple regions feed into each other.
        """
        # Extract and concatenate inputs in consistent order
        inputs = []
        for source_name, _port in self.sources:  # Port is for documentation only
            if source_name not in source_outputs:
                # Use zero tensor for missing source (no spikes yet)
                # This is biologically plausible and prevents deadlocks
                expected_size = self.input_sizes[source_name]
                zero_input = torch.zeros(expected_size, device=self.device)
                inputs.append(zero_input)
                continue

            source_output = source_outputs[source_name]

            # NOTE: Port extraction already happened in DynamicBrain._schedule_downstream_events
            # before the output was buffered. The 'port' in sources tuple is for documentation only.

            # Validate size
            expected_size = self.input_sizes[source_name]
            if source_output.shape[0] != expected_size:
                raise ValueError(
                    f"Size mismatch for source '{source_name}': "
                    f"expected {expected_size}, got {source_output.shape[0]}"
                )

            inputs.append(source_output)

        # Concatenate all inputs
        combined_input = torch.cat(inputs, dim=0)

        # Forward through parent SpikingPathway
        # This handles all the spiking dynamics, STDP, etc.
        return super().forward(combined_input, dt=dt)

    def _extract_port(
        self,
        output: torch.Tensor,
        source_name: str,
        port: str,
    ) -> torch.Tensor:
        """Extract specific port from source output.

        For layered cortex, this extracts L2/3 or L5 output.
        For other regions, this is a no-op.

        Args:
            output: Full output from source region
            source_name: Name of source (for error messages)
            port: Port identifier (e.g., 'l23', 'l5')

        Returns:
            Port-specific output slice
        """
        # This is a simplified version - in reality, we'd need to know
        # the port sizes from the source region configuration
        # For now, assume equal split for l23/l5
        if port in ("l23", "l5"):
            mid = output.shape[0] // 2
            if port == "l23":
                return output[:mid]
            else:
                return output[mid:]
        else:
            # Unknown port - return full output
            return output

    # === GROWTH METHODS ===
    def grow_source(self, source_name: str, new_size: int) -> None:
        """Grow input dimension for a specific source (MultiSourcePathway only).

        Called when one source region in a multi-source connection grows.
        Only that source's contribution to the total input expands.

        Args:
            source_name: Name of the source region that grew
            new_size: New total size for that source (not delta!)

        Effects:
            - Updates self.input_sizes[source_name]
            - Resizes weight matrix to accommodate new total input size
            - Preserves weights from other sources
            - Updates config.n_input to sum of all source sizes

        Example:
            >>> pathway.input_sizes  # {'cortex': 100, 'hippocampus': 64}
            >>> pathway.grow_source('cortex', 120)  # Cortex grew by 20
            >>> pathway.input_sizes  # {'cortex': 120, 'hippocampus': 64}
            >>> pathway.config.n_input  # 184 (was 164)
        """
        if source_name not in self.input_sizes:
            raise ValueError(
                f"Unknown source '{source_name}'. Valid sources: {list(self.input_sizes.keys())}"
            )

        old_size = self.input_sizes[source_name]
        if new_size <= old_size:
            return  # No growth needed

        # Update size tracking
        self.input_sizes[source_name] = new_size

        # Grow the weight matrix
        self._resize_weights()

    def _resize_weights(self) -> None:
        """Resize weight matrix to match current input sizes.

        This is called when any source grows. We create a new larger weight matrix
        and copy existing weights, initializing new connections.
        """
        # Calculate new total input size
        new_input_size = sum(self.input_sizes.values())
        old_input_size = self.weights.shape[1]
        n_output = self.weights.shape[0]

        if new_input_size == old_input_size:
            return  # No change needed

        # Create new weight matrix
        new_weights = WeightInitializer.gaussian(
            n_output=n_output,
            n_input=new_input_size,
            mean=self.config.init_mean,
            std=self.config.init_std,
            device=self.config.device,
        )

        # Copy existing weights to preserve learned connections
        # We need to copy source-by-source to maintain alignment
        old_offset = 0
        new_offset = 0
        for source_name, _ in self.sources:  # Port is for documentation only
            old_size = min(self.input_sizes[source_name], old_input_size - old_offset)
            new_size = self.input_sizes[source_name]

            if old_size > 0:
                # Copy existing weights for this source
                new_weights[:, new_offset:new_offset + old_size] = \
                    self.weights[:, old_offset:old_offset + old_size]

            old_offset += old_size
            new_offset += new_size

        # Update weights parameter
        self.weights = nn.Parameter(new_weights, requires_grad=False)

        # Update config for consistency
        self.config.n_input = new_input_size

        # Also resize other tensors that depend on input size
        self._resize_input_dependent_tensors(new_input_size)

    def _resize_input_dependent_tensors(self, new_input_size: int) -> None:
        """Resize tensors that depend on input size.

        This includes:
        - Axonal delays
        - Connectivity mask
        - Delay buffer
        - Trace manager tensors
        """
        old_input_size = self.axonal_delays.shape[1]
        n_output = self.axonal_delays.shape[0]

        if new_input_size == old_input_size:
            return  # No resize needed

        # Resize axonal delays
        new_delays = WeightInitializer.gaussian(
            n_output=n_output,
            n_input=new_input_size,
            mean=self.config.axonal_delay_ms,
            std=self.config.delay_variability * self.config.axonal_delay_ms,
            device=self.config.device,
        ).clamp(min=0.1)

        # Copy what we can from old delays
        copy_size = min(old_input_size, new_input_size)
        new_delays[:, :copy_size] = self.axonal_delays[:, :copy_size]
        self.axonal_delays = new_delays

        # Resize connectivity mask if present
        if self.connectivity_mask is not None:
            new_mask = WeightInitializer.sparse_random(
                n_output=n_output,
                n_input=new_input_size,
                sparsity=self.config.sparsity,
                device=self.config.device,
            )
            copy_size = min(old_input_size, new_input_size)
            new_mask[:, :copy_size] = self.connectivity_mask[:, :copy_size]
            self.connectivity_mask = new_mask

        # Resize delay buffer
        max_delay_steps = self.delay_buffer.shape[0]
        old_delay_input = self.delay_buffer.shape[1]
        new_delay_buffer = torch.zeros(
            max_delay_steps,
            new_input_size,
            device=self.config.device,
        )
        copy_size = min(old_delay_input, new_input_size)
        new_delay_buffer[:, :copy_size] = self.delay_buffer[:, :copy_size]
        self.delay_buffer = new_delay_buffer

        # Resize trace manager - recreate with correct size
        # We can't use grow_dimension reliably because we don't know the current state
        from thalia.learning.eligibility.trace_manager import EligibilityTraceManager, STDPConfig

        # Copy config from existing trace manager if possible
        old_config = self._trace_manager.config if self._trace_manager else STDPConfig()

        self._trace_manager = EligibilityTraceManager(
            n_input=new_input_size,
            n_output=n_output,
            config=old_config,
            device=self.config.device,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get pathway diagnostics including per-source information.

        Returns:
            Dict with standard pathway diagnostics plus source information
        """
        diag = super().get_diagnostics()

        # Add multi-source specific information
        diag["multi_source"] = {
            "sources": [f"{name}:{port if port else 'full'}" for name, port in self.sources],
            "input_sizes": self.input_sizes.copy(),
            "total_input_size": sum(self.input_sizes.values()),
            "target": self.target,
        }

        return diag

    def __repr__(self) -> str:
        """String representation showing all sources."""
        sources_str = ", ".join(
            f"{name}:{port if port else 'full'}" for name, port in self.sources
        )
        return (
            f"MultiSourcePathway({sources_str} → {self.target}, "
            f"inputs={sum(self.input_sizes.values())}, outputs={self.config.n_output})"
        )
