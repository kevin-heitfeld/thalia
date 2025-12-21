"""Growth mixin for region neuron expansion.

This mixin provides utility methods for expanding weight matrices and state tensors
during region growth. All regions implement their own grow_output() methods and use
ConductanceLIF.grow_neurons() for neuron population growth.

Utility Methods:
- _expand_weights(): Expand weight matrix by adding output neurons
- _expand_state_tensors(): Expand state tensors (traces, memory, etc.)

Historical Context:
- Prior to this mixin, grow_output() was duplicated across 4+ regions (~320 lines)
- This mixin consolidates weight and state expansion utilities
- As of Dec 2025: All neurons use ConductanceLIF with direct grow_neurons() support
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from thalia.components.synapses.weight_init import WeightInitializer
from thalia.utils.core_utils import clamp_weights


class GrowthMixin:
    """Mixin providing utility methods for region neuron growth.

    This mixin provides standardized helper methods for:
    1. Expanding weight matrices when adding output neurons
    2. Expanding state tensors (traces, memory, eligibility)

    All regions implement their own grow_output() methods and call
    neurons.grow_neurons() directly for neuron population growth.

    Example Usage:
        class MyRegion(NeuralRegion, GrowthMixin):
            def grow_output(self, n_new, initialization='xavier', sparsity=0.1):
                # 1. Expand weights
                self.weights = self._expand_weights(
                    self.weights, n_new, initialization, sparsity
                )

                # 2. Update config
                self.config = replace(self.config, n_output=self.config.n_output + n_new)

                # 3. Grow neurons
                self.neurons.grow_neurons(n_new)

                # 4. Expand state tensors (if needed)
                expanded = self._expand_state_tensors(
                    {'traces': self.traces}, n_new
                )
                self.traces = expanded['traces']

    Attributes:
        config: Region configuration with w_min/w_max and n_output
        device: PyTorch device for tensor creation
        neurons: ConductanceLIF neuron population
    """

    def _expand_weights(
        self,
        current_weights: nn.Parameter,
        n_new: int,
        initialization: str = 'xavier',
        sparsity: float = 0.1,
        scale: Optional[float] = None,
    ) -> nn.Parameter:
        """Expand weight matrix by adding n_new output neurons.

        This helper consolidates weight expansion logic that was duplicated
        across 6+ regions. Handles initialization strategies and clamping.

        Args:
            current_weights: Existing weight matrix [n_output, n_input]
            n_new: Number of new output neurons to add
            initialization: Strategy ('xavier', 'sparse_random', 'uniform')
            sparsity: Connection sparsity for sparse_random (0.0-1.0)
            scale: Optional scale factor for new weights (defaults to w_max * 0.2)

        Returns:
            Expanded weight matrix [n_output + n_new, n_input]

        Example:
            >>> # In a region's grow_output() method:
            >>> self.weights = self._expand_weights(
            ...     self.weights, n_new=10, initialization='xavier'
            ... )
        """
        n_input = current_weights.shape[1]
        device = current_weights.device

        # Default scale: 20% of w_max (common across regions)
        if scale is None:
            scale = self.config.w_max * 0.2

        # Initialize new weights using specified strategy
        if initialization == 'xavier':
            new_weights = WeightInitializer.xavier(
                n_output=n_new,
                n_input=n_input,
                gain=0.2,
                device=device,
            ) * self.config.w_max
        elif initialization == 'sparse_random':
            new_weights = WeightInitializer.sparse_random(
                n_output=n_new,
                n_input=n_input,
                sparsity=sparsity,
                scale=scale,
                device=device,
            )
        else:  # uniform
            new_weights = WeightInitializer.uniform(
                n_output=n_new,
                n_input=n_input,
                low=0.0,
                high=scale,
                device=device,
            )

        # Clamp to config bounds
        new_weights = clamp_weights(new_weights, self.config.w_min, self.config.w_max, inplace=False)

        # Concatenate with existing weights
        expanded = torch.cat([current_weights.data, new_weights], dim=0)
        return nn.Parameter(expanded)

    def _expand_state_tensors(
        self,
        state_dict: Dict[str, torch.Tensor],
        n_new: int,
    ) -> Dict[str, torch.Tensor]:
        """Expand 1D state tensors (membrane, traces, etc.) by n_new neurons.

        This helper consolidates state expansion logic that was duplicated
        across regions. Handles any 1D or 2D tensor.

        Args:
            state_dict: Dict of tensors to expand {name: tensor[n_neurons] or [n_neurons, dim]}
            n_new: Number of new neurons to add

        Returns:
            Dict with expanded tensors {name: tensor[n_neurons + n_new, ...]}
            New neuron entries are initialized to zero.

        Example:
            >>> # In a region's grow_output() method:
            >>> expanded = self._expand_state_tensors({
            ...     'output_trace': self.output_trace,
            ...     'working_memory': self.working_memory,
            ... }, n_new=10)
            >>> self.output_trace = expanded['output_trace']
            >>> self.working_memory = expanded['working_memory']
        """
        expanded = {}
        for name, tensor in state_dict.items():
            if tensor is None:
                expanded[name] = None
                continue

            device = tensor.device

            # Handle 1D tensors [n_neurons]
            if tensor.dim() == 1:
                new_values = torch.zeros(n_new, device=device, dtype=tensor.dtype)
                expanded[name] = torch.cat([tensor, new_values], dim=0)
            # Handle 2D tensors [n_neurons, dim]
            elif tensor.dim() == 2:
                new_values = torch.zeros(n_new, tensor.shape[1], device=device, dtype=tensor.dtype)
                expanded[name] = torch.cat([tensor, new_values], dim=0)
            else:
                raise ValueError(f"Cannot expand tensor '{name}' with {tensor.dim()} dimensions")

        return expanded
