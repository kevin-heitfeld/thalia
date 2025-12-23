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

from thalia.components.synapses import WeightInitializer
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

        # Default scale: Use constant from regulation module (Architecture Review 2025-12-21, Tier 1.3)
        if scale is None:
            from thalia.regulation.region_architecture_constants import GROWTH_NEW_WEIGHT_SCALE
            scale = self.config.w_max * GROWTH_NEW_WEIGHT_SCALE

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

    def _create_new_weights(
        self,
        n_output: int,
        n_input: int,
        initialization: str = 'xavier',
        sparsity: float = 0.1,
    ) -> torch.Tensor:
        """Create new weight tensor using specified initialization strategy.

        Centralized weight creation for grow_input/grow_output methods.
        Eliminates need for per-region `new_weights_for()` helper functions.

        This consolidates a pattern that was duplicated across 8+ regions
        (Architecture Review 2025-12-21, Tier 1.1).

        Args:
            n_output: Number of output neurons
            n_input: Number of input neurons
            initialization: 'xavier', 'sparse_random', or 'uniform'
            sparsity: Connection sparsity for sparse_random (0.0-1.0)

        Returns:
            New weight tensor [n_output, n_input]

        Example:
            >>> # In a region's grow_input() or grow_output() method:
            >>> new_cols = self._create_new_weights(
            ...     n_output=self.config.n_output,
            ...     n_input=n_new,
            ...     initialization='xavier',
            ... )
            >>> self.weights.data = torch.cat([self.weights.data, new_cols], dim=1)
        """
        if initialization == 'xavier':
            return WeightInitializer.xavier(n_output, n_input, device=self.device)
        elif initialization == 'sparse_random':
            return WeightInitializer.sparse_random(
                n_output, n_input, sparsity, device=self.device
            )
        else:  # uniform
            return WeightInitializer.uniform(n_output, n_input, device=self.device)

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

    def _expand_weights_output(
        self,
        n_new: int,
        weight_param_name: str = 'weights',
        init_method: str = 'xavier',
        sparsity: float = 0.1,
    ) -> None:
        """Expand weight matrix with new output rows (add neurons).

        Helper method that encapsulates the common pattern:
        1. Get current weights
        2. Create new rows with specified initialization
        3. Concatenate and update parameter
        4. Clamp to bounds

        This consolidates ~30 lines per region × 6 regions = 180 lines
        (Architecture Review 2025-12-22, Tier 2.2).

        Args:
            n_new: Number of output neurons to add
            weight_param_name: Name of weight parameter attribute (default: 'weights')
            init_method: Initialization method ('xavier', 'sparse_random', 'uniform')
            sparsity: Sparsity for sparse_random initialization

        Example:
            >>> def grow_output(self, n_new: int) -> None:
            ...     self._expand_weights_output(n_new, init_method='xavier')
            ...     self.config = replace(self.config, n_output=self.config.n_output + n_new)
            ...     self.neurons.grow_neurons(n_new)
        """
        current_weights = getattr(self, weight_param_name)
        n_input = current_weights.shape[1]
        device = current_weights.device

        # Create new rows
        if init_method == 'xavier':
            new_rows = WeightInitializer.xavier(
                n_output=n_new,
                n_input=n_input,
                gain=0.2,
                device=device,
            ) * self.config.w_max
        elif init_method == 'sparse_random':
            new_rows = WeightInitializer.sparse_random(
                n_output=n_new,
                n_input=n_input,
                sparsity=sparsity,
                scale=self.config.w_max * 0.2,
                device=device,
            )
        else:  # uniform
            new_rows = WeightInitializer.uniform(
                n_output=n_new,
                n_input=n_input,
                low=0.0,
                high=self.config.w_max * 0.2,
                device=device,
            )

        # Clamp to config bounds
        new_rows = clamp_weights(new_rows, self.config.w_min, self.config.w_max, inplace=False)

        # Concatenate and update parameter
        expanded = torch.cat([current_weights.data, new_rows], dim=0)
        setattr(self, weight_param_name, nn.Parameter(expanded))

    def _expand_weights_input(
        self,
        n_new: int,
        weight_param_name: str = 'weights',
        init_method: str = 'xavier',
        sparsity: float = 0.1,
    ) -> None:
        """Expand weight matrix with new input columns (accept more inputs).

        Helper method that encapsulates the common pattern:
        1. Get current weights
        2. Create new columns with specified initialization
        3. Concatenate and update parameter
        4. Clamp to bounds

        This consolidates ~30 lines per region × 6 regions = 180 lines
        (Architecture Review 2025-12-22, Tier 2.2).

        Args:
            n_new: Number of input dimensions to add
            weight_param_name: Name of weight parameter attribute (default: 'weights')
            init_method: Initialization method ('xavier', 'sparse_random', 'uniform')
            sparsity: Sparsity for sparse_random initialization

        Example:
            >>> def grow_input(self, n_new: int) -> None:
            ...     self._expand_weights_input(n_new, init_method='xavier')
            ...     self.config = replace(self.config, n_input=self.config.n_input + n_new)
        """
        current_weights = getattr(self, weight_param_name)
        n_output = current_weights.shape[0]
        device = current_weights.device

        # Create new columns
        if init_method == 'xavier':
            new_cols = WeightInitializer.xavier(
                n_output=n_output,
                n_input=n_new,
                gain=0.2,
                device=device,
            ) * self.config.w_max
        elif init_method == 'sparse_random':
            new_cols = WeightInitializer.sparse_random(
                n_output=n_output,
                n_input=n_new,
                sparsity=sparsity,
                scale=self.config.w_max * 0.2,
                device=device,
            )
        else:  # uniform
            new_cols = WeightInitializer.uniform(
                n_output=n_output,
                n_input=n_new,
                low=0.0,
                high=self.config.w_max * 0.2,
                device=device,
            )

        # Clamp to config bounds
        new_cols = clamp_weights(new_cols, self.config.w_min, self.config.w_max, inplace=False)

        # Concatenate and update parameter
        expanded = torch.cat([current_weights.data, new_cols], dim=1)
        setattr(self, weight_param_name, nn.Parameter(expanded))

    def _grow_weight_matrix_rows(
        self,
        old_weights: torch.Tensor,
        n_new_rows: int,
        initializer: str = "xavier",
        sparsity: float = 0.1,
    ) -> torch.Tensor:
        """Add new rows to weight matrix (grow output dimension).

        This is a functional version that returns a new tensor without
        mutating self. Useful for NeuralRegion's synaptic_weights dict pattern.

        Args:
            old_weights: Existing weight matrix [n_old, n_input]
            n_new_rows: Number of rows to add
            initializer: 'xavier', 'gaussian', or 'sparse_random'
            sparsity: Connection sparsity for sparse_random (0.0-1.0)

        Returns:
            New weight matrix [n_old + n_new_rows, n_input]

        Example:
            >>> # In NeuralRegion.grow_output()
            >>> for source_name, old_weights in self.synaptic_weights.items():
            ...     new_weights = self._grow_weight_matrix_rows(
            ...         old_weights, n_new, initializer="xavier"
            ...     )
            ...     self.synaptic_weights[source_name] = nn.Parameter(new_weights)
        """
        n_old, n_input = old_weights.shape
        n_new_total = n_old + n_new_rows
        device = old_weights.device

        # Create new matrix with appropriate initializer
        if initializer == "xavier":
            new_weights = WeightInitializer.xavier(
                n_new_total, n_input, gain=0.2, device=device
            ) * self.config.w_max
        elif initializer == "gaussian":
            new_weights = WeightInitializer.gaussian(
                n_new_total, n_input, mean=0.3, std=0.1, device=device
            )
        elif initializer == "sparse_random":
            new_weights = WeightInitializer.sparse_random(
                n_new_total, n_input, sparsity=sparsity, device=device
            )
        else:
            raise ValueError(f"Unknown initializer: {initializer}")

        # Preserve old weights in first n_old rows
        new_weights[:n_old, :] = old_weights

        # Clamp to config bounds
        return clamp_weights(new_weights, self.config.w_min, self.config.w_max, inplace=False)

    def _grow_weight_matrix_cols(
        self,
        old_weights: torch.Tensor,
        n_new_cols: int,
        initializer: str = "xavier",
        sparsity: float = 0.1,
    ) -> torch.Tensor:
        """Add new columns to weight matrix (grow input dimension).

        This is a functional version that returns a new tensor without
        mutating self. Useful for NeuralRegion's synaptic_weights dict pattern.

        Args:
            old_weights: Existing weight matrix [n_output, n_old]
            n_new_cols: Number of columns to add
            initializer: 'xavier', 'gaussian', or 'sparse_random'
            sparsity: Connection sparsity for sparse_random (0.0-1.0)

        Returns:
            New weight matrix [n_output, n_old + n_new_cols]

        Example:
            >>> # In NeuralRegion.grow_input()
            >>> for source_name, old_weights in self.synaptic_weights.items():
            ...     new_weights = self._grow_weight_matrix_cols(
            ...         old_weights, n_new, initializer="xavier"
            ...     )
            ...     self.synaptic_weights[source_name] = nn.Parameter(new_weights)
        """
        n_output, n_old = old_weights.shape
        n_new_total = n_old + n_new_cols
        device = old_weights.device

        # Create new matrix with appropriate initializer
        if initializer == "xavier":
            new_weights = WeightInitializer.xavier(
                n_output, n_new_total, gain=0.2, device=device
            ) * self.config.w_max
        elif initializer == "gaussian":
            new_weights = WeightInitializer.gaussian(
                n_output, n_new_total, mean=0.3, std=0.1, device=device
            )
        elif initializer == "sparse_random":
            new_weights = WeightInitializer.sparse_random(
                n_output, n_new_total, sparsity=sparsity, device=device
            )
        else:
            raise ValueError(f"Unknown initializer: {initializer}")

        # Preserve old weights in first n_old columns
        new_weights[:, :n_old] = old_weights

        # Clamp to config bounds
        return clamp_weights(new_weights, self.config.w_min, self.config.w_max, inplace=False)
