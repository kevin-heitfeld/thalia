"""Growth mixin for region neuron expansion.

This mixin provides utilities and template methods for adding neurons to brain
regions. It consolidates growth patterns that were duplicated across regions.

Design Philosophy:
- Simple regions (Prefrontal) can use the template method pattern
- Multi-layer regions (Hippocampus, LayeredCortex) use helper methods
- All regions benefit from standardized weight/state expansion utilities

Historical Context:
- Prior to this mixin, add_neurons() was duplicated across 4+ regions (~320 lines)
- Base class had _expand_weights() and _recreate_neurons_with_state() helpers
- This mixin consolidates all growth utilities and adds template method for simple cases
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from thalia.components.synapses.weight_init import WeightInitializer
from thalia.core.utils import clamp_weights


class GrowthMixin:
    """Mixin providing neuron growth capabilities for brain regions.

    This mixin supports two usage patterns:

    Pattern 1 - Template Method (for single-layer regions):
        Override _expand_layer_weights() and optionally _update_config_after_growth().
        The base add_neurons() handles orchestration.

        Example:
            class SimpleRegion(NeuralComponent, GrowthMixin):
                def _expand_layer_weights(self, n_new, initialization, **kwargs):
                    self.weights = self._expand_weights(self.weights, n_new, initialization)

                def _update_config_after_growth(self, new_n_output):
                    self.config = replace(self.config, n_output=new_n_output)

    Pattern 2 - Helper Methods (for multi-layer regions):
        Implement custom add_neurons() but use helper methods for weight expansion.

        Example:
            class MultiLayerRegion(NeuralComponent, GrowthMixin):
                def add_neurons(self, n_new, **kwargs):
                    # Custom multi-layer orchestration
                    for layer in self.layers:
                        layer.weights = self._expand_weights(layer.weights, growth)
                    self._update_all_configs(new_sizes)
                    self._recreate_all_neurons(old_sizes)

    Attributes:
        config: Region configuration with w_min/w_max and n_output
        device: PyTorch device for tensor creation
        neurons: Neuron population to recreate after growth
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
            >>> # In a region's add_neurons() method:
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
            >>> # In a region's add_neurons() method:
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

    def _recreate_neurons_with_state(
        self,
        neuron_factory,
        old_n_output: int,
    ) -> Any:
        """Recreate neuron population with larger size, preserving old state.

        This helper consolidates neuron recreation logic that was duplicated
        across regions. Handles state preservation for ConductanceLIF neurons.

        Args:
            neuron_factory: Callable that creates new neurons (e.g., self._create_neurons)
            old_n_output: Number of neurons before growth

        Returns:
            New neuron population with old state preserved in first old_n_output neurons

        Example:
            >>> # In a region's add_neurons() method:
            >>> self.neurons = self._recreate_neurons_with_state(
            ...     self._create_neurons,
            ...     old_n_output=self.config.n_output
            ... )
        """
        # Save old state
        old_state = {}
        if hasattr(self, 'neurons') and self.neurons is not None:
            if hasattr(self.neurons, 'membrane') and self.neurons.membrane is not None:
                old_state['membrane'] = self.neurons.membrane[:old_n_output].clone()
            if hasattr(self.neurons, 'g_E') and self.neurons.g_E is not None:
                old_state['g_E'] = self.neurons.g_E[:old_n_output].clone()
            if hasattr(self.neurons, 'g_I') and self.neurons.g_I is not None:
                old_state['g_I'] = self.neurons.g_I[:old_n_output].clone()
            if hasattr(self.neurons, 'refractory') and self.neurons.refractory is not None:
                old_state['refractory'] = self.neurons.refractory[:old_n_output].clone()
            if hasattr(self.neurons, 'adaptation') and self.neurons.adaptation is not None:
                old_state['adaptation'] = self.neurons.adaptation[:old_n_output].clone()

        # Create new neurons with larger size
        new_neurons = neuron_factory()
        new_neurons.reset_state()

        # Restore old state in first old_n_output positions
        for key, value in old_state.items():
            if hasattr(new_neurons, key):
                getattr(new_neurons, key)[:old_n_output] = value

        return new_neurons

    # =========================================================================
    # Template Method Pattern (for simple single-layer regions)
    # =========================================================================

    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Template method for adding neurons to single-layer regions.

        This provides standard orchestration for simple regions:
        1. Expand weights via _expand_layer_weights()
        2. Update config via _update_config_after_growth()
        3. Recreate neurons with state preservation
        4. Expand state tensors via _expand_state_tensors_after_growth()

        Multi-layer regions (Hippocampus, LayeredCortex) should override this
        entirely and use the helper methods instead.

        Args:
            n_new: Number of neurons to add
            initialization: Weight initialization strategy
            sparsity: Connection sparsity for sparse_random

        Note:
            Regions using this template must implement:
            - _expand_layer_weights()
            - _update_config_after_growth()
            - Optionally _expand_state_tensors_after_growth()
        """
        old_n_output = self.config.n_output
        new_n_output = old_n_output + n_new

        # Step 1: Expand weights (region-specific logic)
        self._expand_layer_weights(n_new, initialization, sparsity)

        # Step 2: Update configuration
        self._update_config_after_growth(new_n_output)

        # Step 3: Recreate neurons with state preservation
        if hasattr(self, 'neurons') and self.neurons is not None:
            self.neurons = self._recreate_neurons_with_state(
                self._create_neurons,
                old_n_output
            )

        # Step 4: Expand state tensors (if any)
        self._expand_state_tensors_after_growth(n_new)

    def _expand_layer_weights(
        self,
        n_new: int,
        initialization: str,
        sparsity: float,
    ) -> None:
        """Expand weight matrices for this region's layers.

        This is the primary extension point for the template method.
        Regions must implement their specific weight expansion logic here.

        Args:
            n_new: Number of neurons to add
            initialization: Weight initialization strategy
            sparsity: Connection sparsity for sparse_random

        Example (single weight matrix):
            self.weights = self._expand_weights(
                self.weights, n_new, initialization, sparsity
            )

        Example (multiple pathways):
            self.d1_pathway.weights = self._expand_weights(
                self.d1_pathway.weights, n_new * self.neurons_per_action, initialization
            )
            self.d2_pathway.weights = self._expand_weights(
                self.d2_pathway.weights, n_new * self.neurons_per_action, initialization
            )
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._expand_layer_weights() must be implemented "
            f"to use the GrowthMixin template method. "
            f"See src/thalia/mixins/growth_mixin.py for examples."
        )

    def _update_config_after_growth(self, new_n_output: int) -> None:
        """Update region configuration after neuron growth.

        This is called after weight expansion to update config objects.

        Args:
            new_n_output: Total number of neurons after growth

        Example:
            from dataclasses import replace
            self.config = replace(self.config, n_output=new_n_output)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._update_config_after_growth() must be implemented "
            f"to use the GrowthMixin template method."
        )

    def _expand_state_tensors_after_growth(self, n_new: int) -> None:
        """Expand state tensors after neuron growth.

        Override this if your region has state tensors that need expansion
        (e.g., traces, working memory, eligibility).

        Default implementation does nothing (no state tensors to expand).

        Args:
            n_new: Number of neurons added

        Example:
            expanded = self._expand_state_tensors({
                'output_trace': self.output_trace,
                'working_memory': self.working_memory,
            }, n_new)
            self.output_trace = expanded['output_trace']
            self.working_memory = expanded['working_memory']
        """
        pass
