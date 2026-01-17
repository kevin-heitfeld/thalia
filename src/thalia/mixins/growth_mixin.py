"""Growth mixin for region neuron expansion.

This mixin provides utility methods for expanding weight matrices and state tensors
during region growth. All regions implement their own grow_output() methods and use
ConductanceLIF.grow_neurons() for neuron population growth.

Utility Methods:
- _expand_weights(): Expand weight matrix by adding output neurons
- _expand_state_tensors(): Expand state tensors (traces, memory, etc.)
- _update_parameter(): Safely update Parameter data (Phase 1 improvement)
- _auto_grow_stp_modules(): Automatically grow all STP modules (Phase 1 improvement)
- _validate_growth(): Post-growth validation checks (Phase 1 improvement)

Historical Context:
- Prior to this mixin, grow_output() was duplicated across 4+ regions (~320 lines)
- This mixin consolidates weight and state expansion utilities
- As of Dec 2025: All neurons use ConductanceLIF with direct grow_neurons() support
- Jan 2026: Added Phase 1 validation helpers after fixing 18/18 growth tests

Growth Checklist (for implementers of grow_output):
1. ✅ Primary neurons - self.neurons.grow_neurons(n_new)
2. ✅ Weight matrices - Add rows for new neurons
3. ✅ STP modules - Call _auto_grow_stp_modules('post', n_new) or manual stp.grow(n_new, target='post')
4. ✅ State buffers - Expand activity tracking, gates, memory
5. ✅ Interneurons - FSI, TRN, or other inhibitory populations (if applicable)
6. ✅ Config updates - config.n_output, region-specific sizes
7. ✅ Subcomponents - Deep nuclei, granule layers, pathways (if applicable)

Growth Checklist (for implementers of grow_input):
1. ✅ Weight matrices - Add columns for new inputs
2. ✅ STP modules - Call _auto_grow_stp_modules('pre', n_new) or manual stp.grow(n_new, target='pre')
3. ✅ Input filters - Center-surround, preprocessing layers (if applicable)
4. ✅ Config updates - config.n_input
5. ❌ NO neuron growth - Input growth does NOT add neurons
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from thalia.components.synapses import WeightInitializer
from thalia.typing import StateDict
from thalia.utils.core_utils import clamp_weights


class GrowthMixin:
    """Mixin providing utility methods for region neuron growth.

    This mixin provides standardized helper methods for:
    1. Expanding weight matrices when adding output neurons
    2. Expanding state tensors (traces, memory, eligibility)
    3. (Phase 2) Opt-in registration for automatic subcomponent growth

    All regions implement their own grow_output() methods and call
    neurons.grow_neurons() directly for neuron population growth.

    Example Usage:
        class MyRegion(NeuralRegion, GrowthMixin):
            def __init__(self, config):
                super().__init__(config)
                # ... initialization ...

                # Phase 2: Opt-in registration (optional)
                self._register_stp('stp_recurrent', direction='both')
                self._register_subcomponent('fsi_neurons', ratio=0.2)

            def grow_output(self, n_new, initialization='xavier', sparsity=0.1):
                old_n_output = self.config.n_output

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

                # 5. Auto-grow registered components (Phase 2)
                self._auto_grow_registered_components('output', n_new)

                # 6. Validate
                self._validate_output_growth(old_n_output, n_new)

    Attributes:
        config: Region configuration with w_min/w_max and n_output
        device: PyTorch device for tensor creation
        neurons: ConductanceLIF neuron population
        _registered_stp: Dict[str, str] - STP name -> direction ('pre'|'post'|'both')
        _registered_subcomponents: Dict[str, tuple] - name -> (ratio, attr_name)
    """

    # Type stubs for attributes provided by mixed-in class
    # These will be overridden by the actual implementation
    config: Any  # Will be NeuralComponentConfig or subclass
    device: torch.device
    n_output: int
    neurons: Any  # Will be ConductanceLIF

    def named_modules(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Type stub for nn.Module.named_modules() - provided by mixed-in class."""
        raise NotImplementedError("This method should be provided by the mixed-in class")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize growth mixin with registration tracking.

        Note: This is automatically called via MRO when using multiple inheritance.
        """
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        # Phase 2: Component registration (opt-in)
        self._registered_stp: Dict[str, tuple[str, bool]] = {}
        self._registered_subcomponents: Dict[str, tuple[float, Optional[str]]] = {}

    def _expand_weights(
        self,
        current_weights: nn.Parameter,
        n_new: int,
        initialization: str = "xavier",
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
            from thalia.constants.architecture import GROWTH_NEW_WEIGHT_SCALE

            scale = self.config.w_max * GROWTH_NEW_WEIGHT_SCALE

        # Initialize new weights using specified strategy
        if initialization == "xavier":
            new_weights = (
                WeightInitializer.xavier(
                    n_output=n_new,
                    n_input=n_input,
                    gain=0.2,
                    device=device,
                )
                * self.config.w_max
            )
        elif initialization == "sparse_random":
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
        new_weights = clamp_weights(
            new_weights, self.config.w_min, self.config.w_max, inplace=False
        )

        # Concatenate with existing weights
        expanded = torch.cat([current_weights.data, new_weights], dim=0)
        return nn.Parameter(expanded)

    def _create_new_weights(
        self,
        n_output: int,
        n_input: int,
        initialization: str = "xavier",
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
        if initialization == "xavier":
            return WeightInitializer.xavier(n_output, n_input, device=self.device)
        elif initialization == "sparse_random":
            return WeightInitializer.sparse_random(n_output, n_input, sparsity, device=self.device)
        else:  # uniform
            return WeightInitializer.uniform(n_output, n_input, device=self.device)

    def _expand_state_tensors(self, state_dict: StateDict, n_new: int) -> StateDict:
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
        expanded: StateDict = {}
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
        weight_param_name: str = "weights",
        init_method: str = "xavier",
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
        if init_method == "xavier":
            new_rows = (
                WeightInitializer.xavier(
                    n_output=n_new,
                    n_input=n_input,
                    gain=0.2,
                    device=device,
                )
                * self.config.w_max
            )
        elif init_method == "sparse_random":
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
        weight_param_name: str = "weights",
        init_method: str = "xavier",
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
        if init_method == "xavier":
            new_cols = (
                WeightInitializer.xavier(
                    n_output=n_output,
                    n_input=n_new,
                    gain=0.2,
                    device=device,
                )
                * self.config.w_max
            )
        elif init_method == "sparse_random":
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
            new_weights = (
                WeightInitializer.xavier(n_new_total, n_input, gain=0.2, device=device)
                * self.config.w_max
            )
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
            new_weights = (
                WeightInitializer.xavier(n_output, n_new_total, gain=0.2, device=device)
                * self.config.w_max
            )
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

    # =========================================================================
    # Phase 2: Component Registration API (January 2026)
    # =========================================================================

    def _register_stp(
        self,
        stp_attr: str,
        direction: str = "post",
        recurrent: bool = False,
    ) -> None:
        """Register an STP module for automatic growth (opt-in, Phase 2).

        This allows regions to declare their STP modules in __init__() and have
        them automatically grown during grow_output() or grow_input().

        Args:
            stp_attr: Attribute name of the STP module (e.g., 'stp_recurrent')
            direction: When to grow:
                - 'pre': Grow only during grow_input (tracks external inputs)
                - 'post': Grow only during grow_output (tracks outputs)
                - 'both': Participates in both contexts (grow 'pre' during grow_input,
                         'post' during grow_output)
            recurrent: If True, during grow_output also grow 'pre' dimension
                      (for recurrent STP where pre and post are the same population).
                      Only meaningful when direction='post' or 'both'.

        Example:
            >>> # Non-recurrent STP that grows in both contexts:
            >>> self._register_stp('stp_sensory_relay', direction='both')
            >>>
            >>> # Recurrent STP (same population for pre/post):
            >>> self._register_stp('stp_l6_feedback', direction='post', recurrent=True)
            >>>
            >>> # In grow_output():
            >>> self._auto_grow_registered_components('output', n_new)
            >>> # Automatically grows registered STP modules

        Note:
            This is opt-in. Regions with complex STP routing (like Cerebellum,
            Hippocampus) can skip registration and grow STP modules manually.
        """
        if direction not in ("pre", "post", "both"):
            raise ValueError(f"Invalid direction '{direction}', must be 'pre', 'post', or 'both'")

        # Lazy initialization (mixin __init__ may not be called due to MRO)
        if not hasattr(self, "_registered_stp"):
            self._registered_stp = {}
        self._registered_stp[stp_attr] = (direction, recurrent)

    def _register_subcomponent(
        self,
        component_attr: str,
        ratio: float = 1.0,
        size_attr: Optional[str] = None,
    ) -> None:
        """Register a subcomponent for proportional growth (opt-in, Phase 2).

        This allows regions to declare subcomponents (like FSI neurons, TRN neurons)
        that should grow proportionally with the main population.

        Args:
            component_attr: Attribute name of the component (e.g., 'fsi_neurons')
            ratio: Growth ratio (0.0-1.0). If main grows by 10, component grows by ratio*10
            size_attr: Optional attribute tracking current size (e.g., 'fsi_size')

        Example:
            >>> # In region __init__():
            >>> self._register_subcomponent('fsi_neurons', ratio=0.2, size_attr='fsi_size')
            >>>
            >>> # In grow_output():
            >>> self._auto_grow_registered_components('output', n_new=10)
            >>> # Automatically: fsi_neurons.grow_neurons(2), fsi_size += 2

        Note:
            This handles simple proportional scaling. Complex cases (like Striatum's
            FSI with conditional weights) should use manual growth.
        """
        # Lazy initialization (mixin __init__ may not be called due to MRO)
        if not hasattr(self, "_registered_subcomponents"):
            self._registered_subcomponents = {}
        self._registered_subcomponents[component_attr] = (ratio, size_attr)

    def _auto_grow_registered_components(
        self,
        growth_type: str,
        n_new: int,
    ) -> Dict[str, int]:
        """Automatically grow all registered components (Phase 2).

        Called at the end of grow_output() or grow_input() to automatically
        grow STP modules and subcomponents that were registered in __init__().

        Args:
            growth_type: 'output' or 'input' - determines which STP direction to grow
            n_new: Number of neurons/inputs added

        Returns:
            Dict with counts: {'stp_grown': 2, 'subcomponents_grown': 1}

        Example:
            >>> # In grow_output():
            >>> self._auto_grow_registered_components('output', n_new)
            >>> # Grows all registered STP with direction='post' or 'both'
            >>> # Grows all registered subcomponents proportionally

        Note:
            This is a convenience wrapper. Regions can still use
            _auto_grow_stp_modules() and manual growth if preferred.
        """
        from thalia.components.synapses.stp import ShortTermPlasticity

        counts = {"stp_grown": 0, "subcomponents_grown": 0}

        # Lazy initialization (mixin __init__ may not be called due to MRO)
        if not hasattr(self, "_registered_stp"):
            self._registered_stp = {}
        if not hasattr(self, "_registered_subcomponents"):
            self._registered_subcomponents = {}

        # Grow registered STP modules
        for stp_attr, registration in self._registered_stp.items():
            if not hasattr(self, stp_attr):
                continue

            stp = getattr(self, stp_attr)
            if stp is None or not isinstance(stp, ShortTermPlasticity):
                continue

            # Unpack registration (handle legacy single value or new tuple)
            if isinstance(registration, tuple):
                direction, recurrent = registration
            else:
                direction = registration
                recurrent = direction == "both"  # Legacy: 'both' implied recurrent

            # Determine growth based on context and direction
            if growth_type == "output":
                # Always grow 'post' if direction is 'post' or 'both'
                if direction in ("post", "both"):
                    stp.grow(n_new, target="post")
                    counts["stp_grown"] += 1
                # Also grow 'pre' if recurrent
                if recurrent:
                    stp.grow(n_new, target="pre")
            elif growth_type == "input":
                # Grow 'pre' if direction is 'pre' or 'both'
                if direction in ("pre", "both"):
                    stp.grow(n_new, target="pre")
                    counts["stp_grown"] += 1

        # Grow registered subcomponents (only for output growth)
        if growth_type == "output":
            for component_attr, (ratio, size_attr) in self._registered_subcomponents.items():
                if not hasattr(self, component_attr):
                    continue

                component = getattr(self, component_attr)
                if component is None:
                    continue

                # Calculate proportional growth
                n_new_component = int(n_new * ratio)
                if n_new_component <= 0:
                    continue

                # Grow the component if it has grow_neurons method
                if hasattr(component, "grow_neurons"):
                    component.grow_neurons(n_new_component)
                    counts["subcomponents_grown"] += 1

                    # Update size attribute if provided
                    if size_attr and hasattr(self, size_attr):
                        old_size = getattr(self, size_attr)
                        setattr(self, size_attr, old_size + n_new_component)

        return counts

    # =========================================================================
    # Phase 1: Growth Improvements (January 2026)
    # =========================================================================

    def _update_parameter(self, name: str, new_data: torch.Tensor) -> None:
        """Safely update a Parameter's data (handles PyTorch registration correctly).

        **Problem**: Directly assigning `self.param = nn.Parameter(data)` raises
        KeyError if the parameter already exists in the module.

        **Solution**: Update the `.data` attribute directly for existing parameters.

        Args:
            name: Name of the parameter attribute (e.g., 'weights', 'bias')
            new_data: New tensor data to assign

        Example:
            >>> # WRONG - Raises KeyError if weights exists
            >>> self.weights = nn.Parameter(expanded_weights)

            >>> # CORRECT - Use this helper
            >>> self._update_parameter('weights', expanded_weights)

        Note:
            If the parameter doesn't exist yet, this creates it as a new Parameter.
        """
        if hasattr(self, name):
            # Parameter exists - update data in-place
            getattr(self, name).data = new_data
        else:
            # Parameter doesn't exist - register new one
            setattr(self, name, nn.Parameter(new_data))

    def _auto_grow_stp_modules(
        self,
        direction: str,
        n_new: int,
    ) -> int:
        """Automatically grow all ShortTermPlasticity modules in this region.

        **Problem**: Easy to forget growing STP modules when inputs/outputs change.
        6 regions × 2 methods × multiple STP types = many potential bugs.

        **Solution**: Auto-detect and grow all STP modules in one call.

        Args:
            direction: 'pre' for grow_input, 'post' for grow_output
            n_new: Number of neurons/inputs to add

        Returns:
            Number of STP modules grown

        Example:
            >>> # In grow_output():
            >>> def grow_output(self, n_new):
            ...     self.neurons.grow_neurons(n_new)
            ...     self._expand_weights(n_new)
            ...     self._auto_grow_stp_modules('post', n_new)  # Auto-handles all STP

            >>> # In grow_input():
            >>> def grow_input(self, n_new):
            ...     self._expand_input_weights(n_new)
            ...     self._auto_grow_stp_modules('pre', n_new)  # Auto-handles all STP

        Note:
            This uses PyTorch's module tree traversal to find all STP instances.
            For manual control, call stp.grow(n_new, target=direction) directly.
        """
        from thalia.components.synapses.stp import ShortTermPlasticity

        count = 0
        for name, module in self.named_modules():
            if isinstance(module, ShortTermPlasticity):
                module.grow(n_new, target=direction)
                count += 1

        return count

    def _validate_output_growth(
        self,
        old_n_output: int,
        n_new: int,
        check_neurons: bool = True,
        check_config: bool = True,
        check_state_buffers: bool = True,
    ) -> None:
        """Validate that grow_output() was performed correctly (post-growth checks).

        **Problem**: Easy to forget updating config, growing neurons, or expanding state buffers.

        **Solution**: Call this at the end of grow_output() for automatic validation.

        Args:
            old_n_output: Original n_output before growth
            n_new: Number of neurons that should have been added
            check_neurons: Validate neuron count increased correctly
            check_config: Validate config.n_output was updated
            check_state_buffers: Validate state buffers expanded (if they exist)

        Raises:
            AssertionError: If validation fails

        Example:
            >>> def grow_output(self, n_new):
            ...     old_n_output = self.config.n_output
            ...
            ...     # ... perform growth ...
            ...
            ...     # Validate at the end
            ...     self._validate_output_growth(old_n_output, n_new)

        Note:
            Some regions (like Striatum with population coding) may need custom
            validation. In those cases, pass check_neurons=False and validate manually.
        """
        expected_n_output = old_n_output + n_new

        # Check n_output updated (instance variable, not config - new pattern)
        if check_config:
            assert self.n_output == expected_n_output, (
                f"Growth validation failed: n_output not updated correctly. "
                f"Expected {expected_n_output}, got {self.n_output}"
            )

        # Check neurons grown (if applicable)
        if check_neurons and hasattr(self, "neurons"):
            actual_neurons = self.neurons.n_neurons
            assert actual_neurons == expected_n_output, (
                f"Growth validation failed: neurons not grown correctly. "
                f"Expected {expected_n_output} neurons, got {actual_neurons}"
            )

        # Check state buffers expanded (if region has state)
        if check_state_buffers:
            self._validate_state_buffer_sizes(expected_size=expected_n_output)

        # Validate STP sizes (if STP modules exist)
        self._validate_stp_sizes()

    def _validate_input_growth(
        self,
        old_n_input: int,
        n_new: int,
        check_config: bool = True,
    ) -> None:
        """Validate that grow_input() was performed correctly (post-growth checks).

        **Problem**: Easy to forget updating config when growing input dimension.

        **Solution**: Call this at the end of grow_input() for automatic validation.

        Args:
            old_n_input: Original n_input before growth
            n_new: Number of input neurons that should have been added
            check_config: Validate config.n_input was updated

        Raises:
            AssertionError: If validation fails

        Example:
            >>> def grow_input(self, n_new):
            ...     old_n_input = self.config.n_input
            ...
            ...     # ... perform growth ...
            ...
            ...     # Validate at the end
            ...     self._validate_input_growth(old_n_input, n_new)

        Note:
            Input growth does NOT add neurons (only expands weight columns),
            so we don't check neuron count here.
        """
        expected_n_input = old_n_input + n_new

        # Check instance variable updated (size is structural, not in config)
        if check_config:
            actual_n_input = getattr(self, "input_size", None)
            if actual_n_input is None:
                # Fallback: try config.n_input for regions that haven't migrated
                actual_n_input = getattr(self.config, "n_input", None)

            if actual_n_input is not None:
                assert actual_n_input == expected_n_input, (
                    f"Input growth validation failed: input size not updated correctly. "
                    f"Expected {expected_n_input}, got {actual_n_input}"
                )

        # Validate STP sizes (if STP modules exist)
        self._validate_stp_sizes()

    def _validate_state_buffer_sizes(self, expected_size: int) -> None:
        """Validate that state buffers match expected neuron count.

        Checks common state buffer patterns (working_memory, gates, traces, etc.).

        Args:
            expected_size: Expected size for state buffers (usually n_output)

        Raises:
            AssertionError: If any state buffer has incorrect size

        Example:
            >>> # Called automatically by _validate_output_growth
            >>> region.grow_output(10)
            >>> region._validate_state_buffer_sizes(region.config.n_output)
        """
        if not hasattr(self, "state") or self.state is None:
            return  # No state to validate

        # Check common state attributes
        state_attrs = [
            "working_memory",
            "update_gate",
            "active_rule",
            "recent_spikes",
            "eligibility_trace",
            "activity",
        ]

        for attr in state_attrs:
            if hasattr(self.state, attr):
                buffer = getattr(self.state, attr)
                if buffer is not None and isinstance(buffer, torch.Tensor):
                    # Check first dimension matches expected size
                    if buffer.dim() > 0 and buffer.shape[0] != expected_size:
                        raise AssertionError(
                            f"State buffer validation failed: state.{attr} has size {buffer.shape[0]}, "
                            f"expected {expected_size}"
                        )

    def _validate_stp_sizes(self) -> None:
        """Validate that all STP modules have correct pre/post sizes.

        This is called automatically by _validate_growth(), but can also be
        called manually for debugging.

        Raises:
            AssertionError: If any STP module has incorrect dimensions

        Example:
            >>> # Manual validation during debugging
            >>> region.grow_output(10)
            >>> region._validate_stp_sizes()  # Check all STP updated correctly
        """
        from thalia.components.synapses.stp import ShortTermPlasticity

        for name, module in self.named_modules():
            if isinstance(module, ShortTermPlasticity):
                # For regions, we can check if STP sizes match expected dimensions
                # This is a soft check - regions with complex STP may override
                if hasattr(self, "config"):
                    if hasattr(module, "n_post") and hasattr(self.config, "n_output"):
                        # Note: Some regions (like Striatum) have multiple STP modules
                        # with different sizes, so we just check n_post > 0
                        if module.n_post is not None:
                            assert (
                                module.n_post > 0
                            ), f"STP module '{name}' has n_post=0 after growth"
                    if hasattr(module, "n_pre") and hasattr(self.config, "n_input"):
                        if module.n_pre is not None:
                            assert module.n_pre > 0, f"STP module '{name}' has n_pre=0 after growth"
