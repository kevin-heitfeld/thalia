# pyright: strict
"""
Growth Mechanisms - Dynamic Capacity Expansion for Neural Circuits.

Supports adding neurons and synapses WITHOUT disrupting existing weights,
enabling curriculum learning with capacity expansion at stage transitions.

**Design Philosophy**:
======================
1. **CONSERVATIVE GROWTH**: Only add capacity when truly needed (not proactive)
2. **PRESERVE KNOWLEDGE**: New neurons/synapses don't interfere with trained circuits
3. **TRACK HISTORY**: Log all growth events for analysis and debugging
4. **CHECKPOINT COMPATIBLE**: Growth state fully serializable/restorable
5. **COORDINATED GROWTH**: Regions and pathways can grow together

**When to Grow**:
=================
- **Saturation**: >80% of neurons consistently firing (no capacity left)
- **Poor differentiation**: Neurons learning redundant features
- **Stage transitions**: New task complexity requires more capacity
- **Performance plateau**: Learning stalled despite continued training

**How Growth Preserves Knowledge**:
===================================
1. **New neurons start weak**: Low initial weights → minimal disruption
2. **Existing connections unchanged**: Keep trained circuit intact
3. **Gradual integration**: New capacity slowly joins existing network
4. **No catastrophic forgetting**: Old knowledge remains accessible

**Architecture Pattern**:
=========================
Both regions AND pathways implement growth:
- Regions: Add more neurons (increase population size)
- Pathways: Add more synapses (increase connectivity)
- Coordinated: Growing region → connected pathways also grow

**Author**: Thalia Team
**Date**: December 2025 (Updated from Dec 7, 2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from thalia.components.coding import compute_firing_rate


@dataclass
class GrowthEvent:
    """Record of a single growth operation."""

    timestamp: str
    component_name: str  # Region or pathway name
    component_type: str  # 'region' or 'pathway'
    event_type: str  # 'grow_output', 'grow_input', 'add_synapses'
    n_neurons_added: int = 0
    n_synapses_added: int = 0
    reason: str = ""
    metrics_before: Dict[str, float] = field(default_factory=lambda: {})
    metrics_after: Dict[str, float] = field(default_factory=lambda: {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "event_type": self.event_type,
            "n_neurons_added": self.n_neurons_added,
            "n_synapses_added": self.n_synapses_added,
            "reason": self.reason,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GrowthEvent:
        """Reconstruct from dict."""
        return cls(**data)


@dataclass
class CapacityMetrics:
    """Standardized capacity metrics for growth decisions.

    This dataclass provides a consistent interface for all brain components
    (regions and pathways) to report their capacity utilization. The growth
    manager uses these metrics to make informed decisions about when and how
    much to grow.

    Core Metrics (always present):
        utilization: Overall capacity utilization (0.0 to 1.0)
            Combines firing rate, weight saturation, and synapse usage
        total_neurons: Current number of neurons in the component
        active_neurons: Number of neurons that were recently active
        growth_recommended: Whether growth is advised
        growth_amount: Suggested number of neurons to add (if growing)

    Optional Detailed Metrics:
        firing_rate: Average firing rate (0.0 to 1.0)
        silence_fraction: Fraction of neurons that never fire (0.0 to 1.0)
        saturation_fraction: Fraction of weights near max value (0.0 to 1.0)
        synapse_usage: Fraction of synapses actively used (0.0 to 1.0)
        synapse_count: Total number of synapses
        growth_reason: Human-readable explanation for growth recommendation

    Usage:
        >>> metrics = region.get_capacity_metrics()
        >>> if metrics.growth_recommended:
        ...     region.grow_output(n_new=metrics.growth_amount)
    """

    # Core metrics (required)
    utilization: float  # Overall capacity utilization (0.0 to 1.0)
    total_neurons: int  # Current neuron count
    active_neurons: int  # Recently active neurons
    growth_recommended: bool = False  # Should we grow?
    growth_amount: int = 0  # Suggested neurons to add

    # Optional detailed metrics (for diagnostics)
    firing_rate: Optional[float] = None  # Average firing rate
    silence_fraction: Optional[float] = None  # Fraction never firing
    saturation_fraction: Optional[float] = None  # Weights near max
    synapse_usage: Optional[float] = None  # Active synapses fraction
    synapse_count: Optional[int] = None  # Total synapses
    growth_reason: str = ""  # Why growth is recommended

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for logging."""
        result = {
            "utilization": self.utilization,
            "total_neurons": float(self.total_neurons),
            "active_neurons": float(self.active_neurons),
            "growth_recommended": float(self.growth_recommended),
            "growth_amount": float(self.growth_amount),
        }

        # Add optional metrics if present
        if self.firing_rate is not None:
            result["firing_rate"] = self.firing_rate
        if self.silence_fraction is not None:
            result["silence_fraction"] = self.silence_fraction
        if self.saturation_fraction is not None:
            result["saturation_fraction"] = self.saturation_fraction
        if self.synapse_usage is not None:
            result["synapse_usage"] = self.synapse_usage
        if self.synapse_count is not None:
            result["synapse_count"] = float(self.synapse_count)

        return result


class GrowthManager:
    """Manages growth operations and history for a brain region.

    Responsibilities:
    - Coordinate neuron/synapse addition
    - Track growth history
    - Compute capacity metrics
    - Determine growth needs

    Usage:
        growth = GrowthManager(region_name="cortex")
        metrics = growth.get_capacity_metrics(region)

        if metrics.growth_recommended:
            growth.grow_component(region, n_new=100)
    """

    def __init__(self, region_name: str):
        """Initialize growth manager.

        Args:
            region_name: Name of region being managed
        """
        self.region_name = region_name
        self.history: List[GrowthEvent] = []

    def get_capacity_metrics(
        self,
        component: Any,  # Region or pathway (any component with weights)
        saturation_threshold: float = 0.9,
        usage_threshold: float = 0.1,
    ) -> CapacityMetrics:
        """Compute capacity utilization metrics.

        Args:
            component: Brain region or pathway to analyze
            saturation_threshold: Weight value considered "saturated"
            usage_threshold: Minimum weight magnitude to count as "used"

        Returns:
            CapacityMetrics with current utilization
        """
        # Get region properties
        n_neurons = (
            component.n_output if hasattr(component, "n_output") else 0
        )  # Estimate synapse count from weight matrices
        n_synapses = 0
        weight_saturation = 0.0
        synapse_usage = 0.0

        if hasattr(component, "weights"):
            # Single weight matrix
            w = component.weights
            n_synapses = w.numel()

            # Saturation: fraction near max weight value
            weight_max = w.abs().max()
            if weight_max > 0:
                weight_saturation = (
                    (w.abs() > saturation_threshold * weight_max).float().mean().item()
                )

            # Usage: fraction with significant magnitude
            synapse_usage = (w.abs() > usage_threshold).float().mean().item()

        elif hasattr(component, "_get_weight_tensors"):
            # Multiple weight matrices
            weight_tensors = component._get_weight_tensors()
            total_elements = 0
            saturated_count = 0
            used_count = 0

            for w in weight_tensors:
                total_elements += w.numel()
                weight_max = w.abs().max()
                if weight_max > 0:
                    saturated_count += (w.abs() > saturation_threshold * weight_max).sum().item()
                used_count += (w.abs() > usage_threshold).sum().item()

            n_synapses = total_elements
            weight_saturation = saturated_count / total_elements if total_elements > 0 else 0.0
            synapse_usage = used_count / total_elements if total_elements > 0 else 0.0

        # Estimate firing rate from recent activity
        firing_rate = 0.0
        if hasattr(component, "state") and hasattr(component.state, "spikes"):
            if component.state.spikes is not None:
                firing_rate = compute_firing_rate(component.state.spikes)

        # Count active neurons (those with spikes in current state)
        active_neurons = 0
        if hasattr(component, "state") and hasattr(component.state, "spikes"):
            if component.state.spikes is not None:
                active_neurons = (component.state.spikes > 0).sum().item()

        # Compute overall utilization (weighted combination)
        # Weight: 40% firing rate, 40% weight saturation, 20% synapse usage
        utilization = 0.4 * firing_rate + 0.4 * weight_saturation + 0.2 * synapse_usage

        # Determine if growth is recommended
        growth_recommended = False
        growth_reason = ""
        growth_amount = 0

        # Growth triggers (conservative thresholds)
        if weight_saturation > 0.85:
            growth_recommended = True
            growth_reason = f"Weight saturation high ({weight_saturation:.2f})"
            # Suggest 20% capacity increase
            growth_amount = max(10, int(n_neurons * 0.2))
        elif firing_rate > 0.9:
            growth_recommended = True
            growth_reason = f"Firing rate very high ({firing_rate:.2f})"
            growth_amount = max(10, int(n_neurons * 0.15))
        elif synapse_usage > 0.95:
            growth_recommended = True
            growth_reason = f"Synapse usage very high ({synapse_usage:.2f})"
            growth_amount = max(10, int(n_neurons * 0.15))

        return CapacityMetrics(
            utilization=utilization,
            total_neurons=n_neurons,
            active_neurons=active_neurons,
            growth_recommended=growth_recommended,
            growth_amount=growth_amount,
            # Optional detailed metrics
            firing_rate=firing_rate,
            saturation_fraction=weight_saturation,
            synapse_usage=synapse_usage,
            synapse_count=n_synapses,
            growth_reason=growth_reason,
        )

    def grow_component(
        self,
        component: Any,  # Region or pathway (any component with grow_output method)
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
        reason: str = "",
        component_type: str = "region",
    ) -> GrowthEvent:
        """Grow component output dimension without disrupting existing weights.

        Strategy:
        1. Measure capacity before growth
        2. Call component.grow_output() to expand weights
        3. Measure capacity after growth
        4. Record growth event in history

        Args:
            component: Brain region or pathway to grow
            n_new: Number of neurons to add
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
            reason: Human-readable reason for growth
            component_type: 'region' or 'pathway'

        Returns:
            GrowthEvent record
        """
        # Get metrics before growth
        metrics_before = self.get_capacity_metrics(component)

        # Perform growth (component-specific implementation)
        if not hasattr(component, "grow_output"):
            raise NotImplementedError(
                f"Component {self.region_name} does not implement grow_output()"
            )

        # Striatum grows by actions, not individual neurons
        # Convert neuron count to action count if needed
        actual_n_new = n_new
        if hasattr(component, "neurons_per_action") and component.neurons_per_action > 1:
            # Striatum: n_new should be number of actions, not neurons
            # Round to nearest action population
            actual_n_new = max(1, round(n_new / component.neurons_per_action))

        component.grow_output(
            n_new=actual_n_new,
            initialization=initialization,
            sparsity=sparsity,
        )

        # Get metrics after growth
        metrics_after = self.get_capacity_metrics(component)

        # Create growth event
        synapse_count_before = metrics_before.synapse_count or 0
        synapse_count_after = metrics_after.synapse_count or 0
        event = GrowthEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            component_name=self.region_name,
            component_type=component_type,
            event_type="grow_output",
            n_neurons_added=n_new,
            n_synapses_added=synapse_count_after - synapse_count_before,
            reason=reason or metrics_before.growth_reason,
            metrics_before=metrics_before.to_dict(),
            metrics_after=metrics_after.to_dict(),
        )

        # Record in history
        self.history.append(event)

        return event

    def get_history(self) -> List[Dict[str, Any]]:
        """Get growth history as list of dicts (for checkpointing).

        Returns:
            List of growth event dicts
        """
        return [event.to_dict() for event in self.history]

    def load_history(self, history_data: List[Dict[str, Any]]) -> None:
        """Load growth history from checkpoint.

        Args:
            history_data: List of growth event dicts
        """
        self.history = [GrowthEvent.from_dict(data) for data in history_data]

    def get_state(self) -> Dict[str, Any]:
        """Get growth manager state for checkpointing.

        Returns:
            State dict with region_name and history
        """
        return {
            "region_name": self.region_name,
            "history": self.get_history(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load growth manager state from checkpoint.

        Args:
            state: State dict from get_state()
        """
        self.region_name = state["region_name"]
        self.load_history(state.get("history", []))


# =============================================================================
# Growth Coordination - Synchronized Region and Pathway Growth
# =============================================================================


class GrowthCoordinator:
    """Coordinates growth across multiple brain regions and their connected pathways.

    When a region grows, its input and output pathways must also grow to maintain
    connectivity. This coordinator ensures synchronized, coordinated growth to
    prevent dimensional mismatches.

    Responsibilities:
    1. Identify all pathways connected to a growing region
    2. Grow pathways to match new region dimensions
    3. Log coordinated growth events
    4. Prevent broken connectivity after growth

    Example:
        >>> coordinator = GrowthCoordinator(brain)
        >>>
        >>> # Grow cortex by 100 neurons
        >>> events = coordinator.coordinate_growth(
        ...     region_name='cortex',
        ...     n_new_neurons=100,
        ...     reason="Curriculum stage transition"
        ... )
        >>>
        >>> # events contains:
        >>> # - cortex growth event
        >>> # - visual_to_cortex pathway growth event (input pathway)
        >>> # - cortex_to_hippocampus pathway growth event (output pathway)

    Biological Justification:
        In biology, neurogenesis is coupled with synaptogenesis:
        - New neurons form connections with existing neurons
        - New dendrites and axons extend to integrate new capacity
        - Growth is coordinated at circuit level, not just local
    """

    def __init__(self, brain: Any):
        """Initialize growth coordinator.

        Args:
            brain: DynamicBrain instance
        """
        self.brain = brain
        self.pathway_manager = brain.pathway_manager
        self.history: List[Dict[str, Any]] = []

    def coordinate_growth(
        self,
        region_name: str,
        n_new_neurons: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
        reason: str = "",
    ) -> List[GrowthEvent]:
        """Grow a region and all its connected pathways.

        This is the main coordination method. It:
        1. Grows the specified region
        2. Identifies all pathways connected to that region
        3. Grows input pathways (pre-synaptic connections)
        4. Grows output pathways (post-synaptic connections)
        5. Returns all growth events

        Args:
            region_name: Name of region to grow
            n_new_neurons: Number of neurons to add to region
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
            reason: Human-readable reason for growth

        Returns:
            List of GrowthEvent records (region + pathways)

        Raises:
            KeyError: If region_name not found in brain
            AttributeError: If region doesn't support growth
        """
        if region_name not in self.brain.components:
            raise KeyError(f"Region '{region_name}' not found in brain")

        region = self.brain.components[region_name]
        events = []

        # 1. Grow the region itself
        if hasattr(region, "grow_output"):
            # Call grow_output() directly on region
            region.grow_output(
                n_new=n_new_neurons,
                initialization=initialization,
                sparsity=sparsity,
            )

            # Create growth event manually
            region_event = GrowthEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                component_name=region_name,
                component_type="region",
                event_type="grow_output",
                n_neurons_added=n_new_neurons,
                n_synapses_added=0,  # Estimated from pathways
                reason=reason,
                metrics_before={},
                metrics_after={},
            )
            events.append(region_event)
        else:
            raise AttributeError(
                f"Region '{region_name}' does not have grow_output() method. "
                f"Growth not supported for this region type."
            )

        # 2. Identify connected pathways
        input_pathways = self._find_input_pathways(region_name)
        output_pathways = self._find_output_pathways(region_name)

        # 3. Grow input pathways (pre-synaptic → new post-synaptic)
        # These pathways send spikes TO the growing region
        # Need to add neurons to target side to match region growth
        # events list already created above, append to it
        for pathway_name, pathway in input_pathways:
            if hasattr(pathway, "grow_output"):
                # Skip routing pathways (AxonalProjection) - they have no learnable weights
                # v3.0 architecture: routing pathways just transmit spikes, regions handle learning
                has_learnable_params = any(p.requires_grad for p in pathway.parameters())
                if not has_learnable_params:
                    continue  # Skip routing pathways

                # Get source size before growth to calculate synapses added
                n_source = pathway.config.n_input if hasattr(pathway, "config") else 0

                pathway.grow_output(
                    n_new=n_new_neurons,
                    initialization=initialization,
                    sparsity=sparsity,
                )

                # Calculate synapses added: new target neurons × source neurons
                # Each new target neuron gets n_source incoming synapses
                n_synapses_added = n_new_neurons * n_source

                # Record pathway growth event
                pathway_event = GrowthEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    component_name=pathway_name,
                    component_type="pathway",
                    event_type="grow_output",
                    n_neurons_added=n_new_neurons,
                    n_synapses_added=n_synapses_added,
                    reason=f"Input pathway to {region_name}",
                    metrics_before={},
                    metrics_after={},
                )
                events.append(pathway_event)

        # 4. Grow output pathways (new pre-synaptic → existing post-synaptic)
        # These pathways receive spikes FROM the growing region
        # Need to add neurons to source side to match region growth
        for pathway_name, pathway in output_pathways:
            if hasattr(pathway, "grow_input"):
                # Skip routing pathways (AxonalProjection) - they have no learnable weights
                # v3.0 architecture: routing pathways just transmit spikes, regions handle learning
                has_learnable_params = any(p.requires_grad for p in pathway.parameters())
                if not has_learnable_params:
                    continue  # Skip routing pathways

                # Get target size before growth to calculate synapses added
                n_target = pathway.config.n_output if hasattr(pathway, "config") else 0

                pathway.grow_input(
                    n_new=n_new_neurons,
                    initialization=initialization,
                    sparsity=sparsity,
                )

                # Calculate synapses added: new source neurons × target neurons
                # Each target neuron gets n_new additional incoming synapses
                n_synapses_added = n_new_neurons * n_target

                # Record pathway growth event
                pathway_event = GrowthEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    component_name=pathway_name,
                    component_type="pathway",
                    event_type="grow_input",
                    n_neurons_added=n_new_neurons,
                    n_synapses_added=n_synapses_added,
                    reason=f"Output pathway from {region_name}",
                    metrics_before={},
                    metrics_after={},
                )
                events.append(pathway_event)

                # ============================================================
                # CRITICAL: Grow target region's input for this source!
                # ============================================================
                # Multi-source architecture (v3.0): Target regions have separate
                # weight matrices per source. When a source region grows, the
                # target must expand its weights for that specific source.
                target_region_name = self._get_pathway_target(pathway_name)
                if target_region_name and target_region_name in self.brain.components:
                    target_region = self.brain.components[target_region_name]
                    if hasattr(target_region, "grow_source"):
                        # Get new total size of source region
                        source_region = self.brain.components[region_name]
                        new_source_size = source_region.n_output

                        # Grow target's weights for this source
                        # Note: source_name might include port (e.g., "cortex:l5")
                        # For now, use region_name; port-based routing handled elsewhere
                        target_region.grow_source(
                            source_name=region_name,
                            new_size=new_source_size,
                            initialization=initialization,
                            sparsity=sparsity,
                        )

                        # Record region input growth event
                        region_input_event = GrowthEvent(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            component_name=target_region_name,
                            component_type="region",
                            event_type="grow_source",
                            n_neurons_added=0,  # No new neurons, just input expansion
                            n_synapses_added=n_new_neurons * target_region.n_output,
                            reason=f"Source '{region_name}' expanded by {n_new_neurons}",
                            metrics_before={},
                            metrics_after={},
                        )
                        events.append(region_input_event)

        # 5. Record coordinated growth in history
        coordinated_event = {
            "timestamp": datetime.now().isoformat(),
            "region": region_name,
            "n_neurons_added": n_new_neurons,
            "reason": reason,
            "events": [e.to_dict() for e in events],
        }
        self.history.append(coordinated_event)

        return events

    def _find_input_pathways(self, region_name: str) -> List[tuple[str, Any]]:
        """Find all pathways that send spikes TO this region.

        Args:
            region_name: Target region name

        Returns:
            List of (pathway_name, pathway) tuples
        """
        input_pathways: list[tuple[str, Any]] = []
        for name, pathway in self.pathway_manager.get_all_pathways().items():
            # Parse pathway name: "source_to_target" or special names
            if "_to_" in name:
                parts = name.split("_to_")
                if len(parts) == 2:
                    target = parts[1]
                    if target == region_name:
                        input_pathways.append((name, pathway))
        return input_pathways

    def _find_output_pathways(self, region_name: str) -> List[tuple[str, Any]]:
        """Find all pathways that receive spikes FROM this region.

        Args:
            region_name: Source region name

        Returns:
            List of (pathway_name, pathway) tuples
        """
        output_pathways: list[tuple[str, Any]] = []
        for name, pathway in self.pathway_manager.get_all_pathways().items():
            # Parse pathway name: "source_to_target"
            if "_to_" in name:
                parts = name.split("_to_")
                if len(parts) == 2:
                    source = parts[0]
                    if source == region_name:
                        output_pathways.append((name, pathway))
        return output_pathways

    def _get_pathway_target(self, pathway_name: str) -> Optional[str]:
        """Extract target region name from pathway name.

        Args:
            pathway_name: Pathway name in format "source_to_target"

        Returns:
            Target region name, or None if not parseable
        """
        if "_to_" in pathway_name:
            parts = pathway_name.split("_to_")
            if len(parts) == 2:
                return parts[1]
        return None

    def get_growth_history(self) -> List[Dict[str, Any]]:
        """Get coordinated growth history.

        Returns:
            List of coordinated growth events
        """
        return self.history

    def get_state(self) -> Dict[str, Any]:
        """Get coordinator state for checkpointing.

        Returns:
            State dict with growth history
        """
        return {
            "history": self.history,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load coordinator state from checkpoint.

        Args:
            state: State dict from get_state()
        """
        self.history = state.get("history", [])
