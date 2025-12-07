"""Growth mechanisms for Thalia brain regions and pathways.

Supports adding neurons and synapses without disrupting existing weights,
enabling curriculum learning with capacity expansion.

Design Philosophy:
- CONSERVATIVE growth: Only add capacity when truly needed
- PRESERVE existing knowledge: New neurons/synapses don't interfere with trained circuits
- TRACK history: Log all growth events for analysis and debugging
- CHECKPOINT compatible: Growth state serializable/restorable
- COORDINATED growth: Regions and pathways can grow together

Author: Thalia Team
Date: December 7, 2025
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class GrowthEvent:
    """Record of a single growth operation."""

    timestamp: str
    component_name: str  # Region or pathway name
    component_type: str  # 'region' or 'pathway'
    event_type: str  # 'add_neurons', 'add_synapses'
    n_neurons_added: int = 0
    n_synapses_added: int = 0
    reason: str = ""
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)

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
    def from_dict(cls, data: Dict[str, Any]) -> 'GrowthEvent':
        """Reconstruct from dict."""
        return cls(**data)


@dataclass
class CapacityMetrics:
    """Metrics for determining if region needs growth."""

    firing_rate: float  # Average firing rate (0-1)
    weight_saturation: float  # Fraction of weights near max (0-1)
    synapse_usage: float  # Fraction of synapses actively used (0-1)
    neuron_count: int
    synapse_count: int
    growth_recommended: bool = False
    growth_reason: str = ""

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for logging."""
        return {
            "firing_rate": self.firing_rate,
            "weight_saturation": self.weight_saturation,
            "synapse_usage": self.synapse_usage,
            "neuron_count": float(self.neuron_count),
            "synapse_count": float(self.synapse_count),
            "growth_recommended": float(self.growth_recommended),
        }


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
            growth.add_neurons(region, n_new=100)
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
        component: Any,  # BrainRegion or Pathway
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
        n_neurons = component.n_output if hasattr(component, 'n_output') else 0        # Estimate synapse count from weight matrices
        n_synapses = 0
        weight_saturation = 0.0
        synapse_usage = 0.0

        if hasattr(component, 'weights'):
            # Single weight matrix
            w = component.weights
            n_synapses = w.numel()

            # Saturation: fraction near max weight value
            weight_max = w.abs().max()
            if weight_max > 0:
                weight_saturation = (w.abs() > saturation_threshold * weight_max).float().mean().item()

            # Usage: fraction with significant magnitude
            synapse_usage = (w.abs() > usage_threshold).float().mean().item()

        elif hasattr(component, '_get_weight_tensors'):
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
        if hasattr(component, 'state') and hasattr(component.state, 'spikes'):
            if component.state.spikes is not None:
                firing_rate = component.state.spikes.float().mean().item()

        # Determine if growth is recommended
        growth_recommended = False
        growth_reason = ""

        # Growth triggers (conservative thresholds)
        if weight_saturation > 0.85:
            growth_recommended = True
            growth_reason = f"Weight saturation high ({weight_saturation:.2f})"
        elif firing_rate > 0.9:
            growth_recommended = True
            growth_reason = f"Firing rate very high ({firing_rate:.2f})"
        elif synapse_usage > 0.95:
            growth_recommended = True
            growth_reason = f"Synapse usage very high ({synapse_usage:.2f})"

        return CapacityMetrics(
            firing_rate=firing_rate,
            weight_saturation=weight_saturation,
            synapse_usage=synapse_usage,
            neuron_count=n_neurons,
            synapse_count=n_synapses,
            growth_recommended=growth_recommended,
            growth_reason=growth_reason,
        )

    def add_neurons(
        self,
        component: Any,  # BrainRegion or Pathway
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
        reason: str = "",
        component_type: str = "region",
    ) -> GrowthEvent:
        """Add neurons to component (region or pathway) without disrupting existing weights.

        Strategy:
        1. Measure capacity before growth
        2. Call component.add_neurons() to expand weights
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
        if not hasattr(component, 'add_neurons'):
            raise NotImplementedError(
                f"Component {self.region_name} does not implement add_neurons()"
            )

        component.add_neurons(
            n_new=n_new,
            initialization=initialization,
            sparsity=sparsity,
        )

        # Get metrics after growth
        metrics_after = self.get_capacity_metrics(component)

        # Create growth event
        from datetime import timezone
        event = GrowthEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            component_name=self.region_name,
            component_type=component_type,
            event_type='add_neurons',
            n_neurons_added=n_new,
            n_synapses_added=metrics_after.synapse_count - metrics_before.synapse_count,
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
