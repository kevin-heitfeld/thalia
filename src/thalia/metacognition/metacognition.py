"""
Metacognition - Thinking about thinking.

This module implements the network's capacity for self-monitoring,
enabling it to:

- Track confidence in its own processing
- Estimate uncertainty in predictions and beliefs
- Detect errors and conflicts in reasoning
- Monitor cognitive load and resource usage
- Adjust processing strategies based on self-assessment

Metacognition is based on higher-order theories of consciousness
and enables the network to have insight into its own operations.

Key components:
- ConfidenceTracker: Tracks confidence in processing outcomes
- UncertaintyEstimator: Estimates uncertainty in predictions
- ErrorDetector: Detects conflicts and errors in processing
- CognitiveMonitor: Monitors overall cognitive state
- MetacognitiveController: Adjusts processing based on monitoring

References:
- Flavell (1979) - Metacognition and Cognitive Monitoring
- Nelson & Narens (1990) - Metamemory
- Frith (2012) - The Cognitive Neuropsychology of Schizophrenia
- Fleming & Dolan (2012) - The Neural Basis of Metacognitive Ability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable
from enum import Enum, auto
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.core.neuron import LIFNeuron, LIFConfig


class ConfidenceLevel(Enum):
    """Discrete confidence levels."""
    VERY_LOW = auto()    # < 0.2
    LOW = auto()         # 0.2 - 0.4
    MEDIUM = auto()      # 0.4 - 0.6
    HIGH = auto()        # 0.6 - 0.8
    VERY_HIGH = auto()   # > 0.8

    @classmethod
    def from_value(cls, value: float) -> "ConfidenceLevel":
        """Convert numeric confidence to level."""
        if value < 0.2:
            return cls.VERY_LOW
        elif value < 0.4:
            return cls.LOW
        elif value < 0.6:
            return cls.MEDIUM
        elif value < 0.8:
            return cls.HIGH
        else:
            return cls.VERY_HIGH


class ErrorType(Enum):
    """Types of errors that can be detected."""
    PREDICTION_ERROR = auto()   # Prediction didn't match reality
    CONFLICT = auto()           # Conflicting information/beliefs
    INCONSISTENCY = auto()      # Inconsistent reasoning
    TIMEOUT = auto()            # Processing took too long
    OVERLOAD = auto()           # Cognitive overload
    UNCERTAINTY = auto()        # High uncertainty detected


@dataclass
class ConfidenceEstimate:
    """An estimate of confidence with metadata.

    Attributes:
        value: Confidence value between 0 and 1
        level: Discrete confidence level
        source: What generated this estimate
        timestamp: When this was computed
        evidence: Supporting evidence strength
    """
    value: float
    level: ConfidenceLevel
    source: str = "unknown"
    timestamp: int = 0
    evidence: float = 1.0

    def __post_init__(self):
        self.value = max(0.0, min(1.0, self.value))
        self.level = ConfidenceLevel.from_value(self.value)


@dataclass
class ErrorSignal:
    """A detected error or conflict.

    Attributes:
        error_type: Type of error
        magnitude: How severe (0-1)
        location: Where the error occurred
        timestamp: When detected
        details: Additional information
    """
    error_type: ErrorType
    magnitude: float
    location: str = "unknown"
    timestamp: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.magnitude = max(0.0, min(1.0, self.magnitude))


@dataclass
class CognitiveState:
    """Current state of the cognitive system.

    Attributes:
        load: Current cognitive load (0-1)
        confidence: Overall confidence
        uncertainty: Overall uncertainty
        active_errors: List of active errors
        processing_mode: Current mode of processing
    """
    load: float = 0.0
    confidence: float = 0.5
    uncertainty: float = 0.5
    active_errors: List[ErrorSignal] = field(default_factory=list)
    processing_mode: str = "normal"

    def is_overloaded(self, threshold: float = 0.8) -> bool:
        """Check if system is overloaded."""
        return self.load > threshold

    def is_confident(self, threshold: float = 0.6) -> bool:
        """Check if system is confident."""
        return self.confidence > threshold

    def has_errors(self) -> bool:
        """Check if there are active errors."""
        return len(self.active_errors) > 0


@dataclass
class MetacognitiveConfig:
    """Configuration for metacognitive system.

    Attributes:
        hidden_dim: Hidden layer dimension
        confidence_decay: How fast confidence decays without evidence
        error_threshold: Threshold for error detection
        uncertainty_window: Window for uncertainty estimation
        load_capacity: Maximum cognitive load
        tau_mem: Membrane time constant
    """
    hidden_dim: int = 64
    confidence_decay: float = 0.95
    error_threshold: float = 0.3
    uncertainty_window: int = 10
    load_capacity: float = 1.0
    tau_mem: float = 20.0


class ConfidenceTracker(nn.Module):
    """Tracks confidence in processing outcomes.

    Monitors the reliability of network outputs and maintains
    running estimates of confidence levels.

    Example:
        >>> tracker = ConfidenceTracker(input_dim=128)
        >>> confidence = tracker.estimate(activity)
        >>> print(f"Confidence: {confidence.value:.2f}")
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        decay: float = 0.95,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decay = decay

        # Confidence estimation network
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Running confidence estimate
        self.register_buffer("running_confidence", torch.tensor(0.5))
        self.register_buffer("evidence_count", torch.tensor(0.0))

        # History
        self._history: List[ConfidenceEstimate] = []
        self._timestep = 0

    def reset(self) -> None:
        """Reset tracker state."""
        self.running_confidence.fill_(0.5)
        self.evidence_count.fill_(0.0)
        self._history = []
        self._timestep = 0

    def estimate(
        self,
        activity: torch.Tensor,
        source: str = "activity",
    ) -> ConfidenceEstimate:
        """Estimate confidence from activity pattern.

        Args:
            activity: Neural activity pattern
            source: Source of this estimate

        Returns:
            Confidence estimate
        """
        # Flatten if needed
        if activity.dim() > 2:
            activity = activity.view(activity.shape[0], -1)
        if activity.dim() == 1:
            activity = activity.unsqueeze(0)

        # Estimate confidence
        with torch.no_grad():
            raw_confidence = self.confidence_net(activity)
            confidence_value = raw_confidence.mean().item()

        # Update running estimate
        self.evidence_count += 1
        alpha = 1.0 / self.evidence_count.item()
        self.running_confidence = (
            (1 - alpha) * self.running_confidence +
            alpha * confidence_value
        )

        # Create estimate
        estimate = ConfidenceEstimate(
            value=confidence_value,
            level=ConfidenceLevel.from_value(confidence_value),
            source=source,
            timestamp=self._timestep,
            evidence=self.evidence_count.item(),
        )

        self._history.append(estimate)
        self._timestep += 1

        return estimate

    def estimate_from_consistency(
        self,
        patterns: List[torch.Tensor],
    ) -> ConfidenceEstimate:
        """Estimate confidence from pattern consistency.

        Higher consistency = higher confidence.

        Args:
            patterns: List of activity patterns

        Returns:
            Confidence estimate based on consistency
        """
        if len(patterns) < 2:
            return ConfidenceEstimate(
                value=0.5,
                level=ConfidenceLevel.MEDIUM,
                source="consistency",
                timestamp=self._timestep,
            )

        # Compute pairwise similarities
        similarities = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                p1 = patterns[i].flatten()
                p2 = patterns[j].flatten()
                sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
                similarities.append(sim.item())

        # Mean similarity as confidence
        mean_sim = sum(similarities) / len(similarities)
        confidence = (mean_sim + 1) / 2  # Map from [-1, 1] to [0, 1]

        estimate = ConfidenceEstimate(
            value=confidence,
            level=ConfidenceLevel.from_value(confidence),
            source="consistency",
            timestamp=self._timestep,
        )

        self._history.append(estimate)
        self._timestep += 1

        return estimate

    def decay_confidence(self) -> float:
        """Apply decay to running confidence."""
        self.running_confidence *= self.decay
        return self.running_confidence.item()

    def get_running_confidence(self) -> float:
        """Get current running confidence."""
        return self.running_confidence.item()

    def get_history(self) -> List[ConfidenceEstimate]:
        """Get confidence history."""
        return list(self._history)


class UncertaintyEstimator(nn.Module):
    """Estimates uncertainty in predictions and beliefs.

    Uses ensemble-based and entropy-based methods to
    quantify uncertainty in network outputs.

    Example:
        >>> estimator = UncertaintyEstimator(input_dim=128)
        >>> uncertainty = estimator.estimate(prediction)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_samples: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples

        # Uncertainty estimation with dropout for MC sampling
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

        # Running statistics
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.register_buffer("sample_count", torch.tensor(0.0))

    def reset(self) -> None:
        """Reset estimator state."""
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.sample_count.fill_(0.0)

    def estimate_epistemic(
        self,
        input_pattern: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """Estimate epistemic uncertainty using MC dropout.

        Epistemic uncertainty comes from lack of knowledge
        and can be reduced with more data.

        Args:
            input_pattern: Input pattern

        Returns:
            (scalar uncertainty, per-dimension uncertainty)
        """
        if input_pattern.dim() == 1:
            input_pattern = input_pattern.unsqueeze(0)

        # MC sampling with dropout
        self.uncertainty_net.train()  # Enable dropout
        samples = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                output = self.uncertainty_net(input_pattern)
                samples.append(output)

        self.uncertainty_net.eval()

        # Stack samples
        samples = torch.stack(samples, dim=0)  # (n_samples, batch, dim)

        # Variance across samples = epistemic uncertainty
        variance = samples.var(dim=0)

        # Scalar uncertainty
        scalar_uncertainty = variance.mean().item()

        return scalar_uncertainty, variance.squeeze(0)

    def estimate_aleatoric(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> float:
        """Estimate aleatoric uncertainty from prediction distribution.

        Aleatoric uncertainty comes from inherent noise in data
        and cannot be reduced with more data.

        Args:
            predictions: Model predictions
            targets: Optional ground truth

        Returns:
            Aleatoric uncertainty estimate
        """
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)

        # Use entropy as uncertainty measure
        # Normalize to probability-like values
        probs = F.softmax(predictions, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # Normalize by max entropy
        max_entropy = math.log(predictions.shape[-1])
        normalized_entropy = entropy / max_entropy

        return normalized_entropy.mean().item()

    def estimate_total(
        self,
        input_pattern: torch.Tensor,
    ) -> Dict[str, float]:
        """Estimate total uncertainty (epistemic + aleatoric).

        Args:
            input_pattern: Input pattern

        Returns:
            Dictionary with uncertainty components
        """
        epistemic, _ = self.estimate_epistemic(input_pattern)
        aleatoric = self.estimate_aleatoric(input_pattern)

        return {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": epistemic + aleatoric,
        }

    def update_statistics(self, pattern: torch.Tensor) -> None:
        """Update running statistics with new pattern."""
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)

        batch_mean = pattern.mean(dim=0)
        # Handle single sample case - use zeros for variance
        if pattern.shape[0] > 1:
            batch_var = pattern.var(dim=0)
        else:
            batch_var = torch.zeros_like(batch_mean)

        # Welford's online algorithm
        self.sample_count += 1
        delta = batch_mean - self.running_mean
        self.running_mean += delta / self.sample_count
        delta2 = batch_mean - self.running_mean
        self.running_var += delta * delta2

    def get_novelty(self, pattern: torch.Tensor) -> float:
        """Estimate novelty of pattern based on running statistics.

        Novel patterns that deviate from the mean have higher uncertainty.
        """
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)

        if self.sample_count < 2:
            return 0.5  # No basis for comparison

        # Compute Mahalanobis-like distance
        var = self.running_var / (self.sample_count - 1)
        var = var.clamp(min=1e-6)

        diff = pattern - self.running_mean
        distance = (diff ** 2 / var).mean().item()

        # Convert to 0-1 scale
        novelty = 1 - math.exp(-distance)

        return novelty


class ErrorDetector(nn.Module):
    """Detects errors and conflicts in processing.

    Monitors for prediction errors, logical conflicts,
    and processing anomalies.

    Example:
        >>> detector = ErrorDetector(input_dim=128)
        >>> errors = detector.check(predicted, actual)
        >>> for error in errors:
        ...     print(f"Error: {error.error_type.name}")
    """

    def __init__(
        self,
        input_dim: int,
        threshold: float = 0.3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.threshold = threshold
        self.hidden_dim = hidden_dim

        # Conflict detection network
        self.conflict_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Error history
        self._errors: List[ErrorSignal] = []
        self._timestep = 0

    def reset(self) -> None:
        """Reset detector state."""
        self._errors = []
        self._timestep = 0

    def check_prediction_error(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> Optional[ErrorSignal]:
        """Check for prediction error.

        Args:
            predicted: Predicted pattern
            actual: Actual observed pattern

        Returns:
            Error signal if error exceeds threshold
        """
        # Compute prediction error
        error = F.mse_loss(predicted, actual).item()

        if error > self.threshold:
            signal = ErrorSignal(
                error_type=ErrorType.PREDICTION_ERROR,
                magnitude=min(1.0, error),
                location="prediction",
                timestamp=self._timestep,
                details={"mse": error},
            )
            self._errors.append(signal)
            self._timestep += 1
            return signal

        self._timestep += 1
        return None

    def check_conflict(
        self,
        pattern_a: torch.Tensor,
        pattern_b: torch.Tensor,
    ) -> Optional[ErrorSignal]:
        """Check for conflict between patterns.

        Conflict occurs when patterns should be similar but aren't.

        Args:
            pattern_a: First pattern
            pattern_b: Second pattern

        Returns:
            Error signal if conflict detected
        """
        # Flatten patterns
        a = pattern_a.flatten()
        b = pattern_b.flatten()

        # Use network to detect conflict
        combined = torch.cat([a, b]).unsqueeze(0)

        with torch.no_grad():
            conflict_score = self.conflict_net(combined).item()

        if conflict_score > self.threshold:
            signal = ErrorSignal(
                error_type=ErrorType.CONFLICT,
                magnitude=conflict_score,
                location="belief_system",
                timestamp=self._timestep,
                details={"conflict_score": conflict_score},
            )
            self._errors.append(signal)
            self._timestep += 1
            return signal

        self._timestep += 1
        return None

    def check_consistency(
        self,
        patterns: List[torch.Tensor],
    ) -> Optional[ErrorSignal]:
        """Check for inconsistency across patterns.

        Args:
            patterns: List of patterns that should be consistent

        Returns:
            Error signal if inconsistency detected
        """
        if len(patterns) < 2:
            return None

        # Compute pairwise similarities
        similarities = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                p1 = patterns[i].flatten()
                p2 = patterns[j].flatten()
                sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
                similarities.append(sim.item())

        # Check variance in similarities
        mean_sim = sum(similarities) / len(similarities)
        variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)

        if variance > self.threshold:
            signal = ErrorSignal(
                error_type=ErrorType.INCONSISTENCY,
                magnitude=min(1.0, variance),
                location="reasoning",
                timestamp=self._timestep,
                details={"similarity_variance": variance},
            )
            self._errors.append(signal)
            self._timestep += 1
            return signal

        self._timestep += 1
        return None

    def check_timeout(
        self,
        elapsed_time: float,
        timeout: float,
    ) -> Optional[ErrorSignal]:
        """Check for processing timeout.

        Args:
            elapsed_time: Time elapsed
            timeout: Maximum allowed time

        Returns:
            Error signal if timeout exceeded
        """
        if elapsed_time > timeout:
            signal = ErrorSignal(
                error_type=ErrorType.TIMEOUT,
                magnitude=min(1.0, elapsed_time / timeout - 1),
                location="processing",
                timestamp=self._timestep,
                details={"elapsed": elapsed_time, "timeout": timeout},
            )
            self._errors.append(signal)
            self._timestep += 1
            return signal

        self._timestep += 1
        return None

    def get_active_errors(self) -> List[ErrorSignal]:
        """Get list of detected errors."""
        return list(self._errors)

    def clear_errors(self) -> None:
        """Clear error history."""
        self._errors = []

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of errors by type."""
        summary = {}
        for error in self._errors:
            type_name = error.error_type.name
            summary[type_name] = summary.get(type_name, 0) + 1
        return summary


class CognitiveMonitor(nn.Module):
    """Monitors overall cognitive state.

    Integrates signals from confidence, uncertainty, and error
    detection to maintain a picture of cognitive health.

    Example:
        >>> monitor = CognitiveMonitor(input_dim=128)
        >>> state = monitor.get_state()
        >>> if state.is_overloaded():
        ...     print("Need to reduce load!")
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[MetacognitiveConfig] = None,
    ):
        super().__init__()
        self.config = config or MetacognitiveConfig()
        self.input_dim = input_dim

        # Sub-components
        self.confidence_tracker = ConfidenceTracker(
            input_dim,
            self.config.hidden_dim,
            self.config.confidence_decay,
        )
        self.uncertainty_estimator = UncertaintyEstimator(
            input_dim,
            self.config.hidden_dim,
        )
        self.error_detector = ErrorDetector(
            input_dim,
            self.config.error_threshold,
            self.config.hidden_dim,
        )

        # Load tracking
        self.register_buffer("current_load", torch.tensor(0.0))
        self.register_buffer("load_history", torch.zeros(100))
        self._load_idx = 0

        # State
        self._current_state = CognitiveState()
        self._state_history: List[CognitiveState] = []

    def reset(self) -> None:
        """Reset monitor state."""
        self.confidence_tracker.reset()
        self.uncertainty_estimator.reset()
        self.error_detector.reset()
        self.current_load.fill_(0.0)
        self.load_history.fill_(0.0)
        self._load_idx = 0
        self._current_state = CognitiveState()
        self._state_history = []

    def update_load(self, activity: torch.Tensor) -> float:
        """Update cognitive load estimate.

        Load is estimated from neural activity level.

        Args:
            activity: Current neural activity

        Returns:
            Current load estimate
        """
        # Estimate load from activity magnitude
        activity_level = activity.abs().mean().item()

        # Normalize to 0-1
        load = min(1.0, activity_level / self.config.load_capacity)

        # Update running average
        self.load_history[self._load_idx % 100] = load
        self._load_idx += 1

        # Compute smoothed load
        valid_entries = min(self._load_idx, 100)
        self.current_load = self.load_history[:valid_entries].mean()

        return self.current_load.item()

    def update(
        self,
        activity: torch.Tensor,
        predicted: Optional[torch.Tensor] = None,
        actual: Optional[torch.Tensor] = None,
    ) -> CognitiveState:
        """Update cognitive state with new information.

        Args:
            activity: Current neural activity
            predicted: Optional prediction
            actual: Optional actual outcome

        Returns:
            Updated cognitive state
        """
        # Update load
        load = self.update_load(activity)

        # Update confidence
        confidence = self.confidence_tracker.estimate(activity)

        # Update uncertainty
        uncertainty = self.uncertainty_estimator.estimate_total(activity)

        # Check for errors
        errors = []
        if predicted is not None and actual is not None:
            error = self.error_detector.check_prediction_error(predicted, actual)
            if error:
                errors.append(error)

        # Check for overload
        if load > 0.9:
            errors.append(ErrorSignal(
                error_type=ErrorType.OVERLOAD,
                magnitude=load,
                location="system",
            ))

        # Check for high uncertainty
        if uncertainty["total"] > 0.8:
            errors.append(ErrorSignal(
                error_type=ErrorType.UNCERTAINTY,
                magnitude=uncertainty["total"],
                location="estimation",
            ))

        # Determine processing mode
        if load > 0.8:
            mode = "overloaded"
        elif uncertainty["total"] > 0.6:
            mode = "uncertain"
        elif confidence.value < 0.4:
            mode = "low_confidence"
        elif errors:
            mode = "error_recovery"
        else:
            mode = "normal"

        # Update state
        self._current_state = CognitiveState(
            load=load,
            confidence=confidence.value,
            uncertainty=uncertainty["total"],
            active_errors=errors,
            processing_mode=mode,
        )

        self._state_history.append(self._current_state)

        return self._current_state

    def get_state(self) -> CognitiveState:
        """Get current cognitive state."""
        return self._current_state

    def get_recommendations(self) -> List[str]:
        """Get recommendations based on current state."""
        recommendations = []
        state = self._current_state

        if state.is_overloaded():
            recommendations.append("Reduce cognitive load - simplify task or pause")

        if not state.is_confident():
            recommendations.append("Gather more evidence before proceeding")

        if state.uncertainty > 0.6:
            recommendations.append("High uncertainty - consider alternative interpretations")

        if state.has_errors():
            error_types = set(e.error_type.name for e in state.active_errors)
            if ErrorType.PREDICTION_ERROR.name in error_types:
                recommendations.append("Update predictions based on new evidence")
            if ErrorType.CONFLICT.name in error_types:
                recommendations.append("Resolve conflicting beliefs")
            if ErrorType.INCONSISTENCY.name in error_types:
                recommendations.append("Review reasoning for consistency")

        if not recommendations:
            recommendations.append("Continue current processing")

        return recommendations

    def get_state_history(self) -> List[CognitiveState]:
        """Get history of cognitive states."""
        return list(self._state_history)


class MetacognitiveController(nn.Module):
    """Controls processing based on metacognitive signals.

    Adjusts processing parameters and strategies based on
    confidence, uncertainty, and error signals.

    Example:
        >>> controller = MetacognitiveController(input_dim=128)
        >>> adjustments = controller.get_adjustments(state)
        >>> # Apply adjustments to processing
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[MetacognitiveConfig] = None,
    ):
        super().__init__()
        self.config = config or MetacognitiveConfig()
        self.input_dim = input_dim

        # Control network
        self.control_net = nn.Sequential(
            nn.Linear(input_dim + 3, self.config.hidden_dim),  # +3 for load, conf, uncert
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
        )

        # Output heads for different adjustments
        self.noise_head = nn.Linear(self.config.hidden_dim, 1)
        self.attention_head = nn.Linear(self.config.hidden_dim, 1)
        self.threshold_head = nn.Linear(self.config.hidden_dim, 1)

        # Action history
        self._actions: List[Dict[str, float]] = []

    def reset(self) -> None:
        """Reset controller state."""
        self._actions = []

    def get_adjustments(
        self,
        activity: torch.Tensor,
        state: CognitiveState,
    ) -> Dict[str, float]:
        """Compute processing adjustments based on state.

        Args:
            activity: Current neural activity
            state: Current cognitive state

        Returns:
            Dictionary of adjustment parameters
        """
        if activity.dim() == 1:
            activity = activity.unsqueeze(0)

        # Build input with state information
        state_info = torch.tensor(
            [state.load, state.confidence, state.uncertainty],
            device=activity.device,
        ).unsqueeze(0)

        combined = torch.cat([activity, state_info], dim=-1)

        # Get control signal
        with torch.no_grad():
            hidden = self.control_net(combined)

            noise_adj = torch.sigmoid(self.noise_head(hidden)).item()
            attention_adj = torch.sigmoid(self.attention_head(hidden)).item()
            threshold_adj = torch.sigmoid(self.threshold_head(hidden)).item()

        # Apply heuristic adjustments based on state
        adjustments = {
            "noise_scale": noise_adj,
            "attention_focus": attention_adj,
            "threshold_scale": threshold_adj,
        }

        # Override with rule-based adjustments for extreme states
        # Priority: overload > high uncertainty > low confidence
        if state.is_overloaded():
            adjustments["noise_scale"] = 0.2  # Reduce noise when overloaded
            adjustments["attention_focus"] = 0.9  # Increase focus
        elif state.uncertainty > 0.7:
            adjustments["noise_scale"] = 0.9  # High exploration for uncertainty
        elif not state.is_confident():
            adjustments["noise_scale"] = 0.8  # Increase exploration
            adjustments["threshold_scale"] = 0.7  # Lower thresholds

        self._actions.append(adjustments)

        return adjustments

    def get_processing_strategy(
        self,
        state: CognitiveState,
    ) -> str:
        """Determine processing strategy based on state.

        Args:
            state: Current cognitive state

        Returns:
            Strategy name
        """
        if state.is_overloaded():
            return "simplify"
        elif state.uncertainty > 0.7:
            return "explore"
        elif not state.is_confident():
            return "gather_evidence"
        elif state.has_errors():
            error_types = [e.error_type for e in state.active_errors]
            if ErrorType.PREDICTION_ERROR in error_types:
                return "update_model"
            elif ErrorType.CONFLICT in error_types:
                return "resolve_conflict"
            else:
                return "debug"
        else:
            return "proceed"

    def get_action_history(self) -> List[Dict[str, float]]:
        """Get history of control actions."""
        return list(self._actions)


class MetacognitiveNetwork(nn.Module):
    """Complete metacognitive network.

    Integrates monitoring, control, and adjustment into
    a unified metacognitive system.

    Example:
        >>> network = MetacognitiveNetwork(input_dim=128)
        >>> network.observe(activity)
        >>> state = network.get_cognitive_state()
        >>> adjustments = network.get_adjustments()
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[MetacognitiveConfig] = None,
    ):
        super().__init__()
        self.config = config or MetacognitiveConfig()
        self.input_dim = input_dim

        # Core components
        self.monitor = CognitiveMonitor(input_dim, self.config)
        self.controller = MetacognitiveController(input_dim, self.config)

        # SNN for metacognitive processing
        neuron_config = LIFConfig(tau_mem=self.config.tau_mem, noise_std=0.01)
        self.meta_neurons = LIFNeuron(
            n_neurons=self.config.hidden_dim,
            config=neuron_config,
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, self.config.hidden_dim)

        # State
        self._last_activity: Optional[torch.Tensor] = None
        self._last_prediction: Optional[torch.Tensor] = None
        self._timestep = 0

    def reset(self) -> None:
        """Reset network state."""
        self.monitor.reset()
        self.controller.reset()
        self.meta_neurons.reset_state(batch_size=1)
        self._last_activity = None
        self._last_prediction = None
        self._timestep = 0

    def observe(
        self,
        activity: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
        outcome: Optional[torch.Tensor] = None,
    ) -> CognitiveState:
        """Observe activity and update metacognitive state.

        Args:
            activity: Current neural activity
            prediction: Optional prediction made
            outcome: Optional actual outcome

        Returns:
            Updated cognitive state
        """
        # Store for comparison
        self._last_activity = activity
        self._last_prediction = prediction

        # Update monitor
        state = self.monitor.update(activity, prediction, outcome)

        # Process through meta neurons
        if activity.dim() == 1:
            activity = activity.unsqueeze(0)
        meta_input = self.input_proj(activity)
        self.meta_neurons(meta_input)

        self._timestep += 1

        return state

    def get_cognitive_state(self) -> CognitiveState:
        """Get current cognitive state."""
        return self.monitor.get_state()

    def get_adjustments(self) -> Dict[str, float]:
        """Get processing adjustments based on current state."""
        if self._last_activity is None:
            return {"noise_scale": 0.5, "attention_focus": 0.5, "threshold_scale": 0.5}

        state = self.monitor.get_state()
        return self.controller.get_adjustments(self._last_activity, state)

    def get_strategy(self) -> str:
        """Get recommended processing strategy."""
        state = self.monitor.get_state()
        return self.controller.get_processing_strategy(state)

    def get_recommendations(self) -> List[str]:
        """Get human-readable recommendations."""
        return self.monitor.get_recommendations()

    def get_confidence(self) -> float:
        """Get current confidence level."""
        return self.monitor.get_state().confidence

    def get_uncertainty(self) -> float:
        """Get current uncertainty level."""
        return self.monitor.get_state().uncertainty

    def get_load(self) -> float:
        """Get current cognitive load."""
        return self.monitor.get_state().load

    def has_errors(self) -> bool:
        """Check if there are active errors."""
        return self.monitor.get_state().has_errors()

    def get_errors(self) -> List[ErrorSignal]:
        """Get list of active errors."""
        return self.monitor.get_state().active_errors

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metacognitive state."""
        state = self.monitor.get_state()
        return {
            "load": state.load,
            "confidence": state.confidence,
            "uncertainty": state.uncertainty,
            "processing_mode": state.processing_mode,
            "n_errors": len(state.active_errors),
            "strategy": self.get_strategy(),
            "timestep": self._timestep,
        }
