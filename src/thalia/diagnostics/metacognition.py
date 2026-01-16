"""
Metacognitive Monitor for stage-aware confidence estimation.

Implements developmental progression in metacognitive ability:
- Stage 1 (Toddler): Binary confidence (know vs don't know)
- Stage 2 (Preschool): Coarse-grained (high/medium/low)
- Stage 3 (School-age): Continuous but poorly calibrated
- Stage 4 (Adolescent+): Well-calibrated through training

References:
- Lyons & Ghetti (2011): Metacognitive development in early childhood
- Roebers et al. (2012): Confidence judgments in children
- Fleming & Dolan (2012): Neural basis of metacognition
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from thalia.constants.architecture import (
    METACOG_ABSTENTION_STAGE1,
    METACOG_ABSTENTION_STAGE2,
    METACOG_ABSTENTION_STAGE3,
    METACOG_CALIBRATION_LR,
)


class MetacognitiveStage(Enum):
    """Developmental stages of metacognitive ability."""
    TODDLER = 1  # Binary: know vs don't know
    PRESCHOOL = 2  # Coarse: high/medium/low
    SCHOOL_AGE = 3  # Continuous but poorly calibrated
    ADOLESCENT = 4  # Well-calibrated with training


@dataclass
class MetacognitiveMonitorConfig:
    """Configuration for metacognitive monitoring."""
    input_size: int = 256  # Size of population activity
    hidden_size: int = 64  # Hidden layer for calibration network
    stage: MetacognitiveStage = MetacognitiveStage.TODDLER

    # Stage-specific abstention thresholds
    threshold_stage1: float = METACOG_ABSTENTION_STAGE1  # Binary threshold
    threshold_stage2: float = METACOG_ABSTENTION_STAGE2  # Low confidence threshold
    threshold_stage3: float = METACOG_ABSTENTION_STAGE3  # Uncertainty threshold
    threshold_stage4: float = METACOG_ABSTENTION_STAGE2  # Calibrated threshold (same as stage2)

    # Calibration learning
    calibration_lr: float = METACOG_CALIBRATION_LR
    use_dopamine_gating: bool = True

    device: str = "cpu"


class ConfidenceEstimator(nn.Module):
    """
    Estimates confidence from population activity.

    Uses variance in population response as proxy for uncertainty.
    Lower variance → higher confidence (clear winner).
    Higher variance → lower confidence (no clear winner).
    """

    def __init__(self, config: MetacognitiveMonitorConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    def estimate_raw_confidence(self, population_activity: torch.Tensor) -> float:
        """
        Estimate raw confidence from population activity.

        Args:
            population_activity: Neural population activity [pop_size]

        Returns:
            confidence: Raw confidence estimate [0, 1]
        """
        # Ensure on correct device
        population_activity = population_activity.to(self.device)

        # Method: Ratio of max to sum (winner-take-all strength)
        # Clear winner → high ratio → high confidence
        # Uniform distribution → low ratio → low confidence

        total = population_activity.sum()
        if total > 0:
            max_val = population_activity.max()
            # Normalize: ratio of max to average
            ratio = max_val / (total / population_activity.numel())
            # Map to confidence (high ratio = high confidence)
            # Use log-scaled sigmoid to handle wide range (1 to 100+)
            # sigmoid(log(ratio)) gives: ratio=1→0.5, ratio=2.7→0.62, ratio=20→0.88, ratio=150→0.95
            log_ratio = torch.log(torch.tensor(ratio + 1e-8))
            confidence = torch.sigmoid(log_ratio).item()
        else:
            # No activity → no confidence
            confidence = 0.0

        return float(torch.clamp(torch.tensor(confidence), 0.0, 1.0).item())

    def forward(self, population_activity: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility with calibration network."""
        conf = self.estimate_raw_confidence(population_activity)
        return torch.tensor([conf], device=self.device)


class CalibrationNetwork(nn.Module):
    """
    Learns to calibrate confidence estimates (Stage 3-4).

    Maps raw confidence to calibrated confidence based on feedback.
    """

    def __init__(self, config: MetacognitiveMonitorConfig):
        super().__init__()
        self.config = config

        # Simple MLP for calibration
        self.network = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()  # Output calibrated confidence [0, 1]
        )

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.calibration_lr
        )

        # Training statistics
        self.n_updates = 0
        self.total_error = 0.0

    def forward(self, raw_confidence: torch.Tensor) -> torch.Tensor:
        """
        Calibrate raw confidence.

        Args:
            raw_confidence: Raw confidence estimate [batch, 1]

        Returns:
            calibrated: Calibrated confidence [batch, 1]
        """
        return self.network(raw_confidence)

    def update(
        self,
        raw_confidence: float,
        actual_correct: float,
        dopamine: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Update calibration network based on feedback.

        Args:
            raw_confidence: Predicted confidence [0, 1]
            actual_correct: Actual correctness (1.0 or 0.0)
            dopamine: Optional dopamine gating (0-1)

        Returns:
            metrics: Training metrics
        """
        # Dopamine gating (Stage 4 feature)
        if self.config.use_dopamine_gating and dopamine is not None:
            learning_gate = dopamine
        else:
            learning_gate = 1.0

        # EXCEPTION: Temporarily enable gradients for metacognitive calibration
        # This is the ONLY module that uses backpropagation (at a different timescale)
        with torch.enable_grad():
            # Forward pass
            raw_conf_tensor = torch.tensor(
                [[raw_confidence]],
                device=self.network[0].weight.device,
                requires_grad=True
            )
            calibrated = self.forward(raw_conf_tensor)

            # Compute loss (mean squared error)
            target = torch.tensor(
                [[actual_correct]],
                device=calibrated.device
            )
            loss = ((calibrated - target) ** 2).mean()

            # Backward pass with dopamine gating
            self.optimizer.zero_grad()
            loss.backward()

            # Scale gradients by dopamine
            if learning_gate < 1.0:
                for param in self.network.parameters():
                    if param.grad is not None:
                        param.grad.mul_(learning_gate)

        self.optimizer.step()

        # Update statistics
        self.n_updates += 1
        error = abs(calibrated.item() - actual_correct)
        self.total_error = 0.9 * self.total_error + 0.1 * error

        return {
            "loss": loss.item(),
            "error": error,
            "avg_error": self.total_error,
            "n_updates": self.n_updates,
        }


class MetacognitiveMonitor:
    """
    Stage-aware metacognitive monitoring system.

    Developmental progression:
    - Stage 1: Binary confidence (above/below threshold)
    - Stage 2: Coarse-grained (3 levels: high/medium/low)
    - Stage 3: Continuous but poorly calibrated
    - Stage 4: Well-calibrated through experience
    """

    def __init__(self, config: MetacognitiveMonitorConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Components
        self.confidence_estimator = ConfidenceEstimator(config)
        self.calibration_network = CalibrationNetwork(config)

        # Move to device
        self.confidence_estimator.to(self.device)
        self.calibration_network.to(self.device)

        # Statistics
        self.statistics = {
            "n_estimates": 0,
            "n_abstentions": 0,
            "avg_confidence": 0.0,
            "avg_raw_confidence": 0.0,
        }

    def estimate_confidence(
        self,
        population_activity: torch.Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate confidence with stage-appropriate granularity.

        Args:
            population_activity: Neural population response [pop_size]

        Returns:
            confidence: Stage-appropriate confidence [0, 1]
            breakdown: Dict with raw and calibrated values
        """
        # Ensure on correct device
        population_activity = population_activity.to(self.device)

        # Get raw confidence
        raw_confidence = self.confidence_estimator.estimate_raw_confidence(
            population_activity
        )

        # Stage-specific processing
        if self.config.stage == MetacognitiveStage.TODDLER:
            # Binary: know vs don't know
            threshold = self.config.threshold_stage1
            confidence = 1.0 if raw_confidence > threshold else 0.0

        elif self.config.stage == MetacognitiveStage.PRESCHOOL:
            # Coarse-grained: high/medium/low
            if raw_confidence > 0.8:
                confidence = 1.0  # High
            elif raw_confidence > 0.5:
                confidence = 0.5  # Medium
            else:
                confidence = 0.0  # Low

        elif self.config.stage == MetacognitiveStage.SCHOOL_AGE:
            # Continuous but not well calibrated
            confidence = raw_confidence

        elif self.config.stage == MetacognitiveStage.ADOLESCENT:
            # Well-calibrated through learning
            raw_tensor = torch.tensor(
                [[raw_confidence]],
                device=self.device
            )
            calibrated = self.calibration_network(raw_tensor)
            confidence = calibrated.item()

        else:
            confidence = raw_confidence

        # Update statistics
        self.statistics["n_estimates"] += 1
        self.statistics["avg_raw_confidence"] = (
            0.9 * self.statistics["avg_raw_confidence"] + 0.1 * raw_confidence
        )
        self.statistics["avg_confidence"] = (
            0.9 * self.statistics["avg_confidence"] + 0.1 * confidence
        )

        breakdown = {
            "raw": raw_confidence,
            "processed": confidence,
            "stage": self.config.stage.name,
        }

        return confidence, breakdown

    def should_abstain(self, confidence: float) -> bool:
        """
        Decide whether to abstain based on confidence.

        Lower thresholds at higher stages (more willing to try).

        Args:
            confidence: Estimated confidence [0, 1]

        Returns:
            abstain: Whether to abstain from responding
        """
        # Stage-specific thresholds
        threshold_map = {
            MetacognitiveStage.TODDLER: self.config.threshold_stage1,
            MetacognitiveStage.PRESCHOOL: self.config.threshold_stage2,
            MetacognitiveStage.SCHOOL_AGE: self.config.threshold_stage3,
            MetacognitiveStage.ADOLESCENT: self.config.threshold_stage4,
        }

        threshold = threshold_map[self.config.stage]
        abstain = confidence < threshold

        if abstain:
            self.statistics["n_abstentions"] += 1

        return abstain

    def calibrate(
        self,
        population_activity: torch.Tensor,
        actual_correct: bool,
        dopamine: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Train calibration network (Stage 3-4 only).

        Args:
            population_activity: Neural response when answer was given
            actual_correct: Whether the answer was correct
            dopamine: Optional dopamine signal for gating

        Returns:
            metrics: Training metrics
        """
        if self.config.stage not in [
            MetacognitiveStage.SCHOOL_AGE,
            MetacognitiveStage.ADOLESCENT
        ]:
            return {"error": "Calibration only available in Stage 3-4"}

        # Get raw confidence
        raw_confidence = self.confidence_estimator.estimate_raw_confidence(
            population_activity
        )

        # Update calibration network
        metrics = self.calibration_network.update(
            raw_confidence=raw_confidence,
            actual_correct=1.0 if actual_correct else 0.0,
            dopamine=dopamine
        )

        return metrics

    def set_stage(self, stage: MetacognitiveStage):
        """Update developmental stage."""
        self.config.stage = stage

    def get_stage(self) -> MetacognitiveStage:
        """Get current stage."""
        return self.config.stage

    def get_statistics(self) -> Dict[str, any]:
        """Get monitoring statistics."""
        abstention_rate = (
            self.statistics["n_abstentions"] / self.statistics["n_estimates"]
            if self.statistics["n_estimates"] > 0 else 0.0
        )

        return {
            "n_estimates": self.statistics["n_estimates"],
            "n_abstentions": self.statistics["n_abstentions"],
            "abstention_rate": abstention_rate,
            "avg_confidence": self.statistics["avg_confidence"],
            "avg_raw_confidence": self.statistics["avg_raw_confidence"],
            "stage": self.config.stage.name,
            "calibration_updates": self.calibration_network.n_updates,
            "calibration_error": self.calibration_network.total_error,
        }

    def reset_statistics(self):
        """Reset monitoring statistics."""
        self.statistics = {
            "n_estimates": 0,
            "n_abstentions": 0,
            "avg_confidence": 0.0,
            "avg_raw_confidence": 0.0,
        }
