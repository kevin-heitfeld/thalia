"""
THALIA Metacognition Module.

This module implements self-monitoring and meta-awareness capabilities,
allowing the network to monitor its own processing, estimate confidence
and uncertainty, detect errors, and adjust processing strategies.

The metacognition system consists of:
- ConfidenceTracker: Tracks confidence in processing outcomes
- UncertaintyEstimator: Estimates epistemic and aleatoric uncertainty
- ErrorDetector: Detects prediction errors, conflicts, and anomalies
- CognitiveMonitor: Monitors overall cognitive state
- MetacognitiveController: Adjusts processing based on monitoring
- MetacognitiveNetwork: Complete integrated metacognitive system
"""

from thalia.metacognition.metacognition import (
    ConfidenceLevel,
    ErrorType,
    ConfidenceEstimate,
    ErrorSignal,
    CognitiveState,
    MetacognitiveConfig,
    ConfidenceTracker,
    UncertaintyEstimator,
    ErrorDetector,
    CognitiveMonitor,
    MetacognitiveController,
    MetacognitiveNetwork,
)

__all__ = [
    "ConfidenceLevel",
    "ErrorType",
    "ConfidenceEstimate",
    "ErrorSignal",
    "CognitiveState",
    "MetacognitiveConfig",
    "ConfidenceTracker",
    "UncertaintyEstimator",
    "ErrorDetector",
    "CognitiveMonitor",
    "MetacognitiveController",
    "MetacognitiveNetwork",
]
