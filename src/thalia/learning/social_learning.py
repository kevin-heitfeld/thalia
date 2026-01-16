"""
Social Learning Module for developmental curriculum.

Implements fast learning from social cues:
- Imitation learning (2x learning rate from demonstration)
- Pedagogy boost (1.5x learning rate when teaching detected)
- Joint attention (gaze-driven attention modulation)

References:
- Meltzoff & Moore (1977): Imitation in newborns
- Csibra & Gergely (2009): Natural pedagogy
- Tomasello (1995): Joint attention and early learning
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any

import torch


class SocialCueType(Enum):
    """Types of social cues."""
    DEMONSTRATION = "demonstration"  # Observed action
    OSTENSIVE = "ostensive"  # Teaching signal (eye contact, motherese)
    GAZE = "gaze"  # Gaze direction
    JOINT_ATTENTION = "joint_attention"  # Shared focus
    NONE = "none"


@dataclass
class SocialLearningConfig:
    """Configuration for social learning."""
    imitation_boost: float = 2.0  # Multiplier for demonstration learning
    pedagogy_boost: float = 1.5  # Multiplier for teaching contexts
    gaze_influence: float = 0.3  # Weight of gaze in attention
    joint_attention_threshold: float = 0.7  # When to activate joint attention
    device: str = "cpu"


@dataclass
class SocialContext:
    """Context for social learning episode."""
    cue_type: SocialCueType
    demonstration_present: bool = False
    ostensive_cues: Dict[str, bool] = None  # eye_contact, motherese, pointing
    gaze_direction: Optional[torch.Tensor] = None  # Direction vector
    shared_attention: float = 0.0  # Joint attention strength [0, 1]

    def __post_init__(self):
        if self.ostensive_cues is None:
            self.ostensive_cues = {
                "eye_contact": False,
                "motherese": False,
                "pointing": False,
            }


class SocialLearningModule:
    """
    Social learning mechanisms for accelerated learning from others.

    Implements three key mechanisms:
    1. Imitation learning: Fast learning from demonstration
    2. Natural pedagogy: Enhanced learning when teaching is detected
    3. Joint attention: Attention guided by gaze cues
    """

    def __init__(self, config: Optional[SocialLearningConfig] = None):
        self.config = config or SocialLearningConfig()
        self.device = torch.device(self.config.device)

        # Statistics
        self.statistics = {
            "n_demonstrations": 0,
            "n_pedagogy_episodes": 0,
            "n_joint_attention": 0,
            "avg_learning_boost": 1.0,
        }

    def reset_statistics(self):
        """Reset learning statistics."""
        self.statistics = {
            "n_demonstrations": 0,
            "n_pedagogy_episodes": 0,
            "n_joint_attention": 0,
            "avg_learning_boost": 1.0,
        }

    # ========================================================================
    # Imitation Learning
    # ========================================================================

    def imitation_learning(
        self,
        base_lr: float,
        demonstration_present: bool = True,
    ) -> float:
        """
        Boost learning rate when learning from demonstration.

        Biology: Mirror neurons and observational learning enable fast
        acquisition of motor skills and behaviors from others.

        Args:
            base_lr: Base learning rate
            demonstration_present: Whether demonstration is being observed

        Returns:
            modulated_lr: Learning rate with imitation boost
        """
        if demonstration_present:
            self.statistics["n_demonstrations"] += 1
            modulated_lr = base_lr * self.config.imitation_boost

            # Update running average of boost
            self._update_avg_boost(self.config.imitation_boost)

            return modulated_lr
        else:
            return base_lr

    def motor_imitation(
        self,
        observed_action: torch.Tensor,
        own_action: torch.Tensor,
        base_lr: float,
    ) -> Tuple[torch.Tensor, float]:
        """
        Learn motor action from demonstration via error correction.

        Args:
            observed_action: Demonstrated motor command
            own_action: Agent's attempted motor command
            base_lr: Base learning rate

        Returns:
            motor_error: Difference to be corrected
            effective_lr: Boosted learning rate
        """
        # Calculate imitation error
        motor_error = observed_action - own_action

        # Boost learning rate for imitation
        effective_lr = self.imitation_learning(base_lr, demonstration_present=True)

        return motor_error, effective_lr

    # ========================================================================
    # Natural Pedagogy
    # ========================================================================

    def pedagogy_boost(
        self,
        base_lr: float,
        ostensive_cues: Dict[str, bool],
    ) -> float:
        """
        Boost learning rate when teaching signals are detected.

        Ostensive cues (Natural Pedagogy Theory):
        - Eye contact: Teacher looking at learner
        - Motherese: Infant-directed speech (higher pitch, slower)
        - Pointing: Directed gestures

        Biology: Infants are biologically prepared to learn from teaching,
        with enhanced attention and learning when ostensive cues present.

        Args:
            base_lr: Base learning rate
            ostensive_cues: Dictionary of detected teaching signals

        Returns:
            modulated_lr: Learning rate with pedagogy boost
        """
        is_teaching = self._detect_teaching_signal(ostensive_cues)

        if is_teaching:
            self.statistics["n_pedagogy_episodes"] += 1
            modulated_lr = base_lr * self.config.pedagogy_boost

            # Update running average
            self._update_avg_boost(self.config.pedagogy_boost)

            return modulated_lr
        else:
            return base_lr

    def _detect_teaching_signal(self, ostensive_cues: Dict[str, bool]) -> bool:
        """
        Detect whether teaching is occurring based on ostensive cues.

        Teaching detected if:
        - Eye contact + at least one other cue, OR
        - All three cues present

        Args:
            ostensive_cues: Dict with eye_contact, motherese, pointing

        Returns:
            is_teaching: Whether teaching context is detected
        """
        eye_contact = ostensive_cues.get("eye_contact", False)
        motherese = ostensive_cues.get("motherese", False)
        pointing = ostensive_cues.get("pointing", False)

        # Strong signal: eye contact + another cue
        if eye_contact and (motherese or pointing):
            return True

        # Very strong signal: all three
        if eye_contact and motherese and pointing:
            return True

        return False

    # ========================================================================
    # Joint Attention
    # ========================================================================

    def joint_attention(
        self,
        attention_weights: torch.Tensor,
        gaze_direction: Optional[torch.Tensor] = None,
        shared_attention_strength: float = 0.0,
    ) -> torch.Tensor:
        """
        Modulate attention based on gaze cues and joint attention.

        Biology: Infants follow gaze direction to attend to same objects
        as caregivers, enabling efficient learning about relevant stimuli.

        Args:
            attention_weights: Current attention distribution [N]
            gaze_direction: Gaze direction vector (normalized) [N]
            shared_attention_strength: Joint attention strength [0, 1]

        Returns:
            modulated_attention: Attention with gaze influence
        """
        if gaze_direction is None:
            return attention_weights

        # Ensure tensors on correct device
        attention_weights = attention_weights.to(self.device)
        gaze_direction = gaze_direction.to(self.device)

        # Normalize gaze direction (if not already)
        if gaze_direction.abs().sum() > 0:
            gaze_direction = gaze_direction / (gaze_direction.abs().sum() + 1e-8)

        # Weight by gaze influence
        gaze_modulation = self._compute_gaze_modulation(
            gaze_direction,
            shared_attention_strength
        )

        # Combine with existing attention
        # gaze_influence controls how much gaze affects attention
        modulated_attention = (
            (1.0 - self.config.gaze_influence) * attention_weights +
            self.config.gaze_influence * gaze_modulation
        )

        # Normalize
        modulated_attention = modulated_attention / (modulated_attention.sum() + 1e-8)

        # Track joint attention events
        if shared_attention_strength > self.config.joint_attention_threshold:
            self.statistics["n_joint_attention"] += 1

        return modulated_attention

    def _compute_gaze_modulation(
        self,
        gaze_direction: torch.Tensor,
        shared_attention_strength: float,
    ) -> torch.Tensor:
        """
        Compute attention modulation from gaze.

        Args:
            gaze_direction: Normalized gaze direction [N]
            shared_attention_strength: How strongly attending together [0, 1]

        Returns:
            gaze_modulation: Attention boost at gaze target
        """
        # Boost attention at gaze target
        # Higher shared attention → stronger boost
        boost = 1.0 + shared_attention_strength

        gaze_modulation = gaze_direction * boost

        # Ensure non-negative and normalized
        gaze_modulation = torch.clamp(gaze_modulation, min=0.0)
        gaze_modulation = gaze_modulation / (gaze_modulation.sum() + 1e-8)

        return gaze_modulation

    # ========================================================================
    # Combined Social Learning
    # ========================================================================

    def modulate_learning(
        self,
        base_lr: float,
        social_context: SocialContext,
    ) -> float:
        """
        Apply all social learning modulations to base learning rate.

        Combines:
        - Imitation boost (if demonstration present)
        - Pedagogy boost (if teaching detected)

        Note: These are multiplicative (can stack for strong boost).

        Args:
            base_lr: Base learning rate
            social_context: Social context information

        Returns:
            modulated_lr: Final modulated learning rate
        """
        lr = base_lr

        # Apply imitation boost
        if social_context.demonstration_present:
            lr = self.imitation_learning(lr, demonstration_present=True)

        # Apply pedagogy boost
        if social_context.ostensive_cues:
            lr = self.pedagogy_boost(lr, social_context.ostensive_cues)

        return lr

    def modulate_attention(
        self,
        attention_weights: torch.Tensor,
        social_context: SocialContext,
    ) -> torch.Tensor:
        """
        Apply all attention modulations.

        Args:
            attention_weights: Current attention distribution
            social_context: Social context information

        Returns:
            modulated_attention: Attention with social modulation
        """
        # Apply joint attention
        if social_context.gaze_direction is not None:
            attention_weights = self.joint_attention(
                attention_weights,
                social_context.gaze_direction,
                social_context.shared_attention,
            )

        return attention_weights

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _update_avg_boost(self, boost: float):
        """Update running average of learning boost."""
        alpha = 0.1  # Exponential moving average weight
        self.statistics["avg_learning_boost"] = (
            alpha * boost + (1.0 - alpha) * self.statistics["avg_learning_boost"]
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get social learning statistics."""
        total_events = (
            self.statistics["n_demonstrations"] +
            self.statistics["n_pedagogy_episodes"] +
            self.statistics["n_joint_attention"]
        )

        return {
            "n_demonstrations": self.statistics["n_demonstrations"],
            "n_pedagogy_episodes": self.statistics["n_pedagogy_episodes"],
            "n_joint_attention": self.statistics["n_joint_attention"],
            "total_social_events": total_events,
            "avg_learning_boost": self.statistics["avg_learning_boost"],
        }

    def create_social_context(
        self,
        cue_type: SocialCueType = SocialCueType.NONE,
        **kwargs,
    ) -> SocialContext:
        """
        Helper to create social context.

        Args:
            cue_type: Type of social cue
            **kwargs: Additional context parameters

        Returns:
            social_context: Configured social context
        """
        return SocialContext(cue_type=cue_type, **kwargs)


def compute_shared_attention(
    agent_gaze: torch.Tensor,
    other_gaze: torch.Tensor,
    threshold: float = 0.8,
) -> float:
    """
    Compute shared attention strength between agent and other.

    Joint attention occurs when both attend to same location.

    Args:
        agent_gaze: Agent's attention distribution [N]
        other_gaze: Other's attention distribution [N]
        threshold: Similarity threshold for joint attention

    Returns:
        shared_attention: Strength of joint attention [0, 1]
    """
    # Ensure same device
    agent_gaze = agent_gaze / (agent_gaze.sum() + 1e-8)
    other_gaze = other_gaze / (other_gaze.sum() + 1e-8)

    # Compute overlap (dot product of normalized distributions)
    overlap = torch.dot(agent_gaze.flatten(), other_gaze.flatten()).item()

    # Threshold and scale
    if overlap > threshold:
        # Scale to [0, 1] based on how much it exceeds threshold
        shared_attention = min(1.0, (overlap - threshold) / (1.0 - threshold))
    else:
        shared_attention = 0.0

    return shared_attention
