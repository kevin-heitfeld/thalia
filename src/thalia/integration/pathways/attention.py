"""
Enhanced Attention Mechanisms with developmental progression.

Combines bottom-up (stimulus-driven) and top-down (goal-directed) attention
with stage-dependent weighting that shifts from reactive to proactive over
development.

References:
- Posner & Petersen (1990): Attention networks
- Colombo (2001): Infant attention development
- Diamond (2013): Executive function emergence
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import torch
import torch.nn as nn


class AttentionStage(Enum):
    """Developmental stages of attention."""
    INFANT = 0  # Stage 0: Pure bottom-up
    TODDLER = 1  # Stage 1: Mostly bottom-up (70%)
    PRESCHOOL = 2  # Stage 2: Balanced (50/50)
    SCHOOL_AGE = 3  # Stage 3+: Mostly top-down (70%)


@dataclass
class AttentionMechanismsConfig:
    """Configuration for attention mechanisms."""
    input_size: int = 256  # Visual/sensory input size
    stage: AttentionStage = AttentionStage.TODDLER
    
    # Bottom-up parameters
    brightness_contrast_weight: float = 0.4
    motion_weight: float = 0.4
    novelty_weight: float = 0.2
    
    # Top-down parameters (requires existing SpikingAttentionPathway)
    use_top_down: bool = True
    top_down_pathway: Optional[nn.Module] = None
    
    # Developmental weighting (can override stage defaults)
    bottom_up_weight: Optional[float] = None
    top_down_weight: Optional[float] = None
    
    device: str = "cpu"


class AttentionMechanisms:
    """
    Two-pathway attention with developmental progression.
    
    Bottom-up: Stimulus-driven (salience, motion, novelty)
    Top-down: Goal-directed (via PFC → Cortex pathway)
    
    Stage-dependent weighting:
    - Stage 0-1: Dominated by bottom-up (reactive)
    - Stage 2: Balanced control
    - Stage 3+: Dominated by top-down (proactive)
    """
    
    def __init__(self, config: AttentionMechanismsConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Stage-dependent default weights
        self._stage_weights = {
            AttentionStage.INFANT: (1.0, 0.0),  # Pure bottom-up
            AttentionStage.TODDLER: (0.7, 0.3),
            AttentionStage.PRESCHOOL: (0.5, 0.5),
            AttentionStage.SCHOOL_AGE: (0.3, 0.7),
        }
        
        # Get weights (custom or stage default)
        if config.bottom_up_weight is not None and config.top_down_weight is not None:
            self.w_bottom_up = config.bottom_up_weight
            self.w_top_down = config.top_down_weight
        else:
            self.w_bottom_up, self.w_top_down = self._stage_weights[config.stage]
        
        # Normalize weights
        total = self.w_bottom_up + self.w_top_down
        if total > 0:
            self.w_bottom_up /= total
            self.w_top_down /= total
        
        # Previous input for motion detection
        self.prev_input: Optional[torch.Tensor] = None
        
        # Novelty tracking (exponential moving average)
        self.input_history: Optional[torch.Tensor] = None
        self.novelty_alpha = 0.1
        
        # Statistics
        self.statistics = {
            "n_bottom_up": 0,
            "n_top_down": 0,
            "n_combined": 0,
            "avg_bottom_up_strength": 0.0,
            "avg_top_down_strength": 0.0,
        }
    
    def reset_statistics(self):
        """Reset attention statistics."""
        self.statistics = {
            "n_bottom_up": 0,
            "n_top_down": 0,
            "n_combined": 0,
            "avg_bottom_up_strength": 0.0,
            "avg_top_down_strength": 0.0,
        }
    
    def reset_memory(self):
        """Reset motion and novelty memory."""
        self.prev_input = None
        self.input_history = None
    
    # ========================================================================
    # Bottom-Up Salience
    # ========================================================================
    
    def brightness_contrast(self, visual_input: torch.Tensor) -> torch.Tensor:
        """
        Compute brightness/contrast salience.
        
        High-contrast regions attract attention.
        
        Args:
            visual_input: Visual input [input_size]
        
        Returns:
            salience: Contrast-based salience [input_size]
        """
        # Local contrast: difference from neighbors
        # Approximate with variance over local windows
        # For 1D, use sliding window
        
        if visual_input.numel() < 3:
            # Too small for contrast, return uniform
            return torch.ones_like(visual_input)
        
        # Pad for boundary handling (manual for 1D)
        # Replicate first and last elements
        first = visual_input[0].unsqueeze(0)
        last = visual_input[-1].unsqueeze(0)
        padded = torch.cat([first, visual_input, last], dim=0)
        
        # Compute local variance (contrast)
        # center - neighbors
        contrast = torch.abs(padded[1:-1] - 0.5 * (padded[:-2] + padded[2:]))
        
        return contrast
    
    def motion_saliency(self, visual_input: torch.Tensor) -> torch.Tensor:
        """
        Compute motion salience.
        
        Moving regions attract attention.
        
        Args:
            visual_input: Visual input [input_size]
        
        Returns:
            salience: Motion-based salience [input_size]
        """
        if self.prev_input is None:
            # First frame, no motion
            self.prev_input = visual_input.clone()
            return torch.zeros_like(visual_input)
        
        # Temporal difference (motion)
        motion = torch.abs(visual_input - self.prev_input)
        
        # Update history
        self.prev_input = visual_input.clone()
        
        return motion
    
    def novelty_detector(self, visual_input: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty salience.
        
        Novel (unexpected) stimuli attract attention.
        
        Args:
            visual_input: Visual input [input_size]
        
        Returns:
            salience: Novelty-based salience [input_size]
        """
        if self.input_history is None:
            # First input, everything is novel
            self.input_history = visual_input.clone()
            return torch.ones_like(visual_input)
        
        # Novelty = deviation from expectation (moving average)
        novelty = torch.abs(visual_input - self.input_history)
        
        # Update history (exponential moving average)
        self.input_history = (
            self.novelty_alpha * visual_input +
            (1 - self.novelty_alpha) * self.input_history
        )
        
        return novelty
    
    def bottom_up_salience(self, visual_input: torch.Tensor) -> torch.Tensor:
        """
        Compute stimulus-driven attention.
        
        Combines contrast, motion, and novelty.
        
        Args:
            visual_input: Visual input [input_size]
        
        Returns:
            salience: Bottom-up attention weights [input_size]
        """
        # Ensure on correct device
        visual_input = visual_input.to(self.device)
        
        # Compute salience components
        contrast = self.brightness_contrast(visual_input)
        motion = self.motion_saliency(visual_input)
        novelty = self.novelty_detector(visual_input)
        
        # Weighted combination
        salience = (
            self.config.brightness_contrast_weight * contrast +
            self.config.motion_weight * motion +
            self.config.novelty_weight * novelty
        )
        
        # Normalize to [0, 1]
        salience_min = salience.min()
        salience_max = salience.max()
        if salience_max > salience_min:
            salience = (salience - salience_min) / (salience_max - salience_min)
        else:
            salience = torch.ones_like(salience) / salience.numel()
        
        # Update statistics
        self.statistics["n_bottom_up"] += 1
        self.statistics["avg_bottom_up_strength"] = (
            0.9 * self.statistics["avg_bottom_up_strength"] + 0.1 * salience.mean().item()
        )
        
        return salience
    
    # ========================================================================
    # Top-Down Modulation
    # ========================================================================
    
    def top_down_modulation(
        self,
        visual_input: torch.Tensor,
        goal: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Goal-directed attention (via PFC).
        
        Uses existing SpikingAttentionPathway if available.
        
        Args:
            visual_input: Visual input [input_size]
            goal: Goal/task representation (PFC activity) [goal_size]
            dt: Time step in ms
        
        Returns:
            attention: Top-down attention weights [input_size]
        """
        # Update statistics (always, even if no pathway)
        self.statistics["n_top_down"] += 1
        
        if not self.config.use_top_down or self.config.top_down_pathway is None:
            # No top-down pathway, return uniform
            attention = torch.ones_like(visual_input) / visual_input.numel()
            self.statistics["avg_top_down_strength"] = (
                0.9 * self.statistics["avg_top_down_strength"] + 0.1 * attention.mean().item()
            )
            return attention
        
        # Ensure on correct device
        visual_input = visual_input.to(self.device)
        goal = goal.to(self.device)
        
        # Use existing pathway (SpikingAttentionPathway)
        # pathway.modulate(input_signal, pfc_activity, dt) → modulated signal
        attention = self.config.top_down_pathway.modulate(
            visual_input,
            goal,
            dt=dt
        )
        
        # Normalize to attention weights (not modulated signal)
        attention = torch.abs(attention)  # Ensure positive
        attention = attention / (attention.sum() + 1e-8)
        
        # Update statistics
        self.statistics["avg_top_down_strength"] = (
            0.9 * self.statistics["avg_top_down_strength"] + 0.1 * attention.mean().item()
        )
        
        return attention
    
    # ========================================================================
    # Combined Attention
    # ========================================================================
    
    def combined_attention(
        self,
        visual_input: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Combine bottom-up and top-down attention.
        
        Developmental weighting based on stage:
        - Infant/Toddler: Dominated by bottom-up
        - Preschool: Balanced
        - School-age: Dominated by top-down
        
        Args:
            visual_input: Visual input [input_size]
            goal: Optional goal representation for top-down
            dt: Time step in ms
        
        Returns:
            attention: Combined attention weights [input_size]
            components: Dict with 'bottom_up' and 'top_down' components
        """
        # Bottom-up salience (always computed)
        bottom_up = self.bottom_up_salience(visual_input)
        
        # Top-down modulation (if goal provided and pathway available)
        if goal is not None and self.config.use_top_down:
            top_down = self.top_down_modulation(visual_input, goal, dt)
        else:
            # No top-down, use uniform
            top_down = torch.ones_like(visual_input) / visual_input.numel()
        
        # Stage-dependent weighting
        attention = self.w_bottom_up * bottom_up + self.w_top_down * top_down
        
        # Normalize
        attention = attention / (attention.sum() + 1e-8)
        
        # Update statistics
        self.statistics["n_combined"] += 1
        
        components = {
            "bottom_up": bottom_up.detach(),
            "top_down": top_down.detach(),
            "weights": (self.w_bottom_up, self.w_top_down),
        }
        
        return attention, components
    
    def apply_attention(
        self,
        visual_input: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply combined attention to visual input.
        
        Args:
            visual_input: Visual input [input_size]
            goal: Optional goal for top-down attention
            dt: Time step in ms
        
        Returns:
            attended_input: Input with attention applied [input_size]
            components: Dict with attention breakdown
        """
        attention, components = self.combined_attention(visual_input, goal, dt)
        
        # Apply attention (multiplicative)
        attended_input = visual_input * attention * visual_input.numel()  # Scale to preserve energy
        
        return attended_input, components
    
    # ========================================================================
    # Developmental Progression
    # ========================================================================
    
    def set_stage(self, stage: AttentionStage):
        """Update developmental stage and weights."""
        self.config.stage = stage
        self.w_bottom_up, self.w_top_down = self._stage_weights[stage]
        
        # Normalize
        total = self.w_bottom_up + self.w_top_down
        if total > 0:
            self.w_bottom_up /= total
            self.w_top_down /= total
    
    def get_stage_weights(self, stage: AttentionStage) -> Tuple[float, float]:
        """Get bottom-up and top-down weights for a stage."""
        return self._stage_weights[stage]
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_statistics(self) -> dict:
        """Get attention statistics."""
        return {
            "n_bottom_up": self.statistics["n_bottom_up"],
            "n_top_down": self.statistics["n_top_down"],
            "n_combined": self.statistics["n_combined"],
            "avg_bottom_up_strength": self.statistics["avg_bottom_up_strength"],
            "avg_top_down_strength": self.statistics["avg_top_down_strength"],
            "current_stage": self.config.stage.name,
            "current_weights": {
                "bottom_up": self.w_bottom_up,
                "top_down": self.w_top_down,
            },
        }
