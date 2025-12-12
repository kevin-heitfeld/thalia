"""
Learning Strategy Mixin for Brain Regions.

This mixin provides a unified interface for regions to use pluggable learning
strategies from thalia.learning.strategies, eliminating duplicate plasticity code.

Design Pattern:
===============
Instead of implementing custom _apply_plasticity() methods, regions:
1. Inherit from LearningStrategyMixin
2. Set self.learning_strategy to a strategy from learning.strategies
3. Call self.apply_strategy_learning() in forward()

This consolidates:
- Eligibility trace management
- Weight update computation
- Dopamine/neuromodulator gating
- Soft bounds and weight clamping
- Learning metrics collection

Usage Example:
==============
    class MyRegion(LearningStrategyMixin, NeuralComponent):
        def __init__(self, config):
            super().__init__(config)
            # Choose strategy based on learning rule
            self.learning_strategy = create_strategy(
                'stdp',
                learning_rate=config.stdp_lr,
                a_plus=0.01,
                a_minus=0.012,
                tau_plus=20.0,
                tau_minus=20.0,
            )
        
        def forward(self, input_spikes, **kwargs):
            # Process input
            output_spikes = self._compute_output(input_spikes)
            
            # Apply learning automatically
            metrics = self.apply_strategy_learning(
                input_spikes,
                output_spikes,
                weights=self.weights,
            )
            
            return output_spikes
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import torch

from thalia.learning.rules.strategies import (
    BaseStrategy,
    create_strategy,
)


class LearningStrategyMixin:
    """Mixin for regions using pluggable learning strategies.
    
    Provides unified interface to apply learning rules from strategies module.
    Handles dopamine modulation, eligibility gating, and metrics collection.
    
    Regions using this mixin should:
    1. Set self.learning_strategy in __init__
    2. Call self.apply_strategy_learning() during forward()
    3. Access learning metrics via self.last_learning_metrics
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize mixin state.
        
        Note: This should be called via super().__init__() in region __init__.
        """
        super().__init__(*args, **kwargs)  # type: ignore
        
        # Strategy instance (set by subclass)
        self.learning_strategy: Optional[BaseStrategy] = None
        
        # Learning metrics from last update
        self.last_learning_metrics: Dict[str, float] = {}
    
    def apply_strategy_learning(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: Optional[float] = None,
        target: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Apply learning strategy to update weights.
        
        This is the main interface for region learning. It:
        1. Checks plasticity enabled flag
        2. Gets dopamine-modulated learning rate
        3. Applies strategy compute_update()
        4. Updates weights in-place
        5. Returns learning metrics
        
        Args:
            pre_activity: Presynaptic activity (input spikes)
            post_activity: Postsynaptic activity (output spikes)
            weights: Weight matrix to update [n_post, n_pre]
            modulator: Optional modulator (dopamine, reward, etc.)
                      If None, uses self.state.dopamine
            target: Optional target for supervised learning
            **kwargs: Additional strategy-specific parameters
        
        Returns:
            Dict of learning metrics (ltp, ltd, net_change, etc.)
        """
        # Check if plasticity is enabled
        if not getattr(self, 'plasticity_enabled', True):
            return {}
        
        # Check if strategy is configured
        if self.learning_strategy is None:
            return {}
        
        # Get dopamine modulation if not explicitly provided
        if modulator is None and hasattr(self, 'state'):
            modulator = getattr(self.state, 'dopamine', 0.0)
        
        # Get effective learning rate (dopamine modulated)
        base_lr = self.learning_strategy.config.learning_rate
        if hasattr(self, 'get_effective_learning_rate'):
            effective_lr = self.get_effective_learning_rate(base_lr)  # type: ignore
            
            # Early exit if learning rate too small
            if effective_lr < 1e-8:
                return {}
            
            # Temporarily adjust strategy learning rate
            original_lr = self.learning_strategy.config.learning_rate
            self.learning_strategy.config.learning_rate = effective_lr
        else:
            original_lr = base_lr
        
        # Apply strategy
        try:
            new_weights, metrics = self.learning_strategy.compute_update(
                weights=weights,
                pre=pre_activity,
                post=post_activity,
                modulator=modulator if modulator is not None else 0.0,
                target=target,
                **kwargs,
            )
            
            # Update weights in-place
            with torch.no_grad():
                weights.data.copy_(new_weights)
            
            # Store metrics
            self.last_learning_metrics = metrics
            
            return metrics
            
        finally:
            # Restore original learning rate
            if hasattr(self, 'get_effective_learning_rate'):
                self.learning_strategy.config.learning_rate = original_lr
    
    def reset_learning_state(self) -> None:
        """Reset learning strategy state (traces, eligibility, etc.)."""
        if self.learning_strategy is not None:
            self.learning_strategy.reset_state()
        self.last_learning_metrics = {}
    
    def get_learning_metrics(self) -> Dict[str, float]:
        """Get learning metrics from last update."""
        return self.last_learning_metrics.copy()
    
    def create_strategy_from_config(
        self,
        learning_rule: str,
        **config_kwargs: Any,
    ) -> BaseStrategy:
        """Convenience method to create strategy from learning rule name.
        
        Args:
            learning_rule: Learning rule name ('stdp', 'bcm', 'three_factor', etc.)
            **config_kwargs: Strategy configuration parameters
        
        Returns:
            Configured learning strategy
        
        Example:
            self.learning_strategy = self.create_strategy_from_config(
                'stdp',
                learning_rate=self.config.learning_rate,
                a_plus=0.01,
                a_minus=0.012,
            )
        """
        return create_strategy(learning_rule, **config_kwargs)


__all__ = [
    'LearningStrategyMixin',
]
