"""
Online Homeostatic Meta-Learning for SNN Parameter Tuning.

This module implements biologically-inspired meta-learning that adjusts
network hyperparameters based on ongoing activity statistics. Unlike
offline hyperparameter search, this operates continuously during training.

Biological Basis:
- Metaplasticity: STDP learning rate adjusts based on recent activity
- Homeostatic set-points: Target firing rates adapt to network statistics
- Neuromodulatory gain control: Global parameters scale based on performance

Key Principles:
1. All adjustments are LOCAL - based on observable network statistics
2. Changes are SLOW - meta-learning timescale >> learning timescale
3. Rules are HOMEOSTATIC - push toward stable operating regimes

Usage:
    meta = MetaHomeostasis(config)
    
    for cycle in training:
        # ... run network ...
        
        # Update meta-parameters based on cycle statistics
        adjusted_params = meta.update(
            firing_rates=neuron_firing_rates,
            phase_accuracy=correct_phases / total_phases,
            weight_saturation=saturated_weights / total_weights,
        )
        
        # Apply adjusted parameters
        theta_modulation_strength = adjusted_params['theta_modulation_strength']
        ...
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch
import numpy as np


@dataclass
class MetaHomeostasisConfig:
    """Configuration for meta-homeostatic parameter tuning.
    
    Attributes:
        # Learning rates for parameter adjustment (very slow)
        meta_lr: float = 0.01  # Base learning rate for meta-updates
        
        # Target statistics (what we want the network to achieve)
        target_firing_rate_hz: float = 20.0  # Target mean firing rate
        target_firing_rate_cv: float = 0.3   # Target coefficient of variation
        target_phase_accuracy: float = 0.8    # Target phase-winner accuracy
        target_weight_saturation: float = 0.1 # Target fraction of saturated weights
        target_weight_diversity: float = 0.8  # Target: different neurons prefer different inputs
        
        # Parameter bounds (prevent runaway adaptation)
        theta_strength_bounds: Tuple[float, float] = (0.1, 10.0)
        homeostatic_strength_bounds: Tuple[float, float] = (0.001, 0.1)
        som_strength_bounds: Tuple[float, float] = (0.1, 2.0)
        hebbian_lr_bounds: Tuple[float, float] = (0.001, 0.1)
        eligibility_tau_bounds: Tuple[float, float] = (10.0, 200.0)  # In timesteps
        heterosynaptic_bounds: Tuple[float, float] = (0.1, 2.0)  # LTD/LTP ratio
        
        # Adaptation timescales (in cycles)
        firing_rate_tau: float = 10.0   # Smooth firing rate over 10 cycles
        phase_accuracy_tau: float = 20.0 # Smooth accuracy over 20 cycles
        
        # Enable/disable specific adaptations
        adapt_theta_strength: bool = True
        adapt_homeostatic_strength: bool = True
        adapt_som_strength: bool = True
        adapt_hebbian_lr: bool = True
        adapt_eligibility_tau: bool = True  # NEW: Adapt eligibility trace decay
        adapt_heterosynaptic: bool = True   # NEW: Adapt heterosynaptic LTD ratio
    """
    meta_lr: float = 0.01
    
    target_firing_rate_hz: float = 20.0
    target_firing_rate_cv: float = 0.3
    target_phase_accuracy: float = 0.8
    target_weight_saturation: float = 0.1
    target_weight_diversity: float = 0.8  # NEW
    
    theta_strength_bounds: Tuple[float, float] = (0.1, 10.0)
    homeostatic_strength_bounds: Tuple[float, float] = (0.001, 0.1)
    som_strength_bounds: Tuple[float, float] = (0.1, 2.0)
    hebbian_lr_bounds: Tuple[float, float] = (0.001, 0.1)
    eligibility_tau_bounds: Tuple[float, float] = (10.0, 200.0)  # NEW
    heterosynaptic_bounds: Tuple[float, float] = (0.1, 2.0)  # NEW
    
    firing_rate_tau: float = 10.0
    phase_accuracy_tau: float = 20.0
    
    adapt_theta_strength: bool = True
    adapt_homeostatic_strength: bool = True
    adapt_som_strength: bool = True
    adapt_hebbian_lr: bool = True
    adapt_eligibility_tau: bool = True  # NEW
    adapt_heterosynaptic: bool = True   # NEW


class MetaHomeostasis:
    """Online homeostatic meta-learning for SNN parameters.
    
    Monitors network statistics and adjusts hyperparameters to maintain
    stable, effective learning. All rules are local and homeostatic.
    
    Adaptation Rules:
    
    1. Theta Modulation Strength:
       - IF neurons win wrong phases consistently → INCREASE theta strength
       - IF neurons are too phase-locked (no flexibility) → DECREASE theta strength
       
    2. Homeostatic Strength:
       - IF firing rates are unstable (high variance) → INCREASE homeostatic strength
       - IF firing rates are stable but wrong → let homeostasis work normally
       
    3. SOM+ Inhibition Strength:
       - IF multiple neurons win same phase → INCREASE SOM strength (more differentiation)
       - IF neurons are too specialized (dead neurons) → DECREASE SOM strength
       
    4. Hebbian Learning Rate:
       - IF weights are saturating at w_max → DECREASE learning rate
       - IF weights are not changing → INCREASE learning rate
    
    5. Eligibility Trace Tau (NEW):
       - IF weight peaks cluster (all neurons prefer same inputs) → INCREASE tau
       - This is a temporal bias signal - late inputs dominating means tau too short
       
    6. Heterosynaptic LTD Ratio (NEW):
       - IF weight diversity is low → INCREASE heterosynaptic LTD
       - More competition between inputs = better specialization
    """
    
    def __init__(self, config: MetaHomeostasisConfig, initial_params: Dict[str, float]):
        """Initialize meta-homeostasis with current parameter values.
        
        Args:
            config: MetaHomeostasisConfig with targets and bounds
            initial_params: Dict with current values of tunable parameters:
                - theta_modulation_strength
                - homeostatic_strength_hz
                - som_strength
                - hebbian_learning_rate
        """
        self.config = config
        
        # Current parameter values
        self.params = {
            'theta_modulation_strength': initial_params.get('theta_modulation_strength', 2.5),
            'homeostatic_strength_hz': initial_params.get('homeostatic_strength_hz', 0.01),
            'som_strength': initial_params.get('som_strength', 0.5),
            'hebbian_learning_rate': initial_params.get('hebbian_learning_rate', 0.02),
            'eligibility_tau': initial_params.get('eligibility_tau', 20.0),  # NEW
            'heterosynaptic_ratio': initial_params.get('heterosynaptic_ratio', 0.5),  # NEW
        }
        
        # Exponential moving averages of statistics
        self.avg_firing_rate_hz: float = config.target_firing_rate_hz
        self.avg_firing_rate_std: float = config.target_firing_rate_hz * config.target_firing_rate_cv
        self.avg_phase_accuracy: float = 0.5  # Start pessimistic
        self.avg_weight_saturation: float = 0.0
        self.avg_weight_change: float = 0.01  # Start with assumption of some change
        self.avg_weight_diversity: float = 1.0  # Start optimistic (different peaks)
        
        # History for debugging
        self.history: list[Dict[str, float]] = []
        
    def update(
        self,
        firing_rates_hz: torch.Tensor,
        phase_accuracy: float,
        weight_saturation: float,
        weight_change: float,
        cycle_num: int,
        weight_diversity: float = 1.0,  # NEW: fraction of neurons with unique peak inputs
    ) -> Dict[str, float]:
        """Update meta-parameters based on current cycle statistics.
        
        Args:
            firing_rates_hz: Per-neuron firing rates this cycle (Hz)
            phase_accuracy: Fraction of phases with correct winner (0-1)
            weight_saturation: Fraction of weights at w_max (0-1)
            weight_change: Mean absolute weight change this cycle
            cycle_num: Current training cycle (for logging)
            weight_diversity: Fraction of neurons with unique peak inputs (0-1)
                              0 = all neurons prefer same input
                              1 = all neurons prefer different inputs
            
        Returns:
            Updated parameter dictionary
        """
        cfg = self.config
        
        # Update exponential moving averages
        alpha_fr = 1.0 / cfg.firing_rate_tau
        alpha_pa = 1.0 / cfg.phase_accuracy_tau
        
        mean_fr = float(firing_rates_hz.mean().item())
        std_fr = float(firing_rates_hz.std().item())
        
        self.avg_firing_rate_hz = (1 - alpha_fr) * self.avg_firing_rate_hz + alpha_fr * mean_fr
        self.avg_firing_rate_std = (1 - alpha_fr) * self.avg_firing_rate_std + alpha_fr * std_fr
        self.avg_phase_accuracy = (1 - alpha_pa) * self.avg_phase_accuracy + alpha_pa * phase_accuracy
        self.avg_weight_saturation = (1 - alpha_pa) * self.avg_weight_saturation + alpha_pa * weight_saturation
        self.avg_weight_change = (1 - alpha_pa) * self.avg_weight_change + alpha_pa * weight_change
        
        # Compute errors from targets
        fr_error = (self.avg_firing_rate_hz - cfg.target_firing_rate_hz) / cfg.target_firing_rate_hz
        cv = self.avg_firing_rate_std / max(self.avg_firing_rate_hz, 1.0)
        cv_error = cv - cfg.target_firing_rate_cv
        phase_error = cfg.target_phase_accuracy - self.avg_phase_accuracy
        sat_error = self.avg_weight_saturation - cfg.target_weight_saturation
        
        # =================================================================
        # ADAPTATION RULE 1: Theta Modulation Strength
        # =================================================================
        # Goal: Neurons should fire preferentially in their target phase
        # IF phase accuracy is low → increase theta strength (more phase selectivity)
        # IF phase accuracy is very high but neurons are inflexible → slight decrease
        if cfg.adapt_theta_strength:
            if phase_error > 0.1:  # Accuracy below target by >10%
                # Increase theta strength to enforce phase selectivity
                adjustment = cfg.meta_lr * phase_error * 2.0
                self.params['theta_modulation_strength'] *= (1.0 + adjustment)
            elif phase_error < -0.15:  # Accuracy above target by >15%
                # Slight decrease to allow flexibility
                adjustment = cfg.meta_lr * 0.5
                self.params['theta_modulation_strength'] *= (1.0 - adjustment)
            
            # Clamp to bounds
            self.params['theta_modulation_strength'] = np.clip(
                self.params['theta_modulation_strength'],
                cfg.theta_strength_bounds[0],
                cfg.theta_strength_bounds[1]
            )
        
        # =================================================================
        # ADAPTATION RULE 2: Homeostatic Strength
        # =================================================================
        # Goal: Firing rates should be stable around target
        # IF firing rate variance is high → increase homeostatic strength
        # IF firing rates are stuck far from target → also increase
        if cfg.adapt_homeostatic_strength:
            instability = abs(fr_error) + abs(cv_error)
            if instability > 0.3:  # Rates are unstable or far from target
                adjustment = cfg.meta_lr * instability
                self.params['homeostatic_strength_hz'] *= (1.0 + adjustment)
            elif instability < 0.1:  # Rates are stable and on target
                # Slight decrease to allow learning
                adjustment = cfg.meta_lr * 0.2
                self.params['homeostatic_strength_hz'] *= (1.0 - adjustment)
            
            self.params['homeostatic_strength_hz'] = np.clip(
                self.params['homeostatic_strength_hz'],
                cfg.homeostatic_strength_bounds[0],
                cfg.homeostatic_strength_bounds[1]
            )
        
        # =================================================================
        # ADAPTATION RULE 3: SOM+ Inhibition Strength
        # =================================================================
        # Goal: Neurons should specialize on different inputs (no overlap)
        # Proxy: If CV of firing rates is low, neurons are too similar → increase SOM
        # If some neurons are dead (very low rates), SOM might be too strong → decrease
        if cfg.adapt_som_strength:
            min_rate = float(firing_rates_hz.min().item())
            
            if cv < 0.2:  # Neurons too similar
                adjustment = cfg.meta_lr * (0.2 - cv) * 2.0
                self.params['som_strength'] *= (1.0 + adjustment)
            elif min_rate < cfg.target_firing_rate_hz * 0.1:  # Dead neurons
                adjustment = cfg.meta_lr * 0.5
                self.params['som_strength'] *= (1.0 - adjustment)
            
            self.params['som_strength'] = np.clip(
                self.params['som_strength'],
                cfg.som_strength_bounds[0],
                cfg.som_strength_bounds[1]
            )
        
        # =================================================================
        # ADAPTATION RULE 4: Hebbian Learning Rate
        # =================================================================
        # Goal: Weights should change but not saturate
        # IF weights are saturating → decrease learning rate
        # IF weights are not changing → increase learning rate
        if cfg.adapt_hebbian_lr:
            if sat_error > 0.1:  # Too many saturated weights
                adjustment = cfg.meta_lr * sat_error * 3.0
                self.params['hebbian_learning_rate'] *= (1.0 - adjustment)
            elif self.avg_weight_change < 0.001:  # Weights not changing
                adjustment = cfg.meta_lr * 1.0
                self.params['hebbian_learning_rate'] *= (1.0 + adjustment)
            
            self.params['hebbian_learning_rate'] = np.clip(
                self.params['hebbian_learning_rate'],
                cfg.hebbian_lr_bounds[0],
                cfg.hebbian_lr_bounds[1]
            )
        
        # =================================================================
        # ADAPTATION RULE 5: Eligibility Trace Tau (NEW)
        # =================================================================
        # Goal: All inputs should have equal chance to influence learning
        # IF weight diversity is low (all neurons prefer late inputs) → INCREASE tau
        # This is the key signal for temporal bias in STDP
        if cfg.adapt_eligibility_tau:
            # Update weight diversity EMA
            self.avg_weight_diversity = (1 - alpha_pa) * self.avg_weight_diversity + alpha_pa * weight_diversity
            
            diversity_error = cfg.target_weight_diversity - self.avg_weight_diversity
            
            if diversity_error > 0.2:  # Low diversity (all neurons prefer same inputs)
                # INCREASE tau to give earlier inputs more credit
                adjustment = cfg.meta_lr * diversity_error * 3.0
                self.params['eligibility_tau'] *= (1.0 + adjustment)
            elif diversity_error < -0.1:  # Very high diversity (maybe too much)
                # Slight decrease
                adjustment = cfg.meta_lr * 0.3
                self.params['eligibility_tau'] *= (1.0 - adjustment)
            
            self.params['eligibility_tau'] = np.clip(
                self.params['eligibility_tau'],
                cfg.eligibility_tau_bounds[0],
                cfg.eligibility_tau_bounds[1]
            )
        
        # =================================================================
        # ADAPTATION RULE 6: Heterosynaptic LTD Ratio (NEW)
        # =================================================================
        # Goal: Neurons should specialize on different inputs
        # IF weight diversity is low → INCREASE heterosynaptic LTD
        # More competition = better differentiation
        if cfg.adapt_heterosynaptic:
            diversity_error = cfg.target_weight_diversity - self.avg_weight_diversity
            
            if diversity_error > 0.2:  # Low diversity
                adjustment = cfg.meta_lr * diversity_error * 2.0
                self.params['heterosynaptic_ratio'] *= (1.0 + adjustment)
            elif self.avg_weight_saturation > 0.3:  # Too much saturation
                # Reduce LTD to prevent collapse
                adjustment = cfg.meta_lr * 0.5
                self.params['heterosynaptic_ratio'] *= (1.0 - adjustment)
            
            self.params['heterosynaptic_ratio'] = np.clip(
                self.params['heterosynaptic_ratio'],
                cfg.heterosynaptic_bounds[0],
                cfg.heterosynaptic_bounds[1]
            )
        
        # Record history
        self.history.append({
            'cycle': cycle_num,
            'avg_fr_hz': self.avg_firing_rate_hz,
            'avg_fr_std': self.avg_firing_rate_std,
            'avg_phase_acc': self.avg_phase_accuracy,
            'avg_weight_sat': self.avg_weight_saturation,
            'avg_weight_div': self.avg_weight_diversity,
            'theta_strength': self.params['theta_modulation_strength'],
            'homeostatic_strength': self.params['homeostatic_strength_hz'],
            'som_strength': self.params['som_strength'],
            'hebbian_lr': self.params['hebbian_learning_rate'],
            'eligibility_tau': self.params['eligibility_tau'],
            'heterosynaptic_ratio': self.params['heterosynaptic_ratio'],
        })
        
        return self.params.copy()
    
    def get_params(self) -> Dict[str, float]:
        """Get current parameter values."""
        return self.params.copy()
    
    def print_status(self, cycle_num: int) -> None:
        """Print current meta-learning status."""
        print(f"  Meta-Homeostasis (cycle {cycle_num}):")
        print(f"    Avg firing rate: {self.avg_firing_rate_hz:.1f} Hz (target: {self.config.target_firing_rate_hz})")
        print(f"    Avg phase accuracy: {self.avg_phase_accuracy:.2f} (target: {self.config.target_phase_accuracy})")
        print(f"    Weight saturation: {self.avg_weight_saturation:.2%} (target: {self.config.target_weight_saturation:.0%})")
        print(f"    Weight diversity: {self.avg_weight_diversity:.2f} (target: {self.config.target_weight_diversity})")
        print(f"    Parameters: theta={self.params['theta_modulation_strength']:.2f}, "
              f"homeo={self.params['homeostatic_strength_hz']:.4f}, "
              f"som={self.params['som_strength']:.2f}, "
              f"lr={self.params['hebbian_learning_rate']:.4f}")
        print(f"    NEW params: eligibility_tau={self.params['eligibility_tau']:.1f}, "
              f"heterosynaptic={self.params['heterosynaptic_ratio']:.2f}")


def compute_weight_diversity(weights: torch.Tensor) -> float:
    """Compute weight diversity: fraction of neurons with unique peak inputs.
    
    This metric detects when all neurons prefer the same inputs (bad)
    vs when neurons specialize on different inputs (good).
    
    Args:
        weights: Weight matrix (n_output, n_input)
        
    Returns:
        Diversity score 0-1:
        - 0 = all neurons have same peak input (temporal bias)
        - 1 = all neurons have unique peak inputs (good specialization)
    """
    if weights.dim() != 2:
        return 1.0  # Can't compute for non-2D
    
    n_output, n_input = weights.shape
    
    # Find peak input for each output neuron
    peak_inputs = weights.argmax(dim=1)  # (n_output,)
    
    # Count unique peaks
    unique_peaks = len(torch.unique(peak_inputs))
    
    # Diversity = unique / total (1.0 if all different, low if many same)
    # Normalize by min(n_output, n_input) since that's max possible unique
    max_unique = min(n_output, n_input)
    diversity = unique_peaks / max_unique
    
    return float(diversity)


def compute_temporal_bias(weights: torch.Tensor) -> float:
    """Compute temporal bias: how much weights favor late vs early inputs.
    
    This metric detects recency bias in STDP learning.
    
    Args:
        weights: Weight matrix (n_output, n_input)
        
    Returns:
        Bias score:
        - 0.5 = balanced (equal weight to early and late inputs)
        - >0.5 = late bias (recency effect)
        - <0.5 = early bias
    """
    if weights.dim() != 2:
        return 0.5
    
    n_output, n_input = weights.shape
    
    # Compute mean weight per input position
    mean_per_input = weights.mean(dim=0)  # (n_input,)
    
    # Compute weighted average of input positions
    positions = torch.arange(n_input, dtype=weights.dtype, device=weights.device)
    
    # Normalize weights for each neuron then compute position expectation
    total_weight = mean_per_input.sum()
    if total_weight > 0:
        expected_position = (mean_per_input * positions).sum() / total_weight
        # Normalize to 0-1 range
        bias = expected_position / (n_input - 1)
    else:
        bias = 0.5
    
    return float(bias)


def compute_weight_saturation(weights: torch.Tensor, w_max: float, threshold: float = 0.95) -> float:
    """Compute fraction of weights that are saturated (near w_max).
    
    Args:
        weights: Weight tensor
        w_max: Maximum weight value
        threshold: Fraction of w_max considered "saturated"
        
    Returns:
        Fraction of weights >= threshold * w_max
    """
    saturated = (weights >= threshold * w_max).float().mean().item()
    return saturated


def compute_weight_change(weights: torch.Tensor, prev_weights: torch.Tensor) -> float:
    """Compute mean absolute weight change between timesteps.
    
    Args:
        weights: Current weights
        prev_weights: Previous weights
        
    Returns:
        Mean absolute change
    """
    return float((weights - prev_weights).abs().mean().item())
