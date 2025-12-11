"""
D1 Pathway - Direct/Go Pathway

D1-MSNs express D1 dopamine receptors and form the DIRECT pathway:
- Projects directly to output nuclei (GPi/SNr)
- Dopamine EXCITES D1 receptors → facilitates action initiation
- Learning: DA+ → LTP (strengthen successful actions)
           DA- → LTD (weaken unsuccessful actions)

Biological role: "GO" signal for action selection
"""

from typing import Dict, Any

import torch

from thalia.core.utils import clamp_weights
from .pathway_base import StriatumPathway, StriatumPathwayConfig


class D1Pathway(StriatumPathway):
    """
    D1-MSN population implementing the direct/Go pathway.
    
    Key features:
    - Positive dopamine modulation (DA+ → strengthen, DA- → weaken)
    - Direct inhibitory projections to output nuclei
    - Facilitates action initiation and selection
    
    Three-factor rule for D1:
        Δw = eligibility × dopamine × learning_rate
        
    When dopamine is positive (reward):
        - Eligible synapses strengthen (LTP)
        - Non-eligible synapses weaken slightly (heterosynaptic LTD)
        
    When dopamine is negative (punishment):
        - Eligible synapses weaken (LTD)
        - Non-eligible synapses strengthen slightly (heterosynaptic LTP)
    """
    
    def __init__(self, config: StriatumPathwayConfig):
        super().__init__(config)
        self.pathway_name = "D1"
    
    def apply_dopamine_modulation(
        self,
        dopamine: float,
        heterosynaptic_ratio: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Apply D1-specific dopamine modulation.
        
        D1 RESPONSE TO DOPAMINE:
        - Dopamine burst (DA+): Strengthen eligible synapses (LTP)
        - Dopamine dip (DA-): Weaken eligible synapses (LTD)
        
        This implements the "GO" signal: rewarded actions become stronger.
        
        Args:
            dopamine: Dopamine signal (RPE), typically in [-1, +1]
            heterosynaptic_ratio: Fraction of learning applied to non-eligible synapses
            
        Returns:
            Metrics dict:
                - ltp: Total LTP magnitude
                - ltd: Total LTD magnitude
                - plasticity: Net weight change
                - mean_eligibility: Average eligibility trace
        """
        with torch.no_grad():
            # Homosynaptic plasticity (eligible synapses)
            # Δw = eligibility × dopamine
            # Positive DA → positive Δw (LTP)
            # Negative DA → negative Δw (LTD)
            homo_plasticity = self.eligibility * dopamine
            
            # Heterosynaptic plasticity (non-eligible synapses)
            # Weak opposite-sign changes for stability
            # Helps prevent runaway growth/decay
            hetero_plasticity = -self.eligibility * dopamine * heterosynaptic_ratio
            
            # Total plasticity
            total_plasticity = homo_plasticity + hetero_plasticity
            
            # Apply to weights with clipping
            old_weights = self.weights.data.clone()
            self.weights.data = self.weights.data + total_plasticity
            clamp_weights(self.weights.data, self.config.w_min, self.config.w_max, inplace=True)
            
            # Compute metrics
            actual_change = self.weights.data - old_weights
            ltp = actual_change[actual_change > 0].sum().item()
            ltd = -actual_change[actual_change < 0].sum().item()
            
            return {
                'pathway': 'D1',
                'ltp': ltp,
                'ltd': ltd,
                'plasticity': actual_change.abs().sum().item(),
                'mean_eligibility': self.eligibility.mean().item(),
                'dopamine_sign': 'positive' if dopamine > 0 else 'negative' if dopamine < 0 else 'zero',
            }
