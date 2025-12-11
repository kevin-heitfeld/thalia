"""
D2 Pathway - Indirect/NoGo Pathway

D2-MSNs express D2 dopamine receptors and form the INDIRECT pathway:
- Projects indirectly through GPe → STN → GPi/SNr
- Dopamine INHIBITS D2 receptors → suppresses action inhibition
- Learning: DA+ → LTD (weaken action suppression)
           DA- → LTP (strengthen action suppression)

Biological role: "NOGO" signal for action suppression
"""

from typing import Dict, Any

import torch

from thalia.core.utils import clamp_weights
from .pathway_base import StriatumPathway, StriatumPathwayConfig


class D2Pathway(StriatumPathway):
    """
    D2-MSN population implementing the indirect/NoGo pathway.
    
    Key features:
    - INVERTED dopamine modulation (DA+ → weaken, DA- → strengthen)
    - Indirect pathway through GPe and STN
    - Suppresses inappropriate actions
    
    Three-factor rule for D2 (INVERTED):
        Δw = eligibility × (-dopamine) × learning_rate
        
    When dopamine is positive (reward):
        - Eligible synapses WEAKEN (LTD) - "stop inhibiting the good action"
        - Non-eligible synapses strengthen slightly (heterosynaptic LTP)
        
    When dopamine is negative (punishment):
        - Eligible synapses STRENGTHEN (LTP) - "inhibit the bad action more"
        - Non-eligible synapses weaken slightly (heterosynaptic LTD)
    
    This implements opponent learning:
    - D1 learns to DO rewarded actions
    - D2 learns to NOT DO punished actions
    """
    
    def __init__(self, config: StriatumPathwayConfig):
        super().__init__(config)
        self.pathway_name = "D2"
    
    def apply_dopamine_modulation(
        self,
        dopamine: float,
        heterosynaptic_ratio: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Apply D2-specific dopamine modulation with INVERTED polarity.
        
        D2 RESPONSE TO DOPAMINE (OPPOSITE OF D1):
        - Dopamine burst (DA+): WEAKEN eligible synapses (LTD)
        - Dopamine dip (DA-): STRENGTHEN eligible synapses (LTP)
        
        This implements the "NOGO" signal: punished actions become more inhibited.
        
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
            # D2 KEY DIFFERENCE: INVERTED dopamine response
            # Multiply dopamine by -1 to flip the polarity
            inverted_dopamine = -dopamine
            
            # Homosynaptic plasticity (eligible synapses)
            # D2: Δw = eligibility × (-dopamine)
            # Positive DA → negative Δw (LTD) - release inhibition on good actions
            # Negative DA → positive Δw (LTP) - increase inhibition on bad actions
            homo_plasticity = self.eligibility * inverted_dopamine
            
            # Heterosynaptic plasticity (non-eligible synapses)
            # Opposite sign for stability
            hetero_plasticity = -self.eligibility * inverted_dopamine * heterosynaptic_ratio
            
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
                'pathway': 'D2',
                'ltp': ltp,
                'ltd': ltd,
                'plasticity': actual_change.abs().sum().item(),
                'mean_eligibility': self.eligibility.mean().item(),
                'dopamine_sign': 'positive' if dopamine > 0 else 'negative' if dopamine < 0 else 'zero',
                'inverted_response': True,  # Flag to indicate D2's inverted learning
            }
