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
            heterosynaptic_ratio: Not used (kept for API compatibility)
            
        Returns:
            Metrics dict from strategy:
                - ltp: Total LTP magnitude
                - ltd: Total LTD magnitude
                - net_change: Net weight change
                - modulator: Inverted dopamine value used
                - eligibility_mean: Average eligibility trace
        """
        # D2 KEY DIFFERENCE: INVERTED dopamine response
        # Use strategy with negated dopamine
        inverted_dopamine = -dopamine
        
        new_weights, metrics = self.learning_strategy.compute_update(
            weights=self.weights.data,
            pre=torch.ones(1, device=self.device),  # Dummy (eligibility already accumulated)
            post=torch.ones(1, device=self.device),  # Dummy (eligibility already accumulated)
            modulator=inverted_dopamine,
        )
        
        # Update weights
        self.weights.data = new_weights
        
        # Add pathway identifier
        metrics['pathway'] = 'D2'
        metrics['dopamine_sign'] = 'positive' if dopamine > 0 else 'negative' if dopamine < 0 else 'zero'
        metrics['inverted_response'] = True  # Flag to indicate D2's inverted learning
        
        return metrics
