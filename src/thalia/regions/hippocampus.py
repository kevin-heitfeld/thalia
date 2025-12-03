"""
Hippocampus region implementation.

The hippocampus is specialized for:
- Episodic memory (one-shot learning)
- Sequence learning and recall
- Pattern separation (DG) and pattern completion (CA3)
- Theta-phase timing for encoding vs retrieval

Key differences from other regions:
- Very fast learning (one-shot or few-shot)
- Sparse, decorrelated representations
- Strong recurrent connections (CA3 auto-associative)
- Theta oscillations gate encoding vs retrieval

Learning rules:
- Asymmetric STDP for sequences
- High learning rate (one-shot capable)
- Strong pattern separation via sparse coding
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn

from .base import BrainRegion, RegionConfig, RegionState, LearningRule
from ..core.neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class HippocampusConfig(RegionConfig):
    """Configuration for hippocampus region.
    
    Inherits from RegionConfig which requires n_input and n_output.
    """
    
    # Theta oscillation parameters
    theta_frequency: float = 8.0  # Hz (typical theta: 4-12 Hz)
    theta_phase_encode: float = 0.0  # Radians, peak encoding phase
    theta_phase_retrieve: float = 3.14  # Radians, peak retrieval phase
    theta_phase_width: float = 1.0  # Width of phase window
    
    # One-shot learning parameters (override default)
    learning_rate: float = 0.5  # Very high for one-shot learning
    learning_rate_retrieval: float = 0.01  # Minimal learning during retrieval
    
    # Sparsity for pattern separation
    sparsity_target: float = 0.05  # Only 5% active (DG-like)
    inhibition_strength: float = 2.0  # Strong lateral inhibition
    
    # Sequence learning (asymmetric STDP)
    stdp_tau_plus: float = 20.0  # ms, for forward sequences
    stdp_tau_minus: float = 10.0  # ms, for backward (weaker)
    stdp_asymmetry: float = 3.0  # Forward/backward ratio
    stdp_lr: float = 0.1  # STDP learning rate
    heterosynaptic_ratio: float = 0.3  # LTD for non-active synapses
    
    # Pattern completion
    recurrent_strength: float = 0.3  # CA3-like recurrence
    completion_threshold: float = 0.3  # Fraction of pattern to trigger completion
    
    # Weight constraints (override defaults)
    w_max: float = 3.0  # Allow stronger weights for episodic memory
    w_min: float = 0.0
    
    # Soft bounds for stable learning
    soft_bounds: bool = True
    
    # Synaptic scaling for homeostasis
    synaptic_scaling_enabled: bool = True
    synaptic_scaling_target: float = 0.5  # Target mean weight
    synaptic_scaling_rate: float = 0.001  # Slow scaling rate
    
    # Neuron configuration
    neuron_config: Optional[ConductanceLIFConfig] = None


@dataclass
class HippocampusState(RegionState):
    """State for hippocampus region."""
    
    theta_phase: float = 0.0  # Current theta phase
    is_encoding: bool = True  # Encoding vs retrieval mode
    
    # Sparse activity tracking
    activity_history: Optional[torch.Tensor] = None  # For sparsity control
    
    # Sequence learning
    pre_trace: Optional[torch.Tensor] = None
    post_trace: Optional[torch.Tensor] = None
    sequence_position: int = 0  # Position in current sequence


class ThetaOscillator:
    """Theta rhythm generator for encoding/retrieval gating."""
    
    def __init__(self, config: HippocampusConfig, device: torch.device):
        self.config = config
        self.device = device
        self.phase = 0.0
        self.dt = 1.0  # Will be set from config
        
    def reset(self):
        self.phase = 0.0
        
    def step(self, dt: float = 1.0) -> Tuple[float, bool]:
        """Advance theta phase and return encoding/retrieval mode."""
        self.dt = dt
        
        # Advance phase
        phase_increment = 2 * 3.14159 * self.config.theta_frequency * dt / 1000.0
        self.phase = (self.phase + phase_increment) % (2 * 3.14159)
        
        # Determine if encoding or retrieval phase
        encode_dist = self._phase_distance(self.phase, self.config.theta_phase_encode)
        retrieve_dist = self._phase_distance(self.phase, self.config.theta_phase_retrieve)
        
        is_encoding = encode_dist < retrieve_dist
        
        # Compute encoding/retrieval strength based on phase
        if is_encoding:
            strength = self._gaussian_phase(self.phase, self.config.theta_phase_encode)
        else:
            strength = self._gaussian_phase(self.phase, self.config.theta_phase_retrieve)
            
        return strength, is_encoding
    
    def _phase_distance(self, phase1: float, phase2: float) -> float:
        """Circular distance between phases."""
        diff = abs(phase1 - phase2)
        return min(diff, 2 * 3.14159 - diff)
    
    def _gaussian_phase(self, phase: float, center: float) -> float:
        """Gaussian modulation based on phase distance."""
        dist = self._phase_distance(phase, center)
        return float(torch.exp(torch.tensor(-dist**2 / (2 * self.config.theta_phase_width**2))).item())


class Hippocampus(BrainRegion):
    """
    Hippocampus implementation with one-shot learning and theta-phase gating.
    
    Key features:
    - Fast, one-shot Hebbian learning during encoding phases
    - Minimal plasticity during retrieval phases
    - Sparse coding for pattern separation
    - Asymmetric STDP for sequence learning
    - Recurrent connections for pattern completion
    """
    
    def __init__(self, config: HippocampusConfig):
        """
        Initialize hippocampus.
        
        Args:
            config: Hippocampus configuration with n_input and n_output
        """
        # Store config before calling super().__init__ which calls abstract methods
        self.hippocampus_config = config
        
        # Call parent init (this will call _initialize_weights, _create_neurons, etc.)
        super().__init__(config)
        
        # Recurrent weights (CA3-like auto-association)
        self.rec_weights = nn.Parameter(
            torch.randn(config.n_output, config.n_output, device=self.device) * 0.05
        )
        # No self-connections
        with torch.no_grad():
            self.rec_weights.data.fill_diagonal_(0.0)
        
        # Lateral inhibition for sparsity
        self.inhibition_weights = nn.Parameter(
            torch.ones(config.n_output, config.n_output, device=self.device) * config.inhibition_strength
        )
        with torch.no_grad():
            self.inhibition_weights.data.fill_diagonal_(0.0)
        
        # Theta oscillator
        self.theta = ThetaOscillator(config, self.device)
        
        # Override state with hippocampus-specific state
        self.state = HippocampusState()
        
    def _get_learning_rule(self) -> LearningRule:
        """Hippocampus uses theta-phase dependent one-shot learning."""
        return LearningRule.THETA_PHASE
    
    def _initialize_weights(self) -> torch.Tensor:
        """Initialize feedforward weights."""
        return nn.Parameter(
            torch.randn(
                self.hippocampus_config.n_output, 
                self.hippocampus_config.n_input, 
                device=torch.device(self.hippocampus_config.device)
            ) * 0.1
        )
    
    def _create_neurons(self) -> ConductanceLIF:
        """Create conductance-based LIF neurons."""
        neuron_config = self.hippocampus_config.neuron_config or ConductanceLIFConfig(
            # Use slightly faster dynamics for sequence processing
            g_L=0.04,  # Slightly faster (τ_m ≈ 25ms with C_m=1.0)
            tau_E=3.0,  # Fast excitatory
            tau_I=8.0,  # Slower inhibitory
        )
        return ConductanceLIF(self.hippocampus_config.n_output, neuron_config)
    
    def reset(self) -> None:
        """Reset state for new episode."""
        super().reset()
        
        self.neurons.reset_state(1)
        self.theta.reset()
        
        self.state = HippocampusState(
            membrane=torch.zeros(1, self.config.n_output, device=self.device),
            spikes=torch.zeros(1, self.config.n_output, device=self.device),
            theta_phase=0.0,
            is_encoding=True,
            activity_history=torch.zeros(10, 1, self.config.n_output, device=self.device),
            pre_trace=torch.zeros(1, self.config.n_input, device=self.device),
            post_trace=torch.zeros(1, self.config.n_output, device=self.device),
            sequence_position=0
        )
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset state for new episode with specific batch size."""
        self.neurons.reset_state(batch_size)
        self.theta.reset()
        
        self.state = HippocampusState(
            membrane=torch.zeros(batch_size, self.config.n_output, device=self.device),
            spikes=torch.zeros(batch_size, self.config.n_output, device=self.device),
            theta_phase=0.0,
            is_encoding=True,
            activity_history=torch.zeros(10, batch_size, self.config.n_output, device=self.device),
            pre_trace=torch.zeros(batch_size, self.config.n_input, device=self.device),
            post_trace=torch.zeros(batch_size, self.config.n_output, device=self.device),
            sequence_position=0
        )
        
    def forward(
        self,
        input_spikes: torch.Tensor,
        dt: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Process input through hippocampus.
        
        Args:
            input_spikes: Input spike pattern [batch, n_input]
            dt: Time step in ms
            **kwargs: Additional inputs (ignored)
            
        Returns:
            Output spikes [batch, n_output]
        """
        batch_size = input_spikes.shape[0]
        
        # Ensure state is initialized
        if self.state.membrane is None:
            self.reset_state(batch_size)
        
        # Update theta phase
        theta_strength, is_encoding = self.theta.step(dt)
        self.state.theta_phase = self.theta.phase
        self.state.is_encoding = is_encoding
        
        # Feedforward input
        ff_input = torch.matmul(input_spikes.float(), self.weights.t())
        
        # Recurrent input (pattern completion)
        if self.state.spikes is not None:
            rec_input = torch.matmul(
                self.state.spikes.float(), 
                self.rec_weights.t()
            ) * self.hippocampus_config.recurrent_strength
        else:
            rec_input = torch.zeros_like(ff_input)
        
        # Total excitation
        g_exc = (ff_input + rec_input).clamp(min=0)
        
        # Lateral inhibition for sparsity
        if self.state.spikes is not None:
            inhib = torch.matmul(
                self.state.spikes.float(),
                self.inhibition_weights.t()
            )
        else:
            inhib = torch.zeros_like(g_exc)
        g_inh = inhib.clamp(min=0)
        
        # Run through neurons (returns tuple of spikes, membrane)
        spikes, _ = self.neurons(g_exc, g_inh)
        
        # Apply winner-take-all sparsity if needed
        spikes = self._apply_sparsity(spikes)
        
        # Update STDP traces for sequence learning
        self._update_stdp_traces(input_spikes, spikes, dt)
        
        # Store state
        self.state.spikes = spikes
        # Update activity history (rolling buffer)
        if self.state.activity_history is not None:
            self.state.activity_history = torch.roll(self.state.activity_history, -1, dims=0)
            self.state.activity_history[-1] = spikes
        
        return spikes
    
    def _apply_sparsity(self, spikes: torch.Tensor) -> torch.Tensor:
        """Apply k-winner-take-all for sparse coding."""
        batch_size = spikes.shape[0]
        
        # Target number of active neurons
        k = max(1, int(self.config.n_output * self.hippocampus_config.sparsity_target))
        
        # Get membrane potential for ranking (ConductanceLIF uses 'membrane', LIF uses 'v')
        if hasattr(self.neurons, 'membrane') and self.neurons.membrane is not None:
            v = self.neurons.membrane
        elif hasattr(self.neurons, 'v') and self.neurons.v is not None:
            v = self.neurons.v
        else:
            # No membrane potential available - just do random k-WTA
            sparse_spikes = torch.zeros_like(spikes)
            for b in range(batch_size):
                active_idx = spikes[b].nonzero().squeeze(-1)
                if len(active_idx.shape) == 0:
                    active_idx = active_idx.unsqueeze(0)
                if len(active_idx) > k:
                    # Randomly select k active neurons
                    keep_idx = active_idx[torch.randperm(len(active_idx))[:k]]
                    sparse_spikes[b, keep_idx] = 1.0
                else:
                    sparse_spikes[b] = spikes[b]
            return sparse_spikes
        
        # For each batch, keep only top-k
        sparse_spikes = torch.zeros_like(spikes)
        for b in range(batch_size):
            if spikes[b].sum() > k:
                # Too many active - keep top k by membrane potential
                active_idx = spikes[b].nonzero().squeeze(-1)
                if len(active_idx.shape) == 0:
                    active_idx = active_idx.unsqueeze(0)
                
                if len(active_idx) > 0:
                    active_v = v[b, active_idx]
                    _, top_k_rel = torch.topk(active_v, min(k, len(active_idx)))
                    top_k_idx = active_idx[top_k_rel]
                    sparse_spikes[b, top_k_idx] = 1.0
            else:
                sparse_spikes[b] = spikes[b]
                
        return sparse_spikes
    
    def _update_stdp_traces(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float
    ):
        """Update STDP traces for sequence learning."""
        if self.state.pre_trace is None:
            return
            
        # Decay traces
        decay_pre = torch.exp(torch.tensor(-dt / self.hippocampus_config.stdp_tau_plus))
        decay_post = torch.exp(torch.tensor(-dt / self.hippocampus_config.stdp_tau_minus))
        
        self.state.pre_trace = self.state.pre_trace * decay_pre + pre_spikes.float()
        self.state.post_trace = self.state.post_trace * decay_post + post_spikes.float()
    
    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        dt: float = 1.0,
        force_encoding: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply spike-based STDP learning gated by theta phase.
        
        Uses asymmetric STDP for sequence learning:
        - Forward sequences (pre before post) → strong LTP
        - Backward sequences (post before pre) → weaker LTP/LTD
        
        Learning is modulated by theta phase:
        - Encoding phase: high learning rate for one-shot memory
        - Retrieval phase: minimal plasticity to protect stored patterns
        
        Args:
            input_spikes: Presynaptic activity [batch, n_input]
            output_spikes: Postsynaptic activity [batch, n_output]
            dt: Time step in ms
            force_encoding: Force encoding mode regardless of theta phase
            **kwargs: Additional learning signals (ignored)
            
        Returns:
            Dictionary with learning metrics
        """
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)
        if output_spikes.dim() == 1:
            output_spikes = output_spikes.unsqueeze(0)
            
        cfg = self.hippocampus_config
        
        # Determine learning rate based on theta phase
        if force_encoding or self.state.is_encoding:
            lr = cfg.stdp_lr
            mode = "encoding"
        else:
            lr = cfg.stdp_lr * 0.01  # Minimal learning during retrieval
            mode = "retrieval"
        
        # ======================================================================
        # STDP-based Feedforward Weight Update
        # ======================================================================
        # Use spike traces for proper temporal credit assignment
        if self.state.pre_trace is None or self.state.post_trace is None:
            self._update_stdp_traces(input_spikes, output_spikes, dt)
        
        # LTP: post spike when pre trace is high (pre before post)
        ltp = torch.outer(output_spikes.squeeze(), self.state.pre_trace.squeeze())
        
        # LTD: pre spike when post trace is high (post before pre) - weaker
        ltd = torch.outer(self.state.post_trace.squeeze(), input_spikes.squeeze())
        
        # Apply soft bounds if enabled
        if cfg.soft_bounds:
            w_normalized = (self.weights - cfg.w_min) / (cfg.w_max - cfg.w_min + 1e-6)
            ltp_factor = 1.0 - w_normalized  # More room for LTP when weights are low
            ltd_factor = w_normalized  # More room for LTD when weights are high
            ltp = ltp * ltp_factor
            ltd = ltd * ltd_factor
        
        # Compute weight change with asymmetry (forward sequences stronger)
        dW_ff = lr * (cfg.stdp_asymmetry * ltp - cfg.heterosynaptic_ratio * ltd)
        
        # Apply feedforward weight update
        with torch.no_grad():
            self.weights.data += dW_ff
            self.weights.data.clamp_(cfg.w_min, cfg.w_max)
            
            # Synaptic scaling for homeostasis
            if cfg.synaptic_scaling_enabled:
                mean_weight = self.weights.data.mean()
                scaling_factor = 1.0 + cfg.synaptic_scaling_rate * (cfg.synaptic_scaling_target - mean_weight)
                self.weights.data *= scaling_factor
                self.weights.data.clamp_(cfg.w_min, cfg.w_max)
        
        # ======================================================================
        # STDP-based Recurrent Weight Update (CA3-like auto-association)
        # ======================================================================
        # Asymmetric STDP for sequences: connect neurons that fire in sequence
        pre_trace_mean = self.state.pre_trace.mean(dim=0)
        post_trace_mean = self.state.post_trace.mean(dim=0)
        post_mean = output_spikes.float().mean(dim=0)
        pre_mean = input_spikes.float().mean(dim=0)
        
        # Forward sequence (pre trace × current post) - strong for sequences
        n_out = min(self.config.n_output, pre_trace_mean.shape[0])
        dW_forward = torch.outer(post_mean, pre_trace_mean[:n_out]) * cfg.stdp_asymmetry
        
        # Backward (current pre × post trace) - weaker
        n_out = min(self.config.n_output, pre_mean.shape[0])
        dW_backward = torch.outer(post_trace_mean, pre_mean[:n_out])
        
        dW_rec = lr * 0.5 * (dW_forward + dW_backward)
        
        # Apply soft bounds to recurrent weights
        if cfg.soft_bounds:
            w_rec_norm = (self.rec_weights - cfg.w_min) / (cfg.w_max - cfg.w_min + 1e-6)
            dW_rec = torch.where(dW_rec > 0, dW_rec * (1 - w_rec_norm), dW_rec * w_rec_norm)
        
        with torch.no_grad():
            self.rec_weights.data += dW_rec
            self.rec_weights.data.fill_diagonal_(0.0)  # No self-connections
            self.rec_weights.data.clamp_(cfg.w_min, cfg.w_max)
        
        # Compute metrics
        total_ltp = (dW_ff > 0).sum().item()
        total_ltd = (dW_ff < 0).sum().item()
        
        return {
            "learning_rate_used": lr,
            "mode": mode,
            "theta_phase": self.state.theta_phase,
            "ltp_count": total_ltp,
            "ltd_count": total_ltd,
            "ff_weight_mean": self.weights.data.mean().item(),
            "rec_weight_mean": self.rec_weights.data.mean().item(),
            "sparsity": output_spikes.float().mean().item()
        }
    
    def store_pattern(
        self,
        pattern: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Store a pattern in one-shot (episodic memory).
        
        Args:
            pattern: Pattern to store [batch, neurons] or [neurons]
            context: Optional context/cue pattern
            
        Returns:
            Storage metrics
        """
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)
            
        # Use pattern as both pre and post for auto-association
        metrics = self.learn(pattern, pattern, dt=1.0, force_encoding=True)
        
        # If context provided, associate context -> pattern
        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0)
            self.learn(context, pattern, dt=1.0, force_encoding=True)
        
        return {
            **metrics,
            "pattern_sparsity": pattern.float().mean().item(),
            "storage_type": "one-shot"
        }
    
    def recall_pattern(
        self,
        cue: torch.Tensor,
        n_iterations: int = 5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Recall a pattern from partial cue via pattern completion.
        
        Args:
            cue: Partial pattern or context cue [batch, neurons]
            n_iterations: Number of recurrent iterations for completion
            
        Returns:
            Tuple of (recalled pattern, recall metrics)
        """
        if cue.dim() == 1:
            cue = cue.unsqueeze(0)
            
        batch_size = cue.shape[0]
        self.reset_state(batch_size)
        
        # Force retrieval mode (minimal learning)
        original_is_encoding = self.state.is_encoding
        self.state.is_encoding = False
        
        # Initial activation from cue
        output = cue.clone()
        overlap_history = []
        
        for i in range(n_iterations):
            # Forward pass with current activity as input
            output = self.forward(output, dt=1.0)
            
            # Track overlap with cue
            overlap = (output * cue).sum() / (cue.sum() + 1e-6)
            overlap_history.append(overlap.item())
        
        # Restore encoding state
        self.state.is_encoding = original_is_encoding
        
        return output, {
            "n_iterations": n_iterations,
            "final_activity": output.float().mean().item(),
            "cue_overlap": overlap_history[-1] if overlap_history else 0.0,
            "convergence": overlap_history
        }
    
    def store_sequence(
        self,
        sequence: List[torch.Tensor],
        repetitions: int = 1
    ) -> Dict[str, Any]:
        """
        Store a sequence of patterns.
        
        Args:
            sequence: List of patterns forming a sequence
            repetitions: Number of times to present sequence
            
        Returns:
            Sequence storage metrics
        """
        batch_size = sequence[0].shape[0] if sequence[0].dim() > 1 else 1
        
        total_metrics: Dict[str, Any] = {
            "sequence_length": len(sequence),
            "repetitions": repetitions,
            "position_metrics": []
        }
        
        for rep in range(repetitions):
            self.reset_state(batch_size)
            self.state.sequence_position = 0
            
            for i, pattern in enumerate(sequence):
                if pattern.dim() == 1:
                    pattern = pattern.unsqueeze(0)
                
                # Forward pass
                output = self.forward(pattern, dt=1.0)
                
                # Learn current pattern
                metrics = self.learn(pattern, output, dt=1.0, force_encoding=True)
                
                if rep == 0:  # Only store on first repetition
                    total_metrics["position_metrics"].append({
                        "position": i,
                        **metrics
                    })
                
                self.state.sequence_position = i + 1
        
        return total_metrics
    
    def get_state(self) -> HippocampusState:
        """Get current state."""
        return self.state

    # =========================================================================
    # RATE-CODED / HETEROASSOCIATIVE API
    # These methods provide simpler interfaces for rate-coded experiments
    # =========================================================================
    
    def store_association(
        self,
        cue: torch.Tensor,
        target: torch.Tensor,
        learning_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Store a heteroassociative cue → target mapping.
        
        This uses direct Hebbian learning on the weight matrix for 
        rate-coded pattern association (not spike-based).
        
        Args:
            cue: Input cue pattern [n_input] or [batch, n_input]
            target: Target pattern to retrieve [n_output] or [batch, n_output]
            learning_rate: Optional override for learning rate
            
        Returns:
            Dictionary with storage metrics
        """
        if cue.dim() == 1:
            cue = cue.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
            
        lr = learning_rate if learning_rate is not None else self.hippocampus_config.learning_rate
        
        # Hebbian association: W += lr * target^T @ cue
        # This creates mapping: W @ cue ≈ target
        cue_norm = cue / (cue.norm(dim=1, keepdim=True) + 1e-6)
        target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-6)
        
        dW = lr * torch.matmul(target_norm.t(), cue_norm)
        
        with torch.no_grad():
            self.weights.data += dW
            self.weights.data.clamp_(self.hippocampus_config.w_min, self.hippocampus_config.w_max)
        
        return {
            "cue_dim": cue.shape[-1],
            "target_dim": target.shape[-1],
            "weight_change_norm": dW.norm().item(),
            "weight_mean": self.weights.data.mean().item()
        }
    
    def retrieve_association(
        self,
        cue: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Retrieve the target pattern associated with a cue.
        
        This performs a simple linear readout through the weight matrix,
        suitable for rate-coded patterns.
        
        Args:
            cue: Input cue pattern [n_input] or [batch, n_input]
            normalize: Whether to normalize the output
            
        Returns:
            Retrieved target pattern [batch, n_output]
        """
        if cue.dim() == 1:
            cue = cue.unsqueeze(0)
        
        # Linear readout
        output = torch.matmul(cue, self.weights.t())
        
        # Optional normalization
        if normalize:
            output = torch.tanh(output)
        
        return output.squeeze(0) if output.shape[0] == 1 else output
    
    def encode_rate(
        self, 
        input_pattern: torch.Tensor,
        use_recurrent: bool = True
    ) -> torch.Tensor:
        """
        Encode a rate-coded input pattern.
        
        Simple linear encoding through weights with tanh activation.
        Useful for experiments that don't need full spiking dynamics.
        
        Args:
            input_pattern: Rate-coded input [n_input] or [batch, n_input]
            use_recurrent: Whether to apply recurrent dynamics (default True)
            
        Returns:
            Encoded pattern [batch, n_output]
        """
        if input_pattern.dim() == 1:
            input_pattern = input_pattern.unsqueeze(0)
        
        # Feedforward encoding
        encoded = torch.matmul(input_pattern, self.weights.t())
        
        # Add recurrent dynamics for pattern completion (optional)
        if use_recurrent and self.rec_weights is not None:
            rec_input = torch.matmul(encoded, self.rec_weights.t())
            encoded = encoded + self.hippocampus_config.recurrent_strength * rec_input
        
        return torch.tanh(encoded).squeeze(0) if encoded.shape[0] == 1 else torch.tanh(encoded)

    def learn_hebbian_rate(
        self,
        input_pattern: torch.Tensor,
        output_pattern: torch.Tensor,
        learning_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Apply Hebbian learning with rate-coded patterns.
        
        Updates feedforward weights to strengthen input→output mapping.
        
        Args:
            input_pattern: Input pattern [n_input] or [batch, n_input]
            output_pattern: Output pattern [n_output] or [batch, n_output]
            learning_rate: Optional override for learning rate
            
        Returns:
            Dictionary with learning metrics
        """
        if input_pattern.dim() == 1:
            input_pattern = input_pattern.unsqueeze(0)
        if output_pattern.dim() == 1:
            output_pattern = output_pattern.unsqueeze(0)
        
        lr = learning_rate if learning_rate is not None else self.hippocampus_config.learning_rate
        
        # Hebbian: dW = lr * output^T @ input
        # weights are [n_output, n_input], so dW should be same shape
        input_norm = input_pattern / (input_pattern.norm(dim=1, keepdim=True) + 1e-6)
        output_norm = output_pattern / (output_pattern.norm(dim=1, keepdim=True) + 1e-6)
        
        dW = lr * torch.matmul(output_norm.t(), input_norm)
        
        with torch.no_grad():
            self.weights.data += dW
            self.weights.data.clamp_(self.hippocampus_config.w_min, self.hippocampus_config.w_max)
        
        return {
            "weight_change_norm": float(dW.norm().item()),
            "weight_mean": float(self.weights.data.mean().item()),
            "learning_rate": lr
        }
