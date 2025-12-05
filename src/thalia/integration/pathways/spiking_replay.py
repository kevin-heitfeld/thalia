"""
Spiking Replay Pathway - Fully spiking Hippocampus → Cortex consolidation.

This pathway implements memory replay using spiking neurons with
temporal coding. Sharp-wave ripples (SWRs) trigger compressed replay
of hippocampal patterns to cortex.

Key features:
1. LIF neurons for realistic spike dynamics  
2. Sharp-wave ripple detection and generation
3. Time-compressed replay (accelerated sequences)
4. Phase precession for sequence encoding
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn
from ..spiking_pathway import SpikingPathway, SpikingPathwayConfig, TemporalCoding


@dataclass
class SpikingReplayPathwayConfig(SpikingPathwayConfig):
    """Configuration for spiking replay pathway."""
    
    # Override defaults for replay-specific behavior
    temporal_coding: TemporalCoding = TemporalCoding.LATENCY  # Precise timing for replay
    axonal_delay_ms: float = 5.0  # Slower for consolidation
    
    # Sharp-wave ripple parameters
    ripple_frequency: float = 150.0  # Ripple frequency (Hz)
    ripple_duration: float = 80.0  # Ripple duration (ms)
    ripple_threshold: float = 0.5  # Threshold for ripple detection
    
    # Replay parameters
    compression_factor: float = 5.0  # Time compression during replay
    max_replay_length: int = 10  # Maximum sequence length
    replay_gain: float = 3.0  # Amplification during replay
    
    # Sleep stage parameters
    sleep_stages: bool = True  # Enable sleep stage modulation
    sws_replay_boost: float = 2.0  # Boost during slow-wave sleep


class RippleGenerator(nn.Module):
    """
    Generates sharp-wave ripple oscillations.
    
    Ripples are high-frequency (150-200 Hz) oscillations that occur
    during hippocampal replay.
    """
    
    def __init__(self, frequency: float = 150.0, duration: float = 80.0):
        super().__init__()
        self.frequency = frequency
        self.duration = duration
        self.register_buffer("phase", torch.tensor(0.0))
        self.register_buffer("time_in_ripple", torch.tensor(0.0))
        self.active = False
    
    def start_ripple(self):
        """Start a new ripple event."""
        self.active = True
        self.time_in_ripple.zero_()
        self.phase.zero_()
    
    def step(self, dt: float) -> Tuple[bool, float]:
        """
        Step the ripple generator.
        
        Returns:
            (is_active, ripple_phase): Whether ripple is active and current phase
        """
        if not self.active:
            return False, 0.0
        
        # Update time and phase
        self.time_in_ripple += dt
        self.phase = 2 * torch.pi * self.frequency * self.time_in_ripple / 1000  # Convert to radians
        
        # Check if ripple should end
        if self.time_in_ripple >= self.duration:
            self.active = False
            return False, 0.0
        
        # Ripple envelope (gaussian)
        t_center = self.duration / 2
        sigma = self.duration / 4
        envelope = torch.exp(-0.5 * ((self.time_in_ripple - t_center) / sigma) ** 2)
        
        # Ripple oscillation
        ripple_value = envelope * torch.cos(self.phase)
        
        return True, ripple_value.item()


class SpikingReplayPathway(SpikingPathway):
    """
    Spiking pathway for Hippocampus → Cortex memory consolidation.
    
    Key mechanisms:
    - Sharp-wave ripples trigger replay events
    - Time-compressed spike sequences
    - Phase precession encodes temporal order
    - Only active during sleep/rest states
    """
    
    def __init__(self, config: SpikingReplayPathwayConfig):
        """Initialize spiking replay pathway."""
        # Store replay-specific config
        self.ripple_freq = config.ripple_frequency
        self.ripple_duration = config.ripple_duration
        self.ripple_threshold = config.ripple_threshold
        self.compression_factor = config.compression_factor
        self.max_replay_length = config.max_replay_length
        self.replay_gain = config.replay_gain
        self.sleep_stages = config.sleep_stages
        self.sws_boost = config.sws_replay_boost
        
        # Initialize base spiking pathway
        super().__init__(config)
        
        # Ripple generator
        self.ripple_gen = RippleGenerator(config.ripple_frequency, config.ripple_duration)
        
        # Replay buffer (stores hippocampal patterns)
        self.replay_buffer: List[torch.Tensor] = []
        self.max_buffer_size = 100
        
        # Current state
        self.replay_active = False
        self.current_replay_idx = 0
        self.sleep_stage = "wake"  # wake, rem, sws (slow-wave sleep)
        
        # Replay projection (to cortex space)
        self.replay_projection = nn.Sequential(
            nn.Linear(config.source_size, config.target_size),
            nn.LayerNorm(config.target_size),
        )
        
        # Priority scoring for replay selection
        self.priority_network = nn.Sequential(
            nn.Linear(config.source_size, config.source_size // 2),
            nn.ReLU(),
            nn.Linear(config.source_size // 2, 1),
        )
        
        # Track replay statistics
        self.replay_count = 0
        self.last_ripple_time = 0.0
    
    def store_pattern(self, hippocampal_pattern: torch.Tensor, priority: Optional[float] = None):
        """
        Store a hippocampal pattern for potential replay.
        
        Args:
            hippocampal_pattern: Pattern to store [batch, source_size]
            priority: Optional priority score (higher = more likely to replay)
        """
        # Compute priority if not provided
        if priority is None:
            with torch.no_grad():
                priority = self.priority_network(hippocampal_pattern).mean().item()
        
        # Store pattern with priority
        pattern_entry = {
            "pattern": hippocampal_pattern.detach().clone(),
            "priority": priority,
            "age": 0,
        }
        
        self.replay_buffer.append(pattern_entry)
        
        # Trim buffer if needed (remove lowest priority)
        if len(self.replay_buffer) > self.max_buffer_size:
            # Sort by priority and remove lowest
            self.replay_buffer.sort(key=lambda x: x["priority"], reverse=True)
            self.replay_buffer = self.replay_buffer[:self.max_buffer_size]
    
    def set_sleep_stage(self, stage: str):
        """Set the current sleep stage.
        
        Sleep stages (biologically inspired):
        - wake: Normal awake state, no replay
        - n1: Light sleep (drowsy), minimal replay, falling asleep
        - n2: Light sleep with spindles, moderate replay
        - sws (N3): Slow-wave sleep, maximum consolidation replay
        - rem: REM sleep, stochastic replay for generalization
        """
        assert stage in ["wake", "n1", "n2", "sws", "rem"]
        self.sleep_stage = stage
        
        # Replay is active in N2, SWS, and REM (not in wake or N1)
        if stage in ["n2", "sws", "rem"]:
            self.replay_active = True
        else:
            self.replay_active = False
    
    def trigger_ripple(self) -> bool:
        """
        Trigger a sharp-wave ripple event.
        
        Returns:
            success: Whether ripple was triggered
        """
        if not self.replay_active or len(self.replay_buffer) == 0:
            return False
        
        self.ripple_gen.start_ripple()
        self.replay_count += 1
        
        # Select pattern to replay (prioritized sampling)
        priorities = torch.tensor([e["priority"] for e in self.replay_buffer])
        probs = torch.softmax(priorities * 2, dim=0)  # Temperature = 0.5
        idx = torch.multinomial(probs, 1).item()
        self.current_replay_idx = idx
        
        return True
    
    def replay_step(self, dt: float) -> Optional[torch.Tensor]:
        """
        Perform one step of replay during a ripple.
        
        Args:
            dt: Time step in ms
            
        Returns:
            replay_signal: Cortical reactivation signal, or None if no ripple
        """
        # Check ripple state
        ripple_active, ripple_value = self.ripple_gen.step(dt)
        
        if not ripple_active or self.current_replay_idx >= len(self.replay_buffer):
            return None
        
        # Get pattern to replay
        entry = self.replay_buffer[self.current_replay_idx]
        pattern = entry["pattern"]
        
        # Time-compress the replay (faster dynamics)
        compressed_dt = dt * self.compression_factor
        
        # Process through spiking pathway - forward returns just output spikes
        output_spikes = self.forward(pattern.squeeze(), dt=compressed_dt)
        
        # Modulate by ripple phase
        ripple_modulation = 0.5 * (1 + ripple_value)
        modulated_spikes = output_spikes * ripple_modulation
        
        # Apply replay gain
        gain = self.replay_gain
        if self.sleep_stage == "sws":
            gain *= self.sws_boost
        
        replay_signal = modulated_spikes * gain
        
        # Project to cortex space
        cortical_reactivation = self.replay_projection(replay_signal)
        
        # Age the replayed pattern (reduce priority slightly)
        entry["age"] += 1
        entry["priority"] *= 0.99
        
        return cortical_reactivation
    
    def consolidate(
        self,
        hippocampal_activity: torch.Tensor,
        cortical_activity: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Full consolidation step: store, replay, and update.
        
        Args:
            hippocampal_activity: Current hippocampal state [batch, source_size]
            cortical_activity: Current cortical state [batch, target_size]
            dt: Time step in ms
            
        Returns:
            replay_signal: Cortical reactivation (zeros if no replay)
        """
        batch_size = hippocampal_activity.shape[0]
        device = hippocampal_activity.device
        
        # Store current pattern if awake
        if self.sleep_stage == "wake":
            self.store_pattern(hippocampal_activity)
        
        # Attempt replay if in SWS
        if self.sleep_stage == "sws":
            # Probabilistically trigger ripples
            if torch.rand(1).item() < 0.1 * dt / 10:  # ~10% per 10ms
                self.trigger_ripple()
            
            # Get replay signal
            replay_signal = self.replay_step(dt)
            
            if replay_signal is not None:
                # Learning happens during replay
                self.learn(
                    pre_spikes=hippocampal_activity,
                    post_spikes=cortical_activity + replay_signal,
                    reward=0.5,  # Consolidation reward
                    dt=dt
                )
                return replay_signal
        
        # No replay
        return torch.zeros(batch_size, self.config.target_size, device=device)
    
    def get_diagnostics(self) -> dict:
        """Get replay-specific diagnostics."""
        diag = super().get_diagnostics()
        
        diag.update({
            "buffer_size": len(self.replay_buffer),
            "replay_count": self.replay_count,
            "sleep_stage": self.sleep_stage,
            "replay_active": self.replay_active,
            "ripple_active": self.ripple_gen.active,
        })
        
        if len(self.replay_buffer) > 0:
            priorities = [e["priority"] for e in self.replay_buffer]
            diag["mean_priority"] = sum(priorities) / len(priorities)
            diag["max_priority"] = max(priorities)
        
        return diag
