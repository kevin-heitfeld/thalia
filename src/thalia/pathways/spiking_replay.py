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
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
from thalia.managers.component_registry import register_pathway
from thalia.core.base.component_config import PathwayConfig
from thalia.pathways.spiking_pathway import SpikingPathway


@dataclass
class SpikingReplayPathwayConfig(PathwayConfig):
    """Configuration for spiking replay pathway."""

    # Override defaults for replay-specific behavior
    temporal_coding: str = "LATENCY"  # Precise timing for replay
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


@register_pathway(
    "replay",
    aliases=["spiking_replay", "consolidation"],
    description="Spiking Hippocampus → Cortex replay pathway with sharp-wave ripples and time compression",
    version="1.0",
    author="Thalia Project"
)
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
        self.ripple_duration = config.ripple_duration
        self.ripple_threshold = config.ripple_threshold
        self.compression_factor = config.compression_factor
        self.max_replay_length = config.max_replay_length
        self.replay_gain = config.replay_gain
        self.sleep_stages = config.sleep_stages
        self.sws_boost = config.sws_replay_boost

        # Initialize base spiking pathway
        super().__init__(config)

        # Sharp-wave ripple tracking (now from brain oscillator)
        self.ripple_active_local = False  # Track if in ripple event
        self.ripple_start_time = 0.0
        self.current_time = 0.0

        # Replay buffer (stores hippocampal patterns)
        self.replay_buffer: List[torch.Tensor] = []
        self.max_buffer_size = 100

        # Current state
        self.replay_active = False
        self.current_replay_idx = 0
        self.sleep_stage = "wake"  # wake, rem, sws (slow-wave sleep)

        # Replay projection (to cortex space)
        self.replay_projection = nn.Sequential(
            nn.Linear(config.n_input, config.n_output),
            nn.LayerNorm(config.n_output),
        )

        # Priority scoring for replay selection
        self.priority_network = nn.Sequential(
            nn.Linear(config.n_input, config.n_input // 2),
            nn.ReLU(),
            nn.Linear(config.n_input // 2, 1),
        )

        # Track replay statistics
        self.replay_count = 0
        self.last_ripple_time = 0.0

    def store_pattern(self, hippocampal_pattern: torch.Tensor, priority: Optional[float] = None):
        """
        Store a hippocampal pattern for potential replay.

        Args:
            hippocampal_pattern: Pattern to store [n_input] (1D)
            priority: Optional priority score (higher = more likely to replay)
        """
        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        # Ensure 1D input
        if hippocampal_pattern.dim() != 1:
            hippocampal_pattern = hippocampal_pattern.squeeze()

        assert hippocampal_pattern.shape[0] == self.config.n_input, (
            f"SpikingReplayPathway.store_pattern: hippocampal_pattern has shape {hippocampal_pattern.shape} "
            f"but n_input={self.config.n_input}. Check hippocampus output size."
        )

        # Compute priority if not provided
        if priority is None:
            with torch.no_grad():
                # ADR-005: priority_network accepts 1D [n_neurons] directly
                # Convert bool spikes to float for priority network
                pattern_float = hippocampal_pattern.float()
                priority = self.priority_network(pattern_float).mean().item()

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

        Uses brain-wide ripple oscillator for synchronization.

        Returns:
            success: Whether ripple was triggered
        """
        if not self.replay_active or len(self.replay_buffer) == 0:
            return False

        # Mark ripple start (will be detected via brain ripple oscillator)
        self.ripple_active_local = True
        self.ripple_start_time = self.current_time
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

        Uses brain-wide ripple oscillator for synchronized replay.

        Args:
            dt: Time step in ms

        Returns:
            replay_signal: Cortical reactivation signal, or None if no ripple
        """
        # Update current time
        self.current_time += dt

        # Check if we're in a ripple event
        if not self.ripple_active_local:
            return None

        # Check ripple duration (end after configured duration)
        ripple_elapsed = self.current_time - self.ripple_start_time
        if ripple_elapsed >= self.ripple_duration:
            self.ripple_active_local = False
            return None

        # Get ripple phase and amplitude from brain oscillator
        # No fallback - must be provided by Brain for synchronized replay
        if hasattr(self, '_oscillator_phases') and 'ripple' in self._oscillator_phases:
            ripple_phase = self._oscillator_phases['ripple']
        else:
            ripple_phase = 0.0  # No fallback - must be connected to Brain

        # Get ripple amplitude (modulated by other oscillators)
        ripple_amp = 1.0
        if hasattr(self, '_coupled_amplitudes') and 'ripple' in self._coupled_amplitudes:
            ripple_amp = self._coupled_amplitudes['ripple']

        # Compute ripple envelope (gaussian)
        t_center = self.ripple_duration / 2
        sigma = self.ripple_duration / 4
        envelope = torch.exp(torch.tensor(-0.5 * ((ripple_elapsed - t_center) / sigma) ** 2))

        # Ripple value with envelope and oscillation
        ripple_value = envelope * torch.cos(torch.tensor(ripple_phase)) * ripple_amp

        if self.current_replay_idx >= len(self.replay_buffer):
            return None

        # Get pattern to replay
        entry = self.replay_buffer[self.current_replay_idx]
        pattern = entry["pattern"]

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert pattern.shape[-1] == self.config.n_input, (
            f"SpikingReplayPathway.replay_step: pattern has shape {pattern.shape} "
            f"but n_input={self.config.n_input}. Replay buffer corrupted?"
        )

        # Time-compress the replay (faster dynamics)
        compressed_dt = dt * self.compression_factor

        # Process through spiking pathway - forward returns just output spikes
        output_spikes = self.forward(pattern.squeeze())

        # Modulate by ripple phase
        ripple_modulation = 0.5 * (1 + ripple_value.item())
        modulated_spikes = output_spikes * ripple_modulation

        # Apply replay gain
        gain = self.replay_gain
        if self.sleep_stage == "sws":
            gain *= self.sws_boost

        replay_signal = modulated_spikes * gain

        # Output spikes are already in cortical space (target_size) from forward()
        # No need for additional projection - SpikingPathway weights already do this
        cortical_reactivation = replay_signal

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
            hippocampal_activity: Current hippocampal state [n_input] (1D)
            cortical_activity: Current cortical state [n_output] (1D)
            dt: Time step in ms

        Returns:
            replay_signal: Cortical reactivation (zeros if no replay)
        """
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
        return torch.zeros(self.config.n_output, device=device)

    def get_diagnostics(self) -> dict:
        """Get replay-specific diagnostics."""
        diag = super().get_diagnostics()

        diag.update({
            "buffer_size": len(self.replay_buffer),
            "replay_count": self.replay_count,
            "sleep_stage": self.sleep_stage,
            "replay_active": self.replay_active,
            "ripple_active": self.ripple_active_local,  # Now from brain oscillator
        })

        if len(self.replay_buffer) > 0:
            priorities = [e["priority"] for e in self.replay_buffer]
            diag["mean_priority"] = sum(priorities) / len(priorities)
            diag["max_priority"] = max(priorities)

        return diag

    def get_state(self) -> Dict[str, Any]:
        """Get replay pathway state for checkpointing.

        Extends base SpikingPathway state with replay-specific components:
        - replay_projection: Weights and biases
        - priority_network: Weights and biases for pattern prioritization
        - replay_buffer: Stored hippocampal patterns with priorities
        - ripple_gen: Sharp-wave ripple generator state
        - replay_state: Current replay index, sleep stage, statistics
        """
        # Get base pathway state
        state = super().get_state()

        # Add replay-specific state
        state["replay_state"] = {
            "replay_projection": {
                "0.weight": self.replay_projection[0].weight.data.clone(),
                "0.bias": self.replay_projection[0].bias.data.clone(),
                "1.weight": self.replay_projection[1].weight.data.clone(),
                "1.bias": self.replay_projection[1].bias.data.clone(),
            },
            "priority_network": {
                "0.weight": self.priority_network[0].weight.data.clone(),
                "0.bias": self.priority_network[0].bias.data.clone(),
                "2.weight": self.priority_network[2].weight.data.clone(),
                "2.bias": self.priority_network[2].bias.data.clone(),
            },
            "replay_buffer": [
                {
                    "pattern": entry["pattern"].clone(),
                    "priority": entry["priority"],
                    "age": entry["age"],
                }
                for entry in self.replay_buffer
            ],
            "ripple_tracking": {
                "ripple_active_local": self.ripple_active_local,
                "ripple_start_time": self.ripple_start_time,
                "current_time": self.current_time,
            },
            "replay_active": self.replay_active,
            "current_replay_idx": self.current_replay_idx,
            "sleep_stage": self.sleep_stage,
            "replay_count": self.replay_count,
            "last_ripple_time": self.last_ripple_time,
        }

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load replay pathway state from checkpoint.

        Args:
            state: State dictionary from get_state()

        Note:
            Restores base pathway state plus replay-specific components.
        """
        # Load base pathway state
        super().load_state(state)

        # Load replay-specific state
        if "replay_state" in state:
            device = self.weights.device
            replay_state = state["replay_state"]

            # Restore replay projection
            replay_proj = replay_state["replay_projection"]
            self.replay_projection[0].weight.data.copy_(replay_proj["0.weight"].to(device))
            self.replay_projection[0].bias.data.copy_(replay_proj["0.bias"].to(device))
            self.replay_projection[1].weight.data.copy_(replay_proj["1.weight"].to(device))
            self.replay_projection[1].bias.data.copy_(replay_proj["1.bias"].to(device))

            # Restore priority network
            priority_net = replay_state["priority_network"]
            self.priority_network[0].weight.data.copy_(priority_net["0.weight"].to(device))
            self.priority_network[0].bias.data.copy_(priority_net["0.bias"].to(device))
            self.priority_network[2].weight.data.copy_(priority_net["2.weight"].to(device))
            self.priority_network[2].bias.data.copy_(priority_net["2.bias"].to(device))

            # Restore replay buffer
            self.replay_buffer = [
                {
                    "pattern": entry["pattern"].to(device),
                    "priority": entry["priority"],
                    "age": entry["age"],
                }
                for entry in replay_state["replay_buffer"]
            ]

            # Restore ripple tracking (now from brain oscillator)
            ripple_tracking = replay_state["ripple_tracking"]
            self.ripple_active_local = ripple_tracking["ripple_active_local"]
            self.ripple_start_time = ripple_tracking["ripple_start_time"]
            self.current_time = ripple_tracking["current_time"]

            # Restore replay state
            self.replay_active = replay_state["replay_active"]
            self.current_replay_idx = replay_state["current_replay_idx"]
            self.sleep_stage = replay_state["sleep_stage"]
            self.replay_count = replay_state["replay_count"]
            self.last_ripple_time = replay_state["last_ripple_time"]

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete replay pathway state for checkpointing (BrainComponent protocol).

        Returns:
            Dictionary with complete state for checkpointing
        """
        base_state = super().get_full_state()
        # Add replay-specific state (already captured in get_state())
        return base_state

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete replay pathway state from checkpoint (BrainComponent protocol).

        Args:
            state: Dictionary from get_full_state()
        """
        # Load base pathway state (which internally calls load_state())
        super().load_full_state(state)
        # Note: Replay-specific state is loaded via load_state() called by base class

    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Add neurons to replay pathway (extends base implementation).

        Grows the pathway's output dimension (cortex size) and updates
        replay-specific layers accordingly.

        Args:
            n_new: Number of output neurons to add
            initialization: Weight init strategy for base pathway
            sparsity: Connection sparsity for new neurons

        Updates:
            - Base pathway weights/delays/traces (via super())
            - replay_projection: [n_input, n_output] → [n_input, n_output+n_new]
            - priority_network: stays [n_input, n_input//2, 1] (input-only)
            - replay_buffer: patterns stay [n_input] (hippocampus size unchanged)
        """
        old_output_size = self.config.n_output
        device = self.weights.device

        # 1. Grow base pathway (weights, delays, neurons, traces)
        super().add_neurons(n_new, initialization, sparsity)

        new_output_size = self.config.n_output  # Updated by super()

        # 2. Expand replay_projection: [n_input, old_output] → [n_input, new_output]
        # Layer 0: Linear
        old_proj_weight = self.replay_projection[0].weight.data.clone()  # [old_output, n_input]
        old_proj_bias = self.replay_projection[0].bias.data.clone()      # [old_output]

        self.replay_projection[0] = nn.Linear(self.config.n_input, new_output_size).to(device)

        # Copy old weights, initialize new outputs small
        self.replay_projection[0].weight.data[:old_output_size, :] = old_proj_weight
        self.replay_projection[0].weight.data[old_output_size:, :] *= 0.1  # Small init
        self.replay_projection[0].bias.data[:old_output_size] = old_proj_bias
        self.replay_projection[0].bias.data[old_output_size:].zero_()

        # Layer 1: LayerNorm
        self.replay_projection[1] = nn.LayerNorm(new_output_size).to(device)

        # 3. priority_network stays unchanged (operates on input patterns only)
        # No expansion needed - it scores which pattern to replay, not output size

        # 4. replay_buffer patterns stay [n_input] - hippocampus size unchanged
        # Buffer already stores correct-sized patterns, no modification needed
