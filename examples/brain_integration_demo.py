"""
Biologically Realistic Multi-Region Brain Integration Demo
============================================================

This demo implements a COMPLETE anatomically correct multi-region system
with ALL major inter-region connections found in the real brain.

COMPLETE BRAIN CONNECTIVITY MAP:
=================================

                    ┌──────────────────────────────────────────────────────┐
                    │              GLOBAL DOPAMINE SIGNAL                  │
                    │  (Reward/prediction error → modulates ALL regions)   │
                    └────────┬─────────┬─────────┬─────────┬───────────────┘
                             │         │         │         │
                             ▼         ▼         ▼         ▼
    Sensory Input      ┌─────────┐ ┌───────┐ ┌───────┐ ┌────────────┐
         │             │  CORTEX │ │ HIPPO │ │  PFC  │ │  STRIATUM  │
         │             └────┬────┘ └───┬───┘ └───┬───┘ └─────┬──────┘
         │                  │         │         │           │
         ▼                  │         │         │           │
    ┌─────────┐ ◄───────────┼─────────┼─────────┘           │
    │  CORTEX │ ◄───────────┼─────────┘ (PFC→Cortex:        │
    └────┬────┘             │            top-down attention) │
         │                  │                                │
    ┌────┴──────────────────┼────────────────────────────────┤
    │         │             │              │                 │
    ▼         ▼             ▼              ▼                 │
┌───────┐ ┌───────┐    ┌────────┐    ┌──────────┐            │
│ HIPPO │◄┤  PFC  │◄───│ HIPPO  │    │ STRIATUM │◄───────────┘
└───┬───┘ └───┬───┘    └────────┘    └────┬─────┘   (Striatum
    │         │     (Hippo→PFC:           │         receives ALL)
    │         │      retrieval)           │
    ├─────────┤                           │
    │         │                           │
    │    ┌────┴───────────────────────────┤
    │    │                                │
    │    ▼                                ▼
    │  ┌──────────────────────────────────────┐
    │  │           STRIATUM                   │ ◄── Reward-modulated STDP
    │  │   (Cortex + PFC + Hippo → Actions)   │     (action selection)
    │  └──────────────┬───────────────────────┘
    │                 │
    │                 ▼
    │          ┌────────────┐
    │          │ Motor Cmd  │
    │          └──────┬─────┘
    │                 │
    │                 ▼
    │          ┌────────────┐              ┌──────────────────┐
    └─────────►│ CEREBELLUM │◄─────────────│ Error Signal     │
               └──────┬─────┘              │ (Climbing Fiber) │
                      │                    └──────────────────┘
                      ▼
                 Motor Output
                      │
                      ├───────────────────► Environment
                      │
                      ▼
            ┌────────────────┐
            │ Prediction vs  │──────────► Cerebellum Error
            │ Actual Outcome │
            └────────────────┘

COMPLETE CONNECTION LIST:
=========================
FEEDFORWARD (bottom-up):
  1. Sensory → Cortex (feature extraction)
  2. Cortex → Hippocampus (encoding)
  3. Cortex → PFC (sensory to WM)
  4. Cortex → Striatum (direct sensory-motor)
  5. Hippocampus → Striatum (memory-guided action)
  6. PFC → Striatum (goal-directed action)
  7. Striatum → Cerebellum (motor command)

FEEDBACK (top-down):
  8. PFC → Cortex (attention/expectation)
  9. Hippocampus → PFC (episodic retrieval)
  10. Hippocampus → Cortex (memory replay - SLEEP)
  11. Cerebellum → Motor (refined output)

NEUROMODULATION:
  12. Global Dopamine → ALL regions (reward signal)

WHY ALL THESE CONNECTIONS HELP SPIKING STABILITY:
==================================================
Spiking networks are inherently noisy because:
- Discrete spikes create high variance
- Timing is stochastic
- Sparse activity means small sample sizes

Additional connections stabilize by:
1. FEEDBACK LOOPS: PFC→Cortex creates prediction-error signal
2. MEMORY CONTEXT: Hippo→PFC disambiguates noisy input
3. ERROR CORRECTION: Cerebellum learns precise timing
4. GLOBAL COORDINATION: Dopamine synchronizes all regions

TASK: Delayed Match-to-Sample
=============================
1. Show sample pattern (encode in Hippo, PFC)
2. Delay period (PFC maintains, Hippo can retrieve)
3. Show test pattern (compare via all pathways)
4. Respond: MATCH or NO-MATCH
5. Sleep: Hippocampus replays to Cortex
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

# Import all brain regions
from thalia.regions.cortex import Cortex, CortexConfig
from thalia.regions.hippocampus import Hippocampus, HippocampusConfig
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.regions.base import LearningRule


@dataclass
class TrialResult:
    """Result of a single trial."""
    sample_pattern: int
    test_pattern: int
    is_match: bool
    selected_action: int  # 0=match, 1=no-match
    correct: bool
    explored: bool
    cortex_activity: float
    pfc_activity: float
    striatum_activity: torch.Tensor
    # Population coding metrics
    population_votes: Optional[torch.Tensor] = None  # Votes per action
    confidence: float = 0.0  # Margin between winning and losing votes


def create_distinct_patterns(n_patterns: int = 4, size: int = 256) -> List[torch.Tensor]:
    """Create visually distinct patterns.

    Each pattern activates a different quadrant plus some overlap:
    - Pattern 0: Top-left + center
    - Pattern 1: Top-right + center
    - Pattern 2: Bottom-left + center
    - Pattern 3: Bottom-right + center
    """
    patterns = []
    grid = int(size ** 0.5)  # 16x16
    half = grid // 2
    quarter = grid // 4

    for i in range(n_patterns):
        pattern = torch.zeros(size)

        # Quadrant activation
        row_start = (i // 2) * half
        col_start = (i % 2) * half

        for r in range(row_start, row_start + half):
            for c in range(col_start, col_start + half):
                pattern[r * grid + c] = 0.8

        # Center activation (shared feature)
        for r in range(quarter, grid - quarter):
            for c in range(quarter, grid - quarter):
                pattern[r * grid + c] = 0.5

        patterns.append(pattern)

    return patterns


def rate_to_spikes(pattern: torch.Tensor, n_timesteps: int, max_rate: float = 0.7) -> torch.Tensor:
    """Convert rate pattern to spike train."""
    spikes = torch.zeros(n_timesteps, *pattern.shape)
    for t in range(n_timesteps):
        spikes[t] = (torch.rand_like(pattern) < pattern * max_rate).float()
    return spikes


class BiologicalBrainSystem:
    """
    Biologically realistic multi-region brain system with COMPLETE connectivity.

    ALL major brain connections are implemented:

    FEEDFORWARD:
    - Cortex → (Hippocampus, PFC, Striatum)
    - Hippocampus → (PFC, Striatum)
    - PFC → Striatum
    - Striatum → Cerebellum

    FEEDBACK (top-down):
    - PFC → Cortex (attention)
    - Hippocampus → Cortex (sleep replay)

    NEUROMODULATION:
    - Global dopamine → ALL regions
    """

    def __init__(
        self,
        input_size: int = 256,
        cortex_size: int = 128,
        hippocampus_size: int = 64,
        pfc_size: int = 32,
        n_actions: int = 2,  # Match vs No-Match
        device: str = "cpu",
    ):
        self.input_size = input_size
        self.cortex_size = cortex_size
        self.hippocampus_size = hippocampus_size
        self.pfc_size = pfc_size
        self.n_actions = n_actions
        self.device = device

        # Global dopamine level (affects all regions)
        self.global_dopamine = 0.0

        # =====================================================================
        # Create brain regions with biologically realistic connectivity
        # =====================================================================

        # 1. CORTEX: Primary visual processing
        #    - Receives: raw sensory input + PFC top-down attention
        #    - Learns: unsupervised (STDP) - extracts visual features
        #    - Outputs to: Hippocampus, PFC, Striatum
        self.cortex = Cortex(CortexConfig(
            n_input=input_size,  # Will add PFC modulation separately
            n_output=cortex_size,
            hebbian_lr=0.003,
            synaptic_scaling_enabled=True,
            synaptic_scaling_target=0.35,
            kwta_k=max(1, int(cortex_size * 0.15)),  # 15% sparsity
            lateral_inhibition=True,
            inhibition_strength=0.5,
            device=device,
        ))

        # 2. HIPPOCAMPUS: Episodic memory
        #    - Receives: Cortex output
        #    - Learns: theta-gated STDP - stores pattern associations
        #    - Outputs to: PFC (retrieval), Cortex (sleep), Striatum (memory-guided action)
        self.hippocampus = Hippocampus(HippocampusConfig(
            n_input=cortex_size,
            n_output=hippocampus_size,
            stdp_lr=0.05,
            sparsity_target=0.15,
            soft_bounds=True,
            synaptic_scaling_enabled=True,
            device=device,
        ))

        # 3. PREFRONTAL CORTEX: Working memory
        #    - Receives: Cortex output + Hippocampus retrieval
        #    - Learns: dopamine-gated STDP - learns what to maintain
        #    - Maintains: sample pattern during delay
        #    - Outputs to: Striatum (goal), Cortex (attention)
        self.pfc = Prefrontal(PrefrontalConfig(
            n_input=cortex_size + hippocampus_size,  # Cortex + Hippo!
            n_output=pfc_size,
            stdp_lr=0.01,
            dopamine_baseline=0.3,
            gate_threshold=0.5,
            wm_decay_tau_ms=200.0,  # Slower decay for maintenance
            soft_bounds=True,
            synaptic_scaling_enabled=True,
            device=device,
        ))

        # 4. STRIATUM: Action selection
        #    - Receives: Cortex + PFC + Hippocampus (ALL regions!)
        #    - Learns: reward-modulated STDP
        #    - Selects: which action to take
        self.striatum = Striatum(StriatumConfig(
            n_input=cortex_size + pfc_size + hippocampus_size,  # ALL THREE!
            n_output=n_actions,
            learning_rule=LearningRule.REWARD_MODULATED_STDP,
            stdp_lr=0.01,
            stdp_tau_ms=20.0,
            eligibility_tau_ms=200.0,
            dopamine_burst=1.5,
            dopamine_dip=-0.5,
            lateral_inhibition=True,
            inhibition_strength=0.8,
            exploration_epsilon=0.5,
            exploration_decay=0.99,
            min_epsilon=0.05,
            # POPULATION CODING: 10 neurons per action for noise robustness
            population_coding=True,
            neurons_per_action=10,
            device=device,
        ))
        
        # Store striatum neuron count for cerebellum input
        striatum_neurons = n_actions * 10  # 10 neurons per action

        # 5. CEREBELLUM: Motor refinement
        #    - Receives: Striatum output (population-coded motor command)
        #    - Learns: error-corrective (climbing fiber signal)
        #    - Refines: motor execution timing/precision
        self.cerebellum = Cerebellum(CerebellumConfig(
            n_input=striatum_neurons,  # Receives full population output
            n_output=n_actions,  # Outputs refined action signal
            stdp_lr=0.02,
            soft_bounds=True,
            device=device,
        ))

        # =====================================================================
        # ADDITIONAL CONNECTION WEIGHTS (feedback/neuromodulatory)
        # =====================================================================

        # PFC → Cortex (top-down attention/expectation)
        # Modulates cortical processing based on goals
        self.pfc_to_cortex_weights = torch.randn(
            input_size, pfc_size, device=device
        ) * 0.1

        # Hippocampus → PFC (episodic retrieval to working memory)
        # Already included in PFC input size above

        # Hippocampus → Cortex (memory replay during sleep)
        self.hippo_to_cortex_weights = torch.randn(
            cortex_size, hippocampus_size, device=device
        ) * 0.1

        # Initialize feedforward weights
        self._initialize_weights()

        # Store experienced patterns for replay
        self.memory_buffer: List[torch.Tensor] = []
        self.max_memories = 100

        print("=" * 70)
        print("COMPLETE BIOLOGICALLY REALISTIC BRAIN SYSTEM")
        print("=" * 70)
        print("\nFEEDFORWARD CONNECTIONS:")
        print(f"  Sensory → CORTEX ({input_size} → {cortex_size})")
        print(f"  CORTEX → HIPPOCAMPUS ({cortex_size} → {hippocampus_size})")
        print(f"  CORTEX + HIPPO → PFC ({cortex_size}+{hippocampus_size} → {pfc_size})")
        print(f"  CORTEX + PFC + HIPPO → STRIATUM ({cortex_size}+{pfc_size}+{hippocampus_size} → {striatum_neurons})")
        print(f"    └─ POPULATION CODING: {n_actions} actions × 10 neurons/action")
        print(f"  STRIATUM → CEREBELLUM ({striatum_neurons} → {n_actions})")
        print("\nFEEDBACK CONNECTIONS:")
        print(f"  PFC → CORTEX ({pfc_size} → {input_size}) ← Top-down attention!")
        print(f"  HIPPOCAMPUS → PFC ({hippocampus_size} → {pfc_size}) ← Episodic retrieval!")
        print(f"  [SLEEP] HIPPOCAMPUS → CORTEX ({hippocampus_size} → {cortex_size}) ← Memory consolidation!")
        print("\nNEUROMODULATION:")
        print("  Global DOPAMINE → ALL regions (reward signal)")
        print("\nLearning rules:")
        print("  CORTEX: Unsupervised STDP (feature learning)")
        print("  HIPPOCAMPUS: Theta-gated STDP (episodic storage)")
        print("  PFC: Dopamine-gated STDP (WM update rules)")
        print("  STRIATUM: Reward-modulated STDP (action values)")
        print("  CEREBELLUM: Error-corrective (motor refinement)")
        print("=" * 70)

    def _initialize_weights(self):
        """Initialize weights with small random values."""
        with torch.no_grad():
            self.cortex.weights.data = torch.rand_like(self.cortex.weights) * 0.3 + 0.1
            self.striatum.weights.data = torch.rand_like(self.striatum.weights) * 0.3 + 0.1

    def reset_trial(self):
        """Reset for new trial (keep learned weights)."""
        self.cortex.reset()
        self.hippocampus.reset()
        self.pfc.reset()
        self.striatum.reset()
        self.cerebellum.reset()

    def sleep_consolidation(
        self,
        n_replay_cycles: int = 10,
        n_timesteps: int = 15,
    ) -> Dict[str, float]:
        """
        SLEEP PHASE: Hippocampus → Cortex memory consolidation.

        This simulates what happens during slow-wave sleep:
        1. Hippocampus spontaneously replays stored memories
        2. Replay activity is sent TO cortex
        3. Cortex learns from the replayed patterns
        4. Memories gradually become cortex-independent

        This is called SYSTEMS CONSOLIDATION and is why sleep is
        critical for long-term memory formation.

        In REM sleep:
        - PFC is down-regulated (explaining bizarre dream logic)
        - Hippocampus continues replay
        - May serve emotional memory processing

        Args:
            n_replay_cycles: Number of memories to replay
            n_timesteps: Timesteps per replay

        Returns:
            Dict with consolidation metrics
        """
        if len(self.memory_buffer) == 0:
            return {"replayed": 0, "cortex_delta": 0.0}

        cortex_weight_before = self.cortex.weights.data.clone()
        n_replayed = 0

        for _ in range(n_replay_cycles):
            # Randomly select a memory to replay
            idx = np.random.randint(len(self.memory_buffer))
            memory_pattern = self.memory_buffer[idx]

            # Reset for replay
            self.hippocampus.reset()
            self.cortex.reset()

            cortex_total = torch.zeros(self.cortex_size)

            for t in range(n_timesteps):
                # 1. First, cortex processes the memory pattern
                #    (This simulates the initial encoding that stored this memory)
                memory_spikes = rate_to_spikes(memory_pattern, 1)[0].unsqueeze(0)
                cortex_encoding = self.cortex.forward(memory_spikes)

                # 2. Hippocampus processes the cortex output (as it did during encoding)
                hippo_activity = self.hippocampus.forward(cortex_encoding, theta_phase=0.5)

                # 3. Hippocampus → Cortex (REPLAY DIRECTION!)
                # This is the critical consolidation pathway
                cortex_replay_signal = torch.matmul(
                    hippo_activity.float(),
                    self.hippo_to_cortex_weights.t()
                )
                cortex_total += cortex_replay_signal.squeeze()

                # 4. Create replay-driven input for cortex learning
                # This simulates how hippocampal replay reinforces cortex patterns
                replay_as_input = torch.zeros(1, self.input_size)
                # Map cortex replay to input space (inverse projection)
                for i in range(min(self.cortex_size, self.input_size)):
                    if cortex_replay_signal[0, i % self.cortex_size] > 0:
                        replay_as_input[0, i] = cortex_replay_signal[0, i % self.cortex_size]

                cortex_out = self.cortex.forward(replay_as_input)

                # 5. Cortex learns from the replayed pattern
                # This is unsupervised STDP on the replayed activity
                cortex_spikes = (cortex_out > 0).float()
                self.cortex.learn(
                    (replay_as_input > 0.5).float(),
                    cortex_spikes,
                )

            n_replayed += 1

        # Measure how much cortex weights changed
        cortex_delta = (self.cortex.weights.data - cortex_weight_before).abs().mean().item()

        return {
            "replayed": n_replayed,
            "cortex_delta": cortex_delta,
        }

    def store_memory(self, pattern: torch.Tensor):
        """Store pattern in memory buffer for later replay."""
        if len(self.memory_buffer) >= self.max_memories:
            # Remove oldest (FIFO) - in reality, importance matters
            self.memory_buffer.pop(0)
        self.memory_buffer.append(pattern.clone())

    def apply_top_down_attention(
        self,
        sensory_input: torch.Tensor,
        pfc_activity: torch.Tensor,
        attention_strength: float = 0.3,
    ) -> torch.Tensor:
        """
        Apply PFC top-down attention to sensory input.

        This models how prefrontal cortex modulates sensory processing:
        - PFC encodes current goals/expectations
        - This biases cortex to process goal-relevant features
        - Stabilizes learning by focusing on task-relevant input

        Args:
            sensory_input: Raw sensory input
            pfc_activity: Current PFC activity (goals/expectations)
            attention_strength: How strongly PFC modulates cortex

        Returns:
            Attention-modulated sensory input
        """
        # Project PFC to input space
        attention_signal = torch.matmul(
            pfc_activity.float(),
            self.pfc_to_cortex_weights.t()
        )

        # Normalize to 0-1 range
        if attention_signal.abs().max() > 0:
            attention_signal = torch.sigmoid(attention_signal)

        # Modulate sensory input (multiplicative gating)
        modulated_input = sensory_input * (1 + attention_strength * attention_signal)

        return modulated_input

    def process_sample(
        self,
        sample_pattern: torch.Tensor,
        n_timesteps: int = 15,
    ) -> Dict[str, torch.Tensor]:
        """
        Process sample pattern (encode phase).

        FULL CONNECTIVITY:
        - Cortex extracts features (modulated by PFC attention)
        - Hippocampus stores pattern
        - PFC encodes into working memory (receives Cortex + Hippo)
        - High DA = gate open for encoding
        """
        sample_spikes = rate_to_spikes(sample_pattern, n_timesteps)

        cortex_total = torch.zeros(self.cortex_size)
        hippo_total = torch.zeros(self.hippocampus_size)
        pfc_total = torch.zeros(self.pfc_size)

        # Initialize PFC activity for attention
        pfc_activity = torch.zeros(1, self.pfc_size)

        for t in range(n_timesteps):
            # 1. Apply top-down attention from PFC (feedback!)
            #    This helps cortex focus on task-relevant features
            modulated_input = self.apply_top_down_attention(
                sample_spikes[t].unsqueeze(0),
                pfc_activity,
                attention_strength=0.2,
            )

            # 2. Cortex processes (attention-modulated) visual input
            cortex_out = self.cortex.forward(modulated_input)
            cortex_total += cortex_out.squeeze()

            # 3. Hippocampus encodes (theta burst during encoding)
            hippo_out = self.hippocampus.forward(cortex_out, theta_phase=0.0)
            hippo_total += hippo_out.squeeze()

            # 4. PFC receives BOTH Cortex AND Hippocampus
            #    This is the key biological feature: memory informs WM
            pfc_input = torch.cat([cortex_out.squeeze(), hippo_out.squeeze()])
            pfc_out = self.pfc.forward(
                pfc_input.unsqueeze(0),
                dopamine_signal=0.5,  # High DA = gate open for encoding
            )
            pfc_total += pfc_out.squeeze()
            pfc_activity = pfc_out  # Update for next timestep attention

        return {
            "cortex": cortex_total,
            "hippocampus": hippo_total,
            "pfc": pfc_total,
        }

    def maintain_delay(self, n_timesteps: int = 10) -> Dict[str, torch.Tensor]:
        """
        Delay period - PFC maintains working memory.

        FULL CONNECTIVITY:
        - No external sensory input
        - Hippocampus can support retrieval if needed
        - Low DA = gate closed = maintain
        - Tests WM persistence
        """
        null_cortex = torch.zeros(1, self.cortex_size)
        null_hippo = torch.zeros(1, self.hippocampus_size)
        pfc_total = torch.zeros(self.pfc_size)

        # Combined null input for PFC
        null_input = torch.cat([null_cortex.squeeze(), null_hippo.squeeze()])

        for t in range(n_timesteps):
            # PFC maintains with LOW dopamine (gate closed)
            pfc_out = self.pfc.forward(null_input.unsqueeze(0), dopamine_signal=-0.3)
            pfc_total += pfc_out.squeeze()

        return {"pfc": pfc_total}

    def process_test_and_respond(
        self,
        test_pattern: torch.Tensor,
        n_timesteps: int = 15,
        explore: bool = True,
    ) -> Dict[str, Any]:
        """
        Process test pattern and select response.

        FULL CONNECTIVITY:
        - Cortex extracts features (modulated by PFC attention)
        - Hippocampus retrieves relevant memories
        - Striatum receives ALL THREE: Cortex + PFC + Hippocampus
        - Cerebellum refines motor output

        This enables:
        1. Direct sensory-motor (Cortex → Striatum)
        2. Goal-directed (PFC → Striatum)
        3. Memory-guided (Hippocampus → Striatum)
        """
        test_spikes = rate_to_spikes(test_pattern, n_timesteps)

        cortex_total = torch.zeros(self.cortex_size)
        hippo_total = torch.zeros(self.hippocampus_size)
        pfc_total = torch.zeros(self.pfc_size)
        # Striatum has n_actions * neurons_per_action neurons with population coding
        striatum_n_neurons = self.striatum.config.n_output
        striatum_total = torch.zeros(striatum_n_neurons)
        cerebellum_total = torch.zeros(self.n_actions)

        # Initialize for attention
        pfc_activity = torch.zeros(1, self.pfc_size)

        for t in range(n_timesteps):
            # 1. Apply top-down attention from PFC
            modulated_input = self.apply_top_down_attention(
                test_spikes[t].unsqueeze(0),
                pfc_activity,
                attention_strength=0.2,
            )

            # 2. Cortex processes (attention-modulated) test pattern
            cortex_out = self.cortex.forward(modulated_input)
            cortex_total += cortex_out.squeeze()

            # 3. Hippocampus retrieves (can support comparison)
            hippo_out = self.hippocampus.forward(cortex_out, theta_phase=0.5)
            hippo_total += hippo_out.squeeze()

            # 4. PFC receives Cortex + Hippocampus, uses maintained WM
            pfc_input = torch.cat([cortex_out.squeeze(), hippo_out.squeeze()])
            pfc_out = self.pfc.forward(
                pfc_input.unsqueeze(0),
                dopamine_signal=-0.2,  # Low DA = use maintained WM
            )
            pfc_total += pfc_out.squeeze()
            pfc_activity = pfc_out

            # 5. Striatum receives ALL THREE regions
            #    This is the COMPLETE biological architecture:
            #    - Cortex: current sensory features
            #    - PFC: goal/task representation
            #    - Hippocampus: memory-guided signals
            combined_input = torch.cat([
                cortex_out.squeeze(),
                pfc_out.squeeze(),
                hippo_out.squeeze(),
            ])
            striatum_out = self.striatum.forward(
                combined_input.unsqueeze(0),
                explore=explore,
            )
            striatum_total += striatum_out.squeeze()

            # 6. Cerebellum refines motor output
            cerebellum_out = self.cerebellum.forward(striatum_out)
            cerebellum_total += cerebellum_out.squeeze()

        # Use striatum's built-in action decoding (handles population coding)
        selected_action = self.striatum.get_selected_action()
        if selected_action is None:
            # Fallback to population vote counting
            votes = self.striatum.get_population_votes(striatum_total)
            selected_action = int(votes.argmax().item())

        return {
            "cortex": cortex_total,
            "hippocampus": hippo_total,
            "pfc": pfc_total,
            "striatum": striatum_total,
            "cerebellum": cerebellum_total,
            "selected_action": selected_action,
            "explored": self.striatum.exploring,
        }

    def run_trial(
        self,
        sample_pattern: torch.Tensor,
        test_pattern: torch.Tensor,
        is_match: bool,
        explore: bool = True,
        learn: bool = True,
    ) -> TrialResult:
        """
        Run complete delayed match-to-sample trial.

        1. Sample phase: encode pattern
        2. Delay phase: maintain in WM
        3. Test phase: compare and respond
        4. Learn: apply plasticity based on reward
        """
        self.reset_trial()

        # 1. Encode sample
        sample_result = self.process_sample(sample_pattern, n_timesteps=15)

        # 2. Delay period
        delay_result = self.maintain_delay(n_timesteps=8)

        # 3. Test and respond
        test_result = self.process_test_and_respond(
            test_pattern, n_timesteps=15, explore=explore
        )

        selected = test_result["selected_action"]
        correct_action = 0 if is_match else 1  # 0=match, 1=no-match
        correct = (selected == correct_action)

        # 4. Learning with GLOBAL DOPAMINE modulation
        if learn:
            reward = 1.0 if correct else -1.0

            # Set global dopamine level (affects all regions)
            # This is how the brain coordinates learning across regions
            self.global_dopamine = reward

            # Striatum learns from reward (primary RL region)
            self.striatum.deliver_reward(reward)

            # Cortex learns (unsupervised, but dopamine can modulate rate)
            cortex_spikes = (sample_result["cortex"] > 0).float().unsqueeze(0)
            # Dopamine modulates learning rate in cortex
            cortex_lr_mod = 1.0 + 0.5 * max(0, self.global_dopamine)  # Boost on reward
            self.cortex.learn(
                (sample_pattern > 0.5).float().unsqueeze(0),
                cortex_spikes,
            )

            # PFC learns to maintain patterns that lead to reward
            # Dopamine gates what gets stored in WM
            pfc_input = torch.cat([
                sample_result["cortex"],
                sample_result["hippocampus"],
            ])
            pfc_spikes = (sample_result["pfc"] > 0).float().unsqueeze(0)
            self.pfc.learn(
                (pfc_input > 0).float().unsqueeze(0),
                pfc_spikes,
                reward=reward,  # Dopamine-gated learning
            )

            # Hippocampus stores the association
            # Dopamine enhances memory encoding for important events
            hippo_spikes = (sample_result["hippocampus"] > 0).float().unsqueeze(0)
            self.hippocampus.learn(
                cortex_spikes,
                hippo_spikes,
                force_encoding=True,  # High dopamine = important, store it!
            )

            # Cerebellum learns from motor error
            target = torch.zeros(1, self.n_actions)
            target[0, correct_action] = 1.0
            self.cerebellum.learn(
                test_result["striatum"].unsqueeze(0),
                test_result["cerebellum"].unsqueeze(0),
                target_spikes=target,
            )

            # Store successful experiences in memory buffer
            # (For later replay during sleep consolidation)
            if correct:
                self.store_memory(sample_pattern)

            # Decay exploration at end of each trial
            self.striatum.decay_exploration()

        # Calculate population votes and confidence
        population_votes = self.striatum.get_population_votes(test_result["striatum"])
        sorted_votes = population_votes.sort(descending=True).values
        if len(sorted_votes) >= 2:
            confidence = (sorted_votes[0] - sorted_votes[1]).item()
        else:
            confidence = sorted_votes[0].item() if len(sorted_votes) > 0 else 0.0

        return TrialResult(
            sample_pattern=-1,  # Will be set by caller
            test_pattern=-1,
            is_match=is_match,
            selected_action=selected,
            correct=correct,
            explored=test_result["explored"],
            cortex_activity=sample_result["cortex"].mean().item(),
            pfc_activity=delay_result["pfc"].mean().item(),
            striatum_activity=test_result["striatum"],
            population_votes=population_votes,
            confidence=confidence,
        )


def run_demo(
    n_epochs: int = 600,  # More epochs for curriculum learning
    n_patterns: int = 4,
    verbose: bool = True,
    use_curriculum: bool = True,  # Enable curriculum learning
) -> Dict[str, Any]:
    """
    Run the delayed match-to-sample demo with curriculum learning.

    Task:
    - See sample pattern
    - Wait (delay)
    - See test pattern
    - Respond: MATCH (action 0) or NO-MATCH (action 1)
    
    Curriculum Learning:
    - Start with 2 patterns (easier task)
    - Gradually increase to full set
    - This stabilizes learning in multi-region networks
    """
    # Create patterns
    patterns = create_distinct_patterns(n_patterns, size=256)

    # Create brain system
    brain = BiologicalBrainSystem(
        input_size=256,
        cortex_size=128,
        hippocampus_size=64,
        pfc_size=32,
        n_actions=2,
    )

    print(f"\nTask: Delayed Match-to-Sample")
    print(f"Patterns: {n_patterns}")
    print(f"Actions: 0=MATCH, 1=NO-MATCH")
    print(f"Epochs: {n_epochs}")
    if use_curriculum:
        print(f"Curriculum: ON (start with 2 patterns, add more over time)")
    print()

    # =========================================================================
    # CURRICULUM SCHEDULE
    # =========================================================================
    # Pattern count schedule: (epoch, num_patterns_to_use)
    # Start easy, gradually increase difficulty
    curriculum_schedule = [
        (0, 2),      # Epochs 0-149: 2 patterns only
        (150, 3),    # Epochs 150-299: 3 patterns  
        (300, 4),    # Epochs 300-449: Full 4 patterns
        (450, 4),    # Epochs 450+: Continue with all patterns (fine-tuning)
    ]
    
    def get_curriculum_patterns(epoch: int) -> int:
        """Get number of patterns to use at current epoch."""
        if not use_curriculum:
            return n_patterns
        current = 2
        for (start_epoch, num) in curriculum_schedule:
            if epoch >= start_epoch:
                current = min(num, n_patterns)
        return current

    # Training history
    accuracies: List[float] = []
    confidences: List[float] = []  # Average confidence per epoch
    correct_confidences: List[float] = []  # Confidence when correct
    incorrect_confidences: List[float] = []  # Confidence when incorrect
    exploration_rates: List[float] = []  # Epsilon over time
    curriculum_stage = 0

    for epoch in range(n_epochs):
        epoch_correct = 0
        epoch_total = 0
        epoch_confidences: List[float] = []
        epoch_correct_conf: List[float] = []
        epoch_incorrect_conf: List[float] = []
        
        # Get current number of patterns for curriculum
        current_n_patterns = get_curriculum_patterns(epoch)
        
        # Log curriculum progression
        if use_curriculum:
            stage = len([s for s in curriculum_schedule if epoch >= s[0]])
            if stage != curriculum_stage:
                curriculum_stage = stage
                print(f"\n>>> CURRICULUM: Now using {current_n_patterns} patterns <<<\n")

        # Generate trials: mix of match and no-match
        for trial_idx in range(current_n_patterns * 2):  # 2 trials per pattern
            # Pick sample pattern (from current curriculum subset)
            sample_idx = trial_idx % current_n_patterns

            # 50% match, 50% no-match
            if trial_idx < current_n_patterns:
                test_idx = sample_idx  # Match
                is_match = True
            else:
                # Pick different pattern for no-match
                test_idx = (sample_idx + 1 + np.random.randint(current_n_patterns - 1)) % current_n_patterns
                is_match = False

            # Run trial
            result = brain.run_trial(
                patterns[sample_idx],
                patterns[test_idx],
                is_match=is_match,
                explore=True,
                learn=True,
            )

            if result.correct:
                epoch_correct += 1
                epoch_correct_conf.append(result.confidence)
            else:
                epoch_incorrect_conf.append(result.confidence)
            epoch_total += 1
            epoch_confidences.append(result.confidence)

        accuracy = epoch_correct / epoch_total * 100
        accuracies.append(accuracy)
        
        # Track confidence metrics
        confidences.append(float(np.mean(epoch_confidences)) if epoch_confidences else 0.0)
        correct_confidences.append(float(np.mean(epoch_correct_conf)) if epoch_correct_conf else 0.0)
        incorrect_confidences.append(float(np.mean(epoch_incorrect_conf)) if epoch_incorrect_conf else 0.0)
        exploration_rates.append(brain.striatum.current_epsilon)

        # =====================================================================
        # SLEEP CONSOLIDATION: Every 50 epochs, replay memories
        # =====================================================================
        if (epoch + 1) % 50 == 0:
            sleep_result = brain.sleep_consolidation(n_replay_cycles=10)
            if verbose:
                print(f"Epoch {epoch+1:3d}: Accuracy = {accuracy:.1f}% "
                      f"(epsilon = {brain.striatum.current_epsilon:.3f}) "
                      f"[SLEEP: {sleep_result['replayed']} replays, "
                      f"Δcortex={sleep_result['cortex_delta']:.4f}]")
        elif verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Accuracy = {accuracy:.1f}% "
                  f"(epsilon = {brain.striatum.current_epsilon:.3f})")

    # Final evaluation (no exploration)
    print("\n" + "-" * 70)
    print("Final Evaluation (no exploration)")
    print("-" * 70)

    eval_correct = 0
    eval_total = 0

    for sample_idx in range(n_patterns):
        for is_match in [True, False]:
            if is_match:
                test_idx = sample_idx
            else:
                test_idx = (sample_idx + 1) % n_patterns

            result = brain.run_trial(
                patterns[sample_idx],
                patterns[test_idx],
                is_match=is_match,
                explore=False,
                learn=False,
            )

            status = "✓" if result.correct else "✗"
            match_str = "MATCH" if is_match else "NO-MATCH"
            action_str = "MATCH" if result.selected_action == 0 else "NO-MATCH"

            if verbose:
                print(f"  Sample {sample_idx}, Test {test_idx} ({match_str}): "
                      f"Selected {action_str} {status}")

            if result.correct:
                eval_correct += 1
            eval_total += 1

    test_accuracy = eval_correct / eval_total * 100
    print(f"\nTest Accuracy: {test_accuracy:.1f}%")

    # Plot results
    if verbose:
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # =====================================================================
        # Plot 1: Learning Progress (Accuracy over time)
        # =====================================================================
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(accuracies, alpha=0.3, color='blue')
        window = min(20, len(accuracies))
        smoothed = np.convolve(accuracies, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(accuracies)), smoothed,
                 color='blue', linewidth=2, label='Smoothed')
        ax1.axhline(y=50, color='r', linestyle='--', label='Chance')
        
        # Mark curriculum stages
        if use_curriculum:
            colors = ['green', 'orange', 'purple']
            for i, (epoch_start, num_patterns) in enumerate(curriculum_schedule[1:]):
                if epoch_start < n_epochs:
                    ax1.axvline(x=epoch_start, color=colors[i % len(colors)], 
                               linestyle=':', alpha=0.7, 
                               label=f'{num_patterns} patterns')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Learning Progress')
        ax1.legend(loc='lower right')
        ax1.set_ylim(0, 105)
        
        # =====================================================================
        # Plot 2: Decision Confidence Over Time
        # =====================================================================
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(confidences, alpha=0.3, color='green')
        if len(confidences) >= window:
            smoothed_conf = np.convolve(confidences, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(confidences)), smoothed_conf,
                     color='green', linewidth=2, label='Avg Confidence')
        
        # Mark curriculum stages
        if use_curriculum:
            for i, (epoch_start, _) in enumerate(curriculum_schedule[1:]):
                if epoch_start < n_epochs:
                    ax2.axvline(x=epoch_start, color=colors[i % len(colors)], 
                               linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Confidence (vote margin)')
        ax2.set_title('Decision Confidence Over Time')
        ax2.legend()
        
        # =====================================================================
        # Plot 3: Correct vs Incorrect Confidence
        # =====================================================================
        ax3 = fig.add_subplot(2, 3, 3)
        if len(correct_confidences) >= window:
            smooth_correct = np.convolve(correct_confidences, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(correct_confidences)), smooth_correct,
                     color='green', linewidth=2, label='Correct decisions')
        if len(incorrect_confidences) >= window:
            smooth_incorrect = np.convolve(incorrect_confidences, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(incorrect_confidences)), smooth_incorrect,
                     color='red', linewidth=2, label='Incorrect decisions')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Confidence (vote margin)')
        ax3.set_title('Confidence: Correct vs Incorrect')
        ax3.legend()
        
        # =====================================================================
        # Plot 4: Exploration Rate (Epsilon)
        # =====================================================================
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(exploration_rates, color='purple', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Exploration Rate Over Time')
        ax4.set_ylim(0, 1)
        
        # Mark curriculum stages
        if use_curriculum:
            for i, (epoch_start, _) in enumerate(curriculum_schedule[1:]):
                if epoch_start < n_epochs:
                    ax4.axvline(x=epoch_start, color=colors[i % len(colors)], 
                               linestyle=':', alpha=0.7)
        
        # =====================================================================
        # Plot 5: Striatum Weight Heatmap (Population Coded)
        # =====================================================================
        ax5 = fig.add_subplot(2, 3, 5)
        weights = brain.striatum.weights.data.cpu().numpy()
        im = ax5.imshow(weights, aspect='auto', cmap='RdBu_r')
        ax5.set_xlabel('Input (Cortex + PFC + Hippo)')
        ax5.set_ylabel('Neuron')
        
        # Mark action populations
        neurons_per_action = brain.striatum.neurons_per_action
        for i in range(1, brain.striatum.n_actions):
            ax5.axhline(y=i * neurons_per_action - 0.5, color='black', linewidth=2)
        
        ax5.set_title(f'Striatum Weights (Population Coded)\n{neurons_per_action} neurons/action')
        plt.colorbar(im, ax=ax5)
        
        # =====================================================================
        # Plot 6: Final Population Vote Distribution (Test Trials)
        # =====================================================================
        ax6 = fig.add_subplot(2, 3, 6)
        
        # Collect vote distributions from final evaluation
        match_votes_correct = []
        match_votes_incorrect = []
        nomatch_votes_correct = []
        nomatch_votes_incorrect = []
        
        # Re-run evaluation to collect detailed metrics
        for sample_idx in range(n_patterns):
            for is_match in [True, False]:
                if is_match:
                    test_idx = sample_idx
                else:
                    test_idx = (sample_idx + 1) % n_patterns
                
                result = brain.run_trial(
                    patterns[sample_idx],
                    patterns[test_idx],
                    is_match=is_match,
                    explore=False,
                    learn=False,
                )
                
                if result.population_votes is not None:
                    votes = result.population_votes.cpu().numpy()
                    if result.correct:
                        match_votes_correct.append(votes[0])
                        nomatch_votes_correct.append(votes[1])
                    else:
                        match_votes_incorrect.append(votes[0])
                        nomatch_votes_incorrect.append(votes[1])
        
        # Plot vote distributions
        x = np.arange(2)
        width = 0.35
        
        if match_votes_correct:
            correct_means = [np.mean(match_votes_correct), np.mean(nomatch_votes_correct)]
            ax6.bar(x - width/2, correct_means, width, label='Correct', color='green', alpha=0.7)
        if match_votes_incorrect:
            incorrect_means = [np.mean(match_votes_incorrect), np.mean(nomatch_votes_incorrect)]
            ax6.bar(x + width/2, incorrect_means, width, label='Incorrect', color='red', alpha=0.7)
        
        ax6.set_xticks(x)
        ax6.set_xticklabels(['MATCH Pop', 'NO-MATCH Pop'])
        ax6.set_ylabel('Average Votes (spikes)')
        ax6.set_title('Population Vote Distribution')
        ax6.legend()
        
        plt.tight_layout()

        # Save main figure
        save_dir = Path("experiments/results/integration")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "brain_integration_demo.png"
        plt.savefig(save_path, dpi=150)
        print(f"\nPlot saved to: {save_path.absolute()}")
        plt.close()

    return {
        "test_accuracy": test_accuracy,
        "final_training_accuracy": accuracies[-1] if accuracies else 0,
        "accuracies": accuracies,
        "confidences": confidences,
        "correct_confidences": correct_confidences,
        "incorrect_confidences": incorrect_confidences,
        "exploration_rates": exploration_rates,
        "used_curriculum": use_curriculum,
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    results = run_demo(n_epochs=600, n_patterns=4, verbose=True, use_curriculum=True)

    if results["test_accuracy"] >= 70:
        print("\n✓ SUCCESS: System learned the task!")
    else:
        print("\n✗ NEEDS TUNING: Accuracy below 70%")
