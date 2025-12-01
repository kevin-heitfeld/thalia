"""
Diagnostic data collectors for SNN experiments.

These classes collect, aggregate, and report diagnostic data during training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
import torch

from .config import DiagnosticConfig, DiagnosticLevel


@dataclass
class SpikeTimingStats:
    """Statistics about spike timing relative to expected phases."""
    # Per-neuron stats
    mean_spike_phase: np.ndarray  # Average phase when each neuron fires
    expected_phase: np.ndarray     # Expected phase for each neuron
    phase_error: np.ndarray        # Difference between actual and expected
    
    # Aggregate stats
    mean_absolute_error: float
    correctly_timed_fraction: float  # Fraction firing in expected phase


class SpikeTimingAnalyzer:
    """
    Tracks when neurons fire relative to their expected phases.
    
    Helps diagnose issues like:
    - Neurons firing in wrong phases (recency bias)
    - Neurons firing too early/late within their phase
    - Phase selectivity development over training
    """
    
    def __init__(self, n_neurons: int, n_phases: int, device: torch.device):
        self.n_neurons = n_neurons
        self.n_phases = n_phases
        self.device = device
        
        # Accumulate spike counts per phase per neuron: [n_phases, n_neurons]
        self.phase_spike_counts = torch.zeros(n_phases, n_neurons, device=device)
        
        # Track first spike timing within each phase (for latency analysis)
        # Accumulate sum of first-spike times and count
        self.first_spike_time_sum = torch.zeros(n_phases, n_neurons, device=device)
        self.first_spike_count = torch.zeros(n_phases, n_neurons, device=device)
        
        # Per-cycle tracking for consistency analysis
        self.cycle_phase_winners: List[np.ndarray] = []  # Which neuron won each phase
        
    def record_spike(self, phase: int, neuron_idx: int, time_in_phase: float = 0.0):
        """Record a spike from a neuron in a specific phase."""
        self.phase_spike_counts[phase, neuron_idx] += 1
        
        # Track first spike timing (only if this is first spike in this phase/neuron)
        if self.first_spike_count[phase, neuron_idx] == 0 or time_in_phase < self.first_spike_time_sum[phase, neuron_idx] / max(1, self.first_spike_count[phase, neuron_idx]):
            self.first_spike_time_sum[phase, neuron_idx] += time_in_phase
            self.first_spike_count[phase, neuron_idx] += 1
    
    def record_phase_spikes(self, phase: int, spikes: torch.Tensor, time_in_phase: float = 0.0):
        """Record all spikes in a phase (spikes is [n_neurons] binary tensor)."""
        self.phase_spike_counts[phase] += spikes.float()
        
        # Track timing for first spikes
        firing_neurons = spikes.nonzero(as_tuple=True)[0]
        for neuron_idx in firing_neurons:
            self.first_spike_time_sum[phase, neuron_idx] += time_in_phase
            self.first_spike_count[phase, neuron_idx] += 1
    
    def record_cycle_winners(self, winners: np.ndarray):
        """Record which neuron won each phase in this cycle."""
        self.cycle_phase_winners.append(winners.copy())
    
    def reset_cycle(self):
        """Call at start of each cycle to reset per-cycle tracking."""
        pass  # Currently we accumulate across cycles
    
    def get_stats(self, expected_neuron_per_phase: Optional[np.ndarray] = None) -> SpikeTimingStats:
        """
        Compute timing statistics.
        
        Args:
            expected_neuron_per_phase: Which neuron should win each phase [n_phases]
        """
        counts = self.phase_spike_counts.cpu().numpy()
        
        # Mean phase for each neuron (weighted by spike counts)
        phases = np.arange(self.n_phases)
        total_spikes = counts.sum(axis=0)  # [n_neurons]
        
        # Avoid division by zero
        total_spikes_safe = np.maximum(total_spikes, 1)
        
        # Weighted average phase for each neuron
        mean_spike_phase = (counts.T @ phases) / total_spikes_safe  # [n_neurons]
        
        # Expected phase for each neuron (based on input mapping)
        if expected_neuron_per_phase is not None:
            # Invert: for each neuron, which phases should it respond to?
            expected_phase = np.zeros(self.n_neurons)
            for phase, neuron in enumerate(expected_neuron_per_phase):
                if neuron < self.n_neurons:
                    expected_phase[neuron] = phase  # Last expected phase
        else:
            # Default: assume linear mapping (neuron i responds to phases i*ratio to (i+1)*ratio-1)
            ratio = self.n_phases // self.n_neurons
            expected_phase = np.array([(i * ratio + (i + 1) * ratio - 1) / 2 for i in range(self.n_neurons)])
        
        phase_error = mean_spike_phase - expected_phase
        
        # Correctly timed: neuron's peak phase matches expected
        peak_phases = counts.argmax(axis=0)  # [n_neurons]
        correctly_timed = 0
        for neuron in range(self.n_neurons):
            ratio = self.n_phases // self.n_neurons
            expected_start = neuron * ratio
            expected_end = expected_start + ratio
            if expected_start <= peak_phases[neuron] < expected_end:
                correctly_timed += 1
        
        return SpikeTimingStats(
            mean_spike_phase=mean_spike_phase,
            expected_phase=expected_phase,
            phase_error=phase_error,
            mean_absolute_error=np.abs(phase_error).mean(),
            correctly_timed_fraction=correctly_timed / self.n_neurons,
        )
    
    def get_phase_selectivity(self) -> np.ndarray:
        """
        Compute phase selectivity for each neuron.
        
        Returns [n_neurons] array where higher values = more phase-selective.
        Uses entropy-based measure: low entropy = high selectivity.
        """
        counts = self.phase_spike_counts.cpu().numpy()
        total = counts.sum(axis=0, keepdims=True)
        total = np.maximum(total, 1)  # Avoid division by zero
        
        probs = counts / total  # [n_phases, n_neurons]
        
        # Entropy (lower = more selective)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=0)
        
        # Convert to selectivity (higher = more selective)
        max_entropy = np.log(self.n_phases)
        selectivity = 1 - entropy / max_entropy
        
        return selectivity
    
    def print_summary(self, expected_neuron_per_phase: Optional[np.ndarray] = None):
        """Print human-readable summary."""
        stats = self.get_stats(expected_neuron_per_phase)
        selectivity = self.get_phase_selectivity()
        
        print("\n  === Spike Timing Analysis ===")
        print(f"  Correctly timed neurons: {stats.correctly_timed_fraction:.1%}")
        print(f"  Mean phase error: {stats.mean_absolute_error:.2f} phases")
        print(f"  Phase selectivity: mean={selectivity.mean():.3f}, min={selectivity.min():.3f}, max={selectivity.max():.3f}")
        
        # Show per-neuron details
        counts = self.phase_spike_counts.cpu().numpy()
        peak_phases = counts.argmax(axis=0)
        for neuron in range(self.n_neurons):
            ratio = self.n_phases // self.n_neurons
            expected_start = neuron * ratio
            expected_end = expected_start + ratio - 1
            is_correct = expected_start <= peak_phases[neuron] <= expected_end
            status = "✓" if is_correct else "✗"
            print(f"    Neuron {neuron}: peak_phase={peak_phases[neuron]}, expected={expected_start}-{expected_end}, selectivity={selectivity[neuron]:.3f} {status}")


@dataclass
class WeightChangeRecord:
    """Record of weight changes from a single update."""
    cycle: int
    phase: int
    mechanism: str  # "hebbian_ltp", "hebbian_ltd", "heterosynaptic", "scaling", etc.
    delta_weights: float  # Mean absolute change
    ltp_fraction: float   # Fraction of changes that were positive
    affected_synapses: int  # Number of synapses changed


class WeightChangeTracker:
    """
    Tracks weight changes attributed to different mechanisms.
    
    Helps diagnose issues like:
    - LTP/LTD imbalance causing saturation
    - Heterosynaptic LTD not balancing Hebbian LTP
    - Which mechanism is dominating learning
    """
    
    def __init__(self, n_pre: int, n_post: int, device: torch.device):
        self.n_pre = n_pre
        self.n_post = n_post
        self.device = device
        
        # Accumulate changes per mechanism
        self.mechanism_totals: Dict[str, float] = defaultdict(float)
        self.mechanism_counts: Dict[str, int] = defaultdict(int)
        self.mechanism_ltp_total: Dict[str, float] = defaultdict(float)
        
        # Detailed records (if needed for analysis)
        self.records: List[WeightChangeRecord] = []
        self.store_details = False  # Set True for detailed analysis
        
        # Track weight distribution over time
        self.weight_snapshots: List[Tuple[int, np.ndarray]] = []
        
    def record_change(
        self, 
        cycle: int, 
        phase: int, 
        mechanism: str, 
        old_weights: torch.Tensor, 
        new_weights: torch.Tensor
    ):
        """Record weight change from a mechanism."""
        delta = new_weights - old_weights
        delta_np = delta.cpu().numpy()
        
        abs_change = np.abs(delta_np).sum()
        ltp_change = np.maximum(delta_np, 0).sum()
        ltd_change = np.abs(np.minimum(delta_np, 0)).sum()
        affected = (delta_np != 0).sum()
        
        self.mechanism_totals[mechanism] += abs_change
        self.mechanism_counts[mechanism] += 1
        self.mechanism_ltp_total[mechanism] += ltp_change
        
        if self.store_details:
            self.records.append(WeightChangeRecord(
                cycle=cycle,
                phase=phase,
                mechanism=mechanism,
                delta_weights=abs_change / max(affected, 1),
                ltp_fraction=ltp_change / max(ltp_change + ltd_change, 1e-10),
                affected_synapses=affected,
            ))
    
    def record_snapshot(self, cycle: int, weights: torch.Tensor):
        """Store weight snapshot for evolution analysis."""
        self.weight_snapshots.append((cycle, weights.cpu().numpy().copy()))
    
    def get_mechanism_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics per mechanism."""
        summary = {}
        for mechanism in self.mechanism_totals:
            total = self.mechanism_totals[mechanism]
            count = self.mechanism_counts[mechanism]
            ltp = self.mechanism_ltp_total[mechanism]
            
            summary[mechanism] = {
                "total_change": total,
                "mean_change": total / max(count, 1),
                "update_count": count,
                "ltp_fraction": ltp / max(total, 1e-10),
            }
        return summary
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n  === Weight Change Attribution ===")
        summary = self.get_mechanism_summary()
        
        if not summary:
            print("  No weight changes recorded")
            return
        
        # Sort by total change
        sorted_mechanisms = sorted(summary.items(), key=lambda x: -x[1]["total_change"])
        
        total_all = sum(s["total_change"] for s in summary.values())
        
        for mechanism, stats in sorted_mechanisms:
            pct = 100 * stats["total_change"] / max(total_all, 1e-10)
            ltp_pct = 100 * stats["ltp_fraction"]
            print(f"  {mechanism:25s}: {pct:5.1f}% of total, {ltp_pct:5.1f}% LTP, {stats['update_count']:5d} updates")


@dataclass  
class MechanismState:
    """Snapshot of a homeostatic mechanism's state."""
    name: str
    values: np.ndarray
    bounds: Optional[Tuple[float, float]]
    at_lower_bound: int  # Count of neurons at lower bound
    at_upper_bound: int  # Count of neurons at upper bound
    mean: float
    std: float


class MechanismStateTracker:
    """
    Tracks state of homeostatic mechanisms over time.
    
    Helps diagnose issues like:
    - Mechanisms saturating at bounds (can't adapt further)
    - Wide variance indicating unstable dynamics
    - Mechanisms not being used (stuck at initial values)
    """
    
    def __init__(self, n_neurons: int, device: torch.device):
        self.n_neurons = n_neurons
        self.device = device
        
        # Define tracked mechanisms and their bounds
        self.mechanism_bounds: Dict[str, Tuple[float, float]] = {
            "g_tonic": (0.0, 0.5),
            "excitability": (-10.0, 10.0),
            "bcm_threshold": (0.0, 100.0),
            "sfa_current": (0.0, float("inf")),
            "som_current": (0.0, float("inf")),
        }
        
        # History of states per mechanism
        self.history: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        
        # Latest values
        self.latest: Dict[str, np.ndarray] = {}
        
    def record(self, cycle: int, mechanism: str, values: torch.Tensor, 
               bounds: Optional[Tuple[float, float]] = None):
        """Record mechanism state."""
        values_np = values.cpu().numpy().copy()
        self.latest[mechanism] = values_np
        self.history[mechanism].append((cycle, values_np))
        
        if bounds is not None:
            self.mechanism_bounds[mechanism] = bounds
    
    def get_state(self, mechanism: str) -> Optional[MechanismState]:
        """Get current state of a mechanism."""
        if mechanism not in self.latest:
            return None
        
        values = self.latest[mechanism]
        bounds = self.mechanism_bounds.get(mechanism)
        
        at_lower = 0
        at_upper = 0
        if bounds:
            at_lower = (values <= bounds[0] + 1e-6).sum()
            at_upper = (values >= bounds[1] - 1e-6).sum()
        
        return MechanismState(
            name=mechanism,
            values=values,
            bounds=bounds,
            at_lower_bound=at_lower,
            at_upper_bound=at_upper,
            mean=values.mean(),
            std=values.std(),
        )
    
    def get_all_states(self) -> Dict[str, MechanismState]:
        """Get current state of all tracked mechanisms."""
        return {name: self.get_state(name) for name in self.latest if self.get_state(name)}
    
    def check_saturation(self) -> List[str]:
        """Return list of mechanisms that are saturated (>50% at bounds)."""
        saturated = []
        for name in self.latest:
            state = self.get_state(name)
            if state and state.bounds:
                pct_at_bounds = (state.at_lower_bound + state.at_upper_bound) / self.n_neurons
                if pct_at_bounds > 0.5:
                    saturated.append(name)
        return saturated
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n  === Mechanism States ===")
        
        for name in sorted(self.latest.keys()):
            state = self.get_state(name)
            if state is None:
                continue
            
            bounds_str = f"[{state.bounds[0]:.3f}, {state.bounds[1]:.3f}]" if state.bounds else "unbounded"
            saturation = ""
            if state.bounds:
                pct_lower = 100 * state.at_lower_bound / self.n_neurons
                pct_upper = 100 * state.at_upper_bound / self.n_neurons
                if pct_lower > 10 or pct_upper > 10:
                    saturation = f" ⚠️ {pct_lower:.0f}% at min, {pct_upper:.0f}% at max"
            
            print(f"  {name:20s}: mean={state.mean:.4f}, std={state.std:.4f}, bounds={bounds_str}{saturation}")


class WinnerConsistencyTracker:
    """
    Tracks consistency of which neurons win which phases over training.
    
    Helps diagnose issues like:
    - Neurons not stabilizing to specific phases
    - Multiple neurons competing for same phase
    - Winner changes disrupting learning
    """
    
    def __init__(self, n_neurons: int, n_phases: int, device: torch.device):
        self.n_neurons = n_neurons
        self.n_phases = n_phases
        self.device = device
        
        # Count how many times each neuron won each phase: [n_phases, n_neurons]
        self.win_counts = torch.zeros(n_phases, n_neurons, device=device)
        
        # Track winner history per cycle
        self.winner_history: List[np.ndarray] = []  # List of [n_phases] arrays
        
        # Track when winners change
        self.winner_changes: List[Tuple[int, int, int, int]] = []  # (cycle, phase, old_winner, new_winner)
        self.last_winners: Optional[np.ndarray] = None
        
    def record_winner(self, phase: int, winner: int):
        """Record a phase winner."""
        if winner >= 0:  # -1 means no winner
            self.win_counts[phase, winner] += 1
    
    def record_cycle(self, cycle: int, winners: np.ndarray):
        """Record all phase winners for a cycle."""
        self.winner_history.append(winners.copy())
        
        # Track changes from last cycle
        if self.last_winners is not None:
            for phase in range(len(winners)):
                if winners[phase] != self.last_winners[phase]:
                    self.winner_changes.append((
                        cycle, phase, 
                        int(self.last_winners[phase]), 
                        int(winners[phase])
                    ))
        self.last_winners = winners.copy()
        
        # Update counts
        for phase, winner in enumerate(winners):
            self.record_winner(phase, int(winner))
    
    def get_consistency(self) -> np.ndarray:
        """
        Compute consistency score for each phase.
        
        Returns [n_phases] array where higher = more consistent winner.
        """
        counts = self.win_counts.cpu().numpy()
        total = counts.sum(axis=1, keepdims=True)
        total = np.maximum(total, 1)
        
        # Consistency = fraction of wins by the most common winner
        consistency = counts.max(axis=1) / total.squeeze()
        return consistency
    
    def get_dominant_winners(self) -> np.ndarray:
        """Get the most common winner for each phase."""
        counts = self.win_counts.cpu().numpy()
        return counts.argmax(axis=1)
    
    def compute_accuracy(
        self, 
        expected_winners: Optional[np.ndarray] = None,
        ratio: int = 2,
    ) -> Tuple[int, int]:
        """Compute accuracy: how many phases have the expected winner.
        
        Args:
            expected_winners: Array of expected winner for each phase.
                If None, uses diagonal mapping (phase // ratio).
            ratio: Inputs per logical phase (only used if expected_winners is None).
        
        Returns:
            Tuple of (correct_count, total_phases).
        """
        dominant = self.get_dominant_winners()
        
        if expected_winners is None:
            # Default diagonal mapping
            expected_winners = np.array([p // ratio for p in range(self.n_phases)])
        
        correct = np.sum(dominant == expected_winners)
        return int(correct), self.n_phases
    
    def get_neuron_specialization(self) -> np.ndarray:
        """
        Compute how specialized each neuron is.
        
        Returns [n_neurons] array where higher = wins fewer phases more consistently.
        """
        counts = self.win_counts.cpu().numpy().T  # [n_neurons, n_phases]
        total = counts.sum(axis=1, keepdims=True)
        total = np.maximum(total, 1)
        
        probs = counts / total
        
        # Entropy-based: low entropy = high specialization
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        max_entropy = np.log(self.n_phases)
        
        specialization = 1 - entropy / max_entropy
        return specialization
    
    def print_summary(self, ratio: int = 2, expected_winners: Optional[np.ndarray] = None):
        """Print human-readable summary.
        
        Args:
            ratio: Inputs per logical phase (only used if expected_winners is None).
            expected_winners: Array of expected winner for each phase.
                If None, uses diagonal mapping (phase // ratio).
        """
        print("\n  === Winner Consistency Analysis ===")
        
        consistency = self.get_consistency()
        dominant = self.get_dominant_winners()
        specialization = self.get_neuron_specialization()
        
        print(f"  Phase consistency: mean={consistency.mean():.3f}, min={consistency.min():.3f}")
        print(f"  Neuron specialization: mean={specialization.mean():.3f}, min={specialization.min():.3f}")
        print(f"  Winner changes: {len(self.winner_changes)} total")
        
        # Compute expected winners if not provided
        if expected_winners is None:
            expected_winners = np.array([phase // ratio for phase in range(self.n_phases)])
        
        # Check if dominant winners match expected
        correct, total = self.compute_accuracy(expected_winners)
        print(f"  Correct phase->neuron mapping: {correct}/{total} ({100*correct/max(total,1):.0f}%)")
        
        # Show per-phase details (for arbitrary mappings, per-neuron doesn't make sense)
        print("\n  Per-phase winners:")
        counts = self.win_counts.cpu().numpy()
        for phase in range(self.n_phases):
            expected = expected_winners[phase]
            actual = dominant[phase]
            phase_counts = counts[phase, :]
            total_for_phase = phase_counts.sum()
            if total_for_phase > 0:
                pct = 100 * phase_counts[actual] / total_for_phase
                status = "✓" if actual == expected else "✗"
                print(f"    Phase {phase}: expected={expected}, actual={actual} ({pct:.0f}% wins) {status}")
            else:
                print(f"    Phase {phase}: expected={expected}, no spikes")


class EligibilityTracker:
    """
    Tracks eligibility trace dynamics for three-factor learning.
    
    Helps diagnose issues like:
    - Correct neurons not building eligibility (credit assignment failure)
    - Eligibility decay too fast/slow
    - Subthreshold eligibility contribution
    """
    
    def __init__(self, n_pre: int, n_post: int, device: torch.device):
        self.n_pre = n_pre
        self.n_post = n_post
        self.device = device
        
        # Track per-cycle stats
        self.cycle_stats: List[Dict[str, Any]] = []
        
        # Track eligibility per neuron over time
        self.eligibility_per_neuron: List[np.ndarray] = []  # List of [n_post] arrays
        
        # Track which neurons have non-zero eligibility
        self.active_eligibility_counts: List[int] = []
        
    def record_eligibility(
        self, 
        cycle: int, 
        eligibility: torch.Tensor,
        target_neuron: Optional[int] = None,
    ):
        """Record eligibility trace state at end of cycle.
        
        Args:
            cycle: Current cycle number
            eligibility: Eligibility trace tensor, shape (n_post, n_pre)
            target_neuron: Optional - the correct target neuron for this phase
        """
        elig_np = eligibility.detach().cpu().numpy()
        
        # Per-neuron eligibility (sum across inputs)
        elig_per_neuron = elig_np.sum(axis=1)
        self.eligibility_per_neuron.append(elig_per_neuron.copy())
        
        # Count neurons with non-zero eligibility
        active = (elig_per_neuron > 1e-6).sum()
        self.active_eligibility_counts.append(active)
        
        # Record stats
        stats = {
            "cycle": cycle,
            "max_elig": float(elig_np.max()),
            "mean_elig": float(elig_np.mean()),
            "total_elig": float(elig_np.sum()),
            "active_neurons": active,
            "elig_per_neuron": elig_per_neuron,
        }
        
        # If target neuron specified, record its eligibility
        if target_neuron is not None and 0 <= target_neuron < self.n_post:
            stats["target_elig"] = float(elig_per_neuron[target_neuron])
            stats["target_rank"] = int((elig_per_neuron > elig_per_neuron[target_neuron]).sum())
        
        self.cycle_stats.append(stats)
    
    def get_credit_assignment_score(self) -> float:
        """
        Compute how well eligibility is assigned to target neurons.
        
        Returns fraction of cycles where target had highest eligibility.
        """
        correct = 0
        total = 0
        for stats in self.cycle_stats:
            if "target_rank" in stats:
                total += 1
                if stats["target_rank"] == 0:  # Target had highest eligibility
                    correct += 1
        return correct / max(total, 1)
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n  === Eligibility Trace Analysis ===")
        
        if not self.cycle_stats:
            print("  No eligibility data recorded")
            return
        
        # Early vs late comparison
        early = self.cycle_stats[:5]
        late = self.cycle_stats[-5:]
        
        early_max = np.mean([s["max_elig"] for s in early])
        late_max = np.mean([s["max_elig"] for s in late])
        early_active = np.mean([s["active_neurons"] for s in early])
        late_active = np.mean([s["active_neurons"] for s in late])
        
        print(f"  Max eligibility: early={early_max:.4f}, late={late_max:.4f}")
        print(f"  Active neurons: early={early_active:.1f}, late={late_active:.1f}")
        
        # Credit assignment score
        ca_score = self.get_credit_assignment_score()
        print(f"  Credit assignment score: {ca_score:.1%} (target had highest elig)")
        
        # Show per-neuron eligibility for last few cycles
        if len(self.eligibility_per_neuron) >= 1:
            last_elig = self.eligibility_per_neuron[-1]
            print(f"  Final eligibility per neuron: {last_elig.round(3)}")


class DopamineTracker:
    """
    Tracks dopamine signal dynamics for reward-modulated learning.
    
    Helps diagnose issues like:
    - Dopamine always negative (always wrong) or positive (always right)
    - Dopamine not triggering weight updates
    - Reward signal not reflecting learning progress
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Track dopamine events
        self.events: List[Dict[str, Any]] = []
        
        # Aggregate stats per cycle
        self.cycle_stats: List[Dict[str, float]] = []
        
        # Current cycle accumulator
        self.current_cycle_bursts = 0
        self.current_cycle_dips = 0
        self.current_cycle_total = 0.0
        self.current_cycle_count = 0
        
    def record_dopamine(
        self, 
        timestep: int, 
        dopamine_level: float,
        was_correct: Optional[bool] = None,
        winner: Optional[int] = None,
        target: Optional[int] = None,
    ):
        """Record a dopamine signal event."""
        if abs(dopamine_level) < 0.01:
            return  # Skip baseline
        
        self.events.append({
            "t": timestep,
            "dopamine": dopamine_level,
            "correct": was_correct,
            "winner": winner,
            "target": target,
        })
        
        # Accumulate for cycle stats
        if dopamine_level > 0:
            self.current_cycle_bursts += 1
        else:
            self.current_cycle_dips += 1
        self.current_cycle_total += dopamine_level
        self.current_cycle_count += 1
    
    def end_cycle(self, cycle: int):
        """Record end of cycle stats and reset accumulators."""
        if self.current_cycle_count > 0:
            self.cycle_stats.append({
                "cycle": cycle,
                "bursts": self.current_cycle_bursts,
                "dips": self.current_cycle_dips,
                "mean_dopamine": self.current_cycle_total / self.current_cycle_count,
                "event_count": self.current_cycle_count,
            })
        
        # Reset
        self.current_cycle_bursts = 0
        self.current_cycle_dips = 0
        self.current_cycle_total = 0.0
        self.current_cycle_count = 0
    
    def get_burst_ratio(self) -> float:
        """Get fraction of dopamine events that were bursts (positive)."""
        total_bursts = sum(s["bursts"] for s in self.cycle_stats)
        total_events = sum(s["event_count"] for s in self.cycle_stats)
        return total_bursts / max(total_events, 1)
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n  === Dopamine Signal Analysis ===")
        
        if not self.cycle_stats:
            print("  No dopamine events recorded")
            return
        
        total_bursts = sum(s["bursts"] for s in self.cycle_stats)
        total_dips = sum(s["dips"] for s in self.cycle_stats)
        total_events = total_bursts + total_dips
        
        print(f"  Total events: {total_events} ({total_bursts} bursts, {total_dips} dips)")
        print(f"  Burst ratio: {100 * total_bursts / max(total_events, 1):.1f}%")
        
        # Early vs late comparison
        early = self.cycle_stats[:5]
        late = self.cycle_stats[-5:]
        
        early_burst_ratio = sum(s["bursts"] for s in early) / max(sum(s["event_count"] for s in early), 1)
        late_burst_ratio = sum(s["bursts"] for s in late) / max(sum(s["event_count"] for s in late), 1)
        
        print(f"  Burst ratio evolution: early={early_burst_ratio:.1%}, late={late_burst_ratio:.1%}")
        
        if late_burst_ratio > early_burst_ratio:
            print("  ✓ Learning progress: more correct responses over time")
        elif late_burst_ratio < early_burst_ratio:
            print("  ✗ Regression: fewer correct responses over time")
        else:
            print("  ⚠ No change: learning may be stuck")


class ExperimentDiagnostics:
    """
    Central collector for all diagnostic data during an experiment.
    
    Usage:
        diagnostics = ExperimentDiagnostics(config, n_neurons, n_phases, device)
        
        # During training loop:
        diagnostics.start_cycle(cycle_num)
        diagnostics.record_phase_spikes(phase, spikes)
        diagnostics.record_weight_change(cycle, phase, "hebbian", old_w, new_w)
        diagnostics.record_mechanism_state(cycle, "g_tonic", g_tonic, bounds=(0, 0.5))
        diagnostics.end_cycle(cycle_num)
        
        # After training:
        diagnostics.print_final_summary()
    """
    
    def __init__(
        self, 
        config: DiagnosticConfig,
        n_neurons: int,
        n_phases: int,
        n_inputs: int,
        device: torch.device,
    ):
        self.config = config
        self.n_neurons = n_neurons
        self.n_phases = n_phases
        self.n_inputs = n_inputs
        self.device = device
        
        # Initialize collectors based on config
        self.spike_timing: Optional[SpikeTimingAnalyzer] = None
        self.weight_changes: Optional[WeightChangeTracker] = None
        self.mechanism_states: Optional[MechanismStateTracker] = None
        self.winner_consistency: Optional[WinnerConsistencyTracker] = None
        
        if config.collect_spike_timing:
            self.spike_timing = SpikeTimingAnalyzer(n_neurons, n_phases, device)
        
        if config.collect_weight_changes:
            self.weight_changes = WeightChangeTracker(n_inputs, n_neurons, device)
            self.weight_changes.store_details = (config.level == DiagnosticLevel.DEBUG)
        
        if config.collect_mechanism_states:
            self.mechanism_states = MechanismStateTracker(n_neurons, device)
        
        if config.collect_winner_consistency:
            self.winner_consistency = WinnerConsistencyTracker(n_neurons, n_phases, device)
        
        # Supervised learning specific trackers
        self.eligibility_tracker: Optional[EligibilityTracker] = None
        self.dopamine_tracker: Optional[DopamineTracker] = None
        
        if config.collect_eligibility:
            self.eligibility_tracker = EligibilityTracker(n_inputs, n_neurons, device)
        
        if config.collect_dopamine:
            self.dopamine_tracker = DopamineTracker(device)
        
        # Current cycle state
        self.current_cycle = 0
        self.current_phase = 0
        self.cycle_winners: List[int] = []
        
        # Timing stats
        self.timing_stats: Dict[str, float] = defaultdict(float)
        self.timing_counts: Dict[str, int] = defaultdict(int)
        
    def start_cycle(self, cycle: int):
        """Called at start of each training cycle."""
        self.current_cycle = cycle
        self.current_phase = 0
        self.cycle_winners = []
        
        if self.spike_timing:
            self.spike_timing.reset_cycle()
    
    def start_phase(self, phase: int):
        """Called at start of each phase within a cycle."""
        self.current_phase = phase
    
    def record_phase_spikes(self, phase: int, spikes: torch.Tensor, time_in_phase: float = 0.0):
        """Record spikes during a phase."""
        if self.spike_timing:
            self.spike_timing.record_phase_spikes(phase, spikes, time_in_phase)
    
    def record_winner(self, phase: int, winner: int):
        """Record the winner of a phase."""
        # Track for consistency analysis
        while len(self.cycle_winners) <= phase:
            self.cycle_winners.append(-1)
        self.cycle_winners[phase] = winner
    
    def record_weight_change(
        self, 
        mechanism: str, 
        old_weights: torch.Tensor, 
        new_weights: torch.Tensor
    ):
        """Record weight change from a mechanism."""
        if self.weight_changes:
            self.weight_changes.record_change(
                self.current_cycle, self.current_phase, 
                mechanism, old_weights, new_weights
            )
    
    def record_mechanism_state(
        self, 
        mechanism: str, 
        values: torch.Tensor,
        bounds: Optional[Tuple[float, float]] = None
    ):
        """Record current state of a homeostatic mechanism."""
        if self.mechanism_states:
            self.mechanism_states.record(self.current_cycle, mechanism, values, bounds)
    
    def record_weight_snapshot(self, weights: torch.Tensor):
        """Store weight snapshot if configured."""
        if self.weight_changes and self.config.track_weight_snapshots:
            if self.current_cycle % self.config.weight_snapshot_interval == 0:
                self.weight_changes.record_snapshot(self.current_cycle, weights)
    
    def record_timing(self, operation: str, elapsed: float):
        """Record timing for an operation."""
        self.timing_stats[operation] += elapsed
        self.timing_counts[operation] += 1
    
    def record_eligibility(
        self, 
        eligibility: torch.Tensor,
        target_neuron: Optional[int] = None,
    ):
        """Record eligibility trace state (for three-factor learning)."""
        if self.eligibility_tracker:
            self.eligibility_tracker.record_eligibility(
                self.current_cycle, eligibility, target_neuron
            )
    
    def record_dopamine(
        self, 
        timestep: int,
        dopamine_level: float,
        was_correct: Optional[bool] = None,
        winner: Optional[int] = None,
        target: Optional[int] = None,
    ):
        """Record dopamine signal event (for reward-modulated learning)."""
        if self.dopamine_tracker:
            self.dopamine_tracker.record_dopamine(
                timestep, dopamine_level, was_correct, winner, target
            )
    
    def get_dominant_winners(self) -> Optional[np.ndarray]:
        """Get the most common winner for each phase.
        
        Returns:
            Array of shape [n_phases] with the dominant winner for each phase,
            or None if winner consistency tracking is disabled.
        """
        if self.winner_consistency:
            return self.winner_consistency.get_dominant_winners()
        return None
    
    def compute_accuracy(
        self, 
        expected_winners: Optional[np.ndarray] = None,
        ratio: int = 2,
    ) -> Tuple[int, int]:
        """Compute accuracy: how many phases have the expected winner.
        
        Args:
            expected_winners: Array of expected winner for each phase.
                If None, uses diagonal mapping (phase // ratio).
            ratio: Inputs per logical phase (only used if expected_winners is None).
        
        Returns:
            Tuple of (correct_count, total_phases).
            Returns (0, 0) if winner consistency tracking is disabled.
        """
        if self.winner_consistency:
            return self.winner_consistency.compute_accuracy(expected_winners, ratio)
        return 0, 0

    def end_cycle(self, cycle: int):
        """Called at end of each training cycle."""
        # Record winners for consistency tracking
        if self.winner_consistency and self.cycle_winners:
            # Pad to n_phases if needed
            while len(self.cycle_winners) < self.n_phases:
                self.cycle_winners.append(-1)
            self.winner_consistency.record_cycle(cycle, np.array(self.cycle_winners))
        
        # End cycle for dopamine tracker
        if self.dopamine_tracker:
            self.dopamine_tracker.end_cycle(cycle)
        
        # Print cycle summary if configured
        if self.should_print_cycle(cycle):
            self.print_cycle_summary(cycle)
    
    def should_print_cycle(self, cycle: int) -> bool:
        """Check if we should print summary for this cycle."""
        if self.config.level == DiagnosticLevel.NONE:
            return False
        if cycle <= self.config.early_cycle_details:
            return self.config.level.value >= DiagnosticLevel.VERBOSE.value
        return cycle % self.config.report_every_n_cycles == 0
    
    def print_cycle_summary(self, cycle: int):
        """Print summary for current cycle."""
        print(f"\n  --- Cycle {cycle} Diagnostics ---")
        
        # Show saturation warnings
        if self.mechanism_states:
            saturated = self.mechanism_states.check_saturation()
            if saturated:
                print(f"  ⚠️ SATURATED MECHANISMS: {', '.join(saturated)}")
        
        # Brief stats
        if self.winner_consistency and cycle > 1:
            consistency = self.winner_consistency.get_consistency()
            print(f"  Winner consistency: {consistency.mean():.3f}")
        
        # Detailed output for verbose/debug
        if self.config.level.value >= DiagnosticLevel.VERBOSE.value:
            if self.mechanism_states:
                self.mechanism_states.print_summary()
    
    def print_final_summary(self, ratio: int = 2, expected_winners: Optional[np.ndarray] = None):
        """Print comprehensive summary at end of experiment.
        
        Args:
            ratio: Inputs per logical phase (only used if expected_winners is None).
            expected_winners: Array of expected winner for each phase.
                If None, uses diagonal mapping (phase // ratio).
        """
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        if self.spike_timing:
            self.spike_timing.print_summary()
        
        if self.weight_changes:
            self.weight_changes.print_summary()
        
        if self.mechanism_states:
            self.mechanism_states.print_summary()
        
        if self.winner_consistency:
            self.winner_consistency.print_summary(ratio=ratio, expected_winners=expected_winners)
        
        # Supervised learning diagnostics
        if self.eligibility_tracker:
            self.eligibility_tracker.print_summary()
        
        if self.dopamine_tracker:
            self.dopamine_tracker.print_summary()
        
        # Print timing summary
        if self.timing_stats:
            print("\n  === Timing Summary ===")
            total_time = sum(self.timing_stats.values())
            for op, time in sorted(self.timing_stats.items(), key=lambda x: -x[1]):
                pct = 100 * time / max(total_time, 1e-10)
                count = self.timing_counts[op]
                avg_ms = 1000 * time / max(count, 1)
                print(f"  {op:30s}: {pct:5.1f}% ({time:.2f}s, {count} calls, {avg_ms:.3f}ms avg)")
