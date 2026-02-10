"""
Temporal Sequence Learning Evaluation for Default Brain.

This script tests the default brain architecture on temporal sequence prediction,
evaluating:
- Next-step prediction accuracy (hippocampal memory)
- Pattern completion (given A-B, predict C)
- Violation detection (prediction error when patterns break)
- Generalization (novel symbol combinations)

Success Metrics:
- Basic: >50% accuracy on single pattern type
- Intermediate: >70% on multiple patterns
- Advanced: Detect violations, one-shot learning, noise robustness
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as scipy_signal
from tqdm import tqdm

from thalia.brain import BrainConfig, BrainBuilder, DynamicBrain
from thalia.brain.regions import (
    NeuralRegion,
    Cortex,
    Hippocampus,
    Thalamus,
)
from thalia.datasets import PatternType, SequenceConfig, TemporalSequenceDataset
from thalia.typing import BrainSpikesDict


# ============================================================================
# EEG/LFP Simulator
# ============================================================================


class NeuralSignalRecorder:
    """Simulates EEG-like Local Field Potential (LFP) signals from brain regions."""

    def __init__(self, sampling_rate_hz: float = 1000.0, buffer_size_sec: float = 5.0):
        """
        Args:
            sampling_rate_hz: Sampling rate (1000 Hz = 1ms resolution)
            buffer_size_sec: How many seconds of signal to keep for spectral analysis
        """
        self.sampling_rate = sampling_rate_hz
        self.buffer_size = int(sampling_rate_hz * buffer_size_sec)

        # LFP buffers for each region (circular buffers)
        self.lfp_hippocampus: List[float] = []
        self.lfp_cortex: List[float] = []
        self.lfp_combined: List[float] = []  # Simulated scalp EEG

        self.timestep = 0

    def record_timestep(self, brain: DynamicBrain, brain_output: Optional[BrainSpikesDict] = None) -> None:
        """Record LFP for current timestep using spike activity as proxy.

        Note: In real EEG, LFP comes from synchronized dendritic currents.
        We approximate this by using spike rate (normalized) as a proxy signal.
        This captures the rhythm and amplitude modulation of neural activity.

        Args:
            brain: The brain instance
            brain_output: Optional dict of region outputs from forward() for accurate spike counts
        """
        # Get recent spike activity from brain regions
        # Use spike count as proxy for population activity (normalized)

        thalamus = brain.get_first_region_of_type(Thalamus)
        cortex = brain.get_first_region_of_type(Cortex)
        hippocampus = brain.get_first_region_of_type(Hippocampus)

        # Hippocampus activity (CA1 output)
        hipp_lfp = 0.0
        if brain_output and "hippocampus" in brain_output:
            # Use current timestep output
            hipp_spikes = sum(s.float().sum().item() for s in brain_output["hippocampus"].values() if s is not None)
            hipp_lfp = hipp_spikes / 800.0  # Normalize by population size
        elif hippocampus is not None:
            if hasattr(hippocampus, 'output_spikes') and hippocampus.output_spikes is not None:
                hipp_spikes = hippocampus.output_spikes.float().sum().item()
                hipp_lfp = hipp_spikes / 800.0

        # Cortex activity (superficial layers generate most EEG)
        cortex_lfp = 0.0
        if brain_output and "cortex" in brain_output:
            # Use current timestep output
            cortex_spikes = sum(s.float().sum().item() for s in brain_output["cortex"].values() if s is not None)
            cortex_lfp = cortex_spikes / 200.0  # Normalize by population size
        elif cortex is not None:
            if hasattr(cortex, 'output_spikes') and cortex.output_spikes is not None:
                cortex_spikes = cortex.output_spikes.float().sum().item()
                cortex_lfp = cortex_spikes / 200.0

        # Combined signal (weighted by cortical contribution to scalp EEG)
        # Cortex contributes more to scalp EEG than hippocampus (deeper structure)
        combined_lfp = 0.7 * cortex_lfp + 0.3 * hipp_lfp

        # Add to buffers
        self.lfp_hippocampus.append(hipp_lfp)
        self.lfp_cortex.append(cortex_lfp)
        self.lfp_combined.append(combined_lfp)

        # Keep buffer size limited (check each list individually)
        if len(self.lfp_hippocampus) > self.buffer_size:
            self.lfp_hippocampus.pop(0)
        if len(self.lfp_cortex) > self.buffer_size:
            self.lfp_cortex.pop(0)
        if len(self.lfp_combined) > self.buffer_size:
            self.lfp_combined.pop(0)

        self.timestep += 1

    def get_power_spectrum(self, signal_type: str = 'combined') -> tuple:
        """
        Compute power spectral density using Welch's method.

        Args:
            signal_type: Which signal to analyze ('hippocampus', 'cortex', or 'combined')
        """
        if signal_type == 'hippocampus':
            signal_data = self.lfp_hippocampus
        elif signal_type == 'cortex':
            signal_data = self.lfp_cortex
        else:
            signal_data = self.lfp_combined

        if len(signal_data) < 10:
            return np.array([]), np.array([])

        # Convert list to numpy array for scipy
        signal_array = np.array(signal_data)

        # Welch's method for spectral estimation
        frequencies, psd = scipy_signal.welch(
            signal_array,
            fs=self.sampling_rate,
            nperseg=min(256, len(signal_data)),
            scaling='density'
        )

        return frequencies, psd

    def get_band_power(self, signal_type: str = 'combined') -> Dict[str, float]:
        """
        Compute power in standard EEG frequency bands.

        Returns:
            Dictionary with theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-100 Hz)
        """
        frequencies, psd = self.get_power_spectrum(signal_type)

        if len(frequencies) == 0:
            return {'theta': 0.0, 'alpha': 0.0, 'beta': 0.0, 'gamma': 0.0}

        # Define frequency bands
        bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
        }

        band_powers = {}
        for band_name, (low, high) in bands.items():
            idx = np.logical_and(frequencies >= low, frequencies <= high)
            # Use trapezoid (trapz was deprecated/removed in newer numpy)
            band_powers[band_name] = np.trapezoid(psd[idx], frequencies[idx]) if idx.any() else 0.0

        return band_powers


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for temporal sequence training."""

    # Brain configuration
    thalamus_relay_size: int = 50  # FIX: 50 neurons = 10 per symbol (population coding)
    cortex_size: int = 200  # Reduced from default 1000
    pfc_executive_size: int = 150
    striatum_actions: int = 5  # One per symbol

    # cerebellum_sizes = {'purkinje_size': 200, 'granule_size': 800}
    # cortex_sizes = {'l23_size': 133, 'l4_size': 66, 'l5_size': 66, 'l6a_size': 10, 'l6b_size': 50, 'vta_da': 20000}
    # hippocampus_sizes = {'dg_size': 796, 'ca3_size': 398, 'ca2_size': 199, 'ca1_size': 398, 'vta_da': 20000}
    # pfc_sizes = {'executive_size': 150, 'vta_da': 20000}
    # striatum_sizes = {'d1_size': 50, 'd2_size': 50, 'n_actions': 5, 'neurons_per_action': 10, 'vta_da': 20000}
    # thalamus_sizes = {'relay_size': 50, 'trn_size': 15}

    # Dataset configuration
    n_symbols: int = 5
    sequence_length: int = 10
    pattern_types: Optional[List[str]] = None  # ["ABC", "ABA", "AAB"]

    # Training configuration
    n_training_trials: int = 1000
    n_test_trials: int = 100
    n_violation_trials: int = 50

    # Timing (biological timesteps)
    timestep_ms: float = 1.0  # 1ms resolution
    timesteps_per_symbol: int = 15  # FIX: 15 timesteps = 3 theta cycles (theta ~5Hz) for reliable hippocampal spiking
    trial_duration_ms: float = 100.0  # 100ms per trial
    inter_trial_interval_ms: float = 50.0  # 50ms between trials

    # Learning parameters
    use_dopamine_modulation: bool = True
    dopamine_baseline: float = 0.0  # No external baseline - VTA handles tonic dopamine internally
    dopamine_lr: float = 1.0  # Full strength reward signal
    # With baseline=0.0 and lr=1.0:
    # - Correct: DA = 0.0 + (1.0)*1.0 = +1.0 (strong LTP burst)
    # - Incorrect: DA = 0.0 + (-0.5)*1.0 = -0.5 (LTD dip)
    # - Incorrect: DA = 0.15 + (-0.5)*0.5 = -0.10 (mild LTD dip)
    # This 4:1 ratio (0.65 vs -0.10) compensates for 80% errors during exploration!
    reward_on_correct: float = 1.0
    penalty_on_error: float = -0.5

    # Spike encoding
    spike_rate_active: float = 0.95  # INCREASED: 95% firing rate for stronger input signal
    spike_rate_baseline: float = 0.02  # Low background rate (keep sparse)

    # Evaluation
    min_spikes_for_prediction: int = 1  # Minimum spikes to count as prediction

    # Output
    output_dir: str = "data/results/sequence_learning"
    save_brain: bool = False
    verbose: bool = True

    # Real-time plotting
    enable_plots: bool = True
    plot_update_interval: int = 1  # Update plots every N trials

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.pattern_types is None:
            self.pattern_types = ["ABC", "ABA", "AAB"]


# ============================================================================
# Helper Functions
# ============================================================================


def encode_symbol_to_spikes(
    symbol: int,
    n_symbols: int,
    spike_rate_active: float,
    spike_rate_baseline: float,
    device: torch.device,
    thalamus_size: Optional[int] = None,
    add_temporal_jitter: bool = True,
) -> torch.Tensor:
    """
    Encode a symbol as spike rates (one-hot style).

    FIX (Feb 2026): Changed from Bernoulli sampling to deterministic population coding.
    Bernoulli with rate=0.8 produces 0-2 spikes per timestep (too sparse).
    Now uses population of neurons to reliably encode each symbol.

    FIX (Feb 2026): Added temporal jitter to prevent FFI from blocking repeated presentations.

    Args:
        symbol: Symbol index (0 to n_symbols-1)
        n_symbols: Total number of symbols
        spike_rate_active: Firing rate for active symbol (now interpreted as fraction of population)
        spike_rate_baseline: Background firing rate
        device: Torch device
        thalamus_size: Target size for thalamic input (pad if larger than n_symbols)
        add_temporal_jitter: If True, randomly shuffle which neurons spike to prevent FFI blocking

    Returns:
        Spike pattern (n_symbols or thalamus_size,) with deterministic spikes
    """
    # Calculate neurons per symbol
    target_size = thalamus_size if thalamus_size is not None else n_symbols
    neurons_per_symbol = target_size // n_symbols

    # Create spike pattern: each symbol gets a population of neurons
    spikes = torch.zeros(target_size, dtype=torch.bool, device=device)

    # Active symbol: activate a population of neurons
    start_idx = symbol * neurons_per_symbol
    end_idx = start_idx + neurons_per_symbol
    n_active = int(neurons_per_symbol * spike_rate_active)

    if add_temporal_jitter:
        # Randomly select which neurons spike (prevents FFI from detecting "no change")
        symbol_neurons = torch.arange(start_idx, end_idx, device=device)
        active_indices = symbol_neurons[torch.randperm(neurons_per_symbol, device=device)[:n_active]]
        spikes[active_indices] = True
    else:
        # Deterministic (always same neurons)
        spikes[start_idx:start_idx + n_active] = True

    # Baseline activity: sparse random spikes in remaining neurons
    if spike_rate_baseline > 0:
        # Add sparse baseline spikes to non-active neurons
        remaining = torch.arange(target_size, device=device)
        remaining = remaining[(remaining < start_idx) | (remaining >= end_idx)]
        n_baseline = int(len(remaining) * spike_rate_baseline)
        if n_baseline > 0:
            baseline_idx = remaining[torch.randperm(len(remaining), device=device)[:n_baseline]]
            spikes[baseline_idx] = True

    return spikes


def decode_spikes_to_symbol(
    spike_counts: torch.Tensor,
    min_spikes: int = 1,
) -> Optional[int]:
    """
    Decode spike counts to most active symbol.

    Args:
        spike_counts: Accumulated spike counts per symbol (n_symbols,)
        min_spikes: Minimum total spikes to count as valid prediction

    Returns:
        Symbol index or None if below threshold
    """
    # Ensure float for comparison
    if spike_counts.dtype == torch.bool:
        spike_counts = spike_counts.float()

    total_spikes = spike_counts.sum()
    if total_spikes < min_spikes:
        return None

    # Return symbol with most spikes (ties go to first index)
    return int(torch.argmax(spike_counts).item())


def compute_prediction_accuracy(
    predictions: List[Optional[int]],
    targets: List[int],
) -> float:
    """
    Compute prediction accuracy.

    Args:
        predictions: List of predicted symbols (None = no prediction)
        targets: List of target symbols

    Returns:
        Accuracy (0.0 to 1.0)
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(targets)}")

    correct = sum(1 for pred, tgt in zip(predictions, targets) if pred == tgt)
    return correct / len(targets) if targets else 0.0


def compute_prediction_error(
    prediction_spikes: torch.Tensor,
    target_symbol: int,
    n_symbols: int,
) -> float:
    """
    Compute prediction error for dopamine signal.

    Uses a simple error: actual_rate - predicted_rate for target symbol.

    Args:
        prediction_spikes: Predicted spike pattern (n_symbols,)
        target_symbol: Actual next symbol
        n_symbols: Total number of symbols

    Returns:
        Error magnitude (0.0 to 1.0, where 0 = perfect prediction)
    """
    # DIAGNOSTIC: Check if we're getting any prediction spikes
    total_spikes = prediction_spikes.sum().item()

    # Create target one-hot
    target_onehot = torch.zeros(n_symbols, device=prediction_spikes.device)
    target_onehot[target_symbol] = 1.0

    # Normalize prediction spikes to rates
    # If no spikes at all, this is maximum error (uniform distribution = 0.2 for each symbol)
    if total_spikes < 0.1:
        # No prediction = uniform distribution
        pred_rates = torch.ones(n_symbols, device=prediction_spikes.device) / n_symbols
    else:
        pred_rates = prediction_spikes / (prediction_spikes.sum() + 1e-6)

    # Mean squared error
    error = F.mse_loss(pred_rates, target_onehot).item()

    return error


# ============================================================================
# Main Training Script
# ============================================================================


class SequenceLearningExperiment:
    """Runs temporal sequence learning experiment."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.brain: Optional[DynamicBrain] = None
        self.dataset: Optional[TemporalSequenceDataset] = None

        # Spike accumulation buffer (for temporal decoding)
        self.spike_buffer: Optional[torch.Tensor] = None
        self.buffer_size: int = 5  # Accumulate over 5 timesteps

        # Metrics
        self.metrics: Dict[str, List[float]] = {
            "train_accuracy": [],
            "train_dopamine": [],
            "test_accuracy": [],
            "violation_detection": [],
        }

        # Real-time tracking for plots
        self.trial_accuracies: List[float] = []
        self.trial_hippocampus_spikes: List[float] = []
        self.trial_cortex_spikes: List[float] = []
        self.trial_dopamine: List[float] = []

        # LFP/EEG recorder
        self.lfp_recorder = NeuralSignalRecorder(sampling_rate_hz=1000.0, buffer_size_sec=3.0)

        # Setup plotting if enabled
        self.fig = None
        self.axes = None
        if config.enable_plots:
            self._setup_plots()

    def setup(self):
        """Initialize brain and dataset."""
        if self.config.verbose:
            print("=" * 80)
            print("TEMPORAL SEQUENCE LEARNING EVALUATION")
            print("=" * 80)
            print(f"\nDevice: {self.device}")
            print(f"Output directory: {self.output_dir}")

        # Build brain with default architecture
        if self.config.verbose:
            print("\n[1/2] Building default brain architecture...")

        brain_config = BrainConfig(device=self.device, dt_ms=self.config.timestep_ms)
        self.brain = BrainBuilder.preset(
            name="default",
            brain_config=brain_config,
            thalamus_relay_size=self.config.thalamus_relay_size,
            cortex_size=self.config.cortex_size,
            pfc_executive_size=self.config.pfc_executive_size,
            striatum_actions=self.config.striatum_actions,
        )

        thalamus = self.brain.get_first_region_of_type(Thalamus)
        cortex = self.brain.get_first_region_of_type(Cortex)
        hippocampus = self.brain.get_first_region_of_type(Hippocampus)

        if self.config.verbose:
            print(f"   ✓ Brain created with {self._count_parameters()} parameters")
            print(f"   ✓ Regions: {list(self.brain.regions.keys())}")

            def _print_region_info(region: Optional[NeuralRegion], additional_attrs: Optional[List[str]] = None):
                if region is not None:
                    print(f"  {region.__class__.__name__}:")
                    print(f"    - Learning strategy: {region.learning_strategy.__class__.__name__ if region.learning_strategy else 'NONE'}")
                    print(f"    - Input sources: {list(region.input_sources.keys())}")
                    print(f"    - Synaptic weights: {list(region.synaptic_weights.keys())}")
                    for attr in additional_attrs or []:
                        print(f"    - {attr}: {getattr(region, attr) if hasattr(region, attr) else 'N/A'}")

            # DEBUG: Verify critical connectivity and learning setup
            print("\n[CONNECTIVITY & LEARNING DEBUG]")
            _print_region_info(thalamus, ["relay_size"])
            _print_region_info(cortex)
            _print_region_info(hippocampus)

        if self.config.verbose:
            print("\n[CRITICAL FIX] Attaching learning strategies...")

        from thalia.learning.strategies import ThreeFactorStrategy, ThreeFactorConfig

        # Create dopamine-modulated learning for hippocampus
        hipp_learning_config = ThreeFactorConfig(
            learning_rate=0.001,  # Conservative rate for stability
            eligibility_tau=100.0,  # Eligibility trace decay (ms) - matches our symbol timing
            modulator_tau=50.0,  # Modulator (dopamine) decay (ms)
            device=self.device,
        )
        hippocampus.learning_strategy = ThreeFactorStrategy(
            config=hipp_learning_config
        )

        # Create dopamine-modulated learning for cortex
        cortex_learning_config = ThreeFactorConfig(
            learning_rate=0.001,
            eligibility_tau=100.0,
            modulator_tau=50.0,
            device=self.device,
        )
        cortex.learning_strategy = ThreeFactorStrategy(
            config=cortex_learning_config
        )

        if self.config.verbose:
            print(f"   ✓ Hippocampus: ThreeFactorStrategy (lr={hipp_learning_config.learning_rate}, elig_tau={hipp_learning_config.eligibility_tau}ms)")
            print(f"   ✓ Cortex: ThreeFactorStrategy (lr={cortex_learning_config.learning_rate}, elig_tau={cortex_learning_config.eligibility_tau}ms)")

        # Create dataset
        if self.config.verbose:
            print("\n[2/2] Creating temporal sequence dataset...")

        pattern_types = [PatternType(pt.lower()) for pt in self.config.pattern_types]
        seq_config = SequenceConfig(
            n_symbols=self.config.n_symbols,
            sequence_length=self.config.sequence_length,
            pattern_types=pattern_types,
            violation_probability=0.0,  # Controlled separately
            device=self.device,
        )
        self.dataset = TemporalSequenceDataset(seq_config)

        if self.config.verbose:
            print(f"   ✓ Dataset created: {self.config.n_symbols} symbols, "
                  f"{self.config.sequence_length} steps")
            print(f"   ✓ Pattern types: {self.config.pattern_types}")

    def _accumulate_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Accumulate spikes over a temporal window for rate-based decoding.

        Args:
            spikes: Current timestep spikes (n_symbols,) bool tensor

        Returns:
            Accumulated spike counts (n_symbols,) float tensor
        """
        if self.spike_buffer is None:
            self.spike_buffer = torch.zeros(
                self.buffer_size,
                self.config.n_symbols,
                device=self.device
            )

        # Shift buffer and add new spikes
        self.spike_buffer = torch.roll(self.spike_buffer, shifts=-1, dims=0)
        self.spike_buffer[-1] = spikes.float()

        # Return accumulated counts
        return self.spike_buffer.sum(dim=0)

    def _reset_spike_buffer(self):
        """Reset spike accumulation buffer between trials."""
        self.spike_buffer = None

    def _setup_plots(self):
        """Setup real-time plotting with EEG/LFP visualization."""
        plt.ion()  # Enable interactive mode
        self.fig, self.axes = plt.subplots(3, 2, figsize=(14, 10))
        self.fig.suptitle('Sequence Learning Training Progress + Neural Signals (EEG-like)', fontsize=14, fontweight='bold')

        # Configure subplots
        ax_acc, ax_spikes, ax_da, ax_weights, ax_lfp, ax_spectrum = self.axes.flatten()

        ax_acc.set_title('Rolling Accuracy (50-trial window)')
        ax_acc.set_xlabel('Trial')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_ylim(0, 1)
        ax_acc.axhline(y=0.2, color='r', linestyle='--', alpha=0.3, label='Random baseline')
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend()

        ax_spikes.set_title('Neural Activity per Trial')
        ax_spikes.set_xlabel('Trial')
        ax_spikes.set_ylabel('Average Spikes')
        ax_spikes.grid(True, alpha=0.3)

        ax_da.set_title('Dopamine Signal')
        ax_da.set_xlabel('Trial')
        ax_da.set_ylabel('Dopamine')
        ax_da.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax_da.grid(True, alpha=0.3)

        ax_weights.set_title('CA3 Recurrent Weight Statistics')
        ax_weights.set_xlabel('Trial')
        ax_weights.set_ylabel('Weight Mean')
        ax_weights.grid(True, alpha=0.3)

        # LFP time series (last 2 seconds)
        ax_lfp.set_title('Local Field Potential (LFP) - Last 2 seconds')
        ax_lfp.set_xlabel('Time (ms)')
        ax_lfp.set_ylabel('Membrane Potential (mV)')
        ax_lfp.grid(True, alpha=0.3)

        # Power spectrum
        ax_spectrum.set_title('Power Spectrum (Hippocampus)')
        ax_spectrum.set_xlabel('Frequency (Hz)')
        ax_spectrum.set_ylabel('Power (log scale)')
        ax_spectrum.set_xlim(0, 50)  # Focus on theta-gamma range
        ax_spectrum.grid(True, alpha=0.3)
        # Add frequency band markers
        ax_spectrum.axvspan(4, 8, alpha=0.1, color='blue', label='Theta (4-8 Hz)')
        ax_spectrum.axvspan(8, 13, alpha=0.1, color='green', label='Alpha (8-13 Hz)')
        ax_spectrum.axvspan(30, 50, alpha=0.1, color='red', label='Gamma (30-50 Hz)')
        ax_spectrum.legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        plt.show(block=False)

    def _update_plots(self):
        """Update real-time plots with current metrics."""
        if not self.config.enable_plots or self.fig is None:
            return

        ax_acc, ax_spikes, ax_da, ax_weights, ax_lfp, ax_spectrum = self.axes.flatten()

        # Calculate rolling accuracy
        window = 50
        if len(self.trial_accuracies) >= window:
            rolling_acc = [
                np.mean(self.trial_accuracies[max(0, i-window):i+1])
                for i in range(len(self.trial_accuracies))
            ]
        else:
            rolling_acc = self.trial_accuracies.copy()

        trials = list(range(1, len(self.trial_accuracies) + 1))

        # Update accuracy plot
        ax_acc.clear()
        ax_acc.set_title('Rolling Accuracy (50-trial window)')
        ax_acc.set_xlabel('Trial')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_ylim(0, 1)
        ax_acc.axhline(y=0.2, color='r', linestyle='--', alpha=0.3, label='Random baseline')
        if rolling_acc:
            ax_acc.plot(trials, rolling_acc, 'b-', linewidth=2, label='Accuracy')
            ax_acc.scatter(trials[-1:], rolling_acc[-1:], c='blue', s=50, zorder=5)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend()

        # Update spike activity plot
        ax_spikes.clear()
        ax_spikes.set_title('Neural Activity per Trial')
        ax_spikes.set_xlabel('Trial')
        ax_spikes.set_ylabel('Average Spikes')
        if self.trial_hippocampus_spikes:
            ax_spikes.plot(trials, self.trial_hippocampus_spikes, 'g-', label='Hippocampus', linewidth=1.5)
        if self.trial_cortex_spikes:
            ax_spikes.plot(trials, self.trial_cortex_spikes, 'orange', label='Cortex', linewidth=1.5, alpha=0.7)
        ax_spikes.grid(True, alpha=0.3)
        ax_spikes.legend()

        # Update dopamine plot
        ax_da.clear()
        ax_da.set_title('Dopamine Signal')
        ax_da.set_xlabel('Trial')
        ax_da.set_ylabel('Dopamine')
        ax_da.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        if self.trial_dopamine:
            ax_da.plot(trials, self.trial_dopamine, 'purple', linewidth=1.5)
            # Add rolling mean
            if len(self.trial_dopamine) > 10:
                rolling_da = [
                    np.mean(self.trial_dopamine[max(0, i-10):i+1])
                    for i in range(len(self.trial_dopamine))
                ]
                ax_da.plot(trials, rolling_da, 'purple', linewidth=2, alpha=0.5, linestyle='--', label='10-trial avg')
                ax_da.legend()
        ax_da.grid(True, alpha=0.3)

        # Update weights plot (sample periodically to avoid slowdown)
        ax_weights.clear()
        ax_weights.set_title('CA3 Recurrent Weight Statistics')
        ax_weights.set_xlabel('Trial')
        ax_weights.set_ylabel('Weight Mean')
        if hasattr(self, 'weight_history'):
            weight_trials = list(range(1, len(self.weight_history) + 1))
            ax_weights.plot(weight_trials, self.weight_history, 'teal', linewidth=2)
        ax_weights.grid(True, alpha=0.3)

        # Update LFP time series (last 2000 ms)
        ax_lfp.clear()
        ax_lfp.set_title('Local Field Potential (LFP) - Last 2 seconds')
        ax_lfp.set_xlabel('Time (ms)')
        ax_lfp.set_ylabel('Membrane Potential (mV)')

        if len(self.lfp_recorder.lfp_hippocampus) > 0:
            # Show last 2000 ms (2000 samples at 1ms resolution)
            n_samples = min(2000, len(self.lfp_recorder.lfp_hippocampus))
            time_axis = np.arange(n_samples)

            hipp_signal = self.lfp_recorder.lfp_hippocampus[-n_samples:]
            cortex_signal = self.lfp_recorder.lfp_cortex[-n_samples:]

            ax_lfp.plot(time_axis, hipp_signal, 'g-', label='Hippocampus', linewidth=1, alpha=0.8)
            ax_lfp.plot(time_axis, cortex_signal, 'orange', label='Cortex', linewidth=1, alpha=0.6)
            ax_lfp.legend(fontsize=8)
        ax_lfp.grid(True, alpha=0.3)

        # NEW: Update power spectrum
        ax_spectrum.clear()
        ax_spectrum.set_title('Power Spectrum (Hippocampus)')
        ax_spectrum.set_xlabel('Frequency (Hz)')
        ax_spectrum.set_ylabel('Power (log scale)')
        ax_spectrum.set_xlim(0, 50)

        # Get power spectrum
        frequencies, psd = self.lfp_recorder.get_power_spectrum('hippocampus')
        if len(frequencies) > 0:
            # Plot on log scale
            ax_spectrum.semilogy(frequencies, psd, 'g-', linewidth=2)

            # Highlight frequency bands
            ax_spectrum.axvspan(4, 8, alpha=0.1, color='blue', label='Theta (4-8 Hz)')
            ax_spectrum.axvspan(8, 13, alpha=0.1, color='green', label='Alpha (8-13 Hz)')
            ax_spectrum.axvspan(30, 50, alpha=0.1, color='red', label='Gamma (30-50 Hz)')

            # Show band powers as text
            band_powers = self.lfp_recorder.get_band_power('hippocampus')
            text_y = ax_spectrum.get_ylim()[1] * 0.7
            ax_spectrum.text(6, text_y, f"θ: {band_powers['theta']:.2e}", fontsize=8, color='blue')
            ax_spectrum.text(10.5, text_y, f"α: {band_powers['alpha']:.2e}", fontsize=8, color='green')
            ax_spectrum.text(40, text_y, f"γ: {band_powers['gamma']:.2e}", fontsize=8, color='red')

            ax_spectrum.legend(fontsize=8, loc='upper right')
        ax_spectrum.grid(True, alpha=0.3)

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _close_plots(self):
        """Close plotting windows."""
        if self.fig is not None:
            plt.close(self.fig)

    def train(self):
        """Run training phase."""
        assert self.brain is not None and self.dataset is not None, (
            "Brain and dataset must be initialized before training."
        )

        thalamus = self.brain.get_first_region_of_type(Thalamus)
        cortex = self.brain.get_first_region_of_type(Cortex)
        hippocampus = self.brain.get_first_region_of_type(Hippocampus)

        if self.config.verbose:
            print("\n" + "=" * 80)
            print("TRAINING PHASE")
            print("=" * 80)

        train_predictions = []
        train_targets = []
        train_dopamine_values = []

        progress = tqdm(
            range(self.config.n_training_trials),
            desc="Training",
            disable=not self.config.verbose,
        )

        for trial_idx in progress:
            # Reset spike buffer at start of each trial
            self._reset_spike_buffer()

            # Generate sequence (brain maintains continuous learning state)
            sequence, targets, _pattern_type = self.dataset.generate_sequence()

            # Convert to symbol indices (assuming one-hot encoding)
            symbol_sequence = [torch.argmax(seq).item() for seq in sequence]
            target_sequence = [torch.argmax(tgt).item() for tgt in targets]

            # Present sequence timestep by timestep
            trial_predictions = []
            trial_targets = []
            trial_dopamine = []

            for t in range(len(symbol_sequence) - 1):
                # Current symbol
                current_symbol = symbol_sequence[t]
                next_symbol = target_sequence[t]  # What should come next

                # FIX (Feb 2026): Theta-Aware Presentation
                # ==========================================
                # The hippocampus exhibits realistic theta oscillations (~5 Hz, spiking every ~5 timesteps).
                # To ensure hippocampus receives input during its receptive phases, we present each
                # symbol for multiple timesteps (15ms = 3 theta cycles). This allows:
                # 1. Hippocampus to spike during at least 2-3 theta peaks (t=1, 6, 11)
                # 2. Sufficient integration time for cortex→hippocampus propagation
                # 3. Hebbian learning to occur during theta trough encoding phases
                #
                # Accumulate predictions across all timesteps (hippocampus spikes at different times)
                accumulated_prediction = torch.zeros(self.config.n_symbols, device=self.device)

                # CRITICAL FIX: Reset spike buffer at start of each symbol
                # The spike_buffer accumulates across calls to _accumulate_spikes()
                # If not reset, predictions accumulate across the entire sequence!
                if self.spike_buffer is not None:
                    self.spike_buffer.zero_()

                # Track neural activity for plotting
                total_hipp_spikes = 0
                total_cortex_spikes = 0

                # DELIVER DOPAMINE NOW (before presenting symbol)
                # Three-factor learning needs dopamine DURING neural activity to modulate eligibility traces.
                # If we wait until after all 15 timesteps, traces decay and learning doesn't happen.
                # Solution: Deliver dopamine from the PREVIOUS prediction before presenting the NEXT symbol.
                if self.config.use_dopamine_modulation and t > 0:  # Skip first symbol (no previous prediction)
                    # Use the prediction from the PREVIOUS symbol to compute reward
                    prev_prediction = trial_predictions[-1] if trial_predictions else None
                    prev_target = trial_targets[-1] if trial_targets else None

                    if prev_prediction is not None and prev_target is not None:
                        # BIOLOGICALLY-ACCURATE REWARD-BASED DOPAMINE
                        # ============================================
                        # Real dopamine neurons encode REWARD PREDICTION ERROR (RPE), not MSE.
                        # RPE = (actual reward) - (expected reward)
                        #
                        # For supervised learning:
                        # - Correct prediction → Positive reward → Dopamine BURST (LTP)
                        # - Incorrect prediction → Negative reward → Dopamine DIP (LTD)
                        #
                        # This creates strong learning signal:
                        # - Correct: DA ~ +0.5 (strengthen correct associations)
                        # - Incorrect: DA ~ -0.5 (weaken incorrect associations)
                        #
                        # Previous approach used MSE between distributions (~0.15 always),
                        # which produced flat DA signal and prevented learning.

                        # Binary reward based on task success
                        if prev_prediction == prev_target:
                            reward = self.config.reward_on_correct  # +1.0 for correct
                        else:
                            reward = self.config.penalty_on_error  # -0.5 for incorrect

                        # Convert reward to dopamine (scaled by learning rate)
                        # NOTE: VTA handles tonic baseline internally - don't add it here!
                        dopamine = reward * self.config.dopamine_lr
                        dopamine = np.clip(dopamine, -1.0, 1.0)

                        # Deliver dopamine NOW, while presenting current symbol
                        # (eligibility traces from previous symbol are still ~85% active with tau=100ms)
                        self.brain.deliver_reward(external_reward=dopamine)

                        # DEBUG: Verify dopamine actually reaches hippocampus
                        if trial_idx < 3:  # Only first 3 trials for brevity
                            actual_da = hippocampus.dopamine.item() if hasattr(hippocampus, 'dopamine') else None
                            da_str = f"{actual_da:.3f}" if actual_da is not None else "N/A"
                            print(f"    [DOPAMINE DEBUG] Delivered: {dopamine:.3f}, Hippocampus received: {da_str}")

                        trial_dopamine.append(dopamine)

                for ts in range(self.config.timesteps_per_symbol):
                    # FIX: Regenerate spikes each timestep with temporal jitter
                    # This prevents FFI from blocking and allows sustained cortical/hippocampal activity
                    input_spikes = encode_symbol_to_spikes(
                        current_symbol,
                        self.config.n_symbols,
                        self.config.spike_rate_active,
                        self.config.spike_rate_baseline,
                        self.device,
                        thalamus_size=thalamus.relay_size,
                        add_temporal_jitter=True,  # Randomize which neurons spike each timestep
                    )

                    # Forward through brain
                    # Thalamus receives external sensory input via ascending pathways
                    # (e.g., retina→LGN, cochlea→MGN, mechanoreceptors→VPN)
                    brain_input = {"thalamus": {"sensory": input_spikes}}
                    brain_output = self.brain.forward(brain_input)

                    # COLLECT CA3 SPIKES for learning (accumulate across timesteps)
                    if ts == 0:
                        symbol_ca3_spikes = torch.zeros_like(hippocampus.ca3_spikes, dtype=torch.float32)
                    symbol_ca3_spikes += hippocampus.ca3_spikes.float()

                    # Record LFP for this timestep (EEG-like signal)
                    # Pass brain_output for accurate spike counts
                    self.lfp_recorder.record_timestep(self.brain, brain_output)

                    # Track activity for plots
                    if "hippocampus" in brain_output:
                        hipp_spikes = sum(s.float().sum().item() for s in brain_output["hippocampus"].values())
                        total_hipp_spikes += hipp_spikes
                    if "cortex" in brain_output:
                        cortex_spikes = sum(s.float().sum().item() for s in brain_output["cortex"].values())
                        total_cortex_spikes += cortex_spikes

                    # Read prediction at THIS timestep and accumulate
                    # Read hippocampus CA1 output (memory prediction)
                    # brain_output is Dict[region_name, Dict[population_name, Tensor]]
                    # Get first available population from preferred region
                    if "hippocampus" in brain_output and brain_output["hippocampus"]:
                        prediction_spikes = list(brain_output["hippocampus"].values())[0]
                    elif "cortex" in brain_output and brain_output["cortex"]:
                        prediction_spikes = list(brain_output["cortex"].values())[0]
                    elif "pfc" in brain_output and brain_output["pfc"]:
                        prediction_spikes = list(brain_output["pfc"].values())[0]
                    else:
                        # Fallback: use any available output from any region
                        for region_output in brain_output.values():
                            if region_output:
                                prediction_spikes = list(region_output.values())[0]
                                break

                    # FIX: Pool distributed hippocampal activity into symbol space
                    # Hippocampus outputs 796D distributed representation, not 5D symbol space
                    # Solution: Divide neurons into chunks and sum spikes per chunk
                    if prediction_spikes.shape[0] > self.config.n_symbols:
                        # Pool by dividing population into symbol-sized chunks
                        neurons_per_symbol = prediction_spikes.shape[0] // self.config.n_symbols
                        symbol_counts = torch.zeros(self.config.n_symbols, device=self.device)

                        for symbol_idx in range(self.config.n_symbols):
                            start_idx = symbol_idx * neurons_per_symbol
                            end_idx = start_idx + neurons_per_symbol
                            symbol_counts[symbol_idx] = prediction_spikes[start_idx:end_idx].float().sum()

                        prediction_spikes = symbol_counts
                    elif prediction_spikes.shape[0] < self.config.n_symbols:
                        # Pad if needed
                        padding = torch.zeros(
                            self.config.n_symbols - prediction_spikes.shape[0],
                            device=self.device,
                            dtype=prediction_spikes.dtype,
                        )
                        prediction_spikes = torch.cat([prediction_spikes, padding])

                    # Accumulate this timestep's prediction
                    accumulated_prediction += prediction_spikes.float()

                # Save last activity counts for this trial
                self._last_hippocampus_spikes = total_hipp_spikes / self.config.timesteps_per_symbol
                self._last_cortex_spikes = total_cortex_spikes / self.config.timesteps_per_symbol

                # CRITICAL: Trigger learning after each symbol
                # Learning happens after dopamine delivery (at start of next symbol)
                # Eligibility traces from previous symbol are still active (~85% after 15ms)
                if self.config.use_dopamine_modulation:
                    # Update hippocampus recurrent weights (CA3→CA3)
                    if hasattr(hippocampus, 'learning_strategy') and hippocampus.learning_strategy:
                        # Use accumulated CA3 spikes from the symbol period
                        ca3_spikes = symbol_ca3_spikes if 'symbol_ca3_spikes' in locals() else None

                        if ca3_spikes is not None and "ca3_ca3" in hippocampus.synaptic_weights:
                            # DEBUG: First trial only
                            if trial_idx == 0 and t == 0:
                                print("\n[LEARNING DEBUG]")
                                print(f"  Symbol CA3 spikes: shape={ca3_spikes.shape}, sum={ca3_spikes.sum().item():.1f}")
                                print(f"  CA3 weights shape: {hippocampus.synaptic_weights['ca3_ca3'].shape}")
                                print(f"  Dopamine: {hippocampus.neuromodulator_state.dopamine:.3f}")
                                print(f"  Learning strategy: {hippocampus.learning_strategy.__class__.__name__}")

                            metrics = hippocampus._apply_strategy_learning(
                                pre_activity=ca3_spikes,
                                post_activity=ca3_spikes,  # Recurrent connection
                                weights=hippocampus.synaptic_weights["ca3_ca3"],
                                modulator=hippocampus.neuromodulator_state.dopamine,
                            )

                            # DEBUG: Print metrics
                            if trial_idx == 0 and t == 0 and metrics:
                                print(f"  Learning metrics: {metrics}")
                        elif trial_idx == 0 and t == 0:
                            print("\n[LEARNING DEBUG] Symbol CA3 spikes NOT available!")
                            print(f"  symbol_ca3_spikes in locals: {'symbol_ca3_spikes' in locals()}")
                            print(f"  Has _ca3_spikes attr: {hasattr(hippocampus, '_ca3_spikes')}")
                            print(f"  Available attrs: {[a for a in dir(hippocampus) if 'ca3' in a.lower() and 'spike' in a.lower()]}")

                # Use accumulated prediction across all timesteps
                spike_counts = self._accumulate_spikes(accumulated_prediction)

                # Decode prediction from accumulated counts
                predicted_symbol = decode_spikes_to_symbol(
                    spike_counts,
                    self.config.min_spikes_for_prediction,
                )

                trial_predictions.append(predicted_symbol)
                trial_targets.append(next_symbol)

                # Save spike counts for next iteration's dopamine delivery
                # (dopamine is delivered at START of next symbol, while traces are fresh)
                self._last_prediction_spikes = spike_counts

                # DIAGNOSTIC: Print prediction details every 10th trial, first step
                if self.config.use_dopamine_modulation and (trial_idx < 5 or (trial_idx % 10 == 0 and t == 0)):
                    # Compute reward for diagnostic display
                    is_correct = (predicted_symbol == next_symbol)
                    reward = self.config.reward_on_correct if is_correct else self.config.penalty_on_error
                    dopamine = reward * self.config.dopamine_lr  # No baseline - VTA handles it
                    dopamine = np.clip(dopamine, -1.0, 1.0)

                    print(f"\n[Trial {trial_idx}, Step {t}] Prediction:")
                    print(f"  Current symbol: {current_symbol}, Target: {next_symbol}, Predicted: {predicted_symbol} {'✓' if is_correct else '✗'}")
                    print(f"  Spike counts: {spike_counts.cpu().numpy()}")
                    print(f"  Reward: {'+' if reward > 0 else ''}{reward:.2f} ({'correct' if is_correct else 'incorrect'})")
            # DELIVER FINAL DOPAMINE for last symbol in sequence
            if self.config.use_dopamine_modulation and trial_predictions and trial_targets:
                # Use binary reward for last prediction
                last_prediction = trial_predictions[-1]
                last_target = trial_targets[-1]

                if last_prediction == last_target:
                    reward = self.config.reward_on_correct
                else:
                    reward = self.config.penalty_on_error

                dopamine = reward * self.config.dopamine_lr  # No baseline - VTA handles it
                dopamine = np.clip(dopamine, -1.0, 1.0)
                self.brain.deliver_reward(external_reward=dopamine)
                trial_dopamine.append(dopamine)

            # Accumulate metrics
            train_predictions.extend(trial_predictions)
            train_targets.extend(trial_targets)
            train_dopamine_values.extend(trial_dopamine)

            # Track per-trial metrics for plotting
            trial_acc = compute_prediction_accuracy(trial_predictions, trial_targets) if trial_predictions else 0.0
            self.trial_accuracies.append(trial_acc)

            # Track average dopamine this trial
            avg_trial_da = np.mean(trial_dopamine) if trial_dopamine else 0.0
            self.trial_dopamine.append(avg_trial_da)

            # Track neural activity (sample from last forward pass)
            # Get activity counts from the sequence presentation
            if hasattr(self, '_last_hippocampus_spikes'):
                self.trial_hippocampus_spikes.append(self._last_hippocampus_spikes)
            if hasattr(self, '_last_cortex_spikes'):
                self.trial_cortex_spikes.append(self._last_cortex_spikes)

            # DIAGNOSTIC: Print trial summary for first 5 trials and every 10th
            if trial_idx < 5 or trial_idx % 10 == 0:
                print(f"\n{'='*60}")
                print(f"TRIAL {trial_idx} SUMMARY:")
                print(f"  Accuracy: {trial_acc:.2%} ({sum(1 for p, t in zip(trial_predictions, trial_targets) if p == t)}/{len(trial_predictions)} correct)")
                print(f"  Predictions: {trial_predictions}")
                print(f"  Targets:     {trial_targets}")
                print(f"  Avg Dopamine: {avg_trial_da:.4f} (range: [{min(trial_dopamine):.4f}, {max(trial_dopamine):.4f}])")
                print(f"  Hippocampus spikes/symbol: {self._last_hippocampus_spikes:.1f}")

                # DEBUG: Learning diagnostics
                hipp_has_learning = hasattr(hippocampus, 'learning_strategy') and hippocampus.learning_strategy is not None
                hipp_learning_name = hippocampus.learning_strategy.__class__.__name__ if hipp_has_learning else 'NONE'

                # Check eligibility traces (stored in learning strategy)
                elig_trace = 0.0
                if hipp_has_learning and hasattr(hippocampus.learning_strategy, 'eligibility_trace'):
                    if isinstance(hippocampus.learning_strategy.eligibility_trace, torch.Tensor):
                        elig_trace = hippocampus.learning_strategy.eligibility_trace.mean().item()

                print(f"  [LEARNING] Strategy: {hipp_learning_name}")
                print(f"  [LEARNING] CA3 eligibility trace: {elig_trace:.6f}")
                if hasattr(hippocampus, 'dopamine'):
                    print(f"  [LEARNING] Hippocampus DA: {hippocampus.dopamine.item():.3f}")
                current_w = hippocampus.synaptic_weights["ca3_ca3"].mean().item()
                if trial_idx == 0:
                    self._initial_weight = current_w
                    print(f"  CA3 weights: {current_w:.6f} (initial)")
                else:
                    weight_change = current_w - self._initial_weight
                    print(f"  CA3 weights: {current_w:.6f} (Δ{weight_change:+.6f} from initial)")

                    # Check if weights are changing at all
                    if abs(weight_change) < 1e-6:
                        print("  ⚠ WARNING: Weights not changing! Learning may not be working.")
                print(f"{'='*60}\n")

            # Update plots periodically
            if self.config.enable_plots and (trial_idx + 1) % self.config.plot_update_interval == 0:
                # Sample CA3 weights every 10 trials
                if not hasattr(self, 'weight_history'):
                    self.weight_history = []
                w_mean = hippocampus.synaptic_weights["ca3_ca3"].mean().item()
                self.weight_history.append(w_mean)

                self._update_plots()

            # Update progress bar every 100 trials
            if trial_idx % 100 == 0 and trial_idx > 0:
                recent_acc = compute_prediction_accuracy(
                    train_predictions[-100:],
                    train_targets[-100:],
                )
                progress.set_postfix({"accuracy": f"{recent_acc:.2%}"})

        # Compute final training accuracy
        train_accuracy = compute_prediction_accuracy(train_predictions, train_targets)
        avg_dopamine = np.mean(train_dopamine_values) if train_dopamine_values else 0.0

        self.metrics["train_accuracy"].append(train_accuracy)
        self.metrics["train_dopamine"].append(avg_dopamine)

        if self.config.verbose:
            print("\n✓ Training complete:")
            print(f"   Accuracy: {train_accuracy:.2%}")
            print(f"   Avg Dopamine: {avg_dopamine:.3f}")
            print(f"   Trials: {self.config.n_training_trials}")

    def test(self):
        """Run testing phase (novel sequences)."""
        assert self.brain is not None and self.dataset is not None, (
            "Brain and dataset must be initialized before testing."
        )

        thalamus = self.brain.get_first_region_of_type(Thalamus)
        cortex = self.brain.get_first_region_of_type(Cortex)
        hippocampus = self.brain.get_first_region_of_type(Hippocampus)

        if self.config.verbose:
            print("\n" + "=" * 80)
            print("TESTING PHASE")
            print("=" * 80)

        test_predictions = []
        test_targets = []

        progress = tqdm(
            range(self.config.n_test_trials),
            desc="Testing",
            disable=not self.config.verbose,
        )

        for _trial_idx in progress:
            # Reset spike buffer at start of each trial
            self._reset_spike_buffer()

            # Generate NEW sequence (not seen in training)
            sequence, targets, pattern_type = self.dataset.generate_sequence()
            symbol_sequence = [torch.argmax(seq).item() for seq in sequence]
            target_sequence = [torch.argmax(tgt).item() for tgt in targets]

            # Present sequence (no learning)
            for t in range(len(symbol_sequence) - 1):
                current_symbol = symbol_sequence[t]
                next_symbol = target_sequence[t]

                input_spikes = encode_symbol_to_spikes(
                    current_symbol,
                    self.config.n_symbols,
                    self.config.spike_rate_active,
                    self.config.spike_rate_baseline,
                    self.device,
                    thalamus_size=thalamus.relay_size,
                )

                # Thalamus receives external sensory input via ascending pathways
                brain_input = {"thalamus": {"sensory": input_spikes}}
                brain_output = self.brain.forward(brain_input)

                # Get prediction (same logic as training)
                if "hippocampus" in brain_output and brain_output["hippocampus"]:
                    prediction_spikes = list(brain_output["hippocampus"].values())[0]
                elif "cortex" in brain_output and brain_output["cortex"]:
                    prediction_spikes = list(brain_output["cortex"].values())[0]
                else:
                    for region_output in brain_output.values():
                        if region_output:
                            prediction_spikes = list(region_output.values())[0]
                            break

                if prediction_spikes.shape[0] > self.config.n_symbols:
                    prediction_spikes = prediction_spikes[: self.config.n_symbols]
                elif prediction_spikes.shape[0] < self.config.n_symbols:
                    padding = torch.zeros(
                        self.config.n_symbols - prediction_spikes.shape[0],
                        device=self.device,
                        dtype=prediction_spikes.dtype,
                    )
                    prediction_spikes = torch.cat([prediction_spikes, padding])

                # Accumulate spikes over temporal window
                spike_counts = self._accumulate_spikes(prediction_spikes)

                predicted_symbol = decode_spikes_to_symbol(
                    spike_counts,
                    self.config.min_spikes_for_prediction,
                )

                test_predictions.append(predicted_symbol)
                test_targets.append(next_symbol)

        # Compute test accuracy
        test_accuracy = compute_prediction_accuracy(test_predictions, test_targets)
        self.metrics["test_accuracy"].append(test_accuracy)

        if self.config.verbose:
            print("\n✓ Testing complete:")
            print(f"   Accuracy: {test_accuracy:.2%}")
            print(f"   Trials: {self.config.n_test_trials}")
            print(f"   Random baseline: {1.0 / self.config.n_symbols:.2%}")

    def test_violation_detection(self):
        """Test if brain detects pattern violations."""
        assert self.brain is not None and self.dataset is not None, (
            "Brain and dataset must be initialized before testing."
        )

        thalamus = self.brain.get_first_region_of_type(Thalamus)
        cortex = self.brain.get_first_region_of_type(Cortex)
        hippocampus = self.brain.get_first_region_of_type(Hippocampus)

        if self.config.verbose:
            print("\n" + "=" * 80)
            print("VIOLATION DETECTION")
            print("=" * 80)

        # Test 1: Normal patterns
        normal_errors = []
        progress = tqdm(
            range(self.config.n_violation_trials),
            desc="Normal patterns",
            disable=not self.config.verbose,
        )

        for _trial_idx in progress:
            # Reset spike buffer at start of each trial
            self._reset_spike_buffer()

            sequence, targets, _ = self.dataset.generate_sequence(include_violation=False)

            # Measure prediction error on normal sequence
            error_sum = 0.0
            for t in range(len(sequence) - 1):
                current_symbol = torch.argmax(sequence[t]).item()
                next_symbol = torch.argmax(targets[t]).item()

                input_spikes = encode_symbol_to_spikes(
                    current_symbol,
                    self.config.n_symbols,
                    self.config.spike_rate_active,
                    self.config.spike_rate_baseline,
                    self.device,
                    thalamus_size=thalamus.relay_size,
                )

                # Thalamus receives external sensory input via ascending pathways
                brain_input = {"thalamus": {"sensory": input_spikes}}
                brain_output = self.brain.forward(brain_input)

                if "hippocampus" in brain_output and brain_output["hippocampus"]:
                    prediction_spikes = list(brain_output["hippocampus"].values())[0]
                else:
                    for region_output in brain_output.values():
                        if region_output:
                            prediction_spikes = list(region_output.values())[0]
                            break

                if prediction_spikes.shape[0] > self.config.n_symbols:
                    prediction_spikes = prediction_spikes[: self.config.n_symbols]
                elif prediction_spikes.shape[0] < self.config.n_symbols:
                    padding = torch.zeros(
                        self.config.n_symbols - prediction_spikes.shape[0],
                        device=self.device,
                        dtype=prediction_spikes.dtype,
                    )
                    prediction_spikes = torch.cat([prediction_spikes, padding])

                spike_counts = self._accumulate_spikes(prediction_spikes)
                error = compute_prediction_error(
                    spike_counts,
                    next_symbol,
                    self.config.n_symbols,
                )
                error_sum += error

            normal_errors.append(error_sum / (len(sequence) - 1))

        # Test 2: Violation patterns
        violation_errors = []
        progress = tqdm(
            range(self.config.n_violation_trials),
            desc="Violation patterns",
            disable=not self.config.verbose,
        )

        for _trial_idx in progress:
            # Reset spike buffer at start of each trial
            self._reset_spike_buffer()

            sequence, targets, _ = self.dataset.generate_sequence(include_violation=True)

            error_sum = 0.0
            for t in range(len(sequence) - 1):
                current_symbol = torch.argmax(sequence[t]).item()
                next_symbol = torch.argmax(targets[t]).item()

                input_spikes = encode_symbol_to_spikes(
                    current_symbol,
                    self.config.n_symbols,
                    self.config.spike_rate_active,
                    self.config.spike_rate_baseline,
                    self.device,
                    thalamus_size=thalamus.relay_size,
                )

                # Thalamus receives external sensory input via ascending pathways
                brain_input = {"thalamus": {"sensory": input_spikes}}
                brain_output = self.brain.forward(brain_input)

                if "hippocampus" in brain_output and brain_output["hippocampus"]:
                    prediction_spikes = list(brain_output["hippocampus"].values())[0]
                else:
                    for region_output in brain_output.values():
                        if region_output:
                            prediction_spikes = list(region_output.values())[0]
                            break

                if prediction_spikes.shape[0] > self.config.n_symbols:
                    prediction_spikes = prediction_spikes[: self.config.n_symbols]
                elif prediction_spikes.shape[0] < self.config.n_symbols:
                    padding = torch.zeros(
                        self.config.n_symbols - prediction_spikes.shape[0],
                        device=self.device,
                        dtype=prediction_spikes.dtype,
                    )
                    prediction_spikes = torch.cat([prediction_spikes, padding])

                spike_counts = self._accumulate_spikes(prediction_spikes)
                error = compute_prediction_error(
                    spike_counts,
                    next_symbol,
                    self.config.n_symbols,
                )
                error_sum += error

            violation_errors.append(error_sum / (len(sequence) - 1))

        # Compare errors
        normal_error = np.mean(normal_errors)
        violation_error = np.mean(violation_errors)
        detection_score = (violation_error - normal_error) / (normal_error + 1e-6)

        self.metrics["violation_detection"].append(detection_score)

        if self.config.verbose:
            print("\n✓ Violation detection complete:")
            print(f"   Normal error: {normal_error:.3f}")
            print(f"   Violation error: {violation_error:.3f}")
            print(f"   Detection score: {detection_score:.2%}")
            print("   (Higher = better violation detection)")

    def save_results(self):
        """Save metrics and configuration."""
        # Save configuration
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Save brain if requested
        if self.config.save_brain:
            brain_path = self.output_dir / "brain.pt"
            torch.save(self.brain.state_dict(), brain_path)
            if self.config.verbose:
                print(f"\n✓ Brain saved to {brain_path}")

        if self.config.verbose:
            print(f"\n✓ Results saved to {self.output_dir}")

    def print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        train_acc = self.metrics["train_accuracy"][-1] if self.metrics["train_accuracy"] else 0.0
        test_acc = self.metrics["test_accuracy"][-1] if self.metrics["test_accuracy"] else 0.0
        violation_score = self.metrics["violation_detection"][-1] if self.metrics["violation_detection"] else 0.0

        random_baseline = 1.0 / self.config.n_symbols

        print(f"\n📊 Performance:")
        print(f"   Training accuracy:   {train_acc:.2%} (baseline: {random_baseline:.2%})")
        print(f"   Test accuracy:       {test_acc:.2%}")
        print(f"   Violation detection: {violation_score:.2%}")

        print(f"\n🎯 Goals:")
        print(f"   Basic (>50%):        {'✓' if test_acc > 0.50 else '✗'}")
        print(f"   Intermediate (>70%): {'✓' if test_acc > 0.70 else '✗'}")
        print(f"   Advanced (>85%):     {'✓' if test_acc > 0.85 else '✗'}")

        print(f"\n📁 Results saved to: {self.output_dir}")
        print("=" * 80)

    def _count_parameters(self) -> int:
        """Count total parameters in brain."""
        return sum(p.numel() for p in self.brain.parameters())

    def run(self):
        """Run complete experiment."""
        try:
            self.setup()
            self.train()
            self.test()
            self.test_violation_detection()
            self.save_results()
            self.print_summary()
        finally:
            # Keep plots open at the end
            if self.config.enable_plots and self.fig is not None:
                print("\n📊 Plots are displayed. Close the plot window to exit.")
                plt.ioff()  # Disable interactive mode
                plt.show()  # Block until user closes window


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test default brain on temporal sequence learning"
    )

    # Training parameters
    parser.add_argument(
        "--n-training-trials",
        type=int,
        default=1000,
        help="Number of training trials",
    )
    parser.add_argument(
        "--n-test-trials",
        type=int,
        default=100,
        help="Number of test trials",
    )
    parser.add_argument(
        "--n-symbols",
        type=int,
        default=5,
        help="Number of distinct symbols",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Length of each sequence",
    )

    # Architecture parameters
    parser.add_argument(
        "--cortex-size",
        type=int,
        default=200,
        help="Cortex size",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results/sequence_learning",
        help="Output directory",
    )
    parser.add_argument(
        "--save-brain",
        action="store_true",
        help="Save trained brain state",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        n_training_trials=args.n_training_trials,
        n_test_trials=args.n_test_trials,
        n_symbols=args.n_symbols,
        sequence_length=args.sequence_length,
        cortex_size=args.cortex_size,
        output_dir=args.output_dir,
        save_brain=args.save_brain,
        verbose=not args.quiet,
        device=args.device,
    )

    # Run experiment
    experiment = SequenceLearningExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
