#!/usr/bin/env python
"""
Experiment 01: Language Learning with Local Plasticity Rules.

This experiment investigates whether spiking neural networks can learn
next-token prediction using only local learning rules (no backpropagation).

Research Questions:
==================
1. Can STDP/BCM/Hebbian rules learn token associations?
2. How does accuracy scale with network size?
3. What is the effect of different learning rule combinations?
4. How does hippocampal sequence memory contribute to prediction?

Experimental Design:
===================
- Independent Variables:
  - Network size (neurons per region): 64, 128, 256, 512
  - Learning rules: STDP only, BCM only, All rules
  - Epochs: 1, 2, 5, 10

- Dependent Variables:
  - Token prediction accuracy
  - Average spike rate
  - Training time
  - Weight change magnitude

- Control:
  - Fixed random seed for reproducibility
  - Character-level tokenization
  - Same training corpus

Output:
======
- JSON results in experiments/results/exp01/
- Console summary with key findings
- Charts showing learning dynamics

Usage:
    python experiments/scripts/exp01_language_learning.py [--quick]

    --quick: Run minimal configuration for testing (default: full sweep)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch

from thalia.config import (
    ThaliaConfig,
    GlobalConfig,
    BrainConfig,
    RegionSizes,
    LanguageConfig,
    TrainingConfig,
    CortexType,
)
from thalia.core.brain import EventDrivenBrain
from thalia.core.diagnostics import DiagnosticLevel
from thalia.language.model import LanguageBrainInterface
from thalia.memory.sequence import SequenceMemory
from thalia.training.data_pipeline import TextDataPipeline, DataConfig
from thalia.training.local_trainer import LocalTrainer


# =============================================================================
# BRAIN DIAGNOSTICS TRACKER
# =============================================================================

@dataclass
class BrainDiagnosticsSnapshot:
    """Snapshot of brain state at a point in time."""
    step: int
    timestamp: float

    # Cortex metrics
    cortex_l4_spikes: float = 0.0
    cortex_l23_spikes: float = 0.0
    cortex_l5_spikes: float = 0.0
    cortex_weight_mean: float = 0.0
    cortex_weight_std: float = 0.0
    cortex_weight_change: float = 0.0  # Change from previous snapshot

    # Hippocampus metrics
    hippo_dg_spikes: float = 0.0
    hippo_ca3_spikes: float = 0.0
    hippo_ca1_spikes: float = 0.0

    # PFC metrics
    pfc_spikes: float = 0.0
    pfc_wm_active: float = 0.0

    # VTA Dopamine System (centralized in brain)
    dopamine_global: float = 0.0   # Combined signal to regions
    dopamine_tonic: float = 0.0    # Slow baseline (intrinsic prediction quality)
    dopamine_phasic: float = 0.0   # Fast bursts/dips (external rewards)

    # Prediction metrics (if using predictive cortex)
    prediction_error: float = 0.0
    free_energy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BrainDiagnosticsTracker:
    """Track brain diagnostics over training."""

    def __init__(self, brain: EventDrivenBrain):
        self.brain = brain
        self.snapshots: List[BrainDiagnosticsSnapshot] = []
        self._prev_cortex_weights: Optional[torch.Tensor] = None
        self._start_time = time.time()

    def snapshot(self, step: int) -> BrainDiagnosticsSnapshot:
        """Take a diagnostic snapshot of the brain state."""
        snap = BrainDiagnosticsSnapshot(
            step=step,
            timestamp=time.time() - self._start_time,
        )

        # Cortex diagnostics - now both cortex types use the same format
        cortex = self.brain.cortex
        cortex_diag = cortex.get_diagnostics()

        # Both cortex types now use spike_diagnostics with "l4_active_count" format
        snap.cortex_l4_spikes = cortex_diag.get("l4_active_count", 0.0)
        snap.cortex_l23_spikes = cortex_diag.get("l23_active_count", 0.0)
        snap.cortex_l5_spikes = cortex_diag.get("l5_active_count", 0.0)

        # Weight statistics - check different weight sources
        # LayeredCortex: l4_l23_weight_mean, PredictiveCortex: pred_weight_mean
        snap.cortex_weight_mean = cortex_diag.get(
            "l4_l23_weight_mean",
            cortex_diag.get("pred_weight_mean", 0.0)
        )
        snap.cortex_weight_std = cortex_diag.get(
            "l4_l23_weight_std",
            cortex_diag.get("pred_weight_std", 0.0)
        )

        # Compute weight change from previous snapshot
        weights = None
        if hasattr(cortex, 'w_l4_l23'):
            weights = cortex.w_l4_l23.data
        elif hasattr(cortex, 'prediction_layer') and cortex.prediction_layer is not None:
            if hasattr(cortex.prediction_layer, 'W_pred'):
                weights = cortex.prediction_layer.W_pred.data

        if weights is not None:
            if self._prev_cortex_weights is not None:
                snap.cortex_weight_change = (weights - self._prev_cortex_weights).abs().mean().item()
            self._prev_cortex_weights = weights.clone()

        # Hippocampus diagnostics - nested structure
        hippo = self.brain.hippocampus
        hippo_diag = hippo.get_diagnostics()

        # Access nested dicts from hippocampus diagnostics
        dg_diag = hippo_diag.get("dg", {})
        ca3_diag = hippo_diag.get("ca3", {})
        ca1_diag = hippo_diag.get("ca1", {})

        snap.hippo_dg_spikes = dg_diag.get("active_count", 0.0)
        snap.hippo_ca3_spikes = ca3_diag.get("active_count", 0.0)
        snap.hippo_ca1_spikes = ca1_diag.get("active_count", 0.0)

        # PFC diagnostics
        pfc = self.brain.pfc
        if hasattr(pfc, 'state') and pfc.state is not None:
            if pfc.state.spikes is not None:
                snap.pfc_spikes = pfc.state.spikes.sum().item()
            if pfc.state.working_memory is not None:
                snap.pfc_wm_active = (pfc.state.working_memory > 0.1).sum().item()

        # VTA Dopamine System (centralized in brain)
        snap.dopamine_global = self.brain._global_dopamine
        snap.dopamine_tonic = self.brain._tonic_dopamine
        snap.dopamine_phasic = self.brain._phasic_dopamine

        # Predictive coding metrics (if using predictive cortex)
        if hasattr(cortex, '_total_free_energy'):
            snap.free_energy = cortex._total_free_energy
        if hasattr(cortex, 'prediction_layer') and cortex.prediction_layer is not None:
            pred_diag = cortex.prediction_layer.get_diagnostics()
            snap.prediction_error = pred_diag.get("mean_error", 0.0)

        self.snapshots.append(snap)
        return snap

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full diagnostic history as list of dicts."""
        return [s.to_dict() for s in self.snapshots]

    def print_summary(self):
        """Print summary of brain diagnostics."""
        if not self.snapshots:
            print("No diagnostics collected.")
            return

        print("\n" + "="*70)
        print("BRAIN DIAGNOSTICS SUMMARY")
        print("="*70)

        # Get first and last snapshots
        first = self.snapshots[0]
        last = self.snapshots[-1]

        print(f"\nSnapshots collected: {len(self.snapshots)}")
        print(f"Training time: {last.timestamp:.1f}s")

        # Cortex
        print("\n--- CORTEX ---")
        print(f"  L4 spikes:  first={first.cortex_l4_spikes:.1f}, last={last.cortex_l4_spikes:.1f}")
        print(f"  L2/3 spikes: first={first.cortex_l23_spikes:.1f}, last={last.cortex_l23_spikes:.1f}")
        print(f"  L5 spikes:  first={first.cortex_l5_spikes:.1f}, last={last.cortex_l5_spikes:.1f}")
        print(f"  Weight mean: first={first.cortex_weight_mean:.4f}, last={last.cortex_weight_mean:.4f}")
        print(f"  Weight std:  first={first.cortex_weight_std:.4f}, last={last.cortex_weight_std:.4f}")

        # Calculate average weight change
        avg_weight_change = sum(s.cortex_weight_change for s in self.snapshots) / len(self.snapshots)
        print(f"  Avg weight change per step: {avg_weight_change:.6f}")

        # Hippocampus
        print("\n--- HIPPOCAMPUS ---")
        print(f"  DG spikes:  first={first.hippo_dg_spikes:.1f}, last={last.hippo_dg_spikes:.1f}")
        print(f"  CA3 spikes: first={first.hippo_ca3_spikes:.1f}, last={last.hippo_ca3_spikes:.1f}")
        print(f"  CA1 spikes: first={first.hippo_ca1_spikes:.1f}, last={last.hippo_ca1_spikes:.1f}")

        # PFC
        print("\n--- PFC ---")
        print(f"  Spikes: first={first.pfc_spikes:.1f}, last={last.pfc_spikes:.1f}")
        print(f"  WM active: first={first.pfc_wm_active:.1f}, last={last.pfc_wm_active:.1f}")

        # VTA Dopamine System
        avg_global = sum(s.dopamine_global for s in self.snapshots) / len(self.snapshots)
        avg_tonic = sum(s.dopamine_tonic for s in self.snapshots) / len(self.snapshots)
        avg_phasic = sum(s.dopamine_phasic for s in self.snapshots) / len(self.snapshots)
        print(f"\n--- VTA DOPAMINE ---")
        print(f"  Global (combined): first={first.dopamine_global:.4f}, last={last.dopamine_global:.4f}, avg={avg_global:.4f}")
        print(f"  Tonic (intrinsic): first={first.dopamine_tonic:.4f}, last={last.dopamine_tonic:.4f}, avg={avg_tonic:.4f}")
        print(f"  Phasic (reward):   first={first.dopamine_phasic:.4f}, last={last.dopamine_phasic:.4f}, avg={avg_phasic:.4f}")

        # Prediction error (if available)
        if any(s.prediction_error > 0 for s in self.snapshots):
            avg_pred_error = sum(s.prediction_error for s in self.snapshots) / len(self.snapshots)
            print(f"\n--- PREDICTION ---")
            print(f"  Avg prediction error: {avg_pred_error:.4f}")
            print(f"  Final free energy: {last.free_energy:.4f}")

        print("="*70)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Training corpus - classic sentences with varied structure
TRAINING_CORPUS = """
# In the beginning was the Word.
# Knowledge is power.
# To be or not to be, that is the question.
# I think therefore I am.
# A journey of a thousand miles begins with a single step.
"""

# TRAINING_CORPUS = """
# In the beginning was the Word.
# Knowledge is power.
# To be or not to be, that is the question.
# I think therefore I am.
# A journey of a thousand miles begins with a single step.
# The only thing we have to fear is fear itself.
# The unexamined life is not worth living.
# Actions speak louder than words.
# Time flies like an arrow.
# The quick brown fox jumps over the lazy dog.
# All that glitters is not gold.
# Beauty is in the eye of the beholder.
# Early to bed and early to rise makes a man healthy wealthy and wise.
# Fortune favors the bold.
# Hope springs eternal in the human breast.
# """

# Test prompts for evaluation
TEST_PROMPTS = [
    "The quick brown ",
    "To be or ",
    "Knowledge is ",
    "I think ",
    "All that ",
]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    n_neurons: int
    n_epochs: int
    use_stdp: bool
    use_bcm: bool
    use_hebbian: bool
    cortex_type: CortexType = CortexType.LAYERED
    seed: int = 42
    context_length: int = 32


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: Dict[str, Any]
    final_accuracy: float
    final_spike_rate: float
    training_time_s: float
    epochs_completed: int
    sequences_processed: int
    accuracy_history: List[float]
    spike_rate_history: List[float]
    test_results: Dict[str, Dict[str, Any]]
    timestamp: str
    # Brain diagnostics
    brain_diagnostics: Optional[List[Dict[str, Any]]] = None
    weight_change_history: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        # Clean up None values
        if d.get("brain_diagnostics") is None:
            d["brain_diagnostics"] = []
        if d.get("weight_change_history") is None:
            d["weight_change_history"] = []
        return d


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class LanguageLearningExperiment:
    """Runner for language learning experiments."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []

    def run_single(self, exp_config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment configuration."""
        print(f"\n{'='*60}")
        print(f"Running: {exp_config.name}")
        print(f"  Neurons: {exp_config.n_neurons}")
        print(f"  Epochs: {exp_config.n_epochs}")
        print(f"  Cortex: {exp_config.cortex_type.value}")
        print(f"  Rules: STDP={exp_config.use_stdp}, BCM={exp_config.use_bcm}, Hebbian={exp_config.use_hebbian}")
        print(f"{'='*60}")

        # Set random seed
        torch.manual_seed(exp_config.seed)

        # Create configuration
        thalia_config = ThaliaConfig(
            global_=GlobalConfig(
                device="cpu",
                vocab_size=100,
                theta_frequency_hz=8.0,
            ),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=exp_config.n_neurons,
                    cortex_size=exp_config.n_neurons,
                    hippocampus_size=exp_config.n_neurons // 2,
                    pfc_size=exp_config.n_neurons // 4,
                    n_actions=2,
                ),
                cortex_type=exp_config.cortex_type,
                encoding_timesteps=5,
            ),
            language=LanguageConfig(),
            training=TrainingConfig(
                n_epochs=exp_config.n_epochs,
                use_stdp=exp_config.use_stdp,
                use_bcm=exp_config.use_bcm,
                use_hebbian=exp_config.use_hebbian,
            ),
        )

        # Create data pipeline
        data_config = DataConfig(
            tokenizer_type="char",
            context_length=exp_config.context_length,
            batch_size=1,
        )
        data_pipeline = TextDataPipeline(data_config)
        data_pipeline.load_text(TRAINING_CORPUS)

        # Update vocab size
        thalia_config.global_.vocab_size = data_pipeline.vocab_size

        # Create model components
        brain = EventDrivenBrain.from_thalia_config(thalia_config)
        lang_interface = LanguageBrainInterface(
            brain,
            thalia_config.to_language_interface_config()
        )
        sequence_memory = SequenceMemory(thalia_config.to_sequence_memory_config())

        # Create trainer
        trainer = LocalTrainer(thalia_config.to_training_config())

        # Create brain diagnostics tracker
        diagnostics_tracker = BrainDiagnosticsTracker(brain)

        # Track accuracy history
        accuracy_history: List[float] = []
        spike_rate_history: List[float] = []

        def progress_callback(epoch: int, step: int, metrics):
            if step % 10 == 0:
                accuracy_history.append(metrics.prediction_accuracy)
                spike_rate_history.append(metrics.spike_rate)
                print(f"  Step {step}: acc={metrics.prediction_accuracy:.4f}, "
                      f"spikes={metrics.spike_rate:.1f}")
                # Collect brain diagnostics every 10 steps
                diagnostics_tracker.snapshot(step)

        # Train
        start_time = time.time()
        final_metrics = trainer.train(
            model=lang_interface,
            data_pipeline=data_pipeline,
            memory=sequence_memory,
            progress_callback=progress_callback,
        )
        training_time = time.time() - start_time

        # Print brain diagnostics summary
        diagnostics_tracker.print_summary()

        # Test on prompts
        test_results = self._evaluate_prompts(
            lang_interface,
            sequence_memory,
            data_pipeline,
            exp_config.context_length,
        )

        # Create result - convert enum to string for JSON serialization
        config_dict = asdict(exp_config)
        config_dict["cortex_type"] = exp_config.cortex_type.value

        result = ExperimentResult(
            config=config_dict,
            final_accuracy=final_metrics.prediction_accuracy,
            final_spike_rate=final_metrics.spike_rate,
            training_time_s=training_time,
            epochs_completed=exp_config.n_epochs,
            sequences_processed=final_metrics.sequences_processed,
            accuracy_history=accuracy_history,
            spike_rate_history=spike_rate_history,
            test_results=test_results,
            timestamp=datetime.now().isoformat(),
            brain_diagnostics=diagnostics_tracker.get_history(),
        )

        self.results.append(result)
        return result

    def _evaluate_prompts(
        self,
        lang_interface: LanguageBrainInterface,
        memory: SequenceMemory,
        data_pipeline: TextDataPipeline,
        context_length: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate model on test prompts."""
        results = {}

        for prompt in TEST_PROMPTS:
            prompt_ids = data_pipeline.encode(prompt)
            if len(prompt_ids) > context_length:
                prompt_ids = prompt_ids[-context_length:]
            prompt_tensor = prompt_ids.unsqueeze(0)

            # Process through brain
            brain_result = lang_interface.process_tokens(prompt_tensor)

            # Get activity stats
            if brain_result["results"]:
                last_result = brain_result["results"][-1]
                cortex_activity = last_result.get("cortex_activity", torch.zeros(1))
                active_neurons = (cortex_activity > 0).sum().item()
                events_processed = last_result.get("events_processed", 0)
            else:
                active_neurons = 0
                events_processed = 0

            # Get memory prediction
            memory_result = memory.predict_next(prompt_tensor)
            predicted_pattern = memory_result["predicted_pattern"]
            memory_active = (predicted_pattern > 0).sum().item()

            results[prompt.strip()] = {
                "cortex_active_neurons": active_neurons,
                "events_processed": events_processed,
                "memory_active_neurons": memory_active,
            }

        return results

    def run_sweep(self, configs: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run a sweep of experiments."""
        print(f"\n{'#'*60}")
        print(f"# EXPERIMENT SWEEP: {len(configs)} configurations")
        print(f"{'#'*60}")

        for config in configs:
            self.run_single(config)

        return self.results

    def save_results(self, filename: str = "results.json"):
        """Save all results to JSON."""
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                indent=2,
            )

        print(f"\nResults saved to: {output_path}")

    def print_summary(self):
        """Print summary of all experiments."""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")

        print(f"\n{'Name':<35} {'Acc':>8} {'Spikes':>10} {'Time':>8}")
        print("-" * 65)

        for r in self.results:
            name = r.config["name"][:35]
            print(f"{name:<35} {r.final_accuracy:>8.4f} {r.final_spike_rate:>10.1f} {r.training_time_s:>7.1f}s")

        # Best result
        if self.results:
            best = max(self.results, key=lambda r: r.final_accuracy)
            print(f"\nBest configuration: {best.config['name']}")
            print(f"  Accuracy: {best.final_accuracy:.4f}")
            print(f"  Spike rate: {best.final_spike_rate:.1f}")


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

def get_quick_configs() -> List[ExperimentConfig]:
    """Minimal configs for quick testing."""
    return [
        ExperimentConfig(
            name="quick_predictive",
            n_neurons=64,
            n_epochs=1,
            use_stdp=True,
            use_bcm=True,
            use_hebbian=True,
            cortex_type=CortexType.PREDICTIVE,
        ),
    ]


def get_full_configs() -> List[ExperimentConfig]:
    """Full sweep of configurations."""
    configs = []

    # ==========================================================================
    # CORTEX TYPE COMPARISON (key experiment!)
    # ==========================================================================
    # Compare layered vs predictive cortex with same settings
    for cortex_type in [CortexType.LAYERED, CortexType.PREDICTIVE]:
        configs.append(ExperimentConfig(
            name=f"cortex_{cortex_type.value}_128",
            n_neurons=128,
            n_epochs=2,
            use_stdp=True,
            use_bcm=True,
            use_hebbian=True,
            cortex_type=cortex_type,
        ))

    # ==========================================================================
    # SIZE SWEEP (with predictive cortex)
    # ==========================================================================
    for n_neurons in [64, 128, 256]:
        configs.append(ExperimentConfig(
            name=f"predictive_size_{n_neurons}",
            n_neurons=n_neurons,
            n_epochs=2,
            use_stdp=True,
            use_bcm=True,
            use_hebbian=True,
            cortex_type=CortexType.PREDICTIVE,
        ))

    # ==========================================================================
    # LEARNING RULE ABLATION (128 neurons, predictive cortex)
    # ==========================================================================
    configs.extend([
        ExperimentConfig(
            name="predictive_stdp_only",
            n_neurons=128,
            n_epochs=2,
            use_stdp=True,
            use_bcm=False,
            use_hebbian=False,
            cortex_type=CortexType.PREDICTIVE,
        ),
        ExperimentConfig(
            name="predictive_no_global_rules",
            n_neurons=128,
            n_epochs=2,
            use_stdp=False,
            use_bcm=False,
            use_hebbian=False,
            cortex_type=CortexType.PREDICTIVE,
        ),
    ])

    # ==========================================================================
    # EPOCH SWEEP (128 neurons, predictive cortex)
    # ==========================================================================
    for n_epochs in [1, 2, 5]:
        configs.append(ExperimentConfig(
            name=f"predictive_epochs_{n_epochs}",
            n_neurons=128,
            n_epochs=n_epochs,
            use_stdp=True,
            use_bcm=True,
            use_hebbian=True,
            cortex_type=CortexType.PREDICTIVE,
        ))

    return configs


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Language Learning Experiment with Local Plasticity"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run minimal configuration for testing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "exp01",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Select configurations
    if args.quick:
        configs = get_quick_configs()
        print("Running QUICK mode (minimal config for testing)")
    else:
        configs = get_full_configs()
        print(f"Running FULL sweep ({len(configs)} configurations)")

    # Run experiments
    experiment = LanguageLearningExperiment(args.output_dir)
    experiment.run_sweep(configs)

    # Save and summarize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment.save_results(f"results_{timestamp}.json")
    experiment.print_summary()

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("  1. Current accuracy is 0% - model does not learn token prediction")
    print("  2. Local rules (STDP/BCM/Hebbian) increase spike activity but not accuracy")
    print("  3. The connection from brain activity to token decoding needs work")
    print("\nNext Steps:")
    print("  - Implement supervised error signal propagation")
    print("  - Add predictive coding between layers")
    print("  - Train decoder weights with available gradients")
    print("="*60)


if __name__ == "__main__":
    main()
