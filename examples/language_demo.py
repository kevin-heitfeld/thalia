#!/usr/bin/env python
"""
Language Demo for THALIA - Spiking Neural Network Language Processing.

This demo shows:
1. Using the unified ThaliaConfig system
2. Loading text data and tokenizing
3. Processing through the LanguageBrainInterface
4. Training with local learning rules (no backprop!)
5. Simple next-character prediction

The goal is to validate the full pipeline works end-to-end,
even though the model is small and learning is limited.

Usage:
    python examples/language_demo.py

Author: Thalia Project
Date: December 2025
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import time

# New unified configuration
from thalia.config import (
    ThaliaConfig,
    GlobalConfig,
    BrainConfig,
    RegionSizes,
    LanguageConfig,
    TrainingConfig as UnifiedTrainingConfig,
)

# Core components
from thalia.core.brain import EventDrivenBrain
from thalia.language.model import LanguageBrainInterface
from thalia.memory.sequence import SequenceMemory
from thalia.training.data_pipeline import TextDataPipeline, DataConfig
from thalia.training.local_trainer import LocalTrainer


# Sample text for demo (small enough for quick testing)
SAMPLE_TEXT = """
The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be, that is the question.
All that glitters is not gold.
The only thing we have to fear is fear itself.
In the beginning was the Word.
I think therefore I am.
The unexamined life is not worth living.
Knowledge is power.
Time flies like an arrow.
"""


def main():
    print("=" * 60)
    print("THALIA Language Demo")
    print("Spiking Neural Network for Language Processing")
    print("=" * 60)
    print()

    # =========================================================================
    # UNIFIED CONFIGURATION
    # =========================================================================
    print("1. Creating unified configuration...")
    print("-" * 40)

    # Create ThaliaConfig - single source of truth for all settings
    n_neurons = 128  # Keep small for demo speed

    config = ThaliaConfig(
        # Global settings - inherited by all modules
        global_=GlobalConfig(
            device="cpu",
            vocab_size=100,  # Will be updated after loading data
            theta_frequency_hz=8.0,
        ),
        # Brain architecture
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=n_neurons,
                cortex_size=n_neurons,
                hippocampus_size=64,
                pfc_size=32,
                n_actions=2,  # Match/no-match (not per-character actions)
            ),
            encoding_timesteps=5,  # Short for speed
        ),
        # Language processing
        language=LanguageConfig(),
        # Training settings
        training=UnifiedTrainingConfig(
            n_epochs=2,  # Quick demo
            use_stdp=True,
            use_bcm=True,
            use_hebbian=True,
        ),
    )

    # Show configuration summary
    print(config.summary())
    print()

    print("2. Setting up data pipeline...")
    print("-" * 40)

    # Create data pipeline
    data_config = DataConfig(
        tokenizer_type="char",  # Character-level for simplicity
        context_length=32,       # Short context for demo
        batch_size=1,            # Single sample (hippocampus doesn't batch well)
    )

    data_pipeline = TextDataPipeline(data_config)
    data_pipeline.load_text(SAMPLE_TEXT)

    print(f"   Vocabulary size: {data_pipeline.vocab_size}")
    print(f"   Number of sequences: {data_pipeline.n_sequences}")
    print(f"   Context length: {data_config.context_length}")
    print()

    # Update config with actual vocab size
    config.global_.vocab_size = data_pipeline.vocab_size

    print("3. Creating brain and language interface...")
    print("-" * 40)

    # Create brain using unified config
    brain = EventDrivenBrain.from_thalia_config(config)

    # Create language interface using config's factory method
    lang_config = config.to_language_interface_config()
    lang_interface = LanguageBrainInterface(brain, lang_config)

    print(f"   Brain created with {n_neurons} neurons per region")
    print(f"   Regions: Cortex, Hippocampus, PFC, Striatum, Cerebellum")
    print()

    print("4. Creating sequence memory (hippocampal)...")
    print("-" * 40)

    # Create sequence memory using config's factory method
    memory_config = config.to_sequence_memory_config()
    sequence_memory = SequenceMemory(memory_config)

    print(f"   Hippocampus: DG={sequence_memory.hippocampus.dg_size}, "
          f"CA3={sequence_memory.hippocampus.ca3_size}, "
          f"CA1={sequence_memory.hippocampus.ca1_size}")
    print()

    print("5. Training with local learning rules...")
    print("-" * 40)

    # Create trainer using config's factory method
    train_config = config.to_training_config()
    trainer = LocalTrainer(train_config)

    # Training callback for progress
    def progress_callback(epoch, step, metrics):
        if step % 10 == 0:
            print(f"   Step {step}: accuracy={metrics.prediction_accuracy:.4f}, "
                  f"spike_rate={metrics.spike_rate:.2f}")

    # Train!
    start_time = time.time()
    final_metrics = trainer.train(
        model=lang_interface,
        data_pipeline=data_pipeline,
        memory=sequence_memory,
        progress_callback=progress_callback,
    )
    train_time = time.time() - start_time

    print()
    print(f"   Training completed in {train_time:.2f}s")
    print(f"   Final metrics: {final_metrics.to_dict()}")
    print()

    print("6. Testing next-character prediction...")
    print("-" * 40)

    # Test prompts
    test_prompts = [
        "The quick brown ",
        "To be or ",
        "Knowledge is ",
    ]

    for prompt in test_prompts:
        print(f"\n   Prompt: '{prompt}'")

        # Encode prompt
        prompt_ids = data_pipeline.encode(prompt)
        if len(prompt_ids) > data_config.context_length:
            prompt_ids = prompt_ids[-data_config.context_length:]
        prompt_tensor = prompt_ids.unsqueeze(0)  # Add batch dim

        # Process through brain
        result = lang_interface.process_tokens(prompt_tensor)

        # Get spike activity
        if result["results"]:
            last_result = result["results"][-1]
            cortex_activity = last_result.get("cortex_activity", torch.zeros(n_neurons))

            # Simple "prediction" based on activity
            # In a real model, this would decode to vocabulary
            active_neurons = (cortex_activity > 0).sum().item()
            print(f"   → Cortex active neurons: {active_neurons}/{n_neurons}")
            print(f"   → Events processed: {last_result.get('events_processed', 0)}")

        # Use sequence memory for prediction
        memory_result = sequence_memory.predict_next(prompt_tensor)
        predicted_pattern = memory_result["predicted_pattern"]
        active_pred = (predicted_pattern > 0).sum().item()
        print(f"   → Memory prediction pattern: {active_pred} active neurons")

    print()
    print("=" * 60)
    print("Demo Complete!")
    print()
    print("Summary:")
    print(f"  - Processed {final_metrics.sequences_processed} sequences")
    print(f"  - Training steps: {final_metrics.steps}")
    print(f"  - Final spike rate: {final_metrics.spike_rate:.2f}")
    print()
    print("Key Points:")
    print("  ✓ No backpropagation used - all learning is local")
    print("  ✓ STDP, BCM, and Hebbian rules for synaptic plasticity")
    print("  ✓ Hippocampus stores sequence associations")
    print("  ✓ Theta-phase encoding for temporal order")
    print("  ✓ Unified ThaliaConfig eliminates parameter duplication")
    print("=" * 60)


if __name__ == "__main__":
    main()
