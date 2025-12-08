"""
Example: Using Task-Specific Datasets for Curriculum Training

Demonstrates how to use each dataset for curriculum training across stages:
- Stage 0: Temporal sequences
- Stage 1: CIFAR-10 with spike encoding
- Stage 2: Grammar learning
- Stage 3: Reading comprehension
"""

import torch
from thalia.datasets import (
    # Stage 0
    create_stage0_temporal_dataset,
    TemporalSequenceDataset,
    PatternType,
    # Stage 1
    create_stage1_cifar_datasets,
    CIFARForThalia,
    # Stage 2
    create_stage2_grammar_dataset,
    GrammarDataset,
    GrammarRule,
    # Stage 3
    create_stage3_reading_dataset,
    ReadingDataset,
    ReadingTask,
)


def demo_temporal_sequences():
    """Stage 0: Temporal sequence learning."""
    print("=" * 60)
    print("Stage 0: Temporal Sequence Dataset")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = create_stage0_temporal_dataset(device=device)
    
    # Generate single sequence
    sequence, targets, pattern_type = dataset.generate_sequence()
    print(f"\nSingle sequence:")
    print(f"  Pattern type: {pattern_type.value}")
    print(f"  Sequence shape: {sequence.shape}")  # (length, n_symbols)
    print(f"  Targets shape: {targets.shape}")
    
    # Generate batch
    sequences, targets_batch, pattern_types = dataset.generate_batch(
        batch_size=32,
        balance_patterns=True,
    )
    print(f"\nBatch:")
    print(f"  Sequences shape: {sequences.shape}")  # (32, length, n_symbols)
    print(f"  Pattern distribution:")
    for pattern_type in [PatternType.ABC, PatternType.ABA, PatternType.AAB]:
        count = sum(1 for pt in pattern_types if pt == pattern_type)
        print(f"    {pattern_type.value}: {count}")
    
    # Compute prediction error
    mock_predictions = torch.rand_like(targets)
    mock_predictions = torch.softmax(mock_predictions, dim=-1)
    error = dataset.compute_prediction_error(mock_predictions, targets)
    print(f"\nMock prediction error: {error:.4f}")


def demo_cifar10():
    """Stage 1: CIFAR-10 with spike encoding."""
    print("\n" + "=" * 60)
    print("Stage 1: CIFAR-10 Spike Encoding Dataset")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create train/test datasets
    train_dataset, test_dataset = create_stage1_cifar_datasets(
        device=device,
        encoding="rate",  # "rate", "temporal", or "phase"
        n_timesteps=100,
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Get single image
    spikes, label = train_dataset[0]
    print(f"\nSingle image:")
    print(f"  Spikes shape: {spikes.shape}")  # (n_timesteps, C, H, W)
    print(f"  Label: {label} ({train_dataset.classes[label]})")
    
    # Get batch
    indices = list(range(16))
    spikes_batch, labels_batch = train_dataset.get_batch(indices)
    print(f"\nBatch:")
    print(f"  Spikes shape: {spikes_batch.shape}")  # (16, n_timesteps, C, H, W)
    print(f"  Labels shape: {labels_batch.shape}")
    
    # Analyze encoding statistics
    stats = train_dataset.analyze_encoding_statistics(n_samples=100)
    print(f"\nEncoding statistics:")
    print(f"  Mean firing rate: {stats['mean_firing_rate']:.4f}")
    print(f"  Sparsity: {stats['sparsity']:.4f}")
    
    # Compute accuracy (mock predictions)
    mock_predictions = torch.randn(16, 10, device=device)  # 10 classes
    accuracy = train_dataset.compute_accuracy(mock_predictions, labels_batch)
    print(f"\nMock accuracy: {accuracy:.4f}")


def demo_grammar():
    """Stage 2: Grammar learning."""
    print("\n" + "=" * 60)
    print("Stage 2: Grammar Dataset")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = create_stage2_grammar_dataset(
        device=device,
        multilingual=False,  # Set True for SOV word order
    )
    
    print(f"\nVocabulary size: {dataset.vocab.vocab_size}")
    
    # Generate single phrase
    phrase, is_grammatical, rule = dataset.generate_phrase()
    phrase_words = dataset.vocab.decode(phrase)
    print(f"\nSingle phrase:")
    print(f"  Words: {' '.join(phrase_words)}")
    print(f"  Grammatical: {is_grammatical}")
    print(f"  Rule tested: {rule.value}")
    
    # Generate batch
    phrases, labels, rules = dataset.generate_batch(
        batch_size=32,
        balance_rules=True,
        balance_violations=True,
    )
    print(f"\nBatch:")
    print(f"  Phrases shape: {phrases.shape}")  # (32, max_length)
    print(f"  Labels shape: {labels.shape}")  # (32,)
    print(f"  Rule distribution:")
    for rule in [GrammarRule.SUBJECT_VERB_AGREEMENT, GrammarRule.NOUN_ADJECTIVE]:
        count = sum(1 for r in rules if r == rule)
        print(f"    {rule.value}: {count}")
    
    # Compute accuracy (mock predictions)
    mock_predictions = torch.rand(32, device=device)
    accuracy = dataset.compute_accuracy(mock_predictions, labels)
    print(f"\nMock accuracy: {accuracy:.4f}")


def demo_reading():
    """Stage 3: Reading comprehension."""
    print("\n" + "=" * 60)
    print("Stage 3: Reading Comprehension Dataset")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = create_stage3_reading_dataset(device=device)
    
    print(f"\nVocabulary:")
    print(f"  Words: {dataset.vocab.vocab_size}")
    print(f"  Phonemes: {dataset.vocab.n_phonemes}")
    
    # Phoneme → word task
    print("\n--- Task 1: Phoneme → Word ---")
    task_data, label, task_type = dataset.generate_task(ReadingTask.PHONEME_TO_WORD)
    print(f"  Task type: {task_type.value}")
    print(f"  Input shape: {task_data['phonemes'].shape}")
    print(f"  Target word: {dataset.vocab.idx2word[label.item()]}")
    
    # Word → meaning task
    print("\n--- Task 2: Word → Meaning ---")
    task_data, label, task_type = dataset.generate_task(ReadingTask.WORD_TO_MEANING)
    print(f"  Task type: {task_type.value}")
    print(f"  Input word: {dataset.vocab.idx2word[task_data['word'].item()]}")
    print(f"  Semantic features: {label.tolist()}")
    print(f"    [is_animate, is_object, is_action, size]")
    
    # Sentence completion task
    print("\n--- Task 3: Sentence Completion ---")
    task_data, label, task_type = dataset.generate_task(ReadingTask.SENTENCE_COMPLETION)
    print(f"  Task type: {task_type.value}")
    sentence_words = [
        dataset.vocab.idx2word.get(idx.item(), '<PAD>')
        for idx in task_data['sentence']
        if idx.item() != dataset.vocab.pad_idx
    ]
    print(f"  Sentence: {' '.join(sentence_words)}")
    print(f"  Missing position: {task_data['missing_position'].item()}")
    print(f"  Answer: {dataset.vocab.idx2word[label.item()]}")
    
    # Simple QA task
    print("\n--- Task 4: Simple Question Answering ---")
    task_data, label, task_type = dataset.generate_task(ReadingTask.SIMPLE_QA)
    print(f"  Task type: {task_type.value}")
    sentence_words = [
        dataset.vocab.idx2word.get(idx.item(), '<PAD>')
        for idx in task_data['sentence']
        if idx.item() != dataset.vocab.pad_idx
    ]
    question_words = [
        dataset.vocab.idx2word.get(idx.item(), '<PAD>')
        for idx in task_data['question']
        if idx.item() != dataset.vocab.pad_idx
    ]
    print(f"  Sentence: {' '.join(sentence_words)}")
    print(f"  Question: {' '.join(question_words)}")
    print(f"  Answer: {dataset.vocab.idx2word.get(label.item(), '<UNK>')}")


def demo_integration_with_curriculum():
    """Example: Using datasets in curriculum training loop."""
    print("\n" + "=" * 60)
    print("Integration with Curriculum Training")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create all datasets
    temporal_dataset = create_stage0_temporal_dataset(device=device)
    cifar_train, cifar_test = create_stage1_cifar_datasets(device=device)
    grammar_dataset = create_stage2_grammar_dataset(device=device)
    reading_dataset = create_stage3_reading_dataset(device=device)
    
    print("\nDatasets created for all stages:")
    print(f"  ✅ Stage 0: Temporal sequences")
    print(f"  ✅ Stage 1: CIFAR-10 ({len(cifar_train)} train, {len(cifar_test)} test)")
    print(f"  ✅ Stage 2: Grammar ({grammar_dataset.vocab.vocab_size} words)")
    print(f"  ✅ Stage 3: Reading ({reading_dataset.vocab.vocab_size} words)")
    
    print("\nExample curriculum training loop structure:")
    print("""
    # Stage 0: Learn temporal patterns
    for step in range(60000):
        seq, targets, pattern = temporal_dataset.generate_sequence()
        brain_output = brain.forward(seq)
        loss = compute_loss(brain_output, targets)
        # ... learning update
    
    # Stage 1: Learn visual categories
    for step in range(80000):
        spikes, label = cifar_train[step % len(cifar_train)]
        brain_output = brain.forward(spikes)
        # ... learning update
    
    # Stage 2: Learn grammar rules
    for step in range(100000):
        phrase, is_gram, rule = grammar_dataset.generate_phrase()
        brain_output = brain.judge_grammaticality(phrase)
        # ... learning update
    
    # Stage 3: Learn reading comprehension
    for step in range(120000):
        task_data, label, task_type = reading_dataset.generate_task()
        brain_output = brain.process_reading_task(task_data, task_type)
        # ... learning update
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Task-Specific Datasets Demo")
    print("=" * 60)
    print("\nDemonstrating datasets for curriculum training stages 0-3\n")
    
    try:
        # Demo each dataset
        demo_temporal_sequences()
        demo_cifar10()
        demo_grammar()
        demo_reading()
        demo_integration_with_curriculum()
        
        print("\n" + "=" * 60)
        print("✅ All datasets working correctly!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Use these datasets in curriculum_training_example.py")
        print("2. Integrate with CurriculumTrainer.train_stage()")
        print("3. Implement stage-specific evaluation functions")
        print("4. Run full curriculum training pipeline")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
