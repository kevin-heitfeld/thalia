# Task-Specific Datasets Quick Reference

**Created**: December 8, 2025  
**Status**: All Stage 0-3 datasets complete ✅

---

## Overview

Task-specific datasets for curriculum training stages 0-3, each biologically motivated and stage-appropriate.

| Stage | Dataset | Purpose | File |
|-------|---------|---------|------|
| 0 | Temporal Sequences | A-B-C pattern learning | `temporal_sequences.py` |
| 0 | Phonology | Phoneme discrimination | `phonology.py` (existing) |
| 1 | CIFAR-10 | Visual object recognition | `cifar_wrapper.py` |
| 2 | Grammar | Grammar rule learning | `grammar.py` |
| 3 | Reading | Reading comprehension | `reading.py` |

---

## Stage 0: Temporal Sequences

**File**: `src/thalia/datasets/temporal_sequences.py`

### Quick Start
```python
from thalia.datasets import create_stage0_temporal_dataset

dataset = create_stage0_temporal_dataset(device=device)
sequence, targets, pattern_type = dataset.generate_sequence()
# sequence: (length, n_symbols) one-hot encoded
# targets: (length, n_symbols) next-symbol predictions
# pattern_type: PatternType.ABC, .ABA, or .AAB
```

### Pattern Types
- **ABC**: Linear sequences (A→B→C)
- **ABA**: Repetition with gap (A→B→A)
- **AAB**: Immediate repetition (A→A→B)
- **ABAC**: Hierarchical patterns
- **RANDOM**: No structure (control)

### Configuration
```python
from thalia.datasets import SequenceConfig, PatternType

config = SequenceConfig(
    n_symbols=5,              # Number of distinct symbols
    sequence_length=10,       # Length per sequence
    pattern_types=[PatternType.ABC, PatternType.ABA],
    violation_probability=0.1, # Unexpected symbols
    encoding="one_hot",       # or "distributed"
    device=device,
)
dataset = TemporalSequenceDataset(config)
```

### Batch Generation
```python
sequences, targets, patterns = dataset.generate_batch(
    batch_size=32,
    balance_patterns=True,  # Equal distribution
)
# sequences: (32, length, n_symbols)
```

### Evaluation
```python
error = dataset.compute_prediction_error(predictions, targets)
results = dataset.analyze_pattern_learning(brain, n_test_sequences=100)
# results: {pattern_type → accuracy}
```

---

## Stage 1: CIFAR-10

**File**: `src/thalia/datasets/cifar_wrapper.py`

### Quick Start
```python
from thalia.datasets import create_stage1_cifar_datasets

train_dataset, test_dataset = create_stage1_cifar_datasets(
    device=device,
    encoding="rate",    # "rate", "temporal", or "phase"
    n_timesteps=100,
)
spikes, label = train_dataset[0]
# spikes: (n_timesteps, 3, 32, 32) binary spikes
# label: 0-9 class label
```

### Encoding Types

**Rate Coding** (default):
- Firing probability ∝ pixel intensity
- Poisson spikes over time window
- Most biologically realistic

**Temporal Coding**:
- Brighter pixels spike earlier
- Latency encodes intensity
- One spike per pixel

**Phase Coding**:
- Spike timing relative to gamma (40 Hz)
- Phase within cycle encodes intensity
- Requires oscillator

### Configuration
```python
from thalia.datasets import CIFARConfig

config = CIFARConfig(
    encoding="rate",
    n_timesteps=100,
    max_firing_rate=0.8,      # Max spikes/timestep
    min_intensity=0.1,        # Ignore dark pixels
    normalize=True,           # Use ImageNet stats
    augment=True,             # Random crops/flips (train only)
    flatten=False,            # Keep spatial structure
    device=device,
)
dataset = CIFARForThalia(config, train=True)
```

### Batch Processing
```python
indices = list(range(16))
spikes_batch, labels = dataset.get_batch(indices)
# spikes_batch: (16, n_timesteps, 3, 32, 32)
```

### Statistics
```python
stats = dataset.analyze_encoding_statistics(n_samples=100)
# stats: {mean_firing_rate, sparsity, n_samples_analyzed}
```

---

## Stage 2: Grammar

**File**: `src/thalia/datasets/grammar.py`

### Quick Start
```python
from thalia.datasets import create_stage2_grammar_dataset

dataset = create_stage2_grammar_dataset(
    device=device,
    multilingual=False,  # Set True for SOV word order
)
phrase, is_grammatical, rule = dataset.generate_phrase()
# phrase: List[int] (word indices)
# is_grammatical: bool
# rule: GrammarRule tested
```

### Grammar Rules
- **SUBJECT_VERB_AGREEMENT**: "The cat runs" vs "The cats run"
- **NOUN_ADJECTIVE**: "The big cat" (correct order)
- **WORD_ORDER_SVO**: Subject-Verb-Object (English)
- **WORD_ORDER_SOV**: Subject-Object-Verb (Japanese, Turkish)
- **PLURAL_MORPHOLOGY**: Determiner-noun agreement

### Configuration
```python
from thalia.datasets import GrammarConfig, GrammarRule

config = GrammarConfig(
    max_phrase_length=5,
    rules_to_test=[
        GrammarRule.SUBJECT_VERB_AGREEMENT,
        GrammarRule.NOUN_ADJECTIVE,
        GrammarRule.WORD_ORDER_SVO,
    ],
    violation_probability=0.2,  # Grammatical errors
    embedding_dim=64,
    multilingual=False,
    device=device,
)
dataset = GrammarDataset(config)
```

### Batch Generation
```python
phrases, labels, rules = dataset.generate_batch(
    batch_size=32,
    balance_rules=True,      # Equal rule distribution
    balance_violations=True, # 50% grammatical, 50% violations
)
# phrases: (32, max_length) padded word indices
# labels: (32,) 1=grammatical, 0=violation
```

### Vocabulary
```python
vocab = dataset.vocab
word_idx = vocab.word2idx['cat']
word = vocab.decode([word_idx])  # ['cat']
print(f"Vocabulary size: {vocab.vocab_size}")
```

### Evaluation
```python
accuracy = dataset.compute_accuracy(predictions, labels)
results = dataset.analyze_rule_learning(brain, n_test_phrases=100)
# results: {rule → accuracy}
```

---

## Stage 3: Reading

**File**: `src/thalia/datasets/reading.py`

### Quick Start
```python
from thalia.datasets import create_stage3_reading_dataset

dataset = create_stage3_reading_dataset(device=device)
task_data, label, task_type = dataset.generate_task()
# task_data: Dict with task-specific inputs
# label: Target output (varies by task)
# task_type: ReadingTask enum
```

### Task Types

**PHONEME_TO_WORD**: Decode phonemes → word
```python
task_data, label, _ = dataset.generate_task(ReadingTask.PHONEME_TO_WORD)
# task_data: {'phonemes': tensor of phoneme indices}
# label: word index
```

**WORD_TO_MEANING**: Map word → semantic features
```python
task_data, label, _ = dataset.generate_task(ReadingTask.WORD_TO_MEANING)
# task_data: {'word': word index}
# label: [is_animate, is_object, is_action, size] (4D vector)
```

**SENTENCE_COMPLETION**: Fill in missing word
```python
task_data, label, _ = dataset.generate_task(ReadingTask.SENTENCE_COMPLETION)
# task_data: {'sentence': indices with <UNK>, 'missing_position': int}
# label: missing word index
```

**SIMPLE_QA**: Answer who/what/where questions
```python
task_data, label, _ = dataset.generate_task(ReadingTask.SIMPLE_QA)
# task_data: {'sentence': indices, 'question': indices}
# label: answer word index
```

**SEMANTIC_ROLE**: Identify agent/action/patient
```python
task_data, label, _ = dataset.generate_task(ReadingTask.SEMANTIC_ROLE)
# task_data: {'sentence': indices}
# label: [agent_pos, action_pos, patient_pos] (3D vector)
```

### Configuration
```python
from thalia.datasets import ReadingConfig, ReadingTask

config = ReadingConfig(
    vocab_size=500,
    max_sentence_length=10,
    max_phonemes=15,
    tasks_to_test=[
        ReadingTask.PHONEME_TO_WORD,
        ReadingTask.WORD_TO_MEANING,
        ReadingTask.SENTENCE_COMPLETION,
        ReadingTask.SIMPLE_QA,
    ],
    embedding_dim=64,
    device=device,
)
dataset = ReadingDataset(config)
```

### Vocabulary
```python
vocab = dataset.vocab
print(f"Words: {vocab.vocab_size}")
print(f"Phonemes: {vocab.n_phonemes}")
print(f"Example word: {vocab.all_words['cat']}")  # ['k', 'æ', 't']
```

### Evaluation
```python
accuracy = dataset.compute_accuracy(predictions, labels, task_type)
# Task-specific accuracy computation
```

---

## Integration with Curriculum Training

### Example: Using datasets in training loop

```python
from thalia.datasets import (
    create_stage0_temporal_dataset,
    create_stage1_cifar_datasets,
    create_stage2_grammar_dataset,
    create_stage3_reading_dataset,
)

# Initialize datasets
temporal_dataset = create_stage0_temporal_dataset(device)
cifar_train, cifar_test = create_stage1_cifar_datasets(device)
grammar_dataset = create_stage2_grammar_dataset(device)
reading_dataset = create_stage3_reading_dataset(device)

# Stage 0: Temporal patterns
for step in range(60000):
    seq, targets, pattern = temporal_dataset.generate_sequence()
    brain_output = brain.forward(seq)
    loss = compute_loss(brain_output, targets)
    # ... learning

# Stage 1: Visual categories
for step in range(80000):
    spikes, label = cifar_train[step % len(cifar_train)]
    brain_output = brain.forward(spikes)
    # ... learning

# Stage 2: Grammar rules
for step in range(100000):
    phrase, is_gram, rule = grammar_dataset.generate_phrase()
    brain_output = brain.judge_grammaticality(phrase)
    # ... learning

# Stage 3: Reading comprehension
for step in range(120000):
    task_data, label, task_type = reading_dataset.generate_task()
    brain_output = brain.process_reading_task(task_data, task_type)
    # ... learning
```

---

## Running the Demo

```bash
python examples/task_specific_datasets_demo.py
```

**Output**:
- Temporal sequences: Pattern generation and statistics
- CIFAR-10: Encoding methods and sparsity analysis
- Grammar: Rule testing and violation detection
- Reading: All task types demonstrated

---

## Files Created

```
src/thalia/datasets/
├── temporal_sequences.py      # Stage 0 - 500+ lines
├── cifar_wrapper.py          # Stage 1 - 600+ lines
├── grammar.py                # Stage 2 - 700+ lines
├── reading.py                # Stage 3 - 800+ lines
└── __init__.py               # Updated exports

examples/
└── task_specific_datasets_demo.py  # Comprehensive demo
```

---

## Next Steps

1. **Integration**: Use datasets in `CurriculumTrainer.train_stage()`
2. **Evaluation**: Implement real metric computation in `stage_evaluation.py`
3. **Testing**: Write unit tests for each dataset
4. **Stage 4+**: Create abstract reasoning and expert domain datasets

---

**Last Updated**: December 8, 2025  
**Status**: All Stage 0-3 datasets complete and tested ✅
