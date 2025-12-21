# Datasets Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 19:26:47
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all dataset classes and factory functions for curriculum training stages.

Total: 4 dataset classes, 4 factory functions

## Dataset Classes

### `GrammarDataset`

**Source**: `thalia\datasets\grammar.py`

**Description**: Grammar learning dataset for Stage 2.

---

### `PhonologicalDataset`

**Source**: `thalia\datasets\phonology.py`

**Description**: Phonological tasks for categorical perception learning.

---

### `ReadingDataset`

**Source**: `thalia\datasets\reading.py`

**Description**: Reading comprehension dataset for Stage 3.

---

### `TemporalSequenceDataset`

**Source**: `thalia\datasets\temporal_sequences.py`

**Description**: Temporal sequence dataset for Stage 0 (Phonology).

---

## Factory Functions

### `create_stage0_temporal_dataset()`

**Source**: `thalia\datasets\temporal_sequences.py`

**Parameters**:

- `device`

**Description**: Create temporal sequence dataset for Stage 0.

---

### `create_stage1_cifar_datasets()`

**Source**: `thalia\datasets\cifar_wrapper.py`

**Parameters**:

- `device`
- `encoding`
- `n_timesteps`

**Description**: Create CIFAR-10 train/test datasets for Stage 1.

---

### `create_stage2_grammar_dataset()`

**Source**: `thalia\datasets\grammar.py`

**Parameters**:

- `device`
- `multilingual`
- `language`

**Description**: Create grammar dataset for Stage 2.

---

### `create_stage3_reading_dataset()`

**Source**: `thalia\datasets\reading.py`

**Parameters**:

- `device`
- `language`

**Description**: Create reading dataset for Stage 3.

---

