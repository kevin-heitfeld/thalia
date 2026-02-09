# Temporal Sequence Learning Evaluation

This script evaluates the default brain architecture on temporal sequence prediction tasks.

## Quick Start

```bash
# Basic run with defaults (1000 training trials, 100 test trials)
python training/test_sequence_learning.py

# Quick test run
python training/test_sequence_learning.py --n-training-trials 100 --n-test-trials 20

# Full evaluation with brain saving
python training/test_sequence_learning.py --n-training-trials 5000 --save-brain

# Custom architecture
python training/test_sequence_learning.py --cortex-size 500 --n-symbols 8
```

## What It Tests

### 1. **Next-Step Prediction**
Given a sequence A-B-C-A-B-C, can the brain predict the next symbol?
- Tests hippocampal episodic memory
- Engages CA3 autoassociative recurrence
- Requires temporal binding via theta oscillations

### 2. **Pattern Types**
- **ABC**: Linear sequences (A→B→C)
- **ABA**: Repetition with gap (A→B→A)
- **AAB**: Immediate repetition (A→A→B)

### 3. **Violation Detection**
Can the brain detect when patterns break?
- Measures prediction error magnitude
- Tests hippocampal prediction mechanisms
- Relevant for dopaminergic learning

### 4. **Generalization**
Test on novel symbol combinations not seen during training.

## Success Criteria

| Level | Accuracy | Description |
|-------|----------|-------------|
| **Basic** | >50% | Better than random (20% for 5 symbols) |
| **Intermediate** | >70% | Multiple pattern types, some generalization |
| **Advanced** | >85% | Robust generalization, violation detection |

## Output

Results are saved to `results/sequence_learning/`:
- `config.json`: Experiment configuration
- `metrics.json`: Training/test accuracies, dopamine values
- `brain.pt`: Trained brain state (if `--save-brain`)

## Architecture

The default brain architecture includes:
- **Thalamus**: Receives sensory input (symbol encoding)
- **Cortex**: Feature extraction and pattern recognition
- **Hippocampus**: Sequence memory (CA3 recurrence, CA1 output)
- **Prefrontal**: Working memory for context
- **Striatum**: Action selection (symbol prediction)
- **Medial Septum**: Theta oscillations for temporal binding

## Biological Accuracy

This task leverages:
- **Local learning rules**: STDP, BCM (no backprop)
- **Three-factor learning**: STDP × dopamine modulation
- **Theta oscillations**: Medial septum → hippocampus
- **Sparse coding**: <5% active neurons per timestep
- **Temporal credit assignment**: Eligibility traces

## Command-Line Options

```
--n-training-trials N    Training trials (default: 1000)
--n-test-trials N        Test trials (default: 100)
--n-symbols N            Number of distinct symbols (default: 5)
--sequence-length N      Length of each sequence (default: 10)
--cortex-size N          Cortex neurons (default: 200)
--output-dir DIR         Output directory (default: results/sequence_learning)
--save-brain             Save trained brain state
--quiet                  Suppress verbose output
--device DEVICE          cuda or cpu (auto-detected)
```

## Expected Results

With default parameters (1000 training trials):
- **Training accuracy**: 40-60%
- **Test accuracy**: 35-55%
- **Violation detection**: 10-30% increase in error

Results will improve with:
- More training trials (5000+)
- Larger architectures (cortex_size=500+)
- Tuned learning rates (requires modifying brain builder)

## Troubleshooting

**Low accuracy (<30%)**:
- Check that learning rules are enabled
- Verify dopamine modulation is working
- Increase training trials
- Check for dead neurons (no activity)

**No improvement over training**:
- Learning rates may be too low
- Synaptic weights may be saturated
- Check for silent regions (no spikes)

**Out of memory**:
- Reduce `--cortex-size`
- Use CPU instead of GPU
- Reduce sequence length

## Next Steps

After validating basic sequence learning:
1. **Add noise robustness**: Test with noisy input
2. **One-shot learning**: Learn new patterns in <10 exposures
3. **Hierarchical sequences**: ABAC, ABCABD patterns
4. **Cross-modal**: Visual → auditory sequence associations
5. **Real phonemes**: Use phonological dataset instead of symbols
