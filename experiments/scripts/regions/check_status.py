"""Check status of all experiments."""
import json
from pathlib import Path

results_dir = Path("experiments/results/regions")

print("=" * 60)
print("EXPERIMENT STATUS SUMMARY")
print("=" * 60)

# Exp2: Cerebellum XOR
try:
    with open(results_dir / "exp2_cerebellum_xor_20251201_073404.json") as f:
        d = json.load(f)
    print("\nExp2: Cerebellum XOR")
    print(f"  Train accuracy: {d['metrics']['train_final_accuracy']:.1%}")
    print(f"  Test accuracy: {d['metrics']['test_accuracy']:.1%}")
    print(f"  Perceptron baseline: {d['metrics']['perceptron_accuracy']:.1%}")
    # Check if passed
    if d['metrics']['test_accuracy'] > 0.6 and d['metrics']['test_accuracy'] > d['metrics']['perceptron_accuracy']:
        print("  Status: ✓ PASSED (beats linear baseline)")
    else:
        print("  Status: ✗ NEEDS WORK")
except Exception as e:
    print(f"\nExp2: Error loading - {e}")

# Exp3: Striatum Bandit
try:
    with open(results_dir / "exp3_striatum_bandit_20251201_073506.json") as f:
        d = json.load(f)
    print("\nExp3: Striatum Bandit")
    print(f"  Striatum final optimal: {d['metrics']['striatum_final_optimal']:.1%}")
    print(f"  Baseline final optimal: {d['metrics']['baseline_final_optimal']:.1%}")
    print(f"  Striatum total reward: {d['metrics']['striatum_total_reward']}")
    print(f"  Baseline total reward: {d['metrics']['baseline_total_reward']}")
    if d['metrics']['striatum_final_optimal'] > 0.5:
        print("  Status: ✓ PASSED")
    else:
        print("  Status: ✗ NEEDS WORK (not learning to select optimal arm)")
except Exception as e:
    print(f"\nExp3: Error loading - {e}")

# Exp4: Hippocampus Memory
try:
    with open(results_dir / "exp4_hippocampus_memory_20251201_074822.json") as f:
        d = json.load(f)
    print("\nExp4: Hippocampus Memory")
    print(f"  Full cue F1: {d['metrics']['full_cue_f1']:.3f}")
    print(f"  Full cue recall: {d['metrics']['full_cue_recall']:.3f}")
    print(f"  Full cue precision: {d['metrics']['full_cue_precision']:.3f}")
    if d['metrics']['full_cue_f1'] > 0.5:
        print("  Status: ✓ PASSED")
    else:
        print("  Status: ✗ NEEDS WORK (low F1 score)")
except Exception as e:
    print(f"\nExp4: Error loading - {e}")

# Exp5: Prefrontal Delay
try:
    with open(results_dir / "exp5_prefrontal_delay_20251201_073957.json") as f:
        d = json.load(f)
    print("\nExp5: Prefrontal Delay")
    # Check delay sweep
    delays = d['delay_sweep']['delay_lengths']
    accuracies = d['delay_sweep']['accuracies']
    print(f"  Delay sweep: {list(zip(delays, [f'{a:.1%}' for a in accuracies]))}")
    # Check if maintains above chance (25% for 4 classes) at long delays
    if accuracies[-1] > 0.5:  # 50 timestep delay
        print("  Status: ✓ PASSED (maintains memory at long delays)")
    else:
        print("  Status: ✗ NEEDS WORK")
except Exception as e:
    print(f"\nExp5: Error loading - {e}")

print("\n" + "=" * 60)
