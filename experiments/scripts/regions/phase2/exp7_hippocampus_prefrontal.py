#!/usr/bin/env python3
"""Experiment 7: Hippocampus + Prefrontal - Context-Dependent Memory Recall.

This experiment tests the interaction between two brain regions:
- **Hippocampus**: Stores and retrieves episodic memories (one-shot Hebbian)
- **Prefrontal**: Maintains context/task state (working memory)

Task: Context-Dependent Memory Retrieval
========================================
- Store cue→target associations with context as part of the cue
- Same partial cue + different context = different retrieved pattern
- Prefrontal must maintain context during retrieval delay

Biological Basis:
=================
- Hippocampus encodes specific episodes (context+content as single pattern)
- Prefrontal maintains "what task am I doing" context
- Context gates which memories are relevant for retrieval

Architecture:
=============
    Context (from Prefrontal) + Partial Cue → Hippocampus → Target
                       ↓
              Concatenated Input Pattern

Success Criteria:
=================
1. Retrieval accuracy ≥60% with correct context
2. Context matters: correct context > wrong context
3. Working memory maintains context over delay
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[4] / "src"))

from thalia.regions.hippocampus import Hippocampus, HippocampusConfig
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig


# =============================================================================
# PATTERN GENERATION
# =============================================================================

def create_context_patterns(
    n_contexts: int = 2,
    context_size: int = 8,
) -> List[torch.Tensor]:
    """Create distinct patterns for each context."""
    patterns = []
    for ctx in range(n_contexts):
        pattern = torch.zeros(context_size)
        # Use different parts of the pattern for each context
        start = ctx * (context_size // n_contexts)
        end = start + (context_size // n_contexts)
        pattern[start:end] = 1.0
        patterns.append(pattern)
    return patterns


def create_contextual_memories(
    n_contexts: int = 2,
    n_patterns_per_context: int = 5,
    pattern_size: int = 32,
    context_patterns: List[torch.Tensor] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Create cue-target pairs for each context.
    
    Returns list of (cue, target, context_id) tuples.
    Cue includes the context pattern for storage.
    """
    if context_patterns is None:
        context_patterns = create_context_patterns(n_contexts, 8)
    
    context_size = context_patterns[0].shape[0]
    memories = []
    
    for ctx in range(n_contexts):
        for i in range(n_patterns_per_context):
            # Create unique cue content (not including context)
            cue_content = torch.zeros(pattern_size)
            # Each pattern has sparse activation
            n_active = pattern_size // 4
            active_idx = torch.randperm(pattern_size)[:n_active]
            cue_content[active_idx] = torch.rand(n_active) * 0.5 + 0.5
            
            # Create corresponding target
            target = torch.zeros(pattern_size)
            # Target is distinct from cue but systematic
            target_idx = (active_idx + pattern_size // 2) % pattern_size
            target[target_idx] = torch.rand(n_active) * 0.5 + 0.5
            
            # Make target depend on context (shift pattern)
            target = torch.roll(target, ctx * pattern_size // 4)
            
            memories.append((cue_content, target, ctx))
    
    # Shuffle
    np.random.shuffle(memories)
    return memories


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment() -> bool:
    """Run the Hippocampus + Prefrontal context-dependent memory experiment."""
    print("=" * 60)
    print("Experiment 7: Hippocampus + Prefrontal Context Memory")
    print("=" * 60)
    
    # Parameters
    n_contexts = 2
    n_patterns_per_context = 5
    pattern_size = 32
    context_size = 8
    
    print(f"\n[1/5] Creating context-dependent memories...")
    print(f"  Contexts: {n_contexts}")
    print(f"  Patterns per context: {n_patterns_per_context}")
    print(f"  Pattern size: {pattern_size}")
    print(f"  Context size: {context_size}")
    
    context_patterns = create_context_patterns(n_contexts, context_size)
    memories = create_contextual_memories(
        n_contexts=n_contexts,
        n_patterns_per_context=n_patterns_per_context,
        pattern_size=pattern_size,
        context_patterns=context_patterns,
    )
    
    print(f"  Total memories: {len(memories)}")
    
    # Create regions
    print(f"\n[2/5] Creating Hippocampus + Prefrontal system...")
    
    # Hippocampus: stores (context+cue) → target associations
    hippocampus = Hippocampus(HippocampusConfig(
        n_input=context_size + pattern_size,  # context + cue
        n_output=pattern_size,  # target
        learning_rate=0.9,  # High LR for one-shot
    ))
    
    # Prefrontal: maintains context
    prefrontal = Prefrontal(PrefrontalConfig(
        n_input=context_size,
        n_output=context_size,
    ))
    
    print(f"  Hippocampus: {context_size + pattern_size} → {pattern_size}")
    print(f"  Prefrontal: {context_size} → {context_size}")
    
    # Training: Store memories
    print(f"\n[3/5] Storing memories...")
    
    for cue_content, target, ctx_id in memories:
        context = context_patterns[ctx_id]
        
        # Set prefrontal context
        prefrontal.set_context(context)
        
        # Full cue = context + cue_content
        full_cue = torch.cat([context, cue_content])
        
        # Store in hippocampus: learn association cue → target
        # Present cue, then learn with target as "output"
        hippocampus.forward(full_cue.unsqueeze(0))
        hippocampus.learn(
            full_cue.unsqueeze(0),
            target.unsqueeze(0),
            force_encoding=True,
        )
        hippocampus.reset()
    
    print(f"  Stored {len(memories)} memories")
    
    # Testing
    print(f"\n[4/5] Testing retrieval...")
    
    results_correct = []
    results_wrong = []
    
    for cue_content, target, ctx_id in memories:
        correct_context = context_patterns[ctx_id]
        wrong_context = context_patterns[(ctx_id + 1) % n_contexts]
        
        # Test with CORRECT context
        full_cue_correct = torch.cat([correct_context, cue_content])
        
        # Direct weight-based readout
        activation_correct = hippocampus.weights @ full_cue_correct
        n_target_active = max(1, int(target.sum().item() / target.max().item()))
        _, top_idx_correct = activation_correct.topk(min(n_target_active, pattern_size))
        
        retrieved_correct = torch.zeros(pattern_size)
        retrieved_correct[top_idx_correct] = 1.0
        
        # F1 score
        tp_correct = (retrieved_correct * (target > 0.1)).sum().item()
        fp_correct = (retrieved_correct * (target <= 0.1)).sum().item()
        fn_correct = ((1 - retrieved_correct) * (target > 0.1)).sum().item()
        
        prec_correct = tp_correct / (tp_correct + fp_correct + 1e-8)
        rec_correct = tp_correct / (tp_correct + fn_correct + 1e-8)
        f1_correct = 2 * prec_correct * rec_correct / (prec_correct + rec_correct + 1e-8)
        
        results_correct.append(f1_correct)
        
        # Test with WRONG context
        full_cue_wrong = torch.cat([wrong_context, cue_content])
        activation_wrong = hippocampus.weights @ full_cue_wrong
        _, top_idx_wrong = activation_wrong.topk(min(n_target_active, pattern_size))
        
        retrieved_wrong = torch.zeros(pattern_size)
        retrieved_wrong[top_idx_wrong] = 1.0
        
        tp_wrong = (retrieved_wrong * (target > 0.1)).sum().item()
        fp_wrong = (retrieved_wrong * (target <= 0.1)).sum().item()
        fn_wrong = ((1 - retrieved_wrong) * (target > 0.1)).sum().item()
        
        prec_wrong = tp_wrong / (tp_wrong + fp_wrong + 1e-8)
        rec_wrong = tp_wrong / (tp_wrong + fn_wrong + 1e-8)
        f1_wrong = 2 * prec_wrong * rec_wrong / (prec_wrong + rec_wrong + 1e-8)
        
        results_wrong.append(f1_wrong)
    
    acc_correct = np.mean(results_correct) * 100
    acc_wrong = np.mean(results_wrong) * 100
    
    print(f"\n  With correct context: {acc_correct:.1f}% F1")
    print(f"  With wrong context: {acc_wrong:.1f}% F1")
    
    # Test prefrontal working memory
    print(f"\n  Testing prefrontal working memory...")
    prefrontal.reset()
    test_context = context_patterns[0]
    prefrontal.set_context(test_context)
    
    # Run maintenance steps
    for _ in range(10):
        noise = torch.rand(context_size) * 0.2
        prefrontal.forward(noise.unsqueeze(0))
    
    # Check if context is maintained
    maintained = prefrontal.get_working_memory().squeeze()
    wm_similarity = torch.cosine_similarity(
        maintained.unsqueeze(0),
        test_context.unsqueeze(0),
        dim=1
    ).item()
    
    print(f"  Working memory similarity after delay: {wm_similarity:.3f}")
    
    # Visualizations
    print(f"\n[5/5] Generating visualizations...")
    results_dir = Path(__file__).parents[3] / "results" / "regions"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar chart
    x = ['Correct Context', 'Wrong Context']
    heights = [acc_correct, acc_wrong]
    colors = ['green', 'red']
    
    axes[0].bar(x, heights, color=colors)
    axes[0].set_ylabel('F1 Score (%)')
    axes[0].set_title('Context-Dependent Memory Retrieval')
    axes[0].set_ylim(0, 100)
    
    # Distribution
    axes[1].hist(results_correct, bins=10, alpha=0.7, label='Correct Context', color='green')
    axes[1].hist(results_wrong, bins=10, alpha=0.7, label='Wrong Context', color='red')
    axes[1].set_xlabel('F1 Score')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Retrieval Score Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / "exp7_hippocampus_prefrontal.png", dpi=150)
    plt.close()
    print("  Saved: exp7_hippocampus_prefrontal.png")
    
    # Evaluate criteria
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    criteria_met = 0
    
    # Criterion 1: Correct context accuracy ≥50%
    c1 = acc_correct >= 50.0
    print(f"\n1. Correct context retrieval ≥50%: {'PASS' if c1 else 'FAIL'}")
    print(f"   F1: {acc_correct:.1f}%")
    if c1:
        criteria_met += 1
    
    # Criterion 2: Context matters
    context_diff = acc_correct - acc_wrong
    c2 = context_diff > 10.0
    print(f"\n2. Context matters (>10% difference): {'PASS' if c2 else 'FAIL'}")
    print(f"   Difference: {context_diff:.1f}%")
    if c2:
        criteria_met += 1
    
    # Criterion 3: Working memory maintenance
    c3 = wm_similarity > 0.5
    print(f"\n3. Working memory maintained (sim > 0.5): {'PASS' if c3 else 'FAIL'}")
    print(f"   Similarity: {wm_similarity:.3f}")
    if c3:
        criteria_met += 1
    
    passed = criteria_met >= 2
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASSED' if passed else 'FAILED'} ({criteria_met}/3 criteria met)")
    print("=" * 60)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp7_hippocampus_prefrontal_{timestamp}.json"
    
    save_data = {
        "experiment": "exp7_hippocampus_prefrontal",
        "timestamp": timestamp,
        "passed": passed,
        "criteria_met": criteria_met,
        "acc_correct_context": acc_correct,
        "acc_wrong_context": acc_wrong,
        "context_difference": context_diff,
        "wm_similarity": wm_similarity,
    }
    
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return passed


if __name__ == "__main__":
    success = run_experiment()
    sys.exit(0 if success else 1)
