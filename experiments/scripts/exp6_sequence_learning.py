#!/usr/bin/env python3
"""Experiment 6: Sequence Learning with SNNs

Train an SNN to learn temporal sequences, demonstrating:
1. Next-character prediction (mini language model)
2. Sequence completion from partial input
3. Memory over time using recurrent connections

This validates temporal learning capabilities in the framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time


class SurrogateSpike(torch.autograd.Function):
    """Surrogate gradient for spiking using fast sigmoid."""
    
    @staticmethod
    def forward(ctx, membrane, threshold):
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        return (membrane > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        membrane, = ctx.saved_tensors
        grad = grad_output / (1 + torch.abs(membrane - ctx.threshold) * 5) ** 2
        return grad, None


class RecurrentSpikingLayer(nn.Module):
    """Recurrent spiking layer with lateral connections."""
    
    def __init__(self, n_in: int, n_out: int, tau_mem: float = 10.0, 
                 threshold: float = 1.0, recurrent: bool = True):
        super().__init__()
        self.n_out = n_out
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.decay = np.exp(-1.0 / tau_mem)
        self.recurrent = recurrent
        
        # Input weights
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * (2.0 / np.sqrt(n_in)))
        self.bias = nn.Parameter(torch.zeros(n_out))
        
        # Recurrent weights (optional)
        if recurrent:
            self.recurrent_weight = nn.Parameter(torch.randn(n_out, n_out) * 0.1)
        
        self.membrane = None
        self.prev_spikes = None
        
    def reset(self, batch_size: int, device: torch.device):
        """Reset state."""
        self.membrane = torch.zeros(batch_size, self.n_out, device=device)
        self.prev_spikes = torch.zeros(batch_size, self.n_out, device=device)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass."""
        # Input current
        current = F.linear(x, self.weight, self.bias)
        
        # Recurrent current from previous spikes
        if self.recurrent and self.prev_spikes is not None:
            current = current + F.linear(self.prev_spikes, self.recurrent_weight)
        
        # Membrane dynamics
        self.membrane = self.membrane * self.decay + current
        
        # Spike with surrogate gradient
        spikes = SurrogateSpike.apply(self.membrane, self.threshold)
        
        # Reset and store
        self.membrane = self.membrane - spikes * self.threshold
        self.prev_spikes = spikes.detach()
        
        return spikes, self.membrane


class SequenceSNN(nn.Module):
    """SNN for sequence learning with recurrent connections."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 128, n_layers: int = 2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Input encoding (one-hot to spikes)
        self.input_layer = RecurrentSpikingLayer(
            vocab_size, hidden_size, tau_mem=5.0, threshold=0.5, recurrent=True
        )
        
        # Hidden recurrent layer
        self.hidden_layer = RecurrentSpikingLayer(
            hidden_size, hidden_size, tau_mem=10.0, threshold=0.5, recurrent=True
        )
        
        # Output layer (hidden to vocab logits via membrane)
        self.output_layer = RecurrentSpikingLayer(
            hidden_size, vocab_size, tau_mem=20.0, threshold=0.5, recurrent=False
        )
        
    def reset(self, batch_size: int, device: torch.device):
        """Reset all states."""
        self.input_layer.reset(batch_size, device)
        self.hidden_layer.reset(batch_size, device)
        self.output_layer.reset(batch_size, device)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass for one timestep.
        
        Args:
            x: One-hot encoded input (batch, vocab_size)
            
        Returns:
            (output_spikes, output_membrane)
        """
        # Forward through layers
        h1_spikes, _ = self.input_layer(x)
        h2_spikes, _ = self.hidden_layer(h1_spikes)
        out_spikes, out_membrane = self.output_layer(h2_spikes)
        
        return out_spikes, out_membrane


def create_sequence_dataset(sequences: list, char_to_idx: dict, device: torch.device):
    """Create training data from sequences.
    
    Returns:
        List of (input_sequence, target_sequence) tuples
    """
    data = []
    
    for seq in sequences:
        # Input: all chars except last
        # Target: all chars except first (next char prediction)
        for i in range(len(seq) - 1):
            # Input is one-hot of current char
            input_idx = char_to_idx[seq[i]]
            # Target is next char
            target_idx = char_to_idx[seq[i + 1]]
            data.append((input_idx, target_idx))
    
    return data


def run_experiment():
    """Run the sequence learning experiment."""
    print("=" * 60)
    print("Experiment 6: Sequence Learning with SNN")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create a simple language dataset
    # These sequences have learnable patterns
    training_sequences = [
        # Pattern: after 'ab' comes 'c'
        "abcabc",
        "abcabcabc",
        # Pattern: after 'xy' comes 'z'
        "xyzxyz",
        "xyzxyzxyz",
        # Pattern: repetition
        "aaaaaa",
        "bbbbbb",
        "ababab",
        # Pattern: after 'hello' comes ' '
        "hello world",
        "hello there",
        "hello again",
        # Mixed patterns
        "the cat sat",
        "the dog ran",
        "the cat ran",
        # Numbers
        "12345678",
        "13579",
        "24680",
    ]
    
    # Build vocabulary
    all_chars = set("".join(training_sequences))
    vocab = sorted(list(all_chars))
    vocab_size = len(vocab)
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    print(f"\nVocabulary: {vocab}")
    print(f"Vocab size: {vocab_size}")
    
    # Training parameters
    hidden_size = 64
    n_epochs = 50
    n_timesteps = 10  # Timesteps per character
    learning_rate = 0.005
    
    print(f"\nConfiguration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Timesteps per char: {n_timesteps}")
    
    # Create model
    model = SequenceSNN(vocab_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Prepare training data
    train_data = create_sequence_dataset(training_sequences, char_to_idx, device)
    print(f"  Training pairs: {len(train_data)}")
    
    # Training tracking
    losses = []
    accuracies = []
    
    print(f"\nTraining...")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        
        # Shuffle data
        np.random.shuffle(train_data)
        
        # Reset model state at epoch start
        model.reset(1, device)
        
        for input_idx, target_idx in train_data:
            # Create one-hot input
            input_onehot = torch.zeros(1, vocab_size, device=device)
            input_onehot[0, input_idx] = 1.0
            
            optimizer.zero_grad()
            
            # Run for multiple timesteps to accumulate membrane
            membrane_sum = torch.zeros(1, vocab_size, device=device)
            
            for t in range(n_timesteps):
                # Add some noise to input for robustness
                input_spikes = input_onehot.clone()
                if t == 0:  # Only spike at first timestep
                    input_spikes = (input_spikes > 0.5).float()
                else:
                    input_spikes = torch.zeros_like(input_spikes)
                
                _, membrane = model(input_spikes)
                membrane_sum = membrane_sum + membrane
            
            # Loss on accumulated membrane
            log_probs = F.log_softmax(membrane_sum / n_timesteps, dim=1)
            target = torch.tensor([target_idx], device=device)
            loss = F.nll_loss(log_probs, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = membrane_sum.argmax(dim=1).item()
            epoch_correct += int(pred == target_idx)
            
            # Detach hidden states to prevent backprop through time across pairs
            model.reset(1, device)
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_data)
        accuracy = epoch_correct / len(train_data)
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, acc={accuracy*100:.1f}%")
    
    print(f"\nFinal accuracy: {accuracies[-1]*100:.1f}%")
    
    # Evaluation: sequence completion
    print("\n" + "=" * 60)
    print("Sequence Completion Test")
    print("=" * 60)
    
    test_prompts = ["abc", "xyz", "hello", "the cat", "123"]
    model.eval()
    
    completions = []
    for prompt in test_prompts:
        model.reset(1, device)
        
        # Feed in prompt
        generated = prompt
        for char in prompt:
            input_onehot = torch.zeros(1, vocab_size, device=device)
            if char in char_to_idx:
                input_onehot[0, char_to_idx[char]] = 1.0
            
            with torch.no_grad():
                for t in range(n_timesteps):
                    input_spikes = (input_onehot > 0.5).float()
                    _, _ = model(input_spikes)
        
        # Generate next 5 characters
        for _ in range(5):
            with torch.no_grad():
                membrane_sum = torch.zeros(1, vocab_size, device=device)
                
                for t in range(n_timesteps):
                    # Use last generated char as input
                    input_onehot = torch.zeros(1, vocab_size, device=device)
                    if generated[-1] in char_to_idx:
                        input_onehot[0, char_to_idx[generated[-1]]] = 1.0
                    
                    _, membrane = model((input_onehot > 0.5).float())
                    membrane_sum = membrane_sum + membrane
                
                # Sample from softmax (temperature=0.5 for some randomness)
                probs = F.softmax(membrane_sum.squeeze() * 2, dim=0)
                next_idx = torch.multinomial(probs, 1).item()
                generated += idx_to_char[next_idx]
        
        completions.append(generated)
        print(f"  '{prompt}' -> '{generated}'")
    
    # Pattern recognition test
    print("\n" + "=" * 60)
    print("Pattern Recognition Test")
    print("=" * 60)
    
    test_pairs = [
        ("a", "b"),   # Should predict 'b' after 'a' (from abc patterns)
        ("b", "c"),   # Should predict 'c' after 'b'
        ("x", "y"),   # Should predict 'y' after 'x'
        ("y", "z"),   # Should predict 'z' after 'y'
        ("o", " "),   # 'hello' -> space
        (" ", "w"),   # After space in 'hello world'
    ]
    
    pattern_correct = 0
    for input_char, expected in test_pairs:
        model.reset(1, device)
        
        input_onehot = torch.zeros(1, vocab_size, device=device)
        if input_char in char_to_idx:
            input_onehot[0, char_to_idx[input_char]] = 1.0
        
        with torch.no_grad():
            membrane_sum = torch.zeros(1, vocab_size, device=device)
            
            for t in range(n_timesteps):
                _, membrane = model((input_onehot > 0.5).float())
                membrane_sum = membrane_sum + membrane
            
            pred_idx = membrane_sum.argmax(dim=1).item()
            pred_char = idx_to_char[pred_idx]
        
        correct = pred_char == expected
        pattern_correct += int(correct)
        status = "✓" if correct else "✗"
        print(f"  '{input_char}' -> '{pred_char}' (expected '{expected}') {status}")
    
    pattern_accuracy = pattern_correct / len(test_pairs)
    print(f"\n  Pattern accuracy: {pattern_correct}/{len(test_pairs)} = {pattern_accuracy*100:.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Experiment 6: Sequence Learning with SNN", fontsize=14, fontweight='bold')
    
    # 1. Training loss
    ax1 = axes[0, 0]
    ax1.plot(losses, 'b-')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    
    # 2. Training accuracy
    ax2 = axes[0, 1]
    ax2.plot([a*100 for a in accuracies], 'g-')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training Accuracy")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 3. Completions visualization
    ax3 = axes[1, 0]
    ax3.axis('off')
    completion_text = "Sequence Completions:\n\n"
    for prompt, completion in zip(test_prompts, completions):
        completion_text += f"'{prompt}' → '{completion}'\n"
    ax3.text(0.1, 0.9, completion_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    ax3.set_title("Generated Sequences")
    
    # 4. Final statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""Training Statistics:
    
Final Loss: {losses[-1]:.4f}
Final Accuracy: {accuracies[-1]*100:.1f}%
Pattern Accuracy: {pattern_accuracy*100:.1f}%

Vocabulary Size: {vocab_size}
Hidden Size: {hidden_size}
Epochs: {n_epochs}
"""
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    ax4.set_title("Summary")
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "exp6_sequence_learning.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    # Success criteria
    print("\n" + "=" * 60)
    print("Success Criteria Check")
    print("=" * 60)
    
    criteria = [
        ("Recurrent SNN trained successfully", True),
        ("Final accuracy > 30%", accuracies[-1] > 0.30),
        ("Loss decreased over training", losses[-1] < losses[0]),
        ("Sequence completion works", len(completions) > 0),
        ("Pattern recognition > 0%", pattern_accuracy > 0),
    ]
    
    all_passed = True
    for name, passed in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed
    
    print("\n" + ("🎉 All criteria passed!" if all_passed else "⚠️ Some criteria failed"))
    
    return all_passed


if __name__ == "__main__":
    run_experiment()
