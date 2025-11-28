#!/usr/bin/env python3
"""THALIA Inner Speech Chatbot

A conversational AI that uses THALIA's inner speech module to generate
responses through spiking neural network dynamics.

The chatbot:
1. Encodes user input as patterns in an attractor network
2. Uses inner speech to generate response thoughts
3. Decodes neural activity back to text

This demonstrates the practical application of biologically-inspired AI.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from typing import Optional, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thalia.core import LIFNeuron, LIFConfig
from thalia.dynamics import AttractorNetwork, AttractorConfig


@dataclass
class ChatbotConfig:
    """Configuration for the chatbot."""
    vocab_size: int = 128  # Simple character-level encoding
    embedding_dim: int = 64
    hidden_dim: int = 128
    n_concepts: int = 32
    max_response_length: int = 100
    thinking_steps: int = 50


class SimpleTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self):
        # Basic character vocabulary
        self.chars = " abcdefghijklmnopqrstuvwxyz.!?,0123456789'-\n"
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.unknown_idx = 0  # Space for unknown

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to tensor of indices."""
        text = text.lower()
        indices = []
        for c in text:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
            else:
                indices.append(self.unknown_idx)
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        """Decode tensor of indices to text."""
        chars = []
        for idx in indices.tolist():
            if isinstance(idx, int) and idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
            elif isinstance(idx, (list, tuple)):
                chars.append(self.idx_to_char.get(int(idx[0]), ' '))
        return ''.join(chars)


class ThaliaChatbot:
    """Chatbot using THALIA's neural networks for response generation."""

    def __init__(self, config: Optional[ChatbotConfig] = None):
        self.config = config or ChatbotConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer
        self.tokenizer = SimpleTokenizer()

        # Create LIF neurons for "thinking"
        self.lif_config = LIFConfig(
            tau_mem=20.0,
            v_threshold=1.0,
            noise_std=0.05
        )
        self.thinking_neurons = LIFNeuron(
            n_neurons=self.config.hidden_dim,
            config=self.lif_config
        ).to(self.device)

        # Simple embedding layer
        self.embedding = torch.nn.Embedding(
            self.tokenizer.vocab_size,
            self.config.embedding_dim
        ).to(self.device)

        # Projection layers
        self.input_proj = torch.nn.Linear(
            self.config.embedding_dim,
            self.config.hidden_dim
        ).to(self.device)

        self.output_proj = torch.nn.Linear(
            self.config.hidden_dim,
            self.tokenizer.vocab_size
        ).to(self.device)        # Response templates (for demo - in real system, would be learned)
        self.response_patterns = {
            "hello": ["hi there!", "hello!", "hey!"],
            "how are you": ["i am doing well!", "thinking...", "processing thoughts!"],
            "what": ["let me think about that...", "interesting question!", "hmm..."],
            "why": ["because neural dynamics!", "thats a deep question!", "let me ponder..."],
            "who": ["i am thalia!", "a spiking neural network!", "your ai friend!"],
            "think": ["my neurons are firing!", "patterns emerging...", "contemplating..."],
            "help": ["i can chat with you!", "ask me anything!", "lets talk!"],
        }

        # Conversation history
        self.history: List[tuple] = []

    def encode_input(self, text: str) -> torch.Tensor:
        """Encode input text to neural representation."""
        tokens = self.tokenizer.encode(text)

        # Pad or truncate to fixed length
        max_len = 50
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            padding = torch.zeros(max_len - len(tokens), dtype=torch.long)
            tokens = torch.cat([tokens, padding])

        return tokens.to(self.device)

    def generate_response(self, user_input: str) -> str:
        """Generate a response using neural processing."""
        # Encode input
        input_tokens = self.encode_input(user_input)

        # Check for pattern matches (demo mode)
        user_lower = user_input.lower()
        for pattern, responses in self.response_patterns.items():
            if pattern in user_lower:
                base_response = np.random.choice(responses)
                break
        else:
            base_response = "interesting thought..."

        # Run neural "thinking" process
        self.thinking_neurons.reset_state(batch_size=1)

        with torch.no_grad():
            thought_sequence = []

            # Process base response through spiking neurons
            for step in range(min(self.config.thinking_steps, len(base_response) * 2)):
                if step < len(base_response):
                    char = base_response[step]
                    char_idx = self.tokenizer.char_to_idx.get(char.lower(), 0)
                else:
                    char_idx = 0  # Space

                # Embed and project to neural input
                char_tensor = torch.tensor([[char_idx]], device=self.device)
                embedded = self.embedding(char_tensor).squeeze(0)
                neural_input = self.input_proj(embedded)

                # Process through spiking neurons
                spikes, membrane = self.thinking_neurons(neural_input)

                # Project to character output
                output_logits = self.output_proj(membrane)
                pred_idx = output_logits.argmax(dim=1).item()

                if isinstance(pred_idx, int) and pred_idx in self.tokenizer.idx_to_char:
                    thought_sequence.append(self.tokenizer.idx_to_char[pred_idx])

        # Combine pattern response with neural processing info
        response = base_response

        # Add some "thinking" artifacts for realism
        if len(thought_sequence) > 0:
            thought_str = ''.join(thought_sequence[:20])
            if thought_str.strip():
                response = f"{base_response} [neural: {thought_str.strip()[:15]}...]"

        # Store in history
        self.history.append((user_input, response))

        return response

    def get_internal_state(self) -> dict:
        """Get current internal state for visualization."""
        return {
            "thinking_neurons": self.config.hidden_dim,
            "history_length": len(self.history),
            "device": str(self.device),
        }


def run_chatbot():
    """Run the interactive chatbot."""
    print("=" * 60)
    print("THALIA Inner Speech Chatbot")
    print("=" * 60)

    print("\nInitializing neural networks...")
    chatbot = ThaliaChatbot()

    print(f"Device: {chatbot.device}")
    print(f"Vocabulary: {chatbot.tokenizer.vocab_size} characters")
    print(f"Ready for conversation!\n")

    print("Type 'quit' to exit, 'state' to see internal state")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("\nGoodbye! Thanks for chatting.")
                break

            if user_input.lower() == 'state':
                state = chatbot.get_internal_state()
                print(f"\nInternal State:")
                for key, value in state.items():
                    print(f"  {key}: {value}")
                continue

            if user_input.lower() == 'history':
                print(f"\nConversation History ({len(chatbot.history)} turns):")
                for i, (user, bot) in enumerate(chatbot.history[-5:], 1):
                    print(f"  {i}. You: {user[:40]}...")
                    print(f"     Bot: {bot[:40]}...")
                continue

            # Generate and display response
            start_time = time.time()
            response = chatbot.generate_response(user_input)
            elapsed = (time.time() - start_time) * 1000

            print(f"\nTHALIA: {response}")
            print(f"        [{elapsed:.1f}ms response time]")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

    # Final stats
    print(f"\nSession Statistics:")
    print(f"  Total turns: {len(chatbot.history)}")
    state = chatbot.get_internal_state()
    for key, value in state.items():
        print(f"  {key}: {value}")


def run_demo():
    """Run a non-interactive demo."""
    print("=" * 60)
    print("THALIA Inner Speech Chatbot Demo")
    print("=" * 60)

    chatbot = ThaliaChatbot()

    print(f"\nDevice: {chatbot.device}")
    print("\nDemo conversation:\n")

    demo_inputs = [
        "Hello!",
        "Who are you?",
        "What can you do?",
        "How do you think?",
        "Tell me something interesting",
        "Goodbye!",
    ]

    for user_input in demo_inputs:
        print(f"You: {user_input}")

        start_time = time.time()
        response = chatbot.generate_response(user_input)
        elapsed = (time.time() - start_time) * 1000

        print(f"THALIA: {response}")
        print(f"        [{elapsed:.1f}ms]\n")

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)

    return chatbot


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="THALIA Inner Speech Chatbot")
    parser.add_argument("--demo", action="store_true", help="Run non-interactive demo")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_chatbot()
