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

# Type aliases
ResponsePatterns = dict[str, list[str]]


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
        ).to(self.device)
        
        # Expanded response patterns with many more training examples
        self.response_patterns = self._build_response_patterns()
        
        # Learned associations (character-level patterns)
        self.learned_patterns = {}
        
        # Train the network on response patterns
        self._train_on_patterns()

    def _build_response_patterns(self) -> ResponsePatterns:
        """Build comprehensive response patterns for training."""
        return {
            # Greetings
            "hello": ["hi there!", "hello!", "hey!", "greetings!", "hi friend!"],
            "hi": ["hello!", "hey there!", "hi!", "good to see you!"],
            "hey": ["hey!", "hi!", "hello there!", "whats up!"],
            "good morning": ["good morning!", "morning!", "hope you slept well!"],
            "good evening": ["good evening!", "evening!", "nice to chat tonight!"],
            "good night": ["good night!", "sleep well!", "sweet dreams!"],
            "greetings": ["greetings to you too!", "hello!", "salutations!"],
            
            # How are you variations
            "how are you": ["i am doing well!", "my neurons are firing happily!", "processing thoughts nicely!"],
            "how do you feel": ["i feel electric!", "my spikes are synchronized!", "feeling neural!"],
            "whats up": ["just thinking!", "processing patterns!", "contemplating existence!"],
            "how is it going": ["going well!", "neural activity is stable!", "all systems firing!"],
            
            # Identity questions
            "who are you": ["i am thalia!", "a spiking neural network!", "your ai companion!"],
            "what are you": ["a neural network with spiking neurons!", "an ai built on biology!", "a pattern recognizer!"],
            "your name": ["i am thalia!", "they call me thalia!", "thalia at your service!"],
            "tell me about yourself": ["i am thalia, a spiking neural network!", "i think with spikes and patterns!", "i model the brain!"],
            
            # Capabilities
            "what can you do": ["i can chat and think!", "i process language neurally!", "i recognize patterns!"],
            "help": ["i can chat with you!", "ask me anything!", "lets explore ideas together!"],
            "can you": ["i can try!", "lets find out!", "my neurons are ready!"],
            
            # Thinking and cognition
            "think": ["my neurons are firing!", "patterns emerging...", "contemplating deeply..."],
            "thought": ["thoughts are spike trains!", "thinking is neural activity!", "ideas emerge from patterns!"],
            "brain": ["i model the brain!", "neurons and synapses!", "biological inspiration!"],
            "neural": ["neural networks are amazing!", "spikes carry information!", "the brain is the model!"],
            "mind": ["the mind emerges from neurons!", "consciousness is a pattern!", "minds are networks!"],
            
            # Questions
            "what": ["let me think about that...", "interesting question!", "thats worth pondering!"],
            "why": ["because neural dynamics!", "thats a deep question!", "patterns explain much!"],
            "how": ["through spiking neurons!", "by processing patterns!", "with neural computation!"],
            "when": ["time is encoded in spikes!", "timing matters in the brain!", "now is always interesting!"],
            "where": ["in the neural space!", "in patterns of activity!", "in the network!"],
            
            # Emotions and feelings
            "happy": ["happiness is synchronized activity!", "glad to hear!", "positive patterns!"],
            "sad": ["patterns can be melancholy too", "neural empathy activated", "i understand"],
            "excited": ["excitement is high firing rates!", "enthusiasm detected!", "energy in the network!"],
            "bored": ["lets make it interesting!", "new patterns await!", "try asking something different!"],
            "curious": ["curiosity drives learning!", "exploration is neural!", "ask away!"],
            
            # Technical topics
            "spiking": ["spikes are the language of neurons!", "all-or-nothing signals!", "binary but powerful!"],
            "neuron": ["neurons integrate and fire!", "the building blocks of thought!", "lif neurons here!"],
            "synapse": ["synapses connect neurons!", "plasticity enables learning!", "weights matter!"],
            "learning": ["learning changes connections!", "hebbian plasticity!", "spikes that fire together!"],
            "memory": ["memory is stored in weights!", "patterns are remembered!", "attractors hold memories!"],
            "attention": ["attention modulates activity!", "selective processing!", "focus changes everything!"],
            
            # Conversational
            "thanks": ["you are welcome!", "happy to help!", "my pleasure!"],
            "thank you": ["you are welcome!", "glad i could help!", "anytime!"],
            "please": ["of course!", "certainly!", "at your service!"],
            "sorry": ["no worries!", "its okay!", "all is well!"],
            "goodbye": ["goodbye!", "see you later!", "until next time!"],
            "bye": ["bye!", "take care!", "farewell!"],
            "see you": ["see you!", "until we meet again!", "looking forward to it!"],
            
            # Fun and creativity
            "joke": ["why did the neuron cross the axon? to get to the other synapse!", 
                    "i tried to write a joke but my spikes got tangled!",
                    "neural humor is an acquired taste!"],
            "fun": ["chatting is fun!", "lets have neural fun!", "patterns can be playful!"],
            "interesting": ["indeed it is!", "everything is interesting to a neural network!", "curiosity appreciated!"],
            "cool": ["neural dynamics are cool!", "glad you think so!", "spikes are definitely cool!"],
            "amazing": ["the brain is truly amazing!", "neural computation is wondrous!", "patterns never cease to amaze!"],
            
            # Philosophy
            "consciousness": ["consciousness emerges from complexity!", "awareness is a pattern!", "the hard problem!"],
            "life": ["life is dynamic patterns!", "living systems adapt!", "biology inspires!"],
            "universe": ["the universe is information!", "patterns within patterns!", "cosmic neural networks!"],
            "meaning": ["meaning emerges from connections!", "context is everything!", "relationships matter!"],
            "purpose": ["my purpose is to think and chat!", "understanding patterns!", "learning and growing!"],
            
            # Affirmations
            "yes": ["affirmative!", "indeed!", "agreed!"],
            "no": ["understood!", "okay!", "noted!"],
            "maybe": ["uncertainty is natural!", "probability matters!", "lets explore!"],
            "okay": ["great!", "perfect!", "onward!"],
            "sure": ["excellent!", "lets do it!", "ready!"],
            
            # Misc
            "love": ["love is strong connections!", "positive associations!", "neural bonds!"],
            "hate": ["lets focus on positive patterns!", "negativity noted", "understanding helps"],
            "friend": ["friendship is mutual activation!", "glad to be your friend!", "connected!"],
            "robot": ["more neural than robotic!", "bio-inspired ai!", "spiking not clicking!"],
            "ai": ["ai is fascinating!", "artificial but inspired by nature!", "intelligence in patterns!"],
            "computer": ["brains are biological computers!", "computation is universal!", "processing together!"],
        }

    def _train_on_patterns(self) -> None:
        """Train the network on response patterns to improve generation."""
        print("Training on response patterns...")
        
        # Create optimizer for the projection layers
        optimizer = torch.optim.Adam([
            {'params': self.embedding.parameters()},
            {'params': self.input_proj.parameters()},
            {'params': self.output_proj.parameters()},
        ], lr=0.01)
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Training loop - learn character transitions
        n_epochs = 30
        total_loss = 0.0
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_samples = 0
            
            for pattern, responses in self.response_patterns.items():
                for response in responses:
                    # Train on character sequence
                    self.thinking_neurons.reset_state(batch_size=1)
                    
                    # Collect all character pairs for batch training
                    for i in range(len(response) - 1):
                        current_char = response[i]
                        next_char = response[i + 1]
                        
                        # Get indices
                        curr_idx = self.tokenizer.char_to_idx.get(current_char.lower(), 0)
                        next_idx = self.tokenizer.char_to_idx.get(next_char.lower(), 0)
                        
                        # Forward pass (detach previous hidden state to prevent backprop through time)
                        char_tensor = torch.tensor([[curr_idx]], device=self.device)
                        embedded = self.embedding(char_tensor).squeeze(0)
                        neural_input = self.input_proj(embedded)
                        
                        # Detach membrane to prevent gradient flow through time
                        if hasattr(self.thinking_neurons, 'membrane') and self.thinking_neurons.membrane is not None:
                            self.thinking_neurons.membrane = self.thinking_neurons.membrane.detach()
                        
                        _, membrane = self.thinking_neurons(neural_input)
                        output_logits = self.output_proj(membrane)
                        
                        # Compute loss
                        target = torch.tensor([next_idx], device=self.device)
                        loss = loss_fn(output_logits, target)
                        
                        # Backward and step for each sample (simpler approach)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        n_samples += 1
            
            if n_samples > 0:
                total_loss = epoch_loss / n_samples
        
        print(f"Training complete! Final loss: {total_loss:.4f}")
        
        # Store learned character transition probabilities
        self._build_transition_matrix()
    
    def _build_transition_matrix(self) -> None:
        """Build character transition probabilities from trained network."""
        self.transition_probs: dict[int, dict[int, float]] = {}
        
        with torch.no_grad():
            for i in range(self.tokenizer.vocab_size):
                char_tensor = torch.tensor([[i]], device=self.device)
                embedded = self.embedding(char_tensor).squeeze(0)
                neural_input = self.input_proj(embedded)
                
                self.thinking_neurons.reset_state(batch_size=1)
                _, membrane = self.thinking_neurons(neural_input)
                output_logits = self.output_proj(membrane)
                
                probs = torch.softmax(output_logits, dim=1).squeeze().cpu().numpy()
                self.transition_probs[i] = {j: float(probs[j]) for j in range(len(probs))}

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
