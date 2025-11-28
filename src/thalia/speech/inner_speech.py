"""
Inner Speech - Self-dialogue and verbal reasoning.

This module implements the network's capacity for internal monologue,
enabling it to:

- Generate sequences of "verbal" thoughts
- Reason through problems step-by-step
- Maintain dialogue between different perspectives
- Use language-like sequential processing for deliberation

Inner speech is based on Vygotsky's theory that internal dialogue
develops from externalized speech and becomes a tool for thought.
The network learns to use sequential symbolic representations
to structure and guide its thinking.

Key components:
- Token: Discrete symbolic unit (like a word or concept)
- InnerVoice: Generates sequential token streams
- DialogueManager: Manages multi-voice inner dialogue
- ReasoningChain: Structures step-by-step reasoning
- InnerSpeechNetwork: Full inner speech architecture

References:
- Vygotsky (1934) - Thought and Language
- Fernyhough (2016) - The Voices Within
- Hurlburt & Heavey (2008) - Inner Speech
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable, Iterator
from enum import Enum, auto
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.dynamics.attractor import ConceptNetwork, AttractorConfig


class VoiceType(Enum):
    """Types of inner voices for dialogue."""
    NARRATOR = auto()      # Describes current state
    QUESTIONER = auto()    # Asks questions, challenges
    ANSWERER = auto()      # Provides answers, solutions
    CRITIC = auto()        # Evaluates, criticizes
    SUPPORTER = auto()     # Encourages, supports
    PLANNER = auto()       # Plans next steps


@dataclass
class Token:
    """A discrete symbolic unit in inner speech.

    Tokens are the "words" of inner speech - discrete symbolic
    representations that can be sequenced into thoughts.

    Attributes:
        id: Unique identifier
        name: Human-readable name
        embedding: Neural pattern representing this token
        category: Optional category (noun, verb, concept, etc.)
        associations: Links to related tokens
    """
    id: int
    name: str
    embedding: torch.Tensor
    category: str = "concept"
    associations: Dict[int, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Token({self.id}, '{self.name}')"


@dataclass
class Utterance:
    """A sequence of tokens forming a complete thought.

    Attributes:
        tokens: List of tokens in order
        voice: Which inner voice produced this
        timestamp: When this was generated
        confidence: How confident the network is in this
        is_question: Whether this is a question
    """
    tokens: List[Token]
    voice: VoiceType = VoiceType.NARRATOR
    timestamp: int = 0
    confidence: float = 1.0
    is_question: bool = False

    def to_string(self) -> str:
        """Convert to human-readable string."""
        text = " ".join(t.name for t in self.tokens)
        if self.is_question:
            text += "?"
        return text

    def __repr__(self) -> str:
        return f"[{self.voice.name}]: {self.to_string()}"


@dataclass
class InnerSpeechConfig:
    """Configuration for inner speech system.

    Attributes:
        n_tokens: Size of token vocabulary
        embedding_dim: Dimension of token embeddings
        hidden_dim: Hidden state dimension
        n_voices: Number of inner voices
        max_utterance_length: Maximum tokens per utterance
        temperature: Sampling temperature for generation
        tau_mem: Membrane time constant
        dt: Simulation timestep
    """
    n_tokens: int = 256
    embedding_dim: int = 64
    hidden_dim: int = 128
    n_voices: int = 3
    max_utterance_length: int = 20
    temperature: float = 1.0
    tau_mem: float = 20.0
    dt: float = 1.0


class TokenVocabulary:
    """Manages the vocabulary of tokens for inner speech.

    Example:
        >>> vocab = TokenVocabulary(embedding_dim=64)
        >>> vocab.add_token("think", category="verb")
        >>> vocab.add_token("problem", category="noun")
        >>> token = vocab.get_token("think")
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.tokens: Dict[int, Token] = {}
        self.name_to_id: Dict[str, int] = {}
        self._next_id = 0

        # Special tokens
        self.add_token("<START>", category="special")
        self.add_token("<END>", category="special")
        self.add_token("<PAD>", category="special")
        self.add_token("<UNK>", category="special")

    def add_token(
        self,
        name: str,
        category: str = "concept",
        embedding: Optional[torch.Tensor] = None,
    ) -> Token:
        """Add a token to the vocabulary."""
        if name in self.name_to_id:
            return self.tokens[self.name_to_id[name]]

        token_id = self._next_id
        self._next_id += 1

        if embedding is None:
            embedding = torch.randn(self.embedding_dim) * 0.1

        token = Token(
            id=token_id,
            name=name,
            embedding=embedding,
            category=category,
        )

        self.tokens[token_id] = token
        self.name_to_id[name] = token_id

        return token

    def get_token(self, name_or_id: str | int) -> Optional[Token]:
        """Get a token by name or ID."""
        if isinstance(name_or_id, str):
            token_id = self.name_to_id.get(name_or_id)
            if token_id is None:
                return None
            return self.tokens[token_id]
        return self.tokens.get(name_or_id)

    def get_embedding_matrix(self) -> torch.Tensor:
        """Get matrix of all token embeddings."""
        embeddings = []
        for i in range(len(self.tokens)):
            embeddings.append(self.tokens[i].embedding)
        return torch.stack(embeddings)

    def associate(self, token_a: str | int, token_b: str | int, strength: float = 0.5) -> None:
        """Create association between tokens."""
        t_a = self.get_token(token_a)
        t_b = self.get_token(token_b)
        if t_a and t_b:
            t_a.associations[t_b.id] = strength
            t_b.associations[t_a.id] = strength

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens.values())


class InnerVoice(nn.Module):
    """A single inner voice that generates token sequences.

    Each voice has its own "personality" - biases toward certain
    types of utterances and reasoning styles.

    Example:
        >>> voice = InnerVoice(vocab, VoiceType.QUESTIONER)
        >>> voice.reset_state(batch_size=1)
        >>>
        >>> # Generate a question
        >>> utterance = voice.generate(max_length=10)
        >>> print(utterance)
    """

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        voice_type: VoiceType = VoiceType.NARRATOR,
        hidden_dim: int = 128,
        tau_mem: float = 20.0,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.voice_type = voice_type
        self.hidden_dim = hidden_dim

        n_tokens = len(vocabulary)
        embedding_dim = vocabulary.embedding_dim

        # SNN for sequence generation
        neuron_config = LIFConfig(tau_mem=tau_mem, noise_std=0.02)
        self.neurons = LIFNeuron(n_neurons=hidden_dim, config=neuron_config)

        # Input projection (embedding → hidden)
        self.input_proj = nn.Parameter(
            torch.randn(embedding_dim, hidden_dim) * 0.1 / (embedding_dim ** 0.5)
        )

        # Recurrent weights
        self.recurrent = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.register_buffer("recurrent_mask", 1 - torch.eye(hidden_dim))

        # Output projection (hidden → token logits)
        self.output_proj = nn.Parameter(
            torch.randn(hidden_dim, n_tokens) * 0.1 / (hidden_dim ** 0.5)
        )

        # Voice-specific bias
        self.voice_bias = nn.Parameter(torch.zeros(n_tokens))

        # State
        self._last_hidden: Optional[torch.Tensor] = None
        self._generated_tokens: List[Token] = []
        self._timestep = 0

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset voice state."""
        self.neurons.reset_state(batch_size)
        self._last_hidden = None
        self._generated_tokens = []
        self._timestep = 0

    def _get_device(self) -> torch.device:
        return self.input_proj.device

    def step(
        self,
        input_token: Optional[Token] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one step and return next token distribution.

        Args:
            input_token: Previous token (for autoregressive generation)
            context: Optional context vector to condition on

        Returns:
            logits: Token logits
            hidden: Hidden state
        """
        device = self._get_device()
        batch_size = 1

        if self.neurons.membrane is not None:
            batch_size = self.neurons.membrane.shape[0]
        else:
            self.reset_state(batch_size)

        # Get input embedding
        if input_token is not None:
            embedding = input_token.embedding.to(device).unsqueeze(0)
        else:
            # Use start token
            start_token = self.vocabulary.get_token("<START>")
            embedding = start_token.embedding.to(device).unsqueeze(0) if start_token else torch.zeros(1, self.vocabulary.embedding_dim, device=device)

        # Project to hidden
        hidden_input = torch.matmul(embedding, self.input_proj)

        # Add context if provided
        if context is not None:
            # Project context to hidden size
            if context.shape[-1] != self.hidden_dim:
                context = F.pad(context, (0, self.hidden_dim - context.shape[-1]))[:, :self.hidden_dim]
            hidden_input = hidden_input + context * 0.5

        # Add recurrent input
        if self._last_hidden is not None:
            recurrent_input = torch.matmul(
                self._last_hidden,
                self.recurrent * self.recurrent_mask
            )
            hidden_input = hidden_input + recurrent_input

        # Step neurons
        spikes, membrane = self.neurons(hidden_input)

        # Use membrane potential for continuous output
        hidden = torch.sigmoid(membrane)
        self._last_hidden = hidden

        # Compute token logits
        logits = torch.matmul(hidden, self.output_proj) + self.voice_bias

        self._timestep += 1

        return logits, hidden

    def sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Optional[Token]:
        """Sample a token from logits."""
        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None and top_k < logits.shape[-1]:
            values, indices = torch.topk(logits, top_k, dim=-1)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, indices, values)

        # Sample
        probs = F.softmax(logits, dim=-1)
        token_id = int(torch.multinomial(probs.squeeze(0), 1).item())

        return self.vocabulary.get_token(token_id)

    def generate(
        self,
        max_length: int = 20,
        temperature: float = 1.0,
        context: Optional[torch.Tensor] = None,
        stop_tokens: Optional[List[str]] = None,
    ) -> Utterance:
        """Generate a complete utterance.

        Args:
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            context: Optional context to condition on
            stop_tokens: Token names that end generation

        Returns:
            Generated utterance
        """
        if stop_tokens is None:
            stop_tokens = ["<END>", "."]

        self._generated_tokens = []
        current_token = None

        for _ in range(max_length):
            logits, _ = self.step(current_token, context)
            token = self.sample_token(logits, temperature)

            if token is None or token.name in stop_tokens:
                break

            if token.name not in ["<START>", "<PAD>"]:
                self._generated_tokens.append(token)

            current_token = token

        # Determine if this is a question
        is_question = any(t.name in ["?", "why", "how", "what", "when", "where"]
                         for t in self._generated_tokens)

        return Utterance(
            tokens=list(self._generated_tokens),
            voice=self.voice_type,
            timestamp=self._timestep,
            is_question=is_question,
        )


class DialogueManager(nn.Module):
    """Manages multi-voice inner dialogue.

    Coordinates multiple inner voices to have a dialogue,
    passing context and responses between them.

    Example:
        >>> manager = DialogueManager(vocab, n_voices=3)
        >>>
        >>> # Start a dialogue
        >>> manager.start_dialogue("What should I do?")
        >>>
        >>> # Generate dialogue turns
        >>> for _ in range(5):
        ...     utterance = manager.next_turn()
        ...     print(utterance)
    """

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        n_voices: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.n_voices = n_voices

        # Create voices with different types
        voice_types = list(VoiceType)[:n_voices]
        if len(voice_types) < n_voices:
            voice_types = voice_types * (n_voices // len(voice_types) + 1)
            voice_types = voice_types[:n_voices]

        self.voices = nn.ModuleList([
            InnerVoice(vocabulary, vt, hidden_dim)
            for vt in voice_types
        ])

        # Shared context
        self.register_buffer(
            "shared_context",
            torch.zeros(1, hidden_dim)
        )

        # Dialogue history
        self._history: List[Utterance] = []
        self._current_voice_idx = 0
        self._turn_count = 0

    def reset(self) -> None:
        """Reset dialogue state."""
        for voice in self.voices:
            voice.reset_state(batch_size=1)
        self.shared_context.zero_()
        self._history = []
        self._current_voice_idx = 0
        self._turn_count = 0

    def start_dialogue(self, prompt: Optional[str] = None) -> None:
        """Start a new dialogue, optionally with a prompt."""
        self.reset()

        if prompt:
            # Parse prompt into tokens
            words = prompt.lower().split()
            tokens = []
            for word in words:
                token = self.vocabulary.get_token(word)
                if token is None:
                    token = self.vocabulary.add_token(word)
                tokens.append(token)

            # Create initial utterance
            initial = Utterance(
                tokens=tokens,
                voice=VoiceType.NARRATOR,
                timestamp=0,
                is_question="?" in prompt,
            )
            self._history.append(initial)

            # Update context from prompt
            if tokens:
                embeddings = torch.stack([t.embedding for t in tokens])
                self.shared_context = embeddings.mean(dim=0, keepdim=True)

    def _select_next_voice(self) -> int:
        """Select which voice speaks next."""
        # Simple round-robin with some randomness
        if self._history and self._history[-1].is_question:
            # After a question, prefer ANSWERER
            for i, voice in enumerate(self.voices):
                if voice.voice_type == VoiceType.ANSWERER:
                    return i

        # Otherwise, rotate with some randomness
        if random.random() < 0.3:
            return random.randint(0, len(self.voices) - 1)
        else:
            return (self._current_voice_idx + 1) % len(self.voices)

    def next_turn(
        self,
        temperature: float = 1.0,
        max_length: int = 15,
    ) -> Utterance:
        """Generate the next dialogue turn."""
        # Select voice
        voice_idx = self._select_next_voice()
        voice = self.voices[voice_idx]
        self._current_voice_idx = voice_idx

        # Generate utterance
        voice.reset_state(batch_size=1)
        utterance = voice.generate(
            max_length=max_length,
            temperature=temperature,
            context=self.shared_context,
        )
        utterance.timestamp = self._turn_count

        # Update history and context
        self._history.append(utterance)
        self._turn_count += 1

        # Update shared context
        if utterance.tokens:
            device = self.shared_context.device
            embeddings = torch.stack([t.embedding.to(device) for t in utterance.tokens])
            new_context = embeddings.mean(dim=0, keepdim=True)
            # Project to hidden_dim if needed
            if new_context.shape[-1] != self.shared_context.shape[-1]:
                # Pad or truncate to match hidden_dim
                hidden_dim = self.shared_context.shape[-1]
                if new_context.shape[-1] < hidden_dim:
                    new_context = F.pad(new_context, (0, hidden_dim - new_context.shape[-1]))
                else:
                    new_context = new_context[:, :hidden_dim]
            # Blend with previous context
            self.shared_context = 0.7 * self.shared_context + 0.3 * new_context

        return utterance

    def get_history(self) -> List[Utterance]:
        """Get full dialogue history."""
        return list(self._history)

    def run_dialogue(
        self,
        n_turns: int,
        prompt: Optional[str] = None,
    ) -> List[Utterance]:
        """Run a complete dialogue session."""
        self.start_dialogue(prompt)

        for _ in range(n_turns):
            self.next_turn()

        return self.get_history()


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain.

    Attributes:
        step_number: Position in chain
        premise: Input/premise for this step
        operation: Reasoning operation applied
        conclusion: Result of this step
        confidence: Confidence in this step
    """
    step_number: int
    premise: Utterance
    operation: str
    conclusion: Utterance
    confidence: float = 1.0

    def __repr__(self) -> str:
        return f"Step {self.step_number}: {self.premise.to_string()} → [{self.operation}] → {self.conclusion.to_string()}"


class ReasoningChain:
    """Structures step-by-step reasoning through inner speech.

    Implements chain-of-thought reasoning where each step
    builds on the previous, with explicit operations.

    Example:
        >>> chain = ReasoningChain(voice)
        >>> chain.add_premise("The sky is blue")
        >>> chain.apply_operation("infer")
        >>> chain.apply_operation("conclude")
        >>> print(chain.get_conclusion())
    """

    OPERATIONS = [
        "observe",    # Direct observation
        "recall",     # Retrieve from memory
        "infer",      # Draw inference
        "compare",    # Compare concepts
        "evaluate",   # Evaluate truth/value
        "hypothesize", # Form hypothesis
        "conclude",   # Draw conclusion
    ]

    def __init__(self, voice: InnerVoice):
        self.voice = voice
        self.steps: List[ReasoningStep] = []
        self._current_premise: Optional[Utterance] = None

    def reset(self) -> None:
        """Reset reasoning chain."""
        self.steps = []
        self._current_premise = None
        self.voice.reset_state(batch_size=1)

    def add_premise(self, premise: str | Utterance) -> None:
        """Add initial premise to reason from."""
        if isinstance(premise, str):
            # Convert string to utterance
            tokens = []
            for word in premise.lower().split():
                token = self.voice.vocabulary.get_token(word)
                if token is None:
                    token = self.voice.vocabulary.add_token(word)
                tokens.append(token)
            premise = Utterance(tokens=tokens)

        self._current_premise = premise

    def apply_operation(
        self,
        operation: str,
        temperature: float = 0.8,
    ) -> ReasoningStep:
        """Apply a reasoning operation to current state.

        Args:
            operation: One of OPERATIONS
            temperature: Generation temperature

        Returns:
            The reasoning step produced
        """
        if operation not in self.OPERATIONS:
            operation = "infer"  # Default

        if self._current_premise is None:
            raise ValueError("No premise set. Call add_premise first.")

        # Build context from premise
        device = self.voice._get_device()
        if self._current_premise.tokens:
            embeddings = torch.stack([t.embedding.to(device) for t in self._current_premise.tokens])
            context = embeddings.mean(dim=0, keepdim=True)
        else:
            context = None

        # Generate conclusion
        self.voice.reset_state(batch_size=1)
        conclusion = self.voice.generate(
            max_length=15,
            temperature=temperature,
            context=context,
        )

        # Compute confidence (heuristic)
        confidence = 1.0 / (1 + len(self.steps) * 0.1)  # Decreases with chain length

        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            premise=self._current_premise,
            operation=operation,
            conclusion=conclusion,
            confidence=confidence,
        )

        self.steps.append(step)
        self._current_premise = conclusion  # Chain forward

        return step

    def reason(
        self,
        premise: str,
        operations: List[str],
        temperature: float = 0.8,
    ) -> List[ReasoningStep]:
        """Run a complete reasoning chain.

        Args:
            premise: Starting premise
            operations: Sequence of operations to apply
            temperature: Generation temperature

        Returns:
            All reasoning steps
        """
        self.reset()
        self.add_premise(premise)

        for op in operations:
            self.apply_operation(op, temperature)

        return self.steps

    def get_conclusion(self) -> Optional[Utterance]:
        """Get final conclusion of reasoning chain."""
        if not self.steps:
            return None
        return self.steps[-1].conclusion

    def get_chain_string(self) -> str:
        """Get human-readable chain representation."""
        lines = []
        for step in self.steps:
            lines.append(str(step))
        return "\n".join(lines)


class InnerSpeechNetwork(nn.Module):
    """Complete inner speech network.

    Integrates vocabulary, voices, dialogue, and reasoning
    into a unified inner speech system.

    Example:
        >>> network = InnerSpeechNetwork(InnerSpeechConfig())
        >>>
        >>> # Add vocabulary
        >>> network.add_words(["think", "problem", "solution", "try", "maybe"])
        >>>
        >>> # Generate inner monologue
        >>> monologue = network.think_aloud(steps=5)
        >>>
        >>> # Reason about something
        >>> conclusion = network.reason_about("The problem is complex")
    """

    def __init__(self, config: Optional[InnerSpeechConfig] = None):
        super().__init__()
        self.config = config or InnerSpeechConfig()

        # Vocabulary
        self.vocabulary = TokenVocabulary(self.config.embedding_dim)

        # Dialogue manager
        self.dialogue = DialogueManager(
            self.vocabulary,
            n_voices=self.config.n_voices,
            hidden_dim=self.config.hidden_dim,
        )

        # Primary voice for monologue
        self.primary_voice = InnerVoice(
            self.vocabulary,
            VoiceType.NARRATOR,
            self.config.hidden_dim,
            self.config.tau_mem,
        )

        # Reasoning engine
        self.reasoning = ReasoningChain(self.primary_voice)

        # State
        self._monologue_history: List[Utterance] = []
        self._timestep = 0

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all state."""
        self.primary_voice.reset_state(batch_size)
        self.dialogue.reset()
        self.reasoning.reset()
        self._monologue_history = []
        self._timestep = 0

    def add_word(self, word: str, category: str = "concept") -> Token:
        """Add a word to vocabulary."""
        return self.vocabulary.add_token(word, category)

    def add_words(self, words: List[str], category: str = "concept") -> List[Token]:
        """Add multiple words to vocabulary."""
        return [self.add_word(w, category) for w in words]

    def associate_words(self, word_a: str, word_b: str, strength: float = 0.5) -> None:
        """Create association between words."""
        self.vocabulary.associate(word_a, word_b, strength)

    def speak(
        self,
        context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Utterance:
        """Generate one utterance of inner speech.

        Args:
            context: Optional context to condition on
            temperature: Sampling temperature

        Returns:
            Generated utterance
        """
        temp = temperature or self.config.temperature

        self.primary_voice.reset_state(batch_size=1)
        utterance = self.primary_voice.generate(
            max_length=self.config.max_utterance_length,
            temperature=temp,
            context=context,
        )
        utterance.timestamp = self._timestep

        self._monologue_history.append(utterance)
        self._timestep += 1

        return utterance

    def think_aloud(
        self,
        steps: int = 5,
        temperature: Optional[float] = None,
    ) -> List[Utterance]:
        """Generate a stream of inner speech.

        Args:
            steps: Number of utterances to generate
            temperature: Sampling temperature

        Returns:
            List of generated utterances
        """
        utterances = []
        context = None

        for _ in range(steps):
            utterance = self.speak(context, temperature)
            utterances.append(utterance)

            # Build context from last utterance
            if utterance.tokens:
                device = self.primary_voice._get_device()
                embeddings = torch.stack([t.embedding.to(device) for t in utterance.tokens])
                context = embeddings.mean(dim=0, keepdim=True)

        return utterances

    def have_dialogue(
        self,
        prompt: Optional[str] = None,
        n_turns: int = 5,
    ) -> List[Utterance]:
        """Have an inner dialogue between voices.

        Args:
            prompt: Optional starting prompt
            n_turns: Number of dialogue turns

        Returns:
            Dialogue history
        """
        return self.dialogue.run_dialogue(n_turns, prompt)

    def reason_about(
        self,
        premise: str,
        depth: int = 3,
        temperature: float = 0.8,
    ) -> Utterance:
        """Reason about a premise through chain of thought.

        Args:
            premise: Starting premise
            depth: Number of reasoning steps
            temperature: Generation temperature

        Returns:
            Final conclusion
        """
        # Build operation sequence
        operations = []
        if depth >= 1:
            operations.append("observe")
        if depth >= 2:
            operations.append("infer")
        if depth >= 3:
            operations.extend(["evaluate", "conclude"])
        else:
            operations.append("conclude")

        self.reasoning.reason(premise, operations, temperature)
        return self.reasoning.get_conclusion()

    def get_reasoning_chain(self) -> str:
        """Get the current reasoning chain as string."""
        return self.reasoning.get_chain_string()

    def get_monologue_history(self) -> List[Utterance]:
        """Get history of inner monologue."""
        return list(self._monologue_history)

    def stream_of_consciousness(
        self,
        duration: int = 100,
        temperature: float = 1.2,
    ) -> Iterator[Utterance]:
        """Generate a continuous stream of inner speech.

        Higher temperature for more free-flowing, creative stream.

        Args:
            duration: Number of utterances
            temperature: Sampling temperature (higher = more random)

        Yields:
            Utterances one at a time
        """
        context = None

        for _ in range(duration):
            self.primary_voice.reset_state(batch_size=1)
            utterance = self.primary_voice.generate(
                max_length=self.config.max_utterance_length,
                temperature=temperature,
                context=context,
            )

            yield utterance

            # Update context
            if utterance.tokens:
                device = self.primary_voice._get_device()
                embeddings = torch.stack([t.embedding.to(device) for t in utterance.tokens])
                context = embeddings.mean(dim=0, keepdim=True)
