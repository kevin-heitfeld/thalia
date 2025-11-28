"""Tests for the Inner Speech module."""

import pytest
import torch
import torch.nn as nn

from thalia.speech import (
    Token,
    VoiceType,
    TokenVocabulary,
    InnerVoice,
    Utterance,
    ReasoningStep,
    ReasoningChain,
    DialogueManager,
    InnerSpeechConfig,
    InnerSpeechNetwork,
)


class TestToken:
    """Tests for Token dataclass."""
    
    def test_token_creation(self):
        """Test basic token creation."""
        embedding = torch.randn(64)
        token = Token(
            id=0,
            name="hello",
            embedding=embedding,
            category="word",
        )
        
        assert token.name == "hello"
        assert token.id == 0
        assert torch.equal(token.embedding, embedding)
        assert token.category == "word"
        
    def test_token_with_associations(self):
        """Test token with associations."""
        token = Token(
            id=1,
            name="dog",
            embedding=torch.randn(64),
            category="concept",
            associations={2: 0.9, 3: 0.8},
        )
        
        assert 2 in token.associations
        assert token.associations[2] == 0.9
        
    def test_token_repr(self):
        """Test token string representation."""
        token = Token(
            id=5,
            name="test",
            embedding=torch.randn(64),
        )
        
        assert "Token(5" in repr(token)
        assert "'test'" in repr(token)


class TestTokenVocabulary:
    """Tests for TokenVocabulary class."""
    
    def test_vocabulary_creation(self):
        """Test vocabulary initialization."""
        vocab = TokenVocabulary(embedding_dim=64)
        
        assert vocab.embedding_dim == 64
        # Should have special tokens
        assert len(vocab) >= 4  # START, END, PAD, UNK
        
    def test_add_and_get_token(self):
        """Test adding and retrieving tokens."""
        vocab = TokenVocabulary(embedding_dim=64)
        
        token = vocab.add_token("hello", category="word")
        
        assert token.name == "hello"
        assert token.embedding.shape == (64,)
        
        retrieved = vocab.get_token(token.id)
        assert retrieved is not None
        assert retrieved.name == "hello"
        
    def test_get_token_by_name(self):
        """Test getting token by name."""
        vocab = TokenVocabulary(embedding_dim=64)
        vocab.add_token("world", category="word")
        
        token = vocab.get_token("world")
        assert token is not None
        assert token.name == "world"
        
        missing = vocab.get_token("nonexistent")
        assert missing is None
        
    def test_add_token_with_custom_embedding(self):
        """Test adding token with custom embedding."""
        vocab = TokenVocabulary(embedding_dim=64)
        embedding = torch.randn(64)
        
        token = vocab.add_token("custom", category="concept", embedding=embedding)
        
        assert torch.allclose(token.embedding, embedding)
        
    def test_get_embedding_matrix(self):
        """Test getting all embeddings as matrix."""
        vocab = TokenVocabulary(embedding_dim=64)
        
        # Add some tokens (already has 4 special tokens)
        for i in range(6):
            vocab.add_token(f"word_{i}", category="word")
            
        embeddings = vocab.get_embedding_matrix()
        
        assert embeddings.shape == (len(vocab), 64)
        
    def test_vocabulary_len(self):
        """Test vocabulary length."""
        vocab = TokenVocabulary(embedding_dim=64)
        initial_len = len(vocab)
        
        vocab.add_token("a", category="word")
        vocab.add_token("b", category="word")
        
        assert len(vocab) == initial_len + 2
        
    def test_vocabulary_iteration(self):
        """Test iterating over vocabulary."""
        vocab = TokenVocabulary(embedding_dim=64)
        
        for name in ["x", "y", "z"]:
            vocab.add_token(name, category="word")
            
        names = [token.name for token in vocab]
        assert "x" in names
        assert "y" in names
        assert "z" in names
        
    def test_token_associations(self):
        """Test creating associations between tokens."""
        vocab = TokenVocabulary(embedding_dim=64)
        
        vocab.add_token("cat", category="noun")
        vocab.add_token("dog", category="noun")
        
        vocab.associate("cat", "dog", strength=0.8)
        
        cat = vocab.get_token("cat")
        dog = vocab.get_token("dog")
        
        assert dog.id in cat.associations
        assert cat.id in dog.associations
        
    def test_duplicate_token_returns_existing(self):
        """Test that adding duplicate token returns existing."""
        vocab = TokenVocabulary(embedding_dim=64)
        
        token1 = vocab.add_token("duplicate", category="word")
        token2 = vocab.add_token("duplicate", category="word")
        
        assert token1.id == token2.id


class TestVoiceType:
    """Tests for VoiceType enum."""
    
    def test_voice_types_exist(self):
        """Test all voice types are defined."""
        assert VoiceType.NARRATOR is not None
        assert VoiceType.QUESTIONER is not None
        assert VoiceType.ANSWERER is not None
        assert VoiceType.CRITIC is not None
        assert VoiceType.SUPPORTER is not None
        assert VoiceType.PLANNER is not None


class TestUtterance:
    """Tests for Utterance dataclass."""
    
    def test_utterance_creation(self):
        """Test utterance initialization."""
        tokens = [
            Token(0, "hello", torch.randn(64), "word"),
            Token(1, "world", torch.randn(64), "word"),
        ]
        
        utterance = Utterance(
            tokens=tokens,
            voice=VoiceType.NARRATOR,
            timestamp=10,
            confidence=0.9,
        )
        
        assert len(utterance.tokens) == 2
        assert utterance.timestamp == 10
        assert utterance.confidence == 0.9
        assert utterance.voice == VoiceType.NARRATOR
        
    def test_utterance_to_string(self):
        """Test converting utterance to text."""
        tokens = [
            Token(0, "hello", torch.randn(64), "word"),
            Token(1, "world", torch.randn(64), "word"),
        ]
        
        utterance = Utterance(tokens=tokens)
        
        assert utterance.to_string() == "hello world"
        
    def test_question_utterance(self):
        """Test question utterance."""
        tokens = [
            Token(0, "why", torch.randn(64), "word"),
        ]
        
        utterance = Utterance(tokens=tokens, is_question=True)
        
        assert utterance.to_string() == "why?"
        
    def test_empty_utterance(self):
        """Test empty utterance."""
        utterance = Utterance(tokens=[])
        
        assert len(utterance.tokens) == 0
        assert utterance.to_string() == ""


class TestInnerVoice:
    """Tests for InnerVoice class."""
    
    @pytest.fixture
    def vocabulary(self):
        """Create a test vocabulary."""
        vocab = TokenVocabulary(embedding_dim=64)
        words = ["the", "dog", "runs", "fast", "home", "is", "happy"]
        for word in words:
            vocab.add_token(word, category="word")
        return vocab
        
    def test_inner_voice_creation(self, vocabulary):
        """Test inner voice initialization."""
        voice = InnerVoice(
            vocabulary=vocabulary,
            voice_type=VoiceType.NARRATOR,
            hidden_dim=128,
        )
        
        assert voice.hidden_dim == 128
        assert voice.voice_type == VoiceType.NARRATOR
        
    def test_reset_state(self, vocabulary):
        """Test state reset."""
        voice = InnerVoice(vocabulary=vocabulary, hidden_dim=128)
        
        voice.reset_state(batch_size=1)
        
        # Should be able to step after reset
        logits, hidden = voice.step()
        assert logits is not None
        
    def test_step(self, vocabulary):
        """Test single step."""
        voice = InnerVoice(vocabulary=vocabulary, hidden_dim=128)
        voice.reset_state(batch_size=1)
        
        logits, hidden = voice.step()
        
        assert logits.shape[-1] == len(vocabulary)
        assert hidden.shape[-1] == 128
        
    def test_step_with_token(self, vocabulary):
        """Test step with input token."""
        voice = InnerVoice(vocabulary=vocabulary, hidden_dim=128)
        voice.reset_state(batch_size=1)
        
        token = vocabulary.get_token("dog")
        logits, hidden = voice.step(input_token=token)
        
        assert logits.shape[-1] == len(vocabulary)
        
    def test_sample_token(self, vocabulary):
        """Test token sampling."""
        voice = InnerVoice(vocabulary=vocabulary, hidden_dim=128)
        voice.reset_state(batch_size=1)
        
        logits, _ = voice.step()
        token = voice.sample_token(logits, temperature=1.0)
        
        assert token is not None
        assert isinstance(token, Token)
        assert 0 <= token.id < len(vocabulary)
        
    def test_sample_token_with_temperature(self, vocabulary):
        """Test temperature affects sampling."""
        voice = InnerVoice(vocabulary=vocabulary, hidden_dim=128)
        
        # Create logits with clear preference
        logits = torch.zeros(1, len(vocabulary))
        logits[0, 5] = 10.0  # Strong preference for token 5
        
        # Low temperature should consistently pick highest
        samples_low = []
        for _ in range(10):
            token = voice.sample_token(logits.clone(), temperature=0.01)
            if token:
                samples_low.append(token.id)
            
        # Most samples should be token 5
        assert samples_low.count(5) > 5
        
    def test_sample_token_top_k(self, vocabulary):
        """Test top-k sampling."""
        voice = InnerVoice(vocabulary=vocabulary, hidden_dim=128)
        
        logits = torch.randn(1, len(vocabulary))
        
        # Get top-3 token indices
        _, top_indices = torch.topk(logits.squeeze(0), 3)
        top_set = set(top_indices.tolist())
        
        # Top-k=3 should restrict choices to those 3
        samples = []
        for _ in range(20):
            token = voice.sample_token(logits.clone(), top_k=3)
            if token:
                samples.append(token.id)
            
        # All samples should be in top-3
        for sample in samples:
            assert sample in top_set
        
    def test_generate(self, vocabulary):
        """Test generating complete utterance."""
        voice = InnerVoice(vocabulary=vocabulary, hidden_dim=128)
        voice.reset_state(batch_size=1)
        
        utterance = voice.generate(max_length=10)
        
        assert isinstance(utterance, Utterance)
        assert len(utterance.tokens) <= 10
        
    def test_generate_with_context(self, vocabulary):
        """Test generation with context."""
        voice = InnerVoice(vocabulary=vocabulary, hidden_dim=128)
        voice.reset_state(batch_size=1)
        
        context = torch.randn(1, 128)
        utterance = voice.generate(max_length=5, context=context)
        
        assert isinstance(utterance, Utterance)


class TestReasoningStep:
    """Tests for ReasoningStep class."""
    
    def test_reasoning_step_creation(self):
        """Test reasoning step initialization."""
        premise = Utterance(tokens=[Token(0, "all", torch.randn(64), "word")])
        conclusion = Utterance(tokens=[Token(1, "therefore", torch.randn(64), "word")])
        
        step = ReasoningStep(
            step_number=1,
            premise=premise,
            operation="infer",
            conclusion=conclusion,
            confidence=0.95,
        )
        
        assert step.step_number == 1
        assert step.operation == "infer"
        assert step.confidence == 0.95
        
    def test_reasoning_step_repr(self):
        """Test reasoning step string representation."""
        premise = Utterance(tokens=[Token(0, "A", torch.randn(64), "word")])
        conclusion = Utterance(tokens=[Token(1, "B", torch.randn(64), "word")])
        
        step = ReasoningStep(
            step_number=1,
            premise=premise,
            operation="infer",
            conclusion=conclusion,
        )
        
        repr_str = repr(step)
        assert "Step 1" in repr_str
        assert "infer" in repr_str


class TestReasoningChain:
    """Tests for ReasoningChain class."""
    
    @pytest.fixture
    def voice(self):
        """Create a test voice."""
        vocab = TokenVocabulary(embedding_dim=64)
        for word in ["think", "therefore", "because", "if", "then"]:
            vocab.add_token(word, category="word")
        return InnerVoice(vocab, VoiceType.NARRATOR, hidden_dim=128)
        
    def test_chain_creation(self, voice):
        """Test reasoning chain initialization."""
        chain = ReasoningChain(voice)
        
        assert chain.voice is voice
        assert len(chain.steps) == 0
        
    def test_add_premise(self, voice):
        """Test adding premise."""
        chain = ReasoningChain(voice)
        
        chain.add_premise("The sky is blue")
        
        assert chain._current_premise is not None
        
    def test_apply_operation(self, voice):
        """Test applying reasoning operation."""
        chain = ReasoningChain(voice)
        
        chain.add_premise("The sky is blue")
        step = chain.apply_operation("observe")
        
        assert isinstance(step, ReasoningStep)
        assert step.operation == "observe"
        assert len(chain.steps) == 1
        
    def test_reason_complete_chain(self, voice):
        """Test complete reasoning chain."""
        chain = ReasoningChain(voice)
        
        steps = chain.reason(
            premise="It is raining",
            operations=["observe", "infer", "conclude"],
            temperature=0.8,
        )
        
        assert len(steps) == 3
        assert steps[0].operation == "observe"
        assert steps[1].operation == "infer"
        assert steps[2].operation == "conclude"
        
    def test_get_conclusion(self, voice):
        """Test getting final conclusion."""
        chain = ReasoningChain(voice)
        
        chain.reason(
            premise="Test premise",
            operations=["infer", "conclude"],
        )
        
        conclusion = chain.get_conclusion()
        
        assert conclusion is not None
        assert isinstance(conclusion, Utterance)
        
    def test_empty_chain_conclusion(self, voice):
        """Test empty chain has no conclusion."""
        chain = ReasoningChain(voice)
        
        conclusion = chain.get_conclusion()
        
        assert conclusion is None
        
    def test_reset(self, voice):
        """Test chain reset."""
        chain = ReasoningChain(voice)
        
        chain.reason("Test", ["infer"])
        chain.reset()
        
        assert len(chain.steps) == 0
        assert chain._current_premise is None
        
    def test_get_chain_string(self, voice):
        """Test getting chain as string."""
        chain = ReasoningChain(voice)
        
        chain.reason("Premise", ["observe", "infer"])
        
        chain_str = chain.get_chain_string()
        
        assert "Step 1" in chain_str
        assert "Step 2" in chain_str
        
    def test_valid_operations(self, voice):
        """Test that all valid operations work."""
        chain = ReasoningChain(voice)
        
        for op in ReasoningChain.OPERATIONS:
            chain.reset()
            chain.add_premise("Test")
            step = chain.apply_operation(op)
            assert step.operation == op


class TestDialogueManager:
    """Tests for DialogueManager class."""
    
    @pytest.fixture
    def vocabulary(self):
        """Create a test vocabulary."""
        vocab = TokenVocabulary(embedding_dim=64)
        words = ["yes", "no", "maybe", "but", "and", "because", "therefore", "why", "how"]
        for word in words:
            vocab.add_token(word, category="word")
        return vocab
        
    def test_dialogue_creation(self, vocabulary):
        """Test dialogue initialization."""
        dialogue = DialogueManager(
            vocabulary=vocabulary,
            n_voices=3,
            hidden_dim=128,
        )
        
        assert len(dialogue.voices) == 3
        
    def test_reset(self, vocabulary):
        """Test dialogue reset."""
        dialogue = DialogueManager(vocabulary=vocabulary, n_voices=2)
        
        # Generate some turns
        dialogue.start_dialogue("Test")
        dialogue.next_turn()
        
        dialogue.reset()
        
        assert len(dialogue._history) == 0
        assert dialogue._turn_count == 0
        
    def test_start_dialogue_with_prompt(self, vocabulary):
        """Test starting dialogue with prompt."""
        dialogue = DialogueManager(vocabulary=vocabulary)
        
        dialogue.start_dialogue("What should I do?")
        
        assert len(dialogue._history) == 1
        assert dialogue._history[0].is_question
        
    def test_next_turn(self, vocabulary):
        """Test generating next turn."""
        dialogue = DialogueManager(vocabulary=vocabulary, n_voices=2)
        dialogue.start_dialogue()
        
        utterance = dialogue.next_turn(max_length=5)
        
        assert isinstance(utterance, Utterance)
        assert dialogue._turn_count == 1
        
    def test_run_dialogue(self, vocabulary):
        """Test running complete dialogue."""
        dialogue = DialogueManager(vocabulary=vocabulary, n_voices=2)
        
        history = dialogue.run_dialogue(n_turns=4, prompt="Hello")
        
        # Should have prompt + 4 turns
        assert len(history) >= 4
        
    def test_get_history(self, vocabulary):
        """Test getting dialogue history."""
        dialogue = DialogueManager(vocabulary=vocabulary)
        
        dialogue.run_dialogue(n_turns=3)
        
        history = dialogue.get_history()
        
        assert isinstance(history, list)
        assert all(isinstance(u, Utterance) for u in history)


class TestInnerSpeechConfig:
    """Tests for InnerSpeechConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = InnerSpeechConfig()
        
        assert config.n_tokens == 256
        assert config.embedding_dim == 64
        assert config.hidden_dim == 128
        assert config.n_voices == 3
        assert config.max_utterance_length == 20
        assert config.temperature == 1.0
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = InnerSpeechConfig(
            n_tokens=512,
            hidden_dim=256,
            n_voices=5,
        )
        
        assert config.n_tokens == 512
        assert config.hidden_dim == 256
        assert config.n_voices == 5


class TestInnerSpeechNetwork:
    """Tests for InnerSpeechNetwork class."""
    
    def test_network_creation(self):
        """Test network initialization."""
        config = InnerSpeechConfig(hidden_dim=128)
        network = InnerSpeechNetwork(config)
        
        assert network.vocabulary is not None
        assert network.dialogue is not None
        assert network.primary_voice is not None
        assert network.reasoning is not None
        
    def test_add_word(self):
        """Test adding words to vocabulary."""
        network = InnerSpeechNetwork()
        
        token = network.add_word("hello", category="greeting")
        
        assert token.name == "hello"
        assert token.category == "greeting"
        
    def test_add_words(self):
        """Test adding multiple words."""
        network = InnerSpeechNetwork()
        
        tokens = network.add_words(["one", "two", "three"])
        
        assert len(tokens) == 3
        assert all(isinstance(t, Token) for t in tokens)
        
    def test_associate_words(self):
        """Test word associations."""
        network = InnerSpeechNetwork()
        
        network.add_words(["cat", "dog"])
        network.associate_words("cat", "dog", strength=0.8)
        
        cat = network.vocabulary.get_token("cat")
        dog = network.vocabulary.get_token("dog")
        
        assert dog.id in cat.associations
        
    def test_speak(self):
        """Test generating single utterance."""
        network = InnerSpeechNetwork()
        network.add_words(["think", "about", "this"])
        
        utterance = network.speak()
        
        assert isinstance(utterance, Utterance)
        
    def test_speak_with_context(self):
        """Test speaking with context."""
        network = InnerSpeechNetwork(InnerSpeechConfig(hidden_dim=128))
        network.add_words(["context", "test"])
        
        context = torch.randn(1, 128)
        utterance = network.speak(context=context)
        
        assert isinstance(utterance, Utterance)
        
    def test_think_aloud(self):
        """Test stream of inner speech."""
        network = InnerSpeechNetwork()
        network.add_words(["maybe", "perhaps", "therefore"])
        
        utterances = network.think_aloud(steps=3)
        
        assert len(utterances) == 3
        assert all(isinstance(u, Utterance) for u in utterances)
        
    def test_have_dialogue(self):
        """Test inner dialogue."""
        network = InnerSpeechNetwork()
        network.add_words(["yes", "no", "why"])
        
        history = network.have_dialogue(n_turns=3)
        
        assert len(history) >= 3
        
    def test_reason_about(self):
        """Test reasoning about premise."""
        network = InnerSpeechNetwork()
        network.add_words(["if", "then", "therefore", "because"])
        
        conclusion = network.reason_about("The problem is complex", depth=2)
        
        assert conclusion is not None
        assert isinstance(conclusion, Utterance)
        
    def test_get_reasoning_chain(self):
        """Test getting reasoning chain string."""
        network = InnerSpeechNetwork()
        network.add_words(["test", "words"])
        
        network.reason_about("Test premise", depth=2)
        chain_str = network.get_reasoning_chain()
        
        assert isinstance(chain_str, str)
        assert "Step" in chain_str
        
    def test_get_monologue_history(self):
        """Test getting monologue history."""
        network = InnerSpeechNetwork()
        network.add_words(["a", "b", "c"])
        
        network.think_aloud(steps=3)
        history = network.get_monologue_history()
        
        assert len(history) == 3
        
    def test_stream_of_consciousness(self):
        """Test stream of consciousness generator."""
        network = InnerSpeechNetwork()
        network.add_words(["flow", "of", "thoughts"])
        
        stream = list(network.stream_of_consciousness(duration=5))
        
        assert len(stream) == 5
        assert all(isinstance(u, Utterance) for u in stream)
        
    def test_reset_state(self):
        """Test full state reset."""
        network = InnerSpeechNetwork()
        network.add_words(["test"])
        
        network.think_aloud(steps=2)
        network.reset_state()
        
        assert len(network._monologue_history) == 0


class TestIntegration:
    """Integration tests for the inner speech system."""
    
    def test_full_reasoning_pipeline(self):
        """Test complete reasoning from vocabulary to conclusion."""
        config = InnerSpeechConfig(hidden_dim=128)
        network = InnerSpeechNetwork(config)
        
        # Add vocabulary
        network.add_words([
            "if", "then", "and", "or", "not", "true", "false",
            "therefore", "because", "premise", "conclusion",
        ])
        
        # Run reasoning
        conclusion = network.reason_about(
            "If it rains then the ground is wet",
            depth=3,
        )
        
        assert conclusion is not None
        
        # Check chain
        chain_str = network.get_reasoning_chain()
        assert "Step 1" in chain_str
        
    def test_dialogue_to_reasoning(self):
        """Test combining dialogue and reasoning."""
        network = InnerSpeechNetwork()
        network.add_words([
            "what", "should", "I", "do", "think", "consider", "maybe",
        ])
        
        # Have a dialogue
        dialogue_history = network.have_dialogue("What should I do?", n_turns=2)
        
        # Then reason about it
        conclusion = network.reason_about("I should think carefully", depth=2)
        
        assert len(dialogue_history) >= 2
        assert conclusion is not None
        
    def test_gpu_compatibility(self):
        """Test that components work on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = torch.device("cuda")
        
        network = InnerSpeechNetwork()
        network.add_words(["gpu", "test", "words"])
        network = network.to(device)
        
        # Generate on GPU
        utterance = network.speak()
        assert isinstance(utterance, Utterance)
        
    def test_multiple_sessions(self):
        """Test multiple separate sessions."""
        network = InnerSpeechNetwork()
        network.add_words(["session", "one", "two"])
        
        # Session 1
        network.think_aloud(steps=2)
        history1_len = len(network.get_monologue_history())
        
        # Reset for session 2
        network.reset_state()
        
        network.think_aloud(steps=3)
        history2_len = len(network.get_monologue_history())
        
        assert history1_len == 2
        assert history2_len == 3  # Fresh after reset
        
    def test_vocabulary_persistence(self):
        """Test that vocabulary persists across resets."""
        network = InnerSpeechNetwork()
        
        network.add_words(["persist", "across", "reset"])
        initial_vocab_size = len(network.vocabulary)
        
        network.reset_state()
        
        # Vocabulary should still be there
        assert len(network.vocabulary) == initial_vocab_size
        assert network.vocabulary.get_token("persist") is not None
