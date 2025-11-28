"""
Inner Speech Demo - THALIA's Verbal Reasoning System

This demo showcases the inner speech capabilities:
1. Token vocabulary and associations
2. Inner voice generation
3. Multi-voice dialogue
4. Chain-of-thought reasoning
5. Stream of consciousness

Inner speech enables the network to "think in words" - generating
sequential verbal patterns that structure reasoning and planning.
"""

import torch
from thalia.speech import (
    TokenVocabulary,
    InnerVoice,
    VoiceType,
    DialogueManager,
    ReasoningChain,
    InnerSpeechConfig,
    InnerSpeechNetwork,
)


def demo_vocabulary():
    """Demonstrate vocabulary creation and associations."""
    print("=" * 60)
    print("1. TOKEN VOCABULARY")
    print("=" * 60)

    vocab = TokenVocabulary(embedding_dim=64)

    # Add some words
    words = ["think", "problem", "solution", "maybe", "because",
             "therefore", "if", "then", "and", "or", "not"]

    print(f"\nAdding vocabulary: {words}")
    for word in words:
        token = vocab.add_token(word, category="word")
        print(f"  Added: {token}")

    # Add concepts
    concepts = ["logic", "planning", "decision", "goal"]
    print(f"\nAdding concepts: {concepts}")
    for concept in concepts:
        vocab.add_token(concept, category="concept")

    # Create associations
    print("\nCreating semantic associations...")
    vocab.associate("think", "logic")
    vocab.associate("problem", "solution")
    vocab.associate("if", "then")
    vocab.associate("goal", "planning")

    # Show associations
    token = vocab.get_token("think")
    print(f"\nToken 'think' associations: {token.associations}")

    print(f"\nTotal vocabulary size: {len(vocab)}")

    return vocab


def demo_inner_voice(vocab):
    """Demonstrate single voice generation."""
    print("\n" + "=" * 60)
    print("2. INNER VOICE GENERATION")
    print("=" * 60)

    # Create narrator voice
    voice = InnerVoice(
        vocabulary=vocab,
        voice_type=VoiceType.NARRATOR,
        hidden_dim=128,
    )

    print(f"\nCreated voice: {voice.voice_type.name}")
    print("Generating utterances...\n")

    # Generate several utterances
    for i in range(3):
        voice.reset_state(batch_size=1)
        utterance = voice.generate(
            max_length=8,
            temperature=0.9,
        )
        print(f"  Utterance {i+1}: {utterance}")

    # Generate with lower temperature (more focused)
    print("\nWith lower temperature (more focused):")
    voice.reset_state(batch_size=1)
    utterance = voice.generate(max_length=8, temperature=0.3)
    print(f"  {utterance}")

    # Generate with higher temperature (more creative)
    print("\nWith higher temperature (more creative):")
    voice.reset_state(batch_size=1)
    utterance = voice.generate(max_length=8, temperature=1.5)
    print(f"  {utterance}")


def demo_dialogue(vocab):
    """Demonstrate multi-voice inner dialogue."""
    print("\n" + "=" * 60)
    print("3. MULTI-VOICE INNER DIALOGUE")
    print("=" * 60)

    # Create dialogue manager
    dialogue = DialogueManager(
        vocabulary=vocab,
        n_voices=3,
        hidden_dim=128,
    )

    print("\nVoices in dialogue:")
    for i, voice in enumerate(dialogue.voices):
        print(f"  Voice {i}: {voice.voice_type.name}")

    # Run a dialogue session
    print("\n--- Dialogue Session ---")
    history = dialogue.run_dialogue(
        n_turns=5,
        prompt="What should I do about this problem?",
    )

    for utterance in history:
        print(f"  {utterance}")

    print("\n--- Second Session (without prompt) ---")
    dialogue.reset()
    history = dialogue.run_dialogue(n_turns=4)

    for utterance in history:
        print(f"  {utterance}")


def demo_reasoning(vocab):
    """Demonstrate chain-of-thought reasoning."""
    print("\n" + "=" * 60)
    print("4. CHAIN-OF-THOUGHT REASONING")
    print("=" * 60)

    # Create a voice for reasoning
    voice = InnerVoice(
        vocabulary=vocab,
        voice_type=VoiceType.NARRATOR,
        hidden_dim=128,
    )

    # Create reasoning chain
    chain = ReasoningChain(voice)

    print("\nAvailable reasoning operations:")
    for op in ReasoningChain.OPERATIONS:
        print(f"  - {op}")

    # Run a reasoning chain
    print("\n--- Reasoning Chain ---")
    print("Premise: 'If the goal is clear then planning helps'")

    steps = chain.reason(
        premise="If the goal is clear then planning helps",
        operations=["observe", "infer", "evaluate", "conclude"],
        temperature=0.7,
    )

    print("\nReasoning steps:")
    for step in steps:
        print(f"  {step}")

    # Get conclusion
    conclusion = chain.get_conclusion()
    print(f"\nFinal conclusion: {conclusion.to_string()}")

    # Show the full chain
    print("\n--- Full Chain Representation ---")
    print(chain.get_chain_string())


def demo_full_network():
    """Demonstrate the complete InnerSpeechNetwork."""
    print("\n" + "=" * 60)
    print("5. COMPLETE INNER SPEECH NETWORK")
    print("=" * 60)

    # Create network with custom config
    config = InnerSpeechConfig(
        embedding_dim=64,
        hidden_dim=128,
        n_voices=3,
        max_utterance_length=12,
        temperature=0.9,
    )

    network = InnerSpeechNetwork(config)

    # Build vocabulary
    print("\nBuilding vocabulary...")
    network.add_words([
        # Basic words
        "think", "know", "believe", "want", "need",
        # Logical connectives
        "if", "then", "and", "or", "not", "but",
        # Reasoning words
        "because", "therefore", "maybe", "probably",
        # Action words
        "try", "do", "make", "find", "solve",
        # Concepts
        "problem", "solution", "goal", "plan", "idea",
        # Question words
        "what", "why", "how", "when", "where",
    ])

    # Create associations
    network.associate_words("think", "know")
    network.associate_words("problem", "solution")
    network.associate_words("if", "then")
    network.associate_words("goal", "plan")
    network.associate_words("try", "do")

    print(f"Vocabulary size: {len(network.vocabulary)}")

    # Generate inner speech
    print("\n--- Inner Monologue ---")
    utterances = network.think_aloud(steps=4)
    for i, u in enumerate(utterances):
        print(f"  {i+1}. {u.to_string()}")

    # Have a dialogue
    print("\n--- Inner Dialogue ---")
    dialogue = network.have_dialogue(
        prompt="What should I try next?",
        n_turns=4,
    )
    for u in dialogue:
        print(f"  {u}")

    # Reason about something
    print("\n--- Verbal Reasoning ---")
    network.reason_about(
        premise="The problem needs a creative solution",
        depth=3,
    )
    print(network.get_reasoning_chain())


def demo_stream_of_consciousness():
    """Demonstrate stream of consciousness mode."""
    print("\n" + "=" * 60)
    print("6. STREAM OF CONSCIOUSNESS")
    print("=" * 60)

    # Create a network
    network = InnerSpeechNetwork()

    # Add a larger vocabulary for more varied stream
    words = [
        "flow", "drift", "wander", "wonder", "dream",
        "see", "feel", "sense", "notice", "observe",
        "remember", "imagine", "create", "explore", "discover",
        "light", "dark", "color", "sound", "silence",
        "here", "there", "now", "then", "always", "never",
        "perhaps", "maybe", "somehow", "somewhere",
    ]
    network.add_words(words)

    print("\nGenerating stream of consciousness (high temperature)...")
    print("(Higher randomness for free-flowing thoughts)\n")

    # Generate stream
    stream = network.stream_of_consciousness(
        duration=8,
        temperature=1.3,  # Higher for more creative flow
    )

    print("Stream:")
    for utterance in stream:
        text = utterance.to_string()
        if text:
            print(f"  ... {text} ...")


def demo_gpu():
    """Demonstrate GPU acceleration if available."""
    print("\n" + "=" * 60)
    print("7. GPU ACCELERATION")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nUsing GPU: {torch.cuda.get_device_name()}")

        # Create network on GPU
        network = InnerSpeechNetwork()
        network.add_words(["think", "fast", "compute"])
        network = network.to(device)

        # Time generation
        import time

        # Warmup
        network.speak()

        start = time.perf_counter()
        for _ in range(50):
            network.speak()
        elapsed = time.perf_counter() - start

        print(f"Generated 50 utterances in {elapsed:.3f}s")
        print(f"Rate: {50/elapsed:.1f} utterances/second")
    else:
        print("\nNo GPU available, using CPU")
        print("(GPU would accelerate generation)")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("   THALIA INNER SPEECH DEMONSTRATION")
    print("   Self-Dialogue and Verbal Reasoning")
    print("=" * 60)

    # Run demos
    vocab = demo_vocabulary()
    demo_inner_voice(vocab)
    demo_dialogue(vocab)
    demo_reasoning(vocab)
    demo_full_network()
    demo_stream_of_consciousness()
    demo_gpu()

    print("\n" + "=" * 60)
    print("   DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nInner speech enables the network to:")
    print("  • Generate verbal thought sequences")
    print("  • Conduct internal dialogue between perspectives")
    print("  • Structure chain-of-thought reasoning")
    print("  • Produce stream of consciousness")
    print("\nThis forms the basis for verbal reasoning and")
    print("metacognitive monitoring in THALIA.")
    print()


if __name__ == "__main__":
    main()
