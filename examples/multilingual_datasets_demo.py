"""
Multilingual Datasets Demo

Demonstrates the multilingual support for Grammar and Reading datasets.
Shows examples in English, German, and Spanish.
"""

import torch
from thalia.datasets import (
    create_stage2_grammar_dataset,
    create_stage3_reading_dataset,
    GrammarLanguage,
    ReadingLanguage,
)


def demo_grammar_multilingual():
    """Demo grammar dataset in all three languages."""
    print("=" * 60)
    print("Grammar Dataset - Multilingual Support")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    languages = [
        (GrammarLanguage.ENGLISH, "English"),
        (GrammarLanguage.GERMAN, "German"),
        (GrammarLanguage.SPANISH, "Spanish"),
    ]
    
    for lang_enum, lang_name in languages:
        print(f"\n--- {lang_name} ---")
        dataset = create_stage2_grammar_dataset(
            device=device,
            language=lang_enum,
        )
        
        print(f"Vocabulary size: {dataset.vocab.vocab_size}")
        print(f"Sample nouns (singular): {dataset.vocab.nouns_singular[:5]}")
        print(f"Sample verbs (singular): {dataset.vocab.verbs_singular[:3]}")
        print(f"Sample adjectives: {dataset.vocab.adjectives[:3]}")
        
        # Generate example phrases
        print(f"\n{lang_name} Example Phrases:")
        for i in range(3):
            phrase_indices, is_gram, rule = dataset.generate_phrase()
            phrase_words = dataset.vocab.decode(phrase_indices)
            status = "✓ Grammatical" if is_gram else "✗ Violation"
            print(f"  {i+1}. {' '.join(phrase_words)} [{status}] ({rule.value})")


def demo_reading_multilingual():
    """Demo reading dataset in all three languages."""
    print("\n" + "=" * 60)
    print("Reading Dataset - Multilingual Support")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    languages = [
        (ReadingLanguage.ENGLISH, "English"),
        (ReadingLanguage.GERMAN, "German"),
        (ReadingLanguage.SPANISH, "Spanish"),
    ]
    
    for lang_enum, lang_name in languages:
        print(f"\n--- {lang_name} ---")
        dataset = create_stage3_reading_dataset(
            device=device,
            language=lang_enum,
        )
        
        print(f"Vocabulary size: {dataset.vocab.vocab_size}")
        print(f"Phoneme inventory size: {dataset.vocab.n_phonemes}")
        
        # Show some example words with phonemes
        print(f"\n{lang_name} Word Examples (with IPA phonemes):")
        word_samples = list(dataset.vocab.nouns.items())[:3]
        for word, phonemes in word_samples:
            phoneme_str = ' '.join(phonemes)
            print(f"  {word}: /{phoneme_str}/")
        
        # Show sample task
        from thalia.datasets import ReadingTask
        task_data, label, task_type = dataset.generate_task(
            task_type=ReadingTask.SENTENCE_COMPLETION
        )
        print(f"\n{lang_name} Sample Task (Sentence Completion):")
        sentence_indices = task_data['sentence'].cpu().tolist()
        sentence_words = []
        for idx in sentence_indices:
            if idx == dataset.vocab.pad_idx:
                break
            word = dataset.vocab.idx2word.get(idx, '<UNK>')
            sentence_words.append(word)
        print(f"  Sentence: {' '.join(sentence_words)}")
        print(f"  Missing position: {task_data['missing_position'].item()}")
        answer_word = dataset.vocab.idx2word.get(label.item(), '<UNK>')
        print(f"  Answer: {answer_word}")


def demo_language_comparison():
    """Compare the same grammar rule across languages."""
    print("\n" + "=" * 60)
    print("Cross-Linguistic Comparison")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from thalia.datasets import GrammarRule
    
    print("\nSubject-Verb Agreement Across Languages:")
    print("-" * 60)
    
    languages = [
        (GrammarLanguage.ENGLISH, "English"),
        (GrammarLanguage.GERMAN, "German (Deutsch)"),
        (GrammarLanguage.SPANISH, "Spanish (Español)"),
    ]
    
    for lang_enum, lang_name in languages:
        dataset = create_stage2_grammar_dataset(
            device=device,
            language=lang_enum,
        )
        
        print(f"\n{lang_name}:")
        
        # Generate SV agreement examples
        for _ in range(2):
            phrase_indices, is_gram, rule = dataset.generate_phrase(
                rule=GrammarRule.SUBJECT_VERB_AGREEMENT
            )
            if is_gram:  # Show grammatical examples
                phrase_words = dataset.vocab.decode(phrase_indices)
                print(f"  • {' '.join(phrase_words)}")


def demo_phoneme_differences():
    """Show phoneme inventory differences across languages."""
    print("\n" + "=" * 60)
    print("Phoneme Inventory Comparison")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    languages = [
        (ReadingLanguage.ENGLISH, "English"),
        (ReadingLanguage.GERMAN, "German"),
        (ReadingLanguage.SPANISH, "Spanish"),
    ]
    
    for lang_enum, lang_name in languages:
        dataset = create_stage3_reading_dataset(
            device=device,
            language=lang_enum,
        )
        
        print(f"\n{lang_name}:")
        print(f"  Total phonemes: {dataset.vocab.n_phonemes}")
        print(f"  Phoneme inventory: {' '.join(dataset.vocab.phonemes[:20])}...")
        
        # Show unique phonemes (language-specific)
        if lang_enum == ReadingLanguage.GERMAN:
            print(f"  Unique: /x/ (ach-Laut), /ç/ (ich-Laut), /ʏ/ (ü), /ø/ (ö)")
        elif lang_enum == ReadingLanguage.SPANISH:
            print(f"  Unique: /ɲ/ (ñ), /x/ (j), /β/ (soft b/v), /ð/ (soft d)")
        elif lang_enum == ReadingLanguage.ENGLISH:
            print(f"  Unique: /θ/ (th), /ð/ (th-voiced), /ʃ/ (sh), /dʒ/ (j)")


def demo_curriculum_training_multilingual():
    """Show how to use multilingual datasets in curriculum training."""
    print("\n" + "=" * 60)
    print("Multilingual Curriculum Training Example")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nScenario: Train on English, then test on German and Spanish")
    print("-" * 60)
    
    # Create datasets for all languages
    grammar_en = create_stage2_grammar_dataset(device, language=GrammarLanguage.ENGLISH)
    grammar_de = create_stage2_grammar_dataset(device, language=GrammarLanguage.GERMAN)
    grammar_es = create_stage2_grammar_dataset(device, language=GrammarLanguage.SPANISH)
    
    reading_en = create_stage3_reading_dataset(device, language=ReadingLanguage.ENGLISH)
    reading_de = create_stage3_reading_dataset(device, language=ReadingLanguage.GERMAN)
    reading_es = create_stage3_reading_dataset(device, language=ReadingLanguage.SPANISH)
    
    print("\nDatasets created:")
    print(f"  ✓ Grammar:  English ({grammar_en.vocab.vocab_size} words), "
          f"German ({grammar_de.vocab.vocab_size} words), "
          f"Spanish ({grammar_es.vocab.vocab_size} words)")
    print(f"  ✓ Reading:  English ({reading_en.vocab.vocab_size} words), "
          f"German ({reading_de.vocab.vocab_size} words), "
          f"Spanish ({reading_es.vocab_size} words)")
    
    print("\nSuggested training approach:")
    print("""
    # Stage 2: Grammar Training (Sequential)
    # Week 1-4: English grammar
    for step in range(40000):
        phrase, is_gram, rule = grammar_en.generate_phrase()
        # ... train brain
    
    # Week 5-8: German grammar (transfer learning)
    for step in range(40000):
        phrase, is_gram, rule = grammar_de.generate_phrase()
        # ... train brain (should be faster due to transfer)
    
    # Week 9-12: Spanish grammar
    for step in range(40000):
        phrase, is_gram, rule = grammar_es.generate_phrase()
        # ... train brain
    
    # Stage 3: Reading Training (Interleaved for better transfer)
    for step in range(120000):
        # 60% English, 20% German, 20% Spanish
        lang_roll = np.random.random()
        if lang_roll < 0.6:
            task_data, label, task_type = reading_en.generate_task()
        elif lang_roll < 0.8:
            task_data, label, task_type = reading_de.generate_task()
        else:
            task_data, label, task_type = reading_es.generate_task()
        # ... train brain
    """)
    
    print("\nExpected benefits of multilingual training:")
    print("  ✓ Better grammatical abstraction (cross-linguistic transfer)")
    print("  ✓ Richer phoneme representations")
    print("  ✓ More robust language models")
    print("  ✓ Matches human multilingual acquisition patterns")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Multilingual Dataset Demo")
    print("=" * 60)
    print("\nDemonstrating English, German, and Spanish support\n")
    
    try:
        demo_grammar_multilingual()
        demo_reading_multilingual()
        demo_language_comparison()
        demo_phoneme_differences()
        demo_curriculum_training_multilingual()
        
        print("\n" + "=" * 60)
        print("✅ All multilingual features working correctly!")
        print("=" * 60)
        print("\nKey features:")
        print("  • Grammar datasets in 3 languages")
        print("  • Reading datasets with language-specific phonemes")
        print("  • Easy language switching via Language enum")
        print("  • Ready for multilingual curriculum training")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
