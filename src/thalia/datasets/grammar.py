"""
Grammar Dataset for Stage 2 (Grammar & Composition)

Generates grammar learning tasks:
- Subject-verb agreement (The cat runs / The cats run)
- Noun-adjective composition (blue ball, red car)
- Word order rules (SVO, SOV for multilingual)
- Simple phrase structure (NP + VP)
- Agreement violations for error detection

Biologically relevant:
- Tests compositional language learning
- Engages Broca's area (grammar processing)
- Critical period for grammar (2-7 years)
- Prepares for sentence-level processing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch


class Language(Enum):
    """Supported languages for grammar tasks."""
    ENGLISH = "en"
    GERMAN = "de"
    SPANISH = "es"


class GrammarRule(Enum):
    """Types of grammar rules to test."""
    SUBJECT_VERB_AGREEMENT = "sv_agreement"
    NOUN_ADJECTIVE = "noun_adj"
    WORD_ORDER_SVO = "word_order_svo"
    WORD_ORDER_SOV = "word_order_sov"
    PLURAL_MORPHOLOGY = "plural_morph"
    TENSE_MORPHOLOGY = "tense_morph"


class AgreementType(Enum):
    """Subject-verb agreement types."""
    SINGULAR = "singular"
    PLURAL = "plural"


@dataclass
class GrammarConfig:
    """Configuration for grammar dataset."""
    vocab_size: int = 100  # Total vocabulary size
    max_phrase_length: int = 5  # Maximum words per phrase
    rules_to_test: Optional[List[GrammarRule]] = None
    violation_probability: float = 0.2  # Probability of grammatical error
    embedding_dim: int = 64  # Embedding dimension
    multilingual: bool = False  # Include SOV word order
    language: Language = Language.ENGLISH  # Language to use
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        if self.rules_to_test is None:
            self.rules_to_test = [
                GrammarRule.SUBJECT_VERB_AGREEMENT,
                GrammarRule.NOUN_ADJECTIVE,
                GrammarRule.WORD_ORDER_SVO,
            ]


class GrammarVocabulary:
    """Vocabulary for grammar tasks with multilingual support."""

    def __init__(self, language: Language = Language.ENGLISH):
        self.language = language

        if language == Language.ENGLISH:
            self._init_english()
        elif language == Language.GERMAN:
            self._init_german()
        elif language == Language.SPANISH:
            self._init_spanish()
        else:
            raise ValueError(f"Unsupported language: {language}")

        # Build word→index mapping
        self.word2idx = {}
        self.idx2word = {}

        idx = 0
        for word_list in [
            self.nouns_singular, self.nouns_plural,
            self.verbs_singular, self.verbs_plural,
            self.adjectives,
            self.determiners_singular, self.determiners_plural,
        ]:
            for word in word_list:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

        # Special tokens
        self.word2idx['<PAD>'] = idx
        self.idx2word[idx] = '<PAD>'
        self.pad_idx = idx

        self.vocab_size = len(self.word2idx)

    def _init_english(self):
        """Initialize English vocabulary."""
        # Nouns (subjects/objects)
        self.nouns_singular = ['cat', 'dog', 'bird', 'fish', 'mouse', 'ball', 'book', 'car']
        self.nouns_plural = ['cats', 'dogs', 'birds', 'fish', 'mice', 'balls', 'books', 'cars']

        # Verbs
        self.verbs_singular = ['runs', 'jumps', 'flies', 'swims', 'eats']
        self.verbs_plural = ['run', 'jump', 'fly', 'swim', 'eat']

        # Adjectives
        self.adjectives = ['big', 'small', 'red', 'blue', 'fast', 'slow', 'happy', 'sad']

        # Determiners
        self.determiners_singular = ['the', 'a', 'this']
        self.determiners_plural = ['the', 'these', 'some']

    def _init_german(self):
        """Initialize German vocabulary."""
        # Nouns (with grammatical gender markers)
        # Note: German nouns are capitalized
        self.nouns_singular = ['Katze', 'Hund', 'Vogel', 'Fisch', 'Maus', 'Ball', 'Buch', 'Auto']
        self.nouns_plural = ['Katzen', 'Hunde', 'Vögel', 'Fische', 'Mäuse', 'Bälle', 'Bücher', 'Autos']

        # Verbs (3rd person singular vs plural)
        self.verbs_singular = ['läuft', 'springt', 'fliegt', 'schwimmt', 'isst']
        self.verbs_plural = ['laufen', 'springen', 'fliegen', 'schwimmen', 'essen']

        # Adjectives (uninflected predicative form)
        self.adjectives = ['groß', 'klein', 'rot', 'blau', 'schnell', 'langsam', 'glücklich', 'traurig']

        # Determiners (der/die/das for singular, die for plural)
        self.determiners_singular = ['der', 'die', 'das', 'ein', 'eine']
        self.determiners_plural = ['die', 'diese', 'einige']

    def _init_spanish(self):
        """Initialize Spanish vocabulary."""
        # Nouns (masculine/feminine)
        self.nouns_singular = ['gato', 'perro', 'pájaro', 'pez', 'ratón', 'pelota', 'libro', 'coche']
        self.nouns_plural = ['gatos', 'perros', 'pájaros', 'peces', 'ratones', 'pelotas', 'libros', 'coches']

        # Verbs (3rd person singular vs plural)
        self.verbs_singular = ['corre', 'salta', 'vuela', 'nada', 'come']
        self.verbs_plural = ['corren', 'saltan', 'vuelan', 'nadan', 'comen']

        # Adjectives (need gender agreement in attributive position, but we'll use masculine)
        self.adjectives = ['grande', 'pequeño', 'rojo', 'azul', 'rápido', 'lento', 'feliz', 'triste']

        # Determiners (el/la for singular, los/las for plural)
        self.determiners_singular = ['el', 'la', 'un', 'una', 'este', 'esta']
        self.determiners_plural = ['los', 'las', 'estos', 'estas', 'algunos']

        # Build word→index mapping
        self.word2idx = {}
        self.idx2word = {}

        idx = 0
        for word_list in [
            self.nouns_singular, self.nouns_plural,
            self.verbs_singular, self.verbs_plural,
            self.adjectives,
            self.determiners_singular, self.determiners_plural,
        ]:
            for word in word_list:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

        # Special tokens
        self.word2idx['<PAD>'] = idx
        self.idx2word[idx] = '<PAD>'
        self.pad_idx = idx

        self.vocab_size = len(self.word2idx)

    def encode(self, words: List[str]) -> List[int]:
        """Convert words to indices."""
        return [self.word2idx[w] for w in words]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to words."""
        return [self.idx2word[i] for i in indices]


class GrammarDataset:
    """
    Grammar learning dataset for Stage 2.

    Generates grammatical and ungrammatical phrases for:
    - Agreement learning (subject-verb, noun-adjective)
    - Word order learning (SVO, optionally SOV)
    - Morphology learning (plurals, tense)

    Usage:
        >>> config = GrammarConfig(max_phrase_length=5)
        >>> dataset = GrammarDataset(config)
        >>> phrase, label, rule = dataset.generate_phrase()
        >>> # phrase: List[int] (word indices)
        >>> # label: bool (True=grammatical, False=violation)
        >>> # rule: GrammarRule that was tested
    """

    def __init__(self, config: GrammarConfig):
        self.config = config
        self.vocab = GrammarVocabulary(language=config.language)

        # Generators for each rule type
        self.rule_generators = {
            GrammarRule.SUBJECT_VERB_AGREEMENT: self._generate_sv_agreement,
            GrammarRule.NOUN_ADJECTIVE: self._generate_noun_adj,
            GrammarRule.WORD_ORDER_SVO: self._generate_word_order_svo,
            GrammarRule.WORD_ORDER_SOV: self._generate_word_order_sov,
            GrammarRule.PLURAL_MORPHOLOGY: self._generate_plural_morph,
        }

    def generate_phrase(
        self,
        rule: Optional[GrammarRule] = None,
        force_violation: bool = False,
    ) -> Tuple[List[int], bool, GrammarRule]:
        """
        Generate single phrase.

        Args:
            rule: Specific rule to test (random if None)
            force_violation: Force grammatical violation

        Returns:
            phrase: List of word indices
            is_grammatical: True if grammatical, False if violation
            rule: Grammar rule that was tested
        """
        if rule is None:
            rule = np.random.choice(self.config.rules_to_test)

        # Generate base phrase
        phrase_words, is_grammatical = self.rule_generators[rule]()

        # Maybe add violation
        if force_violation or (not is_grammatical):
            is_grammatical = False

        # Convert to indices
        phrase_indices = self.vocab.encode(phrase_words)

        return phrase_indices, is_grammatical, rule

    def generate_batch(
        self,
        batch_size: int,
        balance_rules: bool = True,
        balance_violations: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[GrammarRule]]:
        """
        Generate batch of phrases.

        Args:
            batch_size: Number of phrases
            balance_rules: Equal distribution of rules
            balance_violations: 50% grammatical, 50% violations

        Returns:
            phrases: (batch_size, max_length) padded word indices
            labels: (batch_size,) grammatical (1) or violation (0)
            rules: List of grammar rules tested
        """
        phrases = []
        labels = []
        rules = []

        for _ in range(batch_size):
            # Maybe force violation for balance
            force_violation = False
            if balance_violations and np.random.random() < 0.5:
                force_violation = True

            phrase, is_gram, rule = self.generate_phrase(force_violation=force_violation)
            phrases.append(phrase)
            labels.append(1 if is_gram else 0)
            rules.append(rule)

        # Pad phrases to same length
        max_len = max(len(p) for p in phrases)
        padded = []
        for phrase in phrases:
            padded_phrase = phrase + [self.vocab.pad_idx] * (max_len - len(phrase))
            padded.append(padded_phrase)

        return (
            torch.tensor(padded, dtype=torch.long, device=self.config.device),
            torch.tensor(labels, dtype=torch.float, device=self.config.device),
            rules,
        )

    def _generate_sv_agreement(self) -> Tuple[List[str], bool]:
        """
        Generate subject-verb agreement phrase.

        Returns:
            phrase: ["the", "cat", "runs"] or ["the", "cats", "run"]
            is_grammatical: True unless forced violation
        """
        # Choose singular or plural
        is_plural = np.random.random() < 0.5

        # Build phrase
        if is_plural:
            det = np.random.choice(self.vocab.determiners_plural)
            noun = np.random.choice(self.vocab.nouns_plural)
            verb = np.random.choice(self.vocab.verbs_plural)
        else:
            det = np.random.choice(self.vocab.determiners_singular)
            noun = np.random.choice(self.vocab.nouns_singular)
            verb = np.random.choice(self.vocab.verbs_singular)

        # Maybe introduce violation
        is_grammatical = True
        if np.random.random() < self.config.violation_probability:
            # Swap verb number
            if is_plural:
                verb = np.random.choice(self.vocab.verbs_singular)
            else:
                verb = np.random.choice(self.vocab.verbs_plural)
            is_grammatical = False

        return [det, noun, verb], is_grammatical

    def _generate_noun_adj(self) -> Tuple[List[str], bool]:
        """
        Generate noun-adjective phrase.

        Returns:
            phrase: ["the", "big", "cat"] (adjective before noun in English)
            is_grammatical: True unless forced violation
        """
        is_plural = np.random.random() < 0.5

        if is_plural:
            det = np.random.choice(self.vocab.determiners_plural)
            noun = np.random.choice(self.vocab.nouns_plural)
        else:
            det = np.random.choice(self.vocab.determiners_singular)
            noun = np.random.choice(self.vocab.nouns_singular)

        adj = np.random.choice(self.vocab.adjectives)

        # Correct order: det + adj + noun
        phrase = [det, adj, noun]
        is_grammatical = True

        # Maybe introduce word order violation
        if np.random.random() < self.config.violation_probability:
            # Wrong order: det + noun + adj
            phrase = [det, noun, adj]
            is_grammatical = False

        return phrase, is_grammatical

    def _generate_word_order_svo(self) -> Tuple[List[str], bool]:
        """
        Generate SVO (Subject-Verb-Object) phrase.

        Returns:
            phrase: ["the", "cat", "eats", "the", "fish"]
            is_grammatical: True unless forced violation
        """
        # Subject
        subj_det = np.random.choice(self.vocab.determiners_singular)
        subj_noun = np.random.choice(self.vocab.nouns_singular)

        # Verb
        verb = np.random.choice(self.vocab.verbs_singular)

        # Object
        obj_det = np.random.choice(self.vocab.determiners_singular)
        obj_noun = np.random.choice(self.vocab.nouns_singular)

        # Correct SVO order
        phrase = [subj_det, subj_noun, verb, obj_det, obj_noun]
        is_grammatical = True

        # Maybe introduce word order violation (SOV or OSV)
        if np.random.random() < self.config.violation_probability:
            if np.random.random() < 0.5:
                # SOV: subject + object + verb
                phrase = [subj_det, subj_noun, obj_det, obj_noun, verb]
            else:
                # OSV: object + subject + verb
                phrase = [obj_det, obj_noun, subj_det, subj_noun, verb]
            is_grammatical = False

        return phrase, is_grammatical

    def _generate_word_order_sov(self) -> Tuple[List[str], bool]:
        """
        Generate SOV (Subject-Object-Verb) phrase.

        For multilingual curriculum (Japanese, Korean, Turkish, etc.)

        Returns:
            phrase: ["the", "cat", "the", "fish", "eats"]
            is_grammatical: True unless forced violation
        """
        # Subject
        subj_det = np.random.choice(self.vocab.determiners_singular)
        subj_noun = np.random.choice(self.vocab.nouns_singular)

        # Object
        obj_det = np.random.choice(self.vocab.determiners_singular)
        obj_noun = np.random.choice(self.vocab.nouns_singular)

        # Verb
        verb = np.random.choice(self.vocab.verbs_singular)

        # Correct SOV order
        phrase = [subj_det, subj_noun, obj_det, obj_noun, verb]
        is_grammatical = True

        # Maybe introduce violation (SVO)
        if np.random.random() < self.config.violation_probability:
            # SVO: subject + verb + object
            phrase = [subj_det, subj_noun, verb, obj_det, obj_noun]
            is_grammatical = False

        return phrase, is_grammatical

    def _generate_plural_morph(self) -> Tuple[List[str], bool]:
        """
        Generate phrase testing plural morphology.

        Returns:
            phrase: ["the", "cats"] or ["these", "cat"] (violation)
            is_grammatical: True unless forced violation
        """
        is_plural = np.random.random() < 0.5

        if is_plural:
            det = np.random.choice(self.vocab.determiners_plural)
            noun = np.random.choice(self.vocab.nouns_plural)
        else:
            det = np.random.choice(self.vocab.determiners_singular)
            noun = np.random.choice(self.vocab.nouns_singular)

        phrase = [det, noun]
        is_grammatical = True

        # Maybe mismatch determiner and noun number
        if np.random.random() < self.config.violation_probability:
            if is_plural:
                noun = np.random.choice(self.vocab.nouns_singular)
            else:
                noun = np.random.choice(self.vocab.nouns_plural)
            phrase = [det, noun]
            is_grammatical = False

        return phrase, is_grammatical

    def compute_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute grammaticality judgment accuracy.

        Args:
            predictions: (batch_size,) predicted grammaticality (0 or 1)
            labels: (batch_size,) true grammaticality (0 or 1)

        Returns:
            accuracy: Fraction correct
        """
        pred_binary = (predictions > 0.5).float()
        correct = (pred_binary == labels).sum().item()
        total = len(labels)
        return correct / total if total > 0 else 0.0

    def analyze_rule_learning(
        self,
        brain,
        n_test_phrases: int = 100,
    ) -> Dict[str, float]:
        """
        Analyze how well brain learned each grammar rule.

        Args:
            brain: Brain instance to test
            n_test_phrases: Number of phrases per rule

        Returns:
            results: Dict mapping rule → accuracy
        """
        results = {}

        for rule in self.config.rules_to_test:
            correct = 0
            total = 0

            for _ in range(n_test_phrases):
                phrase, is_gram, _ = self.generate_phrase(rule=rule)

                # Run brain on phrase
                brain_output = brain.judge_grammaticality(phrase)
                prediction = brain_output > 0.5

                if prediction == is_gram:
                    correct += 1
                total += 1

            results[rule.value] = correct / total if total > 0 else 0.0

        return results


def create_stage2_grammar_dataset(
    device: torch.device = torch.device("cpu"),
    multilingual: bool = False,
    language: Language = Language.ENGLISH,
) -> GrammarDataset:
    """
    Create grammar dataset for Stage 2.

    Args:
        device: Device to place tensors on
        multilingual: Include SOV word order (for multilingual training)
        language: Language to use (English, German, or Spanish)

    Returns:
        dataset: GrammarDataset instance
    """
    rules = [
        GrammarRule.SUBJECT_VERB_AGREEMENT,
        GrammarRule.NOUN_ADJECTIVE,
        GrammarRule.WORD_ORDER_SVO,
        GrammarRule.PLURAL_MORPHOLOGY,
    ]

    if multilingual:
        rules.append(GrammarRule.WORD_ORDER_SOV)

    config = GrammarConfig(
        max_phrase_length=5,
        rules_to_test=rules,
        violation_probability=0.2,
        embedding_dim=64,
        multilingual=multilingual,
        language=language,
        device=device,
    )

    return GrammarDataset(config)
