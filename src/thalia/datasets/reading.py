"""
Reading Comprehension Dataset for Stage 3 (Reading & Sentence Semantics)

Generates reading tasks progressing from:
1. Phoneme → word decoding
2. Word → sentence composition
3. Simple comprehension questions
4. Semantic role labeling (who did what to whom)

Biologically relevant:
- Tests phonological → orthographic mapping
- Engages angular gyrus (reading)
- Requires semantic integration (prefrontal + temporal)
- Prepares for complex language understanding
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class Language(Enum):
    """Supported languages for reading tasks."""

    ENGLISH = "en"
    GERMAN = "de"
    SPANISH = "es"


class ReadingTask(Enum):
    """Types of reading tasks."""

    PHONEME_TO_WORD = "phoneme_to_word"  # Decode phonemes → word
    WORD_TO_MEANING = "word_to_meaning"  # Map word → semantic features
    SENTENCE_COMPLETION = "sentence_completion"  # Fill in missing word
    SIMPLE_QA = "simple_qa"  # Who/what/where questions
    SEMANTIC_ROLE = "semantic_role"  # Agent/action/patient labeling


@dataclass
class ReadingConfig:
    """Configuration for reading dataset."""

    vocab_size: int = 500  # Total vocabulary
    max_sentence_length: int = 10  # Max words per sentence
    max_phonemes: int = 15  # Max phonemes per word
    tasks_to_test: Optional[List[ReadingTask]] = None
    embedding_dim: int = 64
    language: Language = Language.ENGLISH  # Language to use
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        if self.tasks_to_test is None:
            self.tasks_to_test = [
                ReadingTask.PHONEME_TO_WORD,
                ReadingTask.WORD_TO_MEANING,
                ReadingTask.SENTENCE_COMPLETION,
                ReadingTask.SIMPLE_QA,
            ]


class ReadingVocabulary:
    """Vocabulary for reading tasks with multilingual support."""

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

        # Combine all words
        self.all_words = {
            **self.nouns,
            **self.verbs,
            **self.adjectives,
            **self.function_words,
        }

        # Build mappings
        self.word2idx = {word: idx for idx, word in enumerate(self.all_words.keys())}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Phoneme inventory
        self.phonemes: list = []
        phonemes_set = set()
        for phoneme_list in self.all_words.values():
            phonemes_set.update(phoneme_list)
        self.phonemes = sorted(list(phonemes_set))

        self.phoneme2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2phoneme = {idx: p for p, idx in self.phoneme2idx.items()}

        # Special tokens
        self.word2idx["<PAD>"] = len(self.word2idx)
        self.word2idx["<UNK>"] = len(self.word2idx)
        self.pad_idx = self.word2idx["<PAD>"]
        self.unk_idx = self.word2idx["<UNK>"]

        self.phoneme2idx["<PAD>"] = len(self.phoneme2idx)
        self.phoneme_pad_idx = self.phoneme2idx["<PAD>"]

        self.vocab_size = len(self.word2idx)
        self.n_phonemes = len(self.phoneme2idx)

    def _init_english(self):
        """Initialize English vocabulary."""
        # Common nouns
        self.nouns = {
            "cat": ["k", "æ", "t"],
            "dog": ["d", "ɔ", "g"],
            "ball": ["b", "ɔ", "l"],
            "book": ["b", "ʊ", "k"],
            "car": ["k", "ɑ", "r"],
            "tree": ["t", "r", "i"],
            "bird": ["b", "ɜ", "r", "d"],
            "fish": ["f", "ɪ", "ʃ"],
            "house": ["h", "aʊ", "s"],
            "girl": ["g", "ɜ", "r", "l"],
            "boy": ["b", "ɔɪ"],
        }

        # Common verbs
        self.verbs = {
            "runs": ["r", "ʌ", "n", "z"],
            "jumps": ["dʒ", "ʌ", "m", "p", "s"],
            "eats": ["i", "t", "s"],
            "sees": ["s", "i", "z"],
            "reads": ["r", "i", "d", "z"],
            "plays": ["p", "l", "eɪ", "z"],
            "throws": ["θ", "r", "oʊ", "z"],
            "catches": ["k", "æ", "tʃ", "ɪ", "z"],
        }

        # Common adjectives
        self.adjectives = {
            "big": ["b", "ɪ", "g"],
            "small": ["s", "m", "ɔ", "l"],
            "red": ["r", "ɛ", "d"],
            "blue": ["b", "l", "u"],
            "fast": ["f", "æ", "s", "t"],
            "happy": ["h", "æ", "p", "i"],
        }

        # Function words
        self.function_words = {
            "the": ["ð", "ə"],
            "a": ["ə"],
            "in": ["ɪ", "n"],
            "on": ["ɑ", "n"],
            "with": ["w", "ɪ", "θ"],
            "is": ["ɪ", "z"],
        }

    def _init_german(self):
        """Initialize German vocabulary with IPA phonemes."""
        # Common nouns (German nouns are capitalized)
        self.nouns = {
            "Katze": ["k", "a", "t", "s", "ə"],  # cat
            "Hund": ["h", "ʊ", "n", "t"],  # dog
            "Ball": ["b", "a", "l"],  # ball
            "Buch": ["b", "uː", "x"],  # book
            "Auto": ["aʊ", "t", "o"],  # car
            "Baum": ["b", "aʊ", "m"],  # tree
            "Vogel": ["f", "oː", "g", "əl"],  # bird
            "Fisch": ["f", "ɪ", "ʃ"],  # fish
            "Haus": ["h", "aʊ", "s"],  # house
            "Mädchen": ["m", "ɛː", "t", "ç", "ən"],  # girl
            "Junge": ["j", "ʊ", "ŋ", "ə"],  # boy
        }

        # Common verbs
        self.verbs = {
            "läuft": ["l", "ɔɪ", "f", "t"],  # runs
            "springt": ["ʃ", "p", "r", "ɪ", "ŋ", "t"],  # jumps
            "isst": ["ɪ", "s", "t"],  # eats
            "sieht": ["z", "iː", "t"],  # sees
            "liest": ["l", "iː", "s", "t"],  # reads
            "spielt": ["ʃ", "p", "iː", "l", "t"],  # plays
            "wirft": ["v", "ɪ", "r", "f", "t"],  # throws
            "fängt": ["f", "ɛ", "ŋ", "t"],  # catches
        }

        # Common adjectives
        self.adjectives = {
            "groß": ["g", "r", "oː", "s"],  # big
            "klein": ["k", "l", "aɪ", "n"],  # small
            "rot": ["r", "oː", "t"],  # red
            "blau": ["b", "l", "aʊ"],  # blue
            "schnell": ["ʃ", "n", "ɛ", "l"],  # fast
            "glücklich": ["g", "l", "ʏ", "k", "l", "ɪ", "ç"],  # happy
        }

        # Function words
        self.function_words = {
            "der": ["d", "eː", "ɐ"],  # the (masc)
            "die": ["d", "iː"],  # the (fem/plural)
            "das": ["d", "a", "s"],  # the (neut)
            "ein": ["aɪ", "n"],  # a (masc/neut)
            "eine": ["aɪ", "n", "ə"],  # a (fem)
            "in": ["ɪ", "n"],  # in
            "auf": ["aʊ", "f"],  # on
            "mit": ["m", "ɪ", "t"],  # with
            "ist": ["ɪ", "s", "t"],  # is
        }

    def _init_spanish(self):
        """Initialize Spanish vocabulary with IPA phonemes."""
        # Common nouns
        self.nouns = {
            "gato": ["g", "a", "t", "o"],  # cat
            "perro": ["p", "e", "r", "o"],  # dog
            "pelota": ["p", "e", "l", "o", "t", "a"],  # ball
            "libro": ["l", "i", "β", "r", "o"],  # book
            "coche": ["k", "o", "t͡ʃ", "e"],  # car
            "árbol": ["a", "r", "β", "o", "l"],  # tree
            "pájaro": ["p", "a", "x", "a", "r", "o"],  # bird
            "pez": ["p", "e", "s"],  # fish
            "casa": ["k", "a", "s", "a"],  # house
            "niña": ["n", "i", "ɲ", "a"],  # girl
            "niño": ["n", "i", "ɲ", "o"],  # boy
        }

        # Common verbs
        self.verbs = {
            "corre": ["k", "o", "r", "e"],  # runs
            "salta": ["s", "a", "l", "t", "a"],  # jumps
            "come": ["k", "o", "m", "e"],  # eats
            "ve": ["b", "e"],  # sees
            "lee": ["l", "e", "e"],  # reads
            "juega": ["x", "w", "e", "ɣ", "a"],  # plays
            "lanza": ["l", "a", "n", "s", "a"],  # throws
            "atrapa": ["a", "t", "r", "a", "p", "a"],  # catches
        }

        # Common adjectives
        self.adjectives = {
            "grande": ["g", "r", "a", "n", "d", "e"],  # big
            "pequeño": ["p", "e", "k", "e", "ɲ", "o"],  # small
            "rojo": ["r", "o", "x", "o"],  # red
            "azul": ["a", "s", "u", "l"],  # blue
            "rápido": ["r", "a", "p", "i", "ð", "o"],  # fast
            "feliz": ["f", "e", "l", "i", "s"],  # happy
        }

        # Function words
        self.function_words = {
            "el": ["e", "l"],  # the (masc)
            "la": ["l", "a"],  # the (fem)
            "un": ["u", "n"],  # a (masc)
            "una": ["u", "n", "a"],  # a (fem)
            "en": ["e", "n"],  # in
            "sobre": ["s", "o", "β", "r", "e"],  # on
            "con": ["k", "o", "n"],  # with
            "es": ["e", "s"],  # is
        }


class ReadingDataset:
    """
    Reading comprehension dataset for Stage 3.

    Generates tasks progressing from phoneme decoding to comprehension:
    - Phoneme → word mapping
    - Word → meaning mapping
    - Sentence completion
    - Simple question answering

    Usage:
        >>> config = ReadingConfig()
        >>> dataset = ReadingDataset(config)
        >>> task_data, label, task_type = dataset.generate_task()
    """

    def __init__(self, config: ReadingConfig):
        self.config = config
        self.vocab = ReadingVocabulary(language=config.language)

        # Task generators
        self.task_generators = {
            ReadingTask.PHONEME_TO_WORD: self._generate_phoneme_to_word,
            ReadingTask.WORD_TO_MEANING: self._generate_word_to_meaning,
            ReadingTask.SENTENCE_COMPLETION: self._generate_sentence_completion,
            ReadingTask.SIMPLE_QA: self._generate_simple_qa,
            ReadingTask.SEMANTIC_ROLE: self._generate_semantic_role,
        }

    def generate_task(
        self,
        task_type: Optional[ReadingTask] = None,
    ) -> Tuple[Dict, torch.Tensor, ReadingTask]:
        """
        Generate single reading task.

        Args:
            task_type: Specific task type (random if None)

        Returns:
            task_data: Dict with task-specific inputs
            label: Target output
            task_type: Type of task generated
        """
        if task_type is None:
            task_type = np.random.choice(self.config.tasks_to_test)

        task_data, label = self.task_generators[task_type]()

        return task_data, label, task_type

    def _generate_phoneme_to_word(self) -> Tuple[Dict, torch.Tensor]:
        """
        Phoneme decoding: Given phonemes, predict word.

        Returns:
            task_data: {'phonemes': tensor of phoneme indices}
            label: Word index
        """
        # Sample random word
        word = np.random.choice(list(self.vocab.all_words.keys()))
        phonemes = self.vocab.all_words[word]

        # Convert to indices
        phoneme_indices = [self.vocab.phoneme2idx[p] for p in phonemes]

        # Pad to max length
        while len(phoneme_indices) < self.config.max_phonemes:
            phoneme_indices.append(self.vocab.phoneme_pad_idx)

        task_data = {
            "phonemes": torch.tensor(
                phoneme_indices[: self.config.max_phonemes],
                dtype=torch.long,
                device=self.config.device,
            )
        }

        label = torch.tensor(
            self.vocab.word2idx[word],
            dtype=torch.long,
            device=self.config.device,
        )

        return task_data, label

    def _generate_word_to_meaning(self) -> Tuple[Dict, torch.Tensor]:
        """
        Word → meaning: Given word, predict semantic features.

        Semantic features:
        - Is animate? (0/1)
        - Is object? (0/1)
        - Is action? (0/1)
        - Size category (0=small, 1=medium, 2=large)

        Returns:
            task_data: {'word': word index}
            label: (4,) semantic feature vector
        """
        # Sample word
        word = np.random.choice(list(self.vocab.all_words.keys()))

        # Determine semantic features
        is_animate = (
            1
            if word in self.vocab.nouns and word in ["cat", "dog", "bird", "fish", "girl", "boy"]
            else 0
        )
        is_object = 1 if word in self.vocab.nouns else 0
        is_action = 1 if word in self.vocab.verbs else 0

        # Size (simplified)
        if word in ["cat", "dog", "bird", "fish", "ball", "book"]:
            size = 0  # small
        elif word in ["car", "tree", "house"]:
            size = 2  # large
        else:
            size = 1  # medium

        task_data = {
            "word": torch.tensor(
                self.vocab.word2idx[word],
                dtype=torch.long,
                device=self.config.device,
            )
        }

        label = torch.tensor(
            [is_animate, is_object, is_action, size],
            dtype=torch.float,
            device=self.config.device,
        )

        return task_data, label

    def _generate_sentence_completion(self) -> Tuple[Dict, torch.Tensor]:
        """
        Sentence completion: Fill in missing word.

        "The cat ___ the ball" → "catches"

        Returns:
            task_data: {'sentence': partial sentence indices}
            label: Missing word index
        """
        # Generate simple sentence: Det + Noun + Verb + Det + Noun
        det1 = "the"
        subj = np.random.choice(list(self.vocab.nouns.keys()))
        verb = np.random.choice(list(self.vocab.verbs.keys()))
        det2 = "the"
        obj = np.random.choice(list(self.vocab.nouns.keys()))

        sentence = [det1, subj, verb, det2, obj]

        # Remove one content word (noun or verb)
        missing_pos = np.random.choice([1, 2, 4])  # subj, verb, or obj
        missing_word = sentence[missing_pos]
        sentence[missing_pos] = "<UNK>"

        # Convert to indices
        sentence_indices = [self.vocab.word2idx.get(w, self.vocab.unk_idx) for w in sentence]

        # Pad
        while len(sentence_indices) < self.config.max_sentence_length:
            sentence_indices.append(self.vocab.pad_idx)

        task_data = {
            "sentence": torch.tensor(
                sentence_indices[: self.config.max_sentence_length],
                dtype=torch.long,
                device=self.config.device,
            ),
            "missing_position": torch.tensor(
                missing_pos,
                dtype=torch.long,
                device=self.config.device,
            ),
        }

        label = torch.tensor(
            self.vocab.word2idx[missing_word],
            dtype=torch.long,
            device=self.config.device,
        )

        return task_data, label

    def _generate_simple_qa(self) -> Tuple[Dict, torch.Tensor]:
        """
        Simple question answering.

        Sentence: "The girl reads the book"
        Question: "Who reads?" → "girl"

        Returns:
            task_data: {'sentence': indices, 'question': indices}
            label: Answer word index
        """
        # Generate sentence
        det1 = "the"
        subj = np.random.choice(["girl", "boy", "cat", "dog"])
        verb = np.random.choice(list(self.vocab.verbs.keys()))
        det2 = "the"
        obj = np.random.choice(list(self.vocab.nouns.keys()))

        sentence = [det1, subj, verb, det2, obj]

        # Generate question
        question_type = np.random.choice(["who", "what_action", "what_object"])

        if question_type == "who":
            question = ["who", verb]  # Who <verbs>?
            answer = subj
        elif question_type == "what_action":
            question = ["what", "does", subj]  # What does <subj> do?
            answer = verb
        else:  # what_object
            question = ["what", "object"]  # What object?
            answer = obj

        # Convert to indices
        sentence_indices = [self.vocab.word2idx.get(w, self.vocab.unk_idx) for w in sentence]
        question_indices = [self.vocab.word2idx.get(w, self.vocab.unk_idx) for w in question]

        # Pad
        while len(sentence_indices) < self.config.max_sentence_length:
            sentence_indices.append(self.vocab.pad_idx)
        while len(question_indices) < self.config.max_sentence_length:
            question_indices.append(self.vocab.pad_idx)

        task_data = {
            "sentence": torch.tensor(
                sentence_indices[: self.config.max_sentence_length],
                dtype=torch.long,
                device=self.config.device,
            ),
            "question": torch.tensor(
                question_indices[: self.config.max_sentence_length],
                dtype=torch.long,
                device=self.config.device,
            ),
        }

        label = torch.tensor(
            self.vocab.word2idx.get(answer, self.vocab.unk_idx),
            dtype=torch.long,
            device=self.config.device,
        )

        return task_data, label

    def _generate_semantic_role(self) -> Tuple[Dict, torch.Tensor]:
        """
        Semantic role labeling: Identify agent/action/patient.

        Sentence: "The girl throws the ball"
        Roles: agent=girl, action=throws, patient=ball

        Returns:
            task_data: {'sentence': indices}
            label: (3,) roles [agent_idx, action_idx, patient_idx]
        """
        # Generate sentence
        det1 = "the"
        subj = np.random.choice(["girl", "boy", "cat", "dog"])
        verb = np.random.choice(list(self.vocab.verbs.keys()))
        det2 = "the"
        obj = np.random.choice(list(self.vocab.nouns.keys()))

        sentence = [det1, subj, verb, det2, obj]

        # Roles: positions in sentence
        agent_pos = 1  # subj
        action_pos = 2  # verb
        patient_pos = 4  # obj

        sentence_indices = [self.vocab.word2idx.get(w, self.vocab.unk_idx) for w in sentence]

        # Pad
        while len(sentence_indices) < self.config.max_sentence_length:
            sentence_indices.append(self.vocab.pad_idx)

        task_data = {
            "sentence": torch.tensor(
                sentence_indices[: self.config.max_sentence_length],
                dtype=torch.long,
                device=self.config.device,
            ),
        }

        label = torch.tensor(
            [agent_pos, action_pos, patient_pos],
            dtype=torch.long,
            device=self.config.device,
        )

        return task_data, label

    def compute_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        task_type: ReadingTask,
    ) -> float:
        """
        Compute task-specific accuracy.

        Args:
            predictions: Model predictions (shape depends on task)
            labels: True labels
            task_type: Type of task

        Returns:
            accuracy: Fraction correct
        """
        if task_type in [
            ReadingTask.PHONEME_TO_WORD,
            ReadingTask.SENTENCE_COMPLETION,
            ReadingTask.SIMPLE_QA,
        ]:
            # Classification: argmax predictions
            pred_classes = torch.argmax(predictions, dim=-1)
            correct = (pred_classes == labels).sum().item()
            total = len(labels)
        elif task_type == ReadingTask.WORD_TO_MEANING:
            # Multi-label: threshold at 0.5
            pred_binary = (predictions > 0.5).float()
            correct = (pred_binary == labels).all(dim=-1).sum().item()
            total = len(labels)
        elif task_type == ReadingTask.SEMANTIC_ROLE:
            # Multiple positions: all must match
            correct = (predictions == labels).all(dim=-1).sum().item()
            total = len(labels)
        else:
            return 0.0

        return correct / total if total > 0 else 0.0


def create_stage3_reading_dataset(
    device: torch.device = torch.device("cpu"),
    language: Language = Language.ENGLISH,
) -> ReadingDataset:
    """
    Create reading dataset for Stage 3.

    Args:
        device: Device to place tensors on
        language: Language to use (English, German, or Spanish)

    Returns:
        dataset: ReadingDataset instance
    """
    config = ReadingConfig(
        vocab_size=500,
        max_sentence_length=10,
        max_phonemes=15,
        tasks_to_test=[
            ReadingTask.PHONEME_TO_WORD,
            ReadingTask.WORD_TO_MEANING,
            ReadingTask.SENTENCE_COMPLETION,
            ReadingTask.SIMPLE_QA,
        ],
        embedding_dim=64,
        language=language,
        device=device,
    )

    return ReadingDataset(config)
