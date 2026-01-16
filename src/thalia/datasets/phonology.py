"""
Phonological Tasks Dataset for Stage 0 Critical Period Learning.

Implements categorical perception tasks for phoneme discrimination:
1. Voice Onset Time (VOT) continua (/p/ vs /b/, /d/ vs /t/)
2. Vowel categories (formant space)
3. Place of articulation contrasts
4. Phoneme boundaries (native vs non-native)

Designed for critical period gating experiments where the brain learns
to categorize continuous acoustic space into discrete phoneme categories.

References:
- Kuhl et al. (2008): Native language magnet theory
- Werker & Tees (1984): Cross-language speech perception
- Eimas et al. (1971): Categorical perception in infants
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch


class Language(Enum):
    """Supported languages for phonological training."""
    ENGLISH = "en"
    GERMAN = "de"
    SPANISH = "es"


class PhonemeCategory(Enum):
    """Phoneme categories for discrimination tasks (multi-language)."""
    # ===== UNIVERSAL: Voicing contrasts (VOT continuum) =====
    P = "p"  # Voiceless bilabial stop (VOT ~60ms)
    B = "b"  # Voiced bilabial stop (VOT ~0ms)
    T = "t"  # Voiceless alveolar stop (VOT ~70ms)
    D = "d"  # Voiced alveolar stop (VOT ~0ms)
    K = "k"  # Voiceless velar stop (VOT ~80ms)
    G = "g"  # Voiced velar stop (VOT ~0ms)

    # ===== ENGLISH: Vowel categories =====
    AA = "aa"  # /ɑ/ as in "father" (F1=730, F2=1090)
    AE = "ae"  # /æ/ as in "cat" (F1=660, F2=1720)
    AH = "ah"  # /ʌ/ as in "but" (F1=640, F2=1190)
    EH = "eh"  # /ɛ/ as in "bed" (F1=530, F2=1840)
    IH = "ih"  # /ɪ/ as in "bit" (F1=390, F2=1990)
    IY = "iy"  # /i/ as in "beat" (F1=270, F2=2290)
    UH = "uh"  # /ʊ/ as in "book" (F1=440, F2=1020)
    UW = "uw"  # /u/ as in "boot" (F1=300, F2=870)

    # ===== GERMAN: Unique vowels =====
    UE = "ue"  # /y/ as in "über" (F1=270, F2=2100) - high front rounded
    OE = "oe"  # /ø/ as in "schön" (F1=390, F2=1680) - mid front rounded
    AE_DE = "ae_de"  # /ɛː/ as in "Käse" (F1=530, F2=1840) - long mid front

    # ===== GERMAN: Consonants =====
    X = "x"  # /x/ as in "Bach" (F1=1500, F2=2500) - voiceless velar fricative
    R_UVULAR = "r_uvular"  # /ʁ/ German uvular r (F1=500, F2=1400)

    # ===== SPANISH: Vowels (5-vowel system) =====
    A_ES = "a_es"  # /a/ as in "casa" (F1=700, F2=1200)
    E_ES = "e_es"  # /e/ as in "peso" (F1=400, F2=2000)
    I_ES = "i_es"  # /i/ as in "piso" (F1=280, F2=2250)
    O_ES = "o_es"  # /o/ as in "poco" (F1=400, F2=800)
    U_ES = "u_es"  # /u/ as in "puro" (F1=300, F2=700)

    # ===== SPANISH: Critical contrasts =====
    R_TAP = "r_tap"  # /ɾ/ single tap as in "pero" (duration ~30ms)
    R_TRILL = "r_trill"  # /r/ trill as in "perro" (duration ~100ms, multiple taps)

    # ===== SPANISH: Voiced fricatives =====
    B_FRIC = "b_fric"  # /β/ voiced bilabial fricative (intervocalic b)
    D_FRIC = "d_fric"  # /ð/ voiced dental fricative (intervocalic d)
    G_FRIC = "g_fric"  # /ɣ/ voiced velar fricative (intervocalic g)

    # ===== UNIVERSAL: Place of articulation =====
    M = "m"  # Bilabial nasal
    N = "n"  # Alveolar nasal
    NG = "ng"  # Velar nasal


@dataclass
class PhonemeFeatures:
    """Acoustic features for a phoneme."""
    # Voice Onset Time (for stops, in ms)
    vot: Optional[float] = None

    # Formants (for vowels, in Hz)
    f1: Optional[float] = None  # First formant (tongue height)
    f2: Optional[float] = None  # Second formant (tongue frontness)
    f3: Optional[float] = None  # Third formant (lip rounding)

    # Additional features
    duration: float = 150.0  # Duration in ms
    intensity: float = 1.0  # Relative intensity (0-1)
    pitch: float = 120.0  # F0 in Hz


# Standard acoustic features for each phoneme
PHONEME_FEATURES: Dict[PhonemeCategory, PhonemeFeatures] = {
    # ===== UNIVERSAL: Voiceless stops (long VOT) =====
    PhonemeCategory.P: PhonemeFeatures(vot=60.0, duration=100.0),
    PhonemeCategory.T: PhonemeFeatures(vot=70.0, duration=100.0),
    PhonemeCategory.K: PhonemeFeatures(vot=80.0, duration=100.0),

    # ===== UNIVERSAL: Voiced stops (short VOT) =====
    PhonemeCategory.B: PhonemeFeatures(vot=5.0, duration=100.0),
    PhonemeCategory.D: PhonemeFeatures(vot=5.0, duration=100.0),
    PhonemeCategory.G: PhonemeFeatures(vot=5.0, duration=100.0),

    # ===== ENGLISH: Vowels =====
    PhonemeCategory.AA: PhonemeFeatures(f1=730.0, f2=1090.0, f3=2440.0, duration=150.0),
    PhonemeCategory.AE: PhonemeFeatures(f1=660.0, f2=1720.0, f3=2410.0, duration=150.0),
    PhonemeCategory.AH: PhonemeFeatures(f1=640.0, f2=1190.0, f3=2390.0, duration=150.0),
    PhonemeCategory.EH: PhonemeFeatures(f1=530.0, f2=1840.0, f3=2480.0, duration=150.0),
    PhonemeCategory.IH: PhonemeFeatures(f1=390.0, f2=1990.0, f3=2550.0, duration=150.0),
    PhonemeCategory.IY: PhonemeFeatures(f1=270.0, f2=2290.0, f3=3010.0, duration=150.0),
    PhonemeCategory.UH: PhonemeFeatures(f1=440.0, f2=1020.0, f3=2240.0, duration=150.0),
    PhonemeCategory.UW: PhonemeFeatures(f1=300.0, f2=870.0, f3=2240.0, duration=150.0),

    # ===== GERMAN: Unique vowels =====
    PhonemeCategory.UE: PhonemeFeatures(f1=270.0, f2=2100.0, f3=2700.0, duration=150.0),  # /y/ über
    PhonemeCategory.OE: PhonemeFeatures(f1=390.0, f2=1680.0, f3=2300.0, duration=150.0),  # /ø/ schön
    PhonemeCategory.AE_DE: PhonemeFeatures(f1=530.0, f2=1840.0, f3=2480.0, duration=180.0),  # /ɛː/ Käse (longer)

    # ===== GERMAN: Consonants =====
    PhonemeCategory.X: PhonemeFeatures(f1=1500.0, f2=2500.0, f3=3500.0, duration=120.0),  # /x/ Bach (fricative noise)
    PhonemeCategory.R_UVULAR: PhonemeFeatures(f1=500.0, f2=1400.0, f3=2200.0, duration=80.0),  # /ʁ/ uvular r

    # ===== SPANISH: 5-vowel system =====
    PhonemeCategory.A_ES: PhonemeFeatures(f1=700.0, f2=1200.0, f3=2500.0, duration=150.0),  # /a/ casa
    PhonemeCategory.E_ES: PhonemeFeatures(f1=400.0, f2=2000.0, f3=2800.0, duration=150.0),  # /e/ peso
    PhonemeCategory.I_ES: PhonemeFeatures(f1=280.0, f2=2250.0, f3=3000.0, duration=150.0),  # /i/ piso
    PhonemeCategory.O_ES: PhonemeFeatures(f1=400.0, f2=800.0, f3=2300.0, duration=150.0),   # /o/ poco
    PhonemeCategory.U_ES: PhonemeFeatures(f1=300.0, f2=700.0, f3=2200.0, duration=150.0),   # /u/ puro

    # ===== SPANISH: Critical tap vs trill =====
    PhonemeCategory.R_TAP: PhonemeFeatures(f1=500.0, f2=1500.0, f3=2500.0, duration=30.0),   # /ɾ/ pero (single tap)
    PhonemeCategory.R_TRILL: PhonemeFeatures(f1=500.0, f2=1500.0, f3=2500.0, duration=100.0),  # /r/ perro (trill)

    # ===== SPANISH: Voiced fricatives (intervocalic) =====
    PhonemeCategory.B_FRIC: PhonemeFeatures(f1=300.0, f2=900.0, f3=2200.0, duration=80.0),  # /β/ cabo
    PhonemeCategory.D_FRIC: PhonemeFeatures(f1=300.0, f2=1700.0, f3=2600.0, duration=80.0),  # /ð/ cada
    PhonemeCategory.G_FRIC: PhonemeFeatures(f1=300.0, f2=2200.0, f3=2850.0, duration=80.0),  # /ɣ/ hago

    # ===== UNIVERSAL: Nasals =====
    PhonemeCategory.M: PhonemeFeatures(f1=280.0, f2=900.0, f3=2200.0, duration=120.0),
    PhonemeCategory.N: PhonemeFeatures(f1=280.0, f2=1700.0, f3=2600.0, duration=120.0),
    PhonemeCategory.NG: PhonemeFeatures(f1=280.0, f2=2200.0, f3=2850.0, duration=120.0),
}


# Language-specific phoneme sets and contrasts
LANGUAGE_PHONEMES: Dict[Language, List[PhonemeCategory]] = {
    Language.ENGLISH: [
        # Stops
        PhonemeCategory.P, PhonemeCategory.B,
        PhonemeCategory.T, PhonemeCategory.D,
        PhonemeCategory.K, PhonemeCategory.G,
        # Vowels
        PhonemeCategory.AA, PhonemeCategory.AE, PhonemeCategory.AH,
        PhonemeCategory.EH, PhonemeCategory.IH, PhonemeCategory.IY,
        PhonemeCategory.UH, PhonemeCategory.UW,
        # Nasals
        PhonemeCategory.M, PhonemeCategory.N, PhonemeCategory.NG,
    ],
    Language.GERMAN: [
        # Stops (shared with English)
        PhonemeCategory.P, PhonemeCategory.B,
        PhonemeCategory.T, PhonemeCategory.D,
        PhonemeCategory.K, PhonemeCategory.G,
        # German-specific vowels
        PhonemeCategory.UE, PhonemeCategory.OE, PhonemeCategory.AE_DE,
        # Also use some English vowels
        PhonemeCategory.IY, PhonemeCategory.IH,
        PhonemeCategory.UW, PhonemeCategory.UH,
        PhonemeCategory.AA, PhonemeCategory.EH,
        # German consonants
        PhonemeCategory.X, PhonemeCategory.R_UVULAR,
        # Nasals
        PhonemeCategory.M, PhonemeCategory.N, PhonemeCategory.NG,
    ],
    Language.SPANISH: [
        # Stops (shared with English)
        PhonemeCategory.P, PhonemeCategory.B,
        PhonemeCategory.T, PhonemeCategory.D,
        PhonemeCategory.K, PhonemeCategory.G,
        # Spanish 5-vowel system
        PhonemeCategory.A_ES, PhonemeCategory.E_ES, PhonemeCategory.I_ES,
        PhonemeCategory.O_ES, PhonemeCategory.U_ES,
        # Critical tap/trill distinction
        PhonemeCategory.R_TAP, PhonemeCategory.R_TRILL,
        # Voiced fricatives
        PhonemeCategory.B_FRIC, PhonemeCategory.D_FRIC, PhonemeCategory.G_FRIC,
        # Nasals
        PhonemeCategory.M, PhonemeCategory.N,
    ],
}


# Language-specific contrasts for training
LANGUAGE_CONTRASTS: Dict[Language, Dict[str, List[Tuple[PhonemeCategory, PhonemeCategory]]]] = {
    Language.ENGLISH: {
        "voicing": [
            (PhonemeCategory.P, PhonemeCategory.B),
            (PhonemeCategory.T, PhonemeCategory.D),
            (PhonemeCategory.K, PhonemeCategory.G),
        ],
        "vowel": [
            (PhonemeCategory.IY, PhonemeCategory.IH),  # beat vs bit
            (PhonemeCategory.EH, PhonemeCategory.AE),  # bed vs bad
            (PhonemeCategory.UW, PhonemeCategory.UH),  # boot vs book
        ],
        "place": [
            (PhonemeCategory.M, PhonemeCategory.N),
            (PhonemeCategory.N, PhonemeCategory.NG),
        ],
    },
    Language.GERMAN: {
        "voicing": [
            (PhonemeCategory.P, PhonemeCategory.B),
            (PhonemeCategory.T, PhonemeCategory.D),
            (PhonemeCategory.K, PhonemeCategory.G),
        ],
        "vowel": [
            (PhonemeCategory.UE, PhonemeCategory.IY),  # ü vs i (front rounded vs unrounded)
            (PhonemeCategory.OE, PhonemeCategory.EH),  # ö vs e (front rounded vs unrounded)
            (PhonemeCategory.UW, PhonemeCategory.UE),  # u vs ü (back vs front, both rounded)
        ],
        "fricative": [
            (PhonemeCategory.X, PhonemeCategory.K),  # Bach-Laut vs k
            (PhonemeCategory.R_UVULAR, PhonemeCategory.R_TAP),  # German r vs tap
        ],
        "place": [
            (PhonemeCategory.M, PhonemeCategory.N),
            (PhonemeCategory.N, PhonemeCategory.NG),
        ],
    },
    Language.SPANISH: {
        "voicing": [
            (PhonemeCategory.P, PhonemeCategory.B),
            (PhonemeCategory.T, PhonemeCategory.D),
            (PhonemeCategory.K, PhonemeCategory.G),
        ],
        "vowel": [
            (PhonemeCategory.E_ES, PhonemeCategory.I_ES),  # e vs i
            (PhonemeCategory.O_ES, PhonemeCategory.U_ES),  # o vs u
            (PhonemeCategory.A_ES, PhonemeCategory.E_ES),  # a vs e
        ],
        "tap_trill": [  # CRITICAL for Spanish!
            (PhonemeCategory.R_TAP, PhonemeCategory.R_TRILL),  # pero vs perro
        ],
        "fricative": [
            (PhonemeCategory.B, PhonemeCategory.B_FRIC),  # boca vs cabo (stop vs fricative)
            (PhonemeCategory.D, PhonemeCategory.D_FRIC),  # donde vs cada
            (PhonemeCategory.G, PhonemeCategory.G_FRIC),  # gato vs hago
        ],
    },
}


@dataclass
class PhonologicalConfig:
    """Configuration for phonological dataset."""
    # Language selection
    language: Language = Language.ENGLISH  # Primary language for training

    # Acoustic encoding
    n_freq_channels: int = 64  # Mel-frequency channels
    n_time_steps: int = 100  # Time steps per stimulus
    sample_rate: int = 16000  # Audio sample rate

    # Task parameters
    noise_std: float = 0.1  # Acoustic noise (0-1)
    within_category_variance: float = 0.15  # Natural variation (0-1)

    # Difficulty calibration
    continuum_steps: int = 11  # Steps in VOT/formant continua
    boundary_sharpness: float = 2.0  # Sigmoid slope at category boundary

    # Device
    device: str = "cpu"


class PhonologicalDataset:
    """
    Phonological tasks for categorical perception learning.

    Provides:
    1. Categorical discrimination (same/different)
    2. Continuum identification (VOT/formant continua)
    3. Cross-language contrasts (native vs non-native)
    4. Phoneme boundary shifts (warping)

    Used in Stage 0 for critical period phonology learning.
    """

    def __init__(self, config: Optional[PhonologicalConfig] = None, language: Optional[Language] = None):
        self.config = config or PhonologicalConfig()

        # Override language if specified
        if language is not None:
            self.config.language = language

        self.device = torch.device(self.config.device)
        self.language = self.config.language

        # Load language-specific contrasts
        language_contrasts = LANGUAGE_CONTRASTS[self.language]

        # Standard contrasts for training (language-dependent)
        self.voicing_contrasts = language_contrasts.get("voicing", [])
        self.vowel_contrasts = language_contrasts.get("vowel", [])
        self.place_contrasts = language_contrasts.get("place", [])

        # Language-specific contrasts
        if self.language == Language.GERMAN:
            self.fricative_contrasts = language_contrasts.get("fricative", [])
        elif self.language == Language.SPANISH:
            self.tap_trill_contrasts = language_contrasts.get("tap_trill", [])
            self.fricative_contrasts = language_contrasts.get("fricative", [])

        # All available contrasts for this language
        self.all_contrasts = {
            "voicing": self.voicing_contrasts,
            "vowel": self.vowel_contrasts,
            "place": self.place_contrasts,
        }

        if self.language == Language.GERMAN:
            self.all_contrasts["fricative"] = self.fricative_contrasts
        elif self.language == Language.SPANISH:
            self.all_contrasts["tap_trill"] = self.tap_trill_contrasts
            self.all_contrasts["fricative"] = self.fricative_contrasts

        # Statistics for performance tracking
        self.reset_statistics()

    def reset_statistics(self):
        """Reset performance statistics."""
        self.stats = {
            "n_trials": 0,
            "n_correct": 0,
            "by_contrast": {},
        }

    def _encode_phoneme(
        self,
        phoneme: PhonemeCategory,
        add_noise: bool = True,
        add_variance: bool = True,
    ) -> torch.Tensor:
        """
        Encode phoneme as acoustic feature vector.

        Returns mel-spectrogram-like representation (n_freq_channels × n_time_steps).
        """
        features = PHONEME_FEATURES[phoneme]
        n_freq = self.config.n_freq_channels
        n_time = self.config.n_time_steps

        # Initialize spectrogram
        spectrogram = torch.zeros(n_freq, n_time, device=self.device)

        # Encode based on phoneme type
        if features.vot is not None:
            # Stop consonant: encode VOT in temporal onset pattern
            vot = features.vot
            if add_variance:
                vot += np.random.randn() * vot * self.config.within_category_variance

            # VOT determines when voicing starts (energy at low frequencies)
            vot_step = int((vot / 150.0) * n_time)  # Normalize to time steps
            vot_step = max(0, min(n_time - 1, vot_step))

            # Burst at onset (high frequencies)
            spectrogram[n_freq//2:, :5] = 0.8

            # Voicing onset after VOT (low frequencies)
            spectrogram[:n_freq//4, vot_step:] = 0.9

            # Formant transitions (mid frequencies)
            for t in range(vot_step, min(vot_step + 20, n_time)):
                ratio = (t - vot_step) / 20.0
                spectrogram[n_freq//4:n_freq//2, t] = 0.6 * ratio

        elif features.f1 is not None:
            # Vowel: encode formants as energy peaks in frequency dimension
            f1, f2 = features.f1, features.f2

            if add_variance:
                f1 += np.random.randn() * f1 * self.config.within_category_variance
                f2 += np.random.randn() * f2 * self.config.within_category_variance

            # Map formants to frequency channels (mel scale approximation)
            # F1: 200-1000 Hz → channels 0-24
            # F2: 800-3000 Hz → channels 16-56
            f1_channel = int((f1 - 200) / 800 * 24)
            f2_channel = int((f2 - 800) / 2200 * 40) + 16

            f1_channel = max(0, min(n_freq - 1, f1_channel))
            f2_channel = max(0, min(n_freq - 1, f2_channel))

            # Create formant peaks (Gaussian in frequency, sustained in time)
            for f_idx in range(n_freq):
                # F1 peak
                dist_f1 = abs(f_idx - f1_channel)
                energy_f1 = 0.9 * np.exp(-(dist_f1**2) / (2 * 3**2))

                # F2 peak
                dist_f2 = abs(f_idx - f2_channel)
                energy_f2 = 0.8 * np.exp(-(dist_f2**2) / (2 * 3**2))

                # Sustained through duration
                duration_steps = int((features.duration / 200.0) * n_time)
                spectrogram[f_idx, :duration_steps] = max(energy_f1, energy_f2)

        # Add acoustic noise
        if add_noise:
            noise = torch.randn_like(spectrogram) * self.config.noise_std
            spectrogram = spectrogram + noise
            spectrogram = torch.clamp(spectrogram, 0, 1)

        return spectrogram

    def generate_discrimination_pair(
        self,
        contrast: Tuple[PhonemeCategory, PhonemeCategory],
        same: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Generate same/different discrimination task.

        Returns:
            stimulus1: First phoneme (n_freq × n_time)
            stimulus2: Second phoneme (n_freq × n_time)
            label: 1 if same, 0 if different
        """
        phoneme1, phoneme2 = contrast

        if same:
            # Both from same category (with natural variance)
            chosen = phoneme1 if np.random.rand() < 0.5 else phoneme2
            stim1 = self._encode_phoneme(chosen, add_variance=True)
            stim2 = self._encode_phoneme(chosen, add_variance=True)
            label = 1
        else:
            # One from each category
            stim1 = self._encode_phoneme(phoneme1, add_variance=True)
            stim2 = self._encode_phoneme(phoneme2, add_variance=True)
            label = 0

        return stim1, stim2, label

    def generate_continuum(
        self,
        contrast: Tuple[PhonemeCategory, PhonemeCategory],
        n_steps: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Generate continuum between two phonemes (e.g., VOT continuum).

        Returns:
            stimuli: List of acoustic patterns along continuum
            labels: Category labels (0 or 1) based on perceptual boundary
        """
        n_steps = n_steps or self.config.continuum_steps
        phoneme1, phoneme2 = contrast
        features1 = PHONEME_FEATURES[phoneme1]
        features2 = PHONEME_FEATURES[phoneme2]

        stimuli = []
        labels = []

        for step in range(n_steps):
            ratio = step / (n_steps - 1)  # 0.0 to 1.0

            # Create intermediate phoneme by interpolating features
            if features1.vot is not None and features2.vot is not None:
                # VOT continuum
                vot = features1.vot + ratio * (features2.vot - features1.vot)
                intermediate_features = PhonemeFeatures(
                    vot=vot,
                    duration=features1.duration,
                )
            elif features1.f1 is not None and features2.f1 is not None:
                # Formant continuum
                f1 = features1.f1 + ratio * (features2.f1 - features1.f1)
                f2 = features1.f2 + ratio * (features2.f2 - features1.f2)
                intermediate_features = PhonemeFeatures(
                    f1=f1,
                    f2=f2,
                    duration=features1.duration,
                )
            else:
                raise ValueError(f"Cannot create continuum for {phoneme1} and {phoneme2}")

            # Encode with temporarily modified features
            # (We'll use phoneme1 as template and modify its features)
            original_features = PHONEME_FEATURES[phoneme1]
            PHONEME_FEATURES[phoneme1] = intermediate_features
            stimulus = self._encode_phoneme(phoneme1, add_noise=True, add_variance=False)
            PHONEME_FEATURES[phoneme1] = original_features

            stimuli.append(stimulus)

            # Categorical boundary at 0.5 (with sigmoid for sharpness)
            boundary_response = 1 / (1 + np.exp(-self.config.boundary_sharpness * (ratio - 0.5)))
            label = 1 if boundary_response > 0.5 else 0
            labels.append(label)

        return stimuli, labels

    def generate_batch(
        self,
        task_type: str = "discrimination",
        batch_size: int = 32,
        contrast_type: str = "voicing",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batch of phonological tasks.

        Args:
            task_type: "discrimination" or "continuum"
            batch_size: Number of trials
            contrast_type: Language-specific contrast type
                English: "voicing", "vowel", "place"
                German: "voicing", "vowel", "place", "fricative"
                Spanish: "voicing", "vowel", "tap_trill", "fricative"

        Returns:
            stimuli: Batch of acoustic patterns (batch × n_freq × n_time)
            labels: Category labels (batch,)
        """
        # Select contrast set (language-specific)
        if contrast_type not in self.all_contrasts:
            available = list(self.all_contrasts.keys())
            raise ValueError(
                f"Unknown contrast type '{contrast_type}' for {self.language.value}. "
                f"Available: {available}"
            )

        contrasts = self.all_contrasts[contrast_type]

        if len(contrasts) == 0:
            raise ValueError(f"No contrasts available for type '{contrast_type}'")

        stimuli_list = []
        labels_list = []

        if task_type == "discrimination":
            for _ in range(batch_size):
                contrast = contrasts[np.random.randint(len(contrasts))]
                same = np.random.rand() < 0.5

                stim1, stim2, label = self.generate_discrimination_pair(contrast, same)

                # Concatenate along time dimension (sequential presentation)
                combined = torch.cat([stim1, stim2], dim=1)  # (n_freq, 2*n_time)

                stimuli_list.append(combined)
                labels_list.append(label)

        elif task_type == "continuum":
            contrast = contrasts[np.random.randint(len(contrasts))]
            continuum_stimuli, continuum_labels = self.generate_continuum(contrast)

            # Sample from continuum
            for _ in range(batch_size):
                idx = np.random.randint(len(continuum_stimuli))
                stimuli_list.append(continuum_stimuli[idx])
                labels_list.append(continuum_labels[idx])

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Stack into batch tensors (handle empty case)
        if len(stimuli_list) == 0:
            # Return empty tensors with correct shape
            if task_type == "discrimination":
                stimuli = torch.zeros(0, self.config.n_freq_channels, 2*self.config.n_time_steps, device=self.device)
            else:
                stimuli = torch.zeros(0, self.config.n_freq_channels, self.config.n_time_steps, device=self.device)
            labels = torch.zeros(0, dtype=torch.long, device=self.device)
        else:
            stimuli = torch.stack(stimuli_list, dim=0)
            labels = torch.tensor(labels_list, dtype=torch.long, device=self.device)

        return stimuli, labels

    def evaluate_discrimination(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        contrast_type: str,
    ) -> Dict[str, float]:
        """
        Evaluate discrimination performance.

        Args:
            predictions: Model predictions (batch, 2) or (batch,)
            labels: Ground truth (batch,)
            contrast_type: Type of contrast

        Returns:
            metrics: Accuracy, d-prime, etc.
        """
        if predictions.dim() == 2:
            # Logits: take argmax
            predicted_labels = predictions.argmax(dim=1)
        else:
            # Already labels
            predicted_labels = predictions

        correct = (predicted_labels == labels).sum().item()
        total = len(labels)
        accuracy = correct / total

        # Update statistics
        self.stats["n_trials"] += total
        self.stats["n_correct"] += correct

        if contrast_type not in self.stats["by_contrast"]:
            self.stats["by_contrast"][contrast_type] = {"correct": 0, "total": 0}

        self.stats["by_contrast"][contrast_type]["correct"] += correct
        self.stats["by_contrast"][contrast_type]["total"] += total

        # Compute d-prime (signal detection theory)
        # Hit rate: P(respond "different" | actually different)
        # FA rate: P(respond "different" | actually same)
        hits = ((predicted_labels == 0) & (labels == 0)).sum().item()
        false_alarms = ((predicted_labels == 0) & (labels == 1)).sum().item()

        n_different = (labels == 0).sum().item()
        n_same = (labels == 1).sum().item()

        hit_rate = (hits + 0.5) / (n_different + 1)  # Add 0.5 for continuity correction
        fa_rate = (false_alarms + 0.5) / (n_same + 1)

        # Convert to z-scores
        from scipy.stats import norm
        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)

        return {
            "accuracy": accuracy,
            "d_prime": d_prime,
            "hit_rate": hit_rate,
            "fa_rate": fa_rate,
        }

    def get_statistics(self) -> Dict[str, any]:
        """Get cumulative performance statistics."""
        overall_accuracy = (
            self.stats["n_correct"] / self.stats["n_trials"]
            if self.stats["n_trials"] > 0
            else 0.0
        )

        by_contrast_accuracy = {
            contrast: stats["correct"] / stats["total"]
            for contrast, stats in self.stats["by_contrast"].items()
        }

        return {
            "overall_accuracy": overall_accuracy,
            "by_contrast": by_contrast_accuracy,
            "n_trials": self.stats["n_trials"],
        }
