"""
CIFAR-10 Wrapper for Stage 1 (Toddler - Object Permanence & Working Memory)

Wraps torchvision CIFAR-10 with spike encoding for spiking neural networks.
Provides multiple encoding schemes:
- Rate coding (Poisson spikes)
- Temporal coding (latency encoding)
- Phase coding (spike timing relative to oscillations)

Biologically relevant:
- Tests visual cortex hierarchical processing
- Requires object category learning
- Prepares for more complex visual reasoning
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path


@dataclass
class CIFARConfig:
    """Configuration for CIFAR-10 spike encoding."""
    encoding: str = "rate"  # "rate", "temporal", or "phase"
    n_timesteps: int = 100  # Number of timesteps for encoding
    max_firing_rate: float = 1.0  # Maximum firing probability per timestep
    min_intensity: float = 0.0  # Minimum pixel intensity to encode
    image_size: int = 32  # CIFAR-10 native size
    normalize: bool = True  # Normalize pixel values
    augment: bool = False  # Apply data augmentation
    flatten: bool = False  # Flatten spatial dimensions
    device: torch.device = torch.device("cpu")
    data_dir: str = "./data"


class CIFARForThalia:
    """
    CIFAR-10 dataset with spike encoding for spiking neural networks.
    
    Supports multiple encoding schemes:
    - Rate coding: Firing probability proportional to pixel intensity
    - Temporal coding: Brighter pixels spike earlier
    - Phase coding: Spike timing encodes intensity (requires oscillator)
    
    Usage:
        >>> config = CIFARConfig(encoding="rate", n_timesteps=100)
        >>> dataset = CIFARForThalia(config, train=True)
        >>> spikes, label = dataset[0]
        >>> # spikes.shape = (n_timesteps, channels, height, width)
        >>> # or (n_timesteps, channels * height * width) if flatten=True
    """
    
    def __init__(
        self,
        config: CIFARConfig,
        train: bool = True,
        download: bool = True,
    ):
        self.config = config
        self.train = train
        
        # Setup transforms
        transform_list = []
        
        if config.augment and train:
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
        ])
        
        if config.normalize:
            # CIFAR-10 mean/std
            transform_list.append(
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                )
            )
        
        transform = transforms.Compose(transform_list)
        
        # Load CIFAR-10
        data_path = Path(config.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        self.dataset = torchvision.datasets.CIFAR10(
            root=str(data_path),
            train=train,
            download=download,
            transform=transform,
        )
        
        self.classes = self.dataset.classes  # 10 classes
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get spike-encoded image and label.
        
        Args:
            idx: Index of image
            
        Returns:
            spikes: (n_timesteps, C, H, W) or (n_timesteps, C*H*W) if flatten
            label: Class label (0-9)
        """
        image, label = self.dataset[idx]
        
        # Move to device
        image = image.to(self.config.device)
        
        # Encode as spikes
        if self.config.encoding == "rate":
            spikes = self._rate_encode(image)
        elif self.config.encoding == "temporal":
            spikes = self._temporal_encode(image)
        elif self.config.encoding == "phase":
            spikes = self._phase_encode(image)
        else:
            raise ValueError(f"Unknown encoding: {self.config.encoding}")
        
        # Optionally flatten spatial dimensions
        if self.config.flatten:
            # (T, C, H, W) → (T, C*H*W)
            T = spikes.shape[0]
            spikes = spikes.reshape(T, -1)
        
        return spikes, label
    
    def get_batch(
        self,
        indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch of spike-encoded images.
        
        Args:
            indices: List of image indices
            
        Returns:
            spikes: (batch_size, n_timesteps, C, H, W) or (batch, T, C*H*W)
            labels: (batch_size,) class labels
        """
        spikes_list = []
        labels_list = []
        
        for idx in indices:
            spikes, label = self[idx]
            spikes_list.append(spikes)
            labels_list.append(label)
        
        return (
            torch.stack(spikes_list),
            torch.tensor(labels_list, dtype=torch.long, device=self.config.device),
        )
    
    def _rate_encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Rate coding: Firing probability proportional to pixel intensity.
        
        Brighter pixels fire more frequently over time window.
        
        Args:
            image: (C, H, W) normalized image
            
        Returns:
            spikes: (n_timesteps, C, H, W) binary spikes
        """
        # Normalize to [0, 1] range
        if self.config.normalize:
            # Undo normalization
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=image.device).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616], device=image.device).view(3, 1, 1)
            image = image * std + mean
        
        image = torch.clamp(image, 0.0, 1.0)
        
        # Apply minimum intensity threshold
        image = torch.where(
            image < self.config.min_intensity,
            torch.zeros_like(image),
            image,
        )
        
        # Generate Poisson spikes
        # Firing rate proportional to intensity
        firing_rates = image * self.config.max_firing_rate
        
        # Sample spikes for each timestep
        spikes = torch.zeros(
            self.config.n_timesteps,
            *image.shape,
            device=self.config.device,
        )
        
        for t in range(self.config.n_timesteps):
            # Poisson process: P(spike) = rate * dt (where dt=1)
            spike_probs = firing_rates
            spikes[t] = (torch.rand_like(spike_probs) < spike_probs).float()
        
        return spikes
    
    def _temporal_encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Temporal (latency) coding: Brighter pixels spike earlier.
        
        Spike timing encodes intensity:
        - Brightest pixels spike at t=0
        - Darkest pixels spike at t=T-1 (or not at all)
        
        Args:
            image: (C, H, W) normalized image
            
        Returns:
            spikes: (n_timesteps, C, H, W) binary spikes (one spike per pixel)
        """
        # Normalize to [0, 1] range
        if self.config.normalize:
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=image.device).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616], device=image.device).view(3, 1, 1)
            image = image * std + mean
        
        image = torch.clamp(image, 0.0, 1.0)
        
        # Apply minimum intensity threshold
        mask = image >= self.config.min_intensity
        
        # Compute spike times: higher intensity → earlier spike
        # latency = (1 - intensity) * (T - 1)
        spike_times = ((1.0 - image) * (self.config.n_timesteps - 1)).long()
        spike_times = torch.clamp(spike_times, 0, self.config.n_timesteps - 1)
        
        # Create spike tensor
        spikes = torch.zeros(
            self.config.n_timesteps,
            *image.shape,
            device=self.config.device,
        )
        
        # Place one spike per pixel at computed time
        C, H, W = image.shape
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    if mask[c, h, w]:
                        t = int(spike_times[c, h, w].item())
                        spikes[t, c, h, w] = 1.0
        
        return spikes
    
    def _phase_encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Phase coding: Spike timing relative to oscillation encodes intensity.
        
        Brighter pixels spike earlier in oscillation cycle.
        Requires gamma oscillation (40 Hz) for temporal reference.
        
        Args:
            image: (C, H, W) normalized image
            
        Returns:
            spikes: (n_timesteps, C, H, W) binary spikes
        """
        # Normalize to [0, 1] range
        if self.config.normalize:
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=image.device).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616], device=image.device).view(3, 1, 1)
            image = image * std + mean
        
        image = torch.clamp(image, 0.0, 1.0)
        
        # Apply minimum intensity threshold
        mask = image >= self.config.min_intensity
        
        # Gamma oscillation: 40 Hz = 25ms cycle at 1ms timesteps
        gamma_period = 25  # timesteps
        n_cycles = self.config.n_timesteps // gamma_period
        
        # Phase within cycle: higher intensity → earlier phase
        # phase ∈ [0, gamma_period)
        phases = ((1.0 - image) * gamma_period).long()
        phases = torch.clamp(phases, 0, gamma_period - 1)
        
        # Create spike tensor
        spikes = torch.zeros(
            self.config.n_timesteps,
            *image.shape,
            device=self.config.device,
        )
        
        # Place spike at correct phase in each cycle
        C, H, W = image.shape
        for cycle in range(n_cycles):
            cycle_start = cycle * gamma_period
            
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        if mask[c, h, w]:
                            phase = int(phases[c, h, w].item())
                            t = cycle_start + phase
                            if t < self.config.n_timesteps:
                                spikes[t, c, h, w] = 1.0
        
        return spikes
    
    def compute_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute classification accuracy.
        
        Args:
            predictions: (batch_size, n_classes) logits or probabilities
            labels: (batch_size,) true class labels
            
        Returns:
            accuracy: Fraction correct
        """
        pred_classes = torch.argmax(predictions, dim=-1)
        correct = (pred_classes == labels).sum().item()
        total = len(labels)
        return correct / total if total > 0 else 0.0
    
    def get_class_name(self, label: int) -> str:
        """Get human-readable class name."""
        return self.classes[label]
    
    def analyze_encoding_statistics(
        self,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Analyze encoding statistics across samples.
        
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            stats: Dict with mean_firing_rate, sparsity, etc.
        """
        total_spikes = 0
        total_possible = 0
        
        for idx in range(min(n_samples, len(self))):
            spikes, _ = self[idx]
            total_spikes += spikes.sum().item()
            total_possible += spikes.numel()
        
        mean_firing_rate = total_spikes / total_possible if total_possible > 0 else 0.0
        sparsity = 1.0 - mean_firing_rate
        
        return {
            'mean_firing_rate': mean_firing_rate,
            'sparsity': sparsity,
            'n_samples_analyzed': n_samples,
        }


def create_stage1_cifar_datasets(
    device: torch.device = torch.device("cpu"),
    encoding: str = "rate",
    n_timesteps: int = 100,
) -> Tuple[CIFARForThalia, CIFARForThalia]:
    """
    Create CIFAR-10 train/test datasets for Stage 1.
    
    Args:
        device: Device to place tensors on
        encoding: Encoding type ("rate", "temporal", "phase")
        n_timesteps: Number of timesteps for encoding
        
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
    """
    config = CIFARConfig(
        encoding=encoding,
        n_timesteps=n_timesteps,
        max_firing_rate=0.8,  # Conservative for stability
        min_intensity=0.1,  # Ignore very dark pixels
        normalize=True,
        augment=True,  # Only for training
        flatten=False,  # Keep spatial structure
        device=device,
    )
    
    train_dataset = CIFARForThalia(config, train=True, download=True)
    
    # Test dataset without augmentation
    test_config = CIFARConfig(
        encoding=encoding,
        n_timesteps=n_timesteps,
        max_firing_rate=0.8,
        min_intensity=0.1,
        normalize=True,
        augment=False,  # No augmentation for test
        flatten=False,
        device=device,
    )
    test_dataset = CIFARForThalia(test_config, train=False, download=True)
    
    return train_dataset, test_dataset
