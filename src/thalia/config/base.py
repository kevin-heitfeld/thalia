"""
Base Configuration Classes.

This module provides base configuration classes with common fields to reduce
duplication across the codebase. All specific configs should inherit from these.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
import torch


@dataclass
class BaseConfig:
    """Base configuration with common fields for all components.

    This provides standard fields that appear in almost every config:
    - device: Hardware device (cpu/cuda)
    - dtype: Tensor data type
    - seed: Random seed for reproducibility
    """

    device: str = "cpu"
    """Device to run on: 'cpu', 'cuda', 'cuda:0', etc."""

    dtype: str = "float32"
    """Data type for tensors: 'float32', 'float64', 'float16'"""

    seed: Optional[int] = None
    """Random seed for reproducibility. None = no seeding."""

    def get_torch_device(self) -> torch.device:
        """Get PyTorch device object."""
        return torch.device(self.device)

    def get_torch_dtype(self) -> torch.dtype:
        """Get PyTorch dtype object."""
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        if self.dtype not in dtype_map:
            raise ValueError(
                f"Unknown dtype '{self.dtype}'. "
                f"Choose from: {list(dtype_map.keys())}"
            )
        return dtype_map[self.dtype]


@dataclass
class NeuralComponentConfig(BaseConfig):
    """Base config for neural components (regions, pathways, layers).

    Provides common parameters for all neural components:
    - Dimensionality: n_input, n_output, n_neurons
    - Temporal dynamics: dt_ms, axonal_delay_ms
    - Learning: learn, learning_rate
    - Weight bounds: w_min, w_max
    - Device/dtype: From BaseConfig

    Both regions and pathways inherit from this. Regions typically have
    n_input = n_output = n_neurons (recurrent), while pathways have
    n_input (source) and n_output (target) that differ.
    """

    # Dimensionality
    n_neurons: int = 100
    """Number of neurons in the component."""

    n_input: int = 128
    """Input dimension (for regions: input size, for pathways: source region size)."""

    n_output: int = 64
    """Output dimension (for regions: output size, for pathways: target region size)."""

    # Temporal dynamics
    dt_ms: float = 1.0
    """Simulation timestep in milliseconds. Set from GlobalConfig.dt_ms by Brain."""

    axonal_delay_ms: float = 1.0
    """Axonal conduction delay in milliseconds.

    Biological ranges:
    - Within-region (local): 0.5-2ms
    - Inter-region (long-range): 1-10ms
    - Thalamo-cortical: 8-15ms
    - Striato-cortical: 10-20ms

    ALL neural connections have conduction delays - this is not optional.
    Regions and pathways differ only in typical delay values (configuration),
    not in whether delays exist (architectural difference).
    """

    # Learning
    learn: bool = True
    """Whether learning is enabled in this component."""

    learning_rate: float = 0.001
    """Base learning rate for plasticity."""

    # Weight bounds (shared across regions and pathways)
    w_min: float = 0.0
    """Minimum synaptic weight (usually 0.0 for excitatory)."""

    w_max: float = 1.0
    """Maximum synaptic weight (prevents runaway potentiation)."""


@dataclass
class LearningComponentConfig(BaseConfig):
    """Base config for learning components.

    Extends BaseConfig with learning-specific parameters:
    - learning_rate: Base learning rate
    - enabled: Whether learning is enabled
    """

    learning_rate: float = 0.01
    """Base learning rate."""

    enabled: bool = True
    """Whether this learning component is enabled."""


@dataclass
class PathwayConfig(NeuralComponentConfig):
    """Configuration for neural pathways (spiking connections between regions).

    Inherits common parameters from NeuralComponentConfig:
    - n_input, n_output: Source and target region sizes
    - n_neurons: Intermediate neuron population (set to n_output)
    - dt_ms, device, dtype, seed: From NeuralComponentConfig
    - w_min, w_max: Weight bounds
    - learning_rate: Base learning rate
    - axonal_delay_ms: Conduction delay

    All inter-region pathways in Thalia are spike-based, implementing:
    - Leaky integrate-and-fire neurons
    - STDP learning
    - Temporal coding schemes
    - Axonal delays and synaptic filtering

    Example:
        config = PathwayConfig(
            n_input=128,   # Source region size
            n_output=64,   # Target region size
            stdp_lr=0.01,
            temporal_coding=TemporalCoding.PHASE,
        )
    """
    def __post_init__(self):
        """Synchronize n_neurons with n_output for pathway consistency."""
        # For pathways, n_neurons should match n_output (target size)
        self.n_neurons = self.n_output
        
        # Synchronize learning_rate with stdp_lr if stdp_lr was explicitly set
        if hasattr(self, 'stdp_lr') and self.stdp_lr != 0.01:
            self.learning_rate = self.stdp_lr

    # Connectivity
    sparsity: float = 0.1
    """Target sparsity for pathway connections (fraction of non-zero weights)."""

    bidirectional: bool = False
    """Whether pathway supports bidirectional communication."""

    topographic: bool = False
    """Use topographic connectivity pattern."""

    delay_variability: float = 0.2
    """Variability in axonal delays (fraction of mean delay)."""

    # Neuron model parameters
    tau_mem_ms: float = 20.0  # TAU_MEM_STANDARD
    """Membrane time constant in milliseconds."""

    tau_syn_ms: float = 5.0  # TAU_SYN_EXCITATORY
    """Synaptic time constant in milliseconds."""

    v_thresh: float = -50.0  # V_THRESHOLD_STANDARD
    """Spike threshold voltage in mV."""

    v_reset: float = -65.0  # V_RESET_STANDARD
    """Reset voltage after spike in mV."""

    v_rest: float = -70.0  # V_REST_STANDARD
    """Resting membrane potential in mV."""

    refractory_ms: float = 2.0  # TAU_REF_STANDARD
    """Refractory period in milliseconds."""

    # STDP parameters
    learning_rule: str = "STDP"  # SpikingLearningRule enum value
    """Which plasticity rule to use (STDP, PHASE_STDP, TRIPLET_STDP, etc.)."""

    stdp_lr: float = 0.01
    """STDP-specific learning rate."""

    tau_plus_ms: float = 20.0
    """LTP time constant in milliseconds."""

    tau_minus_ms: float = 20.0
    """LTD time constant in milliseconds."""

    a_plus: float = 1.0
    """LTP amplitude."""

    a_minus: float = 1.0
    """LTD amplitude."""

    max_trace: float = 10.0
    """Maximum trace value to prevent runaway accumulation."""

    # Weight initialization
    soft_bounds: bool = True
    """Use soft weight bounds (weight-dependent learning rate)."""

    init_mean: float = 0.3
    """Initial weight mean."""

    init_std: float = 0.1
    """Initial weight standard deviation."""

    # Temporal coding
    temporal_coding: str = "RATE"  # TemporalCoding enum value
    """Which temporal coding scheme (RATE, LATENCY, PHASE, SYNCHRONY, BURST)."""

    oscillation_freq_hz: float = 8.0
    """Oscillation frequency for phase coding (Hz)."""

    phase_precision: float = 0.5
    """How tightly spikes lock to phase (0-1)."""

    # Homeostasis
    synaptic_scaling: bool = True
    """Enable homeostatic synaptic scaling."""

    target_rate: float = 0.1
    """Target firing rate for homeostatic scaling."""

    scaling_tau_ms: float = 1000.0
    """Time constant for homeostatic scaling in milliseconds."""

    # Short-Term Plasticity (STP)
    stp_enabled: bool = False
    """Enable short-term plasticity."""

    stp_type: str = "DEPRESSING"  # STPType enum value
    """Preset synapse type (DEPRESSING, FACILITATING, DUAL)."""

    stp_config: Optional[Any] = None
    """Custom STP parameters (overrides stp_type)."""

    # BCM sliding threshold (metaplasticity)
    bcm_enabled: bool = False
    """Enable BCM sliding threshold rule."""

    bcm_config: Optional[Any] = None
    """Custom BCM parameters."""


__all__ = [
    "BaseConfig",
    "NeuralComponentConfig",
    "LearningComponentConfig",
    "PathwayConfig",
]
