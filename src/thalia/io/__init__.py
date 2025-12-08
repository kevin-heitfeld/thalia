"""
Thalia I/O Module - Binary Checkpoint Format

This module provides efficient binary serialization for brain checkpoints:
- Custom binary format with versioning
- Tensor encoding (dense and sparse CSR)
- Region indexing for efficient loading
- SHA-256 checksums for validation
- Compression support (zstd/lz4)
- Delta checkpoints for curriculum learning

Example:
    from thalia.io import BrainCheckpoint
    
    # Save brain state
    BrainCheckpoint.save(
        brain,
        "checkpoint.thalia",
        metadata={"experiment": "language_learning"}
    )
    
    # Save with compression
    BrainCheckpoint.save(
        brain,
        "checkpoint.thalia.zst",  # Auto-detects zstd compression
        compression='zstd',
        compression_level=3
    )
    
    # Save delta checkpoint (only changes)
    BrainCheckpoint.save_delta(
        brain,
        "stage2.delta.thalia",
        base_checkpoint="stage1.thalia"
    )
    
    # Load brain state (handles compression/delta automatically)
    brain = BrainCheckpoint.load("checkpoint.thalia.zst", device="cuda")
    
    # Inspect without loading
    info = BrainCheckpoint.info("checkpoint.thalia")
"""

from .checkpoint import BrainCheckpoint
from .compression import compress_file, decompress_file, CompressionError
from .delta import save_delta_checkpoint, load_delta_checkpoint
from .precision import PrecisionPolicy, PRECISION_POLICIES, get_precision_statistics

__all__ = [
    "BrainCheckpoint",
    "compress_file",
    "decompress_file",
    "CompressionError",
    "save_delta_checkpoint",
    "load_delta_checkpoint",
    "PrecisionPolicy",
    "PRECISION_POLICIES",
    "get_precision_statistics",
]
