"""
Thalia I/O Module - Binary Checkpoint Format

This module provides efficient binary serialization for brain checkpoints:
- Custom binary format with versioning
- Tensor encoding (dense and sparse CSR)
- Region indexing for efficient loading
- SHA-256 checksums for validation

Example:
    from thalia.io import BrainCheckpoint
    
    # Save brain state
    BrainCheckpoint.save(
        brain,
        "checkpoint.thalia",
        metadata={"experiment": "language_learning"}
    )
    
    # Load brain state
    brain = BrainCheckpoint.load("checkpoint.thalia", device="cuda")
    
    # Inspect without loading
    info = BrainCheckpoint.info("checkpoint.thalia")
"""

from .checkpoint import BrainCheckpoint

__all__ = ["BrainCheckpoint"]
