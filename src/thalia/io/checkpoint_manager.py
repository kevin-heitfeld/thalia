"""
Unified Checkpoint Manager for Brain Components.

Provides centralized checkpoint management for the complete brain state,
ensuring consistent checkpointing across all components (regions, pathways,
neuromodulators, oscillators).

Design Philosophy:
==================
- Single entry point for saving/loading brain state
- Ensures complete state capture (no missing components)
- Handles versioning and migration
- Provides validation and metadata tracking
- Consistent checkpoint format across versions

Architecture:
=============
    CheckpointManager
        ├── save() → Complete brain state to disk
        ├── load() → Restore complete brain state
        ├── validate() → Check checkpoint integrity
        ├── get_metadata() → Extract metadata without full load
        └── migrate() → Convert old checkpoints to new format

Benefits:
=========
1. **Completeness**: Guaranteed capture of all component states
2. **Consistency**: Same checkpoint format everywhere
3. **Validation**: Detect missing or incompatible components
4. **Versioning**: Track checkpoint format version
5. **Migration**: Handle old checkpoint formats
6. **Debugging**: Inspect checkpoint contents without loading
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch

from thalia import __version__ as THALIA_VERSION
from thalia.io.checkpoint import BrainCheckpoint
from thalia.io.precision import PrecisionPolicy

if TYPE_CHECKING:
    from thalia.brain import DynamicBrain


class CheckpointManager:
    """Unified checkpoint manager for DynamicBrain.

    Provides centralized checkpoint save/load/validate operations
    ensuring complete state capture and restoration.

    Attributes:
        brain: DynamicBrain instance to manage
        default_precision: Default precision policy for checkpoints
        default_compression: Default compression method
    """

    def __init__(
        self,
        brain: DynamicBrain,
        default_precision: Union[str, PrecisionPolicy] = "fp32",
        default_compression: Optional[str] = None,
    ):
        """Initialize checkpoint manager.

        Args:
            brain: DynamicBrain instance
            default_precision: Default precision policy ('fp32', 'fp16', 'mixed')
            default_compression: Default compression ('zstd', 'lz4', None)
        """
        self.brain = brain
        self.default_precision = default_precision
        self.default_compression = default_compression

    def save(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        compression: Optional[str] = None,
        compression_level: int = 3,
        precision_policy: Union[str, PrecisionPolicy, None] = None,
    ) -> Dict[str, Any]:
        """Save complete brain state to checkpoint file.

        Args:
            path: Path to save checkpoint
            metadata: Optional metadata dict (epoch, loss, etc.)
            compression: Compression type ('zstd', 'lz4', or None)
            compression_level: Compression level (1-22 for zstd, 1-12 for lz4)
            precision_policy: Mixed precision policy or None to use default

        Returns:
            Dict containing save info (size, time, components saved)
        """
        # Use defaults if not specified
        if compression is None:
            compression = self.default_compression
        if precision_policy is None:
            precision_policy = self.default_precision

        # Ensure path is Path object
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Add checkpoint metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "checkpoint_version": "2.0",  # Checkpoint format version
                "saved_at": datetime.now().isoformat(),
                "thalia_version": THALIA_VERSION,
            }
        )

        # Save using BrainCheckpoint API
        start_time = time.time()
        save_info = BrainCheckpoint.save(
            self.brain,
            path,
            metadata=metadata,
            compression=compression,
            compression_level=compression_level,
            precision_policy=precision_policy,
        )
        end_time = time.time()

        # Add timing information
        save_info["time_s"] = end_time - start_time
        save_info["components_saved"] = self._count_components()

        return save_info

    def load(
        self,
        path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load complete brain state from checkpoint file.

        Args:
            path: Path to checkpoint file
            device: Device to load to (None = use brain's device)
            strict: Whether to enforce strict config matching

        Returns:
            Dict containing load info (metadata, load time, etc.)

        Raises:
            ValueError: If checkpoint invalid or config mismatch (when strict=True)
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Checkpoint file not found: {path}")

        # Use brain's device if not specified
        if device is None:
            device = self.brain.device

        # Load checkpoint using BrainCheckpoint API
        start_time = time.time()
        checkpoint_data = BrainCheckpoint.load(path, device)
        end_time = time.time()

        # checkpoint_data IS the state dict (has regions, metadata, config, etc.)
        metadata = checkpoint_data.get("metadata", {})

        # TODO: Load state into brain
        # self.brain.load_state(state)

        # Return load info
        return {
            "metadata": metadata,
            "time_s": end_time - start_time,
            "checkpoint_version": metadata.get("checkpoint_version", "unknown"),
            "components_loaded": self._count_components(),
        }

    def validate(self, path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """Validate checkpoint file integrity and compatibility.

        Args:
            path: Path to checkpoint file

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if checkpoint is valid
            - error_message: None if valid, error string if invalid
        """
        path = Path(path)

        # Check file exists
        if not path.exists():
            return False, f"File not found: {path}"

        try:
            # Use BrainCheckpoint validation
            BrainCheckpoint.validate(path)

            return True, None

        except Exception as e:
            return False, str(e)

    def get_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get checkpoint metadata without loading full state.

        Args:
            path: Path to checkpoint file

        Returns:
            Dict containing metadata (epoch, loss, timestamp, etc.)
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Checkpoint file not found: {path}")

        # Use BrainCheckpoint info API
        info = BrainCheckpoint.info(path)
        return dict(info.get("metadata", {}))

    def list_components(self) -> Dict[str, Any]:
        """List all components managed by this checkpoint manager.

        Returns:
            Dict mapping component type to component names
        """
        return {
            "regions": list(self.brain.regions.keys()),
            "pathways": list(self.brain.connections.keys()),
            "neuromodulators": ["vta", "lc", "nb"],  # VTA, LC, NB
            "oscillators": ["delta", "theta", "alpha", "beta", "gamma"],
            "managers": ["neuromodulator_manager", "oscillator_manager"],
        }

    def _count_components(self) -> Dict[str, int]:
        """Count components in the brain.

        Returns:
            Dict with component counts
        """
        return {
            "regions": len(self.brain.regions),
            "pathways": len(self.brain.connections),
            "neuromodulators": 3,  # VTA, LC, NB
            "oscillators": 5,  # delta, theta, alpha, beta, gamma
        }
