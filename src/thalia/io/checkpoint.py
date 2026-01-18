"""
Brain Checkpoint API - High-level interface for saving/loading brain states.

Provides simple API for checkpoint persistence:
    BrainCheckpoint.save(brain, path, metadata)
    BrainCheckpoint.load(path, device)
    BrainCheckpoint.info(path)
    BrainCheckpoint.validate(path)
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import warnings
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from thalia import __version__ as THALIA_VERSION

from .binary_format import (
    HEADER_SIZE,
    MAGIC_NUMBER,
    MAJOR_VERSION,
    MINOR_VERSION,
    PATCH_VERSION,
    BinaryReader,
    BinaryWriter,
    CheckpointHeader,
    RegionIndexEntry,
)
from .precision import (
    PrecisionPolicy,
    apply_precision_policy_to_state,
    get_precision_policy,
    get_precision_statistics,
    restore_precision_to_fp32,
)
from .tensor_encoding import decode_tensor, encode_tensor


def _convert_to_json_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, torch.device):
        # Convert torch.device to string representation
        return {"_type": "torch.device", "_value": str(obj)}
    elif is_dataclass(obj):
        # Preserve dataclass type information for reconstruction
        return {
            "_dataclass": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
            "_fields": {k: _convert_to_json_serializable(v) for k, v in asdict(obj).items()},  # type: ignore[arg-type]
        }
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def _convert_from_json(obj: Any) -> Any:
    """Recursively reconstruct objects from JSON-serializable types."""
    if isinstance(obj, dict):
        # Check if this is a torch.device
        if obj.get("_type") == "torch.device":
            return torch.device(obj["_value"])
        # Check if this is a serialized dataclass
        elif "_dataclass" in obj and "_fields" in obj:
            # Reconstruct the dataclass
            module_name, class_name = obj["_dataclass"].rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            dataclass_type = getattr(module, class_name)
            fields = {k: _convert_from_json(v) for k, v in obj["_fields"].items()}
            return dataclass_type(**fields)
        else:
            return {k: _convert_from_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_from_json(item) for item in obj]
    else:
        return obj


class BrainCheckpoint:
    """High-level API for brain checkpoint persistence."""

    @staticmethod
    def save(
        brain: Any,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        compression: Optional[str] = None,
        compression_level: int = 3,
        precision_policy: Union[str, PrecisionPolicy, None] = None,
    ) -> Dict[str, Any]:
        """Save brain state to binary checkpoint file.

        Args:
            brain: Brain instance (DynamicBrain)
            path: Path to save checkpoint
            metadata: Optional metadata dict
            compression: Compression type ('zstd', 'lz4', or None)
                        If None, auto-detects from file extension (.zst or .lz4)
            compression_level: Compression level (1-22 for zstd, 1-12 for lz4)
            precision_policy: Mixed precision policy ('fp32', 'fp16', 'mixed', or PrecisionPolicy)
                            'fp32': All tensors in FP32 (default)
                            'fp16': Weights/traces in FP16, critical params in FP32 (~50% size savings)
                            'mixed': Auto-detect precision based on tensor size

        Returns:
            Summary dict with file info

        Example:
            >>> # Uncompressed FP32
            >>> BrainCheckpoint.save(brain, "checkpoint.thalia")

            >>> # FP16 with zstd compression (~75% total size reduction)
            >>> BrainCheckpoint.save(brain, "checkpoint.thalia.zst", precision_policy='fp16')

            >>> # Custom precision policy
            >>> policy = PrecisionPolicy(weights='fp16', biases='fp32', membrane='fp32')
            >>> BrainCheckpoint.save(brain, "checkpoint.thalia", precision_policy=policy)
        """
        from .compression import compress_data, detect_compression

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect compression if not specified
        if compression is None:
            compression = detect_compression(path)

        # Get brain state
        state = brain.get_full_state()

        # Apply precision policy if specified
        policy = get_precision_policy(precision_policy)
        if precision_policy is not None:
            state = apply_precision_policy_to_state(state, policy, in_place=True)

        # Get precision statistics for metadata
        precision_stats = get_precision_statistics(state)

        # Prepare metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "thalia_version": THALIA_VERSION,
                "checkpoint_format_version": f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}",
                "pytorch_version": torch.__version__,
                "device": str(getattr(brain, "device", getattr(brain.config, "device", "unknown"))),
                "training_steps": state.get("training_steps", 0),
                # Include non-region state
                "config": state.get("config", {}),
                "theta": state.get("theta", {}),
                "scheduler": state.get("scheduler", {}),
                "trial_state": state.get("trial_state", {}),
                # Growth history tracking
                "growth_history": getattr(brain, "_growth_history", []),
                # Precision information
                "precision_policy": str(precision_policy) if precision_policy else "fp32",
                "precision_stats": precision_stats,
            }
        )

        # Count neurons and synapses
        total_neurons = 0
        total_synapses = 0

        for region_name, region_state in state["regions"].items():
            if "neuron_state" in region_state:
                neuron_state = region_state["neuron_state"]
                if "membrane" in neuron_state:
                    total_neurons += neuron_state["membrane"].numel()

            if "weights" in region_state:
                for weight_name, weight_tensor in region_state["weights"].items():
                    if weight_tensor is not None:
                        total_synapses += weight_tensor.numel()

        # Write to file
        with open(path, "w+b") as f:  # Use w+b for read-write
            writer = BinaryWriter(f)

            # Reserve space for header (write placeholder, will update with real header before checksum)
            header_pos = f.tell()
            f.write(b"\x00" * HEADER_SIZE)

            # Write metadata
            metadata_offset = writer.tell()
            metadata_bytes = writer.write_json(metadata)

            # Write region data and build index
            region_index = []

            for region_name, region_state in state["regions"].items():
                region_offset = writer.tell()

                # Encode region state as JSON with tensor references
                # Tensors are written to file, JSON contains their offsets
                region_json = _serialize_region_state(region_state, writer)

                # Write JSON (now all tensors are already written to file)
                json_offset = writer.tell()
                region_bytes = writer.write_json(region_json)

                _ = writer.tell() - region_offset  # region_length - unused

                region_index.append(
                    RegionIndexEntry(
                        region_name=region_name,
                        data_offset=json_offset,  # Point to JSON, not the whole region
                        data_length=region_bytes,  # Only JSON length
                    )
                )

            # Write pathway data (similar to regions)
            if "pathways" in state:
                for pathway_name, pathway_state in state["pathways"].items():
                    pathway_offset = writer.tell()

                    pathway_json = _serialize_region_state(pathway_state, writer)

                    json_offset = writer.tell()
                    pathway_bytes = writer.write_json(pathway_json)

                    _ = writer.tell() - pathway_offset  # pathway_length - unused

                    region_index.append(
                        RegionIndexEntry(
                            region_name=f"pathway:{pathway_name}",
                            data_offset=json_offset,
                            data_length=pathway_bytes,
                        )
                    )

            # Write region index
            region_index_offset = writer.tell()
            region_index_bytes = writer.write_region_index(region_index)

            # Create header with all the metadata
            header = CheckpointHeader(
                magic=MAGIC_NUMBER,
                major_version=MAJOR_VERSION,
                minor_version=MINOR_VERSION,
                patch_version=PATCH_VERSION,
                flags=0,
                timestamp=int(time.time()),
                metadata_offset=metadata_offset,
                metadata_length=metadata_bytes,
                region_index_offset=region_index_offset,
                region_index_length=region_index_bytes,
                connectivity_offset=0,  # Reserved
                connectivity_length=0,
                total_neurons=total_neurons,
                total_synapses=total_synapses,
                training_steps=state.get("training_steps", 0),
                num_regions=len(region_index),
                checksum_type=1,  # SHA256
            )

            # Go back and write the real header
            f.seek(header_pos)
            header_bytes = header.to_bytes()
            f.write(header_bytes)

        # Apply compression if requested (BEFORE checksum)
        if compression is not None:
            # Read uncompressed file
            with open(path, "rb") as f:
                uncompressed_data = f.read()

            # Compress
            compressed_data = compress_data(uncompressed_data, compression, compression_level)  # type: ignore[arg-type]

            # Overwrite with compressed version
            with open(path, "wb") as f:
                f.write(compressed_data)

            # Compute checksum on compressed data
            hasher = hashlib.sha256()
            hasher.update(compressed_data)
            checksum = hasher.digest()
        else:
            # Compute checksum on uncompressed data
            with open(path, "rb") as f:
                hasher = hashlib.sha256()
                while True:
                    chunk = f.read(65536)  # Read in 64KB chunks
                    if not chunk:
                        break
                    hasher.update(chunk)
                checksum = hasher.digest()

        # Append checksum at end of file (works for both compressed and uncompressed)
        with open(path, "ab") as f:
            f.write(checksum)

        file_size = path.stat().st_size

        return {
            "path": str(path),
            "file_size": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "num_regions": len(state["regions"]),
            "num_pathways": len(state.get("pathways", {})),
            "total_neurons": total_neurons,
            "total_synapses": total_synapses,
            "checksum": checksum.hex(),
            "compression": compression,
            "compression_level": compression_level if compression else None,
            "precision_policy": str(precision_policy) if precision_policy else "fp32",
            "precision_stats": precision_stats,
        }

    @staticmethod
    def load(
        path: Union[str, Path],
        device: str = "cpu",
        regions_to_load: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Load brain state from binary checkpoint file.

        Automatically handles compressed checkpoints (.zst, .lz4) and delta checkpoints (.delta.thalia).

        Args:
            path: Path to checkpoint file
            device: Device to load tensors to
            regions_to_load: Optional list of region names to load (None = all)

        Returns:
            State dict suitable for brain.load_full_state()

        Example:
            >>> # Load uncompressed
            >>> brain = BrainCheckpoint.load("checkpoint.thalia")

            >>> # Load compressed (auto-detects)
            >>> brain = BrainCheckpoint.load("checkpoint.thalia.zst")

            >>> # Load delta checkpoint (auto-reconstructs)
            >>> brain = BrainCheckpoint.load("stage3.delta.thalia")
        """
        from .compression import decompress_data, detect_compression
        from .delta import DELTA_MAGIC, load_delta_checkpoint

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Check for compression
        compression = detect_compression(path)

        # Read file
        with open(path, "rb") as f:
            raw_file_data = f.read()

        # Validate checksum FIRST (on compressed or uncompressed data)
        # Checksum is always the last 32 bytes
        if len(raw_file_data) < 32:
            raise ValueError("File too small to contain checksum")

        data_to_validate = raw_file_data[:-32]
        stored_checksum = raw_file_data[-32:]
        computed_checksum = hashlib.sha256(data_to_validate).digest()

        if computed_checksum != stored_checksum:
            raise ValueError("Checksum validation failed - file may be corrupted")

        # Now decompress if needed (excluding checksum bytes)
        if compression is not None:
            file_data = decompress_data(data_to_validate, compression)
        else:
            file_data = data_to_validate

        # Check if this is a delta checkpoint AFTER decompression (magic is 5 bytes: \xCE\x94THL)
        if file_data[:5] == DELTA_MAGIC:
            # This is a delta checkpoint - use special loader
            # For compressed delta, we need to decompress first, then pass to delta loader
            # But delta loader expects a path. We need to handle this differently.
            # The load_delta_checkpoint function will re-read and decompress, so pass the original path
            return load_delta_checkpoint(path, device=device)

        # Now parse the decompressed data (checksum already validated)
        f = io.BytesIO(file_data)  # type: ignore[assignment]
        reader = BinaryReader(f)

        # Read header
        header = reader.read_header()

        # Validate version compatibility
        checkpoint_version = (header.major_version, header.minor_version, header.patch_version)
        current_version = (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)

        # Major version must match (breaking changes)
        if checkpoint_version[0] != current_version[0]:
            raise ValueError(
                f"Incompatible checkpoint format version: "
                f"checkpoint is v{checkpoint_version[0]}.{checkpoint_version[1]}.{checkpoint_version[2]}, "
                f"but current Thalia supports v{current_version[0]}.{current_version[1]}.{current_version[2]}. "
                f"Major version mismatch indicates breaking changes."
            )

        # Warn if minor/patch versions differ (non-breaking changes)
        if checkpoint_version[1:] != current_version[1:]:
            warnings.warn(
                f"Checkpoint format version mismatch: "
                f"checkpoint is v{checkpoint_version[0]}.{checkpoint_version[1]}.{checkpoint_version[2]}, "
                f"current Thalia uses v{current_version[0]}.{current_version[1]}.{current_version[2]}. "
                f"Minor/patch differences are usually compatible.",
                UserWarning,
            )

        # Read metadata
        f.seek(header.metadata_offset)
        metadata_bytes = f.read(header.metadata_length)
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        # Log version info from metadata for debugging
        saved_thalia_version = metadata.get("thalia_version", "unknown")
        if saved_thalia_version != THALIA_VERSION:
            warnings.warn(
                f"Checkpoint was saved with Thalia v{saved_thalia_version}, "
                f"loading with v{THALIA_VERSION}. API changes may cause issues.",
                UserWarning,
            )

        # Read region index
        f.seek(header.region_index_offset)
        index_data = f.read(header.region_index_length)
        n_entries = len(index_data) // 48
        region_index = []
        for i in range(n_entries):
            entry_data = index_data[i * 48 : (i + 1) * 48]
            region_index.append(RegionIndexEntry.from_bytes(entry_data))

        # Load regions
        regions = {}
        pathways = {}

        for entry in region_index:
            # Skip if not in requested regions
            if regions_to_load is not None:
                if entry.region_name not in regions_to_load:
                    continue

            # Read region JSON
            f.seek(entry.data_offset)
            region_json_bytes = f.read(entry.data_length)
            region_json = json.loads(region_json_bytes.decode("utf-8"))

            # Deserialize region state (decode tensors)
            region_state = _deserialize_region_state(region_json, f, device)

            # Store in appropriate dict
            if entry.region_name.startswith("pathway:"):
                pathway_name = entry.region_name[8:]  # Remove "pathway:" prefix
                pathways[pathway_name] = region_state
            else:
                regions[entry.region_name] = region_state

        # Close BytesIO if we created it for decompression
        f.close()

        # Build full state dict
        state = {
            "regions": regions,
            "metadata": metadata,
            # Extract non-region state from metadata
            "config": metadata.get("config", {}),
            "theta": metadata.get("theta", {}),
            "scheduler": metadata.get("scheduler", {}),
            "trial_state": metadata.get("trial_state", {}),
        }

        if pathways:
            state["pathways"] = pathways

        # Restore FP16 tensors to FP32 for computation
        # (Checkpoint may have FP16 for storage, but we compute in FP32)
        if metadata.get("precision_policy") and metadata["precision_policy"] != "fp32":
            state = restore_precision_to_fp32(state, in_place=True)

        return state

    @staticmethod
    def save_delta(
        brain: Any,
        path: Union[str, Path],
        base_checkpoint: Union[str, Path],
        threshold: float = 1e-5,
        metadata: Optional[Dict[str, Any]] = None,
        compression: Optional[str] = None,
        compression_level: int = 3,
        precision_policy: Union[str, PrecisionPolicy, None] = None,
    ) -> Dict[str, Any]:
        """Save delta checkpoint (only weight changes from base).

        Huge savings during curriculum learning - typically 80-95% file size reduction.

        Args:
            brain: Brain instance
            path: Where to save delta checkpoint
            base_checkpoint: Path to base checkpoint
            threshold: Minimum weight change to store (default: 1e-5)
            metadata: Optional metadata
            compression: Optional compression ('zstd', 'lz4', or None)
            compression_level: Compression level if compression is used
            precision_policy: Mixed precision policy (same as save())

        Returns:
            Summary dict with statistics

        Example:
            >>> # Save base checkpoint first
            >>> BrainCheckpoint.save(brain, "stage0.thalia")

            >>> # Train to stage 1
            >>> train_stage(brain, stage=1)

            >>> # Save delta with FP16 + compression (massive savings)
            >>> BrainCheckpoint.save_delta(
            ...     brain,
            ...     "stage1.delta.thalia.zst",
            ...     base_checkpoint="stage0.thalia",
            ...     precision_policy='fp16'
            ... )
        """
        from .compression import compress_file
        from .delta import save_delta_checkpoint

        path = Path(path)
        base_checkpoint = Path(base_checkpoint)

        # Get current state
        current_state = brain.get_full_state()

        # Apply precision policy if specified
        if precision_policy is not None:
            policy = get_precision_policy(precision_policy)
            current_state = apply_precision_policy_to_state(current_state, policy, in_place=True)

        # Save delta checkpoint
        summary = save_delta_checkpoint(
            current_state=current_state,
            base_checkpoint_path=base_checkpoint,
            output_path=path,
            threshold=threshold,
            metadata=metadata,
        )

        # Apply compression if requested
        if compression is not None:
            uncompressed_path = path

            if compression == "zstd":
                compressed_path = path.with_suffix(path.suffix + ".zst")
            elif compression == "lz4":
                compressed_path = path.with_suffix(path.suffix + ".lz4")
            else:
                raise ValueError(f"Unknown compression: {compression}")

            compress_file(
                uncompressed_path,
                compressed_path,
                compression=compression,  # type: ignore[arg-type]
                level=compression_level,
            )

            # Remove uncompressed version
            uncompressed_path.unlink()

            # Update summary
            compressed_size = compressed_path.stat().st_size
            summary["compressed_path"] = str(compressed_path)
            summary["compressed_size_mb"] = compressed_size / (1024 * 1024)
            summary["final_compression_ratio"] = (
                compressed_size / summary["base_size_mb"] / (1024 * 1024)
            )
            summary["final_savings_percent"] = (1 - summary["final_compression_ratio"]) * 100

        return summary

    @staticmethod
    def info(path: Union[str, Path]) -> Dict[str, Any]:
        """Get checkpoint info without loading full state.

        Args:
            path: Path to checkpoint file

        Returns:
            Info dict with metadata and summary
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, "rb") as f:
            reader = BinaryReader(f)

            # Read header
            header = reader.read_header()

            # Read metadata
            metadata = reader.read_json(header.metadata_offset, header.metadata_length)

            # Read region index
            region_index = reader.read_region_index(
                header.region_index_offset, header.region_index_length
            )

        return {
            "file_path": str(path),
            "file_size": path.stat().st_size,
            "version": f"{header.major_version}.{header.minor_version}.{header.patch_version}",
            "timestamp": datetime.fromtimestamp(header.timestamp).isoformat(),
            "num_regions": header.num_regions,
            "total_neurons": header.total_neurons,
            "total_synapses": header.total_synapses,
            "training_steps": header.training_steps,
            "regions": [entry.region_name for entry in region_index],
            "metadata": metadata,
        }

    @staticmethod
    def validate(path: Union[str, Path]) -> Dict[str, Any]:
        """Validate checkpoint file integrity.

        Args:
            path: Path to checkpoint file

        Returns:
            Validation result dict
        """
        path = Path(path)
        issues = []

        if not path.exists():
            return {
                "valid": False,
                "issues": [f"File not found: {path}"],
            }

        try:
            with open(path, "rb") as f:
                # Validate checksum FIRST (same as load)
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                f.seek(0)  # Back to start

                # Read everything except the 32-byte checksum at the end
                data_to_hash = f.read(file_size - 32)
                computed_hash = hashlib.sha256(data_to_hash).digest()

                # Read stored checksum
                stored_checksum = f.read(32)

                if computed_hash != stored_checksum:
                    issues.append("Checksum validation failed")

                # Validate header
                f.seek(0)
                header_data = f.read(HEADER_SIZE)
                header = CheckpointHeader.from_bytes(header_data)
                is_valid, header_issues = header.validate()
                issues.extend(header_issues)

        except Exception as e:
            issues.append(f"Error reading file: {e}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }


def _serialize_value(value: Any, writer: BinaryWriter) -> Any:
    """Recursively serialize a single value (tensor, dict, list, or primitive).

    Args:
        value: Value to serialize
        writer: Binary writer

    Returns:
        JSON-serializable representation
    """
    if isinstance(value, torch.Tensor):
        # Encode tensor inline
        tensor_offset = writer.tell()
        tensor_bytes = encode_tensor(value, writer.file)
        return {
            "_type": "tensor",
            "_offset": tensor_offset,
            "_bytes": tensor_bytes,
        }
    elif isinstance(value, dict):
        # Recursively serialize nested dicts
        return {k: _serialize_value(v, writer) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        # Handle lists and tuples
        return [_serialize_value(item, writer) for item in value]
    elif is_dataclass(value) or isinstance(value, Enum) or isinstance(value, torch.device):
        # Convert special types to JSON-serializable form
        return _convert_to_json_serializable(value)
    elif hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, type(None))):
        # Object with attributes - serialize its __dict__
        return {
            "_type": "object_dict",
            "_class": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "_dict": _serialize_value(value.__dict__, writer),
        }
    else:
        # Plain JSON-serializable value (int, float, str, bool, None)
        return value


def _serialize_region_state(region_state: Dict[str, Any], writer: BinaryWriter) -> Dict[str, Any]:
    """Serialize region state, encoding tensors inline.

    Args:
        region_state: Region state dict
        writer: Binary writer

    Returns:
        JSON dict with tensor references
    """
    return _serialize_value(region_state, writer)


def _deserialize_list(lst: list, file, device: str) -> list:
    """Deserialize list, handling tensors."""
    result = []

    for item in lst:
        if isinstance(item, dict):
            if item.get("_type") == "tensor":
                # Decode tensor - seek and read
                file.seek(item["_offset"])
                result.append(decode_tensor(file, device))
            else:
                result.append(_deserialize_region_state(item, file, device))
        elif isinstance(item, list):
            result.append(_deserialize_list(item, file, device))
        else:
            result.append(item)

    return result


def _deserialize_region_state(json_dict: Dict[str, Any], file, device: str) -> Dict[str, Any]:
    """Deserialize region state, decoding tensors and reconstructing dataclasses.

    Args:
        json_dict: JSON dict with tensor references
        file: Binary file handle
        device: Device to load tensors to

    Returns:
        Deserialized state dict
    """
    state = {}

    for key, value in json_dict.items():
        if isinstance(value, dict):
            if value.get("_type") == "tensor":
                # Decode tensor - seek and read
                file.seek(value["_offset"])
                state[key] = decode_tensor(file, device)
            elif value.get("_dataclass") is not None:
                # Reconstruct dataclass
                state[key] = _convert_from_json(value)
            else:
                # Recursively deserialize nested dict
                state[key] = _deserialize_region_state(value, file, device)
        elif isinstance(value, list):
            # Handle lists
            state[key] = _deserialize_list(value, file, device)
        else:
            # Plain value
            state[key] = value

    return state
