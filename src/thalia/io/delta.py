"""
Delta Checkpoint Implementation - Store only weight differences.

Huge savings during curriculum learning where most weights remain stable.
Typical savings: 80-95% file size for checkpoints after stage 0.

File Structure:
    base_checkpoint.thalia           # Full checkpoint (stage 0)
    stage1.delta.thalia              # Only differences from base
    stage2.delta.thalia              # Only differences from stage1
    stage3.delta.thalia              # Only differences from stage2

Delta Format:
    [DELTA_HEADER]      64 bytes
    [BASE_REFERENCE]    Variable (SHA-256 hash of base checkpoint)
    [REGION_DELTAS]     Multiple regions (only changed weights)

Usage:
    # Initial save (base)
    BrainCheckpoint.save(brain, "stage0.thalia")
    
    # Subsequent saves (delta)
    BrainCheckpoint.save_delta(
        brain,
        "stage1.delta.thalia",
        base_checkpoint="stage0.thalia"
    )
    
    # Load (automatically reconstructs from base + deltas)
    brain = BrainCheckpoint.load("stage3.delta.thalia")
"""

import struct
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

import torch


# Delta magic number
DELTA_MAGIC = b'\xCE\x94THL'  # Δ in UTF-8 + THL

DELTA_VERSION = 1


@dataclass
class DeltaHeader:
    """Delta checkpoint header (64 bytes)."""
    
    magic: bytes  # 5 bytes - "ΔTHL" (Δ is 2 bytes in UTF-8)
    delta_version: int  # 4 bytes
    base_checkpoint_hash: bytes  # 32 bytes (SHA-256)
    base_step: int  # 8 bytes
    current_step: int  # 8 bytes
    # reserved: 7 bytes (to pad to 64 bytes)
    
    def to_bytes(self) -> bytes:
        """Serialize to 64 bytes."""
        data = struct.pack(
            '<5sI32sQQ',  # 5 + 4 + 32 + 8 + 8 = 57 bytes
            self.magic,
            self.delta_version,
            self.base_checkpoint_hash,
            self.base_step,
            self.current_step,
        )
        # Pad to 64 bytes (need 7 more bytes)
        return data + b'\x00' * (64 - len(data))
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'DeltaHeader':
        """Deserialize from 64 bytes."""
        if len(data) < 64:
            raise ValueError(f"Delta header too short: {len(data)} < 64")
        
        fields = struct.unpack('<5sI32sQQ', data[:57])  # Read 57 bytes
        
        return cls(
            magic=fields[0],
            delta_version=fields[1],
            base_checkpoint_hash=fields[2],
            base_step=fields[3],
            current_step=fields[4],
        )


def compute_file_hash(path: Union[str, Path]) -> bytes:
    """Compute SHA-256 hash of entire file.
    
    Args:
        path: Path to file
        
    Returns:
        32-byte SHA-256 hash
    """
    hasher = hashlib.sha256()
    
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    
    return hasher.digest()


def compute_weight_delta(
    current_weights: torch.Tensor,
    base_weights: torch.Tensor,
    threshold: float = 1e-5,
) -> Optional[Dict[str, Any]]:
    """Compute sparse delta between weight matrices.
    
    Only stores weights that changed by more than threshold.
    
    Args:
        current_weights: Current weight tensor
        base_weights: Base weight tensor
        threshold: Minimum change to store (absolute difference)
        
    Returns:
        Dict with sparse delta info, or None if no significant changes
    """
    if current_weights.shape != base_weights.shape:
        # Shape changed (growth) - store full tensor
        return {
            'type': 'full',
            'tensor': current_weights,
            'shape': list(current_weights.shape),
        }
    
    # Compute differences
    delta = current_weights - base_weights
    changed_mask = delta.abs() > threshold
    
    # If <5% changed, store sparse delta
    change_ratio = changed_mask.sum().item() / changed_mask.numel()
    
    if change_ratio < 0.05:
        # Sparse delta: store only changed indices and values
        indices = changed_mask.nonzero(as_tuple=False)  # [n_changed, ndim]
        values = delta[changed_mask]  # [n_changed]
        
        if len(indices) == 0:
            # No changes
            return None
        
        return {
            'type': 'sparse',
            'indices': indices.cpu(),
            'values': values.cpu(),
            'shape': list(current_weights.shape),
        }
    else:
        # Many changes - store full tensor is more efficient
        return {
            'type': 'full',
            'tensor': current_weights,
            'shape': list(current_weights.shape),
        }


def apply_weight_delta(
    base_weights: torch.Tensor,
    delta_info: Dict[str, Any],
    device: str = 'cpu',
) -> torch.Tensor:
    """Apply delta to base weights to reconstruct current weights.
    
    Args:
        base_weights: Base weight tensor
        delta_info: Delta information from compute_weight_delta()
        device: Device to place result on
        
    Returns:
        Reconstructed current weights
    """
    if delta_info['type'] == 'full':
        # Full replacement
        return delta_info['tensor'].to(device)
    
    elif delta_info['type'] == 'sparse':
        # Sparse delta: start with base, apply changes
        result = base_weights.clone().to(device)
        indices = delta_info['indices']
        values = delta_info['values'].to(device)
        
        # Apply changes (handle multi-dimensional indexing)
        if indices.ndim == 2:
            # Multi-dimensional tensor
            for i in range(len(indices)):
                idx = tuple(indices[i].tolist())
                result[idx] += values[i]
        else:
            # 1D tensor
            result[indices] += values
        
        return result
    
    else:
        raise ValueError(f"Unknown delta type: {delta_info['type']}")


def compute_state_delta(
    current_state: Dict[str, Any],
    base_state: Dict[str, Any],
    threshold: float = 1e-5,
) -> Dict[str, Any]:
    """Compute delta between two brain states.
    
    Only includes regions with changed weights or grown neurons.
    
    Args:
        current_state: Current brain state (from get_full_state())
        base_state: Base brain state
        threshold: Minimum weight change threshold
        
    Returns:
        Delta state with only changes
    """
    delta = {
        'regions': {},
        'pathways': {},
        'metadata_changes': {},
    }
    
    # Compare regions
    current_regions = current_state.get('regions', {})
    base_regions = base_state.get('regions', {})
    
    for region_name, current_region in current_regions.items():
        if region_name not in base_regions:
            # New region - store full state
            delta['regions'][region_name] = {
                'type': 'new',
                'state': current_region,
            }
            continue
        
        base_region = base_regions[region_name]
        region_delta = {}
        has_changes = False
        
        # Compare weights
        current_weights = current_region.get('weights', {})
        base_weights = base_region.get('weights', {})
        
        weight_deltas = {}
        for weight_name, current_tensor in current_weights.items():
            if current_tensor is None:
                continue
            
            if weight_name not in base_weights or base_weights[weight_name] is None:
                # New weight matrix
                weight_deltas[weight_name] = {
                    'type': 'full',
                    'tensor': current_tensor,
                    'shape': list(current_tensor.shape),
                }
                has_changes = True
            else:
                # Compute delta
                weight_delta = compute_weight_delta(
                    current_tensor,
                    base_weights[weight_name],
                    threshold=threshold,
                )
                
                if weight_delta is not None:
                    weight_deltas[weight_name] = weight_delta
                    has_changes = True
        
        if has_changes:
            region_delta['weights'] = weight_deltas
            
            # Include non-weight state (always include for changed regions)
            region_delta['config'] = current_region.get('config')
            region_delta['neuron_state'] = current_region.get('neuron_state')
            region_delta['learning_state'] = current_region.get('learning_state')
            region_delta['oscillator_state'] = current_region.get('oscillator_state')
            region_delta['neuromodulator_state'] = current_region.get('neuromodulator_state')
            
            delta['regions'][region_name] = region_delta
    
    # Compare pathways (similar logic)
    current_pathways = current_state.get('pathways', {})
    base_pathways = base_state.get('pathways', {})
    
    for pathway_name, current_pathway in current_pathways.items():
        if pathway_name not in base_pathways:
            delta['pathways'][pathway_name] = {
                'type': 'new',
                'state': current_pathway,
            }
            continue
        
        base_pathway = base_pathways[pathway_name]
        pathway_delta = {}
        has_changes = False
        
        current_weights = current_pathway.get('weights', {})
        base_weights = base_pathway.get('weights', {})
        
        weight_deltas = {}
        for weight_name, current_tensor in current_weights.items():
            if current_tensor is None:
                continue
            
            if weight_name not in base_weights or base_weights[weight_name] is None:
                weight_deltas[weight_name] = {
                    'type': 'full',
                    'tensor': current_tensor,
                    'shape': list(current_tensor.shape),
                }
                has_changes = True
            else:
                weight_delta = compute_weight_delta(
                    current_tensor,
                    base_weights[weight_name],
                    threshold=threshold,
                )
                
                if weight_delta is not None:
                    weight_deltas[weight_name] = weight_delta
                    has_changes = True
        
        if has_changes:
            pathway_delta['weights'] = weight_deltas
            pathway_delta['config'] = current_pathway.get('config')
            pathway_delta['neuron_state'] = current_pathway.get('neuron_state')
            pathway_delta['learning_state'] = current_pathway.get('learning_state')
            
            delta['pathways'][pathway_name] = pathway_delta
    
    # Store current training step
    delta['training_steps'] = current_state.get('training_steps', 0)
    
    return delta


def reconstruct_state_from_delta(
    base_state: Dict[str, Any],
    delta_state: Dict[str, Any],
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Reconstruct full state by applying delta to base.
    
    Args:
        base_state: Base brain state
        delta_state: Delta from compute_state_delta()
        device: Device to place tensors on
        
    Returns:
        Reconstructed full state
    """
    import copy
    
    # Start with deep copy of base
    reconstructed = copy.deepcopy(base_state)
    
    # Apply region deltas
    for region_name, region_delta in delta_state.get('regions', {}).items():
        if region_delta.get('type') == 'new':
            # New region - use full state
            reconstructed['regions'][region_name] = region_delta['state']
        else:
            # Apply weight deltas
            base_region = reconstructed['regions'][region_name]
            
            for weight_name, weight_delta in region_delta.get('weights', {}).items():
                if weight_name in base_region['weights']:
                    base_region['weights'][weight_name] = apply_weight_delta(
                        base_region['weights'][weight_name],
                        weight_delta,
                        device=device,
                    )
                else:
                    # New weight
                    base_region['weights'][weight_name] = weight_delta['tensor'].to(device)
            
            # Update non-weight state
            if 'neuron_state' in region_delta:
                base_region['neuron_state'] = region_delta['neuron_state']
            if 'learning_state' in region_delta:
                base_region['learning_state'] = region_delta['learning_state']
            if 'oscillator_state' in region_delta:
                base_region['oscillator_state'] = region_delta['oscillator_state']
            if 'neuromodulator_state' in region_delta:
                base_region['neuromodulator_state'] = region_delta['neuromodulator_state']
    
    # Apply pathway deltas (similar logic)
    for pathway_name, pathway_delta in delta_state.get('pathways', {}).items():
        if pathway_delta.get('type') == 'new':
            reconstructed['pathways'][pathway_name] = pathway_delta['state']
        else:
            base_pathway = reconstructed['pathways'][pathway_name]
            
            for weight_name, weight_delta in pathway_delta.get('weights', {}).items():
                if weight_name in base_pathway['weights']:
                    base_pathway['weights'][weight_name] = apply_weight_delta(
                        base_pathway['weights'][weight_name],
                        weight_delta,
                        device=device,
                    )
                else:
                    base_pathway['weights'][weight_name] = weight_delta['tensor'].to(device)
            
            if 'neuron_state' in pathway_delta:
                base_pathway['neuron_state'] = pathway_delta['neuron_state']
            if 'learning_state' in pathway_delta:
                base_pathway['learning_state'] = pathway_delta['learning_state']
    
    # Update training steps
    if 'training_steps' in delta_state:
        reconstructed['training_steps'] = delta_state['training_steps']
    
    return reconstructed


def save_delta_checkpoint(
    current_state: Dict[str, Any],
    base_checkpoint_path: Union[str, Path],
    output_path: Union[str, Path],
    threshold: float = 1e-5,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save delta checkpoint.
    
    Args:
        current_state: Current brain state
        base_checkpoint_path: Path to base checkpoint
        output_path: Where to save delta
        threshold: Minimum weight change threshold
        metadata: Optional metadata
        
    Returns:
        Summary dict with statistics
    """
    from .checkpoint import BrainCheckpoint
    
    base_checkpoint_path = Path(base_checkpoint_path)
    output_path = Path(output_path)
    
    # Load base state (returns dict, not brain object)
    base_state = BrainCheckpoint.load(base_checkpoint_path)
    
    # Compute delta
    delta_state = compute_state_delta(current_state, base_state, threshold=threshold)
    
    # Compute base checkpoint hash
    base_hash = compute_file_hash(base_checkpoint_path)
    
    # Create delta header
    header = DeltaHeader(
        magic=DELTA_MAGIC,
        delta_version=DELTA_VERSION,
        base_checkpoint_hash=base_hash,
        base_step=base_state.get('training_steps', 0),
        current_step=current_state.get('training_steps', 0),
    )
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    metadata['delta_info'] = {
        'base_checkpoint': str(base_checkpoint_path),
        'base_hash': base_hash.hex(),
        'base_step': header.base_step,
        'current_step': header.current_step,
        'threshold': threshold,
    }
    
    # Write delta file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(header.to_bytes())
        
        # Write delta state as JSON (with tensor encoding)
        import pickle
        pickle.dump(delta_state, f)
        
        # Write metadata
        metadata_json = json.dumps(metadata, indent=2).encode('utf-8')
        f.write(struct.pack('<I', len(metadata_json)))
        f.write(metadata_json)
    
    # Compute statistics
    base_size = base_checkpoint_path.stat().st_size
    delta_size = output_path.stat().st_size
    compression_factor = base_size / delta_size if delta_size > 0 else 1.0
    savings_percent = (1 - (delta_size / base_size)) * 100 if base_size > 0 else 0.0
    
    num_changed_regions = len(delta_state['regions'])
    num_changed_pathways = len(delta_state['pathways'])
    
    return {
        'base_checkpoint': str(base_checkpoint_path),
        'delta_checkpoint': str(output_path),
        'base_size_mb': base_size / (1024 * 1024),
        'delta_size_mb': delta_size / (1024 * 1024),
        'compression_ratio': compression_factor,  # e.g., 3.0 means 3x smaller
        'savings_percent': savings_percent,  # e.g., 66.7 means 66.7% savings
        'changed_regions': num_changed_regions,
        'changed_pathways': num_changed_pathways,
    }


def load_delta_checkpoint(
    delta_path: Union[str, Path],
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Load delta checkpoint and reconstruct full state.
    
    Automatically finds and loads base checkpoint.
    
    Args:
        delta_path: Path to delta checkpoint (can be compressed)
        device: Device to place tensors on
        
    Returns:
        Reconstructed full state
    """
    from .checkpoint import BrainCheckpoint
    from .compression import detect_compression, decompress_data
    import pickle
    import io
    
    delta_path = Path(delta_path)
    
    # Check for compression and read file
    compression = detect_compression(delta_path)
    
    with open(delta_path, 'rb') as f:
        file_data = f.read()
    
    if compression is not None:
        file_data = decompress_data(file_data, compression)
    
    # Parse decompressed data
    f = io.BytesIO(file_data)
    
    # Read header
    header_bytes = f.read(64)
    header = DeltaHeader.from_bytes(header_bytes)
    
    if header.magic != DELTA_MAGIC:
        raise ValueError(f"Not a delta checkpoint: {delta_path}")
    
    # Read delta state
    delta_state = pickle.load(f)
    
    # Read metadata
    metadata_length_bytes = f.read(4)
    metadata_length = struct.unpack('<I', metadata_length_bytes)[0]
    metadata_json = f.read(metadata_length)
    metadata = json.loads(metadata_json.decode('utf-8'))
    
    # Find base checkpoint
    base_checkpoint_path = metadata['delta_info']['base_checkpoint']
    base_checkpoint_path = Path(base_checkpoint_path)
    
    if not base_checkpoint_path.is_absolute():
        # Try relative to delta checkpoint
        base_checkpoint_path = delta_path.parent / base_checkpoint_path
    
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_checkpoint_path}\n"
            f"Delta checkpoint references: {metadata['delta_info']['base_checkpoint']}"
        )
    
    # Verify base checkpoint hash
    base_hash = compute_file_hash(base_checkpoint_path)
    expected_hash = bytes.fromhex(metadata['delta_info']['base_hash'])
    
    if base_hash != expected_hash:
        raise ValueError(
            f"Base checkpoint hash mismatch!\n"
            f"Expected: {expected_hash.hex()}\n"
            f"Got: {base_hash.hex()}\n"
            f"The base checkpoint may have been modified or corrupted."
        )
    
    # Load base state (returns dict, not brain object)
    base_state = BrainCheckpoint.load(base_checkpoint_path, device=device)
    
    # Reconstruct full state
    reconstructed_state = reconstruct_state_from_delta(base_state, delta_state, device=device)
    
    # Restore FP16 tensors to FP32 (delta may have been saved with FP16)
    # Safe to modify in-place since reconstructed_state is already a fresh copy
    from .precision import restore_precision_to_fp32
    reconstructed_state = restore_precision_to_fp32(reconstructed_state, in_place=True)
    
    return reconstructed_state
