"""
Tensor Encoding - Efficient serialization for PyTorch tensors.

Supports:
- Dense tensors (float32, float64, int32, int64, bool)
- Sparse COO tensors
- Automatic sparsity detection

Encoding format for each tensor:
    [4 bytes] Encoding type (0=dense, 1=sparse_coo)
    [4 bytes] dtype code (0=float32, 1=float64, 2=int32, 3=int64, 4=bool)
    [4 bytes] ndim
    [4*ndim bytes] shape
    [variable] data
"""

from __future__ import annotations

import struct
from enum import IntEnum
from typing import BinaryIO, Tuple

import torch


class EncodingType(IntEnum):
    """Tensor encoding types."""

    DENSE = 0
    SPARSE_COO = 1


class DType(IntEnum):
    """Supported data types."""

    FLOAT32 = 0
    FLOAT64 = 1
    INT32 = 2
    INT64 = 3
    BOOL = 4
    FLOAT16 = 5  # Half precision


# PyTorch dtype to code mapping
DTYPE_TO_CODE = {
    torch.float32: DType.FLOAT32,
    torch.float64: DType.FLOAT64,
    torch.int32: DType.INT32,
    torch.int64: DType.INT64,
    torch.bool: DType.BOOL,
    torch.float16: DType.FLOAT16,
}

CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}


def encode_tensor(tensor: torch.Tensor, file: BinaryIO, sparsity_threshold: float = 0.1) -> int:
    """Encode tensor to binary format.

    Args:
        tensor: PyTorch tensor to encode
        file: Binary file to write to
        sparsity_threshold: If >90% zeros, use sparse encoding

    Returns:
        Number of bytes written
    """
    bytes_written = 0

    # Determine if we should use sparse encoding
    is_sparse = tensor.is_sparse
    if not is_sparse and tensor.numel() > 100:
        # Check if dense tensor is mostly zeros
        zero_ratio = (tensor == 0).float().mean().item()
        is_sparse = zero_ratio > (1.0 - sparsity_threshold)

    # Write encoding type
    encoding_type = EncodingType.SPARSE_COO if is_sparse else EncodingType.DENSE
    file.write(struct.pack("<I", encoding_type))
    bytes_written += 4

    # Write dtype
    dtype_code = DTYPE_TO_CODE.get(tensor.dtype)
    if dtype_code is None:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    file.write(struct.pack("<I", dtype_code))
    bytes_written += 4

    # Write shape
    ndim = len(tensor.shape)
    file.write(struct.pack("<I", ndim))
    bytes_written += 4

    for dim_size in tensor.shape:
        file.write(struct.pack("<I", dim_size))
        bytes_written += 4

    # Write data based on encoding
    if encoding_type == EncodingType.SPARSE_COO:
        bytes_written += _encode_sparse_coo(tensor, file)
    else:
        bytes_written += _encode_dense(tensor, file)

    return bytes_written


def _encode_dense(tensor: torch.Tensor, file: BinaryIO) -> int:
    """Encode dense tensor data."""
    # Convert to contiguous CPU tensor
    data = tensor.detach().cpu().contiguous()

    # Write as raw bytes
    raw_bytes = data.numpy().tobytes()
    file.write(raw_bytes)

    return len(raw_bytes)


def _encode_sparse_coo(tensor: torch.Tensor, file: BinaryIO) -> int:
    """Encode sparse COO tensor data."""
    bytes_written = 0

    # Convert to sparse COO if not already
    if not tensor.is_sparse:
        # Find nonzero indices
        indices = tensor.nonzero(as_tuple=False).t()
        values = tensor[tuple(indices)]
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, tensor.shape, dtype=tensor.dtype
        ).coalesce()
    else:
        sparse_tensor = tensor.coalesce()

    # Get indices and values
    indices = sparse_tensor.indices().cpu()  # [ndim, nnz]
    values = sparse_tensor.values().cpu()  # [nnz]

    nnz = values.numel()

    # Write nnz (number of nonzero elements)
    file.write(struct.pack("<Q", nnz))
    bytes_written += 8

    # Write indices [ndim, nnz]
    indices_bytes = indices.to(torch.int64).numpy().tobytes()
    file.write(indices_bytes)
    bytes_written += len(indices_bytes)

    # Write values [nnz] - detach first to handle tensors with gradients
    values_bytes = values.detach().numpy().tobytes()
    file.write(values_bytes)
    bytes_written += len(values_bytes)

    return bytes_written


def decode_tensor(file: BinaryIO, device: str = "cpu") -> torch.Tensor:
    """Decode tensor from binary format.

    Args:
        file: Binary file to read from
        device: Device to place tensor on

    Returns:
        Decoded PyTorch tensor
    """
    # Read encoding type
    encoding_type = struct.unpack("<I", file.read(4))[0]

    # Read dtype
    dtype_code = struct.unpack("<I", file.read(4))[0]
    dtype = CODE_TO_DTYPE[dtype_code]

    # Read shape
    ndim = struct.unpack("<I", file.read(4))[0]
    shape = tuple(struct.unpack("<I", file.read(4))[0] for _ in range(ndim))

    # Decode data based on encoding type
    if encoding_type == EncodingType.SPARSE_COO:
        tensor = _decode_sparse_coo(file, shape, dtype, device)
        # Convert to dense to avoid copy_() errors when loading into dense tensors
        tensor = tensor.to_dense()
    else:
        tensor = _decode_dense(file, shape, dtype, device)

    return tensor


def _decode_dense(
    file: BinaryIO, shape: Tuple[int, ...], dtype: torch.dtype, device: str
) -> torch.Tensor:
    """Decode dense tensor data."""
    import numpy as np

    # Calculate number of bytes to read
    numel = 1
    for dim in shape:
        numel *= dim

    # Map dtype to numpy dtype
    if dtype == torch.float32:
        np_dtype = np.float32
        bytes_per_elem = 4
    elif dtype == torch.float64:
        np_dtype = np.float64
        bytes_per_elem = 8
    elif dtype == torch.float16:
        np_dtype = np.float16
        bytes_per_elem = 2
    elif dtype == torch.int32:
        np_dtype = np.int32
        bytes_per_elem = 4
    elif dtype == torch.int64:
        np_dtype = np.int64
        bytes_per_elem = 8
    elif dtype == torch.bool:
        np_dtype = np.bool_
        bytes_per_elem = 1
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Read raw bytes
    raw_bytes = file.read(numel * bytes_per_elem)

    # Convert to numpy array
    np_array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(np_array.copy()).to(device)

    return tensor


def _decode_sparse_coo(
    file: BinaryIO, shape: Tuple[int, ...], dtype: torch.dtype, device: str
) -> torch.Tensor:
    """Decode sparse COO tensor data."""
    import numpy as np

    # Read nnz
    nnz = struct.unpack("<Q", file.read(8))[0]

    if nnz == 0:
        # Empty sparse tensor
        return torch.sparse_coo_tensor(
            torch.zeros((len(shape), 0), dtype=torch.int64),
            torch.zeros(0, dtype=dtype),
            shape,
            device=device,
        )

    # Read indices [ndim, nnz]
    ndim = len(shape)
    indices_bytes = file.read(ndim * nnz * 8)  # int64 = 8 bytes
    indices_np = np.frombuffer(indices_bytes, dtype=np.int64).reshape(ndim, nnz)
    indices = torch.from_numpy(indices_np.copy())

    # Read values [nnz]
    if dtype == torch.float32:
        np_dtype = np.float32
        bytes_per_elem = 4
    elif dtype == torch.float64:
        np_dtype = np.float64
        bytes_per_elem = 8
    elif dtype == torch.int32:
        np_dtype = np.int32
        bytes_per_elem = 4
    elif dtype == torch.int64:
        np_dtype = np.int64
        bytes_per_elem = 8
    elif dtype == torch.bool:
        np_dtype = np.bool_
        bytes_per_elem = 1
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    values_bytes = file.read(nnz * bytes_per_elem)
    values_np = np.frombuffer(values_bytes, dtype=np_dtype)
    values = torch.from_numpy(values_np.copy())

    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device)

    return sparse_tensor.coalesce()


def estimate_encoding_size(tensor: torch.Tensor, sparsity_threshold: float = 0.1) -> int:
    """Estimate bytes required to encode tensor.

    Args:
        tensor: Tensor to estimate
        sparsity_threshold: Threshold for sparse encoding

    Returns:
        Estimated bytes
    """
    # Header: encoding_type (4) + dtype (4) + ndim (4) + shape (4*ndim)
    header_size = 12 + 4 * len(tensor.shape)

    # Check if sparse encoding would be used
    is_sparse = tensor.is_sparse
    if not is_sparse and tensor.numel() > 100:
        zero_ratio = (tensor == 0).float().mean().item()
        is_sparse = zero_ratio > (1.0 - sparsity_threshold)

    if is_sparse:
        # Sparse: nnz (8) + indices (ndim * nnz * 8) + values (nnz * bytes_per_elem)
        if tensor.is_sparse:
            nnz = tensor._nnz()
        else:
            nnz = (tensor != 0).sum().item()

        bytes_per_elem = tensor.element_size()
        ndim = len(tensor.shape)
        data_size = 8 + ndim * nnz * 8 + nnz * bytes_per_elem
    else:
        # Dense: numel * bytes_per_elem
        data_size = tensor.numel() * tensor.element_size()

    return header_size + data_size
