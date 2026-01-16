"""
Compression utilities for Thalia checkpoints.

Supports zstd (best compression ratio) and lz4 (fastest compression).
Automatically detects compression from file extension or header flags.

Usage:
    # Automatic from extension
    BrainCheckpoint.save(brain, "checkpoint.thalia.zst")  # zstd
    BrainCheckpoint.save(brain, "checkpoint.thalia.lz4")  # lz4

    # Explicit
    BrainCheckpoint.save(brain, "checkpoint.thalia", compression='zstd', compression_level=3)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union, BinaryIO, Literal

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


CompressionType = Literal['zstd', 'lz4', None]


class CompressionError(Exception):
    """Raised when compression/decompression fails."""
    pass


def detect_compression(path: Union[str, Path]) -> CompressionType:
    """Detect compression type from file extension or magic bytes.

    First checks file extension (.zst, .lz4), then checks magic bytes
    at the start of the file if extension doesn't indicate compression.

    Args:
        path: File path

    Returns:
        'zstd' if .zst extension or zstd magic bytes (28 b5 2f fd)
        'lz4' if .lz4 extension or lz4 magic bytes (04 22 4d 18)
        None otherwise
    """
    path = Path(path)

    # Check extension first (fast)
    if path.suffix == '.zst':
        return 'zstd'
    elif path.suffix == '.lz4':
        return 'lz4'

    # If no extension indicator, check magic bytes
    if path.exists():
        try:
            with open(path, 'rb') as f:
                magic = f.read(4)

            # zstd magic: 28 b5 2f fd (little-endian 0xFD2FB528)
            if magic == b'\x28\xb5\x2f\xfd':
                return 'zstd'
            # lz4 magic: 04 22 4d 18
            elif magic == b'\x04\x22\x4d\x18':
                return 'lz4'
        except (IOError, OSError):
            pass

    return None


def compress_data(
    data: bytes,
    compression: CompressionType = 'zstd',
    level: int = 3,
) -> bytes:
    """Compress data using specified algorithm.

    Args:
        data: Raw bytes to compress
        compression: Compression algorithm ('zstd', 'lz4', or None)
        level: Compression level (1-22 for zstd, 1-12 for lz4)

    Returns:
        Compressed bytes

    Raises:
        CompressionError: If compression library not available or fails
    """
    if compression is None:
        return data

    if compression == 'zstd':
        if not ZSTD_AVAILABLE:
            raise CompressionError(
                "zstd compression requested but zstandard package not installed. "
                "Install with: pip install zstandard"
            )

        try:
            compressor = zstd.ZstdCompressor(level=level)
            return compressor.compress(data)
        except Exception as e:
            raise CompressionError(f"zstd compression failed: {e}")

    elif compression == 'lz4':
        if not LZ4_AVAILABLE:
            raise CompressionError(
                "lz4 compression requested but lz4 package not installed. "
                "Install with: pip install lz4"
            )

        try:
            return lz4.frame.compress(data, compression_level=level)
        except Exception as e:
            raise CompressionError(f"lz4 compression failed: {e}")

    else:
        raise ValueError(f"Unknown compression type: {compression}")


def decompress_data(
    data: bytes,
    compression: CompressionType,
) -> bytes:
    """Decompress data using specified algorithm.

    Args:
        data: Compressed bytes
        compression: Compression algorithm ('zstd', 'lz4', or None)

    Returns:
        Decompressed bytes

    Raises:
        CompressionError: If decompression library not available or fails
    """
    if compression is None:
        return data

    if compression == 'zstd':
        if not ZSTD_AVAILABLE:
            raise CompressionError(
                "zstd decompression required but zstandard package not installed. "
                "Install with: pip install zstandard"
            )

        try:
            decompressor = zstd.ZstdDecompressor()
            return decompressor.decompress(data)
        except Exception as e:
            raise CompressionError(f"zstd decompression failed: {e}")

    elif compression == 'lz4':
        if not LZ4_AVAILABLE:
            raise CompressionError(
                "lz4 decompression required but lz4 package not installed. "
                "Install with: pip install lz4"
            )

        try:
            return lz4.frame.decompress(data)
        except Exception as e:
            raise CompressionError(f"lz4 decompression failed: {e}")

    else:
        raise ValueError(f"Unknown compression type: {compression}")


def compress_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    compression: CompressionType = 'zstd',
    level: int = 3,
) -> Path:
    """Compress an existing checkpoint file.

    Args:
        input_path: Path to uncompressed checkpoint
        output_path: Optional output path (defaults to input_path + extension)
        compression: Compression algorithm
        level: Compression level

    Returns:
        Path to compressed file

    Example:
        >>> compress_file("checkpoint.thalia", compression='zstd', level=9)
        Path("checkpoint.thalia.zst")
    """
    input_path = Path(input_path)

    if output_path is None:
        if compression == 'zstd':
            output_path = input_path.with_suffix(input_path.suffix + '.zst')
        elif compression == 'lz4':
            output_path = input_path.with_suffix(input_path.suffix + '.lz4')
        else:
            raise ValueError("output_path required when compression is None")
    else:
        output_path = Path(output_path)

    # Read input file
    with open(input_path, 'rb') as f:
        data = f.read()

    # Compress
    compressed = compress_data(data, compression=compression, level=level)

    # Write output
    with open(output_path, 'wb') as f:
        f.write(compressed)

    return output_path


def decompress_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Decompress a compressed checkpoint file.

    Args:
        input_path: Path to compressed checkpoint
        output_path: Optional output path (defaults to input without compression extension)

    Returns:
        Path to decompressed file
    """
    input_path = Path(input_path)

    # Detect compression from extension
    compression = detect_compression(input_path)

    if compression is None:
        raise ValueError(f"Cannot detect compression type from {input_path}")

    if output_path is None:
        # Remove compression extension
        if compression == 'zstd' and input_path.suffix == '.zst':
            output_path = input_path.with_suffix('')
        elif compression == 'lz4' and input_path.suffix == '.lz4':
            output_path = input_path.with_suffix('')
        else:
            raise ValueError("output_path required")
    else:
        output_path = Path(output_path)

    # Read compressed file
    with open(input_path, 'rb') as f:
        compressed_data = f.read()

    # Decompress
    data = decompress_data(compressed_data, compression=compression)

    # Write output
    with open(output_path, 'wb') as f:
        f.write(data)

    return output_path


class CompressedFile:
    """Wrapper that provides transparent compression/decompression.

    Usage:
        with CompressedFile('checkpoint.thalia.zst', 'wb', compression='zstd') as f:
            f.write(data)  # Automatically compressed
    """

    def __init__(
        self,
        path: Union[str, Path],
        mode: str,
        compression: Optional[CompressionType] = None,
        level: int = 3,
    ):
        self.path = Path(path)
        self.mode = mode
        self.level = level

        # Auto-detect compression if not specified
        if compression is None:
            compression = detect_compression(self.path)
        self.compression = compression

        self.buffer = io.BytesIO()
        self.real_file = None

    def __enter__(self) -> BinaryIO:
        """Open file for reading/writing."""
        if 'r' in self.mode:
            # Read mode: decompress entire file into buffer
            with open(self.path, 'rb') as f:
                compressed = f.read()
            decompressed = decompress_data(compressed, self.compression)
            self.buffer = io.BytesIO(decompressed)
            return self.buffer
        else:
            # Write mode: use buffer, compress on close
            return self.buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file and compress if writing."""
        if 'w' in self.mode and exc_type is None:
            # Compress buffer and write to file
            data = self.buffer.getvalue()
            compressed = compress_data(data, self.compression, self.level)

            with open(self.path, 'wb') as f:
                f.write(compressed)

        self.buffer.close()
