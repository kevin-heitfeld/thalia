"""
Binary Format Implementation - Low-level binary encoding/decoding.

Implements the Thalia checkpoint binary format specification with:
- 256-byte fixed header
- Variable-length JSON metadata
- Region index for efficient lookup
- SHA-256 checksums

File Format:
    [HEADER]          256 bytes fixed
    [METADATA]        Variable length JSON
    [REGION_INDEX]    Variable length
    [REGION_DATA...]  Multiple regions
    [CHECKSUM]        32 bytes (SHA-256)
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Tuple

# Format version
MAJOR_VERSION = 0
MINOR_VERSION = 1
PATCH_VERSION = 0

# Magic number for Thalia files
MAGIC_NUMBER = b"THAL"

# Header size (fixed)
HEADER_SIZE = 256


@dataclass
class CheckpointHeader:
    """Binary checkpoint header (256 bytes fixed)."""

    magic: bytes  # 4 bytes - "THAL"
    major_version: int  # 2 bytes
    minor_version: int  # 2 bytes
    patch_version: int  # 2 bytes
    flags: int  # 2 bytes (feature flags)
    timestamp: int  # 8 bytes (Unix timestamp)
    metadata_offset: int  # 8 bytes
    metadata_length: int  # 8 bytes
    region_index_offset: int  # 8 bytes
    region_index_length: int  # 8 bytes
    connectivity_offset: int  # 8 bytes (reserved for future)
    connectivity_length: int  # 8 bytes (reserved for future)
    total_neurons: int  # 8 bytes
    total_synapses: int  # 8 bytes
    training_steps: int  # 8 bytes
    num_regions: int  # 4 bytes
    checksum_type: int  # 4 bytes (1=SHA256)
    # reserved: 156 bytes

    def to_bytes(self) -> bytes:
        """Serialize header to 256 bytes."""
        data = struct.pack(
            "<4sHHHHQQQQQQQQQQII",
            self.magic,
            self.major_version,
            self.minor_version,
            self.patch_version,
            self.flags,
            self.timestamp,
            self.metadata_offset,
            self.metadata_length,
            self.region_index_offset,
            self.region_index_length,
            self.connectivity_offset,
            self.connectivity_length,
            self.total_neurons,
            self.total_synapses,
            self.training_steps,
            self.num_regions,
            self.checksum_type,
        )
        # Pad to 256 bytes with zeros
        return data + b"\x00" * (HEADER_SIZE - len(data))

    @classmethod
    def from_bytes(cls, data: bytes) -> CheckpointHeader:
        """Deserialize header from 256 bytes."""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Header data too short: {len(data)} < {HEADER_SIZE}")

        fields = struct.unpack("<4sHHHHQQQQQQQQQQII", data[:100])

        return cls(
            magic=fields[0],
            major_version=fields[1],
            minor_version=fields[2],
            patch_version=fields[3],
            flags=fields[4],
            timestamp=fields[5],
            metadata_offset=fields[6],
            metadata_length=fields[7],
            region_index_offset=fields[8],
            region_index_length=fields[9],
            connectivity_offset=fields[10],
            connectivity_length=fields[11],
            total_neurons=fields[12],
            total_synapses=fields[13],
            training_steps=fields[14],
            num_regions=fields[15],
            checksum_type=fields[16],
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate header fields.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        if self.magic != MAGIC_NUMBER:
            issues.append(f"Invalid magic number: {self.magic!r} != {MAGIC_NUMBER!r}")

        if self.major_version > MAJOR_VERSION:
            issues.append(f"Unsupported major version: {self.major_version} > {MAJOR_VERSION}")

        if self.checksum_type not in (0, 1):  # 0=none, 1=SHA256
            issues.append(f"Invalid checksum type: {self.checksum_type}")

        if self.num_regions == 0:
            issues.append("No regions in checkpoint")

        return (len(issues) == 0, issues)


@dataclass
class RegionIndexEntry:
    """Entry in the region index (48 bytes per region)."""

    region_name: str  # 32 bytes (null-terminated)
    data_offset: int  # 8 bytes
    data_length: int  # 8 bytes

    def to_bytes(self) -> bytes:
        """Serialize entry to 48 bytes."""
        # Encode name as null-terminated string, truncate/pad to 32 bytes
        name_bytes = self.region_name.encode("utf-8")[:31]
        name_bytes = name_bytes + b"\x00" * (32 - len(name_bytes))

        return name_bytes + struct.pack("<QQ", self.data_offset, self.data_length)

    @classmethod
    def from_bytes(cls, data: bytes) -> RegionIndexEntry:
        """Deserialize entry from 48 bytes."""
        if len(data) < 48:
            raise ValueError(f"Region index entry too short: {len(data)} < 48")

        # Extract null-terminated string
        name_bytes = data[:32]
        null_pos = name_bytes.find(b"\x00")
        if null_pos >= 0:
            name_bytes = name_bytes[:null_pos]
        region_name = name_bytes.decode("utf-8")

        data_offset, data_length = struct.unpack("<QQ", data[32:48])

        return cls(
            region_name=region_name,
            data_offset=data_offset,
            data_length=data_length,
        )


class BinaryWriter:
    """Low-level binary writing with checksums."""

    def __init__(self, file: BinaryIO):
        self.file = file
        self.hasher = hashlib.sha256()
        self._write_count = 0

    def write_header(self, header: CheckpointHeader) -> int:
        """Write header and return bytes written."""
        data = header.to_bytes()
        self.file.write(data)
        self.hasher.update(data)
        self._write_count += len(data)
        return len(data)

    def write_json(self, data: Dict[str, Any]) -> int:
        """Write JSON data and return bytes written."""
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.file.write(json_bytes)
        self.hasher.update(json_bytes)
        self._write_count += len(json_bytes)
        return len(json_bytes)

    def write_bytes(self, data: bytes) -> int:
        """Write raw bytes and return bytes written."""
        self.file.write(data)
        self.hasher.update(data)
        self._write_count += len(data)
        return len(data)

    def write_region_index(self, entries: List[RegionIndexEntry]) -> int:
        """Write region index and return bytes written."""
        total_bytes = 0
        for entry in entries:
            entry_bytes = entry.to_bytes()
            self.file.write(entry_bytes)
            self.hasher.update(entry_bytes)
            total_bytes += len(entry_bytes)
            self._write_count += len(entry_bytes)
        return total_bytes

    def finalize(self) -> bytes:
        """Write final checksum and return it."""
        checksum = self.hasher.digest()
        self.file.write(checksum)
        # Don't update hasher with checksum itself
        return checksum

    def tell(self) -> int:
        """Get current file position."""
        return self.file.tell()


class BinaryReader:
    """Low-level binary reading with validation."""

    def __init__(self, file: BinaryIO):
        self.file = file
        self.hasher = hashlib.sha256()
        self._read_count = 0

    def read_header(self) -> CheckpointHeader:
        """Read and validate header."""
        data = self.file.read(HEADER_SIZE)
        if len(data) < HEADER_SIZE:
            raise ValueError(f"File too short for header: {len(data)} < {HEADER_SIZE}")

        self.hasher.update(data)
        self._read_count += len(data)

        header = CheckpointHeader.from_bytes(data)
        is_valid, issues = header.validate()

        if not is_valid:
            raise ValueError(f"Invalid header: {issues}")

        return header

    def read_json(self, offset: int, length: int) -> Dict[str, Any]:
        """Read JSON data at specific offset."""
        self.file.seek(offset)
        data = self.file.read(length)

        if len(data) < length:
            raise ValueError(f"Unexpected EOF reading JSON: {len(data)} < {length}")

        self.hasher.update(data)
        self._read_count += length

        return dict(json.loads(data.decode("utf-8")))

    def read_bytes(self, offset: int, length: int) -> bytes:
        """Read raw bytes at specific offset."""
        self.file.seek(offset)
        data = self.file.read(length)

        if len(data) < length:
            raise ValueError(f"Unexpected EOF: {len(data)} < {length}")

        self.hasher.update(data)
        self._read_count += length

        return data

    def read_region_index(self, offset: int, length: int) -> List[RegionIndexEntry]:
        """Read region index at specific offset."""
        self.file.seek(offset)
        data = self.file.read(length)

        if len(data) < length:
            raise ValueError(f"Unexpected EOF reading region index: {len(data)} < {length}")

        self.hasher.update(data)
        self._read_count += length

        # Parse entries (48 bytes each)
        entries = []
        num_entries = length // 48

        for i in range(num_entries):
            entry_data = data[i * 48 : (i + 1) * 48]
            entries.append(RegionIndexEntry.from_bytes(entry_data))

        return entries

    def validate_checksum(self, expected_checksum: bytes) -> bool:
        """Validate computed checksum against expected."""
        computed = self.hasher.digest()
        return computed == expected_checksum

    def read_checksum(self) -> bytes:
        """Read checksum from end of file (last 32 bytes)."""
        # Seek to 32 bytes before EOF
        self.file.seek(-32, 2)
        return self.file.read(32)
