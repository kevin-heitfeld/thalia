"""Circular buffer utilities for time-indexed numpy arrays."""

from __future__ import annotations

import numpy as np


def linearise_circular(buf: np.ndarray, cursor: int, filled: bool) -> np.ndarray:
    """Linearise a circular buffer so oldest data appears at index 0.

    Parameters
    ----------
    buf:
        The circular buffer array (time axis is axis 0).
    cursor:
        Current write position (next slot to be overwritten).
    filled:
        Whether the buffer has wrapped around at least once.

    Returns a copy; the original is never mutated.
    """
    if not filled:
        return buf[:cursor].copy()
    return np.roll(buf, -cursor, axis=0).copy()


class CircularBuffer:
    """Circular buffer backed by a pre-allocated numpy array.

    Writes wrap around to the beginning when capacity is reached.
    Call :meth:`linearise` to get a time-ordered copy with the oldest
    sample at index 0.

    In the hot path, callers write directly to :attr:`data` at the slot
    returned by :attr:`cursor`, then call :meth:`advance`.
    """

    __slots__ = ("data", "_cursor", "_count")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | type = np.float32,
        fill_value: float = 0.0,
    ) -> None:
        self.data: np.ndarray = np.full(shape, fill_value, dtype=dtype)
        self._cursor: int = 0
        self._count: int = 0

    @property
    def capacity(self) -> int:
        """Number of time-slots along axis 0."""
        return self.data.shape[0]

    @property
    def cursor(self) -> int:
        """Current write position (next slot that will be overwritten)."""
        return self._cursor

    @property
    def filled(self) -> bool:
        """``True`` once the buffer has wrapped at least once."""
        return self._count >= self.capacity

    @property
    def count(self) -> int:
        """Total number of :meth:`advance` calls (may exceed capacity)."""
        return self._count

    def advance(self) -> None:
        """Move the write cursor forward by one slot."""
        self._cursor = (self._cursor + 1) % self.capacity
        self._count += 1

    def linearise(self) -> np.ndarray:
        """Return a time-ordered copy of the buffer contents."""
        return linearise_circular(self.data, self._cursor, self.filled)

    def reset(self, fill_value: float = 0.0) -> None:
        """Zero (or fill) the buffer and reset the cursor."""
        self.data.fill(fill_value)
        self._cursor = 0
        self._count = 0
