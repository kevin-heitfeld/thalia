"""Tests for windowed (streaming) recording components.

Validates the circular buffer linearisation (``linearise_circular``),
spike-flat trimming (``BufferManager.trim_spike_flat_buffers``), and
per-neuron count rebuilding (``BufferManager.rebuild_per_neuron_spike_counts_from_window``).
"""

from __future__ import annotations

import numpy as np

from thalia.diagnostics.buffer_manager import BufferManager
from thalia.diagnostics.circular_buffer import linearise_circular


# ---------------------------------------------------------------------------
# Helpers: create a minimal BufferManager with only the attributes
# needed by the helper methods, avoiding the need for a full Brain.
# ---------------------------------------------------------------------------

def _make_buffer_stub(
    *,
    n_pops: int = 2,
    pop_sizes: list[int] | None = None,
    window_size: int = 10,
    total_n_recorded: int = 0,
) -> BufferManager:
    """Build a BufferManager stub with just enough state for helper tests.

    We bypass __init__ entirely and set the attributes that the helpers read.
    """
    if pop_sizes is None:
        pop_sizes = [8, 4]
    obj = object.__new__(BufferManager)
    obj._window_size = window_size
    obj._windowed = True
    obj.total_n_recorded = total_n_recorded
    # spike flat buffers (empty by default)
    obj.spike_flat_nidx = [np.empty(64, dtype=np.int32) for _ in range(n_pops)]
    obj.spike_flat_ts = [np.empty(64, dtype=np.int32) for _ in range(n_pops)]
    obj.spike_flat_n = np.zeros(n_pops, dtype=np.int64)
    return obj


# =====================================================================
# linearise_circular (standalone function)
# =====================================================================

class TestLineariseCircular:
    """Tests for linearise_circular."""

    def test_not_filled_returns_prefix(self) -> None:
        buf = np.arange(10).reshape(10, 1)
        result = linearise_circular(buf, cursor=3, filled=False)
        np.testing.assert_array_equal(result, buf[:3])

    def test_filled_rolls_correctly(self) -> None:
        buf = np.array([5, 6, 7, 0, 1, 2, 3, 4]).reshape(8, 1)
        # cursor=3 means next write would overwrite index 3; oldest is at 3
        result = linearise_circular(buf, cursor=3, filled=True)
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(8, 1)
        np.testing.assert_array_equal(result, expected)

    def test_filled_cursor_zero(self) -> None:
        """When cursor=0 and filled, the buffer is already in order."""
        buf = np.array([0, 1, 2, 3]).reshape(4, 1)
        result = linearise_circular(buf, cursor=0, filled=True)
        np.testing.assert_array_equal(result, buf)

    def test_multidimensional(self) -> None:
        # shape (4, 2): 4 timesteps, 2 features
        buf = np.array([[20, 21], [30, 31], [0, 1], [10, 11]])
        result = linearise_circular(buf, cursor=2, filled=True)
        expected = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])
        np.testing.assert_array_equal(result, expected)

    def test_returns_copy(self) -> None:
        buf = np.arange(6).reshape(6, 1)
        result = linearise_circular(buf, cursor=2, filled=False)
        assert not np.shares_memory(result, buf)


# =====================================================================
# BufferManager.trim_spike_flat_buffers
# =====================================================================

class TestTrimSpikeFlats:
    """Tests for BufferManager.trim_spike_flat_buffers."""

    def test_no_trim_when_window_not_exceeded(self) -> None:
        stub = _make_buffer_stub(window_size=10, total_n_recorded=5)
        # 3 events in pop 0 at timesteps [1, 2, 3]
        stub.spike_flat_ts[0][:3] = [1, 2, 3]
        stub.spike_flat_nidx[0][:3] = [0, 1, 2]
        stub.spike_flat_n[0] = 3
        stub.trim_spike_flat_buffers()
        assert stub.spike_flat_n[0] == 3  # nothing trimmed

    def test_trims_old_events(self) -> None:
        stub = _make_buffer_stub(window_size=5, total_n_recorded=10)
        # cutoff = 10 - 5 = 5; events with ts < 5 should be removed
        ts = np.array([2, 3, 5, 7, 9], dtype=np.int32)
        nidx = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        stub.spike_flat_ts[0][:5] = ts
        stub.spike_flat_nidx[0][:5] = nidx
        stub.spike_flat_n[0] = 5
        stub.trim_spike_flat_buffers()
        cnt = int(stub.spike_flat_n[0])
        assert cnt == 3
        np.testing.assert_array_equal(stub.spike_flat_ts[0][:cnt], [5, 7, 9])
        np.testing.assert_array_equal(stub.spike_flat_nidx[0][:cnt], [2, 3, 4])

    def test_trims_all_events(self) -> None:
        stub = _make_buffer_stub(window_size=5, total_n_recorded=100)
        # All events are old
        stub.spike_flat_ts[0][:3] = [10, 20, 30]
        stub.spike_flat_nidx[0][:3] = [0, 1, 2]
        stub.spike_flat_n[0] = 3
        stub.trim_spike_flat_buffers()
        assert stub.spike_flat_n[0] == 0

    def test_empty_population_unaffected(self) -> None:
        stub = _make_buffer_stub(window_size=5, total_n_recorded=10)
        stub.spike_flat_n[0] = 0
        stub.spike_flat_n[1] = 0
        stub.trim_spike_flat_buffers()
        assert stub.spike_flat_n[0] == 0
        assert stub.spike_flat_n[1] == 0


# =====================================================================
# BufferManager.rebuild_per_neuron_spike_counts_from_window
# =====================================================================

class TestRebuildPerNeuronCounts:
    """Tests for BufferManager.rebuild_per_neuron_spike_counts_from_window."""

    def test_counts_within_window(self) -> None:
        stub = _make_buffer_stub(n_pops=1, pop_sizes=[4], window_size=5, total_n_recorded=10)
        # window: steps [5, 6, 7, 8, 9]; cutoff = 5
        # neuron 0 fires at steps 3, 6, 8; only 6, 8 in window
        # neuron 2 fires at step 7
        ts = np.array([3, 6, 7, 8], dtype=np.int32)
        nidx = np.array([0, 0, 2, 0], dtype=np.int32)
        stub.spike_flat_ts[0][:4] = ts
        stub.spike_flat_nidx[0][:4] = nidx
        stub.spike_flat_n[0] = 4

        pop_sizes = np.array([4], dtype=np.int32)
        counts = stub.rebuild_per_neuron_spike_counts_from_window(pop_sizes)
        assert len(counts) == 1
        np.testing.assert_array_equal(counts[0], [2, 0, 1, 0])

    def test_no_spikes(self) -> None:
        stub = _make_buffer_stub(n_pops=1, pop_sizes=[3], window_size=5, total_n_recorded=10)
        stub.spike_flat_n[0] = 0
        pop_sizes = np.array([3], dtype=np.int32)
        counts = stub.rebuild_per_neuron_spike_counts_from_window(pop_sizes)
        np.testing.assert_array_equal(counts[0], [0, 0, 0])

    def test_multiple_populations(self) -> None:
        stub = _make_buffer_stub(n_pops=2, pop_sizes=[3, 2], window_size=10, total_n_recorded=10)
        # Pop 0: neuron 1 fires at step 5
        stub.spike_flat_ts[0][:1] = [5]
        stub.spike_flat_nidx[0][:1] = [1]
        stub.spike_flat_n[0] = 1
        # Pop 1: neuron 0 fires at steps 2, 8
        stub.spike_flat_ts[1][:2] = [2, 8]
        stub.spike_flat_nidx[1][:2] = [0, 0]
        stub.spike_flat_n[1] = 2

        pop_sizes = np.array([3, 2], dtype=np.int32)
        counts = stub.rebuild_per_neuron_spike_counts_from_window(pop_sizes)
        np.testing.assert_array_equal(counts[0], [0, 1, 0])
        np.testing.assert_array_equal(counts[1], [2, 0])
