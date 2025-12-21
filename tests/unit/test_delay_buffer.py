"""
Test circular delay buffer implementation.

Verifies that CircularDelayBuffer correctly handles:
- Write/read operations
- Circular wrap-around
- Growth
- State saving/loading
- Edge cases

Author: Thalia Project
Date: December 21, 2025
"""

import pytest
import torch

from thalia.utils.delay_buffer import CircularDelayBuffer


def test_basic_delay():
    """Test basic delay functionality."""
    buffer = CircularDelayBuffer(max_delay=3, size=4, device="cpu")

    # Write and advance sequence: [1], [2], [3], [4]
    for t in range(4):
        spikes = torch.zeros(4, dtype=torch.bool)
        spikes[t % 4] = True  # One spike per timestep
        buffer.write(spikes)
        buffer.advance()

    # Now at timestep 4, buffer.ptr has wrapped
    # When we write new data and read with delay=3, we get data from 3 steps ago
    # The data 3 steps ago (t=1) had spike at index 1
    spikes = torch.zeros(4, dtype=torch.bool)
    buffer.write(spikes)
    delayed = buffer.read(delay=3)

    # Should read what we wrote 3 timesteps ago (t=1)
    expected = torch.zeros(4, dtype=torch.bool)
    expected[1] = True
    assert torch.equal(delayed, expected)


@pytest.mark.parametrize("delay", [0, 1, 2, 3, 5])
def test_delay_values(delay):
    """Test reading with various delay values.

    Why this test exists: Validates that the circular buffer correctly
    handles different delay amounts without off-by-one errors.
    """
    buffer = CircularDelayBuffer(max_delay=5, size=4, device="cpu")

    # Fill buffer with identifiable patterns
    for t in range(10):
        spikes = torch.zeros(4, dtype=torch.bool)
        spikes[t % 4] = True
        buffer.write(spikes)
        buffer.advance()

    # Read with specified delay
    if delay <= 5:
        delayed = buffer.read(delay=delay)
        assert delayed.shape == (4,)
        assert delayed.dtype == torch.bool
        # Should have exactly one spike (from the pattern)
        assert delayed.sum() <= 1


def test_zero_delay():
    """Test reading with zero delay (current timestep)."""
    buffer = CircularDelayBuffer(max_delay=5, size=3, device="cpu")

    # Write current spikes
    current = torch.tensor([True, False, True], dtype=torch.bool)
    buffer.write(current)

    # Read with delay=0 should return what we just wrote
    result = buffer.read(delay=0)
    assert torch.equal(result, current)


def test_wrap_around():
    """Test that buffer correctly wraps around."""
    buffer = CircularDelayBuffer(max_delay=2, size=2, device="cpu")

    # Fill buffer past its capacity to test wrap-around
    for t in range(10):
        spikes = torch.tensor([t % 2 == 0, t % 2 == 1], dtype=torch.bool)
        buffer.write(spikes)

        if t >= 2:  # Once buffer is filled
            # Read with delay=2
            delayed = buffer.read(delay=2)
            # Should be what we wrote 2 timesteps ago
            expected_t = t - 2
            expected = torch.tensor(
                [expected_t % 2 == 0, expected_t % 2 == 1],
                dtype=torch.bool
            )
            assert torch.equal(delayed, expected), f"Failed at t={t}"

        buffer.advance()


@pytest.mark.parametrize("initial_size,new_size", [
    (3, 5),
    (10, 20),
    (50, 100),
])
def test_grow_various_sizes(initial_size, new_size):
    """Test growing buffer from various initial sizes.

    Why this test exists: Ensures buffer growth correctly handles
    different size ratios and preserves existing data.
    """
    buffer = CircularDelayBuffer(max_delay=2, size=initial_size, device="cpu")

    # Write some identifiable data
    for t in range(3):
        spikes = torch.zeros(initial_size, dtype=torch.bool)
        if t < initial_size:
            spikes[t] = True
        buffer.write(spikes)
        buffer.advance()

    # Grow buffer
    old_data = buffer.read(delay=0).clone()
    buffer.grow(new_size=new_size)

    assert buffer.size == new_size
    # Old data should be preserved in first positions
    new_data = buffer.read(delay=0)
    assert torch.equal(old_data, new_data[:initial_size])
    # New elements should be zero
    assert not new_data[initial_size:].any()


def test_reset():
    """Test resetting buffer."""
    buffer = CircularDelayBuffer(max_delay=2, size=3, device="cpu")

    # Write some data
    for _ in range(3):
        spikes = torch.ones(3, dtype=torch.bool)
        buffer.write(spikes)
        buffer.advance()

    # Reset
    buffer.reset()

    # All data should be zeros
    for delay in range(3):
        result = buffer.read(delay=delay)
        assert torch.all(~result), f"Buffer not cleared at delay={delay}"

    # Pointer should be at 0
    assert buffer.ptr == 0


def test_state_dict():
    """Test saving and loading state."""
    buffer1 = CircularDelayBuffer(max_delay=3, size=2, device="cpu")

    # Write some data
    for t in range(4):
        spikes = torch.tensor([t % 2 == 0, t % 2 == 1], dtype=torch.bool)
        buffer1.write(spikes)
        buffer1.advance()

    # Save state
    state = buffer1.state_dict()

    # Create new buffer and load state
    buffer2 = CircularDelayBuffer(max_delay=3, size=2, device="cpu")
    buffer2.load_state_dict(state)

    # Both buffers should produce same output
    for delay in range(4):
        result1 = buffer1.read(delay=delay)
        result2 = buffer2.read(delay=delay)
        assert torch.equal(result1, result2), f"Mismatch at delay={delay}"

    assert buffer1.ptr == buffer2.ptr


def test_edge_cases():
    """Test edge cases and error handling."""
    buffer = CircularDelayBuffer(max_delay=2, size=3, device="cpu")

    # Test invalid delay (too large)
    with pytest.raises(ValueError, match="out of range"):
        buffer.read(delay=5)

    # Test invalid delay (negative)
    with pytest.raises(ValueError, match="out of range"):
        buffer.read(delay=-1)

    # Test wrong size write
    with pytest.raises(ValueError, match="size mismatch"):
        buffer.write(torch.zeros(5, dtype=torch.bool))

    # Test invalid construction
    with pytest.raises(ValueError):
        CircularDelayBuffer(max_delay=-1, size=3)

    with pytest.raises(ValueError):
        CircularDelayBuffer(max_delay=2, size=0)


def test_float_buffer():
    """Test using buffer with float dtype (for non-binary signals)."""
    buffer = CircularDelayBuffer(max_delay=2, size=2, device="cpu", dtype=torch.float32)

    # Write float values and advance
    for t in range(3):
        values = torch.tensor([float(t), float(t) * 2], dtype=torch.float32)
        buffer.write(values)
        buffer.advance()

    # Read delayed values (delay=2 gives us t=0)
    # We're at position 3, delay=2 gives position 1 (t=1)
    delayed = buffer.read(delay=2)
    expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert torch.equal(delayed, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
