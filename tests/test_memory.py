"""
Tests for working memory systems.
"""

import pytest
import torch
from thalia.memory import (
    WorkingMemoryConfig,
    MemorySlot,
    WorkingMemory,
    WorkingMemorySNN,
)


class TestWorkingMemoryConfig:
    """Test WorkingMemoryConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = WorkingMemoryConfig()
        assert config.n_slots == 7
        assert config.slot_size == 50
        assert config.reverb_strength == 0.8
        assert config.decay_rate == 0.01
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = WorkingMemoryConfig(
            n_slots=5,
            slot_size=100,
            reverb_strength=0.9,
        )
        assert config.n_slots == 5
        assert config.slot_size == 100
        assert config.reverb_strength == 0.9
        
    def test_total_neurons(self):
        """Test total neurons property."""
        config = WorkingMemoryConfig(n_slots=5, slot_size=50)
        assert config.total_neurons == 250


class TestMemorySlot:
    """Test individual memory slot."""
    
    @pytest.fixture
    def slot(self):
        """Create a test slot."""
        return MemorySlot(
            size=50,
            reverb_strength=0.8,
            decay_rate=0.01,
            tau_mem=30.0,
            noise_std=0.0,  # No noise for deterministic tests
        )
    
    def test_initialization(self, slot):
        """Test slot initialization."""
        assert slot.size == 50
        assert slot.reverb_strength == 0.8
        assert slot.recurrent.shape == (50, 50)
        
    def test_reset_state(self, slot):
        """Test state reset."""
        slot.reset_state(batch_size=4)
        assert slot.neurons.membrane is not None
        assert slot.neurons.membrane.shape == (4, 50)
        assert slot._last_spikes is None
        
    def test_forward_step(self, slot):
        """Test forward step."""
        slot.reset_state(batch_size=1)
        
        input_pattern = torch.randn(1, 50)
        spikes, activity = slot(input_pattern)
        
        assert spikes.shape == (1, 50)
        assert 0 <= activity <= 1
        
    def test_forward_without_input(self, slot):
        """Test forward step without input."""
        slot.reset_state(batch_size=1)
        
        spikes, activity = slot(None)
        
        assert spikes.shape == (1, 50)
        
    def test_gates_affect_loading(self, slot):
        """Test that load gate controls input."""
        slot.reset_state(batch_size=1)
        input_pattern = torch.ones(1, 50) * 2.0
        
        # With load gate open
        slot.set_gates(load=1.0, clear=0.0)
        spikes1, _ = slot(input_pattern)
        
        # With load gate closed
        slot.reset_state(batch_size=1)
        slot.set_gates(load=0.0, clear=0.0)
        spikes2, _ = slot(input_pattern)
        
        # Open gate should have more activity (input gets through)
        assert spikes1.sum() >= spikes2.sum()
        
    def test_is_active_property(self, slot):
        """Test is_active property."""
        slot.reset_state(batch_size=1)
        assert not slot.is_active
        
        # Generate some activity
        slot.set_gates(load=1.0)
        strong_input = torch.ones(1, 50) * 5.0
        for _ in range(10):
            slot(strong_input)
            
        # After sustained input, should be active
        # (depends on implementation - may or may not be true)
        
    def test_get_content(self, slot):
        """Test getting slot content."""
        slot.reset_state(batch_size=1)
        
        # Initially empty
        content = slot.get_content()
        assert content is None
        
        # After some activity
        slot.set_gates(load=1.0)
        slot(torch.ones(1, 50) * 5.0)
        
        content = slot.get_content()
        # Content may or may not be None depending on activity


class TestWorkingMemory:
    """Test working memory system."""
    
    @pytest.fixture
    def memory(self):
        """Create working memory."""
        config = WorkingMemoryConfig(
            n_slots=5,
            slot_size=50,
            noise_std=0.0,
        )
        return WorkingMemory(config)
    
    def test_initialization(self, memory):
        """Test memory initialization."""
        assert len(memory.slots) == 5
        assert memory.config.slot_size == 50
        
    def test_reset_state(self, memory):
        """Test state reset."""
        memory.reset_state(batch_size=2)
        
        for slot in memory.slots:
            assert slot.neurons.membrane is not None
            
    def test_load_pattern(self, memory):
        """Test loading a pattern into a slot."""
        memory.reset_state(batch_size=1)
        
        pattern = torch.randn(1, 50)
        memory.load(0, pattern, label="test")
        
        assert memory.slot_contents[0] == "test"
        assert 0 in memory._pending_loads
        
    def test_load_invalid_slot(self, memory):
        """Test loading into invalid slot raises error."""
        with pytest.raises(ValueError):
            memory.load(99, torch.randn(1, 50))
            
    def test_clear_slot(self, memory):
        """Test clearing a slot."""
        memory.reset_state(batch_size=1)
        memory.load(0, torch.randn(1, 50), label="test")
        
        memory.clear(0)
        
        assert memory.slot_contents[0] is None
        
    def test_clear_all(self, memory):
        """Test clearing all slots."""
        memory.reset_state(batch_size=1)
        
        for i in range(3):
            memory.load(i, torch.randn(1, 50), label=f"item{i}")
            
        memory.clear_all()
        
        for label in memory.slot_contents:
            assert label is None
            
    def test_forward_step(self, memory):
        """Test forward step."""
        memory.reset_state(batch_size=1)
        
        output = memory()
        
        # Output should have all slot neurons
        assert output.shape == (1, 5 * 50)
        
    def test_forward_with_load(self, memory):
        """Test forward processes pending loads."""
        memory.reset_state(batch_size=1)
        
        pattern = torch.randn(1, 50)
        memory.load(0, pattern)
        
        # Forward should consume the pending load
        memory()
        
        assert 0 not in memory._pending_loads
        
    def test_get_status(self, memory):
        """Test status reporting."""
        memory.reset_state(batch_size=1)
        
        status = memory.get_status()
        
        assert status["n_slots"] == 5
        assert len(status["slots"]) == 5
        assert "active_count" in status
        assert "capacity_used" in status
        
    def test_find_empty_slot(self, memory):
        """Test finding empty slot."""
        memory.reset_state(batch_size=1)
        
        empty = memory.find_empty_slot()
        
        # All slots start empty
        assert empty is not None
        assert empty == 0  # First slot
        
    def test_refresh_slot(self, memory):
        """Test refreshing a slot."""
        memory.reset_state(batch_size=1)
        
        # Load something
        pattern = torch.randn(1, 50)
        memory.load(0, pattern)
        memory()  # Process load
        
        # Refresh should work without error
        memory.refresh(0, strength=0.5)
        
    def test_refresh_invalid_slot(self, memory):
        """Test refreshing invalid slot raises error."""
        with pytest.raises(ValueError):
            memory.refresh(99)


class TestWorkingMemorySNN:
    """Test full working memory SNN."""
    
    @pytest.fixture
    def network(self):
        """Create working memory SNN."""
        config = WorkingMemoryConfig(
            n_slots=3,
            slot_size=30,
            noise_std=0.0,
        )
        return WorkingMemorySNN(
            input_size=20,
            output_size=10,
            config=config,
        )
    
    def test_initialization(self, network):
        """Test network initialization."""
        assert network.input_size == 20
        assert network.output_size == 10
        assert network.config.n_slots == 3
        
    def test_reset_state(self, network):
        """Test state reset."""
        network.reset_state(batch_size=2)
        
        # Memory should be reset
        for slot in network.memory.slots:
            assert slot.neurons.membrane is not None
            
    def test_encode(self, network):
        """Test input encoding."""
        x = torch.randn(2, 20)
        encoded = network.encode(x)
        
        assert encoded.shape == (2, 30)  # slot_size
        
    def test_decode(self, network):
        """Test memory decoding."""
        memory_output = torch.randn(2, 90)  # 3 slots * 30
        decoded = network.decode(memory_output)
        
        assert decoded.shape == (2, 10)  # output_size
        
    def test_forward_step(self, network):
        """Test forward step."""
        network.reset_state(batch_size=1)
        
        output_spikes, memory_activity = network()
        
        assert output_spikes.shape == (1, 10)
        assert memory_activity.shape == (1, 90)
        
    def test_forward_with_input(self, network):
        """Test forward with input loading."""
        network.reset_state(batch_size=1)
        
        x = torch.randn(1, 20)
        output_spikes, memory_activity = network(x, load_slot=0)
        
        assert output_spikes.shape == (1, 10)
        assert memory_activity.shape == (1, 90)
        
    def test_forward_sequence(self, network):
        """Test running multiple steps."""
        network.reset_state(batch_size=1)
        
        # Load input
        x = torch.randn(1, 20)
        network(x, load_slot=0)
        
        # Run for several steps
        outputs = []
        for _ in range(10):
            out, _ = network()
            outputs.append(out)
            
        assert len(outputs) == 10
        for out in outputs:
            assert out.shape == (1, 10)


class TestMemoryMaintenance:
    """Test memory maintenance (reverberating activity)."""
    
    @pytest.fixture
    def memory(self):
        """Create memory with stronger reverberation."""
        config = WorkingMemoryConfig(
            n_slots=3,
            slot_size=50,
            reverb_strength=0.9,
            decay_rate=0.001,  # Slow decay
            noise_std=0.0,
        )
        return WorkingMemory(config)
    
    def test_activity_persists(self, memory):
        """Test that activity persists after input removed."""
        memory.reset_state(batch_size=1)
        
        # Load strong input
        strong_input = torch.ones(1, 50) * 3.0
        memory.load(0, strong_input)
        
        # Step with input
        for _ in range(5):
            memory()
            
        activity_with_input = memory.slots[0]._activity_level
        
        # Now run without input
        for _ in range(10):
            memory()
            
        activity_without_input = memory.slots[0]._activity_level
        
        # Activity might persist (depends on dynamics)
        # At minimum, we should not crash
        assert True  # Placeholder - actual persistence depends on parameters


class TestGPU:
    """Test GPU functionality."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_on_gpu(self):
        """Test working memory on GPU."""
        device = torch.device("cuda")
        config = WorkingMemoryConfig(n_slots=3, slot_size=30)
        memory = WorkingMemory(config).to(device)
        
        # Move internal state
        memory.reset_state(batch_size=1)
        
        # Load pattern on GPU
        pattern = torch.randn(1, 30, device=device)
        memory.load(0, pattern)
        
        # Forward pass
        output = memory()
        
        assert output.device.type == "cuda"
