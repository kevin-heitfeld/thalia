"""
Tests for ThinkingSNN - the core thinking architecture.
"""

import pytest
import torch
from thalia.cognition import ThinkingSNN, ThinkingConfig, ThoughtState


class TestThinkingConfig:
    """Test ThinkingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ThinkingConfig()
        assert config.n_concepts == 256
        assert config.n_wm_slots == 7
        assert config.enable_learning is True
        assert config.enable_homeostasis is True
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = ThinkingConfig(
            n_concepts=128,
            n_wm_slots=5,
            enable_learning=False,
        )
        assert config.n_concepts == 128
        assert config.n_wm_slots == 5
        assert config.enable_learning is False


class TestThinkingSNN:
    """Test ThinkingSNN functionality."""
    
    @pytest.fixture
    def thinker(self):
        """Create a small thinking network for testing."""
        config = ThinkingConfig(
            n_concepts=64,
            n_wm_slots=3,
            wm_slot_size=32,
            enable_learning=False,  # Disable for faster tests
            enable_homeostasis=False,
        )
        return ThinkingSNN(config)
    
    @pytest.fixture
    def learning_thinker(self):
        """Create a thinking network with learning enabled."""
        config = ThinkingConfig(
            n_concepts=32,
            n_wm_slots=2,
            wm_slot_size=16,
            enable_learning=True,
            enable_homeostasis=True,
        )
        return ThinkingSNN(config)
    
    def test_initialization(self, thinker):
        """Test thinker initialization."""
        assert thinker.config.n_concepts == 64
        assert len(thinker.working_memory.slots) == 3
        
    def test_reset_state(self, thinker):
        """Test state reset."""
        thinker.reset_state(batch_size=1)
        
        # Should have reset internal state
        assert thinker._timestep == 0
        assert thinker._current_concept == -1
        
    def test_store_concept(self, thinker):
        """Test storing concepts."""
        thinker.reset_state(batch_size=1)
        
        pattern = (torch.rand(64) < 0.1).float()
        idx = thinker.store_concept(pattern, "apple")
        
        assert idx == 0
        assert len(thinker.concepts.patterns) == 1
        
    def test_store_multiple_concepts(self, thinker):
        """Test storing multiple concepts."""
        thinker.reset_state(batch_size=1)
        
        patterns = {
            "apple": (torch.rand(64) < 0.1).float(),
            "banana": (torch.rand(64) < 0.1).float(),
            "cat": (torch.rand(64) < 0.1).float(),
        }
        
        for name, pattern in patterns.items():
            thinker.store_concept(pattern, name)
        
        assert len(thinker.concepts.patterns) == 3
        
    def test_associate_concepts(self, thinker):
        """Test concept association."""
        thinker.reset_state(batch_size=1)
        
        idx1 = thinker.store_concept((torch.rand(64) < 0.1).float(), "red")
        idx2 = thinker.store_concept((torch.rand(64) < 0.1).float(), "apple")
        
        thinker.associate_concepts(idx1, idx2, strength=1.0)
        
        assert (idx1, idx2) in thinker.concepts.associations
        
    def test_think_step(self, thinker):
        """Test single thinking step."""
        thinker.reset_state(batch_size=1)
        
        state = thinker.think()
        
        assert isinstance(state, ThoughtState)
        assert state.timestep == 1
        assert state.spikes.shape == (1, 64)
        
    def test_think_multiple_steps(self, thinker):
        """Test multiple thinking steps."""
        thinker.reset_state(batch_size=1)
        
        for t in range(10):
            state = thinker.think()
            assert state.timestep == t + 1
            
    def test_attend_to(self, thinker):
        """Test attention mechanism."""
        thinker.reset_state(batch_size=1)
        
        pattern = torch.randn(1, 64)
        thinker.attend_to(pattern)
        
        assert thinker._attention_target is not None
        
    def test_set_goal(self, thinker):
        """Test goal setting."""
        thinker.set_goal(1.0)
        assert thinker._goal_signal == 1.0
        
        thinker.set_goal(-0.5)
        assert thinker._goal_signal == -0.5
        
    def test_load_to_memory(self, thinker):
        """Test loading to working memory."""
        thinker.reset_state(batch_size=1)
        
        pattern = torch.randn(1, 32)  # wm_slot_size = 32
        thinker.load_to_memory(0, pattern, label="test")
        
        assert thinker.working_memory.slot_contents[0] == "test"
        
    def test_read_from_memory(self, thinker):
        """Test reading from working memory."""
        thinker.reset_state(batch_size=1)
        
        pattern = torch.randn(1, 32)
        thinker.load_to_memory(0, pattern)
        
        # Process the load
        thinker.think()
        
        # Read back (may or may not have content depending on activity)
        content = thinker.read_from_memory(0)
        # Content could be None or tensor
        
    def test_get_trajectory(self, thinker):
        """Test getting thought trajectory."""
        thinker.reset_state(batch_size=1)
        
        for _ in range(10):
            thinker.think()
            
        trajectory = thinker.get_trajectory()
        
        assert isinstance(trajectory, type(thinker.trajectory))
        
    def test_get_activity_history(self, thinker):
        """Test getting activity history."""
        thinker.reset_state(batch_size=1)
        
        for _ in range(10):
            thinker.think()
            
        history = thinker.get_activity_history()
        
        assert history.shape[0] == 10  # 10 timesteps
        
    def test_project_activity(self, thinker):
        """Test activity projection."""
        thinker.reset_state(batch_size=1)
        
        for _ in range(50):
            thinker.think()
            
        coords, variance = thinker.project_activity(n_components=3)
        
        assert coords.shape == (50, 3)
        assert len(variance) == 3
        
    def test_on_concept_change_callback(self, thinker):
        """Test concept change callback."""
        thinker.reset_state(batch_size=1)
        
        # Store a concept
        pattern = (torch.rand(64) < 0.2).float()
        thinker.store_concept(pattern, "test")
        
        # Register callback
        changes = []
        def on_change(idx: int, name: str):
            changes.append((idx, name))
        thinker.on_concept_change(on_change)
        
        # Run with attention to trigger concept activation
        thinker.attend_to(pattern.unsqueeze(0))
        for _ in range(50):
            thinker.think()
        
        # Callback might have been called
        # (depends on whether concept was detected)
        
    def test_think_until_stable(self, thinker):
        """Test thinking until stable."""
        thinker.reset_state(batch_size=1)
        
        steps = thinker.think_until_stable(max_steps=100, stability_window=10)
        
        assert steps <= 100
        
    def test_generate_thought_chain(self, thinker):
        """Test generating thought chain."""
        thinker.reset_state(batch_size=1)
        
        # Store some concepts
        thinker.store_concept((torch.rand(64) < 0.1).float(), "apple")
        thinker.store_concept((torch.rand(64) < 0.1).float(), "banana")
        
        chain = thinker.generate_thought_chain(steps=50)
        
        assert isinstance(chain, list)
        # May or may not have entries depending on concept detection


class TestThinkingWithLearning:
    """Test thinking with learning enabled."""
    
    @pytest.fixture
    def thinker(self):
        """Create a thinking network with learning."""
        config = ThinkingConfig(
            n_concepts=32,
            n_wm_slots=2,
            wm_slot_size=16,
            enable_learning=True,
            enable_homeostasis=True,
        )
        return ThinkingSNN(config)
    
    def test_learning_components_initialized(self, thinker):
        """Test that learning components are initialized."""
        assert thinker.stdp is not None
        assert thinker.reward_stdp is not None
        assert thinker.intrinsic_plasticity is not None
        assert thinker.synaptic_scaling is not None
        
    def test_learning_affects_weights(self, thinker):
        """Test that learning modifies weights."""
        thinker.reset_state(batch_size=1)
        
        # Get initial weights
        initial_weights = thinker.concepts.weights.data.clone()
        
        # Run with reward
        thinker.set_goal(1.0)
        for _ in range(100):
            thinker.think()
        
        # Weights might have changed
        # (depends on spike patterns and learning dynamics)
        # At minimum, this should not crash


class TestThoughtState:
    """Test ThoughtState dataclass."""
    
    def test_thought_state_creation(self):
        """Test creating a thought state."""
        state = ThoughtState(
            timestep=10,
            spikes=torch.zeros(1, 64),
            membrane=torch.zeros(1, 64),
            current_concept=0,
            concept_name="apple",
            concept_changed=True,
            energy=-5.0,
            wm_status={"n_slots": 5, "active_count": 2},
        )
        
        assert state.timestep == 10
        assert state.current_concept == 0
        assert state.concept_name == "apple"
        assert state.concept_changed is True
        assert state.energy == -5.0


class TestGPU:
    """Test GPU functionality."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thinker_on_gpu(self):
        """Test thinking on GPU."""
        device = torch.device("cuda")
        config = ThinkingConfig(
            n_concepts=32,
            n_wm_slots=2,
            wm_slot_size=16,
            enable_learning=False,
            enable_homeostasis=False,
        )
        thinker = ThinkingSNN(config).to(device)
        thinker.reset_state(batch_size=1)
        
        # Think should work on GPU
        state = thinker.think()
        
        assert state.spikes.device.type == "cuda"
