"""
Unit tests for checkpoint state management (Phase 1A).

Tests that regions can save and restore their complete state without file I/O.
This validates the get_full_state() and load_full_state() API before
implementing the binary format in Phase 1B.
"""

import pytest
import torch

from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.hippocampus import TrisynapticHippocampus, TrisynapticConfig
from thalia.core.event_system import TrialPhase


class TestStriatumStateRoundtrip:
    """Test that Striatum can save and restore complete state."""

    def test_basic_state_roundtrip(self):
        """Test basic state save/load without any activity."""
        config = StriatumConfig(
            n_input=32,
            n_output=4,  # 4 actions
            device="cpu",
        )
        
        striatum1 = Striatum(config)
        
        # Get initial state
        state = striatum1.get_full_state()
        
        # Create a new instance and load state
        striatum2 = Striatum(config)
        striatum2.load_full_state(state)
        
        # Verify weights match
        assert torch.allclose(striatum1.weights, striatum2.weights)
        assert torch.allclose(striatum1.d1_weights, striatum2.d1_weights)
        assert torch.allclose(striatum1.d2_weights, striatum2.d2_weights)
        
        # Verify config matches
        assert striatum2.striatum_config.n_input == config.n_input
        # n_output gets expanded by neurons_per_action when population_coding=True
        expected_n_output = config.n_output * config.neurons_per_action if config.population_coding else config.n_output
        assert striatum2.striatum_config.n_output == expected_n_output

    def test_state_with_activity(self):
        """Test state save/load after some activity."""
        config = StriatumConfig(
            n_input=32,
            n_output=4,
            device="cpu",
        )
        
        striatum1 = Striatum(config)
        
        # Run some forward passes to build up state (ADR-005: 1D tensors)
        input_spikes = (torch.rand(32) > 0.8).float()
        for _ in range(10):
            output = striatum1.forward(input_spikes)
        
        # Simulate action selection and reward
        action_result = striatum1.finalize_action(explore=False)
        selected_action = action_result["selected_action"]
        striatum1.deliver_reward(reward=1.0)
        
        # Get state after activity
        state = striatum1.get_full_state()
        
        # Create new instance and load
        striatum2 = Striatum(config)
        striatum2.load_full_state(state)
        
        # Verify eligibility traces match
        assert torch.allclose(
            striatum1.eligibility.get(),
            striatum2.eligibility.get(),
            atol=1e-6
        )
        
        # Verify last action matches
        assert striatum1.last_action == striatum2.last_action
        
        # Verify dopamine state matches
        assert abs(striatum1.state.dopamine - striatum2.state.dopamine) < 1e-6

    def test_state_with_value_estimates(self):
        """Test that value estimates (RPE) are preserved."""
        config = StriatumConfig(
            n_input=32,
            n_output=4,
            rpe_enabled=True,
            device="cpu",
        )
        
        striatum1 = Striatum(config)
        
        # Update value estimates
        striatum1.update_value_estimate(action=0, reward=0.5)
        striatum1.update_value_estimate(action=2, reward=1.0)
        
        # Save and load
        state = striatum1.get_full_state()
        striatum2 = Striatum(config)
        striatum2.load_full_state(state)
        
        # Verify value estimates match
        assert torch.allclose(striatum1.value_estimates, striatum2.value_estimates)

    def test_dimension_mismatch_raises_error(self):
        """Test that loading incompatible state raises ValueError."""
        config1 = StriatumConfig(n_input=32, n_output=4, device="cpu")
        config2 = StriatumConfig(n_input=64, n_output=4, device="cpu")
        
        striatum1 = Striatum(config1)
        state = striatum1.get_full_state()
        
        striatum2 = Striatum(config2)
        
        with pytest.raises(ValueError, match="Input dimension mismatch"):
            striatum2.load_full_state(state)


class TestHippocampusStateRoundtrip:
    """Test that TrisynapticHippocampus can save and restore complete state."""

    def test_basic_state_roundtrip(self):
        """Test basic state save/load for hippocampus."""
        config = TrisynapticConfig(
            n_input=64,
            n_output=128,
            dg_expansion=3.0,
            ca3_size_ratio=0.5,
            ca1_size_ratio=1.0,
            device="cpu",
        )
        
        hipp1 = TrisynapticHippocampus(config)
        
        # Get initial state
        state = hipp1.get_full_state()
        
        # Create new instance and load
        hipp2 = TrisynapticHippocampus(config)
        hipp2.load_full_state(state)
        
        # Verify all pathway weights match
        assert torch.allclose(hipp1.w_ec_dg, hipp2.w_ec_dg)
        assert torch.allclose(hipp1.w_dg_ca3, hipp2.w_dg_ca3)
        assert torch.allclose(hipp1.w_ca3_ca1, hipp2.w_ca3_ca1)
        assert torch.allclose(hipp1.w_ca3_ca3, hipp2.w_ca3_ca3)

    def test_state_with_sequences(self):
        """Test state save/load after encoding sequences."""
        config = TrisynapticConfig(
            n_input=64,
            n_output=128,
            dg_expansion=3.0,
            ca3_size_ratio=0.5,
            ca1_size_ratio=1.0,
            device="cpu",
        )
        
        hipp1 = TrisynapticHippocampus(config)
        
        # Encode a sequence (convert bool to float) (ADR-005: 1D patterns)
        sequence_patterns = [(torch.rand(64) > 0.9).float() for _ in range(5)]
        
        for pattern in sequence_patterns:
            output = hipp1.forward(
                pattern,
                phase=TrialPhase.ENCODE,
                encoding_mod=1.0,
                retrieval_mod=0.0,
                dt=1.0,
            )
            if hipp1.gamma_oscillator is not None:
                hipp1.gamma_oscillator.advance(dt_ms=20.0)  # Advance gamma
        
        # Save state
        state = hipp1.get_full_state()
        
        # Create new instance and load
        hipp2 = TrisynapticHippocampus(config)
        hipp2.load_full_state(state)
        
        # Verify oscillator state matches
        if hipp1.gamma_oscillator is not None:
            assert abs(
                hipp1.gamma_oscillator.theta_phase -
                hipp2.gamma_oscillator.theta_phase
            ) < 1e-5
            assert abs(
                hipp1.gamma_oscillator.gamma_phase -
                hipp2.gamma_oscillator.gamma_phase
            ) < 1e-5
        
        # Verify sequence position matches
        assert hipp1._sequence_position == hipp2._sequence_position

    def test_episode_buffer_preserved(self):
        """Test that episodic memory buffer is preserved."""
        config = TrisynapticConfig(
            n_input=64,
            n_output=128,
            device="cpu",
        )
        
        hipp1 = TrisynapticHippocampus(config)
        
        # Store some episodes (ADR-005: 1D patterns)
        for i in range(3):
            state_pattern = torch.rand(100)
            hipp1.store_episode(
                state=state_pattern,
                action=i,
                reward=float(i) * 0.5,
                correct=(i % 2 == 0),
            )
        
        # Save and load
        state = hipp1.get_full_state()
        hipp2 = TrisynapticHippocampus(config)
        hipp2.load_full_state(state)
        
        # Verify episode buffer matches
        assert len(hipp1.episode_buffer) == len(hipp2.episode_buffer)
        for ep1, ep2 in zip(hipp1.episode_buffer, hipp2.episode_buffer):
            assert torch.allclose(ep1.state, ep2.state)
            assert ep1.action == ep2.action
            assert abs(ep1.reward - ep2.reward) < 1e-6
            assert ep1.correct == ep2.correct


class TestCortexStateRoundtrip:
    """Test that LayeredCortex can save and restore complete state."""

    def test_basic_state_roundtrip(self):
        """Test basic state save/load for layered cortex."""
        from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig
        
        config = LayeredCortexConfig(
            n_input=64,
            n_output=128,
            device="cpu",
            bcm_enabled=True,
            stp_l23_recurrent_enabled=True,
        )
        
        cortex1 = LayeredCortex(config)
        
        # Get initial state
        state = cortex1.get_full_state()
        
        # Verify state structure
        assert "weights" in state
        assert "region_state" in state
        assert "learning_state" in state
        assert "neuromodulator_state" in state
        
        # Create new instance and load
        cortex2 = LayeredCortex(config)
        cortex2.load_full_state(state)
        
        # Verify all layer weights match
        assert torch.allclose(cortex1.w_input_l4, cortex2.w_input_l4)
        assert torch.allclose(cortex1.w_l4_l23, cortex2.w_l4_l23)
        assert torch.allclose(cortex1.w_l23_recurrent, cortex2.w_l23_recurrent)
        assert torch.allclose(cortex1.w_l23_l5, cortex2.w_l23_l5)
        assert torch.allclose(cortex1.w_l23_inhib, cortex2.w_l23_inhib)

    def test_state_with_bcm_thresholds(self):
        """Test that BCM thresholds are preserved (when they exist)."""
        from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig
        
        config = LayeredCortexConfig(
            n_input=64,
            n_output=128,
            device="cpu",
            bcm_enabled=True,
        )
        
        cortex1 = LayeredCortex(config)
        
        # Note: BCM theta may be None initially (lazy initialization)
        # Just verify state can be saved/loaded
        state = cortex1.get_full_state()
        
        cortex2 = LayeredCortex(config)
        cortex2.load_full_state(state)
        
        # If BCM was initialized, verify thresholds match
        if (cortex1.bcm_l4 is not None and hasattr(cortex1.bcm_l4, 'theta') and 
            cortex1.bcm_l4.theta is not None and cortex2.bcm_l4 is not None and
            cortex2.bcm_l4.theta is not None):
            assert torch.allclose(cortex1.bcm_l4.theta, cortex2.bcm_l4.theta)


class TestPrefrontalStateRoundtrip:
    """Test that Prefrontal can save and restore complete state."""

    def test_basic_state_roundtrip(self):
        """Test basic state save/load for prefrontal cortex."""
        from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
        
        config = PrefrontalConfig(
            n_input=32,
            n_output=64,
            device="cpu",
            stp_recurrent_enabled=True,
        )
        
        pfc1 = Prefrontal(config)
        
        # Get initial state
        state = pfc1.get_full_state()
        
        # Verify state structure
        assert "weights" in state
        assert "feedforward" in state["weights"]
        assert "recurrent" in state["weights"]
        assert "inhibition" in state["weights"]
        
        # Create new instance and load
        pfc2 = Prefrontal(config)
        pfc2.load_full_state(state)
        
        # Verify all weights match
        assert torch.allclose(pfc1.weights, pfc2.weights)
        assert torch.allclose(pfc1.rec_weights, pfc2.rec_weights)
        assert torch.allclose(pfc1.inhib_weights, pfc2.inhib_weights)

    def test_working_memory_preserved(self):
        """Test that working memory state is preserved."""
        from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
        
        config = PrefrontalConfig(
            n_input=32,
            n_output=64,
            device="cpu",
        )
        
        pfc1 = Prefrontal(config)
        
        # Set working memory (ADR-005: 1D context)
        wm_pattern = torch.rand(64)
        pfc1.set_context(wm_pattern)
        
        # Save and load
        state = pfc1.get_full_state()
        pfc2 = Prefrontal(config)
        pfc2.load_full_state(state)
        
        # Verify working memory matches
        assert pfc1.state.working_memory is not None
        assert pfc2.state.working_memory is not None
        assert torch.allclose(pfc1.state.working_memory, pfc2.state.working_memory)


class TestCerebellumStateRoundtrip:
    """Test that Cerebellum can save and restore complete state."""

    def test_basic_state_roundtrip(self):
        """Test basic state save/load for cerebellum."""
        from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
        
        config = CerebellumConfig(
            n_input=32,
            n_output=16,
            device="cpu",
        )
        
        cereb1 = Cerebellum(config)
        
        # Get initial state
        state = cereb1.get_full_state()
        
        # Verify state structure
        assert "weights" in state
        assert "region_state" in state
        assert "learning_state" in state
        
        # Create new instance and load
        cereb2 = Cerebellum(config)
        cereb2.load_full_state(state)
        
        # Verify weights match
        assert torch.allclose(cereb1.weights, cereb2.weights)

    def test_eligibility_traces_preserved(self):
        """Test that eligibility traces are preserved."""
        from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
        
        config = CerebellumConfig(
            n_input=32,
            n_output=16,
            device="cpu",
        )
        
        cereb1 = Cerebellum(config)
        
        # Build up some eligibility traces (ADR-005: 1D tensors)
        input_spikes = (torch.rand(32) > 0.8).float()
        for _ in range(5):
            cereb1.forward(input_spikes)
        
        # Save and load
        state = cereb1.get_full_state()
        cereb2 = Cerebellum(config)
        cereb2.load_full_state(state)
        
        # Verify traces match
        assert torch.allclose(cereb1.input_trace, cereb2.input_trace, atol=1e-6)
        assert torch.allclose(cereb1.output_trace, cereb2.output_trace, atol=1e-6)
        assert torch.allclose(cereb1.stdp_eligibility, cereb2.stdp_eligibility, atol=1e-6)


class TestBrainStateRoundtrip:
    """Test that EventDrivenBrain can save and restore complete state."""

    def test_basic_brain_state_roundtrip(self):
        """Test basic state save/load for entire brain."""
        from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
        
        config = EventDrivenBrainConfig(
            input_size=64,
            cortex_size=128,
            hippocampus_size=64,
            pfc_size=32,
            n_actions=4,
            device="cpu",
        )
        
        brain1 = EventDrivenBrain(config)
        
        # Get initial state
        state = brain1.get_full_state()
        
        # Verify state structure
        assert "regions" in state
        assert "cortex" in state["regions"]
        assert "hippocampus" in state["regions"]
        assert "pfc" in state["regions"]
        assert "striatum" in state["regions"]
        assert "cerebellum" in state["regions"]
        assert "theta" in state
        assert "scheduler" in state
        assert "trial_state" in state
        assert "config" in state
        
        # Create new instance and load
        brain2 = EventDrivenBrain(config)
        brain2.load_full_state(state)
        
        # Verify cortex weights match
        assert torch.allclose(
            brain1.cortex.impl.w_input_l4,
            brain2.cortex.impl.w_input_l4
        )
        
        # Verify hippocampus weights match
        assert torch.allclose(
            brain1.hippocampus.impl.w_ec_dg,
            brain2.hippocampus.impl.w_ec_dg
        )
        
        # Verify striatum weights match
        assert torch.allclose(
            brain1.striatum.impl.weights,
            brain2.striatum.impl.weights
        )

    def test_brain_state_after_processing(self):
        """Test brain state save/load after processing input."""
        from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
        
        config = EventDrivenBrainConfig(
            input_size=64,
            cortex_size=128,
            hippocampus_size=64,
            pfc_size=32,
            n_actions=4,
            device="cpu",
        )
        
        brain1 = EventDrivenBrain(config)
        
        # Process some input (convert bool to float)
        sample = (torch.rand(64) > 0.8).float()
        brain1.process_sample(sample, n_timesteps=10)
        
        # Save state
        state = brain1.get_full_state()
        
        # Create new instance and load
        brain2 = EventDrivenBrain(config)
        brain2.load_full_state(state)
        
        # Verify trial phase matches (access internal state for testing)
        assert brain1._trial_phase == brain2._trial_phase  # type: ignore[attr-defined]
        
        # Verify event counters match
        assert brain1._events_processed == brain2._events_processed  # type: ignore[attr-defined]

    def test_brain_dimension_mismatch_raises_error(self):
        """Test that loading incompatible brain state raises ValueError."""
        from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
        
        config1 = EventDrivenBrainConfig(
            input_size=64,
            cortex_size=128,
            n_actions=4,
            device="cpu",
        )
        
        config2 = EventDrivenBrainConfig(
            input_size=64,
            cortex_size=256,  # Different size
            n_actions=4,
            device="cpu",
        )
        
        brain1 = EventDrivenBrain(config1)
        state = brain1.get_full_state()
        
        brain2 = EventDrivenBrain(config2)
        
        with pytest.raises(ValueError, match="cortex_size"):
            brain2.load_full_state(state)


class TestPathwayStateRoundtrip:
    """Test state management for spiking pathways."""
    
    def test_attention_pathway_basic_state(self):
        """Test attention pathway state roundtrip."""
        from thalia.integration.pathways.spiking_attention import (
            SpikingAttentionPathway,
            SpikingAttentionPathwayConfig,
        )
        
        config = SpikingAttentionPathwayConfig(
            source_size=64,
            target_size=128,
            input_size=256,
            device="cpu",
        )
        
        pathway1 = SpikingAttentionPathway(config)
        
        # Run some forward passes to accumulate state (ADR-005: 1D tensors)
        pfc_activity = torch.rand(64) > 0.8
        for _ in range(5):
            pathway1.compute_attention(pfc_activity.float(), dt=1.0)
        
        # Save state
        state = pathway1.get_state()
        
        # Create new pathway and load state
        pathway2 = SpikingAttentionPathway(config)
        pathway2.load_state(state)
        
        # Verify weights match
        assert torch.allclose(pathway1.weights, pathway2.weights)
        assert torch.allclose(
            pathway1.input_projection.weight, 
            pathway2.input_projection.weight
        )
        assert torch.allclose(
            pathway1.gain_output.weight,
            pathway2.gain_output.weight
        )
        
        # Verify state matches
        assert torch.allclose(pathway1.membrane, pathway2.membrane)
        assert torch.allclose(pathway1.pre_trace, pathway2.pre_trace)
        assert torch.allclose(pathway1.beta_phase, pathway2.beta_phase)
    
    def test_replay_pathway_basic_state(self):
        """Test replay pathway state roundtrip."""
        from thalia.integration.pathways.spiking_replay import (
            SpikingReplayPathway,
            SpikingReplayPathwayConfig,
        )
        
        config = SpikingReplayPathwayConfig(
            source_size=128,
            target_size=256,
            device="cpu",
        )
        
        pathway1 = SpikingReplayPathway(config)
        
        # Store some patterns in the replay buffer (ADR-005: 1D patterns)
        for i in range(3):
            pattern = torch.rand(128) > 0.7
            pathway1.store_pattern(pattern.float(), priority=float(i + 1))
        
        # Set sleep stage and trigger replay
        pathway1.set_sleep_stage("sws")
        pathway1.trigger_ripple()
        
        # Run some replay steps
        for _ in range(10):
            pathway1.replay_step(dt=1.0)
        
        # Save state
        state = pathway1.get_state()
        
        # Create new pathway and load state
        pathway2 = SpikingReplayPathway(config)
        pathway2.load_state(state)
        
        # Verify weights match
        assert torch.allclose(pathway1.weights, pathway2.weights)
        assert torch.allclose(
            pathway1.replay_projection[0].weight,
            pathway2.replay_projection[0].weight
        )
        assert torch.allclose(
            pathway1.priority_network[0].weight,
            pathway2.priority_network[0].weight
        )
        
        # Verify replay buffer matches
        assert len(pathway2.replay_buffer) == 3
        for entry1, entry2 in zip(pathway1.replay_buffer, pathway2.replay_buffer):
            assert torch.allclose(entry1["pattern"], entry2["pattern"])
            assert entry1["priority"] == entry2["priority"]
            assert entry1["age"] == entry2["age"]
        
        # Verify replay state matches
        assert pathway2.sleep_stage == "sws"
        assert pathway2.replay_active == True
        assert pathway2.replay_count == pathway1.replay_count
    
    def test_brain_with_pathways(self):
        """Test brain state including pathway state."""
        from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
        
        config = EventDrivenBrainConfig(
            input_size=64,
            cortex_size=128,
            hippocampus_size=64,
            pfc_size=32,
            n_actions=4,
            device="cpu",
        )
        
        brain1 = EventDrivenBrain(config)
        
        # Process some input through pathways
        sample = torch.rand(64) > 0.8
        brain1.process_sample(sample.float(), n_timesteps=10)
        
        # Store patterns in replay pathway (ADR-005: 1D patterns)
        hippo_pattern = torch.rand(64) > 0.7
        brain1.replay_pathway.store_pattern(hippo_pattern.float())
        
        # Save state
        state = brain1.get_full_state()
        
        # Verify pathways are in state
        assert "pathways" in state
        assert "attention" in state["pathways"]
        assert "replay" in state["pathways"]
        
        # Create new brain and load state
        brain2 = EventDrivenBrain(config)
        brain2.load_full_state(state)
        
        # Verify pathway weights match
        assert torch.allclose(
            brain1.attention_pathway.weights,
            brain2.attention_pathway.weights
        )
        assert torch.allclose(
            brain1.replay_pathway.weights,
            brain2.replay_pathway.weights
        )
        
        # Verify replay buffer transferred
        assert len(brain2.replay_pathway.replay_buffer) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
