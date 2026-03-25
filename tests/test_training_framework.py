"""Tests for the training framework (no brain required)."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import torch

from training.checkpointing.checkpoint import (
    _NEURON_BATCH_STATE_KEYS,
    _STP_BATCH_STATE_KEYS,
    load_checkpoint,
    save_checkpoint,
)
from training.encoding.spike_decoder import ReadoutGroup
from training.encoding.spike_encoder import encode_population_rate
from training.monitoring.health_monitor import HealthMonitor, TrialHealthSummary
from training.monitoring.training_logger import LoggerConfig, TrainingLogger
from training.tasks.base import Trial, TrialResult
from training.tasks.pattern_association import (
    CONVERGENCE_THRESHOLD,
    CONVERGENCE_WINDOW,
    RELAY_SIZE,
    RESPONSE_WINDOW,
    STIMULUS_END,
    STIMULUS_START,
    TRIAL_DURATION,
    PatternAssociationTask,
    make_readout_groups,
)


# ── spike_encoder ──────────────────────────────────────────────────────────


class TestEncodePopulationRate:
    def test_output_shape(self) -> None:
        indices = torch.arange(0, 50)
        inp = encode_population_rate(250, indices, 0.5, "thalamus_sensory")
        assert len(inp) == 1
        tensor = next(iter(inp.values()))
        assert tensor.shape == (250,)
        assert tensor.dtype == torch.bool

    def test_zero_prob_produces_no_spikes(self) -> None:
        indices = torch.arange(0, 125)
        inp = encode_population_rate(250, indices, 0.0, "thalamus_sensory")
        tensor = next(iter(inp.values()))
        assert tensor.sum().item() == 0

    def test_one_prob_produces_all_spikes(self) -> None:
        indices = torch.arange(0, 125)
        inp = encode_population_rate(250, indices, 1.0, "thalamus_sensory")
        tensor = next(iter(inp.values()))
        assert tensor[:125].sum().item() == 125
        assert tensor[125:].sum().item() == 0

    def test_empty_indices(self) -> None:
        inp = encode_population_rate(250, torch.tensor([], dtype=torch.long), 1.0, "thalamus_sensory")
        tensor = next(iter(inp.values()))
        assert tensor.sum().item() == 0


# ── spike_decoder ──────────────────────────────────────────────────────────


class TestReadoutGroup:
    def test_count_slices_correctly(self) -> None:
        spikes = torch.zeros(10, dtype=torch.bool)
        spikes[2] = True
        spikes[3] = True
        spikes[7] = True
        outputs = {"region": {"pop": spikes}}
        group = ReadoutGroup("test", "region", "pop", 0, 5)
        assert group.count(outputs) == 2  # indices 2, 3

    def test_count_second_half(self) -> None:
        spikes = torch.zeros(10, dtype=torch.bool)
        spikes[7] = True
        spikes[9] = True
        outputs = {"region": {"pop": spikes}}
        group = ReadoutGroup("test", "region", "pop", 5, 10)
        assert group.count(outputs) == 2


# ── pattern_association task ───────────────────────────────────────────────


class TestPatternAssociationTask:
    def setup_method(self) -> None:
        self.task = PatternAssociationTask()

    def test_generate_trial_structure(self) -> None:
        trial = self.task.generate_trial(0)
        assert trial.pattern_id in ("A", "B")
        assert trial.duration_steps == TRIAL_DURATION
        assert trial.response_window == RESPONSE_WINDOW

    def test_make_input_during_stimulus(self) -> None:
        trial = Trial(pattern_id="A", duration_steps=TRIAL_DURATION, response_window=RESPONSE_WINDOW)
        inp = self.task.make_input(trial, STIMULUS_START)
        assert inp is not None
        tensor = next(iter(inp.values()))
        assert tensor.shape == (RELAY_SIZE,)

    def test_make_input_during_baseline_is_none(self) -> None:
        trial = Trial(pattern_id="A", duration_steps=TRIAL_DURATION, response_window=RESPONSE_WINDOW)
        assert self.task.make_input(trial, 0) is None
        assert self.task.make_input(trial, STIMULUS_START - 1) is None

    def test_make_input_during_settling_is_none(self) -> None:
        trial = Trial(pattern_id="A", duration_steps=TRIAL_DURATION, response_window=RESPONSE_WINDOW)
        assert self.task.make_input(trial, STIMULUS_END) is None

    def test_evaluate_correct_a(self) -> None:
        trial = Trial(pattern_id="A", duration_steps=TRIAL_DURATION, response_window=RESPONSE_WINDOW)
        result = self.task.evaluate(trial, {"group_a": 100, "group_b": 50})
        assert result.correct is True
        assert result.reward == 1.0

    def test_evaluate_wrong_a(self) -> None:
        trial = Trial(pattern_id="A", duration_steps=TRIAL_DURATION, response_window=RESPONSE_WINDOW)
        result = self.task.evaluate(trial, {"group_a": 30, "group_b": 80})
        assert result.correct is False
        assert result.reward == -0.5

    def test_evaluate_tie(self) -> None:
        trial = Trial(pattern_id="A", duration_steps=TRIAL_DURATION, response_window=RESPONSE_WINDOW)
        result = self.task.evaluate(trial, {"group_a": 50, "group_b": 50})
        assert result.correct is False
        assert result.reward == 0.0

    def test_evaluate_correct_b(self) -> None:
        trial = Trial(pattern_id="B", duration_steps=TRIAL_DURATION, response_window=RESPONSE_WINDOW)
        result = self.task.evaluate(trial, {"group_a": 30, "group_b": 80})
        assert result.correct is True
        assert result.reward == 1.0

    def test_is_learned_insufficient_window(self) -> None:
        results = [TrialResult(reward=1.0, correct=True)] * (CONVERGENCE_WINDOW - 1)
        assert self.task.is_learned(results) is False

    def test_is_learned_above_threshold(self) -> None:
        n_correct = int(CONVERGENCE_WINDOW * CONVERGENCE_THRESHOLD) + 1
        n_wrong = CONVERGENCE_WINDOW - n_correct
        results = (
            [TrialResult(reward=-0.5, correct=False)] * n_wrong
            + [TrialResult(reward=1.0, correct=True)] * n_correct
        )
        assert self.task.is_learned(results) is True

    def test_is_learned_below_threshold(self) -> None:
        n_correct = int(CONVERGENCE_WINDOW * CONVERGENCE_THRESHOLD) - 5
        n_wrong = CONVERGENCE_WINDOW - n_correct
        results = (
            [TrialResult(reward=-0.5, correct=False)] * n_wrong
            + [TrialResult(reward=1.0, correct=True)] * n_correct
        )
        assert self.task.is_learned(results) is False


class TestReadoutGroups:
    def test_two_groups_cover_readout(self) -> None:
        groups = make_readout_groups()
        assert len(groups) == 2
        assert groups[0].name == "group_a"
        assert groups[1].name == "group_b"
        # Non-overlapping
        assert groups[0].end_neuron <= groups[1].start_neuron


# ── Tier 1 health monitor (no brain needed) ────────────────────────────────


class TestTrialHealthSummary:
    def test_dataclass_fields(self) -> None:
        summary = TrialHealthSummary(
            population_spike_counts={("r", "p"): 42},
            trial_steps=100,
        )
        assert summary.population_spike_counts[("r", "p")] == 42
        assert summary.trial_steps == 100
        assert summary.silent_populations == []
        assert summary.hyperactive_populations == []


class TestHealthMonitorTier1:
    """Test Tier 1 spike accumulation using fake BrainOutput dicts."""

    def _fake_outputs(self, region: str, pop: str, n: int, n_active: int) -> dict:
        spikes = torch.zeros(n, dtype=torch.bool)
        spikes[:n_active] = True
        return {region: {pop: spikes}}

    def test_record_and_end_trial(self) -> None:
        """Verify spike counts accumulate across timesteps."""

        # Create a mock brain-like object with minimal attributes
        class FakePop:
            def __init__(self, n: int) -> None:
                self.n_neurons = n

        class FakeRegion:
            def __init__(self, name: str, pops: dict) -> None:
                self.region_name = name
                self.neuron_populations = {k: FakePop(v) for k, v in pops.items()}

        class FakeBrain:
            def __init__(self) -> None:
                self.regions = {
                    "r1": FakeRegion("r1", {"p1": 10, "p2": 5}),
                }

            def values(self):
                return self.regions.values()

        brain = FakeBrain()
        brain.regions = {"r1": FakeRegion("r1", {"p1": 10, "p2": 5})}

        # Construct monitor using duck-typed brain
        monitor = HealthMonitor.__new__(HealthMonitor)
        monitor.brain = brain  # type: ignore[assignment]
        monitor.hyperactive_hz = 100.0
        monitor._spike_counts = {}
        monitor._trial_steps = 0
        monitor._pop_sizes = {("r1", "p1"): 10, ("r1", "p2"): 5}

        # Simulate 10 timesteps: p1 gets 3 spikes/step, p2 gets 0
        for _ in range(10):
            monitor.record_step(self._fake_outputs("r1", "p1", 10, 3))

        summary = monitor.end_trial()
        assert summary.population_spike_counts[("r1", "p1")] == 30
        assert summary.trial_steps == 10
        # p2 never appeared in outputs, so not tracked (only p1 is)
        assert ("r1", "p1") not in summary.silent_populations

    def test_silent_population_detected(self) -> None:
        """Populations with 0 spikes should be flagged as silent."""
        monitor = HealthMonitor.__new__(HealthMonitor)
        monitor.brain = None  # type: ignore[assignment]
        monitor.hyperactive_hz = 100.0
        monitor._spike_counts = {}
        monitor._trial_steps = 0
        monitor._pop_sizes = {("r1", "p1"): 10}

        # Feed one timestep with all-zero spikes
        monitor.record_step(self._fake_outputs("r1", "p1", 10, 0))
        summary = monitor.end_trial()
        assert ("r1", "p1") in summary.silent_populations

    def test_hyperactive_population_detected(self) -> None:
        """Populations above 100 Hz should be flagged as hyperactive."""
        monitor = HealthMonitor.__new__(HealthMonitor)
        monitor.brain = None  # type: ignore[assignment]
        monitor.hyperactive_hz = 100.0
        monitor._spike_counts = {}
        monitor._trial_steps = 0
        monitor._pop_sizes = {("r1", "p1"): 10}

        # 500 steps × 10 neurons all firing = 5000 spikes
        # Rate = 5000 / (10 neurons × 0.5 s) = 1000 Hz >> 100 Hz
        for _ in range(500):
            monitor.record_step(self._fake_outputs("r1", "p1", 10, 10))

        summary = monitor.end_trial()
        assert ("r1", "p1") in summary.hyperactive_populations

    def test_reset_between_trials(self) -> None:
        """end_trial() should reset accumulators for the next trial."""
        monitor = HealthMonitor.__new__(HealthMonitor)
        monitor.brain = None  # type: ignore[assignment]
        monitor.hyperactive_hz = 100.0
        monitor._spike_counts = {}
        monitor._trial_steps = 0
        monitor._pop_sizes = {("r1", "p1"): 10}

        monitor.record_step(self._fake_outputs("r1", "p1", 10, 5))
        monitor.end_trial()

        # After end_trial, accumulators should be empty
        assert monitor._spike_counts == {}
        assert monitor._trial_steps == 0


# ── Training logger ────────────────────────────────────────────────────────


_TEST_LOG_DIR = "data/training/_test_logs"


class TestTrainingLogger:
    def setup_method(self) -> None:
        # Clean up any leftover test directory
        p = Path(_TEST_LOG_DIR)
        if p.exists():
            shutil.rmtree(p)

    def teardown_method(self) -> None:
        p = Path(_TEST_LOG_DIR)
        if p.exists():
            shutil.rmtree(p)

    def test_creates_run_directory_and_files(self) -> None:
        cfg = LoggerConfig(log_dir=_TEST_LOG_DIR)
        logger = TrainingLogger(cfg)
        assert logger.run_dir.exists()
        assert (logger.run_dir / "trials.csv").exists()
        assert (logger.run_dir / "events.log").exists()
        logger.close()

    def test_log_trial_writes_csv_row(self) -> None:
        cfg = LoggerConfig(log_dir=_TEST_LOG_DIR)
        logger = TrainingLogger(cfg)

        result = TrialResult(
            reward=1.0, correct=True,
            metrics={"pattern": "A", "count_a": 100, "count_b": 50, "margin": 50},
        )
        health = TrialHealthSummary(
            population_spike_counts={}, trial_steps=500,
        )
        logger.log_trial(0, result, health, [result])
        logger.close()

        csv_path = logger.run_dir / "trials.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["trial"] == "0"
        assert rows[0]["correct"] == "1"
        assert rows[0]["pattern"] == "A"

    def test_log_diagnostics_writes_event(self) -> None:
        cfg = LoggerConfig(log_dir=_TEST_LOG_DIR)
        logger = TrainingLogger(cfg)
        logger.log_diagnostics(50, n_critical=2, n_warning=10, brain_state="active/theta/critical")
        logger.close()

        events_path = logger.run_dir / "events.log"
        text = events_path.read_text(encoding="utf-8")
        assert "DIAGNOSTICS" in text
        assert "criticals=2" in text
        assert "active/theta/critical" in text

    def test_log_alert_writes_event(self) -> None:
        cfg = LoggerConfig(log_dir=_TEST_LOG_DIR)
        logger = TrainingLogger(cfg)
        logger.log_alert(99, "test alert message")
        logger.close()

        events_path = logger.run_dir / "events.log"
        text = events_path.read_text(encoding="utf-8")
        assert "ALERT" in text
        assert "test alert message" in text


# ── Checkpointing ─────────────────────────────────────────────────────────


_TEST_CKPT_DIR = "data/training/_test_ckpts"


class _FakeNeuronBatch:
    """Minimal stand-in for ConductanceLIFBatch with all state tensors."""

    def __init__(self, n: int = 10) -> None:
        for key in _NEURON_BATCH_STATE_KEYS:
            dtype = torch.int32 if key == "refractory" else torch.float32
            setattr(self, key, torch.randn(n).to(dtype))


class _FakeSTPBatch:
    """Minimal stand-in for STPBatch with all state tensors."""

    def __init__(self, n: int = 8) -> None:
        for key in _STP_BATCH_STATE_KEYS:
            setattr(self, key, torch.randn(n))


class _FakeBrain(torch.nn.Module):
    """nn.Module with _neuron_batch and _stp_batch for checkpoint testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2, bias=False)
        self._neuron_batch = _FakeNeuronBatch()
        self._stp_batch = _FakeSTPBatch()


class TestCheckpoint:
    def setup_method(self) -> None:
        p = Path(_TEST_CKPT_DIR)
        if p.exists():
            shutil.rmtree(p)

    def teardown_method(self) -> None:
        p = Path(_TEST_CKPT_DIR)
        if p.exists():
            shutil.rmtree(p)

    def test_save_creates_file(self) -> None:
        brain = _FakeBrain()
        path = save_checkpoint(brain, f"{_TEST_CKPT_DIR}/ckpt.pt")  # type: ignore[arg-type]
        assert path.exists()
        assert path.suffix == ".pt"

    def test_roundtrip_restores_module_state(self) -> None:
        brain = _FakeBrain()
        original_w = brain.linear.weight.data.clone()

        save_checkpoint(brain, f"{_TEST_CKPT_DIR}/ckpt.pt")  # type: ignore[arg-type]

        # Corrupt weights
        brain.linear.weight.data.fill_(999.0)
        assert not torch.equal(brain.linear.weight.data, original_w)

        load_checkpoint(brain, f"{_TEST_CKPT_DIR}/ckpt.pt")  # type: ignore[arg-type]
        assert torch.equal(brain.linear.weight.data, original_w)

    def test_roundtrip_restores_neuron_state(self) -> None:
        brain = _FakeBrain()
        original_v = brain._neuron_batch.V_soma.clone()

        save_checkpoint(brain, f"{_TEST_CKPT_DIR}/ckpt.pt")  # type: ignore[arg-type]

        brain._neuron_batch.V_soma.fill_(0.0)
        load_checkpoint(brain, f"{_TEST_CKPT_DIR}/ckpt.pt")  # type: ignore[arg-type]

        assert torch.equal(brain._neuron_batch.V_soma, original_v)

    def test_roundtrip_restores_stp_state(self) -> None:
        brain = _FakeBrain()
        original_u = brain._stp_batch.u.clone()

        save_checkpoint(brain, f"{_TEST_CKPT_DIR}/ckpt.pt")  # type: ignore[arg-type]

        brain._stp_batch.u.fill_(0.0)
        load_checkpoint(brain, f"{_TEST_CKPT_DIR}/ckpt.pt")  # type: ignore[arg-type]

        assert torch.equal(brain._stp_batch.u, original_u)

    def test_metadata_roundtrip(self) -> None:
        brain = _FakeBrain()
        meta = {"trial_idx": 42, "accuracy": 0.85}

        save_checkpoint(brain, f"{_TEST_CKPT_DIR}/ckpt.pt", metadata=meta)  # type: ignore[arg-type]
        restored_meta = load_checkpoint(brain, f"{_TEST_CKPT_DIR}/ckpt.pt")  # type: ignore[arg-type]

        assert restored_meta == meta

    def test_subdirectory_created_automatically(self) -> None:
        brain = _FakeBrain()
        deep_path = f"{_TEST_CKPT_DIR}/deep/nested/ckpt.pt"
        path = save_checkpoint(brain, deep_path)  # type: ignore[arg-type]
        assert path.exists()
