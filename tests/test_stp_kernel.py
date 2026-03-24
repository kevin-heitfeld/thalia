"""Equivalence tests for batched STP kernel vs individual ShortTermPlasticity modules.

Tests verify that:
1. C++ STP kernel matches Python STP module for identical inputs
2. STPBatch produces identical efficacy to individual STP.forward() calls
3. Multi-step consistency: batched and individual paths diverge by < tolerance
"""

from __future__ import annotations

import pytest
import torch

from thalia.brain.synapses.stp import ShortTermPlasticity, STPConfig
from thalia.brain.synapses.stp_batch import STPBatch
from thalia.typing import ReceptorType, SynapseId
from thalia.utils.stp_fused import is_available as stp_cpp_available


def _make_synapse_id(idx: int) -> SynapseId:
    """Create a unique SynapseId for testing."""
    return SynapseId(
        source_region=f"region_a",
        source_population=f"pop_{idx}",
        target_region=f"region_b",
        target_population=f"pop_target",
        receptor_type=ReceptorType.AMPA,
    )


def _make_stp(n_pre: int, config: STPConfig, dt_ms: float = 1.0) -> ShortTermPlasticity:
    """Create an STP module with temporal parameters initialized."""
    stp = ShortTermPlasticity(n_pre=n_pre, config=config)
    stp.update_temporal_parameters(dt_ms)
    return stp


# ─── Configs representing different synapse types ─────────────────────────────
DEPRESSING = STPConfig(U=0.5, tau_d=800.0, tau_f=20.0)
FACILITATING = STPConfig(U=0.1, tau_d=300.0, tau_f=300.0)
MIXED = STPConfig(U=0.25, tau_d=400.0, tau_f=150.0)


class TestSTPKernelEquivalence:
    """Test C++ STP kernel against Python STP module."""

    @pytest.fixture(autouse=True)
    def _require_cpp(self):
        if not stp_cpp_available():
            pytest.skip("STP C++ kernel not available")

    def _run_both(
        self, n_pre: int, config: STPConfig, spikes: torch.Tensor, n_steps: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Python and C++ paths, return (py_efficacy, cpp_efficacy, py_u, cpp_u)."""
        from thalia.utils.stp_fused import stp_step as _stp_step_cpp

        dt_ms = 1.0
        stp_py = _make_stp(n_pre, config, dt_ms)

        # Clone state for C++ path
        u_cpp = stp_py.u.clone()
        x_cpp = stp_py.x.clone()
        U = torch.full((n_pre,), config.U)
        decay_d = stp_py.decay_d.expand(n_pre).clone()
        decay_f = stp_py.decay_f.expand(n_pre).clone()
        recovery_d = stp_py.recovery_d.expand(n_pre).clone()
        recovery_f = stp_py.recovery_f.expand(n_pre).clone()

        py_eff = torch.zeros(n_pre)
        cpp_eff = torch.zeros(n_pre)

        for _ in range(n_steps):
            py_eff = stp_py.forward(spikes.float())
            cpp_eff = _stp_step_cpp(
                u_cpp, x_cpp, U,
                decay_d, decay_f, recovery_d, recovery_f,
                spikes.float(), n_pre,
            )

        return py_eff, cpp_eff, stp_py.u.clone(), u_cpp.clone()

    def test_no_spikes(self):
        """With no spikes, only continuous decay occurs."""
        n = 100
        spikes = torch.zeros(n)
        py_eff, cpp_eff, py_u, cpp_u = self._run_both(n, DEPRESSING, spikes)
        torch.testing.assert_close(py_eff, cpp_eff, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(py_u, cpp_u, atol=1e-6, rtol=1e-6)

    def test_all_spikes_depressing(self):
        """All neurons spike — depressing synapse."""
        n = 50
        spikes = torch.ones(n)
        py_eff, cpp_eff, py_u, cpp_u = self._run_both(n, DEPRESSING, spikes)
        torch.testing.assert_close(py_eff, cpp_eff, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(py_u, cpp_u, atol=1e-6, rtol=1e-6)

    def test_all_spikes_facilitating(self):
        """All neurons spike — facilitating synapse."""
        n = 50
        spikes = torch.ones(n)
        py_eff, cpp_eff, py_u, cpp_u = self._run_both(n, FACILITATING, spikes)
        torch.testing.assert_close(py_eff, cpp_eff, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(py_u, cpp_u, atol=1e-6, rtol=1e-6)

    def test_sparse_spikes(self):
        """10% of neurons spike."""
        n = 200
        spikes = (torch.rand(n) < 0.1).float()
        py_eff, cpp_eff, _, _ = self._run_both(n, MIXED, spikes)
        torch.testing.assert_close(py_eff, cpp_eff, atol=1e-6, rtol=1e-6)

    def test_multi_step_consistency(self):
        """Run 20 steps with random spikes and check final state."""
        n = 100
        dt_ms = 1.0
        stp_py = _make_stp(n, DEPRESSING, dt_ms)

        from thalia.utils.stp_fused import stp_step as _stp_step_cpp

        u_cpp = stp_py.u.clone()
        x_cpp = stp_py.x.clone()
        U = torch.full((n,), DEPRESSING.U)
        decay_d = stp_py.decay_d.expand(n).clone()
        decay_f = stp_py.decay_f.expand(n).clone()
        recovery_d = stp_py.recovery_d.expand(n).clone()
        recovery_f = stp_py.recovery_f.expand(n).clone()

        torch.manual_seed(42)
        for _ in range(20):
            spikes = (torch.rand(n) < 0.05).float()
            py_eff = stp_py.forward(spikes)
            cpp_eff = _stp_step_cpp(
                u_cpp, x_cpp, U,
                decay_d, decay_f, recovery_d, recovery_f,
                spikes, n,
            )

        torch.testing.assert_close(stp_py.u, u_cpp, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(stp_py.x, x_cpp, atol=1e-5, rtol=1e-5)

    def test_small_population(self):
        """Edge case: 3 neurons."""
        n = 3
        spikes = torch.tensor([1.0, 0.0, 1.0])
        py_eff, cpp_eff, _, _ = self._run_both(n, DEPRESSING, spikes)
        torch.testing.assert_close(py_eff, cpp_eff, atol=1e-6, rtol=1e-6)

    def test_large_population(self):
        """1000 neurons to exercise parallel_for."""
        n = 1000
        spikes = (torch.rand(n) < 0.1).float()
        py_eff, cpp_eff, _, _ = self._run_both(n, FACILITATING, spikes)
        torch.testing.assert_close(py_eff, cpp_eff, atol=1e-6, rtol=1e-6)


class TestSTPBatchEquivalence:
    """Test STPBatch vs individual ShortTermPlasticity modules."""

    def _build_batch_and_individual(
        self,
        configs: list[tuple[int, STPConfig]],
        dt_ms: float = 1.0,
    ) -> tuple[
        STPBatch,
        dict[SynapseId, ShortTermPlasticity],
        dict[SynapseId, ShortTermPlasticity],
    ]:
        """Build an STPBatch and a set of reference individual STP modules.

        Returns: (batch, batch_stps, ref_stps)
        - batch: the STPBatch
        - batch_stps: the STP modules whose state is aliased into the batch
        - ref_stps: independent reference modules with identical initial state
        """
        batch_entries: list[tuple[SynapseId, ShortTermPlasticity]] = []
        ref_stps: dict[SynapseId, ShortTermPlasticity] = {}

        for idx, (n_pre, config) in enumerate(configs):
            sid = _make_synapse_id(idx)

            stp_for_batch = _make_stp(n_pre, config, dt_ms)
            stp_ref = _make_stp(n_pre, config, dt_ms)

            # Ensure identical initial state
            stp_ref.u.copy_(stp_for_batch.u)
            stp_ref.x.copy_(stp_for_batch.x)

            batch_entries.append((sid, stp_for_batch))
            ref_stps[sid] = stp_ref

        batch = STPBatch(batch_entries, device=torch.device("cpu"))
        batch_stps = {sid: stp for sid, stp in batch_entries}
        return batch, batch_stps, ref_stps

    def test_single_connection_no_spikes(self):
        """One connection, no spikes — continuous decay only."""
        n = 50
        batch, _, ref_stps = self._build_batch_and_individual([(n, DEPRESSING)])
        sid = _make_synapse_id(0)

        # Build region_inputs with no spikes
        region_inputs = {"region_b": {sid: torch.zeros(n, dtype=torch.bool)}}
        batch_eff = batch.step(region_inputs, last_brain_output=None)

        ref_eff = ref_stps[sid].forward(torch.zeros(n))

        torch.testing.assert_close(batch_eff[sid], ref_eff, atol=1e-6, rtol=1e-6)

    def test_single_connection_with_spikes(self):
        """One connection, sparse spikes."""
        n = 100
        batch, _, ref_stps = self._build_batch_and_individual([(n, DEPRESSING)])
        sid = _make_synapse_id(0)

        torch.manual_seed(99)
        spikes = torch.rand(n) < 0.2

        region_inputs = {"region_b": {sid: spikes}}
        batch_eff = batch.step(region_inputs, last_brain_output=None)
        ref_eff = ref_stps[sid].forward(spikes.float())

        torch.testing.assert_close(batch_eff[sid], ref_eff, atol=1e-6, rtol=1e-6)

    def test_multiple_connections(self):
        """Three connections with different configs."""
        configs = [(50, DEPRESSING), (80, FACILITATING), (30, MIXED)]
        batch, _, ref_stps = self._build_batch_and_individual(configs)

        torch.manual_seed(42)
        spikes_dict: dict[SynapseId, torch.Tensor] = {}
        for idx, (n, _) in enumerate(configs):
            sid = _make_synapse_id(idx)
            spikes_dict[sid] = torch.rand(n) < 0.1

        region_inputs = {"region_b": spikes_dict}
        batch_eff = batch.step(region_inputs, last_brain_output=None)

        for idx, (n, _) in enumerate(configs):
            sid = _make_synapse_id(idx)
            ref_eff = ref_stps[sid].forward(spikes_dict[sid].float())
            torch.testing.assert_close(
                batch_eff[sid], ref_eff,
                atol=1e-5, rtol=1e-5,
                msg=f"Mismatch for connection {idx}",
            )

    def test_multi_step_multiple_connections(self):
        """Run 10 steps over 3 connections with random spikes."""
        configs = [(60, DEPRESSING), (40, FACILITATING), (20, MIXED)]
        batch, _, ref_stps = self._build_batch_and_individual(configs)

        torch.manual_seed(123)
        for step in range(10):
            spikes_dict: dict[SynapseId, torch.Tensor] = {}
            for idx, (n, _) in enumerate(configs):
                sid = _make_synapse_id(idx)
                spikes_dict[sid] = torch.rand(n) < 0.05

            region_inputs = {"region_b": spikes_dict}
            batch_eff = batch.step(region_inputs, last_brain_output=None)

            for idx, (n, _) in enumerate(configs):
                sid = _make_synapse_id(idx)
                ref_eff = ref_stps[sid].forward(spikes_dict[sid].float())
                torch.testing.assert_close(
                    batch_eff[sid], ref_eff,
                    atol=1e-5, rtol=1e-5,
                    msg=f"Mismatch at step {step} for connection {idx}",
                )

    def test_missing_spikes_treated_as_zero(self):
        """If a synapse_id has no spikes in region_inputs, it should be treated as zero."""
        n = 50
        batch, _, ref_stps = self._build_batch_and_individual([(n, DEPRESSING)])
        sid = _make_synapse_id(0)

        # Empty region_inputs — no spike data at all
        region_inputs: dict[str, dict[SynapseId, torch.Tensor]] = {}
        batch_eff = batch.step(region_inputs, last_brain_output=None)

        ref_eff = ref_stps[sid].forward(torch.zeros(n))
        torch.testing.assert_close(batch_eff[sid], ref_eff, atol=1e-6, rtol=1e-6)

    def test_state_views_synchronized(self):
        """Individual STP modules' u/x should be views into the batch arrays."""
        configs = [(30, DEPRESSING), (40, FACILITATING)]
        batch, batch_stps, _ = self._build_batch_and_individual(configs)

        sid0 = _make_synapse_id(0)
        sid1 = _make_synapse_id(1)

        # Run a step with spikes
        spikes0 = torch.ones(30, dtype=torch.bool)
        spikes1 = torch.zeros(40, dtype=torch.bool)
        region_inputs = {"region_b": {sid0: spikes0, sid1: spikes1}}
        batch.step(region_inputs, last_brain_output=None)

        # The individual STP modules' u/x should reflect the batch state
        off0, cnt0 = batch.registry[sid0]
        off1, cnt1 = batch.registry[sid1]
        torch.testing.assert_close(batch_stps[sid0].u, batch.u[off0:off0+cnt0])
        torch.testing.assert_close(batch_stps[sid1].x, batch.x[off1:off1+cnt1])

    def test_python_fallback(self):
        """STPBatch._step_python should match the C++ path."""
        configs = [(50, DEPRESSING), (30, FACILITATING)]
        batch, _, ref_stps = self._build_batch_and_individual(configs)

        torch.manual_seed(7)
        spikes_dict: dict[SynapseId, torch.Tensor] = {}
        for idx, (n, _) in enumerate(configs):
            sid = _make_synapse_id(idx)
            spikes_dict[sid] = torch.rand(n) < 0.15

        # Force Python path
        batch._use_cpp = False
        region_inputs = {"region_b": spikes_dict}
        batch_eff = batch.step(region_inputs, last_brain_output=None)

        for idx, (n, _) in enumerate(configs):
            sid = _make_synapse_id(idx)
            ref_eff = ref_stps[sid].forward(spikes_dict[sid].float())
            torch.testing.assert_close(batch_eff[sid], ref_eff, atol=1e-5, rtol=1e-5)
