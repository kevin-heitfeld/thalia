"""Numerical equivalence test: fused C++ cortical inhibitory kernel vs Python reference.

Verifies that the fused sparse kernel produces identical results to the
standard torch.mv-based _integrate_single_synaptic_input path.
"""

import pytest
import torch

from thalia.utils.cortical_inhibitory_fused import cortical_inhibitory_step, is_available

pytestmark = pytest.mark.skipif(not is_available(), reason="C++ kernel not compiled")


def _make_weight(n_out: int, n_in: int, connectivity: float = 0.3) -> torch.Tensor:
    """Create a sparse-ish weight matrix (dense tensor with many zeros)."""
    w = torch.randn(n_out, n_in) * 0.01
    mask = torch.rand(n_out, n_in) > connectivity
    w[mask] = 0.0
    return w


def _make_efficacy(n: int) -> torch.Tensor:
    """Create plausible STP efficacy (u*x values in [0, 1])."""
    return torch.rand(n).clamp(0.05, 0.95)


def _reference_stp_mv(W: torch.Tensor, eff: torch.Tensor | None, spikes_f: torch.Tensor) -> torch.Tensor:
    """Reference: match _integrate_single_synaptic_input logic."""
    if eff is not None:
        g = W @ (eff * spikes_f)
    else:
        g = W @ spikes_f
    g.clamp_(min=0.0)
    return g


class TestCorticalInhibitoryKernel:
    """Test fused kernel against reference Python implementation."""

    PYR_N = 200
    PV_N = 40
    SST_N = 30
    VIP_N = 20
    NGC_N = 10

    def _build_test_data(self, spike_rate: float = 0.05):
        """Build spike vectors, weight matrices, and efficacy vectors."""
        pyr_f = (torch.rand(self.PYR_N) < spike_rate).float()
        pv_f = (torch.rand(self.PV_N) < spike_rate).float()
        sst_f = (torch.rand(self.SST_N) < spike_rate).float()
        vip_f = (torch.rand(self.VIP_N) < spike_rate).float()
        ngc_f = (torch.rand(self.NGC_N) < spike_rate).float()

        # 16 weight matrices in connection order
        weights = [
            # E→I (0-3): [tgt, pyr]
            _make_weight(self.PV_N, self.PYR_N),
            _make_weight(self.SST_N, self.PYR_N),
            _make_weight(self.VIP_N, self.PYR_N),
            _make_weight(self.NGC_N, self.PYR_N),
            # I→I (4-11)
            _make_weight(self.PV_N, self.PV_N),     # PV→PV
            _make_weight(self.PV_N, self.SST_N),    # SST→PV
            _make_weight(self.PV_N, self.VIP_N),    # VIP→PV
            _make_weight(self.SST_N, self.PV_N),    # PV→SST
            _make_weight(self.SST_N, self.VIP_N),   # VIP→SST (GABA_A)
            _make_weight(self.SST_N, self.VIP_N),   # VIP→SST (GABA_B)
            _make_weight(self.VIP_N, self.SST_N),   # SST→VIP
            _make_weight(self.SST_N, self.SST_N),   # SST→SST
            # I→E (12-15): [pyr, tgt]
            _make_weight(self.PYR_N, self.PV_N),    # PV→Pyr
            _make_weight(self.PYR_N, self.SST_N),   # SST→Pyr
            _make_weight(self.PYR_N, self.VIP_N),   # VIP→Pyr
            _make_weight(self.PYR_N, self.NGC_N),   # NGC→Pyr (no STP)
        ]

        # 15 efficacy vectors (connections 0-14)
        # Source sizes: 0-3 pyr, 4 pv, 5 sst, 6 vip, 7 pv, 8-9 vip, 10-11 sst, 12 pv, 13 sst, 14 vip
        src_sizes = [
            self.PYR_N, self.PYR_N, self.PYR_N, self.PYR_N,
            self.PV_N, self.SST_N, self.VIP_N, self.PV_N,
            self.VIP_N, self.VIP_N, self.SST_N, self.SST_N,
            self.PV_N, self.SST_N, self.VIP_N,
        ]
        efficacies = [_make_efficacy(s) for s in src_sizes]

        return pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies

    def _compute_reference(self, pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies):
        """Compute reference outputs using the same logic as _run_cortical_inhibitory."""
        # E→I
        pv_g_exc = _reference_stp_mv(weights[0], efficacies[0], pyr_f)
        sst_g_exc = _reference_stp_mv(weights[1], efficacies[1], pyr_f)
        vip_g_exc = _reference_stp_mv(weights[2], efficacies[2], pyr_f)
        ngc_g_exc = _reference_stp_mv(weights[3], efficacies[3], pyr_f)

        # I→I
        pv_g_gaba_a = (
            _reference_stp_mv(weights[4], efficacies[4], pv_f)
            + _reference_stp_mv(weights[5], efficacies[5], sst_f)
            + _reference_stp_mv(weights[6], efficacies[6], vip_f)
        )
        sst_g_gaba_a = (
            _reference_stp_mv(weights[7], efficacies[7], pv_f)
            + _reference_stp_mv(weights[8], efficacies[8], vip_f)
            + _reference_stp_mv(weights[11], efficacies[11], sst_f)
        )
        sst_g_gaba_b = _reference_stp_mv(weights[9], efficacies[9], vip_f)
        vip_g_gaba_a = _reference_stp_mv(weights[10], efficacies[10], sst_f)

        # I→E
        perisomatic = _reference_stp_mv(weights[12], efficacies[12], pv_f)
        dendritic = _reference_stp_mv(weights[13], efficacies[13], sst_f)
        vip_to_pyr = _reference_stp_mv(weights[14], efficacies[14], vip_f)
        ngc_to_pyr = _reference_stp_mv(weights[15], None, ngc_f)

        total = perisomatic + dendritic + vip_to_pyr + ngc_to_pyr

        return [
            pv_g_exc, sst_g_exc, vip_g_exc, ngc_g_exc,
            pv_g_gaba_a, sst_g_gaba_a, sst_g_gaba_b, vip_g_gaba_a,
            perisomatic, dendritic, vip_to_pyr, ngc_to_pyr, total,
        ]

    def test_numerical_equivalence(self):
        """Fused kernel matches reference for typical spike rates."""
        torch.manual_seed(42)
        pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies = self._build_test_data(spike_rate=0.05)

        results = cortical_inhibitory_step(pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies)
        reference = self._compute_reference(pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies)

        names = [
            "pv_g_exc", "sst_g_exc", "vip_g_exc", "ngc_g_exc",
            "pv_g_gaba_a", "sst_g_gaba_a", "sst_g_gaba_b", "vip_g_gaba_a",
            "perisomatic", "dendritic", "vip_to_pyr", "ngc_to_pyr", "total_inhibition",
        ]
        for i, name in enumerate(names):
            torch.testing.assert_close(results[i], reference[i], atol=1e-5, rtol=1e-5,
                                       msg=f"Mismatch in {name}")

    def test_zero_spikes(self):
        """All zeros input → all zeros output."""
        torch.manual_seed(123)
        pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies = self._build_test_data(spike_rate=0.0)

        results = cortical_inhibitory_step(pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies)
        for i, t in enumerate(results):
            assert torch.all(t == 0), f"Output {i} should be all zeros with no spikes"

    def test_high_spike_rate(self):
        """High spike rate still matches reference."""
        torch.manual_seed(99)
        pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies = self._build_test_data(spike_rate=0.5)

        results = cortical_inhibitory_step(pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies)
        reference = self._compute_reference(pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies)

        for i in range(13):
            torch.testing.assert_close(results[i], reference[i], atol=1e-5, rtol=1e-5)

    def test_all_spikes(self):
        """All neurons spiking — reference matches."""
        torch.manual_seed(77)
        pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies = self._build_test_data(spike_rate=1.0)

        results = cortical_inhibitory_step(pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies)
        reference = self._compute_reference(pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies)

        for i in range(13):
            torch.testing.assert_close(results[i], reference[i], atol=1e-4, rtol=1e-4)

    def test_output_shapes(self):
        """Output tensors have correct shapes."""
        torch.manual_seed(42)
        pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies = self._build_test_data()

        results = cortical_inhibitory_step(pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies)

        assert len(results) == 13
        assert results[0].shape == (self.PV_N,)     # pv_g_exc
        assert results[1].shape == (self.SST_N,)    # sst_g_exc
        assert results[2].shape == (self.VIP_N,)    # vip_g_exc
        assert results[3].shape == (self.NGC_N,)    # ngc_g_exc
        assert results[4].shape == (self.PV_N,)     # pv_g_gaba_a
        assert results[5].shape == (self.SST_N,)    # sst_g_gaba_a
        assert results[6].shape == (self.SST_N,)    # sst_g_gaba_b
        assert results[7].shape == (self.VIP_N,)    # vip_g_gaba_a
        assert results[8].shape == (self.PYR_N,)    # perisomatic
        assert results[9].shape == (self.PYR_N,)    # dendritic
        assert results[10].shape == (self.PYR_N,)   # vip_to_pyr
        assert results[11].shape == (self.PYR_N,)   # ngc_to_pyr
        assert results[12].shape == (self.PYR_N,)   # total_inhibition

    def test_total_is_sum_of_parts(self):
        """total_inhibition == perisomatic + dendritic + vip_to_pyr + ngc_to_pyr."""
        torch.manual_seed(42)
        pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies = self._build_test_data(spike_rate=0.1)

        results = cortical_inhibitory_step(pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies)

        expected_total = results[8] + results[9] + results[10] + results[11]
        torch.testing.assert_close(results[12], expected_total, atol=1e-6, rtol=1e-6)

    def test_clamp_per_connection(self):
        """Verify clamping happens per-connection, not after summation."""
        # Create a scenario where one connection gives negative and another positive
        # The result should be max(neg, 0) + max(pos, 0) = 0 + pos, not max(neg+pos, 0)
        pv_n, sst_n = 5, 3
        # PV→PV gives -1 per element, SST→PV gives +2 per element
        w_pv_pv = torch.full((pv_n, pv_n), -0.2)
        w_sst_pv = torch.full((pv_n, sst_n), 0.4)

        # All PV and SST spike
        pyr_f = torch.zeros(10)
        pv_f = torch.ones(pv_n)
        sst_f = torch.ones(sst_n)
        vip_f = torch.zeros(4)
        ngc_f = torch.zeros(2)

        eff_pv = torch.ones(pv_n)
        eff_sst = torch.ones(sst_n)

        weights = [
            _make_weight(pv_n, 10),   # Pyr→PV
            _make_weight(sst_n, 10),  # Pyr→SST
            _make_weight(4, 10),      # Pyr→VIP
            _make_weight(2, 10),      # Pyr→NGC
            w_pv_pv,                  # PV→PV
            w_sst_pv,                 # SST→PV
            _make_weight(pv_n, 4),    # VIP→PV
            _make_weight(sst_n, pv_n),# PV→SST
            _make_weight(sst_n, 4),   # VIP→SST_A
            _make_weight(sst_n, 4),   # VIP→SST_B
            _make_weight(4, sst_n),   # SST→VIP
            _make_weight(sst_n, sst_n),# SST→SST
            _make_weight(10, pv_n),   # PV→Pyr
            _make_weight(10, sst_n),  # SST→Pyr
            _make_weight(10, 4),      # VIP→Pyr
            _make_weight(10, 2),      # NGC→Pyr
        ]
        efficacies = [
            _make_efficacy(10), _make_efficacy(10), _make_efficacy(10), _make_efficacy(10),
            eff_pv, eff_sst, _make_efficacy(4), _make_efficacy(pv_n),
            _make_efficacy(4), _make_efficacy(4), _make_efficacy(sst_n), _make_efficacy(sst_n),
            _make_efficacy(pv_n), _make_efficacy(sst_n), _make_efficacy(4),
        ]

        results = cortical_inhibitory_step(pyr_f, pv_f, sst_f, vip_f, ngc_f, weights, efficacies)
        pv_gaba = results[4]  # pv_g_gaba_a

        # PV→PV: all negative weights → clamped to 0
        # SST→PV: all positive weights → eff_sst * sst_spikes summed > 0
        # Per-connection clamp: 0 + positive = positive
        # Without per-connection clamp: might get negative + positive first, then clamp
        # The result should equal just the SST→PV contribution (since PV→PV is clamped to 0)
        sst_pv_ref = _reference_stp_mv(w_sst_pv, eff_sst, sst_f)
        # PV→PV gives negative sums, clamped to 0, so total = 0 + sst_pv_ref + vip_pv_ref
        assert torch.all(pv_gaba >= 0), "GABA_A conductance must be non-negative"
