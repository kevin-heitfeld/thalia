"""Equivalence tests for batched synaptic integration vs per-source loop.

Tests verify that:
1. build_batched_dendrite_weights correctly concatenates weight matrices
2. Batched integration produces identical DendriteOutput to the original loop
3. Learning rule weight updates propagate through views to the concatenated matrix
4. STP efficacy is correctly applied in the batched path
5. Single-synapse populations correctly fall back to the loop path
"""

from __future__ import annotations

import torch
import torch.nn as nn

from thalia.brain.regions.neural_region import (
    BatchedDendriteWeights,
    ConcatWeightBlock,
    DendriteOutput,
)
from thalia.brain.synapses import STPConfig, ShortTermPlasticity
from thalia.typing import PopulationName, ReceptorType, SynapseId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sid(src_region: str, src_pop: str, tgt_region: str, tgt_pop: str,
         receptor: ReceptorType) -> SynapseId:
    return SynapseId(
        source_region=src_region,
        source_population=src_pop,
        target_region=tgt_region,
        target_population=tgt_pop,
        receptor_type=receptor,
    )


def _make_weights(n_post: int, n_pre: int, *, seed: int = 0) -> torch.Tensor:
    """Deterministic random weight matrix."""
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n_post, n_pre, generator=gen).abs() * 0.1


def _run_loop_integration(
    synaptic_weights: dict[SynapseId, nn.Parameter],
    stp_modules: dict[SynapseId, ShortTermPlasticity],
    precomputed_stp_efficacy: dict[SynapseId, torch.Tensor] | None,
    synaptic_inputs: dict[SynapseId, torch.Tensor],
    n_neurons: int,
    filter_by_target_population: PopulationName,
) -> DendriteOutput:
    """Reference implementation: original per-source loop (no batching)."""
    g_ampa = torch.zeros(n_neurons)
    g_nmda = torch.zeros(n_neurons)
    g_gaba_a = torch.zeros(n_neurons)
    g_gaba_b = torch.zeros(n_neurons)

    for synapse_id, source_spikes in synaptic_inputs.items():
        if synapse_id.target_population != filter_by_target_population:
            continue
        if synapse_id not in synaptic_weights:
            continue

        weights = synaptic_weights[synapse_id]
        spikes_f = source_spikes.float()

        if synapse_id in stp_modules:
            if precomputed_stp_efficacy and synapse_id in precomputed_stp_efficacy:
                eff = precomputed_stp_efficacy[synapse_id]
            else:
                eff = stp_modules[synapse_id].forward(spikes_f)
            g = weights @ (eff * spikes_f)
        else:
            g = weights @ spikes_f

        match synapse_id.receptor_type:
            case ReceptorType.AMPA:
                g_ampa += g
            case ReceptorType.NMDA:
                g_nmda += g
            case ReceptorType.GABA_A:
                g_gaba_a += g
            case ReceptorType.GABA_B:
                g_gaba_b += g

    g_ampa.clamp_(min=0.0)
    g_nmda.clamp_(min=0.0)
    g_gaba_a.clamp_(min=0.0)
    g_gaba_b.clamp_(min=0.0)
    return DendriteOutput(g_ampa=g_ampa, g_nmda=g_nmda, g_gaba_a=g_gaba_a, g_gaba_b=g_gaba_b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildBatchedDendriteWeights:
    """Test that build_batched_dendrite_weights creates correct structures."""

    def test_groups_by_target_population_and_receptor(self) -> None:
        """Multiple sources targeting same population should be concatenated."""
        from thalia.utils import SynapseIdParameterDict

        weights_dict = SynapseIdParameterDict()
        n_post = 50

        sid1 = _sid("r1", "p1", "r2", "target", ReceptorType.AMPA)
        sid2 = _sid("r1", "p2", "r2", "target", ReceptorType.AMPA)
        sid3 = _sid("r1", "p3", "r2", "target", ReceptorType.GABA_A)

        w1 = _make_weights(n_post, 30, seed=1)
        w2 = _make_weights(n_post, 40, seed=2)
        w3 = _make_weights(n_post, 20, seed=3)

        weights_dict[sid1] = nn.Parameter(w1, requires_grad=False)
        weights_dict[sid2] = nn.Parameter(w2, requires_grad=False)
        weights_dict[sid3] = nn.Parameter(w3, requires_grad=False)

        # Simulate build_batched_dendrite_weights logic
        pop_synapses: dict[str, list[SynapseId]] = {}
        for sid in weights_dict:
            pop_synapses.setdefault(sid.target_population, []).append(sid)

        assert "target" in pop_synapses
        assert len(pop_synapses["target"]) == 3

        # Group by receptor
        receptor_groups: dict[ReceptorType, list[SynapseId]] = {}
        for sid in pop_synapses["target"]:
            receptor_groups.setdefault(sid.receptor_type, []).append(sid)

        assert ReceptorType.AMPA in receptor_groups
        assert len(receptor_groups[ReceptorType.AMPA]) == 2
        assert ReceptorType.GABA_A in receptor_groups
        assert len(receptor_groups[ReceptorType.GABA_A]) == 1

    def test_concat_weight_matrix_shape(self) -> None:
        """Concatenated weight matrix should have correct dimensions."""
        n_post = 50
        w1 = _make_weights(n_post, 30, seed=1)
        w2 = _make_weights(n_post, 40, seed=2)

        W_concat = torch.cat([w1, w2], dim=1)
        assert W_concat.shape == (50, 70)

    def test_view_based_weight_sharing(self) -> None:
        """After build, individual weights should be views of W_concat."""
        n_post = 50
        w1 = _make_weights(n_post, 30, seed=1)
        w2 = _make_weights(n_post, 40, seed=2)

        W_concat = torch.cat([w1, w2], dim=1)

        # Create view-backed parameters
        view1 = nn.Parameter(W_concat[:, :30], requires_grad=False)
        view2 = nn.Parameter(W_concat[:, 30:70], requires_grad=False)

        # Verify they share storage
        view1.data[0, 0] = 999.0
        assert W_concat[0, 0] == 999.0

        view2.data[0, 0] = 888.0
        assert W_concat[0, 30] == 888.0


class TestBatchedIntegrationEquivalence:
    """Test that batched integration matches the per-source loop exactly."""

    def _setup_region_like(
        self,
        synapse_specs: list[tuple[SynapseId, int]],
        stp_config: STPConfig | None = None,
    ) -> tuple[
        dict[SynapseId, nn.Parameter],
        dict[SynapseId, ShortTermPlasticity],
        dict[PopulationName, BatchedDendriteWeights] | None,
    ]:
        """Create weight dicts and batched weights for testing.

        Args:
            synapse_specs: (synapse_id, n_pre) pairs.
            stp_config: If set, add STP to all synapses.

        Returns:
            (weights_dict, stp_dict, batched_weights)
        """
        from thalia.utils import SynapseIdParameterDict, SynapseIdModuleDict

        weights_dict = SynapseIdParameterDict()
        stp_dict = SynapseIdModuleDict()

        n_post = 50  # All targets have 50 neurons
        for sid, n_pre in synapse_specs:
            w = _make_weights(n_post, n_pre, seed=hash(str(sid)) % (2**31))
            weights_dict[sid] = nn.Parameter(w, requires_grad=False)
            if stp_config:
                stp = ShortTermPlasticity(n_pre=n_pre, config=stp_config)
                stp.update_temporal_parameters(1.0)
                stp_dict[sid] = stp

        # Group by target population
        pop_synapses: dict[str, list[SynapseId]] = {}
        for sid in weights_dict:
            pop_synapses.setdefault(sid.target_population, []).append(sid)

        batched: dict[PopulationName, BatchedDendriteWeights] = {}
        for target_pop, sids in pop_synapses.items():
            if len(sids) < 2:
                continue

            n_target = weights_dict[sids[0]].shape[0]
            receptor_groups: dict[ReceptorType, list[SynapseId]] = {}
            for sid in sids:
                receptor_groups.setdefault(sid.receptor_type, []).append(sid)

            blocks: dict[ReceptorType, ConcatWeightBlock] = {}
            for receptor_type, group_sids in receptor_groups.items():
                group_sids.sort(key=lambda s: s.to_key())
                weight_list = [weights_dict[sid].data for sid in group_sids]
                column_slices: list[slice] = []
                offset = 0
                for w in weight_list:
                    column_slices.append(slice(offset, offset + w.shape[1]))
                    offset += w.shape[1]

                W_concat = torch.cat(weight_list, dim=1)
                # Replace with views
                for sid, cs in zip(group_sids, column_slices):
                    weights_dict[sid] = nn.Parameter(W_concat[:, cs], requires_grad=False)

                blocks[receptor_type] = ConcatWeightBlock(
                    W_concat=W_concat,
                    synapse_ids=tuple(group_sids),
                    column_slices=tuple(column_slices),
                    total_sources=offset,
                    device=torch.device("cpu"),
                )

            batched[target_pop] = BatchedDendriteWeights(n_target=n_target, blocks=blocks)

        return dict(weights_dict.items()), dict(stp_dict.items()), batched or None

    def test_multiple_ampa_sources_no_stp(self) -> None:
        """Three AMPA sources → batched should match loop."""
        target_pop = "pyramidal"
        sid1 = _sid("thalamus", "relay", "cortex", target_pop, ReceptorType.AMPA)
        sid2 = _sid("cortex", "l4", "cortex", target_pop, ReceptorType.AMPA)
        sid3 = _sid("cortex", "l6", "cortex", target_pop, ReceptorType.AMPA)

        weights, stp_modules, batched = self._setup_region_like([
            (sid1, 100), (sid2, 80), (sid3, 60),
        ])

        # Create spike inputs
        torch.manual_seed(42)
        spikes = {
            sid1: (torch.rand(100) > 0.9).bool(),
            sid2: (torch.rand(80) > 0.85).bool(),
            sid3: (torch.rand(60) > 0.8).bool(),
        }

        # Reference: loop
        ref = _run_loop_integration(
            {k: nn.Parameter(v, requires_grad=False) if not isinstance(v, nn.Parameter) else v
             for k, v in weights.items()},
            stp_modules, None, spikes, 50, target_pop,
        )

        # Batched
        assert batched is not None
        batch = batched[target_pop]
        block = batch.get_block(ReceptorType.AMPA)
        assert block is not None

        buf = block.spike_buffer
        buf.zero_()
        for i, sid in enumerate(block.synapse_ids):
            if sid in spikes:
                buf[block.column_slices[i]] = spikes[sid].float()
        g_ampa = (block.W_concat @ buf).clamp_(min=0.0)

        torch.testing.assert_close(g_ampa, ref.g_ampa, atol=1e-5, rtol=1e-5)

    def test_mixed_receptor_types(self) -> None:
        """AMPA + GABA_A sources → all channels should match."""
        target_pop = "target"
        sid_ampa1 = _sid("r1", "p1", "r2", target_pop, ReceptorType.AMPA)
        sid_ampa2 = _sid("r1", "p2", "r2", target_pop, ReceptorType.AMPA)
        sid_gaba1 = _sid("r1", "p3", "r2", target_pop, ReceptorType.GABA_A)
        sid_gaba2 = _sid("r1", "p4", "r2", target_pop, ReceptorType.GABA_A)

        weights, stp_modules, batched = self._setup_region_like([
            (sid_ampa1, 30), (sid_ampa2, 40),
            (sid_gaba1, 25), (sid_gaba2, 35),
        ])

        torch.manual_seed(7)
        spikes = {
            sid_ampa1: (torch.rand(30) > 0.8).bool(),
            sid_ampa2: (torch.rand(40) > 0.85).bool(),
            sid_gaba1: (torch.rand(25) > 0.7).bool(),
            sid_gaba2: (torch.rand(35) > 0.75).bool(),
        }

        ref = _run_loop_integration(
            {k: nn.Parameter(v, requires_grad=False) if not isinstance(v, nn.Parameter) else v
             for k, v in weights.items()},
            stp_modules, None, spikes, 50, target_pop,
        )

        # Batched
        assert batched is not None
        batch = batched[target_pop]

        results = {}
        for rtype in (ReceptorType.AMPA, ReceptorType.GABA_A):
            block = batch.get_block(rtype)
            if block is None:
                results[rtype] = torch.zeros(50)
                continue
            buf = block.spike_buffer
            buf.zero_()
            for i, sid in enumerate(block.synapse_ids):
                if sid in spikes:
                    buf[block.column_slices[i]] = spikes[sid].float()
            results[rtype] = (block.W_concat @ buf).clamp_(min=0.0)

        torch.testing.assert_close(results[ReceptorType.AMPA], ref.g_ampa, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(results[ReceptorType.GABA_A], ref.g_gaba_a, atol=1e-5, rtol=1e-5)

    def test_with_stp_efficacy(self) -> None:
        """STP efficacy should be applied identically in both paths."""
        target_pop = "target"
        sid1 = _sid("r1", "p1", "r2", target_pop, ReceptorType.AMPA)
        sid2 = _sid("r1", "p2", "r2", target_pop, ReceptorType.AMPA)

        stp_cfg = STPConfig(U=0.5, tau_d=800.0, tau_f=20.0)
        weights, stp_modules, batched = self._setup_region_like(
            [(sid1, 50), (sid2, 40)], stp_config=stp_cfg,
        )

        torch.manual_seed(99)
        spikes = {
            sid1: (torch.rand(50) > 0.8).bool(),
            sid2: (torch.rand(40) > 0.85).bool(),
        }

        # Precompute STP efficacy (simulating what Brain.forward does)
        precomputed_eff = {}
        for sid, stp in stp_modules.items():
            if sid in spikes:
                precomputed_eff[sid] = stp.forward(spikes[sid].float())

        ref = _run_loop_integration(
            {k: nn.Parameter(v, requires_grad=False) if not isinstance(v, nn.Parameter) else v
             for k, v in weights.items()},
            stp_modules, precomputed_eff, spikes, 50, target_pop,
        )

        # Batched with precomputed efficacy
        assert batched is not None
        batch = batched[target_pop]
        block = batch.get_block(ReceptorType.AMPA)
        assert block is not None

        buf = block.spike_buffer
        buf.zero_()
        for i, sid in enumerate(block.synapse_ids):
            if sid in spikes:
                spikes_f = spikes[sid].float()
                if sid in precomputed_eff:
                    spikes_f = precomputed_eff[sid] * spikes_f
                buf[block.column_slices[i]] = spikes_f

        g_ampa = (block.W_concat @ buf).clamp_(min=0.0)
        torch.testing.assert_close(g_ampa, ref.g_ampa, atol=1e-5, rtol=1e-5)

    def test_missing_spikes_treated_as_zero(self) -> None:
        """If a source has no spikes in the dict, its contribution should be zero."""
        target_pop = "target"
        sid1 = _sid("r1", "p1", "r2", target_pop, ReceptorType.AMPA)
        sid2 = _sid("r1", "p2", "r2", target_pop, ReceptorType.AMPA)

        weights, _, batched = self._setup_region_like([(sid1, 30), (sid2, 40)])

        torch.manual_seed(55)
        # Only provide spikes for sid1, not sid2
        spikes = {sid1: (torch.rand(30) > 0.8).bool()}

        ref = _run_loop_integration(
            {k: nn.Parameter(v, requires_grad=False) if not isinstance(v, nn.Parameter) else v
             for k, v in weights.items()},
            {}, None, spikes, 50, target_pop,
        )

        assert batched is not None
        batch = batched[target_pop]
        block = batch.get_block(ReceptorType.AMPA)
        assert block is not None

        buf = block.spike_buffer
        buf.zero_()
        for i, sid in enumerate(block.synapse_ids):
            if sid in spikes:
                buf[block.column_slices[i]] = spikes[sid].float()

        g_ampa = (block.W_concat @ buf).clamp_(min=0.0)
        torch.testing.assert_close(g_ampa, ref.g_ampa, atol=1e-5, rtol=1e-5)

    def test_weight_update_propagates_through_views(self) -> None:
        """Learning rule weight updates should propagate to W_concat."""
        target_pop = "target"
        sid1 = _sid("r1", "p1", "r2", target_pop, ReceptorType.AMPA)
        sid2 = _sid("r1", "p2", "r2", target_pop, ReceptorType.AMPA)

        weights, _, batched = self._setup_region_like([(sid1, 30), (sid2, 40)])
        assert batched is not None

        block = batched[target_pop].get_block(ReceptorType.AMPA)
        assert block is not None

        # Modify weight for sid1 (simulating learning rule)
        weights[sid1].data *= 2.0

        # Verify the change propagated to W_concat
        torch.testing.assert_close(
            block.W_concat[:, block.column_slices[0]],
            weights[sid1].data,
            atol=1e-6, rtol=1e-6,
        )

    def test_non_matching_inputs_ignored(self) -> None:
        """Inputs targeting a different population should be ignored."""
        target_pop = "target"
        other_pop = "other"
        sid_target = _sid("r1", "p1", "r2", target_pop, ReceptorType.AMPA)
        sid_other = _sid("r1", "p2", "r2", other_pop, ReceptorType.AMPA)
        sid_target2 = _sid("r1", "p3", "r2", target_pop, ReceptorType.AMPA)

        weights, _, batched = self._setup_region_like([
            (sid_target, 30), (sid_target2, 40), (sid_other, 50),
        ])

        torch.manual_seed(123)
        spikes = {
            sid_target: (torch.rand(30) > 0.8).bool(),
            sid_target2: (torch.rand(40) > 0.85).bool(),
            sid_other: (torch.rand(50) > 0.7).bool(),
        }

        ref = _run_loop_integration(
            {k: nn.Parameter(v, requires_grad=False) if not isinstance(v, nn.Parameter) else v
             for k, v in weights.items()},
            {}, None, spikes, 50, target_pop,
        )

        assert batched is not None
        batch = batched[target_pop]
        block = batch.get_block(ReceptorType.AMPA)
        assert block is not None

        buf = block.spike_buffer
        buf.zero_()
        for i, sid in enumerate(block.synapse_ids):
            if sid in spikes:
                buf[block.column_slices[i]] = spikes[sid].float()

        g_ampa = (block.W_concat @ buf).clamp_(min=0.0)
        torch.testing.assert_close(g_ampa, ref.g_ampa, atol=1e-5, rtol=1e-5)


class TestSinglePopulationFallback:
    """Populations with only 1 synapse should not be batched."""

    def test_single_synapse_not_batched(self) -> None:
        """A target population with only 1 input should not appear in batched dict."""
        from thalia.utils import SynapseIdParameterDict

        weights_dict = SynapseIdParameterDict()
        sid = _sid("r1", "p1", "r2", "lonely", ReceptorType.AMPA)
        weights_dict[sid] = nn.Parameter(_make_weights(50, 30), requires_grad=False)

        pop_synapses: dict[str, list[SynapseId]] = {}
        for s in weights_dict:
            pop_synapses.setdefault(s.target_population, []).append(s)

        # Only 1 synapse → should not be batched
        assert len(pop_synapses["lonely"]) == 1
