"""Weight Initialization Registry - Biologically-Motivated Initialization Strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import torch

from thalia import GlobalConfig


@dataclass
class ConductanceScaledSpec:
    """Deferred parameters for conductance-scaled weight initialization.

    Instead of hand-tuning ``weight_scale``, specify *what you want to happen*:

    .. code-block:: python

        # thalamus (32 Hz, ~400 neurons) drives L4 pyramidals to V_inf ≈ 1.05,
        # providing 70% of total excitatory drive (recurrent handles the rest)
        ConductanceScaledSpec(
            source_rate_hz=32.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.70,
        )

    See :meth:`WeightInitializer.conductance_scaled` for the full derivation.
    """

    source_rate_hz: float
    """Expected mean firing rate of the presynaptic population (Hz)."""

    target_g_L: float
    """Leak conductance of the postsynaptic neuron (dimensionless normalised units)."""

    target_E_E: float = 3.0
    """Excitatory reversal potential of postsynaptic neuron (default 3.0)."""

    target_E_L: float = 0.0
    """Leak reversal potential of postsynaptic neuron (default 0.0)."""

    target_tau_E_ms: float = 5.0
    """AMPA synaptic time-constant of postsynaptic neuron in ms (default 5.0)."""

    target_v_inf: float = 1.05
    """Desired steady-state membrane potential.

    1.05 = just above threshold (1.0) → reliable but not maximal firing.
    Use 0.90–0.95 for sub-threshold seeding; 1.10–1.20 for strongly-driven
    populations such as relay → PV feedforward inhibition.
    """

    fraction_of_drive: float = 1.0
    """Fraction of total required g_AMPA this source supplies.

    Use < 1.0 when multiple excitatory inputs converge on the target.
    Example: thalamus 70% + recurrent 30% → ``fraction_of_drive=0.70``.
    """

    stp_utilization_factor: float = 1.0
    """Expected fraction of the nominal weight that STP leaves effective at the
    source's biological baseline firing rate.

    The formula assumes undepleted Poisson drive.  Short-term plasticity reduces
    the effective per-spike conductance below the nominal weight—for example,
    thalamo-striatal STP (U=0.25) at 30 Hz depletes to ~7.5% of the nominal
    value.  Setting ``stp_utilization_factor=0.075`` compensates by inflating the
    nominal weight so that the *effective* steady-state conductance equals the
    target.

    **Formula adjustment:**  :math:`w_{mean} = g_{ss} / (n \\cdot c \\cdot r \\cdot \\tau_E \\cdot u_{eff})`

    **Computing utilization** for a Tsodyks-Markram STP preset at baseline rate
    *r* Hz with parameters U, τ_d (ms), τ_f (ms):

    .. math::

        u_{ss} = \\frac{U \\,(1 + r \\tau_f / 1000)}{1 + U \\, r \\tau_f / 1000}, \\quad
        x_{ss} = \\frac{1}{1 + u_{ss} \\, r \\, \\tau_d / 1000}, \\quad
        u_{eff} = u_{ss} \\, x_{ss}

    Common presets at their designed operating rates:

    +----------------------------+--------+-------------+-----------+
    | Preset                     | Rate   | Params      | u_eff     |
    +============================+========+=============+===========+
    | CORTICOSTRIATAL            | 10 Hz  | U=0.40      | ≈ 0.24    |
    +----------------------------+--------+-------------+-----------+
    | CORTICOSTRIATAL            |  5 Hz  | U=0.40      | ≈ 0.32    |
    +----------------------------+--------+-------------+-----------+
    | THALAMO_STRIATAL           | 30 Hz  | U=0.25      | ≈ 0.075   |
    +----------------------------+--------+-------------+-----------+
    | SCHAFFER_COLLATERAL        |  5 Hz  | U=0.50      | ≈ 0.21    |
    +----------------------------+--------+-------------+-----------+
    | SCHAFFER_COLLATERAL        |  3 Hz  | U=0.50      | ≈ 0.28    |
    +----------------------------+--------+-------------+-----------+
    | MOSSY_FIBER                |  5 Hz  | U=0.01      | ≈ 0.03    |
    +----------------------------+--------+-------------+-----------+

    Defaults to 1.0 (no STP correction).
    """

    # Sentinel so equality checks and repr work cleanly; no user-facing meaning.
    _tag: str = field(default="conductance_scaled", init=False, repr=False, compare=False)


class WeightInitializer:
    """
    Centralized weight initialization registry.

    Provides standardized initialization methods used across Thalia.
    All methods return torch.Tensor, not nn.Parameter.
    """

    @staticmethod
    def sparse_gaussian(
        n_input: int,
        n_output: int,
        connectivity: float,
        mean: float,
        std: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse Gaussian connectivity with Gaussian weight distribution.

        Creates biological sparse connectivity with Gaussian distributed weights.

        Args:
            n_input: Number of input neurons
            n_output: Number of output neurons
            connectivity: Connection probability (fraction of connections present, 0-1)
            mean: Mean of Gaussian distribution
            std: Standard deviation of Gaussian distribution
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity
        """
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("Connectivity must be between 0 and 1.")
        if mean < 0.0 or std < 0.0:
            raise ValueError("Mean and std must be non-negative for conductance-based synapses.")

        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < connectivity
        weights = torch.randn(n_output, n_input, device=device, requires_grad=False)
        weights = weights * std + mean
        weights = weights * mask.float() * GlobalConfig.SYNAPTIC_WEIGHT_SCALE
        return weights.abs()  # Ensure positive conductances

    @staticmethod
    def sparse_random(
        n_input: int,
        n_output: int,
        connectivity: float,
        weight_scale: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse random connectivity initialization.

        Creates biological sparse connectivity patterns.
        Each output neuron connects to a random subset of inputs.

        Args:
            n_input: Number of input neurons
            n_output: Number of output neurons
            connectivity: Connection probability (fraction of connections present, 0-1)
            weight_scale: Scale of random weights (CONDUCTANCE units, normalized by g_L)
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity
        """
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("Connectivity must be between 0 and 1.")
        if weight_scale < 0.0:
            raise ValueError("Weight scale must be non-negative for conductance-based synapses.")

        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < connectivity
        weights = torch.rand(n_output, n_input, device=device, requires_grad=False)
        weights = weights * weight_scale
        weights = weights * mask.float() * GlobalConfig.SYNAPTIC_WEIGHT_SCALE
        return weights.abs()  # Ensure positive conductances

    @staticmethod
    def sparse_uniform(
        n_input: int,
        n_output: int,
        connectivity: float,
        w_min: float,
        w_max: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse uniform connectivity with uniform weight distribution.

        Creates biological sparse connectivity with uniformly distributed weights.

        Args:
            n_input: Number of input (pre-synaptic) neurons
            n_output: Number of output (post-synaptic) neurons
            connectivity: Connection probability (fraction of connections present, 0-1)
            w_min: Minimum weight value
            w_max: Maximum weight value
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity
        """
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("Connectivity must be between 0 and 1.")
        if w_min < 0.0 or w_max < 0.0:
            raise ValueError("Weight values must be non-negative for conductance-based synapses.")
        if w_min > w_max:
            raise ValueError("Minimum weight cannot be greater than maximum weight.")

        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < connectivity
        weights = torch.rand(n_output, n_input, device=device, requires_grad=False)
        weights = weights * (w_max - w_min) + w_min
        weights = weights * mask.float() * GlobalConfig.SYNAPTIC_WEIGHT_SCALE
        return weights.abs()  # Ensure positive conductances

    @staticmethod
    def sparse_gaussian_no_autapses(
        n_input: int,
        n_output: int,
        connectivity: float,
        mean: float,
        std: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse Gaussian connectivity with diagonal zeroed (no autapses).

        Identical to :meth:`sparse_gaussian` but guarantees the diagonal is zero.
        Only valid for square weight matrices (``n_input == n_output``).

        Args:
            n_input: Number of input neurons (must equal ``n_output``)
            n_output: Number of output neurons (must equal ``n_input``)
            connectivity: Connection probability (fraction of connections present, 0-1)
            mean: Mean of Gaussian distribution
            std: Standard deviation of Gaussian distribution
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity and zero diagonal

        Raises:
            ValueError: If ``n_input != n_output`` (non-square matrices cannot have autapses)
        """
        if n_input != n_output:
            raise ValueError(
                f"sparse_gaussian_no_autapses requires a square matrix "
                f"(n_input={n_input} != n_output={n_output}). "
                "Use sparse_gaussian for non-square weight matrices."
            )
        weights = WeightInitializer.sparse_gaussian(
            n_input=n_input,
            n_output=n_output,
            connectivity=connectivity,
            mean=mean,
            std=std,
            device=device,
        )
        weights.fill_diagonal_(0.0)  # Eliminate autapses (biologically absent)
        return weights

    @staticmethod
    def sparse_random_no_autapses(
        n_input: int,
        n_output: int,
        connectivity: float,
        weight_scale: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse random connectivity with diagonal zeroed (no autapses).

        Identical to :meth:`sparse_random` but guarantees the diagonal is zero.
        Only valid for square weight matrices (``n_input == n_output``).

        Args:
            n_input: Number of input neurons (must equal ``n_output``)
            n_output: Number of output neurons (must equal ``n_input``)
            connectivity: Connection probability (fraction of connections present, 0-1)
            weight_scale: Scale of random weights (CONDUCTANCE units, normalised by g_L)
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity and zero diagonal

        Raises:
            ValueError: If ``n_input != n_output`` (non-square matrices cannot have autapses)
        """
        if n_input != n_output:
            raise ValueError(
                f"sparse_random_no_autapses requires a square matrix "
                f"(n_input={n_input} != n_output={n_output}). "
                "Use sparse_random for non-square weight matrices."
            )
        weights = WeightInitializer.sparse_random(
            n_input=n_input,
            n_output=n_output,
            connectivity=connectivity,
            weight_scale=weight_scale,
            device=device,
        )
        weights.fill_diagonal_(0.0)  # Eliminate autapses (biologically absent)
        return weights

    @staticmethod
    def sparse_uniform_no_autapses(
        n_input: int,
        n_output: int,
        connectivity: float,
        w_min: float,
        w_max: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse uniform connectivity with diagonal zeroed (no autapses).

        Identical to :meth:`sparse_uniform` but guarantees the diagonal is zero.
        Only valid for square weight matrices (``n_input == n_output``).

        Args:
            n_input: Number of input neurons (must equal ``n_output``)
            n_output: Number of output neurons (must equal ``n_input``)
            connectivity: Connection probability (fraction of connections present, 0-1)
            w_min: Minimum weight value
            w_max: Maximum weight value
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity and zero diagonal

        Raises:
            ValueError: If ``n_input != n_output`` (non-square matrices cannot have autapses)
        """
        if n_input != n_output:
            raise ValueError(
                f"sparse_uniform_no_autapses requires a square matrix "
                f"(n_input={n_input} != n_output={n_output}). "
                "Use sparse_uniform for non-square weight matrices."
            )
        weights = WeightInitializer.sparse_uniform(
            n_input=n_input,
            n_output=n_output,
            connectivity=connectivity,
            w_min=w_min,
            w_max=w_max,
            device=device,
        )
        weights.fill_diagonal_(0.0)  # Eliminate autapses (biologically absent)
        return weights

    @staticmethod
    def conductance_scaled(
        n_input: int,
        n_output: int,
        connectivity: float,
        source_rate_hz: float,
        target_g_L: float,
        target_E_E: float = 3.0,
        target_E_L: float = 0.0,
        target_tau_E_ms: float = 5.0,
        target_v_inf: float = 1.05,
        fraction_of_drive: float = 1.0,
        stp_utilization_factor: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Sparse random weights calibrated so the source drives the target to a desired V_inf.

        Eliminates hand-tuned ``weight_scale`` constants by working backwards from the
        ConductanceLIF steady-state membrane equation.  Instead of guessing a small number
        and re-running diagnostics, you specify *what you want to happen* and the correct
        weight falls out analytically.

        **Physics:**

        For a conductance-based LIF at rest (V = V_inf, ignoring inhibition and noise):

        .. math::
            V_{\\infty} = \\frac{g_{AMPA} \\cdot E_E + g_L \\cdot E_L}{g_L + g_{AMPA}}

        Solving for the required steady-state AMPA conductance from *all* sources:

        .. math::
            g_{ss}^{total} = g_L \\cdot \\frac{V_{\\infty} - E_L}{E_E - V_{\\infty}}

        This source is responsible for ``fraction_of_drive`` of that total:

        .. math::
            g_{ss}^{this} = g_{ss}^{total} \\times fraction\\_of\\_drive

        The steady-state conductance produced by one synapse is
        ``weight × rate_per_ms × tau_E_ms`` (geometric series of decaying AMPA pulses
        at mean rate).  With ``n_input × connectivity`` synapses per postsynaptic neuron:

        .. math::
            w_{mean} = \\frac{g_{ss}^{this}}{n_{input} \\cdot c \\cdot (r_{Hz}/1000) \\cdot \\tau_E}

        Weights are drawn from U[0, 2 × w_mean] so that E[w] = w_mean.

        **Usage example** — thalamus (400 neurons, 32 Hz) driving L4 pyramidals (800
        neurons, g_L=0.05, tau_E=5 ms), thalamus providing 60% of L4's total drive:

        .. code-block:: python

            WeightInitializer.conductance_scaled(
                n_input=400, n_output=800, connectivity=0.7,
                source_rate_hz=32.0,
                target_g_L=0.05, target_tau_E_ms=5.0,
                target_v_inf=1.05, fraction_of_drive=0.6,
            )
            # → weight_mean ≈ 0.00065

        Args:
            n_input: Number of presynaptic neurons.
            n_output: Number of postsynaptic neurons.
            connectivity: Connection probability (0–1).
            source_rate_hz: Expected mean firing rate of the source population (Hz).
            target_g_L: Leak conductance of the target neuron (``g_L`` field of
                ``ConductanceLIFConfig``).
            target_E_E: Excitatory reversal potential of target neuron (default 3.0).
            target_E_L: Leak reversal potential of target neuron (default 0.0).
            target_tau_E_ms: AMPA time constant of target neuron in ms (default 5.0).
            target_v_inf: Desired steady-state membrane potential (default 1.05, just
                above threshold=1.0 for reliable but not maximal firing).
                Use 0.90–0.95 for a sub-threshold seed (needs additional input to fire).
                Use 1.10–1.20 for a strongly-driven population (e.g. relay → PV).
            fraction_of_drive: Fraction of total required conductance this source
                should supply.  Use < 1.0 when multiple inputs converge on the target.
                Example: thalamus 60% + recurrent 40% → fraction_of_drive=0.60 and 0.40.
            stp_utilization_factor: Expected effective fraction of each spike's
                nominal weight after STP at the source's baseline firing rate.
                Defaults to 1.0 (no STP).  See :attr:`ConductanceScaledSpec.stp_utilization_factor`
                for a table of common preset values.
            device: Tensor device.

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity and calibrated weights.

        Raises:
            ValueError: If ``target_v_inf >= target_E_E`` (unreachable by excitation alone).
            ValueError: If ``source_rate_hz <= 0`` or other parameters are out of range.
        """
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("connectivity must be in [0, 1].")
        if target_v_inf >= target_E_E:
            raise ValueError(
                f"target_v_inf={target_v_inf} >= target_E_E={target_E_E}: "
                "impossible to reach this V with excitatory inputs alone."
            )
        if source_rate_hz <= 0.0:
            raise ValueError("source_rate_hz must be positive.")
        if not (0.0 < fraction_of_drive <= 1.0):
            raise ValueError("fraction_of_drive must be in (0, 1].")
        if not (0.0 < stp_utilization_factor <= 1.0):
            raise ValueError("stp_utilization_factor must be in (0, 1].")

        # Required steady-state g_AMPA across all sources combined
        g_ss_total = target_g_L * (target_v_inf - target_E_L) / (target_E_E - target_v_inf)

        # Fraction this source is responsible for
        g_ss_this = g_ss_total * fraction_of_drive

        # At steady state: g_ss = weight_mean × n_inputs × rate_per_ms × tau_E × u_eff
        # Each spike contributes `weight` to the AMPA channel; conductance decays
        # with tau_E; at rate r spikes/ms, the geometric series sums to weight × r × tau_E.
        # With STP, each spike's effective weight is reduced by u_eff (utilization factor),
        # so the nominal weight must be inflated by 1/u_eff to reach the target conductance.
        n_inputs_per_neuron = n_input * connectivity
        rate_per_ms = source_rate_hz / 1000.0
        g_ss_per_weight_unit = n_inputs_per_neuron * rate_per_ms * target_tau_E_ms * stp_utilization_factor

        weight_mean = g_ss_this / g_ss_per_weight_unit

        # U[0, 2*weight_mean] → E[w] = weight_mean
        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < connectivity
        weights = torch.rand(n_output, n_input, device=device, requires_grad=False)
        weights = weights * (2.0 * weight_mean)
        weights = weights * mask.float() * GlobalConfig.SYNAPTIC_WEIGHT_SCALE
        return weights.abs()  # Ensure positive conductances
