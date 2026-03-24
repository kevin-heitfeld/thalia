"""Adaptive signal normalization for neuromodulator nuclei."""

from __future__ import annotations


class AdaptiveNormalization:
    """Running-average adaptive normalization for scalar signals.

    Tracks a running mean of the signal magnitude and normalizes new values
    relative to that baseline.  Used by neuromodulator nuclei (LC, DRN, NB,
    VTA) to keep drive/error/uncertainty signals in a stable dynamic range
    regardless of absolute magnitude.

    Two normalization modes are supported:

    * **Ratio** (default): ``signal / (avg + epsilon)``
    * **Centered** (``center=True``): ``(signal - avg) / (|avg| + epsilon)``

    Two alpha schedules are supported:

    * **Warmup** (default): ``alpha = 1 / min(count, window)`` — fast initial
      adaptation that floors at ``1/window``.
    * **No warmup** (``warmup=False``): ``alpha = min(1/window, 1/count)`` —
      capped from the start and continues decreasing past ``1/window``.
    """

    __slots__ = (
        "_window",
        "_epsilon",
        "_avg",
        "_count",
        "_clip_lo",
        "_clip_hi",
        "_track_abs",
        "_center",
        "_warmup",
    )

    def __init__(
        self,
        *,
        window: int = 100,
        epsilon: float = 0.1,
        initial_avg: float = 0.5,
        clip_range: tuple[float, float] | None = None,
        track_abs: bool = False,
        center: bool = False,
        warmup: bool = True,
    ) -> None:
        self._window = window
        self._epsilon = epsilon
        self._avg = initial_avg
        self._count = 0
        self._clip_lo = clip_range[0] if clip_range is not None else None
        self._clip_hi = clip_range[1] if clip_range is not None else None
        self._track_abs = track_abs
        self._center = center
        self._warmup = warmup

    def __call__(self, signal: float) -> float:
        self._count += 1

        if self._warmup:
            alpha = 1.0 / min(self._count, self._window)
        else:
            alpha = min(1.0 / self._window, 1.0 / self._count)

        tracked = abs(signal) if self._track_abs else signal
        self._avg = (1.0 - alpha) * self._avg + alpha * tracked

        if self._center:
            normalized = (signal - self._avg) / (abs(self._avg) + self._epsilon)
        else:
            normalized = signal / (self._avg + self._epsilon)

        if self._clip_lo is not None:
            normalized = max(self._clip_lo, min(self._clip_hi, normalized))  # type: ignore[type-var]

        return float(normalized)
