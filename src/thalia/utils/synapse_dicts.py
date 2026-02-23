"""SynapseId-keyed container classes.

All four ``nn.Module`` subclasses here share the same ``__getitem__`` /
``__setitem__`` / ``__contains__`` / ``__iter__`` / ``items`` / ``keys`` /
``values`` logic via :class:`_SynapseKeyedMixin`.  The mixin avoids ~250 lines
of near-identical boilerplate that previously lived in two separate files.

Serialisation note
------------------
The underlying ``nn.ParameterDict`` / ``nn.ModuleDict`` is stored in an
attribute called ``_inner``, so ``state_dict`` keys look like
``_inner.some|pipe|delimited|key``.  This is stable across all four classes.
"""

from __future__ import annotations

from typing import Generic, Iterator, Optional, TypeVar

import torch
import torch.nn as nn

from thalia.typing import SynapseId


V = TypeVar("V")


# =============================================================================
# SHARED MIXIN
# =============================================================================


class _SynapseKeyedMixin(Generic[V]):
    """Mixin that provides SynapseId-keyed access to a ``_inner`` container.

    Designed to be mixed with ``nn.Module``.  The concrete subclass is
    responsible for:

    1. Calling ``super().__init__()`` (resolves to ``nn.Module.__init__``).
    2. Setting ``self._inner`` to an ``nn.ParameterDict`` or ``nn.ModuleDict``.

    The mixin never calls ``super()`` itself — all ``nn.Module`` bookkeeping
    happens through Python's MRO once the subclass's ``__init__`` runs.
    """

    # Declared here only for type-checker benefit; set by concrete __init__.
    _inner: nn.Module

    # ------------------------------------------------------------------
    # Core protocol
    # ------------------------------------------------------------------

    def __setitem__(self, key: SynapseId, value: V) -> None:
        self._inner[key.to_key()] = value  # type: ignore[index]

    def __getitem__(self, key: SynapseId) -> V:
        return self._inner[key.to_key()]  # type: ignore[index,return-value]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, SynapseId):
            return False
        return key.to_key() in self._inner  # type: ignore[operator]

    def __len__(self) -> int:
        return len(self._inner)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[SynapseId]:
        return (SynapseId.from_key(k) for k in self._inner)  # type: ignore[call-overload]

    # ------------------------------------------------------------------
    # dict-like helpers
    # ------------------------------------------------------------------

    def items(self) -> Iterator[tuple[SynapseId, V]]:
        """Yield ``(SynapseId, value)`` pairs."""
        return (  # type: ignore[return-value]
            (SynapseId.from_key(k), v) for k, v in self._inner.items()  # type: ignore[union-attr]
        )

    def keys(self) -> Iterator[SynapseId]:
        """Yield :class:`~thalia.typing.SynapseId` keys."""
        return (SynapseId.from_key(k) for k in self._inner)  # type: ignore[call-overload]

    def values(self) -> Iterator[V]:
        """Yield stored values."""
        return self._inner.values()  # type: ignore[return-value,union-attr]


# =============================================================================
# CONCRETE CONTAINERS
# =============================================================================


class SynapseIdParameterDict(_SynapseKeyedMixin[nn.Parameter], nn.Module):
    """``nn.ParameterDict`` wrapper keyed by :class:`~thalia.typing.SynapseId`.

    Use this to store learnable (or fixed) synaptic weight matrices:

    .. code-block:: python

        weights = SynapseIdParameterDict()
        weights[sid] = nn.Parameter(torch.zeros(n_post, n_pre), requires_grad=False)
        w = weights[sid]   # nn.Parameter

    All PyTorch semantics (``.to()``, ``.parameters()``, ``state_dict()``)
    work correctly through the underlying ``nn.ParameterDict``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._inner: nn.ParameterDict = nn.ParameterDict()


class SynapseIdModuleDict(_SynapseKeyedMixin[nn.Module], nn.Module):
    """``nn.ModuleDict`` wrapper keyed by :class:`~thalia.typing.SynapseId`.

    Use this to store sub-modules (STP, learning strategies, etc.):

    .. code-block:: python

        stp_modules = SynapseIdModuleDict()
        stp_modules[sid] = ShortTermPlasticity(config)
        stp = stp_modules[sid]

    All PyTorch semantics (``.to()``, ``.parameters()``, ``state_dict()``)
    work correctly through the underlying ``nn.ModuleDict``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._inner: nn.ModuleDict = nn.ModuleDict()

    def get(self, key: SynapseId, default: Optional[nn.Module] = None) -> Optional[nn.Module]:
        """Return the module for *key*, or *default* if absent."""
        if not isinstance(key, SynapseId) or key.to_key() not in self._inner:
            return default
        return self._inner[key.to_key()]


class SynapseIdBufferDict(nn.Module):
    """Dict-like container for per-synapse non-learnable tensors (e.g. eligibility traces).

    Stores each tensor as an ``nn.Parameter(requires_grad=False)`` inside an
    ``nn.ParameterDict`` so that:

    - ``.to(device)`` / ``.cuda()`` move all tensors automatically.
    - ``state_dict()`` / ``load_state_dict()`` save and restore all tensors.
    - Tensors do **not** receive gradients (biologically-accurate learning only).

    ``__setitem__`` updates existing entries **in-place** (``copy_``) to avoid
    creating new graph leaves and to preserve tensor identity.

    Usage::

        elig = SynapseIdBufferDict()
        elig[sid] = torch.zeros(n_post, n_pre, device=device)
        elig[sid] = elig[sid] * decay + update   # in-place copy on assignment
        for sid, tensor in elig.items():
            ...
    """

    def __init__(self) -> None:
        super().__init__()
        self._inner: nn.ParameterDict = nn.ParameterDict()

    # Override __setitem__ for in-place copy semantics (different from the
    # other containers which simply assign).
    def __setitem__(self, key: SynapseId, value: torch.Tensor) -> None:
        """Assign a tensor.  Existing entries are updated in-place (``copy_``)."""
        encoded = key.to_key()
        existing = self._inner.get(encoded)
        if existing is not None:
            existing.data.copy_(value)
        else:
            self._inner[encoded] = nn.Parameter(
                value.detach() if value.requires_grad else value,
                requires_grad=False,
            )

    def __getitem__(self, key: SynapseId) -> torch.Tensor:
        return self._inner[key.to_key()]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, SynapseId):
            return False
        return key.to_key() in self._inner

    def __len__(self) -> int:
        return len(self._inner)

    def __iter__(self) -> Iterator[SynapseId]:
        return (SynapseId.from_key(k) for k in self._inner)

    def get(self, key: SynapseId, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Return tensor for *key*, or *default* if absent."""
        encoded = key.to_key()
        if encoded in self._inner:
            return self._inner[encoded]
        return default

    def items(self) -> Iterator[tuple[SynapseId, torch.Tensor]]:
        """Yield ``(SynapseId, torch.Tensor)`` pairs."""
        return ((SynapseId.from_key(k), v) for k, v in self._inner.items())

    def keys(self) -> Iterator[SynapseId]:
        """Yield :class:`~thalia.typing.SynapseId` keys."""
        return (SynapseId.from_key(k) for k in self._inner)

    def values(self) -> Iterator[torch.Tensor]:
        """Yield tensors."""
        return self._inner.values()  # type: ignore[return-value]
