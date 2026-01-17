"""
Device Management Mixin for Thalia Components.

Provides a consistent interface for setting and using devices across
all components, reducing boilerplate code.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import Union

import torch


class DeviceMixin:
    """Mixin for standardized device management.

    Provides a consistent interface for setting and using devices across
    all components, reducing boilerplate code.

    Usage:
        class MyComponent(DeviceMixin, nn.Module):
            def __init__(self, device: str = "cpu"):
                super().__init__()
                self.init_device(device)

            def forward(self, x):
                x = self.to_device(x)
                # ... rest of forward pass
    """

    def init_device(
        self,
        device: Union[str, torch.device],
    ) -> None:
        """Initialize device from string or torch.device.

        Args:
            device: Device specification ('cpu', 'cuda', 'cuda:0', etc.)
        """
        if isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        if not hasattr(self, "_device"):
            self._device = torch.device("cpu")
        return self._device

    @device.setter
    def device(self, value: Union[str, torch.device]) -> None:
        """Set the device."""
        self.init_device(value)

    def to_device(
        self,
        tensor: torch.Tensor,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Move tensor to this component's device.

        Args:
            tensor: Tensor to move
            non_blocking: Whether to use non-blocking transfer

        Returns:
            Tensor on the correct device
        """
        if tensor.device != self.device:
            return tensor.to(self.device, non_blocking=non_blocking)
        return tensor

    def ensure_device(
        self,
        *tensors: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Ensure multiple tensors are on the correct device.

        Args:
            *tensors: Tensors to move

        Returns:
            Single tensor or tuple of tensors on correct device
        """
        result = tuple(self.to_device(t) for t in tensors)
        if len(result) == 1:
            return result[0]
        return result

    def get_device_type(self) -> str:
        """Get device type string ('cpu', 'cuda', etc.)."""
        return self.device.type

    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self.device.type == "cuda"


__all__ = ["DeviceMixin"]
