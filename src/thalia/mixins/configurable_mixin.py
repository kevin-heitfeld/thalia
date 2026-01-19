"""
Configurable Component Mixin for Thalia.

Provides standard factory method pattern for instantiating components
from the unified configuration system.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from thalia.config import ThaliaConfig


class ConfigurableMixin:
    """Mixin for components that can be created from ThaliaConfig.

    Provides standard factory method pattern for instantiating components
    from the unified configuration system, eliminating boilerplate.

    Subclasses must define CONFIG_CONVERTER_METHOD which specifies the
    ThaliaConfig method to call to get the component-specific config.
    """

    CONFIG_CONVERTER_METHOD: Optional[str] = None

    @classmethod
    def from_thalia_config(cls, config: ThaliaConfig, **kwargs):
        """Create instance from unified ThaliaConfig.

        Automatically extracts the appropriate component config by calling
        the method specified in CONFIG_CONVERTER_METHOD.

        Args:
            config: ThaliaConfig with all settings
            **kwargs: Additional arguments passed to constructor

        Returns:
            Component instance

        Raises:
            NotImplementedError: If CONFIG_CONVERTER_METHOD not defined
        """
        if cls.CONFIG_CONVERTER_METHOD is None:
            raise NotImplementedError(
                f"{cls.__name__} must define CONFIG_CONVERTER_METHOD "
                f"(e.g., 'to_sequence_memory_config')"
            )

        # Get the converter method from ThaliaConfig
        converter = getattr(config, cls.CONFIG_CONVERTER_METHOD, None)
        if converter is None:
            raise AttributeError(f"ThaliaConfig has no method '{cls.CONFIG_CONVERTER_METHOD}'")

        # Convert and instantiate
        component_config = converter()
        return cls(component_config, **kwargs)  # type: ignore[call-arg]


__all__ = ["ConfigurableMixin"]
