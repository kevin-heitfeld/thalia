"""
Configuration validation for THALIA.

This module provides validation functions and declarative validation patterns
to catch configuration errors before brain initialization.

Validation Features:
- Declarative validation rules via ValidatedConfig mixin
- Predefined validators (positive, finite, range, etc.)
- Cross-region size compatibility checking
- Device consistency validation
- Biological plausibility constraints

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Any, Tuple, Callable, Dict
import math

if TYPE_CHECKING:
    from .thalia_config import ThaliaConfig
    from .brain_config import BrainConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""


# =============================================================================
# DECLARATIVE VALIDATION FRAMEWORK
# =============================================================================


class ValidatorRegistry:
    """Registry of predefined validation rules.

    Usage:
        validator = ValidatorRegistry.get_validator('positive')
        validator(0.5, 'learning_rate')  # Passes
        validator(-0.1, 'learning_rate')  # Raises ConfigValidationError
    """

    _validators: Dict[str, Callable[[Any, str], None]] = {}

    @classmethod
    def register(cls, name: str, validator: Callable[[Any, str], None]) -> None:
        """Register a validation function."""
        cls._validators[name] = validator

    @classmethod
    def get_validator(cls, rule: str) -> Callable[[Any, str], None]:
        """Get validator by name or parse compound rule."""
        # Handle range rules: range(min, max)
        if rule.startswith('range('):
            return cls._parse_range_rule(rule)

        # Handle simple named rules
        if rule in cls._validators:
            return cls._validators[rule]

        raise ValueError(f"Unknown validation rule: {rule}")

    @classmethod
    def _parse_range_rule(cls, rule: str) -> Callable[[Any, str], None]:
        """Parse range(min, max) rules."""
        # Extract min and max from "range(0.0, 1.0)"
        inner = rule[6:-1]  # Remove "range(" and ")"
        parts = [p.strip() for p in inner.split(',')]

        if len(parts) != 2:
            raise ValueError(f"Invalid range rule format: {rule}")

        min_val = float(parts[0])
        max_val = float(parts[1])

        def range_validator(value: Any, name: str) -> None:
            if not isinstance(value, (int, float)):
                raise ConfigValidationError(f"{name} must be numeric, got {type(value)}")
            if not (min_val <= value <= max_val):
                raise ConfigValidationError(
                    f"{name}={value} outside valid range [{min_val}, {max_val}]"
                )

        return range_validator


# Register predefined validators
def _register_builtin_validators() -> None:
    """Register standard validation rules."""

    def positive(value: Any, name: str) -> None:
        """Value must be > 0."""
        if not isinstance(value, (int, float)):
            raise ConfigValidationError(f"{name} must be numeric, got {type(value)}")
        if value <= 0:
            raise ConfigValidationError(f"{name}={value} must be positive")

    def non_negative(value: Any, name: str) -> None:
        """Value must be >= 0."""
        if not isinstance(value, (int, float)):
            raise ConfigValidationError(f"{name} must be numeric, got {type(value)}")
        if value < 0:
            raise ConfigValidationError(f"{name}={value} must be non-negative")

    def finite(value: Any, name: str) -> None:
        """Value must be finite (not inf or nan)."""
        if not isinstance(value, (int, float)):
            raise ConfigValidationError(f"{name} must be numeric, got {type(value)}")
        if not math.isfinite(value):
            raise ConfigValidationError(f"{name}={value} must be finite (not inf/nan)")

    def positive_integer(value: Any, name: str) -> None:
        """Value must be a positive integer."""
        if not isinstance(value, int):
            raise ConfigValidationError(f"{name} must be integer, got {type(value)}")
        if value <= 0:
            raise ConfigValidationError(f"{name}={value} must be positive integer")

    def probability(value: Any, name: str) -> None:
        """Value must be in [0, 1]."""
        if not isinstance(value, (int, float)):
            raise ConfigValidationError(f"{name} must be numeric, got {type(value)}")
        if not (0.0 <= value <= 1.0):
            raise ConfigValidationError(f"{name}={value} must be probability in [0, 1]")

    def non_empty_string(value: Any, name: str) -> None:
        """Value must be a non-empty string."""
        if not isinstance(value, str):
            raise ConfigValidationError(f"{name} must be string, got {type(value)}")
        if not value.strip():
            raise ConfigValidationError(f"{name} must be non-empty string")

    ValidatorRegistry.register('positive', positive)
    ValidatorRegistry.register('non_negative', non_negative)
    ValidatorRegistry.register('finite', finite)
    ValidatorRegistry.register('positive_integer', positive_integer)
    ValidatorRegistry.register('probability', probability)
    ValidatorRegistry.register('non_empty_string', non_empty_string)


_register_builtin_validators()


class ValidatedConfig:
    """Mixin for declarative config validation.

    Usage:
        @dataclass
        class MyConfig(BaseConfig, ValidatedConfig):
            learning_rate: float = 0.001
            n_neurons: int = 100
            dropout: float = 0.1

            _validation_rules = {
                'learning_rate': ('positive', 'finite'),
                'n_neurons': ('positive_integer', 'range(1, 10000)'),
                'dropout': ('probability',),
            }

    The validation runs automatically in __post_init__ if defined.
    """

    _validation_rules: Dict[str, Tuple[str, ...]] = {}

    def validate_config(self) -> None:
        """Validate configuration based on _validation_rules.

        Should be called from __post_init__() or manually.

        Raises:
            ConfigValidationError: If any validation fails
        """
        if not hasattr(self, '_validation_rules'):
            return  # No rules defined

        errors: List[str] = []

        for field_name, rules in self._validation_rules.items():
            # Get field value
            if not hasattr(self, field_name):
                errors.append(f"Validation rule for non-existent field: {field_name}")
                continue

            value = getattr(self, field_name)

            # Apply each rule
            for rule in rules:
                try:
                    validator = ValidatorRegistry.get_validator(rule)
                    validator(value, field_name)
                except ConfigValidationError as e:
                    errors.append(str(e))
                except Exception as e:
                    errors.append(f"Validation error for {field_name}: {e}")

        if errors:
            error_msg = (
                f"{self.__class__.__name__} validation failed:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )
            raise ConfigValidationError(error_msg)


# =============================================================================
# LEGACY VALIDATION FUNCTIONS (for ThaliaConfig)
# =============================================================================


def validate_thalia_config(config: "ThaliaConfig") -> None:
    """Validate complete ThaliaConfig before brain creation.

    Checks for common configuration errors that would cause runtime failures:
    - PFC size mismatch with striatum goal conditioning
    - Pathway source/target size mismatches
    - Device inconsistencies
    - Invalid dimension specifications

    Args:
        config: ThaliaConfig to validate

    Raises:
        ConfigValidationError: If validation fails with detailed error message

    Example:
        >>> from thalia.config import ThaliaConfig, validate_thalia_config
        >>> config = ThaliaConfig(...)
        >>> validate_thalia_config(config)  # Raises if invalid
        >>> brain = EventDrivenBrain.from_thalia_config(config)  # Safe to create
    """
    errors: List[str] = []

    # Validate brain configuration
    brain_errors = validate_brain_config(config.brain)
    errors.extend(brain_errors)

    # Validate global consistency
    global_errors = validate_global_consistency(config)
    errors.extend(global_errors)

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        raise ConfigValidationError(error_msg)


def validate_brain_config(brain_config: "BrainConfig") -> List[str]:
    """Validate BrainConfig for internal consistency.

    Args:
        brain_config: BrainConfig to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors: List[str] = []
    sizes = brain_config.sizes

    # =========================================================================
    # CRITICAL: PFC size must match striatum pfc_size for goal conditioning
    # =========================================================================
    if brain_config.use_goal_conditioning:
        pfc_output = sizes.pfc_size
        striatum_pfc_size = brain_config.striatum.pfc_size

        if striatum_pfc_size != pfc_output:
            errors.append(
                f"PFC size mismatch: striatum.pfc_size={striatum_pfc_size} "
                f"but PFC outputs {pfc_output} dimensions. "
                f"Set brain_config.striatum.pfc_size = {pfc_output} or adjust sizes.pfc_size"
            )

    # =========================================================================
    # Sensory pathway dimensions
    # =========================================================================
    # Visual pathway: depends on input encoding
    # Audio pathway: depends on audio features
    # Language pathway: depends on embedding dimension
    # These are validated dynamically during pathway creation

    # =========================================================================
    # Inter-region pathway dimensions
    # =========================================================================
    # Cortex → Hippocampus: cortex L2/3 output → hippocampus input
    cortex_l23_size = sizes.cortex_l23_size
    hippo_input = sizes.hippocampus_size  # Hippocampus uses n_input from config

    # Cortex → Striatum: cortex L5 output → striatum input
    cortex_l5_size = sizes.cortex_l5_size
    striatum_input = sizes.input_size  # Striatum input matches sensory input

    # Hippocampus → PFC: hippocampus output → PFC input
    hippo_output = sizes.hippocampus_size
    pfc_input = sizes.pfc_size

    # =========================================================================
    # Region size constraints
    # =========================================================================
    if sizes.n_actions < 1:
        errors.append(f"n_actions must be >= 1, got {sizes.n_actions}")

    if sizes.input_size < 1:
        errors.append(f"input_size must be >= 1, got {sizes.input_size}")

    if sizes.pfc_size < 1:
        errors.append(f"pfc_size must be >= 1, got {sizes.pfc_size}")

    if sizes.hippocampus_size < 1:
        errors.append(f"hippocampus_size must be >= 1, got {sizes.hippocampus_size}")

    if sizes.cortex_l23_size < 1:
        errors.append(f"cortex_l23_size must be >= 1, got {sizes.cortex_l23_size}")

    if sizes.cortex_l5_size < 1:
        errors.append(f"cortex_l5_size must be >= 1, got {sizes.cortex_l5_size}")

    # =========================================================================
    # Population coding constraints
    # =========================================================================
    if brain_config.use_population_coding:
        neurons_per_action = brain_config.neurons_per_action
        if neurons_per_action < 1:
            errors.append(
                f"neurons_per_action must be >= 1 when use_population_coding=True, "
                f"got {neurons_per_action}"
            )

        # Total striatum neurons = n_actions * neurons_per_action
        total_neurons = sizes.n_actions * neurons_per_action
        if total_neurons > 10000:
            errors.append(
                f"Striatum total neurons ({total_neurons} = {sizes.n_actions} actions × "
                f"{neurons_per_action} neurons/action) exceeds reasonable limit (10000). "
                f"Consider reducing n_actions or neurons_per_action"
            )

    return errors


def validate_global_consistency(config: "ThaliaConfig") -> List[str]:
    """Validate global consistency across all config modules.

    Args:
        config: Complete ThaliaConfig

    Returns:
        List of error messages (empty if valid)
    """
    errors: List[str] = []

    # =========================================================================
    # Device consistency (all modules should use same device)
    # =========================================================================
    devices = set()
    devices.add(config.global_.device)
    devices.add(config.brain.device)

    if len(devices) > 1:
        errors.append(
            f"Device mismatch across configs: {devices}. "
            f"All configs should use the same device (set global_.device)"
        )

    # =========================================================================
    # Timestep consistency
    # =========================================================================
    if config.brain.encoding_timesteps < 1:
        errors.append(
            f"brain.encoding_timesteps must be >= 1, got {config.brain.encoding_timesteps}"
        )

    if config.brain.test_timesteps < 1:
        errors.append(
            f"brain.test_timesteps must be >= 1, got {config.brain.test_timesteps}"
        )

    return errors


def validate_region_sizes(
    source_name: str,
    source_size: int,
    target_name: str,
    target_size: int,
    pathway_name: Optional[str] = None,
) -> None:
    """Validate that pathway source and target sizes are compatible.

    This is a utility for runtime validation during pathway creation.

    Args:
        source_name: Name of source region (for error messages)
        source_size: Output size of source region
        target_name: Name of target region
        target_size: Input size of target region
        pathway_name: Optional pathway name for clearer errors

    Raises:
        ConfigValidationError: If sizes are incompatible

    Example:
        >>> validate_region_sizes("cortex", 64, "hippocampus", 64, "cortex→hippocampus")
        >>> # Passes
        >>> validate_region_sizes("cortex", 64, "hippocampus", 32, "cortex→hippocampus")
        >>> # Raises ConfigValidationError
    """
    pathway_desc = f" ({pathway_name})" if pathway_name else ""

    if source_size != target_size:
        raise ConfigValidationError(
            f"Region size mismatch{pathway_desc}: {source_name} outputs {source_size} "
            f"but {target_name} expects {target_size}. "
            f"Adjust region sizes or add transformation layer."
        )


def check_config_and_warn(config: "ThaliaConfig", raise_on_error: bool = True) -> List[str]:
    """Check configuration and optionally warn instead of raising.

    Useful for exploratory work where you want to see all issues but not crash.

    Args:
        config: ThaliaConfig to validate
        raise_on_error: If True, raise ConfigValidationError. If False, return errors.

    Returns:
        List of validation errors (empty if valid)

    Raises:
        ConfigValidationError: If raise_on_error=True and validation fails
    """
    errors: List[str] = []

    try:
        validate_thalia_config(config)
    except ConfigValidationError as e:
        errors = str(e).split('\n')[1:]  # Skip "Configuration validation failed:" line

        if raise_on_error:
            raise
        else:
            # Print warnings
            print("\n⚠️  Configuration Validation Warnings:")
            for error in errors:
                print(error)
            print()

    return errors
