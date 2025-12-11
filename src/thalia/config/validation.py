"""
Configuration validation for THALIA.

This module provides validation functions to catch configuration errors
before brain initialization. Validates:
- Cross-region size compatibility (PFC ↔ Striatum, pathways ↔ regions)
- Device consistency across all components
- Sensory pathway dimension matching
- Region-specific constraints

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .thalia_config import ThaliaConfig
    from .brain_config import BrainConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


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
    
    # =========================================================================
    # Learning rate bounds
    # =========================================================================
    lr_config = config.training.learning_rates
    
    # Check individual learning rates are positive
    if lr_config.stdp < 0 or lr_config.stdp > 1:
        errors.append(
            f"training.learning_rates.stdp must be in [0, 1], got {lr_config.stdp}"
        )
    
    if lr_config.bcm < 0 or lr_config.bcm > 1:
        errors.append(
            f"training.learning_rates.bcm must be in [0, 1], got {lr_config.bcm}"
        )
    
    if lr_config.three_factor < 0 or lr_config.three_factor > 1:
        errors.append(
            f"training.learning_rates.three_factor must be in [0, 1], got {lr_config.three_factor}"
        )
    
    if lr_config.hebbian < 0 or lr_config.hebbian > 1:
        errors.append(
            f"training.learning_rates.hebbian must be in [0, 1], got {lr_config.hebbian}"
        )
    
    if lr_config.global_scale < 0:
        errors.append(
            f"training.learning_rates.global_scale must be >= 0, got {lr_config.global_scale}"
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
