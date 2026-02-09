"""
Custom exception classes and validation utilities for Thalia.

This module provides:
1. Hierarchical exception classes for different error categories
2. Validation utilities that enforce biological plausibility constraints
3. Consistent error message formatting

Exception Hierarchy:
====================
ThaliaError (base) - Base exception for all Thalia-specific errors
├── ConfigurationError - Invalid configuration parameters

Design Philosophy:
==================
- Specific exception types enable targeted error handling
- Error messages reference ADRs for educational value
- Validation utilities prevent constraint violations early
- Consistent formatting improves debugging experience

Author: Thalia Project
Date: December 12, 2025
"""

from __future__ import annotations

# =============================================================================
# Exception Hierarchy
# =============================================================================


class ThaliaError(Exception):
    """Base exception for all Thalia-specific errors.

    All custom exceptions in Thalia inherit from this class, enabling
    code to catch Thalia errors specifically.
    """


class ConfigurationError(ThaliaError):
    """Invalid configuration parameters.

    Raised when configuration values are out of valid range or incompatible
    with each other.
    """
