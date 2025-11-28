"""
Spike encoding strategies.
"""

from thalia.encoding.poisson import poisson_encode
from thalia.encoding.rate import rate_encode

__all__ = ["poisson_encode", "rate_encode"]
