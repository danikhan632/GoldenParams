"""
Golden Parameters Package

This package provides functionality for computing golden parameters and reverse golden parameters
for neural network models, including sophisticated fluctuation strategies and KL divergence evaluation.
"""

from .core import get_reverse_golden_params
from .mixin import GoldilocksMixin
from .utils import eval_mode

__version__ = "0.1.0"

__all__ = [
    "get_reverse_golden_params",
    "GoldilocksMixin",
    "eval_mode",
]