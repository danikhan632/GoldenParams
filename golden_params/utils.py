"""
Utility functions for golden parameters computation.
"""

from contextlib import contextmanager


@contextmanager
def eval_mode(model):
    """Context manager to temporarily set model to eval mode."""
    prev = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(prev)