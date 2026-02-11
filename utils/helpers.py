# -*- coding: utf-8 -*-

"""Utility functions and decorators."""

import functools
from typing import Callable

import torch


def contiguous(fn: Callable) -> Callable:
    """
    Decorator that ensures all tensor arguments are contiguous before calling the function.
    
    Args:
        fn: Function to wrap
        
    Returns:
        Wrapped function with contiguous tensor arguments
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(
            ctx,
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}
        )
    return wrapper
