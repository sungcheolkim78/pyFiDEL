# Copyright (c) 2024 by Sung-Cheol Kim, All rights reserved 
"""
utils.py - utility functions
"""

import logging
import numpy as np

logger = logging.getLogger("utils")
logging.basicConfig(level=logging.INFO)


def fermi_l(x: np.ndarray, l1: float, l2: float) -> np.ndarray:
    """calculate Fermi-Dirac distribution with np.array x with l1 and l2

    Args:
        x: array of inputs
        l1: l1 coefficient
        l2: l2 coefficient

    Returns:
        Fermi-Dirac distribution
    """

    return 1.0 / (1.0 + np.exp(l2 * x + l1))


def fermi_b(x: np.ndarray, b: float, m: float, normalized: bool = False) -> np.ndarray:
    """calculate Fermi-Dirac distribution of array x with beta and mu.

    Args:
        x: input array
        b: beta
        m: mu
        normalized: option to normalized distribution

    Returns:
        Fermi-Dirac distribution
    """

    if normalized:
        return 1.0 / (1.0 + np.exp(b / float(len(x)) * (x - m * float(len(x)))))
    else:
        return 1.0 / (1.0 + np.exp(b * (x - m)))
