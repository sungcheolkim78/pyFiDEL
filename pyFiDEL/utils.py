"""
utils.py - utility functions

Soli Deo Gloria
"""

import numpy as np


def fermi_l(x: np.ndarray, l1: float, l2: float) -> np.ndarray:
    """calculate fermi-dirac distribution with np.ndarray x with l1 and l2"""

    return 1.0 / (1.0 + np.exp(l2 * x + l1))


def fermi_b(x: np.ndarray, b: float, m: float, normalized: bool = False):
    """calculate fermi-dirac distribution with np.ndarray x with beta and mu"""

    if normalized:
        return 1.0 / (1.0 + np.exp(b / float(len(x)) * (x - m * float(len(x)))))
    else:
        return 1.0 / (1.0 + np.exp(b * (x - m)))
