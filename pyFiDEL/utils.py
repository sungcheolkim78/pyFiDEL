'''
utils.py - utility functions

Soli Deo Gloria
'''

__author__ = 'Sung-Cheol Kim'
__version__ = '1.0.0'

import numpy as np


def fermi_l(x: np.array, l1: float, l2: float) -> np.array:
    ''' calculate fermi-dirac distribution with np.array x with l1 and l2'''

    return 1. / (1. + np.exp(l2 * x + l1))


def fermi_b(x: np.array, b: float, m: float, normalized: bool = False):
    ''' calculate fermi-dirac distribution with np.array x with beta and mu'''

    if normalized:
        return 1. / (1. + np.exp(b / float(len(x)) * (x - m * float(len(x)))))
    else:
        return 1. / (1. + np.exp(b * (x - m)))
