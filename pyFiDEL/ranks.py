'''
ranks.py - rank based metric calculation for structured learning

Soli Deo Gloria
'''

__author__ = 'Sung-Cheol Kim'
__version__ = '1.0.0'

import numpy as np


def auc_rank(scores: list, y: list):
    ''' calculate AUC using rank formula '''

    if len(scores) != len(y):
        raise ValueError(f'length of scores ({len(scores)}) does not match to labels ({len(y)})')

    N = len(y)
    N1 = sum([1 for x in y if x == 'Y'])
    N2 = N - N1

    # calculate the rank of scores O(NlogN)
    rank = np.array(scores).argsort().argsort()
    # calculate AUC using rank formula
    ans = np.abs(rank[y == 'Y'].sum() / N1 - rank[y == 'N'].sum() / N2) / N + .5

    if ans < .5:
        print('... class label might be wrong!')
        ans = 1 - .5

    return ans
