'''
simulator.py - create gaussian score distribution to mimic binary classifier

Soli Deo Gloria
'''

__author__ = 'Sung-Cheol Kim'
__version__ = '1.0.0'

import numpy as np
from scipy import special

from ranks import auc_rank


class SimClassifier(object):
    # label naming
    class1 = 'Y'
    class2 = 'N'

    def __init__(self, N: int = 1000, rho: float = .5):
        self.N = N
        self.rho = rho

    def create_gaussian_scores(self, auc0: float = .9, y: list = [], tol: float = 0.0001, max_iter: int = 2000):
        ''' create gaussian scores to match AUC '''

        N1 = self.N * self.rho
        N2 = self.N - N1

        count = 0
        mu = 2. * special.erfinv(2. * auc0 - 1)
        max_iter = max_iter / ((auc0 - .5) * 10)

        # create score distribution by iterating creation of normal distribution
        simulated_auc = .5
        while abs(simulated_auc - auc0) > tol and count < max_iter:
            score1 = np.random.normal(0, 1, N1)
            score2 = np.random.normal(mu, 1, N2)

            score = np.zeros(self.N)
            score[y == 'Y'] = score1
            score[y == 'N'] = score2

            simulated_auc = auc_rank(score, y)

            count += 1

        print(f'Final AUC: {simulated_auc} (iter: {count}) mu2: {mu}')

        return score
