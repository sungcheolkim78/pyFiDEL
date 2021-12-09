'''
pcr.py - Probability of Class y at given Rank r (PCR)

Soli Deo Gloria
'''

__author__ = 'Sung-Cheol Kim'
__version__ = '1.0.0'

import numpy as np
import pandas as pd


class PCR(object):
    ''' probability of class at given rank '''

    def __init__(self, scores: list, y: list, sample_size: int = 100, sample_n: int = 300, method: str = 'bootstrap'):
        if sample_size == 0:
            self.sample_size = len(scores)
            self.sample_n = 1
        else:
            self.sample_size = sample_size
            self.sample_n = sample_n

        if len(scores) != len(y):
            raise ValueError(f'scores and y does not match size: {len(scores), len(y)}')

        self.scores = scores
        self.y = y
        self.df0 = pd.DataFrame({'scores': scores, 'y': y})
        self.N0 = len(y)
        self.N1 = sum(y == 'Y')
        self.N2 = self.N0 - self.N1
        self.rho = float(self.N1) / float(self.N0)

        if method == 'bootstrap':
            self.pcr = self.pcr_sample()

        # self.pcr_df = self.build_curve_pcr(self.prob)

    def pcr_sample(self):
        ''' calculate pcr using bootstrap method '''

        if len(self.y) - 10 < self.sample_size:
            self.sample_size = len(self.y) - 10
            print(f'... set sample size as {self.sample_size} (original size: {len(self.y)})')

        frac = float(self.sample_size) / float(self.N0)
        ans = np.zeros((self.sample_size, self.sample_n), dtype='float')

        # sample while maintaining the ratio
        for i in range(self.sample_n):
            tmp = self.df0.groupby('y').sample(frac=frac)

            # convert bool to float for averaging
            ans[:, i] = np.array(tmp.sort_values(by=['scores'])['y'].values == 'Y', dtype='float')

        self.sample_table = ans       # For record
        self.pcr = ans.mean(axis=1)

        return self.pcr


