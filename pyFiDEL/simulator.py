"""
simulator.py - create gaussian score distribution to mimic binary classifier

Soli Deo Gloria
"""

__author__ = "Sung-Cheol Kim"
__version__ = "1.0.0"

import numpy as np
from scipy import special
import pandas as pd
import seaborn as sns

from .ranks import auc_rank


class SimClassifier(object):
    # label naming
    class1 = "Y"
    class2 = "N"

    def __init__(self, N: int = 1000, rho: float = 0.5, y: list = None):
        self.N = N
        self.rho = rho
        self.N1 = int(self.N * self.rho)
        self.N2 = N - self.N1

        if y is None:
            y = ["Y"] * self.N1 + ["N"] * self.N2
        elif set(y) != set(["Y", "N"]):
            raise ValueError(f'y should have only "Y"/"N" - {set(y)}')

        self.y = np.array(y)
        self.score = None

    def create_gaussian_scores(
        self, auc0: float = 0.9, tol: float = 1e-4, max_iter: int = 2000
    ):
        """create gaussian scores to match AUC"""

        count = 0
        mu = 2.0 * special.erfinv(2.0 * auc0 - 1.0)
        # as auc approaches to .5, it is harder to converge
        max_iter = max_iter / ((auc0 - 0.5) * 10)

        # create score distribution by iterating creation of normal distribution
        simulated_auc = 0.5
        while abs(simulated_auc - auc0) > tol and count < max_iter:
            score1 = np.random.normal(0, 1, self.N1)
            score2 = np.random.normal(mu, 1, self.N2)

            score = np.zeros(self.N)
            score[self.y == "Y"] = score1
            score[self.y == "N"] = score2

            simulated_auc = auc_rank(score, self.y)

            count += 1

        print(f"Final AUC: {simulated_auc} (iter: {count}) mu2: {mu}")
        self.score = score

        return score

    def create_predictions(self, n_methods: int = 20, auc_list: list = None):
        """create n_methods gaussian score sets"""

        if auc_list is None:
            auc_list = np.linspace(0.51, 0.99, num=n_methods)
        else:
            n_methods = len(auc_list)

        pred = np.zeros((self.N, n_methods))
        for i in range(n_methods):
            pred[:, i] = self.create_gaussian_scores(auc0=auc_list[i])

        return pred

    def plot_o(self):
        """build data for"""

    def plot_score(self):
        """plot histogram of scores"""

        if self.score is None:
            print("create scores first.")
            return

        df = pd.DataFrame()
        df["score"] = self.score
        df["y"] = self.y

        sns.histplot(data=df, x="score", hue="y", hue_order=["Y", "N"])
