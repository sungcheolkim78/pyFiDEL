# Copyright (c) 2024 by Sung-Cheol Kim, All rights reserved
"""
simulator.py - create gaussian score distribution to mimic binary classifier
"""

import logging

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import special

from .ranks import auc_rank

logger = logging.getLogger("simulator")
logging.basicConfig(level=logging.INFO)


class SimClassifier(object):
    """Simulator for the binary classifier.

    It generates multiple Gaussian score distributions to model various
    binary classifier with different AUC.

    Attributes:
        N (int): number of samples
        rho (float): ratio between True/False element counts
        N1 (int): number of True elements
        N2 (int): number of False elements
        y (nd.array): array of ground truth
    """

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
        self.pred = None

    def create_gaussian_scores(self, auc0: float = 0.9, tol: float = 1e-4, max_iter: int = 2000) -> np.ndarray:
        """create gaussian scores to match AUC.

        Args:
            auc0: target Area under ROC curve (AUC)
            tol: tolerance for target AUC
            max_iter: the maximum number of iteration

        Returns:
            score of binary classifier
        """

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

        logger.info("Final AUC: %f (iter: %d) mu2: %f", simulated_auc, count, mu)
        self.score = score

        return score

    def create_predictions(self, n_methods: int = 20, auc_list: list = None) -> np.ndarray:
        """create n_methods gaussian score sets

        Args:
            n_methods: number of classifiers
            auc_list: list of target AUCs

        Returns:
            prediction matrix of shape (N, n_methods)
        """

        if auc_list is None:
            auc_list = np.linspace(0.51, 0.99, num=n_methods)
        else:
            n_methods = len(auc_list)

        pred = np.zeros((self.N, n_methods))
        for i in range(n_methods):
            pred[:, i] = self.create_gaussian_scores(auc0=auc_list[i])

        self.pred = pred

        return pred

    def plot_score(self) -> None:
        """plot histogram of scores"""

        if self.score is None:
            logger.warning("create scores first.")
            return

        df = pd.DataFrame()
        df["score"] = self.score
        df["y"] = self.y

        sns.histplot(data=df, x="score", hue="y", hue_order=["Y", "N"])
