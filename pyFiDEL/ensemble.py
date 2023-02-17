"""
ensemble.py - ensemble method with FiDEL

Soli Deo Gloria
"""

__author__ = "Sung-Cheol Kim"
__version__ = "1.0.0"

import numpy as np
import pandas as pd

from .ranks import auc_rank
from .ranks import get_fermi_root


class FiDEL(object):
    def __init__(self, predictions: np.array, method_names: list = None):
        self.predictions = predictions
        self.n_samples = predictions.shape[0]
        self.n_methods = predictions.shape[1]

        if method_names is None:
            self.method_names = [f"M{i}" for i in range(self.n_methods)]
        else:
            self.method_names = method_names

        # calculate rank matrix for each column
        self.rank_matrix = np.zeros_like(predictions)
        for i in range(self.n_methods):
            self.rank_matrix[:, i] = [
                index
                for e, index in sorted(
                    zip(
                        predictions[:, i],
                        np.linspace(1, self.n_samples, num=self.n_samples),
                    )
                )
            ]

    def calculate_performance(self, y: list, method: str = "FiDEL", alpha: float = 1):
        """calculate ensemble performance"""

        print(f"... sample #: {self.n_samples}, method #: {self.n_methods}")
        self.y = y
        self.N = len(y)
        self.rho = np.sum(y == "Y") / self.N

        # calculate parameters for each methods
        self.df = pd.DataFrame({"Name": self.method_names})
        self.df["AUC"] = [
            auc_rank(self.predictions[:, i], self.y) for i in range(self.n_methods)
        ]
        self.df["beta"] = 1.0
        self.df["mu"] = 0.0
        self.df["r_star"] = 0.0

        for i in range(self.n_methods):
            bm = get_fermi_root(self.df.at[i, "AUC"], self.rho, N=self.n_samples)
            self.df.at[i, "beta"] = bm["beta"]
            self.df.at[i, "mu"] = bm["mu"]
            self.df.at[i, "r_star"] = bm["r_star"]

        if method == "WoC":
            self.df["beta"] = 1

        self.logit_matrix = np.power(self.df["beta"].values, alpha) * (
            self.df["r_star"].values - self.rank_matrix
        )
        self.estimated_logit = -np.sum(self.logit_matrix, axis=1)
        self.estimated_prob = 1.0 / (1.0 + np.exp(-self.estimated_logit))

        self.estimated_auc = auc_rank(self.estimated_logit, y)

        print(f"... estimated auc (ensemble): {self.estimated_auc}")
