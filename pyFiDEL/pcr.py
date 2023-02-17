"""
pcr.py - Probability of Class y at given Rank r (PCR)
"""

import logging
import numpy as np
import pandas as pd

from typing import Tuple

from .ranks import get_fermi_root
from .ci import var_auc_fermi
from .utils import fermi_b

logger = logging.getLogger("utils")
logging.basicConfig(level=logging.INFO)


class PCR(object):
    """probability of class at given rank"""

    def __init__(
        self,
        scores: list,
        y: list,
        sample_size: int = 100,
        sample_n: int = 300,
        method: str = "bootstrap",
    ):
        if sample_size == 0:
            self.sample_size = len(scores)
            self.sample_n = 1
        else:
            self.sample_size = sample_size
            self.sample_n = sample_n

        if len(scores) != len(y):
            raise ValueError(f"scores and y does not match size: {len(scores), len(y)}")

        self.scores = scores
        self.y = y
        self.df0 = pd.DataFrame({"scores": scores, "y": y})
        self.N0 = len(y)
        self.N1 = sum(np.array(y) == "Y")
        self.N2 = self.N0 - self.N1
        self.rho = float(self.N1) / float(self.N0)

        if method == "bootstrap":
            self.pcr = self.pcr_sample()

        # self.pcr_df = self.build_curve_pcr(self.prob)

    def pcr_sample(self):
        """calculate pcr using bootstrap method"""

        if len(self.y) - 10 < self.sample_size:
            self.sample_size = len(self.y) - 10
            print(f"... set sample size as {self.sample_size} (original size: {len(self.y)})")

        frac = float(self.sample_size) / float(self.N0)
        ans = np.zeros((self.sample_size, self.sample_n), dtype="float")

        # sample while maintaining the ratio
        for i in range(self.sample_n):
            tmp = self.df0.groupby("y").sample(frac=frac)

            # convert bool to float for averaging
            ans[:, i] = np.array(tmp.sort_values(by=["scores"])["y"].values == "Y", dtype="float")

        self.sample_table = ans  # For record
        self.pcr = ans.mean(axis=1)

        return self.pcr

    def auc(self):
        """calculate AUC from pcr"""

        N = self.sample_size
        N1 = int(self.sample_size * self.rho)
        N2 = N - N1
        rank = np.linspace(1, N, num=N)

        return np.abs(np.sum(rank * self.pcr) / N1 - np.sum(rank * (1.0 - self.pcr)) / N2) / N + 0.5

    def auprc(self):
        """calculate area under precision recall curve"""

        N = self.sample_size
        prec = np.cumsum(self.pcr) / np.linspace(1, N, num=N)

        return 0.5 * self.rho * (1.0 + np.sum(prec[1: (N - 2)] * prec[2: (N - 1)]) / (N * self.rho * self.rho))

    def build_metric(self) -> Tuple[pd.DataFrame, dict]:
        """calculate metric and parameters from pcr"""

        N = self.sample_size
        N1 = int(self.sample_size * self.rho)
        N2 = N - N1

        print(f"... build metric parameters (N = {N})")

        # calculate curve data
        self.df = pd.DataFrame({"rank": np.linspace(1, N, num=N), "prob": self.pcr})
        self.df["tpr"] = np.cumsum(self.pcr) / N1
        self.df["fpr"] = np.cumsum(1.0 - self.pcr) / N2
        self.df["bac"] = 0.5 * (self.df["tpr"] + 1.0 - self.df["fpr"])
        self.df["prec"] = np.cumsum(self.pcr) / self.df["rank"]

        # calculate metrics
        auc0 = self.auc()

        self.info = {
            "auc_rank": auc0,
            "auc_bac": 2.0 * self.df["bac"].mean() - 0.5,
            "auprc": self.auprc(),
            "rho": self.rho,
        }

        self.info.update(get_fermi_root(auc0, self.rho))
        self.info.update(var_auc_fermi(auc0, self.rho, N))

        return self.df, self.info

    def check_fermi(self):
        """check matching between fermi-dirac distribution and pcr"""

        self.df["fy"] = fermi_b(self.df["rank"], self.info["beta"], self.info["mu"], normalized=True)
        self.df["err"] = self.df["prob"] - self.df["fy"]

        return {
            "MAE": self.df["err"].abs().mean(),
            "RMSE": np.sqrt(np.mean(self.df["err"] * self.df["err"])),
            "SSEV": np.sum(self.df["err"] * self.df["err"]) / self.df["err"].var(),
        }
