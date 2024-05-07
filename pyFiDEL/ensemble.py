# Copyright (c) 2024 by Sung-Cheol Kim, All rights reserved
"""
ensemble.py - ensemble method with FiDEL
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .ranks import auc_rank, get_fermi_root

logger = logging.getLogger("ensemble")
logging.basicConfig(level=logging.INFO)


class FiDEL(object):
    """ensemble classifier using FiDEL

    Examples:
        >>> from pyFiDEL import FiDEL
        >>> fdc = FiDEL()
        >>> fdc.add_prediction(pred1, "RandomForests")
        >>> fdc.add_prediction(pred2, "glm")
        >>> fdc.add_label(y_label)
        >>> fdc.calculate_performance()

    Attributes:
        predictions (np.ndarray): all previous predictions on samples
        n_samples (int): number of samples
        n_methods (int): number of classifiers
        method_names (list): list of method names
        rank_matrix (np.ndarray): collection of ranks based on the predictions
        ensemble_summary (pd.DataFrame): summary of methods
    """

    def __init__(self):
        self.predictions = []
        self.method_names = []
        self.n_samples = 0
        self.n_methods = 0
        self.rank_matrix = []
        self.y_label = []
        self.rho = 0
        self.summary = {
            "Name": [],
            "AUC": [],
            "beta": [],
            "mu": [],
            "r_star": [],
        }

    @property
    def df(self):
        print(self.summary)
        for k, v in self.summary.items():
            print(k, len(v))
        return pd.DataFrame.from_dict(self.summary)

    def add_predictions(self, prediction: list | np.ndarray, method_name: str = "") -> None:
        """add prediction of single method

        Args:
            prediction: model predictions or scores on samples. sample order must be same on each method
            method_name: model or classifier name
        """

        # if predictions has (n_samples, n_methods) shape - multiple predictions
        if isinstance(prediction, np.ndarray):
            self.n_samples, self.n_methods = prediction.shape

            for i in range(self.n_methods):
                pred = prediction[:, i]
                rank = pred.argsort().argsort()
                self.rank_matrix.append(rank)
                self.method_names.append(f"M{i}")
                self.predictions.append(pred)

        # if predictions is a single list
        else:
            self.n_samples = len(prediction)

            rank = np.array(prediction).argsort().argsort()
            self.rank_matrix.append(rank)
            self.method_names.append(method_name)
            self.predictions.append(pred)
            self.n_methods += 1

    def add_label(self, y_label: list, true_value: Any = "Y") -> None:
        """add label list of `Y` and `N`"""
        if self.n_samples == 0:
            logger.warning("add predictions first!")
            return

        assert len(y_label) == self.n_samples, f"sample number mismatch! {self.n_samples} {len(y_label)}"

        self.y_label = np.array(y_label)
        self.rho = np.sum(self.y_label == true_value) / len(y_label)

    def calculate_performance(self, method: str = "FiDEL", alpha: float = 1.0):
        """calculate ensemble performance

        Args:
            method: ensemble method name
            alpha: intensity coefficient of FiDEL calculation
        """

        logger.info("... sample #: %d, method #: %d", self.n_samples, self.n_methods)

        predictions = np.array(self.predictions)
        rank_matrix = np.array(self.rank_matrix).reshape(-1, self.n_methods)

        # calculate parameters for each methods
        self.summary["AUC"] = [auc_rank(pred, self.y_label) for pred in predictions]

        # reset parameters
        for params in ["beta", "mu", "r_star", "Name"]:
            self.summary[params] = []

        for i in range(self.n_methods):
            bm = get_fermi_root(self.summary["AUC"][i], self.rho, N=self.n_samples)
            self.summary["beta"].append(bm["beta"])
            self.summary["mu"].append(bm["mu"])
            self.summary["r_star"].append(bm["r_star"])
            self.summary["Name"].append(self.method_names[i])

        if method == "WoC":
            beta_mtx = np.ones((1, self.n_methods))
        else:
            beta_mtx = np.array(self.summary["beta"]).reshape((-1, self.n_methods))
        rstar_mtx = np.array(self.summary["r_star"]).reshape((-1, self.n_methods))

        self.logit_matrix = np.power(beta_mtx, alpha) * (rstar_mtx - rank_matrix)
        self.estimated_logit = -np.sum(self.logit_matrix, axis=1)
        self.estimated_prob = 1.0 / (1.0 + np.exp(-self.estimated_logit))
        self.estimated_auc = auc_rank(self.estimated_logit, self.y_label)

        logger.info("... estimated auc (ensemble): %f", self.estimated_auc)
