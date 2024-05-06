# Copyright (c) 2024 by Sung-Cheol Kim, All rights reserved 
"""
ranks.py - rank-based metric calculation for supervised learning
"""

import logging
import numpy as np
import pandas as pd
import scipy

from typing import Tuple

logger = logging.getLogger("ranks")
logging.basicConfig(level=logging.INFO)


def auc_rank(scores: list, y: list) -> float:
    """calculate AUC using rank formula"""

    if len(scores) != len(y):
        raise ValueError(f"length of scores ({len(scores)}) does not match to labels ({len(y)})")

    N = len(y)
    N1 = sum([1 for x in y if x == "Y"])
    N2 = N - N1

    # calculate the rank of scores O(NlogN)
    rank = np.array(scores).argsort().argsort()

    # calculate AUC using rank formula
    ans = np.abs(rank[y == "Y"].sum() / N1 - rank[y == "N"].sum() / N2) / N + 0.5

    if ans < 0.5:
        print("... class label might be wrong!")
        ans = 1 - 0.5

    return ans


def build_metric(scores: list, y: list, method="root") -> Tuple[pd.DataFrame, dict]:
    """calculate all metrics using rank formula"""

    # calculate basic parameters
    N = len(y)
    N1 = np.sum(y == "Y")
    N2 = N - N1
    rho = float(N1 / N)
    auc0 = auc_rank(scores, y)

    # prepare data frame
    df = pd.DataFrame({"score": scores, "y": y})
    df.sort_values(by=["score"], ignore_index=True, inplace=True)
    df["rank"] = range(len(y) + 1)[1:]

    t_rank = df["y"] == "Y"
    f_rank = df["y"] == "N"

    df["tpr"] = np.cumsum(t_rank) / N1
    df["fpr"] = np.cumsum(f_rank) / N2
    df["bac"] = 0.5 * (df["tpr"] + 1.0 - df["fpr"])
    df["prec"] = np.cumsum(t_rank) / df["rank"]

    info = {
        "auc_rank": auc0,
        "auc_bac": 2.0 * df["bac"].mean() - 0.5,
        "auprc": 0.5 * N1 / N * (1.0 + N / (N1 * N1) * np.sum(df["prec"][1: N - 2] * df["prec"][2: N - 1])),
        "rho": rho,
    }
    if method == "min":
        info.update(get_fermi_min(auc0, rho))
    else:
        info.update(get_fermi_root(auc0, rho))

    return df, info


def get_fermi_min(auc: float, rho: float, N: int = 1, resol: float = 0.0001, method: str = "beta") -> dict:
    """calculate beta and mu (or l1 and l2) from AUC and rho"""

    # check auc range
    if auc < 0.5:
        auc = 1.0 - auc

    lambdas = get_lambda(auc, rho, N=1000)
    initials = [lambdas["l2"] * 1000, -lambdas["l1"] / (1000 * lambdas["l2"])]

    # find beta, mu
    # temp = scipy.optimize.fmin(_cost, initials, args=(auc, rho, resol), maxiter=8000, ftol=1E-6)
    res = scipy.optimize.minimize(_cost, initials, args=(auc, rho, resol))
    temp = res.x

    r_star = 1.0 / temp[0] * np.log((1.0 - rho) / rho) + temp[1]

    if method == "beta":
        return {
            "beta": temp[0] / N,
            "mu": temp[1] * N,
            "r_star": r_star * N,
        }
    else:
        return {
            "l1": -temp[0] * temp[1],
            "l2": temp[1] / N,
            "r_star": r_star * N,
        }


def _cost0(bm: list, auc: float, rho: float, resol: float) -> float:
    """cost function to minimize using simple approximation"""

    r_prime = np.linspace(0, 1.0, num=int(1.0 / resol), endpoint=False)
    sum1 = np.sum(resol / (1.0 + np.exp(bm[0] * (r_prime - bm[1]))))
    sum2 = np.sum(resol * r_prime / (1.0 + np.exp(bm[0] * (r_prime - bm[1]))))

    diff1 = rho - sum1
    diff2 = 0.5 * rho - rho * (1.0 - rho) * (auc - 0.5) - sum2

    return diff1 * diff1 + diff2 * diff2


def _cost(bm: list, auc: float, rho: float, resol: float) -> float:
    """cost function to minimize using integrate"""

    sum1 = scipy.integrate.quad(lambda x: 1.0 / (1.0 + np.exp(bm[0] * (x - bm[1]))), 0, 1.0)
    sum2 = scipy.integrate.quad(lambda x: x / (1.0 + np.exp(bm[0] * (x - bm[1]))), 0, 1.0)

    diff1 = rho - sum1[0]
    diff2 = 0.5 * rho - rho * (1.0 - rho) * (auc - 0.5) - sum2[0]

    return diff1 * diff1 + diff2 * diff2


def get_fermi_root(auc: float, rho: float, N: int = 1) -> dict:
    """calculate beta and mu from AUC and rho"""

    lambdas = get_lambda(auc, rho, N=1000)
    beta0 = lambdas["l2"] * 1000

    beta = scipy.optimize.brentq(_froot, 0.05, beta0 * 5.0, args=(auc, rho))
    mu = 0.5 - np.log(np.sinh(beta * (1.0 - rho) * 0.5) / np.sinh(beta * rho * 0.5)) / beta
    r_star = 1.0 / beta * np.log((1.0 - rho) / rho) + mu

    return {
        "beta": beta / float(N),
        "mu": mu * float(N),
        "r_star": r_star * float(N),
    }


def _froot(beta: float, auc: float, rho: float) -> float:
    """cost function for auc and rho"""

    mu = 0.5 - np.log(np.sinh(beta * (1.0 - rho) * 0.5) / np.sinh(beta * rho * 0.5)) / beta
    part1 = (scipy.special.spence(1.0 + np.exp(beta * (mu - 1.0))) - scipy.special.spence(1.0 + np.exp(beta * mu)) - beta * np.log(np.exp(beta * (mu - 1.0)) + 1.0)) / (beta * beta)
    part2 = 0.5 * rho - rho * (1.0 - rho) * (auc - 0.5)
    # print(auc, rho, mu, '-', beta, part1 - part2)

    return part1 - part2


def get_lambda(auc: float, rho: float, N: int = 1000) -> dict:
    """calculate lambda1, lambda2 from auc, rho"""

    fN = float(N)
    l1_low = np.log(1.0 / rho - 1.0) - 12.0 * fN * (auc - 0.5) / (fN * fN - 1.0) * ((fN + 1 + fN * rho) * 0.5 - fN * rho * auc)
    l2_low = 12.0 * fN * (auc - 0.5) / (fN * fN - 1.0)

    temp = np.sqrt(rho * (1.0 - rho) * (1.0 - 2.0 * (auc - 0.5)))
    l1_high = -2.0 * rho / (np.sqrt(3) * temp)
    l2_high = 2.0 / (np.sqrt(3) * fN * temp)

    alpha = 2.0 * (auc - 0.5)
    l1 = l1_high * alpha + l1_low * (1.0 - alpha)
    l2 = l2_high * alpha + l2_low * (1.0 - alpha)
    r_star = 1.0 / l2 * np.log((1.0 - rho) / rho) - l1 / l2

    return {
        "l1low": l1_low,
        "l2low": l2_low,
        "l1high": l1_high,
        "l2high": l2_high,
        "l1": l1,
        "l2": l2,
        "r_star": r_star,
    }


def build_correspond_table(auclist: list, rholist: list, resol: float = 0.001, method: str = "root") -> pd.DataFrame:
    """calculate correspondence table between (auc, rho) and (beta, mu)"""

    ans = pd.DataFrame()

    for auc in auclist:
        for rho in rholist:
            row = {"auc": auc, "rho": rho}
            if method == "root":
                row.update(get_fermi_root(auc, rho))
            else:
                row.update(get_fermi_min(auc, rho, resol=resol))
            ans = ans.append(row, ignore_index=True)

    return ans
