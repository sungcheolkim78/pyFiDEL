"""
ci.py - confidence interval calculation

Soli Deo Gloria
"""

__author__ = "Sung-Cheol Kim"
__version__ = "1.0.0"

import logging
import numpy as np

from .utils import fermi_b
from .ranks import get_fermi_root

logger = logging.getLogger("ci")
logging.basicConfig(level=logging.INFO)


def Pxy_int(beta: float, mu: float, rho: float, resol: float = 1e-6) -> float:
    """calculate Pxy with integral formula"""

    logger.info("... Pxy integral calculation with resolution = %f", resol)

    p_table = fermi_b(
        np.linspace(0, 1, num=int(1.0 / resol)), b=beta, m=mu, normalized=False
    )

    A = p_table
    B = np.linspace(len(p_table), 1, num=len(p_table)) - np.cumsum(p_table[::-1])[::-1]

    ans = np.sum(A * B) - np.sum(A * (1.0 - A))
    ans = ans * resol * resol / (rho * (1.0 - rho))

    return ans


def Pxxy_int(beta: float, mu: float, rho: float, resol: float = 1e-6) -> float:
    """calculate Pxxy iwth integral formula"""

    logger.info("... Pxxy integral calculation with resolution = %f", resol)

    p_table = fermi_b(
        np.linspace(0, 1, num=int(1.0 / resol)), b=beta, m=mu, normalized=False
    )

    A = 2.0 * p_table * np.cumsum(p_table) - p_table * p_table
    B = np.linspace(len(p_table), 1, num=len(p_table)) - np.cumsum(p_table[::-1])[::-1]

    ans = (
        np.sum(A * B)
        - np.sum(p_table * p_table * (1.0 - p_table))
        - np.sum(p_table * p_table)
        - 2.0 * np.sum(p_table * (1.0 - p_table))
    )
    ans = ans * resol * resol * resol / (rho * rho * (1.0 - rho))

    return ans


def Pxyy_int(beta: float, mu: float, rho: float, resol: float = 1e-6) -> float:
    """calculate Pxyy iwth integral formula"""

    print(f"... Pxyy integral calculation with resolution = {resol}")

    p_table = fermi_b(
        np.linspace(0, 1, num=int(1.0 / resol)), b=beta, m=mu, normalized=False
    )

    A = p_table
    B = np.cumsum((1.0 - p_table)[::-1])[::-1] ** 2

    ans = (
        np.sum(A * B)
        - np.sum(p_table * (1.0 - p_table) * (1.0 - p_table))
        - np.sum((1.0 - p_table) ** 2)
        - 2.0 * np.sum(p_table * (1.0 - p_table))
    )
    ans = ans * resol * resol * resol / (rho * (1.0 - rho) * (1.0 - rho))

    return ans


def var_auc_fermi(auc: float, rho: float, N: int, resol: float = 1e-6):
    """calculate variance of AUC from Fermi-Dirac distribution"""

    # calculate parameters
    bm = get_fermi_root(auc, rho)
    N1 = int(N * rho)
    N2 = N - N1

    # calculate Pxy, Pxxy, Pxyy
    Pxy_value = Pxy_int(bm["beta"], bm["mu"], rho, resol=resol)
    Pxxy_value = Pxxy_int(bm["beta"], bm["mu"], rho, resol=resol)
    Pxyy_value = Pxyy_int(bm["beta"], bm["mu"], rho, resol=resol)

    # calculate variance and std of AUC
    var_auc = (
        Pxy_value * (1.0 - Pxy_value)
        + (N1 - 1.0) * (Pxxy_value - Pxy_value * Pxy_value)
        + (N2 - 1.0) * (Pxyy_value - Pxy_value * Pxy_value)
    ) / (N1 * N2)
    std_auc = np.sqrt(var_auc)

    return {
        "var_auc": var_auc,
        "Pxy": Pxy_value,
        "Pxxy": Pxxy_value,
        "Pxyy": Pxyy_value,
        "beta": bm["beta"],
        "mu": bm["mu"],
        "auc0": auc,
        "rho": rho,
        "auc sigma": std_auc,
        "95% ci": [auc - 1.96 * std_auc, auc + 1.96 * std_auc],
    }
