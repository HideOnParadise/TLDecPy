"""
Objective functions and fit metrics.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class FitResult:
    """
    Container for deconvolution results.
    """

    params: Dict[str, float]
    cov: Optional[np.ndarray]
    metrics: Dict[str, float]
    y_hat: np.ndarray
    residuals: np.ndarray
    converged: bool
    message: str
    model_type: str


def calculate_metrics(y_obs: np.ndarray, y_fit: np.ndarray, num_params: int) -> Dict[str, float]:
    """
    Calculate FOM, R2, RCS, SSR.
    """
    residuals = y_obs - y_fit
    ssr = np.sum(residuals**2)

    # FOM: Figure of Merit (Balian and Eddy, 1977)
    # Sum of absolute residuals / Sum of observed area
    area_obs = np.sum(y_obs)
    if area_obs == 0:
        fom = np.inf
    else:
        fom = 100.0 * np.sum(np.abs(residuals)) / area_obs

    # R2 (Coefficient of determination)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1 - (ssr / ss_tot) if ss_tot != 0 else 0.0

    # RCS: Reduced Chi-Square
    dof = len(y_obs) - num_params
    rcs = (ssr / dof) if dof > 0 else np.inf

    return {"FOM": round(fom, 4), "R2": round(r2, 6), "SSR": round(ssr, 6), "RCS": round(rcs, 6)}
