"""
Noise generation utilities for simulation.
"""

from __future__ import annotations

import numpy as np


def add_noise(
    y: np.ndarray,
    mode: str = "gaussian",
    sigma: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    r"""
    Inject synthetic noise into a clean signal array.

    Parameters
    ----------
    y : numpy.ndarray
        Clean (noiseless) signal, 1-D float array in arbitrary detector
        counts.  Negative values are clipped to zero before Poisson sampling.
    mode : {"gaussian", "poisson"}, default="gaussian"
        Noise model:

        - ``"gaussian"`` — adds independent identically distributed noise
          :math:`\varepsilon_i \sim \mathcal{N}(0, \sigma^2)`.  Noise
          amplitude is constant across the array (homoscedastic).
        - ``"poisson"`` — draws integer counts
          :math:`\tilde{y}_i \sim \text{Poisson}(y_i)`, giving
          heteroscedastic noise with variance equal to the signal
          :math:`\text{Var}(\tilde{y}_i) = y_i`.  ``sigma`` is ignored.
    sigma : float, default=0.0
        Standard deviation for Gaussian noise in the same units as ``y``.
        Ignored when ``mode="poisson"``.  Setting ``sigma=0.0`` returns the
        input unchanged (Gaussian mode only).
    seed : int | None, optional
        Random seed for the NumPy default RNG.  Pass an integer for
        reproducible results; ``None`` (default) uses a random seed.

    Returns
    -------
    numpy.ndarray
        Noisy signal array, same shape and dtype as ``y`` (float64 for
        Poisson mode, same as input for Gaussian mode).

    Examples
    --------
    >>> import numpy as np
    >>> from tldecpy.simulate.noise import add_noise
    >>> y_clean = np.array([100.0, 500.0, 1000.0, 500.0, 100.0])
    >>> y_gauss = add_noise(y_clean, mode="gaussian", sigma=5.0, seed=0)
    >>> y_pois  = add_noise(y_clean, mode="poisson", seed=0)
    """
    rng = np.random.default_rng(seed)

    if mode == "gaussian":
        noise = rng.normal(0, sigma, size=y.shape)
        return y + noise

    if mode == "poisson":
        # Poisson is integer based, but we approximate for floats
        # Variance = y. Std = sqrt(y).
        # We use gaussian approximation for large N or poisson for small
        y_noisy = rng.poisson(np.maximum(y, 0))
        return y_noisy.astype(float)

    return y
