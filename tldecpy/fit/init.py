"""Initialization helpers: preprocessing, peak detection, and heuristic model seeding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_widths

from tldecpy.fit.bounds import make_bounds_from_init
from tldecpy.schemas import BackgroundSpec, PeakSpec
from tldecpy.utils.constants import KB_EV
from tldecpy.utils.robust import clean_outliers
from tldecpy.utils.sg import safe_savgol


@dataclass
class PeakSeed:
    """Candidate peak descriptor extracted from a preprocessed curve."""

    index: int
    Tm: float
    Im: float
    fwhm: float
    symmetry: float


def preprocess(
    x: np.ndarray,
    y: np.ndarray,
    sg_window: int = 7,
    sg_poly: int = 3,
    remove_outliers: bool = True,
) -> np.ndarray:
    r"""
    Preprocess a TL glow curve before peak detection.

    Parameters
    ----------
    x : numpy.ndarray
        Temperature grid in kelvin.
    y : numpy.ndarray
        Measured intensity values for each temperature sample.
    sg_window : int, default=7
        Savitzky-Golay window length (odd integer in samples).
    sg_poly : int, default=3
        Savitzky-Golay polynomial order.
    remove_outliers : bool, default=True
        If ``True``, apply a robust outlier cleaning pass before smoothing.

    Returns
    -------
    numpy.ndarray
        Smoothed, non-negative intensity signal used by automatic initialization.
    """
    _ = x
    y_proc = y.copy()
    if remove_outliers:
        y_proc = clean_outliers(y_proc)
    y_proc = safe_savgol(y_proc, window_length=sg_window, polyorder=sg_poly)
    return np.maximum(y_proc, 0.0)


def analyze_peak_shape(
    x: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    width_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    """Estimate shape descriptors (T1, T2, FWHM and symmetry) for one detected peak."""
    _ = y
    _, _, left_ips, right_ips = width_data

    def interpolate_temperature(float_idx: float) -> float:
        i = int(float_idx)
        frac = float_idx - i
        if i >= len(x) - 1:
            return float(x[-1])
        return float(x[i] + frac * (x[i + 1] - x[i]))

    t1 = interpolate_temperature(float(left_ips[0]))
    t2 = interpolate_temperature(float(right_ips[0]))
    tm = float(x[peak_idx])

    omega = t2 - t1
    delta = t2 - tm
    mu_g = delta / omega if omega > 0 else 0.5
    return {"T1": t1, "T2": t2, "omega": omega, "mu_g": mu_g}


def estimate_energy_chen(Tm: float, omega: float, order_hint: str = "fo") -> float:
    """Estimate activation energy with Chen-like FWHM heuristics."""
    c_w, b_w = (3.54, 1.0) if order_hint == "so" else (2.52, 1.0)
    energy = c_w * (KB_EV * Tm**2 / (omega + 1e-9)) - b_w * (2.0 * KB_EV * Tm)
    return max(0.1, float(energy))


def pick_peaks(x: np.ndarray, y: np.ndarray, sensitivity: float = 1.0) -> List[PeakSeed]:
    r"""
    Detect candidate TL peaks and return initialization seeds.

    Parameters
    ----------
    x : numpy.ndarray
        Temperature grid in kelvin.
    y : numpy.ndarray
        Preprocessed intensity signal.
    sensitivity : float, default=1.0
        Detection sensitivity multiplier. Higher values lower effective
        prominence/height thresholds and can recover weaker peaks.

    Returns
    -------
    list[PeakSeed]
        Ordered list of peak seeds including :math:`T_m`, :math:`I_m`,
        full-width-at-half-maximum and symmetry descriptors.
    """
    base_prominence_pct = 0.01
    base_height_pct = 0.005

    prominence = float(np.max(y) * (base_prominence_pct / max(sensitivity, 1e-6)))
    height = float(np.max(y) * (base_height_pct / max(sensitivity, 1e-6)))
    distance = max(3, int(7 / max(sensitivity, 1e-6)))

    peaks, _ = find_peaks(y, height=height, prominence=prominence, distance=distance)
    if len(peaks) == 0 and sensitivity <= 1.0:
        peaks, _ = find_peaks(y, height=height, distance=distance)

    try:
        widths = peak_widths(y, peaks, rel_height=0.5)
    except ValueError:
        return []

    seeds: List[PeakSeed] = []
    for i, idx in enumerate(peaks):
        width_data = (
            widths[0][i : i + 1],
            widths[1][i : i + 1],
            widths[2][i : i + 1],
            widths[3][i : i + 1],
        )
        shape = analyze_peak_shape(x, y, int(idx), width_data)
        seeds.append(
            PeakSeed(
                index=int(idx),
                Tm=float(x[idx]),
                Im=float(y[idx]),
                fwhm=float(shape["omega"]),
                symmetry=float(shape["mu_g"]),
            )
        )

    return seeds


def _model_allowed(allow_set: set[str], *aliases: str) -> bool:
    """Return ``True`` when any alias is present in the allowed-model set."""
    return any(alias in allow_set for alias in aliases)


def _select_model(symmetry: float, allow_set: set[str]) -> Tuple[str, str]:
    """Select an initial model key and energy-order hint based on shape + allowed models."""
    fo_preferred = "fo_rq"
    if "fo" not in allow_set and "fo_rq" not in allow_set and "fo_rb" in allow_set:
        fo_preferred = "fo_rb"

    if symmetry < 0.45 and _model_allowed(
        allow_set,
        "fo",
        "fo_rq",
        "fo_rb",
        "fo_ka",
        "fo_wp",
    ):
        return fo_preferred, "fo"
    if symmetry > 0.51 and _model_allowed(allow_set, "so", "so_ks", "so_la"):
        return "so_ks", "so"
    if _model_allowed(allow_set, "go", "go_kg", "go_rq", "go_ge"):
        return "go_kg", "go"
    if _model_allowed(
        allow_set,
        "fo",
        "fo_rq",
        "fo_rb",
        "fo_ka",
        "fo_wp",
    ):
        return fo_preferred, "fo"
    if _model_allowed(allow_set, "so", "so_ks", "so_la"):
        return "so_ks", "so"
    if _model_allowed(allow_set, "otor", "otor_lw", "otor_wo"):
        return "otor_lw", "otor"
    if _model_allowed(
        allow_set,
        "mo",
        "mix",
        "mo_kitis",
        "mo_quad",
        "mo_vej",
    ):
        return "mo_kitis", "mix"
    # Continuous models are only suggested when explicitly allowed.
    if (
        "cont_gauss" in allow_set
        or "gaussian" in allow_set
        or "continuous_gaussian" in allow_set
        or "continuous.continuous_gaussian" in allow_set
        or "dist.continuous_gaussian" in allow_set
    ):
        return "cont_gauss", "continuous"
    if (
        "cont_exp" in allow_set
        or "cont_lorentz" in allow_set
        or "lorentz" in allow_set
        or "exponential" in allow_set
        or "continuous_exponential" in allow_set
        or "continuous.continuous_exponential" in allow_set
        or "dist.continuous_exponential" in allow_set
    ):
        return "cont_exp", "continuous"
    if _model_allowed(allow_set, "continuous", "dist", "cont"):
        return "cont_gauss", "continuous"
    return "fo_rq", "fo"


def autoinit_multi(
    x: np.ndarray,
    y: np.ndarray,
    max_peaks: int = 6,
    allow_models: tuple[str, ...] = ("fo", "go", "otor_lw"),
    bg_mode: Literal["linear", "exponential", "none", "auto"] = "auto",
    sensitivity: float = 1.0,
) -> Tuple[List[PeakSpec], Optional[BackgroundSpec]]:
    r"""
    Build heuristic peak/background initialisation for multi-peak deconvolution.

    The function executes three stages:

    1. **Preprocess** — Savitzky-Golay smoothing + robust outlier removal.
    2. **Detect** — CWT-based peak finder returns candidate positions, FWHM
       and asymmetry coefficient :math:`\mu_g`.
    3. **Seed** — assigns a kinetic model per peak using :math:`\mu_g`, estimates
       activation energy :math:`E` with Chen-style FWHM heuristics, and
       constructs :class:`~tldecpy.schemas.PeakSpec` objects with automatic bounds.

    Parameters
    ----------
    x : numpy.ndarray
        Temperature grid :math:`T` in kelvin. Must be 1-D with at least
        10 points.
    y : numpy.ndarray
        Measured TL intensity :math:`I(T)`. Same length as ``x``. Values
        should be non-negative; negative values are clipped to zero during
        preprocessing.
    max_peaks : int, default=6
        Maximum number of peaks to include in the returned list.  If more
        candidates are detected, the ``max_peaks`` highest-intensity ones
        are kept and sorted by temperature.
    allow_models : tuple[str, ...], default=("fo", "go", "otor_lw")
        Allowed model families or canonical keys.  The seeding algorithm
        selects the best match from this set based on peak asymmetry.
        Continuous models (``"cont_gauss"``, ``"cont_exp"``) are only
        selected when explicitly listed here.
    bg_mode : {"linear", "exponential", "none", "auto"}, default="auto"
        Background initialisation mode.  ``"auto"`` adds an exponential
        background when the signal at the curve boundaries is significantly
        above the noise floor; ``"none"`` suppresses background entirely.
    sensitivity : float, default=1.0
        Peak detection sensitivity multiplier passed to :func:`pick_peaks`.
        Values > 1 lower prominence thresholds and recover weaker peaks;
        values < 1 raise thresholds and suppress weak candidates.
        Range: (0, ∞).

    Returns
    -------
    tuple[list[PeakSpec], BackgroundSpec | None]
        ``peaks`` — list of :class:`~tldecpy.schemas.PeakSpec`, one per
        detected candidate, with ``init``, ``bounds`` and ``name`` filled in.
        ``bg`` — :class:`~tldecpy.schemas.BackgroundSpec` if a background was
        inferred, or ``None``.

    Raises
    ------
    ValueError
        If ``x`` or ``y`` are not 1-D arrays with at least 10 points.

    Notes
    -----
    Activation-energy initialisation uses Chen-style peak-shape heuristics:

    .. math::

        E \approx c_w \frac{k T_m^2}{\omega} - b_w (2 k T_m)

    where :math:`\omega` is the FWHM and the constants :math:`c_w`, :math:`b_w`
    depend on the kinetic order.

    Examples
    --------
    >>> import tldecpy as tl
    >>> T, I = tl.load_refglow("x002")
    >>> peaks, bg = tl.autoinit_multi(T, I, max_peaks=4, allow_models=("fo_rq",))
    >>> print(len(peaks), "peaks seeded")

    References
    ----------
    .. [1] Chen, R., and McKeever, S. W. S. (1997). *Theory of
           thermoluminescence and related phenomena.* World Scientific.
    """
    y_clean = preprocess(x, y)
    seeds = pick_peaks(x, y_clean, sensitivity=sensitivity)
    allow_set = {m.lower() for m in allow_models}

    if len(seeds) > max_peaks:
        seeds = sorted(seeds, key=lambda seed: seed.Im, reverse=True)[:max_peaks]
        seeds = sorted(seeds, key=lambda seed: seed.Tm)

    specs: List[PeakSpec] = []
    for i, seed in enumerate(seeds):
        model, shape_hint = _select_model(seed.symmetry, allow_set)
        if shape_hint == "continuous":
            # Continuous-distribution heuristic used in Benavente (2019): TN ~ TM and IN ~ IM.
            init_params = {"Tn": seed.Tm, "In": seed.Im, "E0": 1.0, "sigma": 0.05}
        else:
            energy_est = estimate_energy_chen(seed.Tm, seed.fwhm, shape_hint)
            init_params = {"Tm": seed.Tm, "Im": seed.Im, "E": energy_est}
            if shape_hint == "go":
                init_params["b"] = 1.5
            if shape_hint == "otor":
                init_params["R"] = 1e-3
            if shape_hint == "mix":
                init_params["alpha"] = 0.5

        bounds = make_bounds_from_init(init_params, model)
        specs.append(
            PeakSpec.model_validate(
                {
                    "name": f"P{i + 1}",
                    "model": model,
                    "init": init_params,
                    "bounds": bounds,
                }
            )
        )

    background_spec: Optional[BackgroundSpec] = None
    if bg_mode == "auto":
        noise_floor = float(np.percentile(y_clean, 5))
        if y_clean[0] > noise_floor * 2.0 or y_clean[-1] > noise_floor * 2.0:
            background_spec = BackgroundSpec.model_validate(
                {"type": "exponential", "init": {"a": 0.0, "b": 1.0, "c": 100.0}}
            )
    elif bg_mode != "none":
        background_spec = BackgroundSpec.model_validate({"type": bg_mode})

    return specs, background_spec
