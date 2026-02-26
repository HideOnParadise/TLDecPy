"""Advanced peak-detection routines based on continuous wavelet transforms."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import find_peaks, peak_widths

from tldecpy.fit.init import PeakSeed, analyze_peak_shape

try:
    from scipy.signal import find_peaks_cwt as _find_peaks_cwt
except Exception:  # pragma: no cover - version-dependent availability
    _find_peaks_cwt = None


def _as_1d_array(values: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    """Convert iterable inputs to validated 1D float arrays."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size < 5:
        raise ValueError(f"{name} must contain at least 5 points.")
    return arr


def _mad_sigma(values: np.ndarray) -> float:
    """Robustly estimate noise scale using median absolute deviation."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return max(1.4826 * mad, np.finfo(float).eps)


def _resolve_widths(length: int, widths: np.ndarray | Iterable[float] | None) -> np.ndarray:
    """Build an adaptive CWT width grid or validate a user-supplied one."""
    if widths is None:
        max_width = max(4, min(48, length // 6))
        return np.arange(2, max_width + 1, dtype=float)

    out = np.asarray(list(widths), dtype=float)
    out = out[np.isfinite(out) & (out > 0.0)]
    if out.size == 0:
        raise ValueError("widths must contain at least one finite positive value.")
    return np.unique(np.sort(out))


def _ricker_wavelet(points: int, scale: float) -> np.ndarray:
    """Mexican-hat (Ricker) wavelet compatible with older SciPy behavior."""
    n = max(3, int(points))
    if n % 2 == 0:
        n += 1
    a = max(float(scale), np.finfo(float).eps)
    x = np.arange(-(n // 2), (n // 2) + 1, dtype=float)
    xa2 = (x / a) ** 2
    norm = 2.0 / (np.sqrt(3.0 * a) * np.pi**0.25)
    return norm * (1.0 - xa2) * np.exp(-0.5 * xa2)


def _cwt_ricker(data: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    Compute a simple CWT matrix with the Ricker wavelet.

    This fallback keeps the algorithm available even when ``scipy.signal.cwt``
    is not exported by the installed SciPy version.
    """
    signal = np.asarray(data, dtype=float)
    coeffs = np.zeros((widths.size, signal.size), dtype=float)
    for i, width in enumerate(widths):
        points = min(signal.size, max(5, int(np.ceil(10.0 * width))))
        wavelet = _ricker_wavelet(points=points, scale=float(width))
        coeffs[i, :] = np.convolve(signal, wavelet, mode="same")
    return coeffs


def detect_peaks_cwt(
    x: np.ndarray,
    y: np.ndarray,
    min_snr: float = 1.0,
    widths: np.ndarray | Iterable[float] | None = None,
) -> list[PeakSeed]:
    r"""
    Detect TL peak candidates using CWT ridge responses (Ricker/Mexican-hat).

    Parameters
    ----------
    x : numpy.ndarray
        Temperature grid :math:`T` in kelvin.
    y : numpy.ndarray
        Intensity signal :math:`I(T)` (typically preprocessed and non-negative).
    min_snr : float, default=1.0
        Minimum signal-to-noise threshold used to retain ridge-consistent candidates.
    widths : numpy.ndarray | Iterable[float] | None, optional
        Wavelet widths in sample units. When omitted, an adaptive range based on
        data length is used.

    Returns
    -------
    list[PeakSeed]
        Peak candidates with position, intensity and shape descriptors compatible
        with the auto-initialization pipeline.

    Notes
    -----
    The implementation uses ``find_peaks_cwt`` when available and falls back to a
    local ridge-threshold detector otherwise.
    """
    x_arr = _as_1d_array(x, "x")
    y_arr = _as_1d_array(y, "y")
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length.")

    y_pos = np.maximum(y_arr, 0.0)
    if float(np.max(y_pos)) <= 0.0:
        return []

    widths_arr = _resolve_widths(y_pos.size, widths)
    y_centered = y_pos - float(np.median(y_pos))
    y_scaled = y_centered / max(float(np.max(np.abs(y_centered))), 1.0)

    # Ridge map across scales: strong responses remain stable for nearby widths.
    coeffs = _cwt_ricker(y_scaled, widths_arr)
    ridge_response = np.max(np.abs(coeffs), axis=0)
    ridge_noise = _mad_sigma(ridge_response)
    ridge_floor = float(np.median(ridge_response))
    ridge_threshold = ridge_floor + max(min_snr, 0.0) * ridge_noise

    if _find_peaks_cwt is not None:
        cwt_candidates = _find_peaks_cwt(
            y_scaled,
            widths_arr,
            min_snr=max(min_snr, 0.5),
            noise_perc=20,
        )
        if cwt_candidates is None:
            cwt_candidates = []
        candidate_idx = np.asarray(cwt_candidates, dtype=int)
    else:
        candidate_idx = np.asarray([], dtype=int)

    if candidate_idx.size == 0:
        fallback, _ = find_peaks(
            ridge_response,
            height=ridge_threshold,
            distance=max(1, int(np.min(widths_arr))),
        )
        candidate_idx = np.asarray(fallback, dtype=int)

    candidate_idx = candidate_idx[(candidate_idx >= 0) & (candidate_idx < y_pos.size)]
    candidate_idx = np.unique(candidate_idx)
    if candidate_idx.size == 0:
        return []

    # Keep only ridge-consistent candidates and order by ridge prominence.
    keep = candidate_idx[ridge_response[candidate_idx] >= ridge_threshold]
    if keep.size == 0:
        return []

    keep = np.asarray(
        sorted(keep.tolist(), key=lambda idx: float(ridge_response[idx]), reverse=True),
        dtype=int,
    )

    # Snap CWT ridge candidates to true local maxima to avoid degenerate width/prominence
    # estimates (common source of PeakPropertyWarning in noisy signals).
    keep_original = keep.copy()
    local_peaks, local_props = find_peaks(y_pos, prominence=0.0)
    if local_peaks.size > 0:
        prominence_by_idx = {
            int(local_peaks[i]): float(local_props["prominences"][i])
            for i in range(local_peaks.size)
        }
        search_radius = max(1, int(np.ceil(np.min(widths_arr))))
        snapped: list[int] = []
        seen: set[int] = set()
        for idx in keep:
            lo = max(1, int(idx) - search_radius)
            hi = min(y_pos.size - 2, int(idx) + search_radius)
            in_window = (local_peaks >= lo) & (local_peaks <= hi)
            if np.any(in_window):
                candidates = local_peaks[in_window]
            else:
                # Fallback to the nearest detected local maximum when no candidate
                # falls inside the target window.
                nearest = int(np.argmin(np.abs(local_peaks - int(idx))))
                candidates = local_peaks[nearest : nearest + 1]
            best = int(candidates[np.argmax(ridge_response[candidates])])
            if best in seen:
                continue
            if prominence_by_idx.get(best, 0.0) <= np.finfo(float).eps:
                continue
            seen.add(best)
            snapped.append(best)

        keep = np.asarray(snapped, dtype=int) if snapped else keep_original
    else:
        keep = keep_original

    keep = keep[(keep > 0) & (keep < y_pos.size - 1)]
    keep = np.unique(keep)
    if keep.size == 0:
        return []

    width_pack = peak_widths(y_pos, keep, rel_height=0.5)
    seeds: list[PeakSeed] = []
    for i, idx in enumerate(keep):
        shape_data = (
            width_pack[0][i : i + 1],
            width_pack[1][i : i + 1],
            width_pack[2][i : i + 1],
            width_pack[3][i : i + 1],
        )
        shape = analyze_peak_shape(x_arr, y_pos, int(idx), shape_data)
        omega = float(shape["omega"])
        if (not np.isfinite(omega)) or omega <= 0.0:
            dx = float(np.median(np.diff(x_arr)))
            omega = max(dx * max(2.0, float(np.min(widths_arr))), np.finfo(float).eps)
        seeds.append(
            PeakSeed(
                index=int(idx),
                Tm=float(x_arr[idx]),
                Im=float(y_pos[idx]),
                fwhm=omega,
                symmetry=float(shape["mu_g"]),
            )
        )

    return sorted(seeds, key=lambda seed: seed.index)
