"""Public API for the ``tldecpy`` package."""

from __future__ import annotations

from .data.refglow import list_refglow, load_refglow
from .errors import (
    ConvergenceError,
    DatasetError,
    DomainError,
    ModelKeyError,
    PygcdError,
    TLDecPyError,
    TypingError,
)
from .fit.automator import iterative_deconvolution
from .fit.detection import detect_peaks_cwt
from .fit.init import autoinit_multi, pick_peaks, preprocess
from .fit.multi import fit_multi
from .fit.solvers import fit_single_peak
from .models.registry import get_model, list_models
from .schemas import (
    BGResult,
    BackgroundSpec,
    BoundsSpec,
    ErrorDetail,
    FitOptions,
    FitResult,
    Metrics,
    MultiFitResult,
    PeakResult,
    PeakSpec,
    RobustOptions,
    SimulationOptions,
    SimulationResult,
    UncertaintyOptions,
    VersionInfo,
)
from .simulate.core import simulate
from .version import __version__, get_version_info

__all__ = [
    "fit_single_peak",
    "fit_multi",
    "iterative_deconvolution",
    "simulate",
    "autoinit_multi",
    "pick_peaks",
    "detect_peaks_cwt",
    "preprocess",
    "list_refglow",
    "load_refglow",
    "list_models",
    "get_model",
    "get_version_info",
    "__version__",
    "PeakSpec",
    "BackgroundSpec",
    "RobustOptions",
    "FitOptions",
    "UncertaintyOptions",
    "SimulationOptions",
    "BoundsSpec",
    "FitResult",
    "MultiFitResult",
    "PeakResult",
    "BGResult",
    "SimulationResult",
    "Metrics",
    "VersionInfo",
    "ErrorDetail",
    "TLDecPyError",
    "PygcdError",
    "ConvergenceError",
    "ModelKeyError",
    "DatasetError",
    "DomainError",
    "TypingError",
]
