"""Public data schemas for configuration, results, and API metadata."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_serializer


def _to_numpy(value: Any) -> np.ndarray:
    """Convert list-like JSON payloads to ``numpy.ndarray``."""
    if isinstance(value, list):
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        return value
    raise ValueError(f"Expected list or numpy.ndarray, got {type(value)}")


def _to_json_compatible(value: Any) -> Any:
    """Recursively convert numpy containers to JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    return value


NumpyArray = Annotated[np.ndarray, BeforeValidator(_to_numpy)]


class PygcdBaseModel(BaseModel):
    """Base schema shared by all public models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("*", when_used="json", check_fields=False)
    def _serialize_numpy(self, value: Any) -> Any:
        """Serialize numpy arrays for JSON output in Pydantic v2 style."""
        return _to_json_compatible(value)


class PygcdImmutableModel(PygcdBaseModel):
    """Immutable schema for result payloads."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class UncertaintyOptions(PygcdBaseModel):
    r"""
    Configuration for uncertainty propagation in TL deconvolution.

    Attributes
    ----------
    enabled : bool
        Enables uncertainty evaluation when ``True``.
    include_parameter_covariance : bool
        Includes covariance from the Jacobian-based local linearization.
    noise_pct : float
        Type-A noise contribution as percentage of fitted maximum intensity.
    noise_from_residuals : bool
        Uses residual statistics to estimate Type-A noise.
    calibration_pct : float
        Type-B calibration contribution in percent.
    heating_rate_pct : float
        Relative uncertainty in heating rate :math:`\beta`.
    reader_drift_pct : float
        Type-B reader drift contribution in percent.
    uc_metric_min_rel_intensity : float
        Minimum relative fitted intensity used to define ROI for ``u_c`` metrics.
    uc_metric_noise_sigma_factor : float
        Residual-noise multiplier used in ROI definition.
    correlations : dict[str, float]
        Correlation coefficients keyed as ``sourceA:sourceB``.
    export_report : bool
        Whether to attach uncertainty report payloads to fit outputs.
    validation_mode : {"none", "monte_carlo", "bootstrap"}
        Validation mode for uncertainty cross-checking.
    n_validation_samples : int
        Number of validation samples/iterations.
    validation_seed : int | None
        Random seed for reproducible validation runs.
    threshold_fom : float
        Recommended upper bound for figure-of-merit.
    threshold_uc_global : float
        Recommended upper bound for global uncertainty criterion.
    threshold_uc_p95 : float
        Recommended upper bound for 95th percentile uncertainty criterion.
    """

    enabled: bool = Field(True, description="Enable uncertainty evaluation.")
    include_parameter_covariance: bool = Field(
        True,
        description="Include parameter-covariance contribution from Jacobian.",
    )
    noise_pct: float = Field(
        0.0,
        ge=0.0,
        description="Type-A noise contribution as percentage of max fitted signal.",
    )
    noise_from_residuals: bool = Field(
        False,
        description="If true, estimate Type-A noise from residual standard deviation.",
    )
    calibration_pct: float = Field(
        0.0,
        ge=0.0,
        description="Type-B calibration contribution as relative percentage.",
    )
    heating_rate_pct: float = Field(
        0.0,
        ge=0.0,
        description="Type-B heating-rate uncertainty as relative percentage of beta.",
    )
    reader_drift_pct: float = Field(
        0.0,
        ge=0.0,
        description="Type-B reader-drift contribution as percentage of max fitted signal.",
    )
    uc_metric_min_rel_intensity: float = Field(
        0.05,
        ge=0.0,
        description=(
            "Relative fitted-signal threshold (fraction of Imax) used to define "
            "ROI channels for u_c summary metrics."
        ),
    )
    uc_metric_noise_sigma_factor: float = Field(
        3.0,
        ge=0.0,
        description=(
            "Noise-floor multiplier used to define ROI channels for u_c summary metrics "
            "when residual noise is available."
        ),
    )
    correlations: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional source correlation coefficients with keys `sourceA:sourceB`.",
    )
    export_report: bool = Field(
        True,
        description="If true, attach technical uncertainty report to fit results.",
    )
    validation_mode: Literal["none", "monte_carlo", "bootstrap"] = Field(
        "none",
        description="Cross-check mode for local linearization uncertainty.",
    )
    n_validation_samples: int = Field(
        0,
        ge=0,
        description="Number of samples/iterations used by uncertainty cross-check validation.",
    )
    validation_seed: Optional[int] = Field(
        42,
        description="Random seed for uncertainty cross-check validation.",
    )
    threshold_fom: float = Field(5.0, ge=0.0, description="Recommended maximum FOM (%).")
    threshold_uc_global: float = Field(
        2.0,
        ge=0.0,
        description="Recommended maximum global uncertainty criterion (%).",
    )
    threshold_uc_p95: float = Field(
        5.0,
        ge=0.0,
        description="Recommended maximum 95th-percentile uncertainty criterion (%).",
    )


class FitOptions(PygcdBaseModel):
    """
    Solver and uncertainty configuration for multi-peak fitting.

    Attributes
    ----------
    local_optimizer : {"trf", "dogbox", "lm"}, default="trf"
        Local least-squares method passed to
        ``scipy.optimize.least_squares``:

        - ``"trf"`` — Trust Region Reflective.  Supports box bounds and all
          robust losses.  **Recommended for most fits.**
        - ``"dogbox"`` — Dogleg method.  Supports bounds and robust losses;
          faster on well-conditioned small problems.
        - ``"lm"`` — Levenberg-Marquardt.  Does **not** support box bounds
          natively (soft penalties are used) and only supports
          ``loss="linear"``.  Use for simple, unconstrained problems.
    max_nfev : int | None, optional
        Maximum number of objective function evaluations.  ``None`` lets
        SciPy choose a default (≈ 100 × number of parameters).
    ftol : float, default=1e-8
        Relative cost-function reduction tolerance for convergence.
    xtol : float, default=1e-8
        Relative parameter-step size tolerance for convergence.
    gtol : float, default=1e-8
        Gradient norm tolerance for convergence.
    uncertainty : UncertaintyOptions | None, optional
        If provided and ``uncertainty.enabled=True``, a full uncertainty
        budget is computed after fitting and attached to ``MultiFitResult``.

    Examples
    --------
    >>> import tldecpy as tl
    >>> opts = tl.FitOptions(local_optimizer="trf")
    >>> opts_uc = tl.FitOptions(
    ...     local_optimizer="trf",
    ...     uncertainty=tl.UncertaintyOptions(
    ...         enabled=True,
    ...         noise_pct=1.0,
    ...         calibration_pct=0.5,
    ...     ),
    ... )
    """

    local_optimizer: Literal["trf", "dogbox", "lm"] = Field(
        "trf",
        description="Local least-squares optimizer.",
    )
    max_nfev: Optional[int] = Field(None, description="Maximum number of function evaluations.")
    ftol: float = Field(1e-8, description="Cost function tolerance.")
    xtol: float = Field(1e-8, description="Parameter tolerance.")
    gtol: float = Field(1e-8, description="Gradient tolerance.")
    uncertainty: Optional[UncertaintyOptions] = Field(
        None,
        description="Optional uncertainty-budget configuration.",
    )


class RobustOptions(PygcdBaseModel):
    r"""
    Robust loss and weighting options for TL curve fitting.

    Attributes
    ----------
    loss : {"linear", "soft_l1", "huber", "cauchy", "arctan", "tukey"}, default="linear"
        Residual loss function :math:`\rho(r)` applied to the normalised
        residuals before summing the cost:

        - ``"linear"`` — :math:`r^2` (ordinary least squares, default).
        - ``"soft_l1"`` — :math:`2(\sqrt{1+r^2}-1)`, smooth L1.
        - ``"huber"`` — quadratic for :math:`|r|<1`, linear otherwise.
        - ``"cauchy"`` — :math:`\ln(1+r^2)`, heavy-tailed downweighting.
        - ``"arctan"`` — :math:`\arctan(r^2)`, bounded loss.
        - ``"tukey"`` — biweight (requires ``loss_param``).

        ``"lm"`` optimizer supports only ``"linear"``.
    f_scale : float, default=1.0
        Residual scale parameter.  Residuals with absolute value < ``f_scale``
        are treated as inliers (OLS regime); larger residuals are robustly
        downweighted.  Set to approximately the noise floor of the detector
        in the same units as ``y``.
    loss_param : float | None, optional
        Extra loss parameter used by Tukey biweight (the Tukey constant ``c``).
        Ignored for other losses.
    weights : {"none", "poisson"}, default="none"
        Residual weighting strategy.

        - ``"none"`` — all channels equally weighted.
        - ``"poisson"`` — each residual is divided by :math:`\sqrt{I_i}`,
          giving inverse-variance weighting for Poisson (counting) noise.
    multi_start : int, default=0
        Number of randomised restarts with ±10 % parameter perturbation.
        The best-cost solution is returned.  ``0`` disables restarts.
    ci_bootstrap : bool, default=False
        If ``True``, compute 95 % bootstrap confidence intervals via residual
        resampling.  Results are stored in ``PeakResult.ci_95``.
        Requires ``n_bootstrap`` extra full solves — can be slow.
    n_bootstrap : int, default=100
        Number of bootstrap iterations when ``ci_bootstrap=True``.
        Use 200–500 for publication-quality intervals.

    Examples
    --------
    >>> import tldecpy as tl
    >>> # Robust fit with Poisson weighting
    >>> opts = tl.RobustOptions(
    ...     loss="soft_l1",
    ...     f_scale=5.0,
    ...     weights="poisson",
    ... )
    >>> # Bootstrap confidence intervals
    >>> opts_ci = tl.RobustOptions(
    ...     loss="soft_l1",
    ...     f_scale=5.0,
    ...     ci_bootstrap=True,
    ...     n_bootstrap=200,
    ... )
    """

    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan", "tukey"] = Field(
        "linear",
        description="Robust loss function.",
    )
    f_scale: float = Field(1.0, description="Residual scale parameter.")
    loss_param: Optional[float] = Field(
        None, description="Extra loss parameter (for example Tukey c)."
    )
    weights: Literal["none", "poisson"] = Field("none", description="Residual weighting strategy.")
    multi_start: int = Field(0, description="Number of randomized restarts.")
    ci_bootstrap: bool = Field(False, description="Compute confidence intervals via bootstrap.")
    n_bootstrap: int = Field(100, description="Bootstrap iterations.")


class SimulationOptions(PygcdBaseModel):
    """
    ODE simulation options for synthetic TL curve generation.

    Attributes
    ----------
    method : str
        ``scipy.integrate.solve_ivp`` method name.
    rtol : float
        Relative integration tolerance.
    atol : float
        Absolute integration tolerance.
    points : int
        Number of output sample points.
    """

    method: str = Field("LSODA", description="`solve_ivp` method.")
    rtol: float = Field(1e-6, description="Relative tolerance.")
    atol: float = Field(1e-9, description="Absolute tolerance.")
    points: int = Field(1000, description="Number of output points.")


class BoundsSpec(PygcdBaseModel):
    """
    Bounds definition for a scalar parameter.

    Attributes
    ----------
    min : float
        Lower bound.
    max : float
        Upper bound.
    """

    min: float = -np.inf
    max: float = np.inf


class PeakSpec(PygcdBaseModel):
    r"""
    Input specification for one TL peak component.

    Attributes
    ----------
    name : str | None
        Human-readable component identifier (e.g. ``"P1"``).  Used as a
        prefix in parameter names within ``MultiFitResult.hit_bounds``
        and ``PeakResult.uncertainties``.  Auto-assigned as ``"P<n>"``
        if omitted.
    model : str
        Kinetic model key.  Accepted values include canonical keys
        (``"fo_rq"``, ``"fo_rb"``, ``"go_kg"``, ``"otor_lw"``,
        ``"cont_gauss"`` …), family short-names (``"fo"``, ``"go"``), and
        registered aliases.  Call ``tl.list_models()`` for the full list.
    init : dict[str, float]
        Initial parameter values passed to the optimizer.  Required keys
        depend on the model family:

        - FO / SO / GO / MO / OTOR: ``"Tm"`` (K), ``"Im"`` (counts),
          ``"E"`` (eV).  GO also requires ``"b"`` (1–2); OTOR requires
          ``"R"`` (0–1); MO requires ``"alpha"`` (0–1).
        - Continuous (``"cont_gauss"``, ``"cont_exp"``): ``"Tn"`` (K),
          ``"In"`` (counts), ``"E0"`` (eV), ``"sigma"`` (eV).
          Legacy aliases ``"Tm"``, ``"Im"``, ``"E"`` are accepted.

        Missing keys receive model-specific defaults inside the fitter.
    bounds : dict[str, tuple[float, float]] | None, optional
        Parameter bounds as ``{name: (lower, upper)}``.  If omitted,
        automatic bounds are derived from ``init`` values inside the fitter.
        Explicit bounds override automatic ones.
    fixed : dict[str, float | bool] | None, optional
        Parameters held constant during optimisation.  Two forms:

        - ``{"b": 1.8}`` — ``b`` is fixed to 1.8 regardless of ``init``.
        - ``{"b": True}`` — ``b`` is fixed to ``init["b"]``.
        - ``{"b": False}`` — ``b`` is free (same as omitting it).

    Examples
    --------
    >>> import tldecpy as tl
    >>> peak = tl.PeakSpec(
    ...     name="P1",
    ...     model="fo_rq",
    ...     init={"Tm": 450.0, "Im": 5000.0, "E": 1.2},
    ...     bounds={"Tm": (400.0, 500.0), "E": (0.8, 2.0)},
    ... )
    >>> go_peak = tl.PeakSpec(
    ...     model="go_kg",
    ...     init={"Tm": 490.0, "Im": 3000.0, "E": 1.5, "b": 1.8},
    ...     fixed={"b": True},          # hold kinetic order fixed
    ... )
    """

    name: Optional[str] = Field(None, description="Unique component name.")
    model: str = Field(..., description="Model key (for example `fo_rq`, `go_kg`, `otor_lw`).")
    init: Dict[str, float] = Field(..., description="Initial parameter values.")
    bounds: Optional[Dict[str, Tuple[float, float]]] = Field(
        None,
        description="Parameter bounds as `(min, max)`.",
    )
    fixed: Optional[Dict[str, float]] = Field(
        None,
        description="Fixed parameters as `{parameter: value}`.",
    )


class ContinuousPeakSpec(PygcdBaseModel):
    r"""
    Input specification for one continuous trap-distribution peak.

    Attributes
    ----------
    name : str | None
        Component identifier.
    model : str
        Continuous model key.
    Tn : float
        Characteristic temperature :math:`T_N` in kelvin.
    In : float
        Characteristic intensity :math:`I_N`.
    E0 : float
        Characteristic activation energy :math:`E_0` in eV.
    sigma : float
        Distribution width in eV.
    """

    name: Optional[str] = Field(None, description="Unique component name.")
    model: Literal[
        "cont_gauss",
        "cont_exp",
        "cont_lorentz",
        "continuous_gaussian",
        "continuous_exponential",
    ] = Field(
        ...,
        description="Continuous trap model key.",
    )
    Tn: float = Field(..., gt=0.0, description="Characteristic temperature `T_N` (K).")
    In: float = Field(..., ge=0.0, description="Characteristic intensity `I_N`.")
    E0: float = Field(..., gt=0.0, description="Characteristic activation energy `E_0` (eV).")
    sigma: float = Field(..., gt=0.0, description="Distribution width `sigma` (eV).")


class BackgroundSpec(PygcdBaseModel):
    """
    Input specification for background contribution in TL fits.

    Attributes
    ----------
    type : {"linear", "exponential", "none"}
        Background functional form.
    init : dict[str, float] | None
        Initial parameter values.
    bounds : dict[str, tuple[float, float]] | None
        Optional parameter bounds.
    fixed : dict[str, float] | None
        Optional fixed parameter values.
    """

    type: Literal["linear", "exponential", "none"] = Field(
        "none", description="Background model type."
    )
    init: Optional[Dict[str, float]] = Field(None, description="Initial parameter values.")
    bounds: Optional[Dict[str, Tuple[float, float]]] = Field(
        None,
        description="Parameter bounds as `(min, max)`.",
    )
    fixed: Optional[Dict[str, float]] = Field(
        None,
        description="Fixed parameters as `{parameter: value}`.",
    )


class Metrics(PygcdImmutableModel):
    """
    Goodness-of-fit and uncertainty summary metrics.

    Attributes
    ----------
    FOM : float
        Figure of Merit in percent.
    R2 : float
        Coefficient of determination.
    RCS : float
        Reduced chi-square surrogate.
    SSR : float
        Sum of squared residuals.
    AIC : float
        Akaike information criterion.
    BIC : float
        Bayesian information criterion.
    uc_global : float | None
        Global combined uncertainty criterion in percent.
    uc_max : float | None
        Maximum channel-wise combined uncertainty in percent.
    uc_p95 : float | None
        95th percentile channel-wise combined uncertainty in percent.
    contrib_E : float | None
        Contribution associated with activation-energy parameters.
    contrib_Tm : float | None
        Contribution associated with characteristic-temperature parameters.
    contrib_Im : float | None
        Contribution associated with characteristic-intensity parameters.
    contrib_bg : float | None
        Contribution associated with background parameters.
    """

    FOM: float = Field(..., description="Figure of Merit in percent.")
    R2: float = Field(..., description="Coefficient of determination.")
    RCS: float = Field(..., description="Reduced chi-square surrogate.")
    SSR: float = Field(..., description="Sum of squared residuals.")
    AIC: float = Field(float("inf"), description="Akaike Information Criterion.")
    BIC: float = Field(float("inf"), description="Bayesian Information Criterion.")
    uc_global: Optional[float] = Field(
        None,
        description="Global combined uncertainty criterion in percent.",
    )
    uc_max: Optional[float] = Field(
        None,
        description="Maximum channel-wise combined uncertainty in percent.",
    )
    uc_p95: Optional[float] = Field(
        None,
        description="95th percentile of channel-wise combined uncertainty.",
    )
    contrib_E: Optional[float] = Field(
        None,
        description="Global uncertainty contribution associated with activation-energy parameters.",
    )
    contrib_Tm: Optional[float] = Field(
        None,
        description="Global uncertainty contribution associated with characteristic-temperature parameters.",
    )
    contrib_Im: Optional[float] = Field(
        None,
        description="Global uncertainty contribution associated with characteristic-intensity parameters.",
    )
    contrib_bg: Optional[float] = Field(
        None,
        description="Global uncertainty contribution associated with background parameters.",
    )


class PeakResult(PygcdImmutableModel):
    r"""
    Fitted result for one deconvolved TL peak.

    Attributes
    ----------
    name : str
        Peak/component label.
    model : str
        Kinetic model key used in fitting.
    params : dict[str, float]
        Estimated parameter values.
    y_hat : numpy.ndarray
        Fitted peak contribution :math:`\hat{I}_{\text{peak}}(T)`.
    area : float
        Integrated area under the fitted peak.
    ci_95 : dict[str, tuple[float, float]] | None
        Optional 95% confidence interval bounds by parameter.
    uncertainties : dict[str, float]
        Parameter standard uncertainties.
    """

    name: str
    model: str
    params: Dict[str, float]
    y_hat: NumpyArray = Field(description="Fitted peak contribution.")
    area: float
    ci_95: Optional[Dict[str, Tuple[float, float]]] = None
    uncertainties: Dict[str, float] = Field(default_factory=dict)


class BGResult(PygcdImmutableModel):
    """
    Fitted result for the background component.

    Attributes
    ----------
    type : str
        Background model type.
    params : dict[str, float]
        Estimated background parameters.
    y_hat : numpy.ndarray
        Fitted background signal.
    uncertainties : dict[str, float]
        Standard uncertainties for background parameters.
    """

    type: str
    params: Dict[str, float]
    y_hat: NumpyArray
    uncertainties: Dict[str, float] = Field(default_factory=dict)


class MultiFitResult(PygcdImmutableModel):
    r"""
    Aggregated output of a multi-peak TL deconvolution.

    Attributes
    ----------
    peaks : list[PeakResult]
        Fitted peak components.
    background : BGResult | None
        Optional fitted background component.
    y_obs : numpy.ndarray
        Observed intensity :math:`I(T)`.
    y_hat_total : numpy.ndarray
        Total fitted intensity :math:`\hat{I}(T)`.
    residuals : numpy.ndarray
        Residual vector :math:`I(T)-\hat{I}(T)`.
    metrics : Metrics
        Fit-quality and uncertainty metrics.
    converged : bool
        Optimizer convergence flag.
    message : str
        Optimizer termination message.
    n_iter : int
        Number of function evaluations/iterations.
    hit_bounds : dict[str, bool]
        Indicates which parameters touched optimization bounds.
    jac_cond : float
        Jacobian condition number estimate.
    loss_used : str
        Robust loss function effectively used by the solver.
    uc_curve : numpy.ndarray | None
        Channel-wise combined uncertainty criterion in percent.
    uncertainty_budget : dict[str, float]
        Global uncertainty contribution summary.
    uncertainty_report : dict[str, Any] | None
        Optional technical uncertainty report payload.
    uncertainty_validation : dict[str, Any] | None
        Optional validation diagnostics (MC/bootstrap vs local linearization).
    """

    peaks: List[PeakResult]
    background: Optional[BGResult] = None
    y_obs: NumpyArray = Field(description="Observed intensity.")
    y_hat_total: NumpyArray = Field(description="Total fitted signal.")
    residuals: NumpyArray
    metrics: Metrics
    converged: bool
    message: str
    n_iter: int
    hit_bounds: Dict[str, bool] = Field(default_factory=dict)
    jac_cond: float = 0.0
    loss_used: str = "linear"
    uc_curve: Optional[NumpyArray] = Field(
        None,
        description="Channel-wise combined uncertainty criterion `u_c(T)` in percent.",
    )
    uncertainty_budget: Dict[str, float] = Field(
        default_factory=dict,
        description="Global uncertainty-budget contributions in percent.",
    )
    uncertainty_report: Optional[Dict[str, Any]] = Field(
        None,
        description="Technical report payload (summary + table) for uncertainty analysis.",
    )
    uncertainty_validation: Optional[Dict[str, Any]] = Field(
        None,
        description="Validation diagnostics comparing local linearization vs MC/bootstrap uncertainty.",
    )


class FitResult(PygcdImmutableModel):
    """
    Legacy single-peak result schema retained for compatibility.

    Attributes
    ----------
    params : dict[str, float]
        Estimated parameters.
    cov : numpy.ndarray | None
        Optional covariance matrix.
    metrics : dict[str, float]
        Legacy metric payload.
    y_hat : numpy.ndarray
        Fitted signal.
    residuals : numpy.ndarray
        Residual signal.
    converged : bool
        Convergence flag.
    message : str
        Solver termination message.
    model_type : str
        Model key used during fitting.
    """

    params: Dict[str, float]
    cov: Optional[NumpyArray]
    metrics: Dict[str, float]
    y_hat: NumpyArray
    residuals: NumpyArray
    converged: bool
    message: str
    model_type: str


class SimulationResult(PygcdImmutableModel):
    r"""
    Output schema for ODE-based TL simulations.

    Attributes
    ----------
    T : numpy.ndarray
        Temperature grid in kelvin.
    I : numpy.ndarray
        Simulated intensity signal.
    states : dict[str, numpy.ndarray] | None
        Optional ODE state trajectories keyed by state name.
    time : numpy.ndarray | None
        Optional integration-time vector.
    """

    T: NumpyArray
    I: NumpyArray  # noqa: E741 - Physical TL intensity notation I(T)
    states: Optional[Dict[str, NumpyArray]] = None
    time: Optional[NumpyArray] = None


class VersionInfo(PygcdImmutableModel):
    """
    Package and API version information.

    Attributes
    ----------
    version : str
        Package version.
    api_version : str
        Public API version.
    build : str | None
        Optional build tag.
    """

    version: str
    api_version: str
    build: Optional[str] = None


class ErrorDetail(PygcdImmutableModel):
    """
    Serializable error payload for API integrations.

    Attributes
    ----------
    error_code : str
        Stable error identifier.
    message : str
        Human-readable error message.
    hint : str | None
        Optional remediation hint.
    """

    error_code: str
    message: str
    hint: Optional[str] = None
