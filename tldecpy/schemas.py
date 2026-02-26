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
    local_optimizer : {"trf", "dogbox", "lm"}
        Local least-squares method used by SciPy.
    max_nfev : int | None
        Maximum number of objective function evaluations.
    ftol : float
        Relative tolerance for objective reduction.
    xtol : float
        Relative tolerance for parameter-step convergence.
    gtol : float
        Gradient norm tolerance.
    uncertainty : UncertaintyOptions | None
        Optional uncertainty propagation settings.
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
    """
    Robust loss and weighting options for TL curve fitting.

    Attributes
    ----------
    loss : {"linear", "soft_l1", "huber", "cauchy", "arctan", "tukey"}
        Residual loss function.
    f_scale : float
        Scale parameter for robust losses.
    loss_param : float | None
        Optional extra loss parameter (for example Tukey constant).
    weights : {"none", "poisson"}
        Residual weighting strategy.
    multi_start : int
        Number of randomized restarts.
    ci_bootstrap : bool
        Enables bootstrap confidence intervals when ``True``.
    n_bootstrap : int
        Number of bootstrap iterations.
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
        Component identifier.
    model : str
        Kinetic model key (for example ``fo_rq``, ``go_kg``, ``otor_lw``).
    init : dict[str, float]
        Initial parameter values. Typical physical parameters include
        :math:`T_m` (K), :math:`I_m`, :math:`E` (eV), and model-specific terms.
    bounds : dict[str, tuple[float, float]] | None
        Optional parameter bounds as ``(min, max)``.
    fixed : dict[str, float] | None
        Parameters fixed to constant numeric values.
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
