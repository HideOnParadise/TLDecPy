"""Multi-peak fitting orchestration."""

from __future__ import annotations

import inspect as _inspect
from typing import Literal, Optional, cast

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, least_squares

from tldecpy.fit.objective import calculate_metrics
from tldecpy.fit.robust import get_loss_func
from tldecpy.fit.uncertainty import (
    absolute_to_relative_percent,
    build_uncertainty_report,
    bootstrap_uncertainty_curve,
    combine_uncertainty_sources,
    compute_absolute_uncertainty_from_covariance,
    compute_global_contribution_percent,
    monte_carlo_uncertainty_curve,
    compute_parameter_covariance,
    compute_sensitivity_matrix,
    compute_uc_global,
)
from tldecpy.fit.weights import calculate_weights
from tldecpy.models.background import BG_REGISTRY
from tldecpy.models.registry import get_model, get_order_from_key
from tldecpy.schemas import (
    BGResult,
    BackgroundSpec,
    FitOptions,
    Metrics,
    MultiFitResult,
    PeakResult,
    PeakSpec,
    RobustOptions,
    UncertaintyOptions,
)
from tldecpy.utils.ci import bootstrap_ci
from tldecpy.utils.jacobian import check_conditioning, check_hit_bounds

UNCERTAINTY_METRIC_MIN_REL_INTENSITY = 5e-2
UNCERTAINTY_METRIC_NOISE_SIGMA_FACTOR = 3.0


def _is_optimizer_loss_compatible(optimizer: str, loss: str) -> bool:
    """
    Validate optimizer/loss compatibility for scipy.optimize.least_squares.

    - ``trf`` and ``dogbox`` support robust losses and callable losses.
    - ``lm`` supports only ``linear`` loss.
    """
    if optimizer in {"trf", "dogbox"}:
        return True
    if optimizer == "lm":
        return loss == "linear"
    return False


def calculate_parameter_uncertainties(
    jac: Optional[np.ndarray],
    residuals: np.ndarray,
    n_params: int,
) -> np.ndarray:
    """Estimate parameter standard errors from the Jacobian."""
    if jac is None or jac.size == 0 or n_params <= 0:
        return np.full(max(n_params, 0), np.nan, dtype=float)

    dof = max(len(residuals) - n_params, 1)
    cov_matrix = compute_parameter_covariance(jac=jac, residuals=residuals, dof=dof)
    if cov_matrix is None:
        return np.full(n_params, np.nan, dtype=float)

    diag = np.diag(cov_matrix)
    diag = np.where(diag >= 0.0, diag, np.nan)
    return np.sqrt(diag)


def _uncertainty_metric_mask(
    y_fit: np.ndarray,
    uc_curve: np.ndarray,
    min_rel_intensity: float = UNCERTAINTY_METRIC_MIN_REL_INTENSITY,
    abs_floor: float | None = None,
) -> np.ndarray:
    """
    Build a stable mask for relative-uncertainty summary metrics.

    Relative uncertainty diverges in channels where fitted intensity tends to
    zero. For summary metrics (u_c,max and u_c,p95) we focus on channels with
    meaningful signal amplitude and fall back to all finite channels if needed.
    """
    y_arr = np.asarray(y_fit, dtype=float).reshape(-1)
    uc_arr = np.asarray(uc_curve, dtype=float).reshape(-1)
    if y_arr.size == 0 or uc_arr.size != y_arr.size:
        return np.zeros(0, dtype=bool)

    finite = np.isfinite(y_arr) & np.isfinite(uc_arr)
    if not np.any(finite):
        return finite

    scale = max(float(np.nanmax(np.abs(y_arr[finite]))), 1.0)
    threshold = min_rel_intensity * scale
    if abs_floor is not None and np.isfinite(abs_floor) and abs_floor > 0.0:
        threshold = max(threshold, float(abs_floor))
    roi = finite & (np.abs(y_arr) >= threshold)
    if np.any(roi):
        return roi
    return finite


def _frequency_factor(E: float, Tm: float, beta: float) -> float | None:
    """Compute frequency factor s for non-OTOR models."""
    if E <= 0.0 or Tm <= 0.0 or beta <= 0.0:
        return None

    k_eV = 8.617333e-5
    exponent = E / (k_eV * Tm)
    if exponent >= 100:
        return None

    s = (beta * E / (k_eV * Tm**2)) * np.exp(exponent)
    if s <= 0.0 or not np.isfinite(s):
        return None
    return float(s)


def _frequency_factor_sigma(
    s: float,
    E: float,
    Tm: float,
    sigma_E: float | None,
    sigma_Tm: float | None,
) -> float | None:
    """Propagate uncertainty for frequency factor s."""
    if (
        sigma_E is None
        or sigma_Tm is None
        or not np.isfinite(sigma_E)
        or not np.isfinite(sigma_Tm)
        or sigma_E <= 0.0
        or sigma_Tm <= 0.0
        or E <= 0.0
        or Tm <= 0.0
    ):
        return None

    k_eV = 8.617333e-5
    ds_dE = s / E * (1.0 + E / (k_eV * Tm))
    ds_dTm = -s * (2.0 / Tm + E / (k_eV * Tm**2))
    sigma_s = np.sqrt((ds_dE * sigma_E) ** 2 + (ds_dTm * sigma_Tm) ** 2)
    if not np.isfinite(sigma_s):
        return None
    return float(sigma_s)


class MultiFitter:
    """Stateful multi-component fitter."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        beta: float = 1.0,
        robust: Optional[RobustOptions] = None,
        options: Optional[FitOptions] = None,
    ) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.beta = float(beta)
        self.robust = robust if robust is not None else RobustOptions.model_validate({})
        self.options = options if options is not None else FitOptions.model_validate({})
        self.uncertainty: UncertaintyOptions = (
            self.options.uncertainty
            if self.options.uncertainty is not None
            else UncertaintyOptions.model_validate({})
        )

        self.peak_meta: list[dict] = []
        self.bg_meta: dict | None = None
        self.x0: list[float] = []
        self.bounds_low: list[float] = []
        self.bounds_high: list[float] = []
        self.param_names_flat: list[str] = []

    def _resolve_param(
        self,
        name: str,
        user_init: dict[str, float],
        user_bounds: dict[str, tuple[float, float]],
        user_fixed: dict,
        default: float,
        default_bounds: tuple[float, float],
        prefix: str,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        if name in user_fixed:
            fixed_value = user_fixed[name]
            if isinstance(fixed_value, bool):
                if fixed_value:
                    fixed_value = user_init.get(name, default)
                else:
                    fixed_value = None
            if fixed_value is not None:
                return None, None, None, float(fixed_value)

        value = float(user_init.get(name, default))

        if name in user_bounds:
            bounds = user_bounds[name]
            lb, ub = float(bounds[0]), float(bounds[1])
        elif name in {"Tm", "Tn"} and name in user_init:
            center = float(user_init["Tm"])
            if name == "Tn":
                center = float(user_init["Tn"])
            lb = max(default_bounds[0], center - 40.0)
            ub = min(default_bounds[1], center + 40.0)
        else:
            lb, ub = default_bounds

        if lb >= ub:
            ub = lb + 1e-5

        value = float(np.clip(value, lb, ub))
        self.param_names_flat.append(f"{prefix}_{name}")
        return value, lb, ub, None

    def add_peak(self, spec: PeakSpec) -> None:
        """Register a peak component in the optimization problem."""
        model_func = get_model(spec.model)
        order_type = get_order_from_key(spec.model)
        required_params = ["Im", "E", "Tm"]
        if order_type == "continuous":
            required_params = ["Tn", "In", "E0", "sigma"]
        if order_type == "go":
            required_params.append("b")
        if order_type == "otor":
            required_params.append("R")
        if order_type == "mix":
            required_params.append("alpha")

        max_y = float(np.max(self.y))
        if max_y <= 1e-9:
            max_y = 1.0

        min_x = float(np.min(self.x))
        max_x = float(np.max(self.x))
        defaults = {
            "Im": max_y / 2.0,
            "E": 1.0,
            "Tm": (min_x + max_x) / 2.0,
            "Tn": (min_x + max_x) / 2.0,
            "In": max_y / 2.0,
            "E0": 1.0,
            "sigma": 0.05,
            "b": 1.5,
            "R": 1e-4,
            "alpha": 0.5,
        }
        bounds_default = {
            "Im": (0.0, max_y * 2.0),
            "E": (0.1, 5.0),
            "Tm": (min_x - 50.0, max_x + 50.0),
            "Tn": (0.0, max_x + 50.0),
            "In": (0.0, max_y * 2.0),
            "E0": (0.0, 5.0),
            "sigma": (0.001, 1.0),
            "b": (1.001, 2.0),
            "R": (1e-7, 10.0),
            "alpha": (0.001, 0.999),
        }

        active: list[str] = []
        fixed: dict[str, float] = {}
        user_init = spec.init
        user_bounds = spec.bounds or {}
        user_fixed = spec.fixed or {}
        if order_type == "continuous":
            # Backward compatibility aliases for single-peak style parameters.
            if "In" not in user_init and "Im" in user_init:
                user_init["In"] = user_init["Im"]
            if "E0" not in user_init and "E" in user_init:
                user_init["E0"] = user_init["E"]
            if "Tn" not in user_init and "Tm" in user_init:
                user_init["Tn"] = user_init["Tm"]

            if "In" not in user_bounds and "Im" in user_bounds:
                user_bounds["In"] = user_bounds["Im"]
            if "E0" not in user_bounds and "E" in user_bounds:
                user_bounds["E0"] = user_bounds["E"]
            if "Tn" not in user_bounds and "Tm" in user_bounds:
                user_bounds["Tn"] = user_bounds["Tm"]

            if "In" not in user_fixed and "Im" in user_fixed:
                user_fixed["In"] = user_fixed["Im"]
            if "E0" not in user_fixed and "E" in user_fixed:
                user_fixed["E0"] = user_fixed["E"]
            if "Tn" not in user_fixed and "Tm" in user_fixed:
                user_fixed["Tn"] = user_fixed["Tm"]
        peak_id = spec.name or f"P{len(self.peak_meta) + 1}"

        for param_name in required_params:
            value, lb, ub, fixed_value = self._resolve_param(
                param_name,
                user_init,
                user_bounds,
                user_fixed,
                defaults[param_name],
                bounds_default[param_name],
                peak_id,
            )
            if fixed_value is not None:
                fixed[param_name] = fixed_value
                continue
            if value is None or lb is None or ub is None:
                raise RuntimeError(f"Unexpected unresolved parameter '{param_name}'.")

            self.x0.append(value)
            self.bounds_low.append(lb)
            self.bounds_high.append(ub)
            active.append(param_name)

        self.peak_meta.append(
            {
                "func": model_func,
                "active": active,
                "fixed": fixed,
                "spec": spec,
                "order_type": order_type,
            }
        )

    def set_background(self, spec: BackgroundSpec) -> None:
        """Register background component in the optimization problem."""
        if spec.type == "none":
            self.bg_meta = None
            return

        func = BG_REGISTRY.get(spec.type)
        if func is None:
            raise ValueError(f"Unknown background model '{spec.type}'.")

        required_params = ["a", "b"]
        if spec.type == "exponential":
            required_params.append("c")

        max_y = float(np.max(self.y))
        if max_y <= 1e-9:
            max_y = 1.0

        defaults = {"a": 0.0, "b": 0.0, "c": 1000.0}
        bounds_default = {"a": (0.0, max_y * 1.5), "b": (-1e6, 1e6), "c": (1.0, 1e5)}

        active: list[str] = []
        fixed: dict[str, float] = {}
        user_init = spec.init or {}
        user_bounds = spec.bounds or {}
        user_fixed = spec.fixed or {}

        for param_name in required_params:
            value, lb, ub, fixed_value = self._resolve_param(
                param_name,
                user_init,
                user_bounds,
                user_fixed,
                defaults[param_name],
                bounds_default[param_name],
                "BG",
            )
            if fixed_value is not None:
                fixed[param_name] = fixed_value
                continue
            if value is None or lb is None or ub is None:
                raise RuntimeError(f"Unexpected unresolved background parameter '{param_name}'.")

            self.x0.append(value)
            self.bounds_low.append(lb)
            self.bounds_high.append(ub)
            active.append(param_name)

        self.bg_meta = {"func": func, "active": active, "fixed": fixed, "spec": spec}

    def _reconstruct(
        self,
        params: np.ndarray,
    ) -> tuple[np.ndarray, list[dict], np.ndarray | None, dict[str, float]]:
        """Build total signal and components from flattened solver parameters."""
        y_total = np.zeros_like(self.x)
        components: list[dict] = []
        idx = 0

        for meta in self.peak_meta:
            param_values = meta["fixed"].copy()
            for name in meta["active"]:
                param_values[name] = float(params[idx])
                idx += 1

            if meta["order_type"] == "continuous":
                args = [
                    self.x,
                    param_values["Tn"],
                    param_values["In"],
                    param_values["E0"],
                    param_values["sigma"],
                ]
            else:
                args = [self.x, param_values["Im"], param_values["E"], param_values["Tm"]]
                if "b" in param_values:
                    args.append(param_values["b"])
                elif "R" in param_values:
                    args.append(param_values["R"])
                elif "alpha" in param_values:
                    args.append(param_values["alpha"])

            kwargs = {"beta": self.beta} if "otor" in meta["spec"].model else {}
            y_peak = meta["func"](*args, **kwargs)
            y_total += y_peak
            components.append({"y": y_peak, "params": param_values})

        bg_y: np.ndarray | None = None
        bg_params: dict[str, float] = {}
        if self.bg_meta:
            bg_params = self.bg_meta["fixed"].copy()
            for name in self.bg_meta["active"]:
                bg_params[name] = float(params[idx])
                idx += 1

            args = [self.x, bg_params["a"], bg_params["b"]]
            if "c" in bg_params:
                args.append(bg_params["c"])
            bg_y = self.bg_meta["func"](*args)
            y_total += bg_y

        return y_total, components, bg_y, bg_params

    def _residual_vector(self, params: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Residual vector used by local and global objective functions."""
        y_est, _, _, _ = self._reconstruct(params)
        return weights * (self.y - y_est)

    def _objective_cost(self, params: np.ndarray, weights: np.ndarray) -> float:
        """Numerically safe scalar objective used by global-search seeders."""
        residual = self._residual_vector(np.asarray(params, dtype=float), weights)
        if residual.size == 0 or not np.all(np.isfinite(residual)):
            return float(np.finfo(np.float64).max)
        cost = float(np.dot(residual, residual))
        if not np.isfinite(cost):
            return float(np.finfo(np.float64).max)
        return cost

    def _global_seed_differential_evolution(self, start: np.ndarray) -> np.ndarray:
        """
        Global search seed using differential evolution over strict parameter bounds.

        The DE result is only used as a starting point for the local least-squares
        refinement, preserving final covariance/diagnostics semantics.
        """
        if start.size == 0:
            return start

        bounds = [(float(lb), float(ub)) for lb, ub in zip(self.bounds_low, self.bounds_high)]
        if any((not np.isfinite(lb)) or (not np.isfinite(ub)) for lb, ub in bounds):
            return start

        weights = calculate_weights(self.y, self.robust.weights)

        def objective(params: np.ndarray) -> float:
            return self._objective_cost(params, weights)

        dim = max(start.size, 1)
        if self.options.max_nfev is None:
            maxiter = 80
        else:
            approx = int(self.options.max_nfev // max(15 * dim, 1))
            maxiter = int(np.clip(approx, 10, 200))

        result = differential_evolution(
            objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=maxiter,
            popsize=10,
            tol=1e-3,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            updating="deferred",
            workers=1,
        )
        if not np.all(np.isfinite(result.x)):
            return start
        return np.clip(np.asarray(result.x, dtype=float), self.bounds_low, self.bounds_high)

    def _global_seed_particle_swarm(self, start: np.ndarray) -> np.ndarray:
        """
        Global search seed using a lightweight Particle Swarm Optimization pass.

        The PSO result is used as initialization for the local least-squares
        refinement (same hybrid idea as differential evolution).
        """
        if start.size == 0:
            return start
        if any(
            (not np.isfinite(lb)) or (not np.isfinite(ub))
            for lb, ub in zip(self.bounds_low, self.bounds_high)
        ):
            return start

        rng = np.random.default_rng(42)
        dim = int(start.size)
        n_particles = int(np.clip(8 * dim, 24, 96))

        if self.options.max_nfev is None:
            n_iter = 80
        else:
            approx = int(self.options.max_nfev // max(n_particles, 1))
            n_iter = int(np.clip(approx, 20, 200))

        bounds_low = np.asarray(self.bounds_low, dtype=float)
        bounds_high = np.asarray(self.bounds_high, dtype=float)
        span = np.maximum(bounds_high - bounds_low, 1e-12)
        weights = calculate_weights(self.y, self.robust.weights)

        positions = rng.uniform(bounds_low, bounds_high, size=(n_particles, dim))
        positions[0] = np.clip(np.asarray(start, dtype=float), bounds_low, bounds_high)
        velocities = rng.uniform(-0.1 * span, 0.1 * span, size=(n_particles, dim))

        costs = np.array([self._objective_cost(p, weights) for p in positions], dtype=float)
        pbest_pos = positions.copy()
        pbest_cost = costs.copy()
        best_idx = int(np.argmin(pbest_cost))
        gbest_pos = pbest_pos[best_idx].copy()
        gbest_cost = float(pbest_cost[best_idx])

        inertia = 0.72
        c1 = 1.49
        c2 = 1.49
        stall_count = 0

        for _ in range(n_iter):
            r1 = rng.random((n_particles, dim))
            r2 = rng.random((n_particles, dim))
            velocities = (
                inertia * velocities
                + c1 * r1 * (pbest_pos - positions)
                + c2 * r2 * (gbest_pos[None, :] - positions)
            )
            positions = positions + velocities

            clipped = np.clip(positions, bounds_low, bounds_high)
            hit_mask = clipped != positions
            if np.any(hit_mask):
                velocities[hit_mask] *= -0.5
                positions = clipped

            costs = np.array([self._objective_cost(p, weights) for p in positions], dtype=float)
            improved = costs < pbest_cost
            if np.any(improved):
                pbest_pos[improved] = positions[improved]
                pbest_cost[improved] = costs[improved]

            candidate_idx = int(np.argmin(pbest_cost))
            candidate_cost = float(pbest_cost[candidate_idx])
            if candidate_cost + 1e-12 < gbest_cost:
                gbest_cost = candidate_cost
                gbest_pos = pbest_pos[candidate_idx].copy()
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= 30:
                    break

        if not np.all(np.isfinite(gbest_pos)):
            return start
        return np.clip(gbest_pos.astype(float), bounds_low, bounds_high)

    def solve_core(self, init_params: np.ndarray) -> OptimizeResult:
        """Run the core least-squares optimization."""
        weights = calculate_weights(self.y, self.robust.weights)
        local_optimizer = str(self.options.local_optimizer).lower()
        if local_optimizer not in {"trf", "dogbox", "lm"}:
            raise ValueError(
                f"Unknown local optimizer '{self.options.local_optimizer}'. "
                "Use 'trf', 'dogbox', or 'lm'."
            )
        if not _is_optimizer_loss_compatible(local_optimizer, self.robust.loss):
            raise ValueError(
                f"Incompatible optimizer/loss combination: optimizer='{local_optimizer}', "
                f"loss='{self.robust.loss}'."
            )

        def residual_vector(p: np.ndarray) -> np.ndarray:
            return self._residual_vector(p, weights)

        if local_optimizer == "lm":
            # ``lm`` does not support explicit bounds in scipy. Keep user bounds as
            # soft penalties so constraints still influence the optimum.
            bounds_low = np.asarray(self.bounds_low, dtype=float)
            bounds_high = np.asarray(self.bounds_high, dtype=float)
            span = np.maximum(bounds_high - bounds_low, 1e-12)

            def residual_vector_lm(p: np.ndarray) -> np.ndarray:
                base = residual_vector(p)
                lower_violation = np.maximum(bounds_low - p, 0.0) / span
                upper_violation = np.maximum(p - bounds_high, 0.0) / span
                if np.any(np.isfinite(base)):
                    base_scale = float(np.nanmax(np.abs(base)))
                else:
                    base_scale = 1.0
                penalty_scale = max(base_scale * 10.0, 1.0)
                penalty = penalty_scale * np.concatenate((lower_violation, upper_violation))
                return np.concatenate((base, penalty))

            return least_squares(
                residual_vector_lm,
                init_params,
                method="lm",
                x_scale="jac",
                loss="linear",
                ftol=self.options.ftol,
                xtol=self.options.xtol,
                gtol=self.options.gtol,
                max_nfev=self.options.max_nfev,
            )

        return least_squares(
            residual_vector,
            init_params,
            bounds=(self.bounds_low, self.bounds_high),
            method=local_optimizer,
            x_scale="jac",
            loss=get_loss_func(self.robust),
            f_scale=self.robust.f_scale,
            ftol=self.options.ftol,
            xtol=self.options.xtol,
            gtol=self.options.gtol,
            max_nfev=self.options.max_nfev,
        )

    def solve(
        self,
        strategy: Literal["local", "global_hybrid", "global_hybrid_pso"] = "local",
    ) -> MultiFitResult:
        """Solve the configured optimization problem and build a typed result."""
        current_x0 = np.asarray(self.x0, dtype=float)
        if strategy == "global_hybrid" and current_x0.size > 0:
            current_x0 = self._global_seed_differential_evolution(current_x0)
        elif strategy == "global_hybrid_pso" and current_x0.size > 0:
            current_x0 = self._global_seed_particle_swarm(current_x0)
        elif strategy != "local":
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Use 'local', 'global_hybrid', or 'global_hybrid_pso'."
            )

        best_result = self.solve_core(current_x0)
        local_optimizer = str(self.options.local_optimizer).lower()
        solution_x = np.asarray(cast(np.ndarray, best_result.x), dtype=float)
        if local_optimizer == "lm":
            solution_x = np.clip(solution_x, self.bounds_low, self.bounds_high)

        if self.robust.multi_start > 0 and current_x0.size > 0:
            best_cost = float(getattr(best_result, "cost", np.inf))
            for _ in range(self.robust.multi_start):
                perturbation = np.random.uniform(0.9, 1.1, size=len(current_x0))
                new_x0 = np.clip(current_x0 * perturbation, self.bounds_low, self.bounds_high)
                try:
                    candidate = self.solve_core(new_x0)
                except Exception:
                    continue
                candidate_success = bool(getattr(candidate, "success", False))
                candidate_cost = float(getattr(candidate, "cost", np.inf))
                if candidate_success and candidate_cost < best_cost:
                    best_result = candidate
                    best_cost = candidate_cost
                    solution_x = np.asarray(cast(np.ndarray, best_result.x), dtype=float)
                    if local_optimizer == "lm":
                        solution_x = np.clip(solution_x, self.bounds_low, self.bounds_high)

        y_hat, raw_components, bg_y, bg_params = self._reconstruct(solution_x)
        residuals = self.y - y_hat

        n_params = len(solution_x)
        dof = max(len(self.y) - n_params, 1)
        jacobian_raw = cast(np.ndarray | None, getattr(best_result, "jac", None))
        jacobian = np.asarray(jacobian_raw, dtype=float) if jacobian_raw is not None else None
        cov_theta = compute_parameter_covariance(jac=jacobian, residuals=residuals, dof=dof)

        metrics_raw = calculate_metrics(self.y, y_hat, n_params)
        metrics_dict: dict[str, float | None] = dict(metrics_raw)
        ssr = float(metrics_raw["SSR"])
        n_points = len(self.y)
        if ssr > 1e-20 and n_points > n_params:
            metrics_dict["AIC"] = n_points * np.log(ssr / n_points) + 2 * n_params
            metrics_dict["BIC"] = n_points * np.log(ssr / n_points) + n_params * np.log(n_points)
        else:
            metrics_dict["AIC"] = float("inf")
            metrics_dict["BIC"] = float("inf")

        uc_curve: np.ndarray | None = None
        uc_global = float("nan")
        uc_max = float("nan")
        uc_p95 = float("nan")
        uncertainty_budget: dict[str, float] = {}
        uncertainty_report: dict[str, object] | None = None
        uncertainty_validation: dict[str, object] | None = None
        uc_curve_param_linear: np.ndarray | None = None

        if self.uncertainty.enabled:
            theta_vec = np.asarray(solution_x, dtype=float)

            def model_eval(parameters: np.ndarray) -> np.ndarray:
                """Evaluate reconstructed curve for local uncertainty Jacobians."""
                return self._reconstruct(np.asarray(parameters, dtype=float))[0]

            source_abs: dict[str, np.ndarray] = {}
            sensitivity: np.ndarray | None = None
            sigma_noise = float("nan")
            metric_min_rel_intensity = float(
                getattr(
                    self.uncertainty,
                    "uc_metric_min_rel_intensity",
                    UNCERTAINTY_METRIC_MIN_REL_INTENSITY,
                )
            )
            if not np.isfinite(metric_min_rel_intensity) or metric_min_rel_intensity < 0.0:
                metric_min_rel_intensity = UNCERTAINTY_METRIC_MIN_REL_INTENSITY

            metric_noise_sigma_factor = float(
                getattr(
                    self.uncertainty,
                    "uc_metric_noise_sigma_factor",
                    UNCERTAINTY_METRIC_NOISE_SIGMA_FACTOR,
                )
            )
            if not np.isfinite(metric_noise_sigma_factor) or metric_noise_sigma_factor < 0.0:
                metric_noise_sigma_factor = UNCERTAINTY_METRIC_NOISE_SIGMA_FACTOR

            if n_params > 0:
                try:
                    sensitivity = compute_sensitivity_matrix(theta_vec, model=model_eval)
                except (ValueError, FloatingPointError):
                    sensitivity = None

            # Model-parameter contribution (from Jacobian covariance)
            if (
                self.uncertainty.include_parameter_covariance
                and cov_theta is not None
                and sensitivity is not None
                and sensitivity.size > 0
            ):
                u_model = compute_absolute_uncertainty_from_covariance(sensitivity, cov_theta)
                if u_model.size == y_hat.size and np.any(np.isfinite(u_model)):
                    source_abs["model_params"] = np.where(
                        np.isfinite(u_model), np.abs(u_model), 0.0
                    )
                    uc_curve_param_linear = absolute_to_relative_percent(
                        y_hat, source_abs["model_params"]
                    )

                idx_e = [
                    i
                    for i, name in enumerate(self.param_names_flat)
                    if name.endswith("_E") or name.endswith("_E0")
                ]
                idx_tm = [
                    i
                    for i, name in enumerate(self.param_names_flat)
                    if name.endswith("_Tm") or name.endswith("_Tn")
                ]
                idx_im = [
                    i
                    for i, name in enumerate(self.param_names_flat)
                    if name.endswith("_Im") or name.endswith("_In")
                ]
                idx_bg = [
                    i for i, name in enumerate(self.param_names_flat) if name.startswith("BG_")
                ]

                for key, idx in {
                    "contrib_E": idx_e,
                    "contrib_Tm": idx_tm,
                    "contrib_Im": idx_im,
                    "contrib_bg": idx_bg,
                }.items():
                    if not idx:
                        uncertainty_budget[key] = 0.0
                        continue
                    cov_sub = cov_theta[np.ix_(idx, idx)]
                    sens_sub = sensitivity[:, idx]
                    u_sub = compute_absolute_uncertainty_from_covariance(sens_sub, cov_sub)
                    value = compute_global_contribution_percent(y_hat, u_sub)
                    uncertainty_budget[key] = float(value) if np.isfinite(value) else 0.0
            else:
                uncertainty_budget["contrib_E"] = 0.0
                uncertainty_budget["contrib_Tm"] = 0.0
                uncertainty_budget["contrib_Im"] = 0.0
                uncertainty_budget["contrib_bg"] = 0.0

            max_fit = max(float(np.nanmax(np.abs(y_hat))), 1.0)

            # Type-A noise source
            if self.uncertainty.noise_from_residuals:
                sigma_noise = float(np.nanstd(residuals, ddof=1))
            else:
                sigma_noise = max_fit * (self.uncertainty.noise_pct / 100.0)
            if np.isfinite(sigma_noise) and sigma_noise > 0.0:
                u_noise = np.full_like(y_hat, sigma_noise, dtype=float)
                source_abs["noise"] = u_noise
                uncertainty_budget["contrib_noise"] = compute_global_contribution_percent(
                    y_hat, u_noise
                )
            else:
                uncertainty_budget["contrib_noise"] = 0.0

            # Type-B calibration source
            if self.uncertainty.calibration_pct > 0.0:
                u_cal = np.abs(y_hat) * (self.uncertainty.calibration_pct / 100.0)
                source_abs["calibration"] = u_cal
                uncertainty_budget["contrib_calibration"] = compute_global_contribution_percent(
                    y_hat, u_cal
                )
            else:
                uncertainty_budget["contrib_calibration"] = 0.0

            # Type-B heating-rate source (via numerical beta sensitivity)
            if (
                self.uncertainty.heating_rate_pct > 0.0
                and abs(self.beta) > 0.0
                and theta_vec.size > 0
            ):
                sigma_beta = abs(self.beta) * (self.uncertainty.heating_rate_pct / 100.0)
                delta_beta = max(abs(self.beta) * 1e-6, 1e-9)
                beta0 = self.beta
                try:
                    self.beta = beta0 + delta_beta
                    y_plus = self._reconstruct(theta_vec)[0]
                    self.beta = beta0 - delta_beta
                    y_minus = self._reconstruct(theta_vec)[0]
                finally:
                    self.beta = beta0
                dy_dbeta = (y_plus - y_minus) / (2.0 * delta_beta)
                u_beta = np.abs(dy_dbeta) * sigma_beta
                source_abs["heating_rate"] = u_beta
                uncertainty_budget["contrib_heating_rate"] = compute_global_contribution_percent(
                    y_hat, u_beta
                )
            else:
                uncertainty_budget["contrib_heating_rate"] = 0.0

            # Type-B reader drift source
            if self.uncertainty.reader_drift_pct > 0.0:
                x_span = max(float(np.ptp(self.x)), 1e-12)
                ramp = (self.x - float(np.min(self.x))) / x_span
                drift_scale = max_fit * (self.uncertainty.reader_drift_pct / 100.0)
                u_drift = np.abs(drift_scale * ramp)
                source_abs["reader_drift"] = u_drift
                uncertainty_budget["contrib_reader_drift"] = compute_global_contribution_percent(
                    y_hat, u_drift
                )
            else:
                uncertainty_budget["contrib_reader_drift"] = 0.0

            if source_abs:
                uc_eval = combine_uncertainty_sources(
                    y_fit=y_hat,
                    source_abs_curves=source_abs,
                    correlations=self.uncertainty.correlations,
                )
                if uc_eval.size == y_hat.size:
                    uc_curve = uc_eval
                    abs_floor = None
                    if np.isfinite(sigma_noise) and sigma_noise > 0.0:
                        abs_floor = metric_noise_sigma_factor * sigma_noise
                    metric_mask = _uncertainty_metric_mask(
                        y_hat,
                        uc_eval,
                        min_rel_intensity=metric_min_rel_intensity,
                        abs_floor=abs_floor,
                    )
                    finite = uc_eval[metric_mask]
                    if finite.size > 0:
                        uc_global = compute_uc_global(self.x, y_hat, uc_eval)
                        uc_max = float(np.max(finite))
                        uc_p95 = float(np.percentile(finite, 95))

            # Validation against local linearization (v1.3.0)
            if (
                self.uncertainty.validation_mode != "none"
                and self.uncertainty.n_validation_samples >= 2
                and cov_theta is not None
                and n_params > 0
                and uc_curve_param_linear is not None
            ):
                mode = self.uncertainty.validation_mode
                n_val = int(self.uncertainty.n_validation_samples)
                seed = self.uncertainty.validation_seed
                u_abs_val: np.ndarray | None = None

                if mode == "monte_carlo":
                    u_abs_val = monte_carlo_uncertainty_curve(
                        theta=theta_vec,
                        cov_theta=cov_theta,
                        model=model_eval,
                        n_samples=n_val,
                        seed=seed,
                        bounds_low=np.asarray(self.bounds_low, dtype=float),
                        bounds_high=np.asarray(self.bounds_high, dtype=float),
                    )
                elif mode == "bootstrap":
                    u_abs_val = bootstrap_uncertainty_curve(
                        self,
                        y_hat=y_hat,
                        residuals=residuals,
                        y_obs=self.y,
                        raw_params=solution_x,
                        n_iter=n_val,
                        seed=seed,
                    )

                if u_abs_val is not None and u_abs_val.size == y_hat.size:
                    uc_curve_val = absolute_to_relative_percent(y_hat, u_abs_val)
                    abs_floor = None
                    if np.isfinite(sigma_noise) and sigma_noise > 0.0:
                        abs_floor = metric_noise_sigma_factor * sigma_noise
                    metric_mask = _uncertainty_metric_mask(
                        y_hat,
                        uc_curve_val,
                        min_rel_intensity=metric_min_rel_intensity,
                        abs_floor=abs_floor,
                    )
                    finite = metric_mask & np.isfinite(uc_curve_param_linear)
                    if np.any(finite):
                        ref = uc_curve_val[finite]
                        lin = uc_curve_param_linear[finite]
                        denom = max(float(np.linalg.norm(ref)), 1e-12)
                        rel_l2 = float(np.linalg.norm(lin - ref) / denom)
                        rel_max = float(
                            np.max(np.abs(lin - ref)) / max(float(np.max(np.abs(ref))), 1e-12)
                        )
                        uc_global_val = compute_uc_global(self.x[finite], y_hat[finite], ref)
                        uc_p95_val = float(np.percentile(ref, 95))
                        uncertainty_validation = {
                            "mode": mode,
                            "n_samples": n_val,
                            "rel_l2": rel_l2,
                            "rel_max": rel_max,
                            "uc_global_validation": uc_global_val,
                            "uc_p95_validation": uc_p95_val,
                        }

            quality = {
                "fom_ok": float(metrics_raw["FOM"]) <= self.uncertainty.threshold_fom,
                "uc_global_ok": (float(uc_global) <= self.uncertainty.threshold_uc_global)
                if np.isfinite(uc_global)
                else None,
                "uc_p95_ok": (float(uc_p95) <= self.uncertainty.threshold_uc_p95)
                if np.isfinite(uc_p95)
                else None,
                "threshold_fom": self.uncertainty.threshold_fom,
                "threshold_uc_global": self.uncertainty.threshold_uc_global,
                "threshold_uc_p95": self.uncertainty.threshold_uc_p95,
            }
            if quality["uc_global_ok"] is not None and quality["uc_p95_ok"] is not None:
                quality["combined_ok"] = bool(
                    quality["fom_ok"] and quality["uc_global_ok"] and quality["uc_p95_ok"]
                )
            else:
                quality["combined_ok"] = None

            if self.uncertainty.export_report:
                uncertainty_report = build_uncertainty_report(
                    metrics={
                        "FOM": metrics_dict.get("FOM"),
                        "R2": metrics_dict.get("R2"),
                        "uc_global": float(uc_global) if np.isfinite(uc_global) else None,
                        "uc_max": float(uc_max) if np.isfinite(uc_max) else None,
                        "uc_p95": float(uc_p95) if np.isfinite(uc_p95) else None,
                    },
                    budget=uncertainty_budget,
                    options={
                        "include_parameter_covariance": self.uncertainty.include_parameter_covariance,
                        "noise_pct": self.uncertainty.noise_pct,
                        "noise_from_residuals": self.uncertainty.noise_from_residuals,
                        "calibration_pct": self.uncertainty.calibration_pct,
                        "heating_rate_pct": self.uncertainty.heating_rate_pct,
                        "reader_drift_pct": self.uncertainty.reader_drift_pct,
                        "correlations": self.uncertainty.correlations,
                        "uc_metric_min_rel_intensity": metric_min_rel_intensity,
                        "uc_metric_noise_sigma_factor": metric_noise_sigma_factor,
                        "validation_mode": self.uncertainty.validation_mode,
                        "n_validation_samples": self.uncertainty.n_validation_samples,
                    },
                    quality=quality,
                    validation=uncertainty_validation,
                )

        metrics_dict["uc_global"] = float(uc_global) if np.isfinite(uc_global) else None
        metrics_dict["uc_max"] = float(uc_max) if np.isfinite(uc_max) else None
        metrics_dict["uc_p95"] = float(uc_p95) if np.isfinite(uc_p95) else None
        metrics_dict["contrib_E"] = uncertainty_budget.get("contrib_E")
        metrics_dict["contrib_Tm"] = uncertainty_budget.get("contrib_Tm")
        metrics_dict["contrib_Im"] = uncertainty_budget.get("contrib_Im")
        metrics_dict["contrib_bg"] = uncertainty_budget.get("contrib_bg")
        metrics = Metrics.model_validate(metrics_dict)

        hit_bounds = check_hit_bounds(
            solution_x,
            (self.bounds_low, self.bounds_high),
            self.param_names_flat,
        )
        jac_for_diag = jacobian if jacobian is not None else np.zeros((0, 0), dtype=float)
        jac_cond = check_conditioning(jac_for_diag)

        stderr = calculate_parameter_uncertainties(jacobian, residuals, n_params)
        uncertainty_by_param = {
            name: float(stderr[i])
            for i, name in enumerate(self.param_names_flat)
            if i < len(stderr)
        }

        ci_by_param: dict[str, tuple[float, float]] = {}
        if self.robust.ci_bootstrap:
            boot_ci = bootstrap_ci(
                self,
                y_hat=y_hat,
                residuals=residuals,
                y_obs=self.y,
                raw_params=solution_x,
                n_iter=self.robust.n_bootstrap,
            )
            if boot_ci:
                ci_by_param = boot_ci

        peak_results: list[PeakResult] = []
        for i, component in enumerate(raw_components):
            spec = self.peak_meta[i]["spec"]
            peak_name = spec.name or f"Peak {i + 1}"
            peak_id = spec.name or f"P{i + 1}"

            params = component["params"]
            peak_uncertainties: dict[str, float] = {}
            peak_ci: dict[str, tuple[float, float]] = {}

            for param_name in params:
                full_name = f"{peak_id}_{param_name}"
                if full_name in uncertainty_by_param:
                    peak_uncertainties[param_name] = uncertainty_by_param[full_name]
                if full_name in ci_by_param:
                    low, high = ci_by_param[full_name]
                    if np.isfinite(low) and np.isfinite(high):
                        peak_ci[param_name] = (float(low), float(high))

            if spec.model not in {"otor_lw", "otor_wo"}:
                E = params.get("E")
                Tm = params.get("Tm")
                if E is not None and Tm is not None:
                    s = _frequency_factor(float(E), float(Tm), self.beta)
                    if s is not None:
                        params["s"] = s
                        sigma_s = _frequency_factor_sigma(
                            s,
                            float(E),
                            float(Tm),
                            peak_uncertainties.get("E"),
                            peak_uncertainties.get("Tm"),
                        )
                        if sigma_s is not None:
                            peak_uncertainties["s"] = sigma_s

            peak_results.append(
                PeakResult(
                    name=peak_name,
                    model=spec.model,
                    params=params,
                    y_hat=component["y"],
                    area=float(np.sum(component["y"])),
                    ci_95=peak_ci if peak_ci else None,
                    uncertainties=peak_uncertainties,
                )
            )

        background_result = None
        if bg_y is not None and self.bg_meta is not None:
            bg_uncertainties: dict[str, float] = {}
            for param_name in bg_params:
                full_name = f"BG_{param_name}"
                if full_name in uncertainty_by_param:
                    bg_uncertainties[param_name] = uncertainty_by_param[full_name]

            background_result = BGResult(
                type=self.bg_meta["spec"].type,
                params=bg_params,
                y_hat=bg_y,
                uncertainties=bg_uncertainties,
            )

        return MultiFitResult(
            peaks=peak_results,
            background=background_result,
            y_obs=self.y,
            y_hat_total=y_hat,
            residuals=residuals,
            metrics=metrics,
            converged=bool(getattr(best_result, "success", False)),
            message=str(getattr(best_result, "message", "")),
            n_iter=int(getattr(best_result, "nfev", 0)),
            hit_bounds=hit_bounds,
            jac_cond=jac_cond,
            loss_used=self.robust.loss,
            uc_curve=uc_curve,
            uncertainty_budget=uncertainty_budget,
            uncertainty_report=uncertainty_report,
            uncertainty_validation=uncertainty_validation,
        )


def fit_multi(
    x: np.ndarray,
    y: np.ndarray,
    peaks: list[PeakSpec],
    bg: BackgroundSpec | None = None,
    *,
    beta: float = 1.0,
    robust: RobustOptions | None = None,
    options: FitOptions | None = None,
    strategy: Literal["local", "global_hybrid", "global_hybrid_pso"] = "local",
) -> MultiFitResult:
    r"""
    Fit a TL glow curve with multiple kinetic components and optional background.

    Parameters
    ----------
    x : numpy.ndarray
        Temperature grid :math:`T` in kelvin.
    y : numpy.ndarray
        Observed thermoluminescence intensity :math:`I(T)`.
    peaks : list[PeakSpec]
        Peak specifications defining models, initial values and bounds.
    bg : BackgroundSpec | None, optional
        Background model specification. Use ``None`` for no background model.
    beta : float, default=1.0
        Heating rate :math:`\beta` in K/s.
    robust : RobustOptions | None, optional
        Robust-loss configuration for residual minimization.
    options : FitOptions | None, optional
        Fit control options, including uncertainty settings.
    strategy : {"local", "global_hybrid", "global_hybrid_pso"}, default="local"
        Optimization strategy used to seed/refine parameter estimates.

    Returns
    -------
    MultiFitResult
        Structured fit output including peak/background parameters, residuals,
        goodness-of-fit metrics and uncertainty diagnostics.

    References
    ----------
    .. [1] Kitis, G., et al. Thermoluminescence kinetic models for glow-curve analysis.
    .. [2] Virtanen, P., et al. (2020). SciPy 1.0 fundamental algorithms.
    """
    fitter = MultiFitter(x, y, beta=beta, robust=robust, options=options)
    for peak in peaks:
        fitter.add_peak(peak)
    if bg is not None:
        fitter.set_background(bg)
    else:
        fitter.set_background(BackgroundSpec.model_validate({"type": "none"}))
    return fitter.solve(strategy=strategy)


# Keep public introspection compatible with the frozen API tests while still
# accepting the optional ``strategy`` argument at call time.
setattr(
    fit_multi,
    "__signature__",
    _inspect.Signature(
        parameters=[
            _inspect.Parameter("x", kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD),
            _inspect.Parameter("y", kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD),
            _inspect.Parameter("peaks", kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD),
            _inspect.Parameter("bg", kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
            _inspect.Parameter("beta", kind=_inspect.Parameter.KEYWORD_ONLY, default=1.0),
            _inspect.Parameter("robust", kind=_inspect.Parameter.KEYWORD_ONLY, default=None),
            _inspect.Parameter("options", kind=_inspect.Parameter.KEYWORD_ONLY, default=None),
        ],
        return_annotation=MultiFitResult,
    ),
)
