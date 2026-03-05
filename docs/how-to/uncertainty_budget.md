# How to compute an uncertainty budget

`UncertaintyOptions` enables a full ISO-GUM-style uncertainty propagation
that combines contributions from parameter covariance, detector noise,
calibration, heating-rate stability, and reader drift.

---

## Minimal setup

```python
import tldecpy as tl

T, I = tl.load_refglow("x003")
peaks, bg = tl.autoinit_multi(T, I, max_peaks=5)

uc_opts = tl.UncertaintyOptions(
    enabled=True,
    include_parameter_covariance=True,  # Jacobian-based u_model
    noise_pct=1.0,                      # 1 % Type-A noise
    calibration_pct=0.5,                # 0.5 % Type-B calibration
)

result = tl.fit_multi(
    T, I, peaks=peaks, bg=bg, beta=1.0,
    options=tl.FitOptions(uncertainty=uc_opts),
)

print(f"FOM           : {result.metrics.FOM:.3f} %")
print(f"u_c global    : {result.metrics.uc_global:.3f} %")
print(f"u_c max       : {result.metrics.uc_max:.3f} %")
print(f"u_c p95       : {result.metrics.uc_p95:.3f} %")
```

---

## Uncertainty sources and their fields

| `UncertaintyOptions` field | Type | Source category | Typical value |
|---|---|---|---|
| `include_parameter_covariance` | bool | Type-A (model params) | `True` |
| `noise_pct` | float \(\geq 0\) | Type-A (detector noise) | 0.5 – 2 % |
| `noise_from_residuals` | bool | Type-A (auto estimate) | `False` |
| `calibration_pct` | float \(\geq 0\) | Type-B (calibration) | 0.5 – 1 % |
| `heating_rate_pct` | float \(\geq 0\) | Type-B (\(\beta\) stability) | 0.1 – 0.5 % |
| `reader_drift_pct` | float \(\geq 0\) | Type-B (reader drift) | 0.1 – 1 % |

Setting `noise_from_residuals=True` estimates the Type-A noise from the
standard deviation of the post-fit residuals, overriding `noise_pct`.

---

## Reading the uc_curve

`result.uc_curve` is a 1-D array of combined relative uncertainty
\(u_c(T)\) in percent, evaluated at each temperature channel.

```python
import numpy as np

uc = result.uc_curve
if uc is not None:
    print(f"Mean u_c : {np.nanmean(uc):.2f} %")
    print(f"Max  u_c : {np.nanmax(uc):.2f} %")
```

The summary scalars in `metrics` use only ROI channels (fitted signal above
5 % of the peak maximum) to avoid divergence at the tails.

---

## Contribution breakdown

```python
budget = result.uncertainty_budget
for source, pct in budget.items():
    print(f"  {source:20s}: {pct:.3f} %")
```

Typical fields: `"model_params"`, `"noise"`, `"calibration"`,
`"heating_rate"`, `"reader_drift"`.

The contribution fields in `metrics` (`contrib_E`, `contrib_Tm`, `contrib_Im`,
`contrib_bg`) show what fraction of the model-parameter uncertainty comes from
each parameter type.

---

## Validate with Monte Carlo or bootstrap

For critical results, cross-check the local Jacobian linearisation against
a stochastic method:

```python
uc_opts_mc = tl.UncertaintyOptions(
    enabled=True,
    include_parameter_covariance=True,
    noise_pct=1.0,
    validation_mode="monte_carlo",      # or "bootstrap"
    n_validation_samples=100,
    validation_seed=42,
)

result_mc = tl.fit_multi(
    T, I, peaks=peaks, bg=bg, beta=1.0,
    options=tl.FitOptions(uncertainty=uc_opts_mc),
)

val = result_mc.uncertainty_validation
if val:
    print(f"MC vs linearisation rel L2 : {val['rel_l2']:.4f}")
    print(f"MC vs linearisation rel max: {val['rel_max']:.4f}")
```

`rel_l2 < 0.1` (10 %) is a reasonable agreement criterion.

---

## Publication quality thresholds

```python
uc_opts_strict = tl.UncertaintyOptions(
    enabled=True,
    noise_pct=1.0,
    calibration_pct=0.5,
    threshold_fom=5.0,          # FOM must be < 5 %
    threshold_uc_global=2.0,    # global u_c must be < 2 %
    threshold_uc_p95=5.0,       # 95th percentile u_c must be < 5 %
    export_report=True,         # attach full report to result
)
```

The full technical report is at `result.uncertainty_report` — a nested dict
with a `"summary"` section and a per-channel `"table"`.
