# How to generate synthetic curves with noise and fit with Poisson weights

Synthetic data generation is essential for:

- Validating that a fitting pipeline recovers known parameters.
- Benchmarking different model variants.
- Studying the effect of noise on parameter uncertainty.

This guide shows how to generate a multi-peak synthetic TL curve, corrupt it
with Poisson (shot) noise, and fit it back using Poisson-weighted residuals.

---

## Step 1 — Generate a clean multi-peak curve

```python
import numpy as np
import tldecpy as tl

# Temperature grid
T = np.linspace(300.0, 620.0, 500)   # kelvin, 500 points
beta = 1.0                            # K/s

# Ground-truth parameters for three peaks
TRUTH = [
    {"model": "fo_rq", "Im": 800.0,  "E": 1.10, "Tm": 400.0},
    {"model": "fo_rq", "Im": 2500.0, "E": 1.35, "Tm": 460.0},
    {"model": "go_kg", "Im": 1200.0, "E": 1.55, "Tm": 520.0, "b": 1.6},
]

# Evaluate each model and sum them
I_clean = np.zeros_like(T)
for p in TRUTH:
    fn = tl.get_model(p["model"])
    args = {k: v for k, v in p.items() if k != "model"}
    I_clean += fn(T, **args)

print(f"Peak intensity : {I_clean.max():.1f}")
print(f"Baseline       : {I_clean.min():.3f}")
```

---

## Step 2 — Add Poisson noise

Photon-counting detectors produce noise whose variance equals the count rate:
\(\text{Var}(I_i) = I_i\), so \(\sigma_i = \sqrt{I_i}\).

```python
rng = np.random.default_rng(0)

# Poisson noise: draw integer counts then convert back to float
I_poisson = rng.poisson(np.maximum(I_clean, 0)).astype(float)

# Signal-to-noise ratio at the main peak
snr = I_clean.max() / np.sqrt(I_clean.max())
print(f"SNR at apex: {snr:.1f}")
```

!!! note "Gaussian approximation"
    For high count rates (\(I_i > 20\)) the Poisson distribution is well
    approximated by \(\mathcal{N}(I_i,\, I_i)\).  At low count rates
    (< 20 counts/channel) use the exact Poisson draw shown above.

---

## Step 3 — Compare fitting with and without Poisson weights

```python
# Build PeakSpec list from ground truth (slightly offset init)
peaks = [
    tl.PeakSpec(
        name=f"P{i+1}",
        model=p["model"],
        init={k: v * 0.95 for k, v in p.items() if k != "model"},
    )
    for i, p in enumerate(TRUTH)
]

# --- Fit A: ordinary least squares (no weights) ---
result_ols = tl.fit_multi(
    T, I_poisson,
    peaks=peaks, bg=None, beta=beta,
    robust=tl.RobustOptions(loss="linear", weights="none"),
)

# --- Fit B: Poisson-weighted soft-l1 ---
result_poi = tl.fit_multi(
    T, I_poisson,
    peaks=peaks, bg=None, beta=beta,
    robust=tl.RobustOptions(
        loss="soft_l1",
        f_scale=np.sqrt(I_poisson.max()) * 0.5,
        weights="poisson",
    ),
)

print(f"OLS  — FOM: {result_ols.metrics.FOM:.3f} %   R²: {result_ols.metrics.R2:.6f}")
print(f"Poisson — FOM: {result_poi.metrics.FOM:.3f} %   R²: {result_poi.metrics.R2:.6f}")
```

With Poisson noise the OLS fit is dominated by the high-intensity apex, while
the Poisson-weighted fit balances contributions from shoulders and apex.

---

## Step 4 — Quantify parameter recovery

```python
print(f"\n{'Peak':>4}  {'Param':>6}  {'Truth':>8}  {'OLS':>8}  {'Poisson':>8}")
print("-" * 42)
for i, p in enumerate(TRUTH):
    for key in ("Im", "E", "Tm"):
        truth_v = p.get(key, float("nan"))
        ols_v   = result_ols.peaks[i].params.get(key, float("nan"))
        poi_v   = result_poi.peaks[i].params.get(key, float("nan"))
        print(f"  P{i+1}  {key:>6}  {truth_v:>8.4g}  {ols_v:>8.4g}  {poi_v:>8.4g}")
```

---

## Step 5 — Add Gaussian noise instead

For systems with dominant electronic readout noise (flat noise floor):

```python
noise_sigma = 10.0   # counts — independent of intensity

rng2 = np.random.default_rng(1)
I_gaussian = I_clean + rng2.normal(0.0, noise_sigma, size=T.shape)
I_gaussian = np.clip(I_gaussian, 0.0, None)

result_gauss = tl.fit_multi(
    T, I_gaussian,
    peaks=peaks, bg=None, beta=beta,
    robust=tl.RobustOptions(
        loss="soft_l1",
        f_scale=noise_sigma * 2,
        weights="none",             # flat noise → no Poisson weighting
    ),
)
print(f"Gaussian noise fit FOM: {result_gauss.metrics.FOM:.3f} %")
```

---

## Summary

| Noise model | Recommended `loss` | Recommended `weights` | `f_scale` guide |
|---|---|---|---|
| Photon counting (Poisson) | `"soft_l1"` | `"poisson"` | ~0.5 × \(\sqrt{I_\max}\) |
| Electronic readout (Gaussian) | `"soft_l1"` or `"linear"` | `"none"` | ~2 × \(\sigma_\text{noise}\) |
| Mixed | `"huber"` | `"poisson"` | ~ noise floor |
| Clean synthetic data | `"linear"` | `"none"` | (any) |

---

## Alternative: ODE-based simulation with `tl.simulate`

The approach above evaluates the analytical model expressions directly via
`tl.get_model()`.  For physically-grounded simulations that integrate the
full trap-kinetics ODE system, use `tl.simulate`:

```python
import tldecpy as tl
from tldecpy.simulate.fo import ode_fo, infer_n0_s

T0, T_end, beta = 300.0, 600.0, 1.0
E, Tm = 1.2, 450.0

n0, s = infer_n0_s(Im=1500.0, Tm=Tm, E=E, beta=beta)

ode_result = tl.simulate(
    ode_fo, params=(s, E),
    T0=T0, T_end=T_end, beta=beta,
    y0=[n0], state_keys=["n"],
    noise_config={"mode": "poisson", "seed": 0},
)

print(f"T shape : {ode_result.T.shape}")
print(f"I max   : {ode_result.I.max():.1f}")
print(f"States  : {list(ode_result.states.keys())}")
```

`tl.simulate` integrates the ODE using `scipy.integrate.solve_ivp` and
returns a `SimulationResult` with `T`, `I`, `states`, and `time`.  The ODE
helper functions (`ode_fo`, `infer_n0_s`, etc.) live in `tldecpy.simulate`
sub-modules and are not part of the public `tl.*` namespace.

Use `tl.simulate` when you need access to the trap-population trajectory
`n(T)` or want to study the effect of different initial occupancies.
