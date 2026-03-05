# OTOR end-to-end fit

This tutorial shows how to fit a **One Trap One Recombination center (OTOR)**
glow curve.  OTOR is the physically most rigorous model for simple two-level
trap systems and introduces a retrapping ratio parameter \(R = A_n / A_m\).

You will:

1. Generate synthetic OTOR data with known ground-truth parameters.
2. Define a `PeakSpec` with physically motivated bounds for \(R\).
3. Run `fit_multi` and recover the original parameters.
4. Understand the two branches of the Lambert W solution.

**Prerequisites:** `pip install tldecpy`

---

## Background: the OTOR model

The OTOR kinetic equations (Kitis & Vlachos 2013, Sadek et al. 2015) describe
electron detrapping with probability of recombination versus retrapping:

\[
P_{\text{recomb}} = \frac{n}{n + R(N - n)}, \quad R = \frac{A_n}{A_m}
\]

where \(A_n\) is the retrapping coefficient and \(A_m\) the recombination
coefficient.

| \(R\) range | Physical regime |
|---|---|
| \(R \to 0\) | Negligible retrapping → first-order behavior |
| \(R \approx 0.5\) | Mixed retrapping / recombination |
| \(R \to 1\) | Near-second-order behavior |

!!! warning "Numerical singularity near R = 0.96"
    The `otor_lw` implementation (Lambert W branch) has a numerical
    singularity near \(R \approx 0.96\) on the \(R < 1\) branch.
    Always keep the upper bound at most `0.90` when \(R < 1\) is expected.

---

## Step 1 — Generate synthetic OTOR data

```python
import numpy as np
import tldecpy as tl

rng = np.random.default_rng(42)

# Temperature grid and heating rate
T = np.linspace(300.0, 620.0, 450)   # kelvin
beta = 1.0                             # K/s

# Ground-truth parameters
TRUE = {"Im": 1400.0, "E": 1.35, "Tm": 470.0, "R": 0.20}

# Evaluate the OTOR model function directly
model_fn = tl.get_model("otor_lw")
I_clean = model_fn(T, **TRUE, beta=beta)

# Add 2 % Gaussian noise
noise_sigma = 0.02 * float(np.max(I_clean))
I_obs = np.clip(
    I_clean + rng.normal(0.0, noise_sigma, size=T.shape),
    0.0, None,
)

print(f"Peak intensity: {I_clean.max():.1f}")
print(f"Noise sigma   : {noise_sigma:.1f}")
```

Using `tl.get_model("otor_lw")` returns the bare model callable —
identical to calling `tldecpy.models.otor_lw.otor_lw` directly.

---

## Step 2 — Define the PeakSpec with OTOR bounds

```python
peak = tl.PeakSpec(
    name="OTOR_P1",
    model="otor_lw",
    init={
        "Im": 1100.0,   # deliberately offset from truth
        "E":  1.10,
        "Tm": 455.0,
        "R":  0.10,
    },
    bounds={
        "Im": (0.0,   3000.0),
        "E":  (0.3,   2.5),
        "Tm": (380.0, 560.0),
        "R":  (1e-6,  0.90),    # keep away from the R=0.96 singularity
    },
)
```

The `R` parameter requires a tight upper bound.  Setting `bounds["R"]` to
`(1e-6, 0.90)` covers the entire physical \(R < 1\) branch safely.

---

## Step 3 — Fit with Poisson weighting

```python
result = tl.fit_multi(
    T, I_obs,
    peaks=[peak],
    bg=None,
    beta=beta,
    robust=tl.RobustOptions(
        loss="soft_l1",     # robust to isolated noisy channels
        f_scale=2.0,        # scale ~ noise floor (~2 % of peak)
        weights="poisson",  # down-weight high-intensity channels
    ),
    options=tl.FitOptions(local_optimizer="trf"),
)
```

`weights="poisson"` divides each residual by \(\sqrt{I_i}\), which is the
correct weighting when the dominant noise source is photon counting statistics.

---

## Step 4 — Compare recovered vs ground-truth parameters

```python
fitted = result.peaks[0].params
unc    = result.peaks[0].uncertainties

print(f"\nConverged : {result.converged}  |  FOM = {result.metrics.FOM:.3f} %")
print(f"\n{'Parameter':>10}  {'Truth':>10}  {'Fitted':>10}  {'Std err':>10}")
print("-" * 46)
for key in ("Im", "E", "Tm", "R"):
    truth  = TRUE[key]
    fit_v  = fitted[key]
    sigma  = unc.get(key, float("nan"))
    print(f"{key:>10}  {truth:>10.4g}  {fit_v:>10.4g}  {sigma:>10.4g}")
```

Expected output (values depend on random noise):

```
Converged : True  |  FOM = 0.xxx %

 Parameter       Truth      Fitted     Std err
----------------------------------------------
        Im      1400.0    ~1400.0       ~10.0
         E       1.350     ~1.350      ~0.003
        Tm       470.0     ~470.0       ~0.5
         R        0.20      ~0.20      ~0.01
```

---

## Step 5 — Check diagnostics

```python
print(f"\nJacobian condition number : {result.jac_cond:.2e}")
print(f"Parameters at bounds      : {[k for k,v in result.hit_bounds.items() if v]}")
print(f"n function evaluations    : {result.n_iter}")
```

A condition number below \(10^8\) indicates a well-posed problem.  If `R`
hits its upper bound (0.90), try extending it to `0.93` but do **not** exceed
`0.95` for `otor_lw`.

---

## Step 6 — otor_wo as a faster alternative

For batch processing, `otor_wo` (Wright Omega approximation) offers the same
physical model at lower cost:

```python
peak_wo = tl.PeakSpec(
    name="OTOR_P1_wo",
    model="otor_wo",            # swap here — identical parameter set
    init=peak.init,
    bounds=peak.bounds,
)
result_wo = tl.fit_multi(T, I_obs, peaks=[peak_wo], bg=None, beta=beta)
print("otor_wo FOM:", result_wo.metrics.FOM)
```

Both models share `(Im, E, Tm, R)`.  Use `otor_lw` for publications (exact
Lambert W solution) and `otor_wo` for exploratory batch runs.

---

## Next steps

- [Fix and freeze parameters](../how-to/fix_parameters.md) — pin `R` to a
  literature value during fit
- [Uncertainty budget](../how-to/uncertainty_budget.md) — propagate Type-A
  noise and calibration uncertainty into reported metrics
- [OTOR physics](../explanation/kinetic_models.md#otor----otor-family) —
  understand the two-level trap model
