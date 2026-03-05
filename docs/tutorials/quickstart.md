# Quick start: automatic multi-peak deconvolution

This tutorial walks through a complete deconvolution of **Refglow x002** — a
four-peak synthetic first-order glow curve from the GLOCANIN benchmark dataset
(Bos et al., 1993).  You will:

1. Load a bundled reference dataset.
2. Run automatic peak initialization.
3. Execute multi-peak deconvolution with a robust loss.
4. Read and interpret every key field of the result.

**Prerequisites**

```bash
pip install tldecpy
```

---

## Step 1 — Load a reference dataset

```python
import tldecpy as tl

# Available IDs: x001 to x010  (list them with tl.list_refglow())
T, I = tl.load_refglow("x002")
print(f"Points : {len(T)}")
print(f"T range: {T[0]:.1f} – {T[-1]:.1f} K")
print(f"I max  : {I.max():.0f}")
```

`load_refglow` returns two 1-D NumPy float64 arrays: temperature in **kelvin**
and TL intensity in arbitrary detector counts.

| Dataset | Description |
|---|---|
| `x001` | Synthetic, 1 peak (FO) |
| `x002` | Synthetic, 4 peaks (FO) |
| `x003`–`x008` | TLD-100, 5 peaks |
| `x009` | TLD-700, 9 peaks (complex) |
| `x010` | TLD-100, low dose |

---

## Step 2 — Automatic initialization

```python
peaks, bg = tl.autoinit_multi(
    T, I,
    max_peaks=4,
    allow_models=("fo_rq", "go_kg"),
    bg_mode="auto",
    sensitivity=1.0,
)

print(f"Seeded {len(peaks)} peaks")
for p in peaks:
    Tm = p.init["Tm"]
    Im = p.init["Im"]
    E  = p.init["E"]
    print(f"  {p.name}: model={p.model}  Tm={Tm:.1f} K  Im={Im:.0f}  E={E:.3f} eV")
```

`autoinit_multi` does three things automatically:

1. **Preprocess** — Savitzky-Golay smoothing and outlier removal.
2. **Detect** — CWT-based peak finder returns candidate positions, FWHM and
   asymmetry \(\mu_g\).
3. **Seed** — Assigns a kinetic model per peak based on \(\mu_g\), estimates
   activation energy \(E\) with Chen-style heuristics, and constructs
   [`PeakSpec`](../reference/schemas.md) objects with automatic bounds.

---

## Step 3 — Deconvolution

```python
result = tl.fit_multi(
    T, I,
    peaks=peaks,
    bg=bg,
    beta=8.4,                                   # heating rate in K/s (Refglow x002)
    robust=tl.RobustOptions(
        loss="soft_l1",                         # robust to outlier channels
        f_scale=50.0,                           # residual scale ~ noise floor
        weights="poisson",                      # heteroscedastic Poisson weighting
    ),
    options=tl.FitOptions(
        local_optimizer="trf",                  # Trust Region Reflective (default)
    ),
)
```

### Optimizer / loss compatibility

| `local_optimizer` | Supported losses |
|---|---|
| `"trf"` (default) | all losses |
| `"dogbox"` | all losses |
| `"lm"` | `"linear"` only |

---

## Step 4 — Interpret the result

```python
print(f"Converged : {result.converged}")
print(f"Message   : {result.message}")
print(f"R²        : {result.metrics.R2:.6f}")
print(f"FOM       : {result.metrics.FOM:.3f} %")
print(f"AIC       : {result.metrics.AIC:.1f}")
print(f"jac_cond  : {result.jac_cond:.2e}")
```

**What to look for**

| Field | Good value | Warning |
|---|---|---|
| `converged` | `True` | `False` → widen bounds or change init |
| `metrics.FOM` | < 5 % | > 5 % → model mismatch or missing peak |
| `metrics.R2` | > 0.999 | — |
| `jac_cond` | < 1e8 | > 1e10 → ill-conditioned, parameters correlated |
| `hit_bounds` | all `False` | `True` → parameter pinned at bound |

```python
# Check for bound violations
for param, hit in result.hit_bounds.items():
    if hit:
        print(f"  WARNING {param} hit its bound")
```

---

## Step 5 — Inspect peak-level results

```python
for peak in result.peaks:
    p = peak.params
    print(
        f"{peak.name} ({peak.model}): "
        f"Tm={p['Tm']:.2f} K  E={p['E']:.4f} eV  "
        f"Im={p['Im']:.1f}  area={peak.area:.0f}"
    )
```

Each [`PeakResult`](../reference/schemas.md) contains:

- `params` — fitted parameter dict (always includes `Tm`, `Im`, `E`; GO adds
  `b`; OTOR adds `R`; FO/SO/GO/MO add derived frequency factor `s`)
- `y_hat` — fitted peak contribution array (same length as `T`)
- `area` — integrated area under the peak
- `uncertainties` — parameter standard errors from Jacobian covariance
- `ci_95` — 95 % confidence intervals (only if `RobustOptions.ci_bootstrap=True`)

---

## Step 6 — Plot (optional)

```python
import numpy as np
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(T, I, "k.", ms=2, label="Observed")
    ax1.plot(T, result.y_hat_total, "r-", lw=1.5, label="Total fit")
    for peak in result.peaks:
        ax1.fill_between(T, 0, peak.y_hat, alpha=0.4, label=peak.name)
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.legend(fontsize=8)

    ax2.plot(T, result.residuals, "b-", lw=0.8)
    ax2.axhline(0, color="k", lw=0.5, ls="--")
    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Residuals")

    plt.tight_layout()
    plt.show()
except ImportError:
    print("Install matplotlib to generate the plot.")
```

---

## Step 7 — Archive the result as JSON

```python
import json, pathlib

payload = result.model_dump_json(indent=2)
pathlib.Path("refglow_x002_result.json").write_text(payload)
print("Saved result to refglow_x002_result.json")
```

The JSON file includes all arrays (stored as lists), all metrics, and all
parameter values.  It can be reloaded with
`MultiFitResult.model_validate_json(...)` for downstream analysis.

---

## Next steps

- [OTOR end-to-end fit](otor_fit.md) — retrapping model with explicit bounds
- [Manual peak setup](../how-to/manual_peak_setup.md) — skip autoinit and
  control every bound
- [Robust fitting](../how-to/robust_fitting.md) — understand every loss option
- [Kinetic models — physics](../explanation/kinetic_models.md) — understand
  model selection
