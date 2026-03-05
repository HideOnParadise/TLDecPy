# Quick start: manual multi-peak deconvolution

This tutorial covers a complete deconvolution of **Refglow x002**, a four-peak synthetic first-order glow curve from the GLOCANIN benchmark (Bos et al., 1993).

In TL deconvolution, peak positions and initial parameter estimates are usually known — either from partial heating experiments, derivative analysis, or published data on the material under study. Defining peaks manually gives full control over the fit and is the standard approach for publication-quality results.

**Prerequisites**

```bash
pip install tldecpy
```

---

## Step 1 — Load the data

```python
import tldecpy as tl

T, I = tl.load_refglow("x002")
print(f"Points : {len(T)}")
print(f"T range: {T[0]:.1f} – {T[-1]:.1f} K")
print(f"I max  : {I.max():.0f}")
```

`load_refglow` returns two 1-D NumPy float64 arrays: temperature in **kelvin** and TL intensity in arbitrary counts.

| Dataset | Description |
|---|---|
| `x001` | Synthetic, 1 peak (FO) |
| `x002` | Synthetic, 4 peaks (FO) |
| `x003`–`x008` | TLD-100, 5 peaks |
| `x009` | TLD-700, 9 peaks |
| `x010` | TLD-100, low dose |

---

## Step 2 — Define the peaks

Each peak is specified with a model, initial parameter estimates, and physical bounds:

```python
peaks = [
    tl.PeakSpec(
        name="P1", model="fo_rq",
        init={"Tm": 417.0, "Im": 12000.0, "E": 1.35},
        bounds={"Tm": (390.0, 435.0), "Im": (0.0, 40000.0), "E": (0.8, 2.2)},
    ),
    tl.PeakSpec(
        name="P2", model="fo_rq",
        init={"Tm": 456.0, "Im": 18000.0, "E": 1.48},
        bounds={"Tm": (435.0, 472.0), "Im": (0.0, 50000.0), "E": (0.8, 2.4)},
    ),
    tl.PeakSpec(
        name="P3", model="fo_rq",
        init={"Tm": 484.0, "Im": 28000.0, "E": 1.60},
        bounds={"Tm": (468.0, 500.0), "Im": (0.0, 70000.0), "E": (0.9, 2.6)},
    ),
    tl.PeakSpec(
        name="P4", model="fo_rq",
        init={"Tm": 512.0, "Im": 45000.0, "E": 2.00},
        bounds={"Tm": (498.0, 530.0), "Im": (0.0, 120000.0), "E": (1.0, 3.0)},
    ),
]
```

**Parameter units**

| Parameter | Unit | Typical range |
|---|---|---|
| `Tm` | K | 300 – 700 K |
| `Im` | detector counts | same scale as `I` |
| `E` | eV | 0.5 – 3.0 eV |
| `b` (GO only) | dimensionless | 1.0 – 2.0 |
| `R` (OTOR only) | dimensionless | 1e-6 – 0.90 |

---

## Step 3 — Run the deconvolution

```python
result = tl.fit_multi(
    T, I,
    peaks=peaks,
    bg=None,
    beta=8.4,                           # heating rate in K/s (Refglow x002)
    robust=tl.RobustOptions(
        loss="soft_l1",
        f_scale=50.0,
        weights="poisson",
    ),
    options=tl.FitOptions(local_optimizer="trf"),
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
print(f"R²        : {result.metrics.R2:.6f}")
print(f"FOM       : {result.metrics.FOM:.3f} %")
print(f"AIC       : {result.metrics.AIC:.1f}")
print(f"jac_cond  : {result.jac_cond:.2e}")
```

| Field | Good value | Warning |
|---|---|---|
| `converged` | `True` | `False` → revise bounds or initial values |
| `metrics.FOM` | < 5 % | > 5 % → model mismatch or missing peak |
| `metrics.R2` | > 0.999 | — |
| `jac_cond` | < 1e8 | > 1e10 → parameters likely correlated |
| `hit_bounds` | all `False` | `True` → parameter pinned at bound |

```python
for param, hit in result.hit_bounds.items():
    if hit:
        print(f"  WARNING: {param} hit its bound")
```

---

## Step 5 — Inspect per-peak results

```python
for peak in result.peaks:
    p = peak.params
    print(
        f"{peak.name} ({peak.model}): "
        f"Tm={p['Tm']:.2f} K  E={p['E']:.4f} eV  "
        f"Im={p['Im']:.1f}  area={peak.area:.0f}"
    )
```

Each `PeakResult` contains:

- `params` — fitted parameter dict
- `y_hat` — individual peak contribution
- `area` — integrated area
- `uncertainties` — standard errors from Jacobian covariance

---

## Step 6 — Plot

```python
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
```

---

## Step 7 — Export the result

```python
import pathlib

pathlib.Path("refglow_x002_result.json").write_text(
    result.model_dump_json(indent=2)
)
```

The JSON file includes all fitted parameters, metrics, and intensity arrays, and can be reloaded with `MultiFitResult.model_validate_json(...)`.

---

## Note on automatic initialization

For unfamiliar datasets where peak positions are not known in advance, `autoinit_multi` can provide rough initial estimates via CWT-based peak detection. These should be treated as a starting point and reviewed carefully — automatic detection can miss overlapping peaks or misassign models. See [Manual peak setup](../how-to/manual_peak_setup.md) for guidance on refining those estimates.

---

## Next steps

- [OTOR end-to-end fit](otor_fit.md) — retrapping model with explicit bounds
- [Robust fitting](../how-to/robust_fitting.md) — loss function selection
- [Kinetic models — physics](../explanation/kinetic_models.md) — model selection criteria
