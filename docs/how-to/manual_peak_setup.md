# How to set up peaks manually

In TL deconvolution, peak positions and kinetic parameters are generally
constrained by the material under study. Defining each component explicitly —
with physically motivated initial values and bounds — gives reproducible,
interpretable results and is the standard approach for publication work.

---

## Define each peak with PeakSpec

```python
import tldecpy as tl

T, I = tl.load_refglow("x002")   # 4-peak FO benchmark, beta = 8.4 K/s

peaks = [
    tl.PeakSpec(
        name="P1",
        model="fo_rq",
        init={"Tm": 417.0, "Im": 12000.0, "E": 1.35},
        bounds={"Tm": (390.0, 435.0), "Im": (0.0, 40000.0), "E": (0.8, 2.2)},
    ),
    tl.PeakSpec(
        name="P2",
        model="fo_rq",
        init={"Tm": 456.0, "Im": 18000.0, "E": 1.48},
        bounds={"Tm": (435.0, 472.0), "Im": (0.0, 50000.0), "E": (0.8, 2.4)},
    ),
    tl.PeakSpec(
        name="P3",
        model="fo_rq",
        init={"Tm": 484.0, "Im": 28000.0, "E": 1.60},
        bounds={"Tm": (468.0, 500.0), "Im": (0.0, 70000.0), "E": (0.9, 2.6)},
    ),
    tl.PeakSpec(
        name="P4",
        model="fo_rq",
        init={"Tm": 512.0, "Im": 45000.0, "E": 2.00},
        bounds={"Tm": (498.0, 530.0), "Im": (0.0, 120000.0), "E": (1.0, 3.0)},
    ),
]
```

**Units**

| Parameter | Unit | Typical range |
|---|---|---|
| `Tm` | K | 300 – 700 K for most TLD materials |
| `Im` | detector counts | same scale as measured `I` |
| `E` | eV | 0.5 – 3.0 eV for common TL materials |
| `b` (GO only) | dimensionless | 1.0 – 2.0 |
| `R` (OTOR only) | dimensionless | 1e-6 – 0.90 |
| `alpha` (MO only) | dimensionless | 0.001 – 0.999 |

---

## Add a background component

```python
# Explicit linear background
bg = tl.BackgroundSpec(
    type="linear",
    init={"a": 100.0, "b": 0.5},
)

# Or let autoinit decide (returns None for clean data)
_, bg_auto = tl.autoinit_multi(T, I, bg_mode="auto")
```

Available background types: `"linear"`, `"exponential"`, `"none"`.

The linear background evaluates as \(a + b \cdot T\); the exponential form
evaluates as \(a \cdot \exp(-T / c) + b\).

---

## Run the fit

```python
result = tl.fit_multi(
    T, I,
    peaks=peaks,
    bg=None,
    beta=8.4,
    options=tl.FitOptions(local_optimizer="trf"),
)

print(f"Converged: {result.converged}  FOM: {result.metrics.FOM:.3f} %")
for pk in result.peaks:
    p = pk.params
    print(f"  {pk.name}: Tm={p['Tm']:.2f} K  E={p['E']:.4f} eV  Im={p['Im']:.0f}")
```

---

## Mix different models across peaks

You can assign a different kinetic model to each peak:

```python
peaks_mixed = [
    tl.PeakSpec(name="P1", model="fo_rq",  init={"Tm": 420.0, "Im": 5000.0, "E": 1.2}),
    tl.PeakSpec(name="P2", model="go_kg",  init={"Tm": 460.0, "Im": 8000.0, "E": 1.5, "b": 1.6}),
    tl.PeakSpec(name="P3", model="otor_lw",init={"Tm": 490.0, "Im": 12000.0,"E": 1.7, "R": 0.1}),
]
```

!!! tip "Model keys"
    Call `tl.list_models()` to see all available canonical keys.
    Call `tl.list_models("fo")` to list only first-order variants.

---

## Inspect bound violations after fitting

```python
violations = {k: v for k, v in result.hit_bounds.items() if v}
if violations:
    print("Parameters at bounds — consider widening:")
    for name in violations:
        print(f"  {name}")
```

A parameter at its bound is a strong signal that the initial estimate is
far from the optimum or that the bound is too tight.
