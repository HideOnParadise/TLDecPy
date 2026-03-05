# How to run one-shot automatic deconvolution

`iterative_deconvolution` combines automatic peak detection, model seeding,
and multi-peak fitting into a **single function call**.  It is the fastest
path from raw data to a `MultiFitResult` when you do not need fine-grained
control over peak initialization.

Internally it runs:

1. `preprocess` — Savitzky-Golay smoothing.
2. `detect_peaks_cwt` — CWT-based peak finder with SNR filtering.
3. Automatic `PeakSpec` construction (model assigned from peak asymmetry \(\mu_g\)).
4. `fit_multi` — multi-peak least-squares optimisation.

---

## Minimal call

```python
import tldecpy as tl

T, I = tl.load_refglow("x002")   # 4-peak FO benchmark

result = tl.iterative_deconvolution(T, I, beta=8.4)

print(f"Converged : {result.converged}")
print(f"Peaks     : {len(result.peaks)}")
print(f"R²        : {result.metrics.R2:.6f}")
print(f"FOM       : {result.metrics.FOM:.3f} %")
```

---

## Control the number of peaks and allowed models

```python
result = tl.iterative_deconvolution(
    T, I,
    max_peaks=5,                            # upper bound on components
    allow_models=("fo_rq", "go_kg"),        # restrict model candidates
    bg_mode="auto",                         # "linear", "exponential", "none", "auto"
    beta=8.4,
)
```

`allow_models` accepts canonical keys, family short-names (`"fo"`, `"go"`),
or any registered alias.  The automatic seeder assigns models based on the
geometric factor \(\mu_g\) estimated for each detected peak.

---

## Add robust fitting

```python
result = tl.iterative_deconvolution(
    T, I,
    max_peaks=4,
    allow_models=("fo_rq", "go_kg"),
    beta=8.4,
    robust=tl.RobustOptions(
        loss="soft_l1",
        f_scale=50.0,
        weights="poisson",
    ),
    options=tl.FitOptions(local_optimizer="trf"),
)
```

All `robust` and `options` arguments are forwarded directly to `fit_multi`.

---

## Use a global optimisation strategy

For complex spectra where the local optimizer stalls in a local minimum:

```python
result = tl.iterative_deconvolution(
    T, I,
    max_peaks=6,
    allow_models=("fo", "go"),
    beta=1.0,
    strategy="global_hybrid",   # Differential Evolution → TRF refinement
)
```

| `strategy` | Method | When to use |
|---|---|---|
| `"local"` | TRF/dogbox/lm directly | Fast, good initial guess (default) |
| `"global_hybrid"` | Differential Evolution → TRF | Many local minima, wider search space |
| `"global_hybrid_pso"` | Particle Swarm → TRF | Alternative when DE stalls |

---

## Inspect and iterate

```python
for pk in result.peaks:
    p = pk.params
    print(
        f"{pk.name} ({pk.model}): "
        f"Tm={p['Tm']:.2f} K  E={p['E']:.4f} eV  Im={p['Im']:.0f}"
    )

# Check for bound violations
violations = [k for k, v in result.hit_bounds.items() if v]
if violations:
    print("Parameters at bounds:", violations)
```

If `iterative_deconvolution` misses a peak or assigns the wrong model, switch
to `autoinit_multi` + `fit_multi` for full manual control — see
[Manual peak setup](manual_peak_setup.md).

---

## Difference from `autoinit_multi` + `fit_multi`

| | `iterative_deconvolution` | `autoinit_multi` + `fit_multi` |
|---|---|---|
| Steps | 1 call | 2 calls |
| Control | `max_peaks`, `allow_models`, `bg_mode` | Full per-peak `PeakSpec` |
| Use case | Rapid exploration, batch processing | Publication fits, custom bounds |

Both paths return the same `MultiFitResult` type.

---

## Next steps

- [Manual peak setup](manual_peak_setup.md) — override automatic initialization
- [Robust fitting and loss functions](robust_fitting.md) — tune noise handling
- [Fit quality metrics](../explanation/fit_metrics.md) — interpret FOM and R²
