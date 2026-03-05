# How to configure robust fitting and loss functions

`RobustOptions` controls how residuals are weighted before the optimizer
minimises them.  The right choice depends on your noise model.

---

## Loss function quick reference

| `loss` | Formula \(\rho(r)\) | When to use |
|---|---|---|
| `"linear"` | \(r^2\) | Clean data, no outliers (default) |
| `"soft_l1"` | \(2(\sqrt{1+r^2}-1)\) | Moderate outlier channels, counting statistics |
| `"huber"` | \(r^2\) if \(\|r\|<1\), else \(2\|r\|-1\) | Isolated spike channels |
| `"cauchy"` | \(\ln(1+r^2)\) | Heavy-tailed noise, aggressive downweighting |
| `"arctan"` | \(\arctan(r^2)\) | Very heavy tails |
| `"tukey"` | biweight (see SciPy docs) | Strong outliers, requires `loss_param` |

All losses reduce to ordinary least squares for small residuals.

---

## The `f_scale` parameter

`f_scale` sets the residual amplitude that separates "small" (OLS regime) from
"large" (robust regime).  In physical units it should be comparable to the
noise floor of your detector.

```python
import tldecpy as tl
import numpy as np

T, I = tl.load_refglow("x001")
noise_floor = float(np.percentile(I, 5))   # estimate from the tail

peaks, bg = tl.autoinit_multi(T, I, max_peaks=1)
result = tl.fit_multi(
    T, I, peaks=peaks, bg=bg, beta=1.0,
    robust=tl.RobustOptions(
        loss="soft_l1",
        f_scale=max(noise_floor, 1.0),   # at least 1 count
    ),
)
print(f"FOM: {result.metrics.FOM:.3f} %")
```

A good default starting point is `f_scale` = 1–5 % of the peak maximum
intensity.

---

## Poisson weighting

```python
robust_poisson = tl.RobustOptions(
    loss="soft_l1",
    f_scale=2.0,
    weights="poisson",          # divide residuals by sqrt(I_i)
)
```

`weights="poisson"` makes each residual contribution scale as
\(w_i = 1 / \sqrt{I_i}\), which is the correct inverse-variance weighting
when noise arises from photon counting statistics (variance \(\propto I_i\)).

Use `weights="poisson"` whenever:

- The detector counts photons directly (PMT, CCD).
- High-intensity channels have proportionally higher noise.
- You see that FOM is dominated by the peak apex rather than the shoulders.

Use `weights="none"` when:

- The data have been normalised or background-subtracted.
- You have a flat noise floor (e.g. electronic readout noise dominates).

---

## Multi-start restarts

For complex multi-peak fits where the optimizer converges to a local minimum:

```python
result = tl.fit_multi(
    T, I, peaks=peaks, bg=bg, beta=1.0,
    robust=tl.RobustOptions(
        loss="soft_l1",
        f_scale=5.0,
        multi_start=5,      # 5 random restarts with ±10 % perturbation
    ),
)
```

The best solution across all restarts is returned.  `multi_start=3`–`10` is
typical; more restarts increase runtime linearly.

---

## Global search strategies

For highly degenerate problems, use `strategy` in `fit_multi`:

```python
result = tl.fit_multi(
    T, I, peaks=peaks, bg=bg, beta=1.0,
    robust=tl.RobustOptions(loss="soft_l1", f_scale=5.0),
    strategy="global_hybrid",       # DE seed + TRF refinement
)
```

| `strategy` | Method | When to use |
|---|---|---|
| `"local"` | TRF/dogbox/lm directly | Good initial guess, fast |
| `"global_hybrid"` | Differential Evolution → TRF | Many local minima, wider bounds |
| `"global_hybrid_pso"` | Particle Swarm → TRF | Alternative when DE stalls |

Global strategies are 5–20× slower than local but can escape local minima.

---

## Bootstrap confidence intervals

```python
result = tl.fit_multi(
    T, I, peaks=peaks, bg=bg, beta=1.0,
    robust=tl.RobustOptions(
        loss="soft_l1",
        f_scale=5.0,
        ci_bootstrap=True,
        n_bootstrap=200,    # default 100; 200+ for publication
    ),
)

# 95 % CI per parameter
for pk in result.peaks:
    if pk.ci_95:
        for param, (lo, hi) in pk.ci_95.items():
            print(f"  {pk.name}.{param}: [{lo:.4g}, {hi:.4g}]")
```

Bootstrap CIs are more reliable than Jacobian-based standard errors for
nonlinear models, but require \(\sim\)100–500 extra solves.
