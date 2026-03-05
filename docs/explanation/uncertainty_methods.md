# Uncertainty propagation

`tldecpy` implements a combined uncertainty budget following the
**ISO GUM (Guide to the Expression of Uncertainty in Measurement)**
framework.  The combined standard uncertainty \(u_c(T)\) at each temperature
channel combines:

1. **Model-parameter uncertainty** — propagated from the Jacobian covariance.
2. **Type-A noise** — estimated from detector noise.
3. **Type-B contributions** — systematic sources (calibration, heating rate,
   reader drift).

---

## Jacobian-based local linearisation

After the least-squares solution \(\hat{\theta}\), the parameter covariance
matrix is estimated as:

\[
C_\theta = (J^\top J)^{-1} \, \sigma_\varepsilon^2
\]

where \(J\) is the Jacobian \(\partial \hat{I} / \partial \theta\) at
\(\hat{\theta}\) and \(\sigma_\varepsilon^2 = \text{SSR}/(n - k)\) is the
residual variance.

The model-parameter contribution to \(u_c(T)\) is:

\[
u_\text{model}(T) = \sqrt{\mathbf{s}(T)^\top C_\theta \, \mathbf{s}(T)}
\]

where \(\mathbf{s}(T) = \partial \hat{I}(T) / \partial \theta\) is the
sensitivity vector, computed by central differences.

This method is fast (no extra solves) but assumes:

- The model is approximately linear near \(\hat{\theta}\).
- The residuals are approximately normally distributed.

---

## Combined uncertainty

Each source \(j\) contributes an absolute standard uncertainty
\(u_j(T)\) in detector counts.  They are combined in quadrature:

\[
u_c(T) = \sqrt{\sum_j u_j(T)^2 + 2 \sum_{j < k} \rho_{jk}\, u_j(T)\, u_k(T)}
\]

where \(\rho_{jk}\) are optional correlation coefficients set via
`UncertaintyOptions.correlations = {"sourceA:sourceB": rho}`.

The relative combined uncertainty (reported in `result.uc_curve`) is:

\[
u_c^\text{rel}(T) = \frac{u_c(T)}{\hat{I}(T)} \times 100\%
\]

---

## Global uncertainty criterion

The area-weighted global criterion (analogous to the integral FOM):

\[
u_{c,\text{global}} = \frac{\int u_c(T)\, \hat{I}(T)\, \mathrm{d}T}
                           {\int \hat{I}(T)\, \mathrm{d}T}
\times 100\%
\]

Available as `result.metrics.uc_global`.

---

## Monte Carlo cross-validation

To verify that the Jacobian linearisation is valid:

```python
uc_opts = tl.UncertaintyOptions(
    enabled=True,
    include_parameter_covariance=True,
    noise_pct=1.0,
    validation_mode="monte_carlo",
    n_validation_samples=200,
    validation_seed=42,
)
```

MC draws \(N\) parameter vectors from \(\mathcal{N}(\hat{\theta}, C_\theta)\),
evaluates the model at each, and estimates \(u_c(T)\) from the spread.
`result.uncertainty_validation` contains the relative L2 difference between
MC and linearisation:

```python
val = result.uncertainty_validation
print(f"rel_l2  = {val['rel_l2']:.3f}")   # < 0.10 is good agreement
print(f"rel_max = {val['rel_max']:.3f}")
```

---

## Bootstrap cross-validation

Bootstrap resamples the residuals and re-solves the fit \(N\) times:

```python
uc_opts_boot = tl.UncertaintyOptions(
    enabled=True,
    noise_pct=1.0,
    validation_mode="bootstrap",
    n_validation_samples=100,
    validation_seed=42,
)
```

Bootstrap is more expensive than MC (requires \(N\) full least-squares solves)
but makes fewer distributional assumptions.  Use it when the linearisation
and MC results disagree.

---

## When to trust the Jacobian estimate

| Condition | Trust level |
|---|---|
| `jac_cond < 1e8` and `converged=True` | High |
| `jac_cond` 1e8 – 1e10 | Moderate — verify with MC |
| `jac_cond > 1e10` | Low — parameters are correlated or degenerate |
| Any `hit_bounds` is `True` | Low — re-constrain or fix the offending parameter |

---

## References

- JCGM 100:2008. *Evaluation of measurement data — Guide to the expression of
  uncertainty in measurement (GUM).* BIPM/ISO.
- Peng, J., et al. (2016). *Semi-analytical expressions for the one-trap one-
  recombination centre (OTOR) model.* Radiat. Meas. 93, 55.
