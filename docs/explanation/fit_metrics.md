# Fit quality metrics

`MultiFitResult.metrics` is a frozen `Metrics` object.  The fields below are
always present after a successful fit.

---

## Figure of Merit (FOM)

\[
\text{FOM} = 100 \cdot \frac{\displaystyle\sum_{i=1}^{n} |I_i - \hat{I}_i|}
                              {\displaystyle\sum_{i=1}^{n} I_i}
\quad [\%]
\]

Defined by Balian & Eddy (1977).  It measures the mean absolute residual
relative to the total observed signal area.

| FOM value | Interpretation |
|---|---|
| < 2.5 % | Excellent — publication quality |
| 2.5 – 5 % | Acceptable for most dosimetry applications |
| 5 – 10 % | Model mismatch or a missing peak component |
| > 10 % | Poor fit — revisit model choice or number of peaks |

FOM is **not** sensitive to systematic under- or over-fitting because it uses
absolute values; complement it with a visual residual plot.

---

## Coefficient of determination (R²)

\[
R^2 = 1 - \frac{\text{SSR}}{\text{SS}_\text{tot}}, \quad
\text{SSR} = \sum_i (I_i - \hat{I}_i)^2, \quad
\text{SS}_\text{tot} = \sum_i (I_i - \bar{I})^2
\]

- \(R^2 = 1\): perfect fit.
- \(R^2 < 0\): fit is worse than the mean.
- Typical good TL fits: \(R^2 > 0.999\).

---

## Reduced chi-square (RCS)

\[
\tilde{\chi}^2 = \frac{\text{SSR}}{n - k}
\]

where \(n\) is the number of data points and \(k\) is the number of free
parameters.

- \(\tilde{\chi}^2 \approx 1\): residuals consistent with assumed noise level.
- \(\tilde{\chi}^2 \gg 1\): underfitting or noise underestimated.
- \(\tilde{\chi}^2 \ll 1\): overfitting or noise overestimated.

RCS is useful for absolute goodness-of-fit, but requires a noise estimate.
In practice, FOM and visual inspection are more diagnostic.

---

## AIC and BIC (information criteria)

Used for **model comparison** (e.g. "does adding a fifth peak improve the
fit beyond the penalty for the extra parameters?"):

\[
\text{AIC} = n \ln\!\left(\frac{\text{SSR}}{n}\right) + 2k
\]

\[
\text{BIC} = n \ln\!\left(\frac{\text{SSR}}{n}\right) + k \ln n
\]

**Lower is better.**  BIC penalises parameters more heavily than AIC for
\(n > 7\).

```python
print(f"AIC (4 peaks): {result_4p.metrics.AIC:.2f}")
print(f"AIC (5 peaks): {result_5p.metrics.AIC:.2f}")
# Negative ΔAIC → 5-peak model is better;
# positive ΔAIC → extra peak is not justified.
delta_aic = result_5p.metrics.AIC - result_4p.metrics.AIC
print(f"ΔAIC = {delta_aic:.2f}  ({'5p better' if delta_aic < 0 else '4p sufficient'})")
```

---

## Convergence diagnostics

Beyond `metrics`, always check:

| Field | Meaning | Action if bad |
|---|---|---|
| `result.converged` | Optimizer exit flag | Check bounds and init values |
| `result.hit_bounds` | Params pinned at bound | Widen bounds or revise init |
| `result.jac_cond` | Jacobian condition number | > 1e10 → parameters correlated |
| `result.n_iter` | Function evaluations | Very high → slow convergence |

---

## Uncertainty metrics

Available when `FitOptions(uncertainty=UncertaintyOptions(enabled=True))`:

| Field | Unit | Meaning |
|---|---|---|
| `metrics.uc_global` | % | Area-weighted combined uncertainty (ISO GUM) |
| `metrics.uc_max` | % | Maximum channel-wise \(u_c(T)\) in the ROI |
| `metrics.uc_p95` | % | 95th percentile of \(u_c(T)\) in the ROI |
| `metrics.contrib_E` | % | Fraction of \(u_c\) from \(E\) parameters |
| `metrics.contrib_Tm` | % | Fraction from \(T_m\) parameters |
| `metrics.contrib_Im` | % | Fraction from \(I_m\) parameters |
| `metrics.contrib_bg` | % | Fraction from background parameters |

The ROI (region of interest) excludes channels where the fitted signal falls
below 5 % of its maximum to avoid division-by-zero in relative uncertainty.

---

## References

- Balian, H. G., & Eddy, N. W. (1977). *Figure-of-merit (FOM), an improved
  criterion over the normalized chi-squared test for assessing goodness-of-fit
  of gamma-ray spectral peaks.* Nucl. Instr. Meth. 145, 389.
