# Kinetic models — physics and model selection

Every TL peak in `tldecpy` is described by a **kinetic model** — a mathematical
expression that relates TL emission intensity \(I(T)\) to a set of physical
parameters.  All models share the same three core parameters.

---

## Shared parameters

| Symbol | Parameter name | Unit | Physical meaning |
|---|---|---|---|
| \(T_m\) | `Tm` | K | Temperature at peak maximum |
| \(I_m\) | `Im` | counts | Intensity at peak maximum |
| \(E\) | `E` | eV | Trap activation energy (depth) |

\(T_m\) and \(I_m\) are **phenomenological anchors** — the model is
normalised so that \(I(T_m) = I_m\) regardless of the underlying kinetics.
\(E\) is a genuine physical quantity that characterises the trap.

---

## First-order kinetics (FO)

**Assumption**: negligible retrapping (\(A_n \ll A_m\)).  After detrapping,
electrons recombine immediately without being recaptured.

The Randall-Wilkins master equation in the FO limit:

\[
I(T) = n_0 \, s \exp\!\left(-\frac{E}{kT}\right)
       \exp\!\left(-\frac{s}{\beta}\int_{T_0}^{T} e^{-E/(kT')} \mathrm{d}T'\right)
\]

The integral \(\int e^{-E/kT'}\mathrm{d}T'\) has no closed form; all four
`fo_*` variants differ only in their numerical approximation:

| Key | Backend for \(\int e^{-E/kT'}\mathrm{d}T'\) | Accuracy |
|---|---|---|
| `fo_rq` | Bos rational polynomial | ~0.1 % |
| `fo_rb` | AAA barycentric rational (from stored \(Q(z)\) data) | ~0.01 % |
| `fo_ka` | Classical asymptotic (Chen & McKeever 1997) | ~1 % |
| `fo_wp` | Weibull empirical approximation | ~2 % |

**Recommendation**: use `fo_rq` or `fo_rb` for all routine fits.

Peak shape: asymmetric with geometric factor \(\mu_g \approx 0.42\).

---

## Second-order kinetics (SO)

**Assumption**: dominant retrapping (\(A_n \gg A_m\)).  Most detrapped
electrons are recaptured before recombining.

\[
I(T) \propto \frac{n^2}{N}
\]

Peak shape: symmetric (\(\mu_g \approx 0.52\)).

| Key | Notes |
|---|---|
| `so_ks` | Garlick-Gibson (1948) standard form. Recommended. |
| `so_la` | Logistic asymmetric approximation |

---

## General-order kinetics (GO)

**Assumption**: phenomenological interpolation between FO and SO via kinetic
order parameter \(b \in (1, 2)\).

\[
I(T) = I_m \, b^{b/(b-1)} \, \exp\!\left(\frac{E}{kT_m}\cdot\frac{T-T_m}{T_m}\right)
       \left[
         (b-1)(1-\delta)\frac{T^2}{T_m^2}\exp\!\left(\frac{E}{kT_m}\cdot\frac{T-T_m}{T_m}\right)
         + 1 + (b-1)\delta_m
       \right]^{-b/(b-1)}
\]

where \(\delta = 2kT/E\) and \(\delta_m = 2kT_m/E\).

| Key | Approximation | Notes |
|---|---|---|
| `go_kg` | Kitis et al. (1998) | Most accurate. Default GO. |
| `go_rq` | Rational-quadratic | Faster but less accurate near \(b=2\) |
| `go_ge` | Exponentialized | Simplest form |

Extra parameter: `b` ∈ (1.001, 2.0).  `b = 1` is equivalent to FO; `b = 2`
to SO.  Allow the optimizer to find `b` freely (e.g. bounds `(1.001, 2.0)`).

---

## Mixed-order kinetics (MO)

**Assumption**: explicit mixture of FO and SO recombination via mixing
parameter \(\alpha \in (0, 1)\).

\(\alpha \to 0\) → FO behavior;  \(\alpha \to 1\) → SO behavior.

| Key | Reference |
|---|---|
| `mo_kitis` | Kitis & Gomez-Ros (2000). **Recommended.** |
| `mo_quad` | Quadratic Gomez-Ros variant |
| `mo_vej` | Vejnovic mixed-order variant |

Extra parameter: `alpha` ∈ (0.001, 0.999).

---

## OTOR — One Trap One Recombination center

**Assumption**: single trap level with a single recombination centre.  The
recombination–retrapping competition is parameterised by:

\[
R = \frac{A_n}{A_m}
\]

where \(A_n\) is the retrapping coefficient (cm³/s) and \(A_m\) the
recombination coefficient (cm³/s).

| \(R\) | Physical meaning |
|---|---|
| \(R \to 0\) | No retrapping → pure FO |
| \(R \approx 0.5\) | Equal retrapping and recombination |
| \(R \to 1\) | Dominant retrapping → near-SO |

The OTOR intensity uses the analytical Lambert W solution
(Kitis & Vlachos 2013):

\[
I(T) \propto \frac{W(T_m) + W(T_m)^2}{W(T) + W(T)^2}
\]

where \(W\) is the Lambert W function evaluated on different branches for
\(R < 1\) and \(R > 1\).

!!! warning "Numerical singularity"
    `otor_lw` exhibits a singularity near \(R \approx 0.96\) on the \(R < 1\)
    branch.  The library clamps \(R \leq 0.95\) internally on that branch.
    **Always set `bounds["R"] = (1e-6, 0.90)`** unless you have evidence
    for \(R > 1\).

| Key | Implementation |
|---|---|
| `otor_lw` | Lambert W exact (recommended) |
| `otor_wo` | Wright Omega fast approximation |

Extra parameter: `R` ∈ (1e-6, 0.90) for \(R < 1\) branch.

---

## Continuous trap distributions

For materials with a broad distribution of trap depths, no single discrete
level is appropriate.  See
[Continuous trap distributions](continuous_traps.md).

---

## Model selection guide

```
Narrow, asymmetric peak (μg < 0.45)?
  → fo_rq  (or fo_rb for higher precision)

Broad, symmetric peak (μg ≈ 0.52)?
  → so_ks

Peak shape between FO and SO?
  → go_kg  (fit b freely, 1.001–2.0)
  → mo_kitis  (if explicit FO/SO mixture is needed)

Material known to have simple two-level trap physics (LiF, etc.)?
  → otor_lw  (physically rigorous)

Broad, structureless hump?
  → cont_gauss  or  cont_exp
```

When uncertain, fit with `go_kg` (freeing `b`) as a flexible baseline; then
compare AIC/BIC against FO or SO to determine whether the extra parameter
is justified.

---

## References

- Randall, J. T., & Wilkins, M. H. F. (1945). *Phosphorescence and electron
  traps.* Proc. R. Soc. London A 184, 366.
- Garlick, G. F. J., & Gibson, A. F. (1948). *The electron trap mechanism of
  luminescence.* Proc. Phys. Soc. 60, 574.
- Kitis, G., Gomez-Ros, J. M., & Tuyn, J. W. N. (1998). *Thermoluminescence
  glow-curve deconvolution functions for first, second and general orders of
  kinetics.* J. Phys. D 31, 2636.
- Kitis, G., & Vlachos, N. D. (2013). *General semi-analytical expressions for
  TL, OSL and other luminescence stimulation modes.* Radiat. Meas. 48, 47.
- Chen, R., & McKeever, S. W. S. (1997). *Theory of Thermoluminescence and
  Related Phenomena.* World Scientific.
