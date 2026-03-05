# Continuous trap distributions

Some TL materials — glasses, feldspars, natural minerals under variable
irradiation history — show broad, structureless emission humps that cannot
be decomposed into a small number of discrete kinetic peaks.  The **continuous
trap-distribution models** (`cont_gauss`, `cont_exp`) account for this by
integrating the first-order kinetic emission over a distribution of activation
energies.

---

## Governing equation

Following Benavente et al. (2019), the TL intensity for a continuous
distribution \(g(E)\) is:

\[
I(T) = I_N \,
\frac{\displaystyle\int_0^\infty g(E)\, e^{-E/kT}
      \exp\!\left(-\frac{s}{\beta}\int_{T_0}^{T} e^{-E/kT'}\mathrm{d}T'\right)\mathrm{d}E}
     {\displaystyle\int_0^\infty g(E)\, e^{-E/kT_N}
      \exp\!\left(-\frac{s}{\beta}\int_{T_0}^{T_N} e^{-E/kT'}\mathrm{d}T'\right)\mathrm{d}E}
\]

where the frequency factor is determined self-consistently from the
characteristic temperature \(T_N\):

\[
\frac{s}{\beta} = \frac{E_0}{k T_N^2}\, e^{E_0/(kT_N)}
\]

The temperature integral uses the \(E_2\) exponential integral approximation
(Abramowitz & Stegun).

---

## Parameters

| Symbol | Parameter name | Unit | Physical meaning |
|---|---|---|---|
| \(T_N\) | `Tn` | K | Characteristic temperature (position of the distribution peak) |
| \(I_N\) | `In` | counts | Characteristic intensity at \(T_N\) |
| \(E_0\) | `E0` | eV | Centre (Gaussian) or lower bound (exponential) of the energy distribution |
| \(\sigma\) | `sigma` | eV | Width of the energy distribution |

!!! note "Legacy aliases"
    The discrete-peak aliases `Tm`, `Im`, `E` are accepted and mapped to
    `Tn`, `In`, `E0` internally for backward compatibility.

---

## Gaussian distribution (`cont_gauss`)

\[
g(E) = \frac{1}{\sigma\sqrt{2\pi}}
        \exp\!\left(-\frac{(E-E_0)^2}{2\sigma^2}\right)
\]

The integration range spans \([E_0 - 8\sigma,\; E_0 + 8\sigma]\).

**When to use**: broad symmetric humps; overlapping peaks in heavily
irradiated or natural samples.

```python
import tldecpy as tl
import numpy as np

T = np.linspace(300.0, 700.0, 600)

peak_cg = tl.PeakSpec(
    name="CG1",
    model="cont_gauss",
    init={"Tn": 480.0, "In": 3000.0, "E0": 1.4, "sigma": 0.12},
    bounds={
        "Tn":    (350.0, 650.0),
        "In":    (0.0,   10000.0),
        "E0":    (0.5,   3.0),
        "sigma": (0.01,  0.5),
    },
)
```

---

## Exponential distribution (`cont_exp`)

\[
g(E) = \frac{1}{\sigma}\, e^{-(E-E_0)/\sigma}, \quad E \geq E_0
\]

The integration range spans \([E_0,\; E_0 + 20\sigma]\).

**When to use**: one-sided trap distribution with a sharp lower energy cutoff;
common in amorphous materials and feldspars.

```python
peak_ce = tl.PeakSpec(
    name="CE1",
    model="cont_exp",
    init={"Tn": 480.0, "In": 3000.0, "E0": 1.2, "sigma": 0.10},
)
```

---

## Performance note

The continuous models evaluate a numerical energy integral (\(n = 801\) nodes
by default) at every optimizer iteration.  They are **5–20× slower** per
evaluation than discrete kinetic models.

For large-scale fitting, reduce `n_energy` to 201–401:

```python
# Direct model call with reduced quadrature
from tldecpy.models.continuous import continuous_gaussian

I = continuous_gaussian(T, Tn=480.0, In=3000.0, E0=1.4, sigma=0.12, n_energy=201)
```

When using `fit_multi`, the `n_energy` parameter is not currently exposed
through `PeakSpec`; the default 801-node grid is used automatically.

---

## References

- Benavente, J. F., et al. (2019). *Continuous trap-distribution models for
  TL glow-curve analysis.* Radiat. Meas. 128, 106180.
- Gomez-Ros, J. M., et al. (2006a). *On the TL glow-curve deconvolution for
  continuous distributions of traps.* Radiat. Meas. 41, 12.
- Chen, R., & McKeever, S. W. S. (1997). *Theory of Thermoluminescence and
  Related Phenomena.* World Scientific.
