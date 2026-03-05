# tldecpy Documentation

`tldecpy` is a Python library for thermoluminescence glow-curve deconvolution. It supports multiple kinetic model families, robust optimization, and uncertainty estimation.

## Installation

```bash
pip install tldecpy
```

## Typical workflow

In TL analysis the number of peaks and their initial parameters are generally known from prior measurements or from the material's published glow-curve structure. The recommended approach is to define each component explicitly:

```python
import tldecpy as tl

T, I = tl.load_refglow("x002")

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

result = tl.fit_multi(T, I, peaks=peaks, bg=None, beta=8.4)

print("Converged:", result.converged)
print("R2:", result.metrics.R2)
print("FOM:", result.metrics.FOM)
```

For exploratory analysis on unfamiliar data, `autoinit_multi` can provide initial peak estimates, but results should always be reviewed and refined manually before publication.

## Public API

- Fitting: `fit_single_peak`, `fit_multi`
- Initialization: `autoinit_multi`, `pick_peaks`, `preprocess`
- Simulation: `simulate`
- Data: `load_refglow`, `list_refglow`
- Models: `list_models`, `get_model`

### Canonical model keys

Use canonical keys from the model registry (e.g. `fo_rq`, `go_kg`, `otor_lw`).

## Notes

- `fo_rb` uses the same FO core as `fo_rq`, with `alpha(z)` evaluated via an AAA barycentric rational backend from `Q(z)=exp(z)E2(z)` (`alpha=1-Q`).

For serialization details, see `serialization.md`.
For versioning policy, see `versioning.md`.
