# tldecpy Documentation

`tldecpy` provides a reproducible workflow for thermoluminescence glow curve analysis:

- typed APIs for fitting and simulation
- physically grounded model families
- robust optimization options
- JSON-safe serialization contracts

## Installation

```bash
pip install tldecpy
```

## Minimal workflow

```python
import tldecpy

T, I = tldecpy.load_refglow("x001")

specs, bg = tldecpy.autoinit_multi(
    T,
    I,
    max_peaks=4,
    allow_models=("fo_rq", "go_kg", "otor_lw"),
)
res = tldecpy.fit_multi(T, I, peaks=specs, bg=bg)

print("Converged:", res.converged)
print("R2:", res.metrics.R2)
print("FOM:", res.metrics.FOM)
```

## Public API

- Fitting: `fit_single_peak`, `fit_multi`
- Initialization: `autoinit_multi`, `pick_peaks`, `preprocess`
- Simulation: `simulate`
- Data: `load_refglow`, `list_refglow`
- Models: `list_models`, `get_model`

### Canonical model keys

Use canonical keys from the model registry (for example `fo_rq`, `go_kg`, `otor_lw`).
Avoid legacy aliases in new scripts.

## Notes

- `fo_rb` uses the same FO core as `fo_rq`, with `alpha(z)` evaluated via an AAA barycentric rational backend from `Q(z)=exp(z)E2(z)` (`alpha=1-Q`).

For serialization details, see `serialization.md`.  
For versioning policy, see `versioning.md`.
