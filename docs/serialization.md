# Serialization contracts (Pydantic v2)

All configuration and result payloads in `tldecpy` use typed Pydantic v2 models.
This guarantees input validation and stable JSON export for reproducible workflows.

---

## Export to JSON

Every result model exposes `.model_dump_json()`:

```python
import pathlib
import tldecpy as tl

T, I = tl.load_refglow("x001")
peaks, bg = tl.autoinit_multi(T, I, max_peaks=2)
res = tl.fit_multi(T, I, peaks=peaks, bg=bg, beta=1.0)

payload = res.model_dump_json(indent=2)
pathlib.Path("result.json").write_text(payload)
print("Saved.")
```

NumPy arrays are serialised as JSON lists automatically.

---

## Reload from JSON (round-trip)

Use `MultiFitResult.model_validate_json()` to reconstruct a typed result from a
saved JSON file:

```python
import pathlib
from tldecpy import MultiFitResult

raw = pathlib.Path("result.json").read_text()
res = MultiFitResult.model_validate_json(raw)

# All fields are restored with correct types
print(f"Converged : {res.converged}")
print(f"R²        : {res.metrics.R2:.6f}")
print(f"FOM       : {res.metrics.FOM:.3f} %")

# Arrays are reconstructed as numpy.ndarray
print(f"y_hat shape: {res.y_hat_total.shape}")

# Per-peak results
for pk in res.peaks:
    print(f"  {pk.name} ({pk.model}): Tm={pk.params['Tm']:.2f} K")
```

!!! note "Array fidelity"
    Arrays round-trip as `float64` with full precision.  Values stored as
    `float32` during fitting are promoted on reload.

---

## Partial export

To export only the metrics without arrays (e.g. for a summary table):

```python
summary = res.metrics.model_dump()
# {'FOM': ..., 'R2': ..., 'AIC': ..., 'BIC': ..., ...}
```

`model_dump()` returns a plain Python dict; `model_dump_json()` returns a JSON
string directly.

---

## NumPy handling

- Inputs that represent arrays are validated into `numpy.ndarray`.
- JSON serialisation converts arrays to standard Python lists.
- `model_validate_json` reconstructs arrays via the `NumpyArray` validator.

---

## Recommended archival artifacts

For publication-grade reproducibility, archive:

- raw input data (`T`, `I`) as a separate CSV or NPZ file
- peak/background configuration (the `PeakSpec` list and `BackgroundSpec`)
- fit options (`beta`, `RobustOptions`, `FitOptions`, bounds/fixed)
- full typed result serialised as JSON via `MultiFitResult.model_dump_json()`
