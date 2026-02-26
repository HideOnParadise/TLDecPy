# Serialization Contracts (Pydantic v2)

All configuration and result payloads in `tldecpy` use typed Pydantic models.  
This guarantees input validation and stable JSON export for reproducible workflows.

## JSON export

Result models expose `.model_dump_json()`:

```python
from tldecpy import PeakSpec, fit_multi, load_refglow

T, I = load_refglow("x001")
specs = [PeakSpec(model="fo_rq", init={"Tm": 490.0, "Im": 1200.0, "E": 1.2})]
res = fit_multi(T, I, peaks=specs)

json_payload = res.model_dump_json(indent=2)
```

## NumPy handling

- Inputs that represent arrays are validated into `numpy.ndarray`.
- JSON serialization converts arrays to standard lists.
- Deserialization recreates arrays where schema validators are defined.

## Recommended archival artifacts

For publication-grade reproducibility, archive:

- raw input data (`T`, `I`)
- peak/background configuration
- fit options (`beta`, robust options, bounds/fixed)
- full typed result (`MultiFitResult`) serialized as JSON
