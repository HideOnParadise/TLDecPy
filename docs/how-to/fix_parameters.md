# How to fix and freeze parameters

`PeakSpec.fixed` holds one or more parameters constant throughout the
optimisation.  This is useful when:

- A parameter is known from literature (e.g. \(E\) from thermally-stimulated
  conductivity measurements).
- The kinetic order \(b\) is to be held at the theoretical value.
- The retrapping ratio \(R\) has been determined from a separate experiment.

---

## Syntax

There are two equivalent forms:

```python
# Form 1: explicit value — parameter is fixed to this number
tl.PeakSpec(
    model="go_kg",
    init={"Tm": 490.0, "Im": 5000.0, "E": 1.5, "b": 1.8},
    fixed={"b": 1.8},          # b is held at 1.8
)

# Form 2: True — parameter is fixed to its value in `init`
tl.PeakSpec(
    model="go_kg",
    init={"Tm": 490.0, "Im": 5000.0, "E": 1.5, "b": 1.8},
    fixed={"b": True},         # b is held at init["b"] = 1.8
)
```

In both cases the parameter does **not** appear in the optimisation vector, so
the Jacobian and covariance matrix are smaller.

---

## Fix E to a literature value

```python
import tldecpy as tl

T, I = tl.load_refglow("x001")

peak = tl.PeakSpec(
    name="P1",
    model="fo_rq",
    init={"Tm": 490.0, "Im": 1500.0, "E": 1.20},
    fixed={"E": 1.20},          # known from TSC measurement
)

result = tl.fit_multi(T, I, peaks=[peak], bg=None, beta=1.0)
fitted = result.peaks[0].params
print(f"Fitted Tm={fitted['Tm']:.2f} K  Im={fitted['Im']:.1f}")
print(f"Fixed  E ={fitted['E']:.4f} eV  (unchanged)")
```

---

## Fix the kinetic order b in a GO model

```python
peak_go_fo = tl.PeakSpec(
    name="P1",
    model="go_kg",
    init={"Tm": 490.0, "Im": 3000.0, "E": 1.4, "b": 1.0},
    fixed={"b": 1.0},           # force first-order behaviour
)

peak_go_so = tl.PeakSpec(
    name="P1",
    model="go_kg",
    init={"Tm": 490.0, "Im": 3000.0, "E": 1.4, "b": 2.0},
    fixed={"b": 2.0},           # force second-order behaviour
)
```

---

## Fix R in the OTOR model

```python
peak_otor = tl.PeakSpec(
    name="OTOR",
    model="otor_lw",
    init={"Tm": 470.0, "Im": 1400.0, "E": 1.35, "R": 0.20},
    fixed={"R": 0.20},          # R from independent experiment
)
```

---

## Fix multiple parameters simultaneously

```python
peak_all_fixed = tl.PeakSpec(
    name="P_ref",
    model="fo_rq",
    init={"Tm": 490.0, "Im": 1000.0, "E": 1.20},
    fixed={"Tm": 490.0, "E": 1.20},    # only Im is free
)
```

With only `Im` free, the fit reduces to a simple linear scaling and is
guaranteed to converge.

---

## Setting `fixed=False` explicitly

Passing `fixed={"b": False}` (boolean `False`) leaves `b` free for
optimisation.  This is the same as not including `b` in `fixed` at all.

---

## Inspect which parameters were fixed

```python
for pk in result.peaks:
    spec_fixed = pk  # PeakResult does not carry the original spec
    # compare pk.params to your original init to see what moved
    for name, val in result.peaks[0].params.items():
        print(f"  {name}: {val:.6g}")
```
