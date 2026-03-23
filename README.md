<p align="center">
  <img src="https://raw.githubusercontent.com/HideOnParadise/TLDecPy/main/assets/logo.png" alt="TLDecPy Logo" width="400"/>
</p>

# TLDecPy: Thermoluminescence Glow-Curve Deconvolution in Python

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/HideOnParadise/TLDecPy/actions/workflows/ci.yml/badge.svg)](https://github.com/HideOnParadise/TLDecPy/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18779807.svg)](https://doi.org/10.5281/zenodo.18779807)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://hideonparadise.github.io/TLDecPy/)
[![Web App](https://img.shields.io/badge/web%20app-tldec.tech-green)](https://tldec.tech)

`tldecpy` is an open-source Python library for thermoluminescence (TL) glow-curve deconvolution and kinetic-model benchmarking.

**[Documentation](https://hideonparadise.github.io/TLDecPy/)** · **[Web interface](https://tldec.tech)** (TLDec Web — powered by tldecpy)

## Supported Kinetic Models

- First-order (FO): `fo_rq`, `fo_ka`, `fo_wp`, `fo_rb`
- Second-order (SO): `so_ks`, `so_la`
- General-order (GO): `go_kg`, `go_rq`, `go_ge`
- One-Trap One-Recombination center (OTOR): `otor_lw` (Lambert W), `otor_wo` (Wright Omega)
- Mixed-order: `mo_kitis`, `mo_quad`, `mo_vej`
- Continuous trap-distribution models (Gaussian and Exponential forms)

## Highlights

- Full manual control over peak number, model selection, initial parameters, and bounds
- Multi-component deconvolution with robust loss functions and optional Poisson weighting
- Uncertainty propagation and fit quality metrics (FOM, R², AIC, BIC) in all fit outputs
- Synthetic curve generation for method validation
- Bundled Refglow benchmark datasets (`x001`–`x010`)
- Typed API with Pydantic v2 schemas (`py.typed`)

## Installation

```bash
pip install tldecpy
```

Or from source:

```bash
git clone https://github.com/HideOnParadise/TLDecPy.git
cd TLDecPy
pip install -e ".[dev]"
```

## Quickstart

The recommended workflow is to define peaks explicitly, with physically motivated initial values and bounds for each component:

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
]

result = tl.fit_multi(
    T, I,
    peaks=peaks,
    bg=None,
    beta=8.4,
    options=tl.FitOptions(local_optimizer="trf"),
)

print(f"Converged: {result.converged}")
print(f"R2: {result.metrics.R2:.4f}   FOM: {result.metrics.FOM:.3f}%")
for p in result.peaks:
    print(p.name, p.model, p.params)
```

For more examples, including OTOR fits and robust loss functions, see the [`examples/`](examples/) directory and the [documentation](https://hideonparadise.github.io/TLDecPy/).

## Reproducibility Scripts

All scripts used to generate paper figures and validation results are in `scripts/`:

- Refglow benchmarks: `phase4_refglow_benchmark.py`, `phase5_refglow_x009_otor_lw.py`
- Continuous-distribution validation: `phase6_gdalo_continuous_validation.py`
- Uncertainty validation: `phase7_uncertainty_validation.py`
- Figure generation: `make_paper_figures.py`

See `scripts/README.md` for details.

## Development

```bash
make lint
make typecheck
make test
make qa
```

## Citation

If TLDecPy contributes to your research, please cite:

- DOI: [10.5281/zenodo.18779807](https://doi.org/10.5281/zenodo.18779807)

```bibtex
@software{romero2026tldecpy,
  title   = {TLDecPy},
  author  = {Romero, Cesar},
  version = {1.0.0},
  year    = {2026},
  doi     = {10.5281/zenodo.18779807},
  url     = {https://doi.org/10.5281/zenodo.18779807}
}
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
