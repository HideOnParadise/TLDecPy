# TLDecPy: Thermoluminescence Glow-Curve Deconvolution in Python

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/HideOnParadise/TLDecPy/actions/workflows/ci.yml/badge.svg)](https://github.com/HideOnParadise/TLDecPy/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18779807.svg)](https://doi.org/10.5281/zenodo.18779807)

`tldecpy` is an open-source scientific Python library for thermoluminescence (TL) glow-curve analysis, nonlinear deconvolution, and kinetic-model benchmarking.

The project is designed for reproducible research workflows in radiation dosimetry and TL materials analysis, with typed APIs and physically motivated models.

## Supported Kinetic Models

TLDecPy includes single-peak and multi-peak deconvolution with:

- First-order (FO): `fo_rq`, `fo_ka`, `fo_wp`, `fo_rb`
- Second-order (SO): `so_ks`, `so_la`
- General-order (GO): `go_kg`, `go_rq`, `go_ge`
- One-Trap One-Recombination center (OTOR): `otor_lw` (Lambert W), `otor_wo` (Wright Omega)
- Mixed-order families: `mo_kitis`, `mo_quad`, `mo_vej`
- Continuous trap-distribution models (Gaussian and Exponential forms)

## Highlights

- Multi-component deconvolution with robust losses and optional Poisson weighting
- Automatic initialization helpers (`autoinit_multi`, peak detection, preprocessing)
- Uncertainty metrics and diagnostics in fit outputs
- Synthetic data generation utilities for method validation
- Bundled Refglow benchmark datasets (`x001` to `x010`)
- Typed models and results (`py.typed`, Pydantic v2 schemas)

## Installation

### From source

```bash
git clone https://github.com/HideOnParadise/TLDecPy.git
cd TLDecPy
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### PyPI install

```bash
pip install tldecpy
```

## Quickstart

```python
import tldecpy as tl

# 1) Load a reference glow curve (Refglow)
T, I = tl.load_refglow("x001")

# 2) Automatic initialization of peaks and optional background
peaks, bg = tl.autoinit_multi(
    T,
    I,
    max_peaks=3,
    allow_models=("fo_rq", "go_kg", "otor_lw"),
    bg_mode="auto",
)

# 3) Run deconvolution
result = tl.fit_multi(
    T,
    I,
    peaks=peaks,
    bg=bg,
    beta=1.0,
    robust=tl.RobustOptions(loss="soft_l1", f_scale=2.0, weights="poisson"),
    options=tl.FitOptions(local_optimizer="trf"),
)

print("Converged:", result.converged)
print("R2:", f"{result.metrics.R2:.4f}", "FOM:", f"{result.metrics.FOM:.3f}%")
for peak in result.peaks:
    print(peak.name, peak.model, peak.params)
```

For an end-to-end OTOR demonstration, see:

- `examples/example_otor_fit.py`
- `examples/example_refglow002_manual_fit.py` (manual Refglow x002 deconvolution)

## Reproducibility Scripts

Validation and paper-generation scripts are available in `scripts/`, including:

- Refglow benchmarks (`phase4_refglow_benchmark.py`, `phase5_refglow_x009_otor_lw.py`)
- Continuous-distribution validation (`phase6_gdalo_continuous_validation.py`)
- Uncertainty validation (`phase7_uncertainty_validation.py`)
- Figure orchestration (`make_paper_figures.py`)

See `scripts/README.md` for full details and smoke-run commands.

## Development Quality Checks

```bash
make lint
make typecheck
make test
make qa
```

## Citation

If TLDecPy contributes to your research, please cite the software archive:
- DOI: [10.5281/zenodo.18779807](https://doi.org/10.5281/zenodo.18779807)

Suggested BibTeX entry:

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

This project is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE).
