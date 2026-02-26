# Validation And Reproducibility Scripts

This folder contains the scripts used to generate validation outputs and paper figures for TLDecPy.

All scripts are written for direct execution from the repository root:

```bash
python scripts/<script_name>.py
```

Generated artifacts are saved under `output/` by default.

## Included datasets

- `scripts/gdalo.csv`: experimental curve used by `phase6_gdalo_continuous_validation.py`.
- `scripts/TLD100Exp.csv`: experimental TLD-100 curve used by `phase7_uncertainty_validation.py`.

## Validation scripts

- `phase1_validate_aux_alpha.py`: numerical validation of `Q(z)` approximations (SciPy vs AAA vs Bos rational).
- `phase1_time_benchmark.py`: runtime benchmark of auxiliary-function evaluators.
- `phase2_benchmark_solvers.py`: solver stress test and parameter recovery on synthetic multi-peak data.
- `phase3_robustness_test.py`: robust-loss comparison under outliers.
- `phase4_refglow_benchmark.py`: Refglow x001/x002/x005 benchmark with first-order canonical models.
- `phase5_refglow_x009_otor_lw.py`: Refglow x009 deconvolution using OTOR Lambert-W and multi-start seeds.
- `phase6_gdalo_continuous_validation.py`: mixed continuous + localized deconvolution benchmark.
- `phase7_uncertainty_validation.py`: uncertainty validation (analytic vs Monte Carlo) on TLD-100 ROI.

## Figure assembly script

- `make_paper_figures.py`: orchestrates selected phase scripts and copies figure outputs into a paper-friendly directory.

## Minimal smoke-run examples

```bash
python scripts/phase2_benchmark_solvers.py --n-runs 2 --n-points 300 --max-nfev 800
python scripts/phase4_refglow_benchmark.py --strategy local --max-nfev 1200
python scripts/phase7_uncertainty_validation.py --n-mc 30 --max-nfev 1200
```

