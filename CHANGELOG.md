# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New first-order model variant `fo_rb` registered separately from `fo_rq`, using the same FO core with AAA barycentric evaluation of `alpha(z)` via `Q(z)=exp(z)E2(z)`.

## [1.0.0] - 2025-11-20

### Added
- **Stable Public API:** Core API surface frozen for v1.0.0.
- **Pydantic Data Models:** Configuration objects (`PeakSpec`) and results (`MultiFitResult`) use immutable Pydantic v2 models with stable JSON serialization contracts.
- **Modern Packaging (PEP 621):** `pyproject.toml` updated to `hatchling` with standardized metadata.
- **Typing Support (PEP 561):** The package now ships type information via `py.typed` for `mypy`.
- **Documentation:** Full docs site built with `mkdocs-material`.
- **CI/CD:** GitHub Actions pipelines for tests, linting, and automatic PyPI publishing with Trusted Publishing.
- **Mixed-Order Kinetics Models:** Added `mo_kitis`, `mo_quad`, and `mo_vej`.
- **ODE Simulation:** Added `tldecpy.simulate` module for synthetic curve generation via ODE integration.
- **Robust Fitting:** Added robust losses (`soft_l1`, `huber`, `tukey`) and weighting (`poisson`).
- **Automatic Initialization:** Added `autoinit_multi` for automated peak detection and initialization.
