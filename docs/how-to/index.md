# How-to guides

Task-focused recipes.  Each guide assumes you know the basics —
see the [Tutorials](../tutorials/index.md) first.

| Guide | Task |
|---|---|
| [One-shot automatic deconvolution](iterative_deconvolution.md) | Run detection, seeding, and fitting in a single call with `iterative_deconvolution` |
| [Manual peak setup](manual_peak_setup.md) | Build `PeakSpec` objects by hand without `autoinit_multi` |
| [Robust fitting and loss functions](robust_fitting.md) | Choose the right `RobustOptions` for your noise model |
| [Uncertainty budget](uncertainty_budget.md) | Configure `UncertaintyOptions` and interpret `uc_curve` |
| [Fix and freeze parameters](fix_parameters.md) | Use `PeakSpec.fixed` to hold parameters constant |
| [Synthetic curves with noise](simulate_synthetic.md) | Generate reference data with Poisson noise and fit with matching weights |
| [Serialize and archive results](../serialization.md) | Export and reload `MultiFitResult` as JSON for reproducible workflows |
