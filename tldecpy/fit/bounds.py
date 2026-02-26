"""
Helper functions to generate bounds based on initial guesses.
"""


def make_bounds_from_init(init_params: dict, model_type: str) -> dict:
    """
    Generate constrained bounds around initial estimates.
    """
    _ = model_type
    bounds = {}

    # Tm: +/- 20K is reasonable for a good guess
    if "Tm" in init_params:
        tm = init_params["Tm"]
        bounds["Tm"] = (tm - 30, tm + 30)

    # E: +/- 30% to allow freedom but prevent unphysical drift
    if "E" in init_params:
        e = init_params["E"]
        e_lb = max(0.1, e * 0.5)
        e_ub = min(5.0, e * 1.5)
        if e_lb >= e_ub:
            e_lb, e_ub = 0.1, 5.0
        bounds["E"] = (e_lb, e_ub)

    # Im: 0 to 2x estimate
    if "Im" in init_params:
        im = init_params["Im"]
        bounds["Im"] = (0, im * 2.0)

    # Continuous model characteristic parameters
    if "Tn" in init_params:
        # Continuous-distribution reparametrization: temperature-like parameter
        bounds["Tn"] = (0.0, float("inf"))
    if "In" in init_params:
        # Continuous-distribution reparametrization: intensity-like parameter
        bounds["In"] = (0.0, float("inf"))
    if "E0" in init_params:
        bounds["E0"] = (0.0, 5.0)
    if "sigma" in init_params:
        bounds["sigma"] = (0.001, 1.0)

    # Kinetic Order b
    if "b" in init_params:
        # General order physics: 1 < b <= 2
        bounds["b"] = (1.001, 2.0)

    # OTOR R
    if "R" in init_params:
        # R > 0
        bounds["R"] = (1e-8, 10.0)

    # Mixed-order interpolation parameter
    if "alpha" in init_params:
        bounds["alpha"] = (0.001, 0.999)

    return bounds
