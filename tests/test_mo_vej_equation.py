import numpy as np

from tldecpy.models.mixed import mo_vej
from tldecpy.utils.constants import KB_EV


def test_mo_vej_matches_vejnovic_eq20() -> None:
    """
    Verify mo_vej against Eq. (20) from:
    Vejnovic et al., Radiation Measurements 43 (2008) 1325-1330.
    """
    temperature = np.linspace(320.0, 620.0, 500)
    Im = 1400.0
    E = 1.2
    Tm = 470.0
    alpha = 0.45

    model_values = mo_vej(temperature, Im=Im, E=E, Tm=Tm, alpha=alpha)

    order_param = 2.0 / (2.0 - alpha)
    delta = (2.0 * KB_EV * temperature) / E
    sigma_term = (E / (KB_EV * Tm)) * ((temperature - Tm) / Tm)
    exp_sigma = np.exp(np.clip(sigma_term, -50.0, 50.0))

    exp_bracket = np.exp(
        np.clip(
            (temperature**2 / Tm**2) * ((2.0 / order_param) - 1.0) * exp_sigma * (1.0 - delta),
            -50.0,
            50.0,
        )
    )
    expected = (
        alpha
        * Im
        * ((2.0 - order_param) ** 2 / (order_param - 1.0))
        * exp_bracket
        * exp_sigma
        / (exp_bracket - alpha) ** 2
    )

    assert np.allclose(model_values, expected, rtol=1e-12, atol=1e-12)
