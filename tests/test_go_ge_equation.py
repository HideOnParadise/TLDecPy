import numpy as np

from tldecpy.models.go import go_ge
from tldecpy.utils.constants import KB_EV


def test_go_ge_matches_closed_form_equation() -> None:
    """
    Validate go_ge against its analytical closed-form equation.

    Equation checked:
        I(T) = Im * exp(A) * (1/b + ((b-1)/b)*exp(A))^(-b/(b-1))
    where:
        A = E/(k*Tm^2) * (T - Tm)
    """
    temperature = np.linspace(320.0, 620.0, 401)
    Im = 2500.0
    E = 1.24
    Tm = 470.0
    b = 1.45

    model_values = go_ge(temperature, Im=Im, E=E, Tm=Tm, b=b)

    a_term = (E / (KB_EV * Tm**2)) * (temperature - Tm)
    exp_term = np.exp(np.clip(a_term, -50.0, 50.0))
    expected_values = Im * exp_term * (
        (1.0 / b) + ((b - 1.0) / b) * exp_term
    ) ** (-b / (b - 1.0))

    assert np.all(np.isfinite(model_values))
    assert np.allclose(model_values, expected_values, rtol=1e-12, atol=1e-12)
