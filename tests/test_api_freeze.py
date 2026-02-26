"""API freeze tests for core public call signatures."""

from __future__ import annotations

import inspect

import tldecpy


def _assert_signature(
    func,
    expected: list[tuple[str, inspect._ParameterKind, object]],
) -> None:
    params = list(inspect.signature(func).parameters.values())
    assert len(params) == len(expected)
    for param, (name, kind, default) in zip(params, expected):
        assert param.name == name
        assert param.kind == kind
        if default is inspect._empty:
            assert param.default is inspect._empty
        else:
            assert param.default == default


def test_fit_multi_signature_frozen() -> None:
    _assert_signature(
        tldecpy.fit_multi,
        [
            ("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect._empty),
            ("y", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect._empty),
            ("peaks", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect._empty),
            ("bg", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
            ("beta", inspect.Parameter.KEYWORD_ONLY, 1.0),
            ("robust", inspect.Parameter.KEYWORD_ONLY, None),
            ("options", inspect.Parameter.KEYWORD_ONLY, None),
        ],
    )


def test_fit_single_peak_signature_frozen() -> None:
    _assert_signature(
        tldecpy.fit_single_peak,
        [
            ("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect._empty),
            ("y", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect._empty),
            ("model", inspect.Parameter.POSITIONAL_OR_KEYWORD, "fo"),
            ("init", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
            ("bounds", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
            ("beta", inspect.Parameter.POSITIONAL_OR_KEYWORD, 1.0),
            ("robust", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        ],
    )


def test_autoinit_signature_frozen() -> None:
    _assert_signature(
        tldecpy.autoinit_multi,
        [
            ("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect._empty),
            ("y", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect._empty),
            ("max_peaks", inspect.Parameter.POSITIONAL_OR_KEYWORD, 6),
            (
                "allow_models",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ("fo", "go", "otor_lw"),
            ),
            ("bg_mode", inspect.Parameter.POSITIONAL_OR_KEYWORD, "auto"),
            ("sensitivity", inspect.Parameter.POSITIONAL_OR_KEYWORD, 1.0),
        ],
    )


def test_simulate_signature_frozen() -> None:
    _assert_signature(
        tldecpy.simulate,
        [
            ("rhs_func", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect._empty),
            ("params", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect._empty),
            ("T0", inspect.Parameter.KEYWORD_ONLY, inspect._empty),
            ("T_end", inspect.Parameter.KEYWORD_ONLY, inspect._empty),
            ("beta", inspect.Parameter.KEYWORD_ONLY, inspect._empty),
            ("y0", inspect.Parameter.KEYWORD_ONLY, inspect._empty),
            ("state_keys", inspect.Parameter.KEYWORD_ONLY, inspect._empty),
            ("method", inspect.Parameter.KEYWORD_ONLY, "LSODA"),
            ("points", inspect.Parameter.KEYWORD_ONLY, 1000),
            ("noise_config", inspect.Parameter.KEYWORD_ONLY, None),
        ],
    )
