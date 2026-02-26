"""Contract tests for model-key resolution."""

from __future__ import annotations

import pytest

from tldecpy.models.registry import ALL_VARIANTS, list_models, resolve_model_key


def test_legacy_tgcd_style_aliases_are_rejected() -> None:
    """
    Legacy TGCD-style aliases must remain unsupported.

    This protects the canonical naming migration and avoids accidental reintroduction.
    """
    legacy_keys = [
        "f1",
        "f2",
        "f3",
        "s1",
        "s2",
        "g1",
        "g2",
        "g3",
        "m1",
        "m2",
        "m3",
        "kgr2000",
        "grk2002",
        "vpd2008",
    ]
    for key in legacy_keys:
        with pytest.raises(ValueError, match="not found"):
            resolve_model_key(key)


def test_web_compatibility_aliases_are_preserved() -> None:
    """
    Aliases currently consumed by TLDecWeb must continue resolving.
    """
    assert resolve_model_key("cont_lorentz") == "cont_exp"
    assert resolve_model_key("continuous_gaussian") == "cont_gauss"
    assert resolve_model_key("continuous_exponential") == "cont_exp"


def test_list_models_defaults_to_canonical_keys() -> None:
    """Default model listing should only expose canonical keys."""
    model_keys = list_models()
    assert "fo_rq" in model_keys
    assert "go_kg" in model_keys
    assert "cont_exp" in model_keys

    for legacy_key in ("f1", "f2", "g1", "s1", "m1", "kgr2000"):
        assert legacy_key not in model_keys


def test_all_variants_does_not_reintroduce_legacy_tgcd_keys() -> None:
    """Global callable map must not include removed TGCD-like aliases."""
    variant_keys = set(ALL_VARIANTS.keys())
    forbidden = {
        "f1",
        "f2",
        "f3",
        "s1",
        "s2",
        "g1",
        "g2",
        "g3",
        "m1",
        "m2",
        "m3",
        "kgr2000",
        "grk2002",
        "vpd2008",
    }
    assert forbidden.isdisjoint(variant_keys)
