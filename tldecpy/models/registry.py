"""Central registry for TL models with rich metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from . import continuous, fo, go, mixed, otor_lw, otor_wo, so

ModelFunc = Callable[..., object]


@dataclass(frozen=True)
class ModelInfo:
    """Metadata and callable for one model variant."""

    key: str
    family: str
    label: str
    description: str
    func: ModelFunc
    aliases: tuple[str, ...] = ()


_MODEL_SPECS: tuple[ModelInfo, ...] = (
    ModelInfo(
        key="cont_gauss",
        family="cont",
        label="CG - Gaussian",
        description="Gaussian trap-energy distribution used in Benavente et al. (2019, Eq. 13), following prior continuous-trap work.",
        func=continuous.continuous_gaussian,
        aliases=(
            "gaussian",
            "continuous_gaussian",
            "continuous.continuous_gaussian",
            "dist.continuous_gaussian",
        ),
    ),
    ModelInfo(
        key="cont_exp",
        family="cont",
        label="CE - Exponential",
        description="Exponential trap-energy distribution used in Benavente et al. (2019, Eq. 14), following prior continuous-trap work.",
        func=continuous.continuous_exponential,
        aliases=(
            "cont_lorentz",
            "lorentz",
            "continuous_exponential",
            "continuous.continuous_exponential",
            "dist.continuous_exponential",
        ),
    ),
    ModelInfo(
        key="fo_rq",
        family="fo",
        label="FO - Rational-Quartic (Kitis)",
        description="Kitis et al. (1998) rational approximation for first-order kinetics.",
        func=fo.fo_rq,
    ),
    ModelInfo(
        key="fo_ka",
        family="fo",
        label="FO - Kinetic-Asymptotic",
        description="Classical asymptotic approximation for isolated first-order peaks (Chen and McKeever).",
        func=fo.fo_ka,
    ),
    ModelInfo(
        key="fo_wp",
        family="fo",
        label="FO - Weibull-Peak",
        description="Empirical first-order peak model based on the Weibull probability density function.",
        func=fo.fo_wp,
    ),
    ModelInfo(
        key="fo_rb",
        family="fo",
        label="FO - Rational-Barycentric (AAA)",
        description="Numerically exact first-order solution using an AAA barycentric rational approximation of the integral.",
        func=fo.fo_rb,
    ),
    ModelInfo(
        key="so_ks",
        family="so",
        label="SO - Kinetic-Standard",
        description="Standard symmetric second-order kinetic equation (Garlick and Gibson, 1948).",
        func=so.so_ks,
    ),
    ModelInfo(
        key="so_la",
        family="so",
        label="SO - Logistic-Asym",
        description="Logistic asymmetric approximation for second-order peaks.",
        func=so.so_la,
    ),
    ModelInfo(
        key="go_kg",
        family="go",
        label="GO - Kinetic-General",
        description="Kitis et al. (1998) general-order kinetic equation.",
        func=go.go_kg,
    ),
    ModelInfo(
        key="go_rq",
        family="go",
        label="GO - Rational-Quadratic",
        description="Rational-quadratic approximation for general-order peaks.",
        func=go.go_rq,
    ),
    ModelInfo(
        key="go_ge",
        family="go",
        label="GO - General-Exponential",
        description="Exponentialized approximation of general-order kinetics.",
        func=go.go_ge,
    ),
    ModelInfo(
        key="mo_kitis",
        family="mo",
        label="MO - Mixed-Kitis",
        description="Mixed-order model based on Kitis and Gomez-Ros (2000).",
        func=mixed.mo_kitis,
    ),
    ModelInfo(
        key="mo_quad",
        family="mo",
        label="MO - Mixed-Quadratic",
        description="Quadratic mixed-order approximation (Gomez-Ros-Kitis style).",
        func=mixed.mo_quad,
    ),
    ModelInfo(
        key="mo_vej",
        family="mo",
        label="MO - Mixed-Vejnovic",
        description="Vejnovic mixed-order kinetic variant.",
        func=mixed.mo_vej,
    ),
    ModelInfo(
        key="otor_lw",
        family="otor",
        label="OTOR - LambertW",
        description="Exact analytical OTOR solution using the Lambert W function (Peng et al., 2016).",
        func=otor_lw.otor_lw,
    ),
    ModelInfo(
        key="otor_wo",
        family="otor",
        label="OTOR - WrightOmega",
        description="Fast numerical OTOR approximation using the Wright Omega function.",
        func=otor_wo.otor_wo,
    ),
)

CANONICAL_MODELS: Dict[str, ModelInfo] = {item.key: item for item in _MODEL_SPECS}

_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for item in _MODEL_SPECS:
    for alias in item.aliases:
        _ALIAS_TO_CANONICAL[alias] = item.key

FAMILY_DEFAULTS: Dict[str, str] = {
    "fo": "fo_rq",
    "so": "so_ks",
    "go": "go_kg",
    "mo": "mo_kitis",
    "mix": "mo_kitis",
    "otor": "otor_lw",
    "cont": "cont_gauss",
    "continuous": "cont_gauss",
    "dist": "cont_gauss",
}

_FAMILY_MEMBERS: Dict[str, tuple[str, ...]] = {
    "fo": ("fo_rq", "fo_rb", "fo_ka", "fo_wp"),
    "so": ("so_ks", "so_la"),
    "go": ("go_kg", "go_rq", "go_ge"),
    "mo": ("mo_kitis", "mo_quad", "mo_vej"),
    "otor": ("otor_lw", "otor_wo"),
    "cont": ("cont_gauss", "cont_exp"),
}

MODEL_REGISTRY: Dict[str, Dict[str, ModelInfo | str]] = {}
for family, members in _FAMILY_MEMBERS.items():
    MODEL_REGISTRY[family] = {"default": FAMILY_DEFAULTS[family]}
    for model_key in members:
        MODEL_REGISTRY[family][model_key] = CANONICAL_MODELS[model_key]


def resolve_model_key(key: str) -> str:
    """Resolve family names and aliases into canonical model keys."""
    normalized = key.strip()
    if not normalized:
        raise ValueError("Model key cannot be empty.")

    if normalized in CANONICAL_MODELS:
        return normalized
    if normalized in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[normalized]
    if normalized in FAMILY_DEFAULTS:
        return FAMILY_DEFAULTS[normalized]

    if "." in normalized:
        family, variant = normalized.split(".", 1)
        family_key = family if family in _FAMILY_MEMBERS else family.lower()
        if family_key in _FAMILY_MEMBERS:
            if variant == "default":
                return FAMILY_DEFAULTS[family_key]
            if variant in CANONICAL_MODELS:
                return variant
            if variant in _ALIAS_TO_CANONICAL:
                return _ALIAS_TO_CANONICAL[variant]

    raise ValueError(
        f"Model key '{key}' not found. Available canonical keys: {list(CANONICAL_MODELS.keys())}"
    )


def get_info(key: str) -> ModelInfo:
    """Return model metadata for any canonical key, alias, or family key."""
    return CANONICAL_MODELS[resolve_model_key(key)]


def get_model_info(key: str) -> ModelInfo:
    """
    Return metadata for a canonical model key, alias, or family key.

    Parameters
    ----------
    key : str
        Canonical key, alias, or family key (for example ``fo``).

    Returns
    -------
    ModelInfo
        Rich model metadata, including callable and display fields.
    """
    return get_info(key)


def get_model(key: str) -> ModelFunc:
    """
    Resolve and return a callable model implementation.

    Parameters
    ----------
    key : str
        Canonical key, alias, or family key.

    Returns
    -------
    collections.abc.Callable
        Numerical model function that maps TL parameters to intensity.
    """
    return get_info(key).func


# Flattened callable map keyed by canonical model IDs.
ALL_VARIANTS: Dict[str, ModelFunc] = {key: info.func for key, info in CANONICAL_MODELS.items()}
for alias, canonical in _ALIAS_TO_CANONICAL.items():
    ALL_VARIANTS[alias] = CANONICAL_MODELS[canonical].func

# Family-prefixed convenience forms.
ALL_VARIANTS["fo.fo_rq"] = CANONICAL_MODELS["fo_rq"].func
ALL_VARIANTS["fo.fo_rb"] = CANONICAL_MODELS["fo_rb"].func
ALL_VARIANTS["go.go_kg"] = CANONICAL_MODELS["go_kg"].func
ALL_VARIANTS["mo.mo_kitis"] = CANONICAL_MODELS["mo_kitis"].func
ALL_VARIANTS["cont.cont_gauss"] = CANONICAL_MODELS["cont_gauss"].func
ALL_VARIANTS["cont.cont_exp"] = CANONICAL_MODELS["cont_exp"].func


def get_order_from_key(key: str) -> str:
    """
    Return internal order family used by the fitting pipeline.

    Note:
    - ``cont`` maps to ``continuous``.
    - ``mo`` maps to ``mix`` for backward compatibility with existing fitter logic.
    """
    family = get_info(key).family
    if family == "cont":
        return "continuous"
    if family == "mo":
        return "mix"
    return family


def list_models(order: str | None = None, include_aliases: bool = False) -> List[str]:
    """
    List available model keys for TL kinetic families.

    Parameters
    ----------
    order : str | None, optional
        Optional family filter (``fo``, ``so``, ``go``, ``mo``, ``otor``, ``cont``).
        Aliases (``mix``, ``continuous``, ``dist``) are also accepted.
    include_aliases : bool, default=False
        When ``True``, include legacy alias keys in the output.

    Returns
    -------
    list[str]
        Canonical model keys (and optional aliases).
    """
    if order is None:
        keys = list(CANONICAL_MODELS.keys())
        if include_aliases:
            keys.extend(sorted(_ALIAS_TO_CANONICAL.keys()))
        return keys

    family_key = order
    if family_key in {"mix"}:
        family_key = "mo"
    if family_key in {"continuous", "dist"}:
        family_key = "cont"

    if family_key not in _FAMILY_MEMBERS:
        raise ValueError(f"Unknown order: {order}. Try: {list(_FAMILY_MEMBERS.keys())}")

    keys = list(_FAMILY_MEMBERS[family_key])
    if include_aliases:
        alias_keys = [
            alias
            for alias, canonical in _ALIAS_TO_CANONICAL.items()
            if CANONICAL_MODELS[canonical].family == family_key
        ]
        keys.extend(sorted(alias_keys))
    return keys
