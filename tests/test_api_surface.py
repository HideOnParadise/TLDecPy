import tldecpy


def test_public_api_exposed():
    """Ensure key functions are exposed at top level."""
    expected = [
        "fit_single_peak",
        "fit_multi",
        "simulate",
        "autoinit_multi",
        "list_refglow",
        "load_refglow",
    ]
    for name in expected:
        assert hasattr(tldecpy, name), f"Missing public API: {name}"


def test_schemas_exposed():
    """Ensure Pydantic models are exposed."""
    assert hasattr(tldecpy, "MultiFitResult")
    assert hasattr(tldecpy, "PeakSpec")
    assert hasattr(tldecpy, "RobustOptions")


def test_version_info():
    info = tldecpy.get_version_info()
    assert info.version == tldecpy.__version__
    assert info.api_version == tldecpy.__version__
