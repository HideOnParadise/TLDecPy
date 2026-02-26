import pytest
from tldecpy.data.refglow import load_refglow, list_refglow


def test_list_refglow():
    available_keys = list_refglow()
    assert "x001" in available_keys
    assert "x009" in available_keys


def test_load_refglow_missing():
    with pytest.raises(ValueError):
        _temperature, _intensity = load_refglow("x999")
