import pytest

from dash_app import _parse_custom_value


def test_parse_custom_value_accepts_none_and_positive_numbers():
    assert _parse_custom_value(None) is None
    assert _parse_custom_value("") is None
    assert _parse_custom_value(12.5) == pytest.approx(12.5)
    assert _parse_custom_value("3.25") == pytest.approx(3.25)


def test_parse_custom_value_rejects_non_positive_numbers():
    with pytest.raises(ValueError):
        _parse_custom_value(0)

    with pytest.raises(ValueError):
        _parse_custom_value(-1)
