import base64
from io import BytesIO
from pathlib import Path
import zipfile

import pytest

import dash_app
from dash_app import (
    _analysis_context,
    _build_analysis_payload,
    _build_download_bundle,
    _build_interactive_figure,
    _parse_custom_value,
    _toggle_excluded_index,
)
import version
from version import _normalize_git_describe


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


def test_build_download_bundle_contains_png_txt_json_and_csv():
    png_bytes = b"fake-png-content"
    analysis_data = {
        "summary": {
            "parameter_key": "Werkstoff",
            "parameter_value": "ZY",
            "order_number": "TTG-CMaC-0039",
            "measurement_series": "sc",
            "unit": "MPa",
            "n": 3,
            "shape_mle": 10.0,
            "scale_mle": 500.0,
            "unbiased_shape": 9.8,
            "ci_shape": [8.5, 11.2],
            "ci_scale": [470.0, 530.0],
            "ad_statistic": 0.12,
            "p_value": 0.45,
            "bootstrap_samples": 500,
            "p_value_method": "Parametric bootstrap AD under fitted Weibull model",
            "ci_method": "Wald confidence intervals from inverse Hessian",
            "confidence_level": 95,
            "custom_value": None,
            "failure_probability": None,
            "code_version": "test-version",
        },
        "raw_data": [480.0, 500.0, 520.0],
        "plot_png": base64.b64encode(png_bytes).decode(),
    }

    bundle = _build_download_bundle(analysis_data, "en")

    with zipfile.ZipFile(BytesIO(bundle), "r") as zf:
        names = set(zf.namelist())
        assert "ZY_plot.png" in names
        assert "ZY_results.txt" in names
        assert "ZY_results.json" in names
        assert "ZY_raw_data.csv" in names
        assert zf.read("ZY_plot.png") == png_bytes
        assert b"index,value" in zf.read("ZY_raw_data.csv")
        assert b"Parameter label: Material" in zf.read("ZY_results.txt")


def test_normalize_git_describe_formats_version_strings():
    assert _normalize_git_describe("v4.1.0\n") == "4.1.0"
    assert _normalize_git_describe("4.1.0-3-gabc1234") == "4.1.0+3.gabc1234"
    assert _normalize_git_describe("4.1.0-3-gabc1234-dirty") == "4.1.0+3.gabc1234.dirty"
    assert _normalize_git_describe("abc1234") == "0+gabc1234"
    assert _normalize_git_describe("abc1234-dirty") == "0+gabc1234.dirty"


def test_layout_uses_current_version(monkeypatch):
    monkeypatch.setattr(dash_app, "get_version", lambda: "9.9.9")
    app, _ = dash_app.create_app()
    layout = app.layout()
    assert layout.children[-1].children == "Version 9.9.9"


def test_resolve_version_uses_version_file_when_git_unavailable(monkeypatch):
    version_file = Path.cwd() / "VERSION.test"
    version_file.write_text("4.2.0\n", encoding="utf-8")
    try:
        monkeypatch.setattr(version, "_VERSION_FILE", version_file)
        monkeypatch.setattr(version, "_version_from_git", lambda: None)
        monkeypatch.delenv("WEIBULL_TOOL_VERSION", raising=False)

        assert version.get_version() == "4.2.0"
    finally:
        if version_file.exists():
            version_file.unlink()


def test_toggle_excluded_index_preserves_minimum_two_points():
    assert _toggle_excluded_index([], 1, 4) == [1]
    assert _toggle_excluded_index([1], 1, 4) == []
    assert _toggle_excluded_index([0, 1], 2, 4) == [0, 1]


def test_build_analysis_payload_tracks_exclusions():
    context = _analysis_context("Werkstoff", "ZrO2", "TEST-1", "sc", "MPa", "s", 0.95, None, "")
    payload = _build_analysis_payload([480.0, 500.0, 520.0, 540.0], context, [1], "en")

    assert payload["excluded_indices"] == [1]
    assert payload["summary"]["excluded_count"] == 1
    assert payload["summary"]["n"] == 3
    assert payload["raw_data"] == [480.0, 520.0, 540.0]


def test_build_interactive_figure_contains_excluded_trace():
    analysis_data = {
        "summary": {
            "parameter_key": "Werkstoff",
            "parameter_value": "ZY",
            "order_number": "TTG-CMaC-0039",
            "measurement_series": "sc",
            "unit": "MPa",
            "n": 3,
            "shape_mle": 10.0,
            "scale_mle": 500.0,
            "unbiased_shape": 9.8,
            "ci_shape": [8.5, 11.2],
            "ci_scale": [470.0, 530.0],
            "ad_statistic": 0.12,
            "p_value": 0.45,
            "bootstrap_samples": 500,
            "p_value_method": "Parametric bootstrap AD under fitted Weibull model",
            "ci_method": "Wald confidence intervals from inverse Hessian",
            "confidence_level": 95,
            "custom_value": None,
            "failure_probability": None,
            "comment": "",
            "code_version": "test-version",
            "excluded_count": 1,
        },
        "raw_data": [480.0, 500.0, 540.0],
        "source_data": [480.0, 500.0, 520.0, 540.0],
        "excluded_indices": [2],
    }

    fig = _build_interactive_figure(analysis_data, "en")

    assert len(fig.data) >= 4
    assert any(trace.name.startswith("Excluded points") for trace in fig.data)
