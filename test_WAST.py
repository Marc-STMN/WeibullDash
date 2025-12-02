import numpy as np
import pandas as pd
import pytest
from io import BytesIO
from scipy.stats import weibull_min

from WAST import (
    calculate_weibull_parameters,
    extract_data,
    load_data,
    load_parameter,
    plot_weibull,
    render_plot_to_png_bytes,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return weibull_min.rvs(c=10, scale=100, size=30)


@pytest.fixture
def sample_excel():
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_params = pd.DataFrame([
            ["Parameter", "Value", "Unit"],
            ["Werkstoff", "ZrO2", "MPa"],
            ["Auftrags-Nr.", "TEST-123", ""],
        ])
        df_params.to_excel(writer, sheet_name="Parameter", index=False, header=False)

        df_results = pd.DataFrame([
            ["sc", "Fmax"],
            ["MPa", "N"],
            [500, 100],
            [550, 110],
            [600, 120],
            [650, 130],
        ])
        df_results.to_excel(writer, sheet_name="Ergebnisse", index=False, header=False)
    output.seek(0)
    return output


def test_calculate_weibull_parameters(sample_data):
    shape, scale, unbiased_shape, ci_shape, ci_scale, d_stat, p_val = calculate_weibull_parameters(
        sample_data, 0.95
    )
    assert isinstance(shape, float)
    assert isinstance(scale, float)
    assert ci_shape[0] < ci_shape[1]
    assert ci_scale[0] < ci_scale[1]
    assert 0 <= d_stat <= 2
    assert 0 <= p_val <= 1


def test_calculate_weibull_parameters_invalid_data():
    with pytest.raises(ValueError):
        calculate_weibull_parameters([], 0.95)
    with pytest.raises(ValueError):
        calculate_weibull_parameters([0, -1, 2], 0.95)


def test_extract_and_load(sample_excel):
    id_val, id_unit, df_data = load_data(sample_excel, ["sc", "Fmax"])
    assert str(id_val).strip() == "sc"
    assert str(id_unit).strip() == "MPa"

    values, col, sym, title = extract_data(df_data)
    assert isinstance(values, np.ndarray)
    assert col in ["sc", "Fmax"]
    assert isinstance(sym, str)
    assert isinstance(title, str)


def test_load_parameter(sample_excel):
    param = load_parameter(sample_excel, "Werkstoff")
    assert param == "MPa"


def test_plot_and_render(sample_excel):
    df = pd.read_excel(sample_excel, sheet_name="Ergebnisse", header=None)
    values, col, sym, title = extract_data(df)
    shape, scale, unbiased_shape, ci_shape, ci_scale, d_stat, p_val = calculate_weibull_parameters(
        values, 0.95
    )
    p_lin = np.linspace(0.01, 0.99, len(values))
    lower_ci = weibull_min.ppf(p_lin, c=ci_shape[0], scale=ci_scale[0])
    upper_ci = weibull_min.ppf(p_lin, c=ci_shape[1], scale=ci_scale[1])
    fig = plot_weibull(
        values,
        "TestLabel",
        "unit",
        scale,
        unbiased_shape,
        lower_ci,
        upper_ci,
        None,
        None,
        "",
        sym,
        title,
        d_stat,
        p_val,
        0.95,
        ci_shape,
        ci_scale,
    )
    img_bytes = render_plot_to_png_bytes(fig)
    assert isinstance(img_bytes, (bytes, bytearray))
    assert len(img_bytes) > 0


def test_load_parameter_missing(sample_excel):
    with pytest.raises(ValueError):
        load_parameter(sample_excel, "NichtVorhanden")
