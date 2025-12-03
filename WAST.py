"""
Weibull Analysis Support Tools
Utility functions for Weibull analysis reused by the Dash UI.
"""

import os
from datetime import datetime
from io import BytesIO

# Force a non-interactive backend early to avoid GUI/thread issues (Dash callbacks, background threads)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import pandas as pd
from scipy.stats import norm, weibull_min
from numpy.typing import ArrayLike

__version__ = "4.0.0"

# Random generator for reproducibility
np.random.seed(42)


def load_data(file_object, search_keys, search_row=0):
    """Read data and metadata from the 'Ergebnisse' sheet.

    Returns id_value, id_unit, and the raw DataFrame. Raises ValueError on errors.
    """
    try:
        excel_file = pd.ExcelFile(file_object)
        df_data = excel_file.parse("Ergebnisse", header=None)
    except Exception as exc:  # pragma: no cover - IO errors are rare in tests
        raise ValueError(f"Error reading 'Ergebnisse' sheet: {type(exc).__name__} - {exc}")

    if isinstance(search_keys, str):
        search_keys = [search_keys]

    for key in search_keys:
        row_values = df_data.iloc[search_row, :].astype(str).str.strip()
        matching_cols = row_values[row_values == key.strip()]
        if not matching_cols.empty:
            col_index = matching_cols.index[0]
            id_value = df_data.iloc[search_row, col_index]
            id_unit = df_data.iloc[search_row + 1, col_index] if search_row + 1 < len(df_data) else "X"
            if pd.isna(id_unit) or str(id_unit).strip() == "":
                id_unit = "X"
            return id_value, id_unit, df_data

    raise ValueError("Search key not found in Ergebnisse.")


def extract_data(df_data):
    """Extract numeric series and metadata from the results sheet."""
    df_data.columns = df_data.iloc[0]
    df_data = df_data.drop(0).reset_index(drop=True)
    column_mapping = {
        "sc": ("I_sigma0", "flexural strength in MPa"),
        "Fmax": ("F_0", "fracture load in N"),
    }
    for column_name, (symbol, title) in column_mapping.items():
        if column_name in df_data.columns:
            values = pd.to_numeric(df_data[column_name], errors="coerce").dropna().values
            return values, column_name, symbol, title

    raise ValueError("Error: Neither 'Fmax' nor 'sc' found in the DataFrame.")


def load_parameter(file_object, search_key, target_row=1):
    """Retrieve a parameter from the 'Parameter' sheet."""
    try:
        excel_file = pd.ExcelFile(file_object)
        df_param = excel_file.parse("Parameter", header=None)
    except Exception as exc:  # pragma: no cover - IO errors are rare in tests
        raise ValueError(f"Error reading 'Parameter' sheet: {type(exc).__name__} - {exc}")

    for idx, cell in enumerate(df_param.iloc[:, 0]):
        if str(cell).strip() == search_key:
            return str(df_param.iloc[idx, 2]).strip() if df_param.shape[1] > 2 else ""

    raise ValueError("Search term not found in the parameter sheet.")


def calculate_weibull_parameters(data, alpha):
    """Perform Weibull MLE estimation and Wald confidence intervals."""
    if len(data) < 2:
        raise ValueError("Need at least 2 data points for Weibull analysis")
    data = np.array(data, dtype=np.float64)
    if np.any(data <= 0):
        raise ValueError("Weibull analysis requires positive values")

    shape_mle, _, scale_mle = weibull_min.fit(data, floc=0)

    def nll(params):
        k, lam = params
        return -np.sum(weibull_min.logpdf(data, c=k, scale=lam))

    hess = nd.Hessian(lambda p: nll(p))([shape_mle, scale_mle])
    cov = np.linalg.inv(hess)
    se_shape, se_scale = np.sqrt(np.diag(cov))

    n = len(data)
    unbias = 1 - 1.61394 * n ** -1.04033
    unbiased_shape = shape_mle * unbias

    z = norm.ppf(1 - (1 - alpha) / 2)
    ci_shape = (
        unbiased_shape - z * se_shape * unbias,
        unbiased_shape + z * se_shape * unbias,
    )
    ci_scale = (scale_mle - z * se_scale, scale_mle + z * se_scale)

    def ad_statistic(sample, shape, scale):
        x = np.sort(sample)
        n0 = len(x)
        probs = weibull_min.cdf(x, c=shape, scale=scale)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        i = np.arange(1, n0 + 1)
        s_val = np.sum((2 * i - 1) * (np.log(probs) + np.log(1 - probs[::-1])))
        return -n0 - s_val / n0

    ad_obs = ad_statistic(data, shape_mle, scale_mle)

    n_boot = 500
    rng = np.random.default_rng(seed=42)
    boot_stats = np.empty(n_boot)
    for j in range(n_boot):
        samp = rng.choice(data, size=n, replace=True)
        k_b, _, lam_b = weibull_min.fit(samp, floc=0)
        boot_stats[j] = ad_statistic(samp, k_b, lam_b)
    p_val_ad = (boot_stats >= ad_obs).mean()

    return shape_mle, scale_mle, unbiased_shape, ci_shape, ci_scale, ad_obs, p_val_ad


def plot_weibull(
    data,
    data_label,
    id_unit,
    scale,
    unbiased_shape,
    lower_ci,
    upper_ci,
    custom_value,
    failure_prob,
    user_comment,
    data_symbol,
    axis_title,
    d_statistic,
    p_value,
    alpha,
    ci_shape,
    ci_scale,
):
    """Generate Weibull plot with Wald confidence intervals."""
    sorted_data = np.sort(data)
    empirical_probs = (np.arange(1, len(data) + 1) - 0.5) / len(data)
    p = np.linspace(0.01, 0.99, len(lower_ci))
    weibull_quantiles = weibull_min.ppf(p, c=unbiased_shape, scale=scale)

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.log(sorted_data),
        np.log(-np.log(1 - empirical_probs)),
        "x",
        label=f"{data_label}, n={len(data)}",
    )

    label_text = (
        f"unbiased Weibull modulus m = {unbiased_shape:.1f}\n"
        f"characteristic value {data_symbol} = {scale:.0f} {id_unit}"
    )
    plt.plot(np.log(weibull_quantiles), np.log(-np.log(1 - p)), "r-", label=label_text)

    q_lo = weibull_min.ppf(p, c=ci_shape[0], scale=ci_scale[0])
    q_lo_hi = weibull_min.ppf(p, c=ci_shape[1], scale=ci_scale[0])
    q_hi = weibull_min.ppf(p, c=ci_shape[1], scale=ci_scale[1])
    q_hi_hi = weibull_min.ppf(p, c=ci_shape[0], scale=ci_scale[1])
    threshold = 1 - np.exp(-1)
    below_threshold = p <= threshold
    above_threshold = p > threshold
    lower_ci_curve = np.where(below_threshold, q_lo, q_hi)
    lower_ci_curve = np.where(above_threshold, q_hi_hi, q_hi)
    upper_ci_curve = np.where(below_threshold, q_hi, q_lo)
    upper_ci_curve = np.where(above_threshold, q_lo_hi, q_lo)

    x_vals: ArrayLike = np.log(-np.log(1 - p))
    lower_log: ArrayLike = np.log(lower_ci_curve)
    upper_log: ArrayLike = np.log(upper_ci_curve)
    plt.plot(lower_log, x_vals, "--", color="grey", linewidth=1)
    plt.plot(upper_log, x_vals, "-", color="grey", linewidth=1)
    plt.fill_betweenx(
        x_vals,
        lower_log,
        upper_log,
        color="grey",
        alpha=0.2,
    )

    if custom_value is not None and failure_prob is not None:
        plt.axvline(
            np.log(custom_value),
            color="blue",
            linestyle="--",
            linewidth=0.5,
            label=f"Failure prob at {custom_value:.0f} {id_unit} = {failure_prob*100:.2f}%",
        )
        y_fp = np.log(-np.log(1 - failure_prob))
        plt.axhline(y=y_fp, color="blue", linestyle="--", linewidth=0.5)

    if user_comment:
        plt.text(
            0.02,
            0.75,
            user_comment,
            transform=plt.gca().transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgrey", alpha=0.8),
        )

    plt.title(
        f"Weibull Plot with {int(alpha*100)}% Confidence Interval (ISO 20501)",
        fontsize=10,
    )
    plt.xlabel(axis_title)
    plt.ylabel("Failure Probability (%)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    x_min, x_max = np.min(lower_ci), np.max(upper_ci)
    interval = 100 if x_max > 700 else 50
    ticks = np.arange(np.floor(x_min / interval) * interval, np.ceil(x_max / interval) * interval + interval, interval)
    plt.xticks(np.log(ticks), [f"{int(t)}" for t in ticks], fontsize=8)

    probs_std = np.array([0.01, 0.05, 0.10, 0.20, 0.40, 0.6325, 0.80, 0.95, 0.99])
    yticks = np.log(-np.log(1 - probs_std))
    plt.yticks(yticks, [f"{p*100:.1f}" for p in probs_std], fontsize=8)

    plt.legend(loc="upper left", fontsize=8)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.text(
        1.02,
        1.00,
        f"Code version: {__version__} / Plot generated on: {current_date}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        rotation=90,
        color="grey",
        fontsize=6,
    )
    plt.text(
        0.97,
        0.03,
        f"AD-stat: {d_statistic:.3f} / p-value: {p_value:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=6,
        color="grey",
        bbox=dict(facecolor="white", edgecolor="white", alpha=1.0),
    )

    return plt.gcf()


def render_plot_to_png_bytes(fig):
    """Convert a Matplotlib figure to PNG bytes for download/embedding."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()


__all__ = [
    "load_data",
    "extract_data",
    "load_parameter",
    "calculate_weibull_parameters",
    "plot_weibull",
    "render_plot_to_png_bytes",
    "__version__",
]
