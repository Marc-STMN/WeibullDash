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
from version import __version__, get_version

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
        "sc": ("I_sigma0", "Biegefestigkeit in MPa"),
        "Fmax": ("F_0", "Bruchlast in N"),
    }
    for column_name, (symbol, title) in column_mapping.items():
        if column_name in df_data.columns:
            values = pd.to_numeric(df_data[column_name], errors="coerce").dropna().values
            return values, column_name, symbol, title

    raise ValueError("Weder 'Fmax' noch 'sc' wurde in der Tabelle gefunden.")


def load_parameter(file_object, search_key, target_col=None):
    """Retrieve a parameter value from the 'Parameter' sheet."""
    try:
        excel_file = pd.ExcelFile(file_object)
        df_param = excel_file.parse("Parameter", header=None)
    except Exception as exc:  # pragma: no cover - IO errors are rare in tests
        raise ValueError(f"Error reading 'Parameter' sheet: {type(exc).__name__} - {exc}")

    for idx, cell in enumerate(df_param.iloc[:, 0]):
        if str(cell).strip() == search_key:
            if target_col is not None:
                return str(df_param.iloc[idx, target_col]).strip() if df_param.shape[1] > target_col else ""

            row_values = df_param.iloc[idx, 1:]
            for value in row_values:
                if pd.notna(value) and str(value).strip():
                    return str(value).strip()
            return ""

    raise ValueError("Search term not found in the parameter sheet.")


def list_parameter_keys(file_object):
    """Return available parameter keys from the 'Parameter' sheet."""
    try:
        excel_file = pd.ExcelFile(file_object)
        df_param = excel_file.parse("Parameter", header=None)
    except Exception as exc:  # pragma: no cover - IO errors are rare in tests
        raise ValueError(f"Error reading 'Parameter' sheet: {type(exc).__name__} - {exc}")

    keys = []
    for idx, cell in enumerate(df_param.iloc[:, 0]):
        if pd.isna(cell):
            continue
        key = str(cell).strip()
        if not key or key.lower() == "parameter":
            continue
        row_values = df_param.iloc[idx, 1:]
        has_value = any(pd.notna(value) and str(value).strip() for value in row_values)
        if has_value and key not in keys:
            keys.append(key)
    return keys


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
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if Hessian is ill-conditioned
        cov = np.linalg.pinv(hess)
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
        # Parametric bootstrap under the fitted Weibull model for the AD goodness-of-fit test.
        samp = weibull_min.rvs(c=shape_mle, scale=scale_mle, size=n, random_state=rng)
        k_b, _, lam_b = weibull_min.fit(samp, floc=0)
        boot_stats[j] = ad_statistic(samp, k_b, lam_b)
    p_val_ad = (boot_stats >= ad_obs).mean()

    return shape_mle, scale_mle, unbiased_shape, ci_shape, ci_scale, ad_obs, p_val_ad, n_boot


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
    language="de",
):
    """Generate Weibull plot with Wald confidence intervals."""
    sorted_data = np.sort(data)
    empirical_probs = (np.arange(1, len(data) + 1) - 0.5) / len(data)
    p = np.linspace(0.01, 0.99, len(lower_ci))
    weibull_quantiles = weibull_min.ppf(p, c=unbiased_shape, scale=scale)
    plot_color = "#0f766e"
    fit_color = "#d97706"
    ci_color = "#94a3b8"
    accent_color = "#0284c7"
    text = {
        "de": {
            "fit": "Fit: m = {m:.1f}, Kennwert = {scale:.0f} {unit}",
            "band": "{confidence} %-Konfidenzband",
            "custom": "P(Ausfall) bei {value:.0f} {unit} = {prob:.2f} %",
            "title": "Weibull-Diagramm mit {confidence} %-Konfidenzband",
            "ylabel": "Ausfallwahrscheinlichkeit (%)",
            "footer_right": "AD = {ad:.3f} | p = {p_value:.3f}",
            "footer_left": "Version {version} | Erstellt am {date}",
        },
        "en": {
            "fit": "Fit: m = {m:.1f}, characteristic value = {scale:.0f} {unit}",
            "band": "{confidence}% confidence band",
            "custom": "P(failure) at {value:.0f} {unit} = {prob:.2f} %",
            "title": "Weibull plot with {confidence}% confidence band",
            "ylabel": "Failure probability (%)",
            "footer_right": "AD = {ad:.3f} | p = {p_value:.3f}",
            "footer_left": "Version {version} | Created on {date}",
        },
    }.get(language if language in {"de", "en"} else "de")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        np.log(sorted_data),
        np.log(-np.log(1 - empirical_probs)),
        linestyle="None",
        marker="x",
        markersize=5.5,
        markeredgewidth=1.2,
        color=plot_color,
        label=f"{data_label} | n = {len(data)}",
    )

    fit_label = text["fit"].format(m=unbiased_shape, scale=scale, unit=id_unit)
    ax.plot(
        np.log(weibull_quantiles),
        np.log(-np.log(1 - p)),
        color=fit_color,
        linewidth=1.8,
        label=fit_label,
    )

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
    ax.plot(lower_log, x_vals, "--", color="grey", linewidth=1)
    ax.plot(upper_log, x_vals, "-", color="grey", linewidth=1)
    ax.fill_betweenx(
        x_vals,
        lower_log,
        upper_log,
        color=ci_color,
        alpha=0.22,
        label=text["band"].format(confidence=int(alpha * 100)),
    )

    if custom_value is not None and failure_prob is not None:
        ax.axvline(
            np.log(custom_value),
            color=accent_color,
            linestyle="--",
            linewidth=0.9,
            label=text["custom"].format(value=custom_value, unit=id_unit, prob=failure_prob * 100),
        )
        y_fp = np.log(-np.log(1 - failure_prob))
        ax.axhline(y=y_fp, color=accent_color, linestyle="--", linewidth=0.9)

    if user_comment:
        ax.text(
            0.02,
            0.72,
            user_comment,
            transform=ax.transAxes,
            fontsize=8,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffbeb", edgecolor="#fbbf24", alpha=0.95),
        )

    ax.set_title(
        text["title"].format(confidence=int(alpha * 100)),
        fontsize=11,
        pad=12,
    )
    ax.set_xlabel(axis_title)
    ax.set_ylabel(text["ylabel"])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#cbd5e1", alpha=0.8)

    axis_candidates = [
        np.asarray(sorted_data, dtype=float),
        np.asarray(lower_ci_curve, dtype=float),
        np.asarray(upper_ci_curve, dtype=float),
        np.asarray(weibull_quantiles, dtype=float),
    ]
    if custom_value is not None:
        axis_candidates.append(np.asarray([custom_value], dtype=float))
    finite_axis = np.concatenate(axis_candidates)
    finite_axis = finite_axis[np.isfinite(finite_axis) & (finite_axis > 0)]
    if finite_axis.size:
        x_min = float(np.min(finite_axis))
        x_max = float(np.max(finite_axis))
        display_min, display_max = _expand_axis_bounds(x_min, x_max)
        ax.set_xlim(np.log(display_min), np.log(display_max))

        ticks = _build_axis_ticks(display_min, display_max)
        if ticks.size:
            ax.set_xticks(np.log(ticks))
            ax.set_xticklabels([_format_tick_label(tick) for tick in ticks], fontsize=8)

    probs_std = np.array([0.01, 0.05, 0.10, 0.20, 0.40, 0.6325, 0.80, 0.95, 0.99])
    yticks = np.log(-np.log(1 - probs_std))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{p*100:.1f}" for p in probs_std], fontsize=8)

    ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor="white", edgecolor="#cbd5e1")
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.99,
        0.055,
        text["footer_right"].format(ad=d_statistic, p_value=p_value),
        ha="right",
        va="center",
        fontsize=7,
        color="#475569",
    )
    fig.text(
        0.01,
        0.055,
        text["footer_left"].format(version=get_version(), date=current_date),
        ha="left",
        va="center",
        fontsize=7,
        color="#64748b",
    )

    return fig


def render_plot_to_png_bytes(fig):
    """Convert a Matplotlib figure to PNG bytes for download/embedding."""
    buffer = BytesIO()
    try:
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        return buffer.read()
    finally:
        plt.close(fig)


def _expand_axis_bounds(x_min, x_max):
    x_min = float(x_min)
    x_max = float(x_max)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min <= 0 or x_max <= 0:
        return x_min, x_max
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if np.isclose(x_min, x_max):
        return x_min * 0.9, x_max * 1.1

    log_min = np.log(x_min)
    log_max = np.log(x_max)
    padding = max((log_max - log_min) * 0.06, 0.08)
    return float(np.exp(log_min - padding)), float(np.exp(log_max + padding))


def _build_axis_ticks(x_min, x_max, max_ticks=9):
    x_min = float(x_min)
    x_max = float(x_max)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min <= 0 or x_max <= 0:
        return np.array([])
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if np.isclose(x_min, x_max):
        return np.array([x_min])

    axis_range = x_max - x_min
    step = _nice_number(axis_range / max(max_ticks - 1, 1), round_result=True)
    if step <= 0:
        return np.array([x_min, x_max])

    tick_start = np.floor(x_min / step) * step
    if tick_start <= 0:
        tick_start = step
    tick_end = np.ceil(x_max / step) * step
    tick_values = np.arange(tick_start, tick_end + step * 0.5, step)
    tick_values = tick_values[np.isfinite(tick_values) & (tick_values > 0)]

    if tick_values.size < 2:
        return np.array([x_min, x_max])
    return tick_values


def _nice_number(value, round_result):
    if not np.isfinite(value) or value <= 0:
        return 0.0

    exponent = np.floor(np.log10(value))
    fraction = value / (10 ** exponent)
    if round_result:
        if fraction < 1.5:
            nice_fraction = 1.0
        elif fraction < 2.25:
            nice_fraction = 2.0
        elif fraction < 3.5:
            nice_fraction = 2.5
        elif fraction < 7.5:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    else:
        if fraction <= 1.0:
            nice_fraction = 1.0
        elif fraction <= 2.0:
            nice_fraction = 2.0
        elif fraction <= 2.5:
            nice_fraction = 2.5
        elif fraction <= 5.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    return float(nice_fraction * (10 ** exponent))


def _format_tick_label(value):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


__all__ = [
    "load_data",
    "extract_data",
    "load_parameter",
    "list_parameter_keys",
    "calculate_weibull_parameters",
    "plot_weibull",
    "render_plot_to_png_bytes",
    "__version__",
    "_build_axis_ticks",
]
