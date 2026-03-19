import base64
from io import BytesIO
import json
import os
import re
import zipfile

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.dcc import send_bytes
import numpy as np
import plotly.graph_objects as go
from scipy.stats import weibull_min

from WAST import (
    _build_axis_ticks,
    _expand_axis_bounds,
    calculate_weibull_parameters,
    extract_data,
    load_data,
    load_parameter,
    list_parameter_keys,
    plot_weibull,
    render_plot_to_png_bytes,
)
from version import get_version


PARAMETER_KEY_LABELS = {
    "Auftrags-Nr.": {"de": "Auftrags-Nr.", "en": "Order No."},
    "Werkstoff": {"de": "Werkstoff", "en": "Material"},
}

DEFAULT_PARAM_KEYS = ["Auftrags-Nr.", "Werkstoff"]

TEXT = {
    "de": {
        "title": "Weibull-Analyse",
        "intro": "Excel-Datei per Drag-and-Drop hochladen, Parameter waehlen und Analyse starten. Die Ergebnisse koennen anschliessend als ZIP-Datei heruntergeladen werden.",
        "upload_prefix": "Datei hier ablegen oder ",
        "upload_link": "auswaehlen",
        "upload_empty": "Keine Datei hochgeladen.",
        "upload_loaded": "Datei geladen: {filename}",
        "upload_error": "Fehler beim Upload: {message}",
        "language_label": "Sprache",
        "language_de": "Deutsch",
        "language_en": "Englisch",
        "param_key_label": "Parameterschluessel",
        "confidence_label": "Konfidenzniveau",
        "comment_label": "Kommentar",
        "comment_placeholder": "Optional",
        "custom_value_label": "Optionaler Wert fuer Ausfallwahrscheinlichkeit",
        "analyze_button": "Analyse starten",
        "download_button": "Ergebnisse herunterladen (ZIP)",
        "download_help": "Beim Download oeffnet der Browser den Dialog zum Speichern.",
        "plot_help": "Punkte im Diagramm anklicken, um sie aus der Analyse aus- oder wieder einzuschliessen.",
        "reset_exclusions": "Ausgeschlossene Punkte zuruecksetzen",
        "run_first": "Bitte zuerst eine Analyse ausfuehren.",
        "download_ready": "Download bereit.",
        "upload_first": "Bitte zuerst eine Excel-Datei hochladen.",
        "error_prefix": "Fehler: {message}",
        "results_heading": "Ergebnisse",
        "summary_sample": "Stichprobe n = {n}, {label}: {value}",
        "summary_measured": "Messspalte: {value}",
        "summary_order": "Auftrags-Nr.: {value}",
        "summary_confidence": "Konfidenzniveau: {value}%",
        "summary_bootstrap": "AD-p-Wert ueber parametrischen Bootstrap (n={n})",
        "summary_excluded": "Ausgeschlossene Punkte: {n}",
        "na": "k. A.",
        "warning_heading": "Analysehinweise",
        "warning_small_sample": "Kleine Stichprobe: Konfidenzintervalle und Anpassungsguete koennen instabil sein.",
        "warning_p_value": "Der AD-p-Wert liegt unter 0,05. Das gefittete Weibull-Modell beschreibt die Daten moeglicherweise nicht gut.",
        "warning_ci_bounds": "Mindestens eine untere Konfidenzgrenze ist nicht positiv. Das Intervall sollte mit Vorsicht interpretiert werden.",
        "warning_extrapolation": "Der optionale Wert liegt ausserhalb des beobachteten Datenbereichs. Die Ausfallwahrscheinlichkeit ist eine Extrapolation.",
        "row_shape_mle": "Weibull-Modul m (MLE)",
        "row_shape_unbiased": "Weibull-Modul m (bias-korrigiert)",
        "row_ci_shape": "Konfidenzintervall m ({confidence}%)",
        "row_scale": "Kennwert",
        "row_ci_scale": "Konfidenzintervall Kennwert ({confidence}%)",
        "row_ad_stat": "AD-Statistik",
        "row_p_value": "AD-p-Wert",
        "row_p_method": "p-Wert-Methode",
        "row_ci_method": "KI-Methode",
        "row_bootstrap": "Bootstrap-Stichproben",
        "range_sep": " bis ",
        "export_code_version": "Code-Version",
        "export_parameter_key": "Parameterschluessel",
        "export_parameter_label": "Parameterbezeichnung",
        "export_parameter_value": "Parameterwert",
        "export_order_number": "Auftrags-Nr.",
        "export_measurement": "Messspalte",
        "export_unit": "Einheit",
        "export_sample_size": "Stichprobe n",
        "export_shape_mle": "Weibull-Modul (MLE)",
        "export_shape_unbiased": "Weibull-Modul (bias-korrigiert)",
        "export_scale": "Kennwert",
        "export_confidence": "Konfidenzniveau",
        "export_ci_shape": "KI m",
        "export_ci_scale": "KI Kennwert",
        "export_ad_stat": "AD-Statistik",
        "export_p_value": "p-Wert",
        "export_p_method": "p-Wert-Methode",
        "export_ci_method": "KI-Methode",
        "export_bootstrap": "Bootstrap-Stichproben",
        "export_custom": "Optionaler Wert",
        "export_failure": "Ausfallwahrscheinlichkeit",
        "export_excluded": "Ausgeschlossene Indizes",
        "p_value_method": "Parametrischer Bootstrap des AD-Tests unter dem gefitteten Weibull-Modell",
        "ci_method": "Wald-Konfidenzintervalle aus der inversen Hesse-Matrix",
    },
    "en": {
        "title": "Weibull Analysis",
        "intro": "Upload the Excel file via drag and drop, choose parameters, and run the analysis. You can then download the results as a ZIP file.",
        "upload_prefix": "Drop file here or ",
        "upload_link": "browse",
        "upload_empty": "No file uploaded.",
        "upload_loaded": "File loaded: {filename}",
        "upload_error": "Upload error: {message}",
        "language_label": "Language",
        "language_de": "German",
        "language_en": "English",
        "param_key_label": "Parameter key",
        "confidence_label": "Confidence level",
        "comment_label": "Comment",
        "comment_placeholder": "Optional",
        "custom_value_label": "Optional value for failure probability",
        "analyze_button": "Run analysis",
        "download_button": "Download results (ZIP)",
        "download_help": "When downloading, the browser opens the save dialog.",
        "plot_help": "Click data points in the plot to exclude or include them in the analysis.",
        "reset_exclusions": "Reset excluded points",
        "run_first": "Please run an analysis first.",
        "download_ready": "Download ready.",
        "upload_first": "Please upload an Excel file first.",
        "error_prefix": "Error: {message}",
        "results_heading": "Results",
        "summary_sample": "Sample n = {n}, {label}: {value}",
        "summary_measured": "Measured column: {value}",
        "summary_order": "Order No.: {value}",
        "summary_confidence": "Confidence level: {value}%",
        "summary_bootstrap": "AD p-value via parametric bootstrap (n={n})",
        "summary_excluded": "Excluded points: {n}",
        "na": "n/a",
        "warning_heading": "Analysis warnings",
        "warning_small_sample": "Small sample size: confidence intervals and goodness-of-fit assessment may be unstable.",
        "warning_p_value": "The AD p-value is below 0.05. The fitted Weibull model may not describe the data well.",
        "warning_ci_bounds": "At least one lower confidence bound is non-positive. Interpret the interval estimate with caution.",
        "warning_extrapolation": "The optional value lies outside the observed data range. The failure probability is an extrapolation.",
        "row_shape_mle": "Weibull modulus m (MLE)",
        "row_shape_unbiased": "Weibull modulus m (bias-corrected)",
        "row_ci_shape": "Confidence interval m ({confidence}%)",
        "row_scale": "Characteristic value",
        "row_ci_scale": "Confidence interval characteristic value ({confidence}%)",
        "row_ad_stat": "AD statistic",
        "row_p_value": "AD p-value",
        "row_p_method": "p-value method",
        "row_ci_method": "CI method",
        "row_bootstrap": "Bootstrap samples",
        "range_sep": " to ",
        "export_code_version": "Code version",
        "export_parameter_key": "Parameter key",
        "export_parameter_label": "Parameter label",
        "export_parameter_value": "Parameter value",
        "export_order_number": "Order number",
        "export_measurement": "Measured column",
        "export_unit": "Unit",
        "export_sample_size": "Sample size n",
        "export_shape_mle": "Weibull modulus (MLE)",
        "export_shape_unbiased": "Weibull modulus (bias-corrected)",
        "export_scale": "Characteristic value",
        "export_confidence": "Confidence level",
        "export_ci_shape": "CI m",
        "export_ci_scale": "CI characteristic value",
        "export_ad_stat": "AD statistic",
        "export_p_value": "P-value",
        "export_p_method": "P-value method",
        "export_ci_method": "CI method",
        "export_bootstrap": "Bootstrap samples",
        "export_custom": "Optional value",
        "export_failure": "Failure probability",
        "export_excluded": "Excluded indices",
        "p_value_method": "Parametric bootstrap of the AD test under the fitted Weibull model",
        "ci_method": "Wald confidence intervals from the inverse Hessian",
    },
}


def _decode_upload(contents: str) -> bytes:
    if not contents:
        raise ValueError("Keine Datei hochgeladen.")
    try:
        _, content_string = contents.split(",")
        return base64.b64decode(content_string)
    except Exception as exc:
        raise ValueError(f"Upload konnte nicht gelesen werden: {exc}")


def _get_language(language):
    return language if language in TEXT else "de"


def _t(language, key, **kwargs):
    template = TEXT[_get_language(language)][key]
    return template.format(**kwargs) if kwargs else template


def _parameter_key_label(parameter_key, language):
    return PARAMETER_KEY_LABELS.get(parameter_key, {}).get(_get_language(language), parameter_key)


def _default_param_options(language):
    return _format_parameter_options(DEFAULT_PARAM_KEYS, language)


def _parse_custom_value(value):
    if value in (None, ""):
        return None

    parsed = float(value)
    if parsed <= 0:
        raise ValueError("Der optionale Wert muss groesser als 0 sein.")
    return parsed


def _slugify_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "weibull_result"


def _format_parameter_options(keys, language):
    options = []
    for key in keys:
        options.append({"label": _parameter_key_label(key, language), "value": key})
    return options


def _choose_parameter_value(keys, current_value=None):
    if current_value in keys:
        return current_value
    if "Werkstoff" in keys:
        return "Werkstoff"
    if "Auftrags-Nr." in keys:
        return "Auftrags-Nr."
    return keys[0] if keys else None


def _build_upload_status(upload_meta, language):
    state = (upload_meta or {}).get("state", "empty")
    if state == "loaded":
        filename = (upload_meta or {}).get("filename") or ("Untitled" if _get_language(language) == "en" else "Ohne Titel")
        return _t(language, "upload_loaded", filename=filename)
    if state == "error":
        return _t(language, "upload_error", message=(upload_meta or {}).get("message", ""))
    return _t(language, "upload_empty")


def _measurement_axis_title(measurement_series, unit, language):
    labels = {
        "sc": {"de": "Biegefestigkeit", "en": "Flexural strength"},
        "Fmax": {"de": "Bruchlast", "en": "Fracture load"},
    }
    base_label = labels.get(measurement_series, {}).get(_get_language(language), measurement_series)
    if unit and unit != "X":
        return f"{base_label} ({unit})"
    return base_label


def _build_plot_label(summary, language):
    parameter_label = _parameter_key_label(summary.get("parameter_key"), language)
    parameter_value = summary.get("parameter_value") or ""
    order_number = summary.get("order_number")
    label = parameter_label
    if parameter_value:
        label = f"{parameter_label}: {parameter_value}"
    if order_number and summary.get("parameter_key") != "Auftrags-Nr.":
        suffix = f" ({order_number})"
        label = f"{label}{suffix}" if label else order_number
    return label or parameter_label


def _render_plot_base64(analysis_data, language):
    summary = analysis_data["summary"]
    data = np.asarray(analysis_data["raw_data"], dtype=float)
    ci_shape = tuple(summary["ci_shape"])
    ci_scale = tuple(summary["ci_scale"])
    alpha = summary["confidence_level"] / 100
    p_lin = np.linspace(0.01, 0.99, 500)
    lower_ci = weibull_min.ppf(p_lin, c=ci_shape[0], scale=ci_scale[0])
    upper_ci = weibull_min.ppf(p_lin, c=ci_shape[1], scale=ci_scale[1])
    fig = plot_weibull(
        data,
        _build_plot_label(summary, language),
        summary["unit"],
        summary["scale_mle"],
        summary["unbiased_shape"],
        lower_ci,
        upper_ci,
        summary.get("custom_value"),
        summary.get("failure_probability"),
        summary.get("comment", ""),
        summary.get("data_symbol", ""),
        _measurement_axis_title(summary["measurement_series"], summary["unit"], language),
        summary["ad_statistic"],
        summary["p_value"],
        alpha,
        ci_shape,
        ci_scale,
        language=language,
    )
    return base64.b64encode(render_plot_to_png_bytes(fig)).decode()


def _weibull_y(probabilities):
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    return np.log(-np.log(1 - probs))


def _analysis_context(
    parameter_key,
    parameter_value,
    order_number,
    measurement_series,
    unit,
    data_symbol,
    alpha,
    custom_value,
    comment,
):
    return {
        "parameter_key": parameter_key,
        "parameter_value": parameter_value,
        "order_number": order_number,
        "measurement_series": measurement_series,
        "unit": unit,
        "data_symbol": data_symbol,
        "alpha": float(alpha),
        "custom_value": None if custom_value is None else float(custom_value),
        "comment": comment or "",
    }


def _bootstrap_samples_for_payload(excluded_indices):
    return 180 if excluded_indices else 320


def _build_analysis_payload(source_data, context, excluded_indices, language):
    source_array = np.asarray(source_data, dtype=float)
    excluded_sorted = sorted(
        {
            int(idx)
            for idx in (excluded_indices or [])
            if 0 <= int(idx) < len(source_array)
        }
    )
    included_mask = np.ones(len(source_array), dtype=bool)
    if excluded_sorted:
        included_mask[excluded_sorted] = False
    active_data = source_array[included_mask]
    if active_data.size < 2:
        raise ValueError("Need at least 2 data points for Weibull analysis")

    bootstrap_samples_requested = _bootstrap_samples_for_payload(excluded_sorted)
    shape, scale, unbiased_shape, ci_shape, ci_scale, d_stat, p_val, bootstrap_samples = calculate_weibull_parameters(
        active_data,
        context["alpha"],
        n_boot=bootstrap_samples_requested,
    )
    custom_value = context.get("custom_value")
    failure_prob = weibull_min.cdf(custom_value, c=unbiased_shape, scale=scale) if custom_value is not None else None

    summary = _build_summary(
        context["parameter_key"],
        context["parameter_value"],
        context["order_number"],
        context["measurement_series"],
        context["unit"],
        active_data,
        shape,
        scale,
        unbiased_shape,
        ci_shape,
        ci_scale,
        d_stat,
        p_val,
        bootstrap_samples,
        context["alpha"],
        custom_value,
        failure_prob,
        context.get("comment", ""),
        language,
    )
    summary["data_symbol"] = context.get("data_symbol", "")
    summary["excluded_count"] = len(excluded_sorted)
    summary["source_n"] = int(len(source_array))

    return {
        "summary": summary,
        "raw_data": active_data.tolist(),
        "source_data": source_array.tolist(),
        "excluded_indices": excluded_sorted,
        "context": context,
    }


def _toggle_excluded_index(excluded_indices, clicked_index, source_length):
    excluded = {int(idx) for idx in (excluded_indices or []) if 0 <= int(idx) < source_length}
    clicked = int(clicked_index)
    if clicked in excluded:
        excluded.remove(clicked)
    else:
        if source_length - len(excluded) <= 2:
            return sorted(excluded)
        excluded.add(clicked)
    return sorted(excluded)


def _extract_clicked_index(click_data, analysis_data=None):
    if not click_data or not click_data.get("points"):
        return None
    point = click_data["points"][0]
    customdata = point.get("customdata")
    if customdata is None:
        click_x = point.get("x")
        source_data = None if analysis_data is None else analysis_data.get("source_data", analysis_data.get("raw_data"))
        if click_x is None or source_data is None:
            return None
        source_array = np.asarray(source_data, dtype=float)
        source_array = source_array[np.isfinite(source_array) & (source_array > 0)]
        if source_array.size == 0:
            return None
        source_logs = np.log(np.asarray(source_data, dtype=float))
        if not np.isfinite(source_logs).any():
            return None
        valid_indices = np.where(np.isfinite(source_logs))[0]
        nearest = valid_indices[np.argmin(np.abs(source_logs[valid_indices] - float(click_x)))]
        return int(nearest)
    if isinstance(customdata, (list, tuple)):
        customdata = customdata[0] if customdata else None
    return None if customdata is None else int(customdata)


def _format_plot_tick(value):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _build_interactive_figure(analysis_data, language):
    summary = analysis_data["summary"]
    source_data = np.asarray(analysis_data.get("source_data") or analysis_data["raw_data"], dtype=float)
    excluded_indices = np.asarray(sorted(analysis_data.get("excluded_indices", [])), dtype=int)
    excluded_set = set(excluded_indices.tolist())
    active_indices = np.array([idx for idx in range(len(source_data)) if idx not in excluded_set], dtype=int)
    active_data = source_data[active_indices]

    alpha = summary["confidence_level"] / 100
    p = np.linspace(0.01, 0.99, 500)
    y_curve = _weibull_y(p)
    fit_quantiles = weibull_min.ppf(p, c=summary["unbiased_shape"], scale=summary["scale_mle"])
    ci_shape = tuple(summary["ci_shape"])
    ci_scale = tuple(summary["ci_scale"])
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

    order_active = np.argsort(active_data)
    active_sorted = active_data[order_active]
    active_sorted_indices = active_indices[order_active]
    empirical_probs = (np.arange(1, len(active_sorted) + 1) - 0.5) / len(active_sorted)

    order_source = np.argsort(source_data)
    source_sorted = source_data[order_source]
    source_probs = (np.arange(1, len(source_sorted) + 1) - 0.5) / len(source_sorted)
    excluded_mask_sorted = np.isin(order_source, excluded_indices)

    palette = {
        "data": "#0f766e",
        "fit": "#d97706",
        "band_fill": "rgba(148, 163, 184, 0.24)",
        "accent": "#0284c7",
        "excluded": "#64748b",
    }
    text = {
        "de": {
            "fit": "Fit: m = {m:.1f}, Kennwert = {scale:.0f} {unit}",
            "band": "{confidence} %-Konfidenzband",
            "custom": "P(Ausfall) bei {value:.0f} {unit} = {prob:.2f} %",
            "title": "Weibull-Diagramm mit {confidence} %-Konfidenzband",
            "ylabel": "Ausfallwahrscheinlichkeit (%)",
            "included": "Eingeschlossene Punkte | n = {n}",
            "excluded": "Ausgeschlossene Punkte | n = {n}",
        },
        "en": {
            "fit": "Fit: m = {m:.1f}, characteristic value = {scale:.0f} {unit}",
            "band": "{confidence}% confidence band",
            "custom": "P(failure) at {value:.0f} {unit} = {prob:.2f} %",
            "title": "Weibull plot with {confidence}% confidence band",
            "ylabel": "Failure probability (%)",
            "included": "Included points | n = {n}",
            "excluded": "Excluded points | n = {n}",
        },
    }[_get_language(language)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([np.log(lower_ci_curve), np.log(upper_ci_curve[::-1])]),
            y=np.concatenate([y_curve, y_curve[::-1]]),
            fill="toself",
            fillcolor=palette["band_fill"],
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name=text["band"].format(confidence=int(alpha * 100)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.log(fit_quantiles),
            y=y_curve,
            mode="lines",
            line=dict(color=palette["fit"], width=2),
            name=text["fit"].format(m=summary["unbiased_shape"], scale=summary["scale_mle"], unit=summary["unit"]),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.log(active_sorted),
            y=_weibull_y(empirical_probs),
            mode="markers",
            marker=dict(color=palette["data"], symbol="circle", size=11, line=dict(width=1.5, color="#ffffff")),
            name=text["included"].format(n=len(active_sorted)),
            customdata=np.array(active_sorted_indices)[:, None],
            text=[f"{value:.3f}" for value in active_sorted],
            hovertemplate="x=%{text}<extra></extra>",
        )
    )
    if excluded_mask_sorted.any():
        fig.add_trace(
            go.Scatter(
                x=np.log(source_sorted[excluded_mask_sorted]),
                y=_weibull_y(source_probs[excluded_mask_sorted]),
                mode="markers",
                marker=dict(
                    color=palette["excluded"],
                    symbol="circle-open",
                    size=11,
                    line=dict(width=2, color=palette["excluded"]),
                ),
                name=text["excluded"].format(n=int(excluded_mask_sorted.sum())),
                customdata=np.array(order_source[excluded_mask_sorted])[:, None],
                text=[f"{value:.3f}" for value in source_sorted[excluded_mask_sorted]],
                hovertemplate="x=%{text}<extra></extra>",
            )
        )

    if summary.get("custom_value") is not None and summary.get("failure_probability") is not None:
        x_custom = np.log(summary["custom_value"])
        y_custom = float(_weibull_y([summary["failure_probability"]])[0])
        fig.add_vline(x=x_custom, line_color=palette["accent"], line_dash="dash", line_width=1)
        fig.add_hline(y=y_custom, line_color=palette["accent"], line_dash="dash", line_width=1)

    axis_candidates = [
        np.asarray(source_data, dtype=float),
        np.asarray(fit_quantiles, dtype=float),
        np.asarray(lower_ci_curve, dtype=float),
        np.asarray(upper_ci_curve, dtype=float),
    ]
    if summary.get("custom_value") is not None:
        axis_candidates.append(np.asarray([summary["custom_value"]], dtype=float))
    finite_axis = np.concatenate(axis_candidates)
    finite_axis = finite_axis[np.isfinite(finite_axis) & (finite_axis > 0)]
    if finite_axis.size:
        display_min, display_max = _expand_axis_bounds(float(np.min(finite_axis)), float(np.max(finite_axis)))
    else:
        display_min, display_max = float(np.min(source_data)), float(np.max(source_data))
    ticks = _build_axis_ticks(display_min, display_max)
    probs_std = np.array([0.01, 0.05, 0.10, 0.20, 0.40, 0.6325, 0.80, 0.95, 0.99])

    fig.update_layout(
        title=dict(
            text=text["title"].format(confidence=int(alpha * 100)),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            pad=dict(t=8, b=0),
        ),
        template="plotly_white",
        height=560,
        margin=dict(l=70, r=30, t=120, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        clickmode="event",
        hovermode="closest",
    )
    fig.update_xaxes(
        title=_measurement_axis_title(summary["measurement_series"], summary["unit"], language),
        tickmode="array",
        tickvals=np.log(ticks) if ticks.size else None,
        ticktext=[_format_plot_tick(tick) for tick in ticks] if ticks.size else None,
        range=[np.log(display_min), np.log(display_max)],
        gridcolor="#cbd5e1",
        zeroline=False,
    )
    fig.update_yaxes(
        title=text["ylabel"],
        tickmode="array",
        tickvals=_weibull_y(probs_std),
        ticktext=[f"{p*100:.1f}" for p in probs_std],
        gridcolor="#cbd5e1",
        zeroline=False,
    )
    if summary.get("comment"):
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.98,
            showarrow=False,
            align="left",
            bgcolor="#fffbeb",
            bordercolor="#fbbf24",
            borderwidth=1,
            text=summary["comment"],
        )
    return fig


def _build_placeholder_figure(language, message=""):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        height=560,
        margin=dict(l=70, r=30, t=40, b=50),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message or _t(language, "run_first"),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#4f5d75"),
            )
        ],
    )
    return fig


def _build_download_bundle(analysis_data, language):
    summary = analysis_data.get("summary", {})
    raw_data = analysis_data.get("raw_data", [])
    label_raw = summary.get("parameter_value") or summary.get("order_number") or "weibull_result"
    label_slug = _slugify_filename(label_raw)
    language = _get_language(language)
    plot_png = analysis_data.get("plot_png") or _render_plot_base64(analysis_data, language)

    lines = [
        f"{_t(language, 'export_code_version')}: {summary.get('code_version', get_version())}",
        f"{_t(language, 'export_parameter_key')}: {summary.get('parameter_key', '')}",
        f"{_t(language, 'export_parameter_label')}: {_parameter_key_label(summary.get('parameter_key'), language)}",
        f"{_t(language, 'export_parameter_value')}: {summary.get('parameter_value', '')}",
        f"{_t(language, 'export_order_number')}: {summary.get('order_number', '')}",
        f"{_t(language, 'export_measurement')}: {summary.get('measurement_series', '')}",
        f"{_t(language, 'export_unit')}: {summary.get('unit', '')}",
        f"{_t(language, 'export_sample_size')}: {summary.get('n', '')}",
        f"{_t(language, 'export_shape_mle')}: {summary.get('shape_mle', '')}",
        f"{_t(language, 'export_shape_unbiased')}: {summary.get('unbiased_shape', '')}",
        f"{_t(language, 'export_scale')}: {summary.get('scale_mle', '')} {summary.get('unit', '')}",
        f"{_t(language, 'export_confidence')}: {summary.get('confidence_level', '')}%",
        f"{_t(language, 'export_ci_shape')}: {summary.get('ci_shape', '')}",
        f"{_t(language, 'export_ci_scale')}: {summary.get('ci_scale', '')}",
        f"{_t(language, 'export_ad_stat')}: {summary.get('ad_statistic', '')}",
        f"{_t(language, 'export_p_value')}: {summary.get('p_value', '')}",
        f"{_t(language, 'export_p_method')}: {summary.get('p_value_method') or _t(language, 'p_value_method')}",
        f"{_t(language, 'export_ci_method')}: {summary.get('ci_method') or _t(language, 'ci_method')}",
        f"{_t(language, 'export_bootstrap')}: {summary.get('bootstrap_samples', '')}",
        f"{_t(language, 'export_custom')}: {summary.get('custom_value', '')}",
        f"{_t(language, 'export_failure')}: {summary.get('failure_probability', '')}",
        f"{_t(language, 'export_excluded')}: {analysis_data.get('excluded_indices', [])}",
    ]
    results_txt = "\n".join(lines)

    csv_lines = ["index,value"]
    csv_lines.extend(f"{idx},{value}" for idx, value in enumerate(raw_data, start=1))
    raw_data_csv = "\n".join(csv_lines)

    json_payload = {
        "summary": summary,
        "raw_data": raw_data,
        "source_data": analysis_data.get("source_data", raw_data),
        "excluded_indices": analysis_data.get("excluded_indices", []),
    }

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(f"{label_slug}_plot.png", base64.b64decode(plot_png))
        zf.writestr(f"{label_slug}_results.txt", results_txt)
        zf.writestr(
            f"{label_slug}_results.json",
            json.dumps(json_payload, indent=2, ensure_ascii=False),
        )
        zf.writestr(f"{label_slug}_raw_data.csv", raw_data_csv)
    buffer.seek(0)
    return buffer.read()


def _build_summary(
    parameter_key,
    parameter_value,
    order_number,
    measurement_series,
    unit,
    data,
    shape,
    scale,
    unbiased_shape,
    ci_shape,
    ci_scale,
    d_stat,
    p_val,
    bootstrap_samples,
    alpha,
    custom_value,
    failure_prob,
    comment,
    language,
):
    return {
        "parameter_key": parameter_key,
        "parameter_value": parameter_value,
        "order_number": order_number,
        "measurement_series": measurement_series,
        "unit": unit,
        "n": int(len(data)),
        "shape_mle": float(shape),
        "scale_mle": float(scale),
        "unbiased_shape": float(unbiased_shape),
        "ci_shape": [float(ci_shape[0]), float(ci_shape[1])],
        "ci_scale": [float(ci_scale[0]), float(ci_scale[1])],
        "ad_statistic": float(d_stat),
        "p_value": float(p_val),
        "bootstrap_samples": int(bootstrap_samples),
        "p_value_method": _t(language, "p_value_method"),
        "ci_method": _t(language, "ci_method"),
        "confidence_level": int(alpha * 100),
        "custom_value": None if custom_value is None else float(custom_value),
        "failure_probability": None if failure_prob is None else float(failure_prob),
        "comment": comment,
        "code_version": get_version(),
    }


def _build_results_table(summary, language):
    rows = [
        (_t(language, "row_shape_mle"), f"{summary['shape_mle']:.3f}"),
        (_t(language, "row_shape_unbiased"), f"{summary['unbiased_shape']:.3f}"),
        (
            _t(language, "row_ci_shape", confidence=summary["confidence_level"]),
            f"{summary['ci_shape'][0]:.3f}{_t(language, 'range_sep')}{summary['ci_shape'][1]:.3f}",
        ),
        (_t(language, "row_scale"), f"{summary['scale_mle']:.3f} {summary['unit']}"),
        (
            _t(language, "row_ci_scale", confidence=summary["confidence_level"]),
            f"{summary['ci_scale'][0]:.3f}{_t(language, 'range_sep')}{summary['ci_scale'][1]:.3f} {summary['unit']}",
        ),
        (_t(language, "row_ad_stat"), f"{summary['ad_statistic']:.4f}"),
        (_t(language, "row_p_value"), f"{summary['p_value']:.4f}"),
        (_t(language, "row_p_method"), _t(language, "p_value_method")),
        (_t(language, "row_ci_method"), _t(language, "ci_method")),
        (_t(language, "row_bootstrap"), str(summary["bootstrap_samples"])),
    ]

    return html.Table(
        className="results-table",
        children=[
            html.Tbody(
                [
                    html.Tr([html.Th(label), html.Td(value)])
                    for label, value in rows
                ]
            )
        ],
    )


def _build_analysis_warnings(summary, data, language):
    warnings = []

    if summary["n"] < 10:
        warnings.append(_t(language, "warning_small_sample"))
    if summary["p_value"] < 0.05:
        warnings.append(_t(language, "warning_p_value"))
    if summary["ci_shape"][0] <= 0 or summary["ci_scale"][0] <= 0:
        warnings.append(_t(language, "warning_ci_bounds"))
    if summary["custom_value"] is not None:
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if summary["custom_value"] < data_min or summary["custom_value"] > data_max:
            warnings.append(_t(language, "warning_extrapolation"))

    if not warnings:
        return None

    return html.Div(
        className="analysis-warnings",
        children=[
            html.H4(_t(language, "warning_heading")),
            html.Ul([html.Li(message) for message in warnings]),
        ],
    )


def _build_analysis_summary(summary, raw_data, language):
    parameter_label = _parameter_key_label(summary["parameter_key"], language)
    summary_items = [
        html.H3(_t(language, "results_heading")),
        html.Ul(
            [
                html.Li(
                    _t(
                        language,
                        "summary_sample",
                        n=summary["n"],
                        label=parameter_label,
                        value=summary["parameter_value"] or _t(language, "na"),
                    )
                ),
                html.Li(_t(language, "summary_measured", value=summary["measurement_series"])),
                html.Li(_t(language, "summary_order", value=summary["order_number"] or _t(language, "na"))),
                html.Li(_t(language, "summary_confidence", value=summary["confidence_level"])),
                html.Li(_t(language, "summary_bootstrap", n=summary["bootstrap_samples"])),
                html.Li(_t(language, "summary_excluded", n=summary.get("excluded_count", 0))),
            ]
        ),
        _build_analysis_warnings(summary, np.asarray(raw_data, dtype=float), language),
        _build_results_table(summary, language),
    ]
    return summary_items


def create_app():
    app = Dash(__name__, title="Weibull-Tool", suppress_callback_exceptions=True)
    server = app.server

    def serve_layout():
        return html.Div(
            className="app",
            children=[
                html.H1(id="app-title"),
                html.P(id="app-intro"),
                html.Div(
                    className="upload",
                    children=[
                        dcc.Upload(
                            id="upload-data",
                            className="upload-dropzone",
                            children=html.Div(id="upload-prompt"),
                            multiple=False,
                        ),
                        html.Div(id="upload-status", className="status"),
                    ],
                ),
                html.Div(
                    className="controls",
                    children=[
                        html.Div(
                            className="control",
                            children=[
                                html.Label(id="label-language"),
                                dcc.RadioItems(
                                    id="language",
                                    options=[
                                        {"label": "Deutsch", "value": "de"},
                                        {"label": "English", "value": "en"},
                                    ],
                                    value="de",
                                    className="radio-compact language-switch",
                                    labelStyle={"display": "inline-flex", "alignItems": "center"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="control",
                            children=[
                                html.Label(id="label-param-key"),
                                dcc.Dropdown(
                                    id="param-key",
                                    options=_default_param_options("de"),
                                    value=_choose_parameter_value(DEFAULT_PARAM_KEYS),
                                    clearable=False,
                                    searchable=False,
                                ),
                            ],
                        ),
                        html.Div(
                            className="control",
                            children=[
                                html.Label(id="label-confidence"),
                                dcc.RadioItems(
                                    id="confidence",
                                    options=[{"label": f"{p}%", "value": p} for p in (90, 95, 99)],
                                    value=95,
                                    className="radio-compact",
                                    labelStyle={"display": "inline-flex", "flexDirection": "column", "alignItems": "center"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="control",
                            children=[
                                html.Label(id="label-custom-value"),
                                dcc.Input(id="custom-value", type="number", min=0.000001, step="any"),
                            ],
                        ),
                        html.Div(
                            className="control",
                            children=[
                                html.Label(id="label-comment"),
                                dcc.Input(id="user-comment", placeholder="Optional", type="text", style={"width": "100%"}),
                            ],
                        ),
                        html.Button(id="analyze", n_clicks=0, className="primary"),
                    ],
                ),
                dcc.Loading(
                    type="default",
                    color="#38bdf8",
                    children=[
                        html.Div(id="analysis-summary", className="summary"),
                        html.Div(
                            id="plot-container",
                            className="plot",
                            children=[
                                html.P(id="plot-help", className="status"),
                                dcc.Graph(
                                    id="weibull-graph",
                                    figure=_build_placeholder_figure("de", ""),
                                    config={"displayModeBar": True, "responsive": True},
                                    style={"width": "100%"},
                                ),
                                html.Button(
                                    id="reset-exclusions",
                                    n_clicks=0,
                                    disabled=True,
                                    className="secondary",
                                    style={"display": "none"},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="download",
                    children=[
                        html.Button(id="download-btn", n_clicks=0, disabled=True),
                        html.P(id="download-help", className="status"),
                        html.Div(id="download-status", className="status"),
                        dcc.Download(id="download-bundle"),
                    ],
                ),
                dcc.Store(id="file-bytes"),
                dcc.Store(id="upload-meta", data={"state": "empty"}),
                dcc.Store(id="parameter-keys", data=DEFAULT_PARAM_KEYS),
                dcc.Store(id="analysis-data"),
                dcc.Store(id="analysis-meta", data={"state": "idle"}),
                html.Div(f"Version {get_version()}", className="version-footer"),
            ],
        )

    app.layout = serve_layout

    @app.callback(
        Output("upload-meta", "data"),
        Output("file-bytes", "data"),
        Output("parameter-keys", "data"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
    )
    def handle_upload(contents, filename):
        if contents is None:
            return {"state": "empty"}, None, DEFAULT_PARAM_KEYS
        try:
            decoded = _decode_upload(contents)
            keys = list_parameter_keys(BytesIO(decoded))
        except ValueError as exc:
            return {"state": "error", "message": str(exc)}, None, DEFAULT_PARAM_KEYS
        name_display = filename or "Untitled"
        return {"state": "loaded", "filename": name_display}, base64.b64encode(decoded).decode(), (keys or DEFAULT_PARAM_KEYS)

    @app.callback(
        Output("app-title", "children"),
        Output("app-intro", "children"),
        Output("upload-prompt", "children"),
        Output("label-language", "children"),
        Output("label-param-key", "children"),
        Output("param-key", "options"),
        Output("param-key", "value"),
        Output("label-confidence", "children"),
        Output("label-comment", "children"),
        Output("user-comment", "placeholder"),
        Output("label-custom-value", "children"),
        Output("analyze", "children"),
        Output("download-btn", "children"),
        Output("download-help", "children"),
        Output("upload-status", "children"),
        Input("language", "value"),
        Input("parameter-keys", "data"),
        Input("upload-meta", "data"),
        State("param-key", "value"),
    )
    def update_interface(language, parameter_keys, upload_meta, current_param_key):
        language = _get_language(language)
        keys = parameter_keys or DEFAULT_PARAM_KEYS
        options = _format_parameter_options(keys, language)
        value = _choose_parameter_value(keys, current_param_key)
        return (
            _t(language, "title"),
            _t(language, "intro"),
            html.Div([_t(language, "upload_prefix"), html.A(_t(language, "upload_link"))]),
            _t(language, "language_label"),
            _t(language, "param_key_label"),
            options,
            value,
            _t(language, "confidence_label"),
            _t(language, "comment_label"),
            _t(language, "comment_placeholder"),
            _t(language, "custom_value_label"),
            _t(language, "analyze_button"),
            _t(language, "download_button"),
            _t(language, "download_help"),
            _build_upload_status(upload_meta, language),
        )

    @app.callback(
        Output("analysis-data", "data"),
        Output("analysis-meta", "data"),
        Output("download-btn", "disabled"),
        Input("analyze", "n_clicks"),
        Input("weibull-graph", "clickData"),
        Input("reset-exclusions", "n_clicks"),
        State("analysis-data", "data"),
        State("file-bytes", "data"),
        State("param-key", "value"),
        State("confidence", "value"),
        State("user-comment", "value"),
        State("custom-value", "value"),
        State("language", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(
        n_clicks,
        click_data,
        reset_clicks,
        current_analysis,
        file_bytes,
        param_key,
        confidence,
        user_comment,
        custom_value,
        language,
    ):
        language = _get_language(language)
        triggered = ctx.triggered_id

        if triggered == "analyze":
            if not file_bytes:
                return None, {"state": "error", "message_key": "upload_first"}, True

            file_content = base64.b64decode(file_bytes)
            alpha = (confidence or 95) / 100
            comment = user_comment or ""

            try:
                custom_val = _parse_custom_value(custom_value)
                parameter_value = load_parameter(BytesIO(file_content), param_key)
                try:
                    order_number = load_parameter(BytesIO(file_content), "Auftrags-Nr.")
                except ValueError:
                    order_number = None
                _, id_unit, df_data = load_data(BytesIO(file_content), ["sc", "Fmax"])
                data, series_key, sym, _ = extract_data(df_data)
                context = _analysis_context(
                    param_key,
                    parameter_value,
                    order_number,
                    series_key,
                    id_unit,
                    sym,
                    alpha,
                    custom_val,
                    comment,
                )
                analysis_payload = _build_analysis_payload(data, context, [], language)
            except Exception as exc:
                return None, {"state": "error", "message": str(exc)}, True

            return analysis_payload, {"state": "ready"}, False

        if not current_analysis:
            return no_update, no_update, no_update

        try:
            source_data = current_analysis.get("source_data", current_analysis.get("raw_data", []))
            excluded_indices = current_analysis.get("excluded_indices", [])
            if triggered == "weibull-graph":
                clicked_index = _extract_clicked_index(click_data, current_analysis)
                if clicked_index is None:
                    return no_update, no_update, no_update
                excluded_indices = _toggle_excluded_index(excluded_indices, clicked_index, len(source_data))
            elif triggered == "reset-exclusions":
                excluded_indices = []
            else:
                return no_update, no_update, no_update

            analysis_payload = _build_analysis_payload(
                source_data,
                current_analysis.get("context", {}),
                excluded_indices,
                language,
            )
        except Exception as exc:
            return current_analysis, {"state": "error", "message": str(exc)}, False

        return analysis_payload, {"state": "ready"}, False

    @app.callback(
        Output("analysis-summary", "children"),
        Output("plot-help", "children"),
        Output("weibull-graph", "figure"),
        Output("weibull-graph", "style"),
        Output("reset-exclusions", "children"),
        Output("reset-exclusions", "disabled"),
        Output("reset-exclusions", "style"),
        Input("analysis-data", "data"),
        Input("analysis-meta", "data"),
        Input("language", "value"),
    )
    def render_analysis(analysis_data, analysis_meta, language):
        language = _get_language(language)
        meta = analysis_meta or {"state": "idle"}
        if meta.get("state") == "error":
            message = _t(language, meta["message_key"]) if meta.get("message_key") else meta.get("message", "")
            return (
                _t(language, "error_prefix", message=message),
                "",
                _build_placeholder_figure(language, _t(language, "error_prefix", message=message)),
                {"width": "100%"},
                _t(language, "reset_exclusions"),
                True,
                {"display": "none"},
            )
        if not analysis_data:
            return None, "", _build_placeholder_figure(language, ""), {"width": "100%"}, _t(language, "reset_exclusions"), True, {"display": "none"}

        summary = analysis_data["summary"]
        return (
            _build_analysis_summary(summary, analysis_data["raw_data"], language),
            _t(language, "plot_help"),
            _build_interactive_figure(analysis_data, language),
            {"width": "100%"},
            _t(language, "reset_exclusions"),
            summary.get("excluded_count", 0) == 0,
            {"display": "inline-block"},
        )

    @app.callback(
        Output("download-bundle", "data"),
        Output("download-status", "children"),
        Input("download-btn", "n_clicks"),
        State("analysis-data", "data"),
        State("language", "value"),
        prevent_initial_call=True,
    )
    def trigger_download(n_clicks, analysis_data, language):
        language = _get_language(language)
        if not analysis_data:
            return no_update, _t(language, "run_first")

        def build_zip_bytes(buffer: BytesIO):
            """Dash send_bytes writer: fills provided buffer with the zip payload."""
            buffer.write(_build_download_bundle(analysis_data, language))

        return send_bytes(build_zip_bytes, "weibull_output.zip"), _t(language, "download_ready")

    return app, server


app, server = create_app()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weibull-Dash-App")
    parser.add_argument("--version", action="store_true", help="Version anzeigen und beenden")
    args = parser.parse_args()

    if args.version:
        print(f"Weibull-Tool Version: {get_version()}")
    else:
        debug = os.getenv("DASH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        port = int(os.getenv("PORT", "8053"))
        app.run_server(debug=debug, host="0.0.0.0", port=port)
