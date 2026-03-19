import base64
from io import BytesIO
import json
import os
import re
import zipfile

from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.dcc import send_bytes
import numpy as np
from scipy.stats import weibull_min

from WAST import (
    calculate_weibull_parameters,
    extract_data,
    load_data,
    load_parameter,
    list_parameter_keys,
    plot_weibull,
    render_plot_to_png_bytes,
    __version__,
)


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


def _build_download_bundle(analysis_data, language):
    summary = analysis_data.get("summary", {})
    raw_data = analysis_data.get("raw_data", [])
    label_raw = summary.get("parameter_value") or summary.get("order_number") or "weibull_result"
    label_slug = _slugify_filename(label_raw)
    language = _get_language(language)
    plot_png = analysis_data.get("plot_png") or _render_plot_base64(analysis_data, language)

    lines = [
        f"{_t(language, 'export_code_version')}: {summary.get('code_version', __version__)}",
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
    ]
    results_txt = "\n".join(lines)

    csv_lines = ["index,value"]
    csv_lines.extend(f"{idx},{value}" for idx, value in enumerate(raw_data, start=1))
    raw_data_csv = "\n".join(csv_lines)

    json_payload = {
        "summary": summary,
        "raw_data": raw_data,
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
        "code_version": __version__,
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
            ]
        ),
        _build_analysis_warnings(summary, np.asarray(raw_data, dtype=float), language),
        _build_results_table(summary, language),
    ]
    return summary_items


def create_app():
    app = Dash(__name__, title="Weibull-Tool", suppress_callback_exceptions=True)
    server = app.server

    app.layout = html.Div(
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
                    html.Div(id="plot-container", className="plot"),
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
            html.Div(f"Version {__version__}", className="version-footer"),
        ],
    )

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
        State("file-bytes", "data"),
        State("param-key", "value"),
        State("confidence", "value"),
        State("user-comment", "value"),
        State("custom-value", "value"),
        State("language", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, file_bytes, param_key, confidence, user_comment, custom_value, language):
        if not file_bytes:
            return None, {"state": "error", "message_key": "upload_first"}, True

        file_content = base64.b64decode(file_bytes)
        alpha = (confidence or 95) / 100
        comment = user_comment or ""
        language = _get_language(language)

        try:
            custom_val = _parse_custom_value(custom_value)
            parameter_value = load_parameter(BytesIO(file_content), param_key)
            try:
                order_number = load_parameter(BytesIO(file_content), "Auftrags-Nr.")
            except ValueError:
                order_number = None
            _, id_unit, df_data = load_data(BytesIO(file_content), ["sc", "Fmax"])
            data, series_key, sym, _ = extract_data(df_data)
            shape, scale, unbiased_shape, ci_shape, ci_scale, d_stat, p_val, bootstrap_samples = (
                calculate_weibull_parameters(data, alpha)
            )
        except Exception as exc:
            return None, {"state": "error", "message": str(exc)}, True

        failure_prob = (
            weibull_min.cdf(custom_val, c=unbiased_shape, scale=scale) if custom_val is not None else None
        )
        summary = _build_summary(
            param_key,
            parameter_value,
            order_number,
            series_key,
            id_unit,
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
            custom_val,
            failure_prob,
            comment,
            language,
        )
        summary["data_symbol"] = sym

        analysis_payload = {
            "summary": summary,
            "raw_data": data.tolist(),
        }

        return analysis_payload, {"state": "ready"}, False

    @app.callback(
        Output("analysis-summary", "children"),
        Output("plot-container", "children"),
        Input("analysis-data", "data"),
        Input("analysis-meta", "data"),
        Input("language", "value"),
    )
    def render_analysis(analysis_data, analysis_meta, language):
        language = _get_language(language)
        meta = analysis_meta or {"state": "idle"}
        if meta.get("state") == "error":
            message = _t(language, meta["message_key"]) if meta.get("message_key") else meta.get("message", "")
            return _t(language, "error_prefix", message=message), None
        if not analysis_data:
            return None, None

        summary = analysis_data["summary"]
        plot_img = html.Img(
            src=f"data:image/png;base64,{_render_plot_base64(analysis_data, language)}",
            style={"maxWidth": "100%"},
        )
        return _build_analysis_summary(summary, analysis_data["raw_data"], language), plot_img

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
        print(f"Weibull-Tool Version: {__version__}")
    else:
        debug = os.getenv("DASH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        port = int(os.getenv("PORT", "8053"))
        app.run_server(debug=debug, host="0.0.0.0", port=port)
