import base64
import json
from io import BytesIO
from pathlib import Path
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
    plot_weibull,
    render_plot_to_png_bytes,
    __version__,
)


def _decode_upload(contents: str) -> bytes:
    if not contents:
        raise ValueError("Keine Datei hochgeladen.")
    try:
        _, content_string = contents.split(",")
        return base64.b64decode(content_string)
    except Exception as exc:
        raise ValueError(f"Upload konnte nicht gelesen werden: {exc}")


def _safe_path(input_path: str) -> Path:
    base_dir = Path.cwd().resolve()
    path = Path(input_path)
    path = path if path.is_absolute() else base_dir / path
    resolved = path.resolve()
    if base_dir not in resolved.parents and resolved != base_dir:
        raise ValueError("Zielordner muss innerhalb des Projektverzeichnisses liegen.")
    return resolved


def _build_summary(
    data_label,
    identifier,
    unit,
    data,
    shape,
    scale,
    unbiased_shape,
    ci_shape,
    ci_scale,
    d_stat,
    p_val,
    alpha,
    custom_value,
    failure_prob,
):
    return {
        "data_label": data_label,
        "identifier": identifier,
        "unit": unit,
        "n": int(len(data)),
        "shape_mle": float(shape),
        "scale_mle": float(scale),
        "unbiased_shape": float(unbiased_shape),
        "ci_shape": [float(ci_shape[0]), float(ci_shape[1])],
        "ci_scale": [float(ci_scale[0]), float(ci_scale[1])],
        "ad_statistic": float(d_stat),
        "p_value": float(p_val),
        "confidence_level": int(alpha * 100),
        "custom_value": None if custom_value is None else float(custom_value),
        "failure_probability": None if failure_prob is None else float(failure_prob),
        "code_version": __version__,
    }


def create_app():
    app = Dash(__name__, title="Weibull Tool", suppress_callback_exceptions=True)
    server = app.server

    app.layout = html.Div(
        className="app",
        children=[
            html.H1("Weibull Analysis Tool - Dash"),
            html.P(
                [
                    "Excel hochladen (Drag & Drop), Parameter waehlen und die Analyse starten. ",
                    "Das Ergebnis kann als ZIP in einen frei waehlbaren Ordner heruntergeladen werden.",
                ]
            ),
            html.Div(
                className="upload",
                children=[
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(["Datei hierher ziehen oder ", html.A("auswaehlen")]),
                        multiple=False,
                        style={
                            "width": "100%",
                            "height": "80px",
                            "lineHeight": "80px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "margin": "10px 0",
                        },
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
                            html.Label("Parameter-Schluessel"),
                            dcc.Dropdown(
                                id="param-key",
                                options=[
                                    {"label": "Auftrags-Nr.", "value": "Auftrags-Nr."},
                                    {"label": "Werkstoff", "value": "Werkstoff"},
                                ],
                                value="Werkstoff",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="control",
                        children=[
                            html.Label("Konfidenzniveau"),
                            dcc.RadioItems(
                                id="confidence",
                                options=[{"label": f"{p}%", "value": p} for p in (90, 95, 99)],
                                value=95,
                                labelStyle={"display": "inline-block", "marginRight": "12px"},
                            ),
                        ],
                    ),
                    html.Div(
                        className="control",
                        children=[
                            html.Label("Benutzer-Kommentar"),
                            dcc.Input(id="user-comment", placeholder="Optional", type="text", style={"width": "100%"}),
                        ],
                    ),
                    html.Div(
                        className="control",
                        children=[
                            html.Label("Optional: Wert fuer Ausfallwahrscheinlichkeit"),
                            dcc.Input(id="custom-value", type="number", min=0, step=1),
                        ],
                    ),
                    html.Div(
                        className="control",
                        children=[
                            html.Label("Zielordner (Server, optional)", title="Standard: ./exports"),
                            dcc.Input(
                                id="save-directory",
                                type="text",
                                placeholder="z.B. exports/meine-auswertung",
                                style={"width": "100%"},
                            ),
                        ],
                    ),
                    html.Button("Analyse starten", id="analyze", n_clicks=0, className="primary"),
                ],
            ),
            html.Div(id="analysis-summary", className="summary"),
            html.Div(id="plot-container", className="plot"),
            html.Div(
                className="download",
                children=[
                    html.Button("Ergebnis herunterladen (ZIP)", id="download-btn", n_clicks=0, disabled=True),
                    html.Div(id="download-status", className="status"),
                    dcc.Download(id="download-bundle"),
                ],
            ),
            dcc.Store(id="file-bytes"),
            dcc.Store(id="analysis-data"),
        ],
    )

    @app.callback(
        Output("upload-status", "children"),
        Output("file-bytes", "data"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
    )
    def handle_upload(contents, filename):
        if contents is None:
            return "Keine Datei hochgeladen.", None
        try:
            decoded = _decode_upload(contents)
        except ValueError as exc:
            return str(exc), None
        name_display = filename or "Unbenannt"
        return f"Datei erhalten: {name_display}", base64.b64encode(decoded).decode()

    @app.callback(
        Output("analysis-summary", "children"),
        Output("plot-container", "children"),
        Output("analysis-data", "data"),
        Output("download-btn", "disabled"),
        Input("analyze", "n_clicks"),
        State("file-bytes", "data"),
        State("param-key", "value"),
        State("confidence", "value"),
        State("user-comment", "value"),
        State("custom-value", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, file_bytes, param_key, confidence, user_comment, custom_value):
        if not file_bytes:
            return "Bitte erst eine Excel-Datei hochladen.", None, None, True

        file_content = base64.b64decode(file_bytes)
        alpha = (confidence or 95) / 100
        comment = user_comment or ""
        custom_val = float(custom_value) if custom_value not in (None, "") else None

        try:
            data_label = load_parameter(BytesIO(file_content), param_key)
            identifier, id_unit, df_data = load_data(BytesIO(file_content), ["sc", "Fmax"])
            data, col, sym, title = extract_data(df_data)
            shape, scale, unbiased_shape, ci_shape, ci_scale, d_stat, p_val = calculate_weibull_parameters(
                data, alpha
            )
        except Exception as exc:
            return f"Fehler: {exc}", None, None, True

        p_lin = np.linspace(0.01, 0.99, 500)
        lower_ci = weibull_min.ppf(p_lin, c=ci_shape[0], scale=ci_scale[0])
        upper_ci = weibull_min.ppf(p_lin, c=ci_shape[1], scale=ci_scale[1])
        failure_prob = (
            weibull_min.cdf(custom_val, c=unbiased_shape, scale=scale) if custom_val is not None else None
        )
        fig = plot_weibull(
            data,
            f"{data_label} ({identifier})",
            id_unit,
            scale,
            unbiased_shape,
            lower_ci,
            upper_ci,
            custom_val,
            failure_prob,
            comment,
            sym,
            title,
            d_stat,
            p_val,
            alpha,
            ci_shape,
            ci_scale,
        )
        img_bytes = render_plot_to_png_bytes(fig)
        img_b64 = base64.b64encode(img_bytes).decode()
        summary = _build_summary(
            data_label,
            identifier,
            id_unit,
            data,
            shape,
            scale,
            unbiased_shape,
            ci_shape,
            ci_scale,
            d_stat,
            p_val,
            alpha,
            custom_val,
            failure_prob,
        )

        summary_items = [
            html.H3("Ergebnis"),
            html.Ul(
                [
                    html.Li(f"Stichprobe n = {summary['n']}, Parameter: {summary['data_label']} ({summary['identifier']})"),
                    html.Li(f"Weibull-Modul (unbiased) m = {summary['unbiased_shape']:.2f}"),
                    html.Li(f"Charakteristischer Wert = {summary['scale_mle']:.1f} {id_unit}"),
                    html.Li(f"Konfidenzniveau: {summary['confidence_level']}%"),
                    html.Li(f"AD-Statistik: {summary['ad_statistic']:.3f} / p-Wert: {summary['p_value']:.3f}"),
                ]
            ),
        ]

        plot_img = html.Img(src=f"data:image/png;base64,{img_b64}", style={"maxWidth": "100%"})

        analysis_payload = {
            "summary": summary,
            "plot_png": img_b64,
            "raw_data": data.tolist(),
        }

        return summary_items, plot_img, analysis_payload, False

    @app.callback(
        Output("download-bundle", "data"),
        Output("download-status", "children"),
        Input("download-btn", "n_clicks"),
        State("analysis-data", "data"),
        State("save-directory", "value"),
        prevent_initial_call=True,
    )
    def trigger_download(n_clicks, analysis_data, save_directory):
        if not analysis_data:
            return no_update, "Bitte zuerst eine Analyse ausfuehren."

        status_messages = []
        if save_directory:
            try:
                target_dir = _safe_path(save_directory)
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / "weibull_plot.png").write_bytes(base64.b64decode(analysis_data["plot_png"]))
                (target_dir / "weibull_results.json").write_text(
                    json.dumps(analysis_data["summary"], indent=2), encoding="utf-8"
                )
                status_messages.append(f"Auf Server gespeichert unter: {target_dir}")
            except Exception as exc:
                status_messages.append(f"Konnte nicht speichern: {exc}")

        def build_zip_bytes(buffer: BytesIO):
            """Dash send_bytes writer: fills provided buffer with the zip payload."""
            with zipfile.ZipFile(buffer, "w") as zf:
                zf.writestr("weibull_plot.png", base64.b64decode(analysis_data["plot_png"]))
                zf.writestr("weibull_results.json", json.dumps(analysis_data["summary"], indent=2))
                zf.writestr("input_data.csv", "\n".join(str(val) for val in analysis_data.get("raw_data", [])))

        return send_bytes(build_zip_bytes, "weibull_output.zip"), " | ".join(status_messages)

    return app, server


app, server = create_app()

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8053)
