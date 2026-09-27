"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe. Achsen werden fest
uebergeben (feste Wertebereiche ueber die ganze Animation hinweg, siehe
feedback_plotly_scaleanchor_explicit_range/feedback_plotly_fixedrange_convention)."""
import numpy as np
import plotly.graph_objects as go

COLOR_POS = "#1f77b4"
COLOR_NEG = "#d62728"
COLOR_LINE = "#2ca02c"


def data_bounds(X: np.ndarray, pad: float = 1.0):
    x_min, y_min = X.min(axis=0) - pad
    x_max, y_max = X.max(axis=0) + pad
    return (float(x_min), float(x_max)), (float(y_min), float(y_max))


def _clip_line_to_box(w, b, x_range, y_range):
    """Schneidet die Gerade w0*x + w1*y + b = 0 mit der Box [x_range]x[y_range]
    und gibt die zwei Randpunkte des sichtbaren Segments zurueck (oder None,
    wenn die Gerade die Box nicht schneidet oder w=(0,0) ist). Robust auch bei
    fast-senkrechten/fast-waagrechten Geraden (kein Divisions-Blowup wie bei
    einer naiven y=f(x)-Auswertung ueber den ganzen x-Bereich)."""
    w0, w1 = w
    xmin, xmax = x_range
    ymin, ymax = y_range
    pts = []
    if abs(w1) > 1e-12:
        for x in (xmin, xmax):
            y_val = -(w0 * x + b) / w1
            if ymin - 1e-9 <= y_val <= ymax + 1e-9:
                pts.append((x, y_val))
    if abs(w0) > 1e-12:
        for y_val in (ymin, ymax):
            x_val = -(w1 * y_val + b) / w0
            if xmin - 1e-9 <= x_val <= xmax + 1e-9:
                pts.append((x_val, y_val))
    uniq = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    if len(uniq) > 2:
        best = max(
            ((uniq[i], uniq[j]) for i in range(len(uniq)) for j in range(i + 1, len(uniq))),
            key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
        )
        return best
    return uniq[0], uniq[1]


def build_boundary_figure(X, y, w, b, x_range, y_range, title=""):
    fig = go.Figure()
    pos = X[y > 0]
    neg = X[y < 0]
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1], mode="markers", name="Klasse +1",
        marker=dict(color=COLOR_POS, size=9),
    ))
    fig.add_trace(go.Scatter(
        x=neg[:, 0], y=neg[:, 1], mode="markers", name="Klasse -1",
        marker=dict(color=COLOR_NEG, size=9),
    ))
    # Entscheidungsgrenze w.x + b = 0, mit der sichtbaren Box geclippt (kein
    # Blowup bei fast-senkrechter/fast-waagrechter Geraden)
    if abs(w[0]) > 1e-12 or abs(w[1]) > 1e-12:
        segment = _clip_line_to_box(w, b, x_range, y_range)
        if segment is not None:
            (x1, y1), (x2, y2) = segment
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2], mode="lines", name="Entscheidungsgrenze",
                line=dict(color=COLOR_LINE, width=3),
            ))
    fig.update_layout(
        title=title, xaxis=dict(range=list(x_range), fixedrange=True, title="x1"),
        yaxis=dict(range=list(y_range), fixedrange=True, title="x2", scaleanchor="x"),
        showlegend=True, height=420, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_errors_figure(errors_per_epoch, current_epoch=None, title="Fehlerzahl je Epoche"):
    epochs = list(range(1, len(errors_per_epoch) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=errors_per_epoch, mode="lines", name="Fehler",
        line=dict(color=COLOR_NEG, width=2),
    ))
    if current_epoch is not None and 1 <= current_epoch <= len(errors_per_epoch):
        fig.add_trace(go.Scatter(
            x=[current_epoch], y=[errors_per_epoch[current_epoch - 1]],
            mode="markers", name="aktuelle Epoche",
            marker=dict(color=COLOR_LINE, size=12, symbol="star"),
        ))
    fig.update_layout(
        title=title, xaxis=dict(title="Epoche", fixedrange=True),
        yaxis=dict(title="Anzahl Fehler", fixedrange=True, rangemode="tozero"),
        showlegend=False, height=300, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_margin_sweep_figure(rows, title="Novikoff-Schranke vs. reale Updates"):
    gammas = [r["gamma"] for r in rows]
    bounds = [r["bound"] for r in rows]
    reals = [r["max_updates"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gammas, y=bounds, mode="lines+markers", name="Novikoff-Schranke (R/γ)²",
        line=dict(color=COLOR_NEG, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=gammas, y=reals, mode="lines+markers", name="reale Updates (Maximum über Seeds)",
        line=dict(color=COLOR_POS, width=2),
    ))
    fig.update_layout(
        title=title, xaxis=dict(title="Marge γ", fixedrange=True, autorange="reversed"),
        yaxis=dict(title="Anzahl Updates", fixedrange=True, type="log"),
        height=360, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
