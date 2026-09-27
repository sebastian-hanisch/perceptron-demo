"""Regressionstest fuer einen echten Bug: bei fast-senkrechten Trennebenen
(kleines w[1]) explodierte die naive y=f(x)-Auswertung der Grenzlinie und riss
die Plotly-Achsen mit (y-Achse zeigte +-600 statt der gesetzten +-2 Box)."""
import numpy as np

import pc_visualization as viz


def test_clip_line_to_box_handles_near_vertical_line_without_blowup():
    w = np.array([1.0, 1e-8])  # fast senkrecht: w1 fast 0
    b = 0.0
    x_range, y_range = (-4.0, 4.0), (-2.0, 2.0)
    segment = viz._clip_line_to_box(w, b, x_range, y_range)
    assert segment is not None
    for (x, y) in segment:
        assert x_range[0] - 1e-6 <= x <= x_range[1] + 1e-6
        assert y_range[0] - 1e-6 <= y <= y_range[1] + 1e-6


def test_clip_line_to_box_handles_near_horizontal_line_without_blowup():
    w = np.array([1e-8, 1.0])  # fast waagrecht: w0 fast 0
    b = 0.5
    x_range, y_range = (-4.0, 4.0), (-2.0, 2.0)
    segment = viz._clip_line_to_box(w, b, x_range, y_range)
    assert segment is not None
    for (x, y) in segment:
        assert x_range[0] - 1e-6 <= x <= x_range[1] + 1e-6
        assert y_range[0] - 1e-6 <= y <= y_range[1] + 1e-6


def test_clip_line_to_box_returns_none_for_zero_weight_vector():
    assert viz._clip_line_to_box(np.array([0.0, 0.0]), 0.0, (-4.0, 4.0), (-2.0, 2.0)) is None


def test_clip_line_to_box_returns_none_when_line_misses_the_box():
    # Gerade x = 100 (weit ausserhalb der Box)
    w = np.array([1.0, 0.0])
    b = -100.0
    assert viz._clip_line_to_box(w, b, (-4.0, 4.0), (-2.0, 2.0)) is None


def test_build_boundary_figure_never_exceeds_the_requested_axis_range():
    X = np.array([[1.0, 0.0], [-1.0, 0.0]])
    y = np.array([1.0, -1.0])
    x_range, y_range = (-4.0, 4.0), (-2.0, 2.0)
    for w1 in (1e-9, 1e-6, 1.0):
        fig = viz.build_boundary_figure(X, y, np.array([1.0, w1]), 0.0, x_range, y_range)
        assert fig.layout.yaxis.range == y_range
        assert fig.layout.xaxis.range == x_range
        for trace in fig.data:
            if trace.name == "Entscheidungsgrenze":
                assert all(y_range[0] - 1e-6 <= v <= y_range[1] + 1e-6 for v in trace.y)
