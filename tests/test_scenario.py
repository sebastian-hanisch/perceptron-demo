import numpy as np
import pytest

import pc_scenario as sc


def test_make_separable_is_reproducible():
    X1, y1 = sc.make_separable(40, seed=3, spread=1.0, gap=4.0)
    X2, y2 = sc.make_separable(40, seed=3, spread=1.0, gap=4.0)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_make_separable_rejects_insufficient_gap():
    with pytest.raises(ValueError):
        sc.make_separable(40, seed=1, spread=1.0, gap=1.5)


def test_make_separable_is_always_linearly_separable_across_many_seeds():
    """Konstruktive Garantie: w=(1,0), b=0 trennt IMMER, wenn gap > 2*spread."""
    spread, gap = 1.0, 4.0
    for seed in range(50):
        X, y = sc.make_separable(40, seed=seed, spread=spread, gap=gap)
        margins = y * X[:, 0]  # y_i * (w.x_i + b) mit w=(1,0), b=0
        assert np.all(margins > 0), f"seed={seed}: nicht getrennt durch w=(1,0)"


def test_make_separable_margin_matches_exact_formula():
    spread, gap = 1.0, 4.0
    gamma_formula = sc.exact_margin_trennbar(spread, gap)
    for seed in range(20):
        X, y = sc.make_separable(40, seed=seed, spread=spread, gap=gap)
        margins = y * X[:, 0]
        assert margins.min() >= gamma_formula - 1e-9


def test_make_xor_has_all_four_quadrant_labels():
    X, y = sc.make_xor(40, seed=1, spread=0.6, gap=3.0)
    # Klassisches XOR: Vorzeichen von x1*x2 sollte (bis auf Rauschgrenze) mit y uebereinstimmen
    signs = np.sign(X[:, 0]) * np.sign(X[:, 1])
    agree = np.mean(signs == y)
    assert agree > 0.95


def test_make_xor_not_linearly_separable_by_any_axis_aligned_or_diagonal_line():
    """Kein einzelnes Perceptron trennt XOR: Testet mehrere naheliegende
    Kandidaten-Trennebenen (Achsen, Diagonalen) und zeigt, dass jede >= 25%
    der Punkte falsch klassifiziert (kein Zufallsartefakt)."""
    X, y = sc.make_xor(80, seed=1, spread=0.6, gap=3.0)
    candidates = [
        (np.array([1.0, 0.0]), 0.0),
        (np.array([0.0, 1.0]), 0.0),
        (np.array([1.0, 1.0]), 0.0),
        (np.array([1.0, -1.0]), 0.0),
    ]
    for w, b in candidates:
        scores = X @ w + b
        pred = np.where(scores >= 0, 1.0, -1.0)
        error_rate = np.mean(pred != y)
        assert error_rate >= 0.25, f"w={w}: Fehlerrate nur {error_rate:.2f}"


def test_build_scenario_dispatches_by_mode():
    s1 = sc.build_scenario("trennbar", 20, 1, 1.0, 4.0)
    assert s1.mode == "trennbar"
    s2 = sc.build_scenario("xor", 20, 1, 0.6, 3.0)
    assert s2.mode == "xor"
    with pytest.raises(ValueError):
        sc.build_scenario("unbekannt", 20, 1, 1.0, 4.0)
