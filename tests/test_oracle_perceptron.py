"""Unabhängiges Orakel: exakt rationale Referenz der Perceptron-Regel (augmentierter
Vektor, Fractions), Novikoff-Schranke (R^2+1)/gamma^2 mit gelerntem Bias, LP-Beweis
der Nicht-Trennbarkeit des XOR-Musters und seed-weise Ausnutzung im Marge-Sweep."""
import random
from fractions import Fraction as F

import numpy as np
import pytest

import pc_evaluation as ev
import pc_perceptron as perc
import pc_scenario as sc


def _ref_fit(X, y, eta, max_epochs):
    Xa = [[F(float(v)) for v in row] + [F(1)] for row in X]
    eta = F(float(eta))
    w = [F(0)] * len(Xa[0])
    updates, errs = 0, []
    for ep in range(max_epochs):
        e = 0
        for xi, yi in zip(Xa, y):
            if int(yi) * sum(a * b for a, b in zip(w, xi)) <= 0:
                w = [a + eta * int(yi) * b for a, b in zip(w, xi)]
                updates += 1
                e += 1
        errs.append(e)
        if e == 0:
            return True, ep + 1, updates, errs, w
    return False, max_epochs, updates, errs, w


def test_fit_matches_exact_rational_reference_and_strict_novikoff_bound():
    rng = random.Random(1)
    for _ in range(120):
        n = rng.choice([2, 3, 5, 10, 20, 40])
        gap = rng.choice([2.05, 2.4, 3.0, 4.0, 8.0])
        eta = rng.choice([0.1, 0.5, 1.0, 2.0])
        seed = rng.randrange(1000)
        X, y = sc.make_separable(n, seed, 1.0, gap)
        gamma = gap / 2 - 1.0
        assert (y * X[:, 0]).min() >= gamma - 1e-12  # konstruktive Marge stimmt
        res = perc.fit(X, y, eta=eta, max_epochs=20000)
        ok, ep, upd, errs, w = _ref_fit(X, y, eta, 20000)
        assert (res.converged, res.epochs, res.updates) == (ok, ep, upd)
        assert res.errors_per_epoch == errs
        np.testing.assert_allclose(np.append(res.w, res.b), [float(v) for v in w], rtol=1e-9, atol=1e-9)
        R = np.linalg.norm(X, axis=1).max()
        assert upd <= (R ** 2 + 1) / gamma ** 2  # strenge Schranke mit Bias als Koordinate


def test_xor_is_not_linearly_separable_by_lp():
    linprog = pytest.importorskip("scipy.optimize").linprog
    for seed in range(5):
        X, y = sc.make_xor(40, seed, 0.6, 3.0)
        A = -(y[:, None] * np.hstack([X, np.ones((len(y), 1))]))
        r = linprog(np.zeros(3), A_ub=A, b_ub=-np.ones(len(y)), bounds=[(None, None)] * 3, method="highs")
        assert r.status == 2  # unzulässig: keine trennende Ebene


def test_margin_sweep_utilisation_is_max_of_per_seed_quotients():
    seeds = range(6)
    rows = ev.margin_sweep(gaps=(4.0, 3.0), seeds=seeds, max_epochs=2000)
    for row in rows:
        best = 0.0
        for s in seeds:
            out = ev.analyse(ev.Settings("trennbar", 40, s, 1.0, 2000, row["gap"]))
            best = max(best, out["result"].updates / out["novikoff_bound"])
        assert row["max_utilisation"] == pytest.approx(best)
