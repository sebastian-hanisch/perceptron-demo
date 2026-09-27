"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen
neu berechnet, mit denselben Presets/Seeds wie im README zitiert."""
import pc_constants as C
import pc_evaluation as ev


def _analyse_preset(key):
    p = C.PRESETS[key]
    settings = ev.Settings(p["mode"], p["n"], p["seed"], p["eta"], p["max_epochs"], p["gap"])
    return ev.analyse(settings)


def test_claim_preset_gross_converges_in_2_epochs_1_update():
    out = _analyse_preset("gross")
    r = out["result"]
    assert r.converged
    assert r.epochs == 2
    assert r.updates == 1
    assert round(out["gamma"], 3) == 1.000
    assert round(out["novikoff_bound"], 1) == 8.6


def test_claim_preset_knapp_converges_fast_despite_huge_bound():
    out = _analyse_preset("knapp")
    r = out["result"]
    assert r.converged
    assert r.epochs == 2
    assert r.updates == 2
    assert round(out["gamma"], 3) == 0.050
    assert out["novikoff_bound"] > 1500  # Schranke ist riesig ...
    assert r.updates / out["novikoff_bound"] < 0.01  # ... aber real kaum ausgenutzt


def test_claim_preset_xor_never_converges_within_2000_epochs():
    out = _analyse_preset("xor")
    r = out["result"]
    assert not r.converged
    assert r.epochs == 2000
    osc = ev.oscillation_metric(r)
    assert osc["never_zero_errors"]
    assert osc["min_errors_in_tail"] == 18
    assert osc["delta_mean"] > 2.0


def test_claim_max_bound_utilisation_over_20_seeds_stays_under_13_percent():
    utilisations = []
    for seed in range(20):
        settings = ev.Settings("trennbar", 40, seed, 1.0, 2000, 4.0)
        out = ev.analyse(settings)
        utilisations.append(out["result"].updates / out["novikoff_bound"])
    assert max(utilisations) * 100 < 13.0


def test_claim_margin_sweep_table_matches_readme():
    rows = ev.margin_sweep()
    expected = {
        4.0: dict(gamma=1.0, max_updates=1, violations=0),
        3.0: dict(gamma=0.5, max_updates=2, violations=0),
        2.4: dict(gamma=0.2, max_updates=2, violations=0),
        2.1: dict(gamma=0.05, max_updates=4, violations=0),
        2.02: dict(gamma=0.01, max_updates=4, violations=0),
    }
    for row in rows:
        exp = expected[row["gap"]]
        assert round(row["gamma"], 2) == exp["gamma"]
        assert row["max_updates"] == exp["max_updates"]
        assert row["violations"] == exp["violations"]
        # Schranke steigt monoton, je kleiner die Marge
    bounds = [r["bound"] for r in rows]
    assert bounds == sorted(bounds)


def test_claim_no_violations_of_novikoff_bound_across_50_seeds():
    for seed in range(50):
        settings = ev.Settings("trennbar", 40, seed, 1.0, 2000, 4.0)
        out = ev.analyse(settings)
        assert out["result"].converged
        assert out["result"].updates <= out["novikoff_bound"]
