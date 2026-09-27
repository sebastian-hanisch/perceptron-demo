import pc_constants as C
import pc_evaluation as ev


def test_analyse_trennbar_reports_bound_that_never_undershoots_real_updates():
    for seed in range(20):
        settings = ev.Settings("trennbar", 40, seed, 1.0, 2000, 4.0)
        out = ev.analyse(settings)
        assert out["result"].converged
        assert out["novikoff_bound"] >= out["result"].updates


def test_analyse_trennbar_bound_is_loose_in_practice():
    """Ergebnis aus der Vormessung: die Schranke wird selten zu mehr als
    einem kleinen Bruchteil ausgenutzt."""
    utilisations = []
    for seed in range(20):
        settings = ev.Settings("trennbar", 40, seed, 1.0, 2000, 4.0)
        out = ev.analyse(settings)
        utilisations.append(out["result"].updates / out["novikoff_bound"])
    assert max(utilisations) < 0.5


def test_analyse_xor_never_converges():
    settings = ev.Settings("xor", 40, 1, 1.0, 2000, 3.0)
    out = ev.analyse(settings)
    assert not out["result"].converged
    assert "novikoff_bound" not in out


def test_oscillation_metric_shows_no_stabilisation_on_xor():
    settings = ev.Settings("xor", 40, 1, 1.0, 2000, 3.0)
    out = ev.analyse(settings)
    osc = ev.oscillation_metric(out["result"])
    assert osc["never_zero_errors"]
    assert osc["min_errors_in_tail"] > 0
    assert osc["delta_mean"] > 0.05


def test_margin_sweep_bound_grows_as_margin_shrinks():
    rows = ev.margin_sweep(gaps=(4.0, 3.0, 2.4), seeds=range(5), max_epochs=20000)
    bounds = [r["bound"] for r in rows]
    assert bounds == sorted(bounds)  # gap faellt -> gamma faellt -> Schranke steigt
    for row in rows:
        assert row["violations"] == 0
