import pc_constants as C
import pc_evaluation as ev
import pc_presets as pr


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert preset["mode"] in C.MODES
        b = pr.bounds(preset["mode"])
        assert b["gap_min"] <= preset["gap"] <= b["gap_max"]
        assert C.N_MIN <= preset["n"] <= C.N_MAX
        assert C.ETA_MIN <= preset["eta"] <= C.ETA_MAX
        assert C.MAX_EPOCHS_MIN <= preset["max_epochs"] <= C.MAX_EPOCHS_MAX


def test_preset_gross_converges_with_few_updates():
    p = C.PRESETS["gross"]
    settings = ev.Settings(p["mode"], p["n"], p["seed"], p["eta"], p["max_epochs"], p["gap"])
    out = ev.analyse(settings)
    assert out["result"].converged
    assert out["result"].updates < 50


def test_preset_knapp_has_much_smaller_margin_than_gross():
    p_gross = C.PRESETS["gross"]
    p_knapp = C.PRESETS["knapp"]
    gamma_gross = pr.bounds(p_gross["mode"])
    out_gross = ev.analyse(ev.Settings(p_gross["mode"], p_gross["n"], p_gross["seed"],
                                       p_gross["eta"], p_gross["max_epochs"], p_gross["gap"]))
    out_knapp = ev.analyse(ev.Settings(p_knapp["mode"], p_knapp["n"], p_knapp["seed"],
                                       p_knapp["eta"], p_knapp["max_epochs"], p_knapp["gap"]))
    assert out_knapp["gamma"] < out_gross["gamma"]
    assert out_knapp["novikoff_bound"] > out_gross["novikoff_bound"]


def test_preset_xor_never_converges():
    p = C.PRESETS["xor"]
    settings = ev.Settings(p["mode"], p["n"], p["seed"], p["eta"], p["max_epochs"], p["gap"])
    out = ev.analyse(settings)
    assert not out["result"].converged


def test_bounds_returns_disjoint_but_valid_ranges_per_mode():
    b_trennbar = pr.bounds(C.MODE_TRENNBAR)
    b_xor = pr.bounds(C.MODE_XOR)
    assert b_trennbar["gap_min"] < b_trennbar["gap_max"]
    assert b_xor["gap_min"] < b_xor["gap_max"]
