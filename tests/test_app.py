from streamlit.testing.v1 import AppTest


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Über mich" in c for c in captions)


def test_default_mode_is_trennbar_and_converges():
    at = _fresh()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Konvergiert?"] == "Ja"


def test_preset_buttons_exist_and_xor_preset_switches_mode():
    at = _fresh()
    labels = [b.label for b in at.button]
    assert "XOR-Muster" in labels
    at.button(key=None).__class__  # smoke: buttons list is non-empty
    xor_button = [b for b in at.button if b.label == "XOR-Muster"][0]
    xor_button.click().run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Konvergiert?"] == "Nein (Max. Epochen erreicht)"


def test_gap_slider_bounds_switch_with_mode():
    at = _fresh()
    xor_button = [b for b in at.button if b.label == "XOR-Muster"][0]
    xor_button.click().run()
    gap_slider = [s for s in at.slider if s.label.startswith("Abstand")][0]
    assert gap_slider.min <= 1.5 + 1e-9
    assert gap_slider.max <= 5.0 + 1e-9


def test_n_slider_extreme_values_do_not_crash():
    at = _fresh()
    n_slider = [s for s in at.slider if s.label.startswith("Anzahl")][0]
    n_slider.set_value(n_slider.min).run()
    assert not at.exception
    n_slider = [s for s in at.slider if s.label.startswith("Anzahl")][0]
    n_slider.set_value(n_slider.max).run()
    assert not at.exception


def test_step_slider_can_move_and_no_duplicate_key_crash():
    at = _fresh()
    step_sliders = [s for s in at.select_slider]
    assert len(step_sliders) == 1
    step = step_sliders[0]
    if len(step.options) > 1:
        step.set_value(step.options[0]).run()
        assert not at.exception
        step.set_value(step.options[-1]).run()
        assert not at.exception
