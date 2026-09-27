"""Perceptron — wo hört eine Gerade auf zu reichen?

Sebastian Hanisch - Operations Research und Machine Learning

Erstes Stück (Wurzel) der "Neuronale Netze"-Reihe der "Konzepte"-Reihe:
Perceptron -> MLP+Backpropagation -> {CNN, RNN -> LSTM -> Attention/Transformer}.
Diese App zeigt die klassische Perceptron-Lernregel (Rosenblatt 1958): auf
linear trennbaren Daten konvergiert sie beweisbar in endlich vielen Schritten
(Novikoff 1962), auf dem XOR-Muster scheitert sie ebenso beweisbar (Minsky &
Papert 1969) - beides hier selbst nachgerechnet statt nur zitiert.

Lauffähig mit: streamlit run app.py
"""
import numpy as np
import streamlit as st

import pc_constants as C
import pc_evaluation as ev
import pc_perceptron as perc
import pc_presets as pr
import pc_visualization as viz

st.set_page_config(page_title="Perceptron", layout="wide")


@st.cache_data(show_spinner=False)
def _analyse(mode, n, seed, eta, max_epochs, gap):
    settings = ev.Settings(mode=mode, n=n, seed=seed, eta=eta, max_epochs=max_epochs, gap=gap)
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _margin_sweep():
    return ev.margin_sweep()


def _german(x: float, digits: int = 0) -> str:
    s = f"{x:,.{digits}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


st.title("🧠 Perceptron — wo hört eine Gerade auf zu reichen?")
st.markdown(
    "Ein Perceptron ist der einfachste lernende Klassifikator: eine Gerade "
    "(bzw. Hyperebene), die per **Fehler-getriebener Regel** so lange verschoben "
    "wird, bis alle Punkte richtig liegen. Zwei Fragen stehen im Zentrum: "
    "Wie schnell konvergiert das, wenn eine trennende Gerade existiert? "
    "Und was passiert, wenn keine existiert?"
)
st.caption(
    "Wurzel der 'Neuronale Netze'-Reihe. Geplante Folgestücke (noch nicht gebaut): "
    "MLP+Backpropagation, CNN, RNN, LSTM, Attention/Transformer."
)

with st.expander("So funktioniert die Perceptron-Lernregel", expanded=True):
    st.markdown(
        "1. Starte mit Gewichten $w=0$, Bias $b=0$.\n"
        "2. Gehe die Punkte der Reihe nach durch. Für jeden Punkt $(x_i, y_i)$ "
        "mit $y_i \\in \\{-1, +1\\}$: berechne die Marge $y_i (w \\cdot x_i + b)$.\n"
        "3. Ist die Marge $\\le 0$ (Punkt falsch oder genau auf der Grenze), "
        "korrigiere: $w \\leftarrow w + \\eta\\, y_i x_i$, $b \\leftarrow b + \\eta\\, y_i$.\n"
        "4. Ein Durchlauf ohne eine einzige Korrektur = **Konvergenz**."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                   use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    mode = st.radio(
        "Datenmodus", C.MODES, format_func=lambda m: C.MODE_LABELS[m],
        key="widget_mode", index=C.MODES.index(ss["mode"]),
        on_change=pr.store_from_widget, args=("mode",),
    )
    ss["mode"] = mode
    b = pr.bounds(mode)
    gap_value = min(max(ss["gap"], b["gap_min"]), b["gap_max"])
    gap = st.slider(
        "Abstand der Zentren (gap)", b["gap_min"], b["gap_max"], gap_value, step=0.05,
        key="widget_gap", on_change=pr.store_from_widget, args=("gap",),
        help="Streuung ist fest (1.0 bzw. 0.6); dieser Regler steuert direkt die Marge γ.",
    )
    ss["gap"] = gap
    n = st.slider("Anzahl Punkte n", C.N_MIN, C.N_MAX, ss["n"], step=4,
                  key="widget_n", on_change=pr.store_from_widget, args=("n",))
    ss["n"] = n
    eta = st.slider("Lernrate η", C.ETA_MIN, C.ETA_MAX, ss["eta"], step=0.1,
                    key="widget_eta", on_change=pr.store_from_widget, args=("eta",))
    ss["eta"] = eta
    max_epochs = st.slider("Max. Epochen", C.MAX_EPOCHS_MIN, C.MAX_EPOCHS_MAX, ss["max_epochs"],
                           step=50, key="widget_max_epochs",
                           on_change=pr.store_from_widget, args=("max_epochs",))
    ss["max_epochs"] = max_epochs
    seed = st.number_input("Seed", value=ss["seed"], step=1,
                           key="widget_seed", on_change=pr.store_from_widget, args=("seed",))
    ss["seed"] = seed
    st.button("🎲 Zufälliger Seed", on_click=pr.randomize_seed)

pr.sync_query_params(dict(mode=mode, n=n, seed=seed, eta=eta, max_epochs=max_epochs, gap=gap))

with st.spinner("Rechne..."):
    out = _analyse(mode, n, int(seed), eta, max_epochs, gap)
scenario = out["scenario"]
result = out["result"]

st.markdown("---")
st.subheader("🎯 Konvergenz Schritt für Schritt")

max_step = len(result.errors_per_epoch)
step = st.select_slider("Epoche", options=list(range(1, max_step + 1)), value=max_step,
                        key=f"step_{mode}_{n}_{seed}_{gap}_{max_epochs}")
w_b = result.w_history[step - 1]
w_step, b_step = w_b[:-1], w_b[-1]
x_range, y_range = viz.data_bounds(scenario.X)

col_left, col_right = st.columns([3, 2])
with col_left:
    fig_boundary = viz.build_boundary_figure(
        scenario.X, scenario.y, w_step, b_step, x_range, y_range,
        title=f"Entscheidungsgrenze nach Epoche {step}",
    )
    st.plotly_chart(fig_boundary, key=f"boundary_{mode}_{step}_{n}_{seed}_{gap}_{max_epochs}",
                    use_container_width=True)
with col_right:
    fig_errors = viz.build_errors_figure(result.errors_per_epoch, current_epoch=step)
    st.plotly_chart(fig_errors, key=f"errors_{mode}_{step}_{n}_{seed}_{gap}_{max_epochs}",
                    use_container_width=True)

st.markdown("---")
st.subheader("🎯 Was am Ende steht")
m1, m2, m3 = st.columns(3)
m1.metric("Konvergiert?", "Ja" if result.converged else "Nein (Max. Epochen erreicht)")
m2.metric("Epochen bis zum Ende", f"{result.epochs}")
m3.metric("Updates insgesamt", f"{_german(result.updates)}")

if mode == C.MODE_TRENNBAR:
    c1, c2, c3 = st.columns(3)
    c1.metric("Marge γ (exakt)", f"{out['gamma']:.3f}")
    c2.metric("R = max‖x‖", f"{out['R']:.2f}")
    c3.metric("Novikoff-Schranke (R/γ)²", f"{_german(out['novikoff_bound'])}")
    st.caption(
        f"Reale Updates ({result.updates}) liegen deutlich unter der Schranke "
        f"({_german(out['novikoff_bound'])}) — die Schranke ist ein gültiges, aber "
        "sehr grobes Worst-Case-Versprechen, kein präziser Vorhersagewert."
    )
else:
    osc = ev.oscillation_metric(result)
    o1, o2, o3 = st.columns(3)
    o1.metric("Min. Fehlerzahl (letzte 200 Epochen)", f"{osc['min_errors_in_tail']}")
    o2.metric("Fehlerzahl je 0?", "Nein" if osc["never_zero_errors"] else "Ja (Bug!)")
    o3.metric("Gewichtsänderung/Epoche (letzte 200)",
              f"{osc['delta_min']:.2f} – {osc['delta_max']:.2f}")
    st.caption(
        "Die Gewichte stabilisieren sich nicht (Änderungsnorm bleibt deutlich über 0) — "
        "das Perceptron oszilliert endlos, statt nur langsam zu konvergieren."
    )

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Daten sind exakt linear trennbar | Perceptron oszilliert endlos statt zu konvergieren "
    "(siehe XOR-Modus) | MLP + Backpropagation (nächstes Stück) |\n"
    "| Feste Reihenfolge der Punkte je Epoche | Andere Reihenfolge kann andere, "
    "gleich gültige Trennebenen liefern (Perceptron ist nicht eindeutig) | — |\n"
    "| Marge γ > 0 bekannt/konstruierbar | In der Praxis unbekannt — die Schranke "
    "ist dann nur a-posteriori auswertbar | — |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Update-Regel** (Rosenblatt 1958): für $(x_i, y_i)$ mit $y_i \in \{-1,+1\}$, falls
$y_i(w \cdot x_i + b) \le 0$: $w \leftarrow w + \eta y_i x_i,\ b \leftarrow b + \eta y_i$.

**Novikoff-Schranke** (Novikoff 1962): existiert eine Trennebene mit Marge
$\gamma = \min_i y_i(w^\* \cdot x_i + b^\*)/\|w^\*\|$ und ist $R = \max_i \|x_i\|$,
dann macht die Perceptron-Regel höchstens $(R/\gamma)^2$ Fehler (Updates) bis zur
Konvergenz. Hier wird die Trennebene $w=(1,0), b=0$ konstruktiv verwendet
(gap > 2·spread garantiert Trennbarkeit), ihre exakte Marge ist
$\gamma = \text{gap}/2 - \text{spread}$ — kein SVM/LP-Fit nötig.
"""
    )
    sweep = _margin_sweep()
    st.plotly_chart(viz.build_margin_sweep_figure(sweep), key="margin_sweep_chart",
                    use_container_width=True)
    st.caption(
        "Je kleiner die Marge γ, desto größer die Schranke (R/γ)² — in der Praxis "
        "bleibt die reale Update-Zahl aber weit darunter (siehe Tabelle im README)."
    )
    st.markdown(
        "**Literatur:** Rosenblatt, F. (1958). *The Perceptron.* Psychological Review, "
        "65(6), 386–408. — Novikoff, A. B. J. (1962). *On Convergence Proofs on "
        "Perceptrons.* Symposium on Mathematical Theory of Automata, 12, 615–622. — "
        "Minsky, M. & Papert, S. (1969). *Perceptrons.* MIT Press."
    )
    st.caption(
        "Implementiert in `pc_perceptron.py` (Lernregel), `pc_scenario.py` (Daten), "
        "`pc_evaluation.py` (Schranke, Sweep), `pc_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) "
    "– Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung "
    "für Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
