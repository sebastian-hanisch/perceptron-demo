# Perceptron – wo hört eine Gerade auf zu reichen? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-perceptron-demo.streamlit.app/)**

Erstes Stück (Wurzel) der **Neuronale-Netze-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Ein Perceptron ist der einfachste lernende Klassifikator: eine Gerade (bzw. Hyperebene), die
per **Fehler-getriebener Regel** (Rosenblatt 1958) so lange verschoben wird, bis alle Punkte
richtig liegen. Zwei Fragen stehen im Zentrum: Wie schnell konvergiert das, wenn eine trennende
Gerade existiert (Novikoff 1962)? Und was passiert, wenn keine existiert (Minsky & Papert 1969,
XOR-Muster)?

**Einordnung in die Reihe:**

```
Perceptron (WURZEL)                              [DIESES STÜCK]
 └─ MLP + Backpropagation                        [gebaut]
      ├─ CNN                                     [gebaut]
      └─ RNN                                     [gebaut]
           └─ LSTM                               [gebaut]
                └─ Attention/Transformer         [gebaut]
```

**Ergebnis in Kürze:** Auf konstruktiv trennbaren Daten konvergiert das Perceptron immer, und
die Novikoff-Schranke $(R/\gamma)^2$ (streng $(R^2+1)/\gamma^2$, da der Bias mitlernt) wird nie überschritten (0 Verletzungen über 50 Seeds) –
**aber sie ist in der Praxis extrem grob**: die reale Update-Zahl bleibt selbst bei winziger
Marge weit darunter (unter 13 % Ausnutzung über 20 Seeds). Auf dem XOR-Muster konvergiert das
Perceptron dagegen **nie** – die Fehlerzahl bleibt über 2000 Epochen durchgehend positiv, die
Gewichte oszillieren statt sich zu stabilisieren.

## Warum dieses Problem

Mindestens acht bestehende Demos in diesem Portfolio implementieren bereits ein eigenes MLP mit
Backpropagation von Hand (alle sechs Stücke der Graph-Neural-Network-Linie, `autoencoder-demo`,
`autoencoder-anomalie-demo`, `deepsvdd-demo`, `mappo-demo`, `pretrained-net-demo`,
`regretnet-demo`) – aber immer nur als Werkzeug für eine andere Frage, nie als eigenes Thema.
Diese Linie liefert das fehlende Fundament: Was lernt ein neuronales Netz überhaupt, und wo
hört die einfachste Variante – eine einzelne Gerade – auf zu reichen?

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| Perceptron konvergiert auf trennbaren Daten in endlich vielen Updates, und die Novikoff-Schranke $(R/\gamma)^2$ hält immer | ✅ 0 Verletzungen über 50 Seeds (`test_claim_no_violations_of_novikoff_bound_across_50_seeds`) |
| Perceptron konvergiert auf dem XOR-Muster **nicht** innerhalb von 2000 Epochen | ✅ (`test_claim_preset_xor_never_converges_within_2000_epochs`) |
| Die eigene Update-Regel stimmt mit einer unabhängigen Referenzimplementierung überein | ✅ 100 % Übereinstimmung mit `sklearn.linear_model.Perceptron` über 5 Seeds |
| ⚠️ Plan-Korrektur: unbeschränktes Gauß-Rauschen im Szenario-Generator garantiert Trennbarkeit **nicht** | ⚠️ In der ersten Vormessung zerstörte ein Ausreißer bei einem Seed die Trennbarkeit selbst (kein Bug in der Lernregel). Fix: beschränktes Rauschen (Scheibe mit festem Radius) + Regel `gap > 2·spread` → Trennbarkeit ist seither konstruktiv garantiert, mit exakt bekannter Marge $\gamma = \text{gap}/2 - \text{spread}$ (kein SVM/LP-Fit nötig). |

## Befunde (gemessen, keine Behauptungen)

**Marge-Sweep** (`ev.margin_sweep()`, je 15 Seeds, $n=40$, $\eta=1{,}0$):

| gap | Marge γ | Novikoff-Schranke | max. reale Updates (über 15 Seeds) | Ausnutzung |
|---|---|---|---|---|
| 4,00 | 1,000 | 8,4 | 1 | 11,8 % |
| 3,00 | 0,500 | 23,3 | 2 | 8,6 % |
| 2,40 | 0,200 | 112,2 | 2 | 1,8 % |
| 2,10 | 0,050 | 1.555,6 | 4 | 0,3 % |
| 2,02 | 0,010 | 37.365,5 | 4 | 0,01 % |

Je kleiner die Marge, desto dramatischer wächst die theoretische Schranke – die reale
Update-Zahl wächst dabei kaum. Über 20 Seeds bei `gap=4,0` liegt die maximale Ausnutzung der
Schranke bei 12,9 %.

**Presets:**

| Preset | Modus | Konvergiert? | Epochen | Updates | γ | Schranke |
|---|---|---|---|---|---|---|
| Große Marge | trennbar | Ja | 2 | 1 | 1,000 | 8,6 |
| Knappe Marge | trennbar | Ja | 2 | 2 | 0,050 | 1.585,1 |
| XOR-Muster | xor | Nein | 2000 (Max. erreicht) | – | – | – |

Bemerkenswert: Auch bei winziger Marge (Preset "Knappe Marge") konvergiert das Perceptron in
der Praxis **genauso schnell** wie bei großer Marge (2 Epochen) – die Schranke ist ein gültiges
Worst-Case-Versprechen, kein typisches Verhalten.

**XOR (Seed 1, Preset "XOR-Muster"):** Fehlerzahl bleibt über alle 2000 Epochen $>0$ (minimal 18
in den letzten 500 Epochen), die Gewichtsänderung je Epoche bleibt im Bereich 0,1–6,7 – keine
Stabilisierung, echte Oszillation.

## Modell und Verfahren

- `pc_scenario.py` – Datengeneratoren: linear trennbar (zwei Wolken um $(\pm\text{gap}/2, 0)$,
  konstruktiv trennbar bei `gap > 2·spread`) und XOR-Muster (vier Quadranten-Wolken).
- `pc_perceptron.py` – die klassische Update-Regel, von Grund auf mit Zählern.
- `pc_evaluation.py` – Novikoff-Schranke, Konvergenzerkennung, Oszillations-Metrik, Marge-Sweep.
- `pc_visualization.py` – Plotly-Plots (Entscheidungsgrenze, Fehlerkurve, Marge-Sweep); die
  Grenzlinie wird robust an der sichtbaren Box geclippt (siehe Grenzen unten).
- `pc_presets.py` – Permalink-Sync und die drei Presets.

## Was die App zeigt

Datenmodus (trennbar/XOR) und Regler (Abstand der Zentren, Anzahl Punkte, Lernrate, max.
Epochen, Seed) in der Sidebar; ein Epochen-Schieberegler zeigt die Entscheidungsgrenze und die
Fehlerkurve Schritt für Schritt; ein "📐"-Abschnitt zeigt die Novikoff-Schranke, ihre Herleitung
und den Marge-Sweep als Diagramm.

## Was nicht funktioniert hat / Grenzen

**Echter Bug gefunden und behoben:** Die erste Version zeichnete die Entscheidungsgrenze über
`y = -(w₀x+b)/w₁` – bei fast-senkrechten Trennebenen (kleines $w_1$) explodierte dieser Wert
(bis zu mehreren Hundert), was Plotlys `scaleanchor`-Kopplung dazu brachte, **beide** Achsen
massiv aufzublähen (y-Achse zeigte ±600 statt der gesetzten ±2-Box). Behoben durch echtes
Linie-Rechteck-Clipping (`pc_visualization._clip_line_to_box`), das die Grenzlinie direkt mit
den vier Boxkanten schneidet statt sie über den ganzen x-Bereich zu extrapolieren – funktioniert
unabhängig von der Steigung, kein Divisions-Blowup mehr (Regressionstests in
`tests/test_visualization.py`).

**Grenzen:** Nur zwei synthetische Vehikel (2D-Punkte), keine echten Datensätze. Die
Novikoff-Schranke wird hier nur über die konstruktive Trennebene $w=(1,0)$ ausgewertet, nicht
über die tatsächliche Maximalmargen-Trennebene (die wäre bei den meisten Seeds noch etwas
großzügiger) – das ändert nichts an der Kernaussage (Schranke gilt immer, ist aber grob), macht
den zitierten Zahlenwert aber zu einer gültigen, nicht unbedingt der bestmöglichen Schranke.

## Tests

45 Tests, `python -m pytest tests/ -v`:
- `test_scenario.py` – Reproduzierbarkeit, konstruktive Trennbarkeitsgarantie über 50 Seeds, XOR-Struktur.
- `test_perceptron.py` – Update-Regel, Bias, Randkonvention, Abgleich gegen `sklearn.linear_model.Perceptron`.
- `test_evaluation.py` – Novikoff-Schranke, Oszillationsmetrik, Marge-Sweep.
- `test_presets.py` – Presets sind gültig und liefern die behaupteten Effekte.
- `test_visualization.py` – Regressionstest für den Achsen-Blowup-Bug.
- `test_claims.py` – jede Zahl oben, direkt aus den echten Auswertungsfunktionen nachgerechnet.
- `test_app.py` – Streamlit `AppTest`: Presets, Modus-Wechsel, Regler-Extremwerte, Footer.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `pc_constants.py` | Regler-Grenzen, Presets |
| `pc_scenario.py` | Datengeneratoren |
| `pc_perceptron.py` | Kernalgorithmus |
| `pc_evaluation.py` | Kennzahlen, Sweeps |
| `pc_visualization.py` | Plotly-Plots |
| `pc_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Kein Pocket-Algorithmus (bestmögliche Trennebene bei nicht-trennbaren Daten) – das würde die
klare "konvergiert/oszilliert"-Botschaft dieses Wurzelstücks verwässern und gehört eher zu
einer möglichen Erweiterung. Keine echte SVM/LP-Maximalmargen-Berechnung (siehe Grenzen oben) –
bewusst durch die konstruktive, exakte Marge ersetzt, um das Stück robust und ohne fragile
Solver-Abhängigkeit zu halten.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and
  Organization in the Brain.* Psychological Review, 65(6), 386–408.
- Novikoff, A. B. J. (1962). *On Convergence Proofs on Perceptrons.* Proceedings of the
  Symposium on Mathematical Theory of Automata, 12, 615–622.
- Minsky, M. & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry.*
  MIT Press.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Neuronale Netze: vom Perceptron zum Transformer](https://sebastianhanisch.net/konzepte-neuronale-netze.html).
