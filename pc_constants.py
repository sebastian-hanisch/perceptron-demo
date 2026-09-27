"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

DEFAULT_SEED = 7

# Geteilte Regler (beide Modi)
N_MIN, N_MAX, N_DEFAULT = 10, 200, 40
ETA_MIN, ETA_MAX, ETA_DEFAULT = 0.1, 2.0, 1.0
MAX_EPOCHS_MIN, MAX_EPOCHS_MAX, MAX_EPOCHS_DEFAULT = 50, 20000, 2000

# Modus-spezifische Regler: Streuung ist FEST (kein Regler), damit die
# konstruktive Trennbarkeits-Garantie (gap > 2*spread) nie durch eine
# Reglerkombination verletzt werden kann (siehe pc_scenario.make_separable).
MODE_TRENNBAR = "trennbar"
MODE_XOR = "xor"
MODES = (MODE_TRENNBAR, MODE_XOR)
MODE_LABELS = {MODE_TRENNBAR: "Linear trennbar", MODE_XOR: "XOR-Muster"}

SPREAD_TRENNBAR = 1.0
SPREAD_XOR = 0.6

# gap muss > 2*SPREAD_TRENNBAR = 2.0 sein, damit Trennbarkeit garantiert ist
GAP_TRENNBAR_MIN, GAP_TRENNBAR_MAX, GAP_TRENNBAR_DEFAULT = 2.02, 8.0, 4.0
GAP_XOR_MIN, GAP_XOR_MAX, GAP_XOR_DEFAULT = 1.5, 5.0, 3.0

# Marge-Sweep fuer den 📐-Abschnitt (feste, kleine Stichprobe - billig genug
# fuer eager statt on-demand Berechnung)
MARGIN_SWEEP_GAPS = (4.0, 3.0, 2.4, 2.1, 2.02)
MARGIN_SWEEP_SEEDS = tuple(range(15))
MARGIN_SWEEP_MAX_EPOCHS = 20000

PRESETS = {
    "gross": dict(
        label="Große Marge",
        mode=MODE_TRENNBAR, n=40, gap=4.0, eta=1.0, max_epochs=2000, seed=7,
        help="Deutlich getrennte Wolken – das Perceptron braucht nur wenige Updates.",
    ),
    "knapp": dict(
        label="Knappe Marge",
        mode=MODE_TRENNBAR, n=40, gap=2.1, eta=1.0, max_epochs=20000, seed=7,
        help="Die Wolken berühren sich fast – die Marge γ ist winzig, die "
             "Novikoff-Schranke schnellt hoch (auch wenn die reale Update-Zahl "
             "moderat bleibt).",
    ),
    "xor": dict(
        label="XOR-Muster",
        mode=MODE_XOR, n=40, gap=3.0, eta=1.0, max_epochs=2000, seed=1,
        help="Vier Quadranten, diagonal gegenüberliegende teilen die Klasse – "
             "ein einzelnes Perceptron konvergiert hier NIE.",
    ),
}
