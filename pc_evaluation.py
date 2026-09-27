"""Kennzahlen und Sweeps: Settings-Dataclass, analyse()-Einstiegspunkt,
Novikoff-Schranke, Oszillations-Metrik fuer XOR, Marge-Sweep."""
from dataclasses import dataclass

import numpy as np

import pc_constants as C
import pc_perceptron as perc
import pc_scenario as sc


@dataclass(frozen=True)
class Settings:
    mode: str
    n: int
    seed: int
    eta: float
    max_epochs: int
    gap: float


def _spread_for(mode: str) -> float:
    return C.SPREAD_TRENNBAR if mode == C.MODE_TRENNBAR else C.SPREAD_XOR


def analyse(settings: Settings) -> dict:
    spread = _spread_for(settings.mode)
    scenario = sc.build_scenario(settings.mode, settings.n, settings.seed, spread, settings.gap)
    result = perc.fit(scenario.X, scenario.y, eta=settings.eta, max_epochs=settings.max_epochs)
    out = {"scenario": scenario, "result": result, "spread": spread}
    if settings.mode == C.MODE_TRENNBAR:
        gamma = sc.exact_margin_trennbar(spread, settings.gap)
        R = float(np.max(np.linalg.norm(scenario.X, axis=1)))
        bound = (R / gamma) ** 2
        out.update(gamma=gamma, R=R, novikoff_bound=bound)
    return out


def oscillation_metric(result: perc.FitResult, last_k: int = 200) -> dict:
    """Fuer den XOR-Modus: zeigt, dass die Gewichte NICHT konvergieren, sondern
    dauerhaft von 0 weg beschraenkte Aenderungen erfahren (Oszillation, nicht
    nur Stillstand kurz vor Ende)."""
    w_hist = np.array(result.w_history)
    tail = w_hist[-last_k:] if len(w_hist) > last_k else w_hist
    deltas = np.linalg.norm(np.diff(tail, axis=0), axis=1)
    errs = result.errors_per_epoch
    err_tail = errs[-last_k:] if len(errs) > last_k else errs
    return {
        "delta_min": float(deltas.min()) if len(deltas) else 0.0,
        "delta_max": float(deltas.max()) if len(deltas) else 0.0,
        "delta_mean": float(deltas.mean()) if len(deltas) else 0.0,
        "min_errors_in_tail": int(min(err_tail)) if err_tail else 0,
        "never_zero_errors": all(e > 0 for e in errs),
    }


def margin_sweep(
    gaps=C.MARGIN_SWEEP_GAPS,
    seeds=C.MARGIN_SWEEP_SEEDS,
    n=C.N_DEFAULT,
    eta=C.ETA_DEFAULT,
    max_epochs=C.MARGIN_SWEEP_MAX_EPOCHS,
) -> list:
    """Wie locker die Novikoff-Schranke bei verschiedenen Margen tatsaechlich
    ist: fuer jede gap-Stufe ueber mehrere Seeds das Verhaeltnis reale
    Updates/Schranke (Ausnutzung)."""
    rows = []
    for gap in gaps:
        gamma = sc.exact_margin_trennbar(C.SPREAD_TRENNBAR, gap)
        updates_list = []
        bound = None
        violations = 0
        for seed in seeds:
            settings = Settings(C.MODE_TRENNBAR, n, seed, eta, max_epochs, gap)
            out = analyse(settings)
            updates_list.append(out["result"].updates)
            bound = out["novikoff_bound"]
            if not out["result"].converged or out["result"].updates > bound:
                violations += 1
        max_updates = max(updates_list)
        rows.append({
            "gap": gap,
            "gamma": gamma,
            "bound": bound,
            "max_updates": max_updates,
            "max_utilisation": max_updates / bound if bound else 0.0,
            "violations": violations,
            "n_seeds": len(list(seeds)),
        })
    return rows
