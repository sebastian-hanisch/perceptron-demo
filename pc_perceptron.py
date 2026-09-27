"""Die klassische Perceptron-Lernregel (Rosenblatt 1958), von Grund auf mit
Zaehlern - kein sklearn/Autodiff im Kernpfad (nur als externe Test-Referenz)."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitResult:
    converged: bool
    epochs: int
    updates: int
    w: np.ndarray
    b: float
    errors_per_epoch: list = field(default_factory=list)
    w_history: list = field(default_factory=list)  # ein (w, b)-Snapshot je Epoche


def fit(X: np.ndarray, y: np.ndarray, eta: float = 1.0, max_epochs: int = 2000) -> FitResult:
    """Ein Durchlauf ueber alle Punkte in fester Reihenfolge = eine Epoche.
    Bei Fehlklassifikation (Marge <= 0): w <- w + eta*y*x, b <- b + eta*y.
    Epoche ohne Fehler = Konvergenz."""
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    updates = 0
    errors_per_epoch: list[int] = []
    w_history: list[np.ndarray] = []
    for epoch in range(max_epochs):
        n_errors = 0
        for xi, yi in zip(X, y):
            margin = yi * (w @ xi + b)
            if margin <= 0:
                w = w + eta * yi * xi
                b = b + eta * yi
                updates += 1
                n_errors += 1
        errors_per_epoch.append(n_errors)
        w_history.append(np.append(w, b).copy())
        if n_errors == 0:
            return FitResult(True, epoch + 1, updates, w, b, errors_per_epoch, w_history)
    return FitResult(False, max_epochs, updates, w, b, errors_per_epoch, w_history)


def predict(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    """Vorzeichenkonvention: Marge genau 0 zaehlt als Klasse +1 (np.sign(0)=0
    waere sonst ein drittes 'Label' - hier bewusst auf +1 abgebildet)."""
    scores = X @ w + b
    return np.where(scores >= 0, 1.0, -1.0)
