import numpy as np
import pytest

import pc_perceptron as perc
import pc_scenario as sc

pytest.importorskip("sklearn")
from sklearn.linear_model import Perceptron as SkPerceptron


def test_single_update_moves_weights_in_correct_direction():
    X = np.array([[1.0, 0.0]])
    y = np.array([1.0])
    result = perc.fit(X, y, eta=1.0, max_epochs=1)
    # erster Punkt ist mit w=0 immer eine "Fehlklassifikation" (Marge=0<=0)
    np.testing.assert_array_almost_equal(result.w, [1.0, 0.0])
    assert result.b == 1.0
    assert result.updates == 1


def test_bias_update_sign_matches_label():
    X = np.array([[0.0, 0.0]])
    y = np.array([-1.0])
    result = perc.fit(X, y, eta=1.0, max_epochs=1)
    assert result.b == -1.0


def test_predict_boundary_convention_at_zero_margin():
    # score genau 0 -> laut Konvention Klasse +1
    w = np.array([1.0, 0.0])
    b = 0.0
    X = np.array([[0.0, 5.0]])  # score = 1*0 + 0*5 + 0 = 0
    assert perc.predict(w, b, X)[0] == 1.0


def test_converges_on_constructively_separable_data():
    X, y = sc.make_separable(40, seed=1, spread=1.0, gap=4.0)
    result = perc.fit(X, y, eta=1.0, max_epochs=2000)
    assert result.converged
    preds = perc.predict(result.w, result.b, X)
    assert np.all(preds == y)


def test_does_not_converge_on_xor_within_2000_epochs():
    X, y = sc.make_xor(40, seed=1, spread=0.6, gap=3.0)
    result = perc.fit(X, y, eta=1.0, max_epochs=2000)
    assert not result.converged
    assert all(e > 0 for e in result.errors_per_epoch)


@pytest.mark.parametrize("seed", [7, 8, 9, 10, 11])
def test_matches_sklearn_perceptron_prediction(seed):
    X, y = sc.make_separable(30, seed=seed, spread=1.0, gap=4.0)
    mine = perc.fit(X, y, eta=1.0, max_epochs=2000)
    sk = SkPerceptron(eta0=1.0, shuffle=False, max_iter=2000, tol=None, fit_intercept=True)
    sk.fit(X, y)
    pred_mine = perc.predict(mine.w, mine.b, X)
    pred_sk = sk.predict(X)
    assert np.mean(pred_mine == pred_sk) == 1.0
