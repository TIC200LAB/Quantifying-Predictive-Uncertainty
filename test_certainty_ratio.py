"""Tests derived from the numerical example reported in the manuscript."""

import numpy as np

from certainty_ratio import evaluate_probabilities


def paper_example():
    classes = ["A", "B", "C"]
    y_true = ["A", "A", "A", "B", "B", "C"]
    Q = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.0, 0.2],
            [0.6, 0.1, 0.3],
            [0.4, 0.3, 0.3],
            [0.1, 0.8, 0.1],
            [0.0, 0.9, 0.1],
        ]
    )
    return evaluate_probabilities(y_true, Q, classes=classes)


def test_paper_confusion_matrices():
    result = paper_example()
    expected_CM = np.array([[3, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    expected_CM_star = np.array(
        [[2.3, 0.2, 0.5], [0.5, 1.1, 0.4], [0.0, 0.9, 0.1]]
    )
    assert np.allclose(result.CM, expected_CM)
    assert np.allclose(result.CM_star, expected_CM_star)


def test_paper_decomposition():
    result = paper_example()
    expected_V = np.array([[2.3, 0.0, 0.0], [0.4, 0.8, 0.0], [0.0, 0.9, 0.0]])
    expected_U = np.array([[0.0, 0.2, 0.5], [0.1, 0.3, 0.4], [0.0, 0.0, 0.1]])
    assert np.allclose(result.V, expected_V)
    assert np.allclose(result.U, expected_U)
    assert np.allclose(result.CM_star, result.V + result.U)


def test_paper_scalar_measures():
    result = paper_example()
    assert np.isclose(result.accuracy, 4 / 6)
    assert np.isclose(result.accuracy_star, 3.5 / 6)
    assert np.isclose(result.lambda_v, 4.4 / 6)
    assert np.isclose(result.lambda_u, 1.6 / 6)
    assert np.isclose(result.lambda_v + result.lambda_u, 1.0)


def test_first_maximum_tie_rule():
    result = evaluate_probabilities(
        [0],
        np.array([[0.4, 0.3, 0.3]]),
    )
    assert result.predicted_index.tolist() == [0]
    assert np.allclose(result.Q_plus, [[0.4, 0.0, 0.0]])
