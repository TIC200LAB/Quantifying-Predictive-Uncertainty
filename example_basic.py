#!/usr/bin/env python3
"""Minimal example using the six-instance, three-class example in the paper."""

import numpy as np

from certainty_ratio import evaluate_probabilities


def main() -> None:
    """Compute and display all framework objects from true labels and Q."""
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
        ],
        dtype=float,
    )

    result = evaluate_probabilities(y_true, Q, classes=classes)

    np.set_printoptions(precision=3, suppress=True)
    for name, matrix in result.matrix_summary().items():
        print(f"\n{name}:\n{matrix}")

    print("\nScalar measures:")
    print(f"Acc       = {result.accuracy:.6f}")
    print(f"Acc_star  = {result.accuracy_star:.6f}")
    print(f"lambda_v  = {result.lambda_v:.6f}")
    print(f"lambda_u  = {result.lambda_u:.6f}")


if __name__ == "__main__":
    main()
