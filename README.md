# Quantifying Predictive Uncertainty: A Probabilistic Framework for Classifier Evaluation

This repository provides a compact reference implementation of the probability-mass evaluation framework described in the manuscript.

The implementation starts from:

- the true class of each instance; and
- the class-probability matrix `Q` returned by any probabilistic classifier.

It computes:

- `T`: one-hot ground-truth matrix;
- `P`: one-hot matrix of tie-resolved argmax predictions;
- `Q`: validated class-probability matrix;
- `CM = T.T @ P`: standard confusion matrix;
- `CM_star = T.T @ Q`: probability-mass confusion matrix;
- `Q_plus`: selected top-1 probability mass;
- `Q_minus = Q - Q_plus`: residual probability mass;
- `V = T.T @ Q_plus`: decisive probability-mass matrix;
- `U = T.T @ Q_minus`: residual-dispersion matrix;
- `lambda_v = sum(V) / n = mean(max(Q, axis=1))`;
- `lambda_u = sum(U) / n = 1 - lambda_v`;
- `Acc = trace(CM) / n`;
- `Acc_star = trace(CM_star) / n`.

The tie convention is deterministic: the first maximum in probability-column order is selected.

## Repository contents

```text
certainty_ratio.py                Core framework in one independent module
reproduce_paper_experiments.py    Reproduction of the RF/MLR experiments
example_basic.py                  Minimal example derived from the manuscript
imcp.py                           Package for Multiclass Classification Performance (MCP) curve.
requirements.txt                  Runtime and test dependencies
pyproject.toml                    Editable-install and pytest configuration
/alldata                          Datasets used in the experiments
```

## Minimal use

```python
import numpy as np
from certainty_ratio import evaluate_probabilities

classes = ["A", "B", "C"]
y_true = ["A", "A", "A", "B", "B", "C"]
Q = np.array([
    [0.9, 0.1, 0.0],
    [0.8, 0.0, 0.2],
    [0.6, 0.1, 0.3],
    [0.4, 0.3, 0.3],
    [0.1, 0.8, 0.1],
    [0.0, 0.9, 0.1],
])

result = evaluate_probabilities(y_true, Q, classes=classes)

print(result.CM)
print(result.CM_star)
print(result.V)
print(result.U)
print(result.scalar_summary())
```

Expected scalar values for this example are:

```text
Acc       = 0.666667
Acc_star  = 0.583333
lambda_v  = 0.733333
lambda_u  = 0.266667
```

Run the complete example with:

```bash
python example_basic.py
```

## API

### `evaluate_probabilities(y_true, probabilities, classes=None)`

This is the recommended entry point. It returns an immutable `ProbabilityMassEvaluation` object containing all matrices and scalar measures.

When labels are strings or do not already equal `0, ..., k-1`, pass `classes` in the exact order used by the columns of `Q`.

### Lower-level functions

The module also exposes separate functions for validation, one-hot encoding, matrix construction, decomposition, weights, and both accuracy definitions. Every public function contains an English docstring.

## Verify the implementation

```bash
python -m pytest
```

The tests reproduce the numerical matrices and scalar values shown in the manuscript example, including:

```text
CM = [[3, 0, 0],
      [1, 1, 0],
      [0, 1, 0]]

CM_star = [[2.3, 0.2, 0.5],
           [0.5, 1.1, 0.4],
           [0.0, 0.9, 0.1]]
```
## Threshold-free decomposition of Q

The matrix `Q` is decomposed row by row into:

```text
Q = Q_plus + Q_minus
```

### Decisive top-1 component

`Q_plus` retains only the selected maximum probability in each row:

```text
Q_plus = [[0.9, 0.0, 0.0],
          [0.8, 0.0, 0.0],
          [0.6, 0.0, 0.0],
          [0.4, 0.0, 0.0],
          [0.0, 0.8, 0.0],
          [0.0, 0.9, 0.0]]
```

This matrix contains the probability mass that determines the hard predictions.

### Residual-dispersion component

The remaining probability mass is:

```text
Q_minus = Q - Q_plus
```

giving:

```text
Q_minus = [[0.0, 0.1, 0.0],
           [0.0, 0.0, 0.2],
           [0.0, 0.1, 0.3],
           [0.0, 0.3, 0.3],
           [0.1, 0.0, 0.1],
           [0.0, 0.0, 0.1]]
```

`Q_minus` describes how the probability mass not selected by the argmax decision is distributed across the remaining classes.

No confidence threshold is required.

---

## Decomposition of the probability-mass confusion matrix

Aggregating `Q_plus` and `Q_minus` by true class gives:

```text
V = T.T @ Q_plus
U = T.T @ Q_minus
```

### Decisive probability-mass matrix

```text
V = [[2.3, 0.0, 0.0],
     [0.4, 0.8, 0.0],
     [0.0, 0.9, 0.0]]
```

The matrix `V` preserves the true-class/output-class allocation of top-1 probability mass.

Its entries distinguish:

- decisive mass supporting correct decisions, represented by diagonal entries;
- decisive mass supporting errors, represented by off-diagonal entries.

For example:

- `V[A, A] = 2.3` supports correct top-1 decisions for class `A`;
- `V[B, A] = 0.4` is top-1 probability mass supporting an incorrect prediction of `A` for a true-`B` instance;
- `V[C, B] = 0.9` is top-1 probability mass supporting an incorrect prediction of `B` for the true-`C` instance.

### Residual-dispersion matrix

```text
U = [[0.0, 0.2, 0.5],
     [0.1, 0.3, 0.4],
     [0.0, 0.0, 0.1]]
```

The matrix `U` preserves the allocation of residual probability mass.

For example:

- `U[A, C] = 0.5` is residual probability mass assigned to class `C` by true-`A` instances;
- `U[B, C] = 0.4` is residual probability mass assigned to class `C` by true-`B` instances;
- `U[B, B] = 0.3` is residual mass retained by the true class for a true-`B` instance whose top-1 prediction was another class;
- `U[C, C] = 0.1` is residual mass assigned to the true class `C` when the top-1 prediction was `B`.

Diagonal entries of `U` need not be zero. They arise when the true class receives some probability mass but is not the selected top-1 class.

The decomposition is exact:

```text
CM_star = V + U
```

Numerically:

```text
CM_star = [[2.3, 0.2, 0.5],
           [0.5, 1.1, 0.4],
           [0.0, 0.9, 0.1]]

V       = [[2.3, 0.0, 0.0],
           [0.4, 0.8, 0.0],
           [0.0, 0.9, 0.0]]

U       = [[0.0, 0.2, 0.5],
           [0.1, 0.3, 0.4],
           [0.0, 0.0, 0.1]]
```

and:

```text
CM_star = V + U
```

This matrix-valued decomposition provides information that cannot be recovered from a scalar confidence or entropy summary alone. It identifies the true/output class pairs receiving decisive and residual probability mass.

---

## Global decisiveness and residual dispersion

The total decisive probability mass is:

```text
sum(V) = 4.4
```

The total residual probability mass is:

```text
sum(U) = 1.6
```

Since the dataset contains six instances:

```text
lambda_v = sum(V) / 6
         = 4.4 / 6
         = 0.733333

lambda_u = sum(U) / 6
         = 1.6 / 6
         = 0.266667
```

## Reproduce the paper experiments

```bash
python reproduce_paper_experiments.py \
    --data-dir alldata \
    --output-dir results
```

The script uses:

- stratified 10-fold outer cross-validation;
- common outer folds for every model condition;
- Random Forest with 100 trees and `random_state=0`;
- standardised multinomial logistic regression with `lbfgs` and `max_iter=2000`;
- nested 3-fold isotonic calibration fitted exclusively on each outer training fold;
- a deterministic first-maximum tie convention.

## Citation 

Quantifying Predictive Uncertainty: A Probabilistic Framework for Classifier Evaluation. 2026.
