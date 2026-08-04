#!/usr/bin/env python3
"""Reproduce the experiments reported in the attached manuscript.

The script evaluates Random Forest (RF) and multinomial logistic regression
(MLR), both without calibration and with nested isotonic calibration. All four
conditions use the same stratified outer folds. Instance-level out-of-fold
probabilities are passed to :func:`certainty_ratio.evaluate_probabilities`,
which computes T, P, Q, Q_plus, Q_minus, CM, CM_star, V, U, lambda_v,
lambda_u, Acc, and Acc_star.

Reported artefacts
------------------
* ``paper_tables.xlsx`` containing paper-style Tables 2, 3, and 4.
* Four detailed workbooks, one for each classifier/calibration condition.
* ``figure_5_rf_not_calibrated.png`` reproducing the RF relationship between
  lambda_v and Acc - MCP.
* ``spearman_correlations.csv`` for all four experimental conditions.
* ``experiment_manifest.json`` recording the run configuration and versions.

Dataset format
--------------
Place the 27 CSV datasets in ``alldata/``. Each file must contain numeric
features and one target column named ``class`` by default. The CSV filename
without extension is used as the dataset identifier.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr, t as student_t
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from certainty_ratio import ProbabilityMassEvaluation, evaluate_probabilities

try:
    from imcp import mcp_score
except ImportError:
    mcp_score = None

ClassifierName = Literal["RF", "MLR"]
Condition = Literal["NOT_CALIBR", "ISOT_CALIBR"]
CLASSIFIERS: tuple[ClassifierName, ...] = ("RF", "MLR")
CONDITIONS: tuple[Condition, ...] = ("NOT_CALIBR", "ISOT_CALIBR")
SCRIPT_VERSION = "paper-reproduction-1.0"


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration matching the experimental protocol."""

    data_dir: Path = Path("alldata")
    output_dir: Path = Path("results")
    class_column: str = "class"
    outer_folds: int = 10
    inner_folds: int = 3
    random_state: int = 0
    mlr_max_iter: int = 2000
    rf_n_estimators: int = 100
    rf_n_jobs: int = 1
    calibration_ensemble: bool = True
    historical_multinomial: bool = True
    require_27_datasets: bool = True
    fail_on_convergence_warning: bool = True


@dataclass
class DatasetResult:
    """Outputs for one dataset, classifier, and calibration condition."""

    dataset: str
    classifier: ClassifierName
    condition: Condition
    n_samples: int
    n_features: int
    n_classes: int
    evaluation: ProbabilityMassEvaluation
    mcp: float

    def metric_row(self) -> dict[str, Any]:
        """Return the dataset-level metrics used in the paper tables."""
        return {
            "Dataset": self.dataset,
            "Classifier": self.classifier,
            "Condition": self.condition,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "Acc": self.evaluation.accuracy,
            "lambda_v": self.evaluation.lambda_v,
            "lambda_u": self.evaluation.lambda_u,
            "Acc*": self.evaluation.accuracy_star,
            "MCP": self.mcp,
        }


# ---------------------------------------------------------------------------
# Data and estimator construction
# ---------------------------------------------------------------------------


def find_target_column(frame: pd.DataFrame, requested: str) -> str:
    """Return the unique target column matching ``requested`` case-insensitively."""
    matches = [column for column in frame.columns if str(column).casefold() == requested.casefold()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one target column matching {requested!r}; found {matches}."
        )
    return str(matches[0])


def load_dataset(
    path: Path,
    class_column: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load one CSV using the preprocessing assumptions in the experiment code."""
    frame = pd.read_csv(path)
    if not frame.columns.is_unique:
        raise ValueError(f"{path.stem}: duplicated column names are not supported.")

    target = find_target_column(frame, class_column)
    X = frame.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
    if X.shape[1] == 0:
        raise ValueError(f"{path.stem}: no feature columns were found.")
    if X.isna().any().any():
        columns = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"{path.stem}: missing or non-numeric feature values in {columns[:10]}."
        )
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"{path.stem}: feature matrix contains non-finite values.")

    raw_y = frame[target].to_numpy()
    if pd.isna(raw_y).any():
        raise ValueError(f"{path.stem}: target labels contain missing values.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_y).astype(int)
    if encoder.classes_.size < 2:
        raise ValueError(f"{path.stem}: at least two classes are required.")
    return X, y, encoder.classes_


def logistic_regression_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    """Return explicit MLR parameters and preserve historical multinomial mode."""
    signature = inspect.signature(LogisticRegression)
    kwargs: dict[str, Any] = {
        "solver": "lbfgs",
        "C": 1.0,
        "fit_intercept": True,
        "tol": 1e-4,
        "max_iter": config.mlr_max_iter,
        "random_state": config.random_state,
    }

    penalty_parameter = signature.parameters.get("penalty")
    if penalty_parameter is not None and penalty_parameter.default == "deprecated":
        kwargs["l1_ratio"] = 0.0
    else:
        kwargs["penalty"] = "l2"

    if config.historical_multinomial:
        if "multi_class" not in signature.parameters:
            raise RuntimeError(
                "This scikit-learn version no longer accepts "
                "multi_class='multinomial'. Use a compatible version or run "
                "with --use-current-logistic-default."
            )
        kwargs["multi_class"] = "multinomial"
    return kwargs


def make_base_estimator(
    classifier: ClassifierName,
    config: ExperimentConfig,
) -> Any:
    """Create the uncalibrated RF or MLR estimator used in the experiments."""
    if classifier == "RF":
        return RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            criterion="gini",
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=config.random_state,
            n_jobs=config.rf_n_jobs,
        )
    if classifier == "MLR":
        return Pipeline(
            steps=[
                ("standardise", StandardScaler()),
                (
                    "multinomial_logistic_regression",
                    LogisticRegression(**logistic_regression_kwargs(config)),
                ),
            ]
        )
    raise ValueError(f"Unsupported classifier: {classifier!r}.")


def make_calibrated_estimator(
    classifier: ClassifierName,
    config: ExperimentConfig,
) -> CalibratedClassifierCV:
    """Create nested isotonic calibration using only outer-training data."""
    inner_cv = StratifiedKFold(n_splits=config.inner_folds, shuffle=False)
    kwargs: dict[str, Any] = {
        "method": "isotonic",
        "cv": inner_cv,
        "ensemble": config.calibration_ensemble,
        "n_jobs": 1,
    }
    signature = inspect.signature(CalibratedClassifierCV)
    if "estimator" in signature.parameters:
        kwargs["estimator"] = make_base_estimator(classifier, config)
    else:
        kwargs["base_estimator"] = make_base_estimator(classifier, config)
    return CalibratedClassifierCV(**kwargs)


def fit_estimator(model: Any, X: pd.DataFrame, y: np.ndarray, config: ExperimentConfig) -> Any:
    """Fit a model while treating convergence warnings according to configuration."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*multi_class.*deprecated.*",
            category=FutureWarning,
        )
        if config.fail_on_convergence_warning:
            warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X, y)
    return model


def align_probability_columns(
    probabilities: np.ndarray,
    estimator_classes: Iterable[Any],
    n_classes: int,
) -> np.ndarray:
    """Align estimator probability columns to encoded order ``0, ..., k-1``."""
    probabilities = np.asarray(probabilities, dtype=float)
    estimator_classes = np.asarray(list(estimator_classes))
    positions: list[int] = []
    for label in range(n_classes):
        matches = np.flatnonzero(estimator_classes == label)
        if matches.size != 1:
            raise ValueError(f"Class {label} is missing or duplicated in model.classes_.")
        positions.append(int(matches[0]))

    aligned = probabilities[:, positions]
    if not np.allclose(aligned.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("Predicted probability rows do not sum to one.")
    return aligned


# ---------------------------------------------------------------------------
# Paired out-of-fold evaluation
# ---------------------------------------------------------------------------


def evaluate_dataset(path: Path, config: ExperimentConfig) -> list[DatasetResult]:
    """Evaluate all four model conditions on common outer folds."""
    X, y, classes = load_dataset(path, config.class_column)
    n_samples = y.size
    n_classes = classes.size
    counts = np.bincount(y, minlength=n_classes)
    if counts.min() < 2:
        raise ValueError(f"{path.stem}: every class needs at least two observations.")

    outer_cv = StratifiedKFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The least populated class in y has only .*members, which is less than n_splits=.*",
            category=UserWarning,
        )
        splits = list(outer_cv.split(X, y))

    probabilities: dict[tuple[ClassifierName, Condition], np.ndarray] = {
        (classifier, condition): np.zeros((n_samples, n_classes), dtype=float)
        for classifier in CLASSIFIERS
        for condition in CONDITIONS
    }
    assigned = np.zeros(n_samples, dtype=bool)

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        y_train = y[train_index]
        training_counts = np.bincount(y_train, minlength=n_classes)
        if training_counts.min() < config.inner_folds:
            raise ValueError(
                f"{path.stem}, fold {fold}: insufficient class count for "
                f"{config.inner_folds}-fold isotonic calibration."
            )

        for classifier in CLASSIFIERS:
            raw_model = fit_estimator(
                make_base_estimator(classifier, config),
                X.iloc[train_index],
                y_train,
                config,
            )
            calibrated_model = fit_estimator(
                make_calibrated_estimator(classifier, config),
                X.iloc[train_index],
                y_train,
                config,
            )

            raw_Q = align_probability_columns(
                raw_model.predict_proba(X.iloc[test_index]),
                raw_model.classes_,
                n_classes,
            )
            calibrated_Q = align_probability_columns(
                calibrated_model.predict_proba(X.iloc[test_index]),
                calibrated_model.classes_,
                n_classes,
            )
            probabilities[(classifier, "NOT_CALIBR")][test_index] = raw_Q
            probabilities[(classifier, "ISOT_CALIBR")][test_index] = calibrated_Q
        assigned[test_index] = True

    if not assigned.all():
        raise AssertionError("At least one instance lacks an out-of-fold prediction.")

    results: list[DatasetResult] = []
    for classifier in CLASSIFIERS:
        for condition in CONDITIONS:
            Q = probabilities[(classifier, condition)]
            evaluation = evaluate_probabilities(y, Q)
            if mcp_score is None:
                raise RuntimeError(
                    "The imcp package is required to reproduce the MCP results."
                )
            mcp = float(mcp_score(y, Q, list(range(n_classes))))
            results.append(
                DatasetResult(
                    dataset=path.stem,
                    classifier=classifier,
                    condition=condition,
                    n_samples=n_samples,
                    n_features=X.shape[1],
                    n_classes=n_classes,
                    evaluation=evaluation,
                    mcp=mcp,
                )
            )
    return results


# ---------------------------------------------------------------------------
# Tables, matrices, statistics, and Figure 5
# ---------------------------------------------------------------------------


def matrix_records(result: DatasetResult) -> list[dict[str, Any]]:
    """Convert the four aggregate framework matrices to long format.

    Only ``CM``, ``CM_star``, ``V``, and ``U`` are written to the global Excel
    workbooks.  The instance-level matrices ``T``, ``P``, ``Q``, ``Q_plus``,
    and ``Q_minus`` can contain millions of cells across all datasets and may
    exceed Excel's limit of 1,048,576 rows when converted to long format.
    They remain available through ``result.evaluation`` during execution.
    """
    records: list[dict[str, Any]] = []
    for matrix_name in ("CM", "CM_star", "V", "U"):
        matrix = getattr(result.evaluation, matrix_name)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                records.append(
                    {
                        "Dataset": result.dataset,
                        "Classifier": result.classifier,
                        "Condition": result.condition,
                        "Matrix": matrix_name,
                        "row": row_index,
                        "column": column_index,
                        "value": float(matrix[row_index, column_index]),
                    }
                )
    return records


def across_dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, sample SD, and two-sided Student-t 95% intervals."""
    rows: list[dict[str, Any]] = []
    for metric in ("Acc", "Acc*", "lambda_v", "lambda_u", "MCP"):
        values = frame[metric].to_numpy(dtype=float)
        n = values.size
        mean = float(values.mean())
        if n > 1:
            sd = float(values.std(ddof=1))
            half_width = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
            lower = mean - half_width
            upper = mean + half_width
        else:
            sd = float("nan")
            lower = float("nan")
            upper = float("nan")
        rows.append(
            {
                "metric": metric,
                "n_datasets": n,
                "mean": mean,
                "sd": sd,
                "ci95_lower": lower,
                "ci95_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def paper_table(frame: pd.DataFrame, classifier: ClassifierName, condition: Condition) -> pd.DataFrame:
    """Create a paper-style dataset table with a final mean row."""
    selected = frame[
        (frame["Classifier"] == classifier) & (frame["Condition"] == condition)
    ][["Dataset", "Acc", "lambda_v", "lambda_u", "Acc*", "MCP"]].copy()
    selected = selected.sort_values("Dataset", key=lambda values: values.str.casefold())
    mean_row = {"Dataset": "Mean"}
    mean_row.update({column: float(selected[column].mean()) for column in selected.columns[1:]})
    selected = pd.concat([selected, pd.DataFrame([mean_row])], ignore_index=True)
    selected.iloc[:, 1:] = selected.iloc[:, 1:].round(2)
    return selected


def table_four(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the four-row mean/SD/CI comparison reported as Table 4."""
    rows: list[dict[str, Any]] = []
    for classifier in CLASSIFIERS:
        for condition in CONDITIONS:
            subset = frame[
                (frame["Classifier"] == classifier) & (frame["Condition"] == condition)
            ]
            row: dict[str, Any] = {
                "Classifier": classifier,
                "Calibration": "None" if condition == "NOT_CALIBR" else "Isotonic",
            }
            for metric in ("Acc", "Acc*", "lambda_v", "MCP"):
                values = subset[metric].to_numpy(dtype=float)
                mean = float(values.mean())
                sd = float(values.std(ddof=1))
                half_width = float(
                    student_t.ppf(0.975, values.size - 1) * sd / np.sqrt(values.size)
                )
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
                row[f"{metric}_ci95_lower"] = mean - half_width
                row[f"{metric}_ci95_upper"] = mean + half_width
            rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the Spearman relationship between lambda_v and Acc - MCP."""
    rows: list[dict[str, Any]] = []
    for classifier in CLASSIFIERS:
        for condition in CONDITIONS:
            subset = frame[
                (frame["Classifier"] == classifier) & (frame["Condition"] == condition)
            ].copy()
            gap = subset["Acc"] - subset["MCP"]
            statistic = spearmanr(subset["lambda_v"], gap)
            rows.append(
                {
                    "Classifier": classifier,
                    "Condition": condition,
                    "n_datasets": len(subset),
                    "spearman_rho": float(statistic.statistic),
                    "p_value": float(statistic.pvalue),
                }
            )
    return pd.DataFrame(rows)


def plot_figure_five(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot lambda_v against Acc - MCP for uncalibrated Random Forest."""
    subset = frame[
        (frame["Classifier"] == "RF") & (frame["Condition"] == "NOT_CALIBR")
    ].copy()
    x = subset["lambda_v"].to_numpy(dtype=float)
    y = (subset["Acc"] - subset["MCP"]).to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.scatter(x, y, label="datasets")
    if x.size >= 2 and np.unique(x).size >= 2:
        coefficients = np.polyfit(x, y, deg=1)
        grid = np.linspace(float(x.min()), float(x.max()), 200)
        axis.plot(grid, np.polyval(coefficients, grid), label="Tendency")
    axis.set_xlabel(r"$\lambda_v$")
    axis.set_ylabel(r"$Acc - MCP$")
    axis.grid(True, linestyle=":", linewidth=0.7)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def package_version_or_unavailable(package: str) -> str:
    """Return an installed package version or an unavailable marker."""
    try:
        return package_version(package)
    except PackageNotFoundError:
        return "unavailable"


def write_outputs(
    results: list[DatasetResult],
    config: ExperimentConfig,
) -> None:
    """Write all tables, detailed matrices, correlations, figure, and manifest."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.DataFrame([result.metric_row() for result in results])
    metric_frame = metric_frame.sort_values(
        ["Dataset", "Classifier", "Condition"],
        key=lambda series: series.str.casefold() if series.dtype == object else series,
    ).reset_index(drop=True)

    matrices = pd.DataFrame(
        [record for result in results for record in matrix_records(result)]
    )

    with pd.ExcelWriter(config.output_dir / "paper_tables.xlsx", engine="openpyxl") as writer:
        paper_table(metric_frame, "RF", "NOT_CALIBR").to_excel(
            writer, sheet_name="Table 2 RF", index=False
        )
        paper_table(metric_frame, "MLR", "ISOT_CALIBR").to_excel(
            writer, sheet_name="Table 3 MLR isotonic", index=False
        )
        table_four(metric_frame).to_excel(writer, sheet_name="Table 4", index=False)
        metric_frame.to_excel(writer, sheet_name="All detailed results", index=False)

    for classifier in CLASSIFIERS:
        for condition in CONDITIONS:
            subset = metric_frame[
                (metric_frame["Classifier"] == classifier)
                & (metric_frame["Condition"] == condition)
            ].copy()
            condition_matrices = matrices[
                (matrices["Classifier"] == classifier)
                & (matrices["Condition"] == condition)
            ].copy()
            output_file = config.output_dir / f"paper_results_{classifier}_{condition}.xlsx"
            with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                subset.to_excel(writer, sheet_name="Detailed results", index=False)
                across_dataset_summary(subset).to_excel(
                    writer, sheet_name="Across-dataset summary", index=False
                )
                condition_matrices.to_excel(writer, sheet_name="Matrices", index=False)

    correlations = correlation_table(metric_frame)
    correlations.to_csv(config.output_dir / "spearman_correlations.csv", index=False)
    plot_figure_five(metric_frame, config.output_dir / "figure_5_rf_not_calibrated.png")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "imcp": package_version_or_unavailable("imcp"),
        "data_directory": str(config.data_dir.resolve()),
        "dataset_count": int(metric_frame["Dataset"].nunique()),
        "outer_folds": config.outer_folds,
        "inner_folds": config.inner_folds,
        "random_state": config.random_state,
        "rf_n_estimators": config.rf_n_estimators,
        "rf_n_jobs": config.rf_n_jobs,
        "mlr_max_iter": config.mlr_max_iter,
        "historical_multinomial": config.historical_multinomial,
        "calibration": "isotonic",
        "calibration_ensemble": config.calibration_ensemble,
        "tie_rule": "first maximum in probability-column order",
    }
    (config.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../alldata"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--class-column", default="class")
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--mlr-max-iter", type=int, default=2000)
    parser.add_argument("--rf-n-estimators", type=int, default=100)
    parser.add_argument("--rf-n-jobs", type=int, default=1)
    parser.add_argument(
        "--use-current-logistic-default",
        action="store_true",
        help="Omit historical multi_class='multinomial' when unsupported.",
    )
    parser.add_argument(
        "--allow-dataset-count-mismatch",
        action="store_true",
        help="Run even when the data directory does not contain exactly 27 CSV files.",
    )
    parser.add_argument(
        "--allow-convergence-warning",
        action="store_true",
        help="Do not convert MLR convergence warnings into errors.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Execute the complete paper-reproduction workflow."""
    args = parse_args(argv)
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        class_column=args.class_column,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        random_state=args.random_state,
        mlr_max_iter=args.mlr_max_iter,
        rf_n_estimators=args.rf_n_estimators,
        rf_n_jobs=args.rf_n_jobs,
        historical_multinomial=not args.use_current_logistic_default,
        require_27_datasets=not args.allow_dataset_count_mismatch,
        fail_on_convergence_warning=not args.allow_convergence_warning,
    )

    if mcp_score is None:
        raise RuntimeError("Install the imcp package before reproducing the paper.")
    if not config.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {config.data_dir.resolve()}")

    files = sorted(config.data_dir.glob("*.csv"), key=lambda path: path.name.casefold())
    if config.require_27_datasets and len(files) != 27:
        raise ValueError(
            f"Expected the 27 paper datasets, but found {len(files)} CSV files. "
            "Use --allow-dataset-count-mismatch for a partial run."
        )
    if not files:
        raise FileNotFoundError(f"No CSV files found in {config.data_dir.resolve()}")

    all_results: list[DatasetResult] = []
    for path in files:
        print(f"Evaluating {path.stem} ...", flush=True)
        all_results.extend(evaluate_dataset(path, config))

    write_outputs(all_results, config)
    print(f"Results written to {config.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
