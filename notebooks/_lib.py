# Databricks notebook source
# MAGIC %md
# MAGIC # Shared library for the repair-cost labs
# MAGIC Notebooks load this with `%run ./_lib`. It generates the synthetic shop data,
# MAGIC locks the feature contract, and holds the metric and diagnostic functions
# MAGIC the modules call.

# COMMAND ----------

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_pinball_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import mlflow
except ImportError:  # local smoke tests can import this file without MLflow
    mlflow = None

RANDOM_STATE = 42

# Visit window. Train / valid / test are cut on these dates so later months
# play the role of "next quarter in the shop."
DATE_START = date(2024, 1, 1)
DATE_END = date(2026, 6, 30)
TRAIN_END = "2025-10-01"
VALID_END = "2026-01-01"

# dtc_count is the sum of the four family counts, so it is stored for the
# collinearity lab and left out of the model matrix. primary_dtc_family is the
# argmax of those counts and is used the same way.
NUMERIC_FEATURES = [
    "mileage",
    "vehicle_age_years",
    "dtc_severity_max",
    "dtc_powertrain",
    "dtc_body",
    "dtc_chassis",
    "dtc_network",
    "has_misfire",
    "has_catalyst",
    "has_wheel_speed",
    "prior_repair_count_12m",
    "prior_repair_cost_12m",
    "days_since_last_repair",
    "repeat_component_90d",
    "warranty_active",
    "is_first_visit",
]

CATEGORICAL_FEATURES = ["make", "powertrain", "region"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "repair_cost_usd"
SECONDARY_TARGET = "line_item_count"
DIAGNOSTIC_COLUMNS = ["dtc_count", "primary_dtc_family"]

# Known at scoring time only if a scanner and the history file are in hand.
# Columns prefixed leak_ use the invoice or a future visit. They stay in the
# table so Module 5 can show how leakage inflates accuracy.
LEAK_COLUMNS = [
    "leak_parts_cost",
    "leak_teardown_quote",
    "leak_prior_cost_including_current",
    "leak_next_visit_cost",
]

MAKE_LEVELS = ["Toyota", "Honda", "Ford", "Chevy", "BMW"]
POWERTRAIN_LEVELS = ["ice", "hybrid", "ev"]
REGION_LEVELS = ["midwest", "south", "northeast", "west"]
FAMILY_LEVELS = ["none", "powertrain", "body", "chassis", "network"]

# drop="first" makes Toyota, ice, midwest, and none the reference levels.
# A linear coefficient is then the contrast against that baseline.
LINEAR_CATEGORIES = [MAKE_LEVELS, POWERTRAIN_LEVELS, REGION_LEVELS]


def feature_contract() -> pd.DataFrame:
    rows = (
        [("visit_id", "id"), ("vehicle_id", "id"), ("visit_date", "id")]
        + [(name, "numeric") for name in NUMERIC_FEATURES]
        + [(name, "categorical") for name in CATEGORICAL_FEATURES]
        + [(name, "diagnostic") for name in DIAGNOSTIC_COLUMNS]
        + [(TARGET, "target"), (SECONDARY_TARGET, "secondary_target")]
        + [("cost_is_catastrophe", "eval_only")]
        + [(name, "leak") for name in LEAK_COLUMNS]
    )
    return pd.DataFrame(rows, columns=["name", "role"])


def time_split(pdf: pd.DataFrame):
    """Split on visit date. The test window is not for tuning."""
    dates = pdf["visit_date"].astype(str)
    train = pdf[dates < TRAIN_END].copy()
    valid = pdf[(dates >= TRAIN_END) & (dates < VALID_END)].copy()
    test = pdf[dates >= VALID_END].copy()
    return train, valid, test


def features_target(pdf: pd.DataFrame, target: str = TARGET):
    missing = [column for column in FEATURES if column not in pdf.columns]
    if missing:
        raise KeyError(f"Visit table is missing features: {missing}")
    return pdf[FEATURES].copy(), pdf[target].astype(float).copy()


def linear_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=LINEAR_CATEGORIES,
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def tree_preprocessor() -> ColumnTransformer:
    """Trees do not need scaled inputs. One-hot keeps every model on one contract."""
    numeric = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=LINEAR_CATEGORIES,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "medae": float(np.median(np.abs(residual))),
        "r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "within_150": float(np.mean(np.abs(residual) <= 150.0)),
    }


def linear_preprocessor_with(extra_numeric: list[str]) -> ColumnTransformer:
    """Same linear pipeline, plus columns that are not part of the honest contract."""
    numeric_columns = NUMERIC_FEATURES + list(extra_numeric)
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=LINEAR_CATEGORIES,
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_columns),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def linear_effects_per_unit(pipeline: Pipeline) -> pd.DataFrame:
    """Convert standardized linear coefficients back to original units.

    Numeric inputs are scaled, so `coef_` is the dollar change for a one-standard-deviation
    move. Dividing by the training standard deviation returns dollars per original unit.
    One-hot coefficients are already dollars against the dropped baseline.
    """
    preprocessor = pipeline.named_steps["prep"]
    estimator = pipeline.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())
    coefficients = pd.Series(estimator.coef_, index=names)
    numeric_names = [name for name in names if name.startswith("num__")]
    scaler = preprocessor.named_transformers_["num"].named_steps["scale"]
    scales = pd.Series(scaler.scale_, index=numeric_names)
    numeric = pd.DataFrame(
        {
            "feature": [name.removeprefix("num__") for name in numeric_names],
            "coef_per_std": coefficients[numeric_names].to_numpy(),
            "std_in_train": scales.to_numpy(),
            "coef_per_unit": (coefficients[numeric_names] / scales).to_numpy(),
        }
    )
    categorical = coefficients.drop(index=numeric_names)
    categorical_rows = pd.DataFrame(
        {
            "feature": [name.removeprefix("cat__") for name in categorical.index],
            "coef_per_std": np.nan,
            "std_in_train": np.nan,
            "coef_per_unit": categorical.to_numpy(),
        }
    )
    return pd.concat([numeric, categorical_rows], ignore_index=True)


def design_condition_number(pipeline: Pipeline, frame: pd.DataFrame) -> float:
    matrix = np.asarray(pipeline.named_steps["prep"].transform(frame), dtype=float)
    design = np.column_stack([np.ones(len(frame)), matrix])
    return float(np.linalg.cond(design.T @ design))


def fit_ols_normal_equation(design: np.ndarray, y) -> np.ndarray:
    """Solve the normal equations (X'X) beta = X'y.

    `design` already contains a leading column of ones, so beta[0] is the intercept.
    `lstsq` is the fallback when collinear columns make X'X singular.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(y, dtype=float)
    gram = design.T @ design
    moment = design.T @ target
    try:
        return np.linalg.solve(gram, moment)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        return beta


def design_matrix(preprocessor, frame: pd.DataFrame, fit: bool) -> np.ndarray:
    matrix = preprocessor.fit_transform(frame) if fit else preprocessor.transform(frame)
    ones = np.ones((matrix.shape[0], 1))
    return np.column_stack([ones, np.asarray(matrix, dtype=float)])


def vif_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor of each column regressed on the others.

    VIF = 1 / (1 - R^2). A value above 10 means the column is nearly a
    recombination of the others, so its coefficient is unstable.
    """
    columns = list(frame.columns)
    matrix = frame.astype(float).to_numpy()
    rows = []
    for index, name in enumerate(columns):
        y = matrix[:, index]
        others = np.delete(matrix, index, axis=1)
        design = np.column_stack([np.ones(len(frame)), others])
        beta = fit_ols_normal_equation(design, y)
        fitted = design @ beta
        sse = float(np.sum((y - fitted) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else 1.0
        vif = float("inf") if r2 >= 1.0 - 1e-8 else float(1.0 / (1.0 - r2))
        rows.append({"feature": name, "r2_on_others": r2, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def pinball_loss(y_true, y_pred, alpha: float) -> float:
    return float(mean_pinball_loss(y_true, y_pred, alpha=alpha))


def quantile_coverage(y_true, y_pred) -> float:
    """Share of outcomes at or below the predicted quantile."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_true <= y_pred))


def population_stability_index(reference, current, bins: int = 10) -> float:
    """PSI on a numeric feature. Values under 0.10 are calm, over 0.25 are a shift."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    cuts = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts = cuts.astype(float).copy()
    cuts[0] = min(cuts[0], float(np.min(current))) - 1e-6
    cuts[-1] = max(cuts[-1], float(np.max(current))) + 1e-6
    ref_pct = np.histogram(reference, bins=cuts)[0].astype(float)
    cur_pct = np.histogram(current, bins=cuts)[0].astype(float)
    ref_pct = np.clip(ref_pct / ref_pct.sum(), 1e-4, None)
    cur_pct = np.clip(cur_pct / cur_pct.sum(), 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference, current) -> float:
    reference = pd.Series(reference).astype(str)
    current = pd.Series(current).astype(str)
    levels = sorted(set(reference) | set(current))
    ref_pct = reference.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    cur_pct = current.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    ref_pct = np.clip(ref_pct, 1e-4, None)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def retrain_decision(
    psi_values: dict,
    recent_mae: float,
    baseline_mae: float,
    new_labels: int,
    days_since_train: int,
    psi_limit: float = 0.25,
    mae_ratio_limit: float = 1.25,
    min_new_labels: int = 500,
    cooldown_days: int = 14,
):
    """Return (should_retrain, reasons).

    A feature shift (PSI) retrains immediately. An accuracy drop waits until
    enough new invoices exist and the cooldown has elapsed, so one noisy week
    does not burn the Free Edition compute quota.
    """
    hard = [
        f"{name} PSI is {value:.2f}, at or above {psi_limit:.2f}"
        for name, value in psi_values.items()
        if value >= psi_limit
    ]
    soft = []
    enough_labels = new_labels >= min_new_labels
    accuracy_dropped = recent_mae > mae_ratio_limit * baseline_mae
    if enough_labels and accuracy_dropped:
        soft.append(
            f"recent MAE {recent_mae:.0f} is above {mae_ratio_limit:.2f}× "
            f"the validation MAE {baseline_mae:.0f}"
        )
    if days_since_train < cooldown_days:
        return (len(hard) > 0, hard)
    return (len(hard) + len(soft) > 0, hard + soft)


def generate_visits(n_vehicles: int = 8000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Synthetic repair visits with a known, point-in-time feature story.

    The target is the shop invoice (parts plus labor), not what the customer
    pays out of pocket. Warranty changes who pays. It does not change the cost
    of the repair, and the generator gives warranty no effect on the invoice.
    """
    rng = np.random.default_rng(seed)
    span_days = (DATE_END - DATE_START).days
    make_effect = {"Toyota": 0.0, "Honda": 0.03, "Ford": 0.08, "Chevy": 0.10, "BMW": 0.48}
    powertrain_effect = {"ice": 0.0, "hybrid": 0.10, "ev": 0.22}
    region_effect = {"midwest": 0.0, "south": 0.02, "northeast": 0.07, "west": 0.09}
    make_p = [0.28, 0.22, 0.22, 0.18, 0.10]
    powertrain_p = [0.72, 0.18, 0.10]
    rows = []
    visit_serial = 0

    for vehicle_serial in range(n_vehicles):
        make = str(rng.choice(MAKE_LEVELS, p=make_p))
        powertrain = str(rng.choice(POWERTRAIN_LEVELS, p=powertrain_p))
        region = str(rng.choice(REGION_LEVELS))
        annual_miles = float(np.clip(rng.normal(13000, 3000), 4000, 28000))
        age_at_start = float(rng.uniform(0.4, 14.0))
        fragility = float(rng.gamma(1.4, 0.35))
        n_visits = int(np.clip(rng.poisson(2.2), 1, 5))
        offsets = np.sort(rng.choice(span_days, size=n_visits, replace=False))
        visit_dates = [DATE_START + timedelta(days=int(offset)) for offset in offsets]

        mileage = float(np.clip(rng.normal(annual_miles * age_at_start, 2500), 400, 280000))
        history = []
        previous_mileage = None

        for visit_date in visit_dates:
            if previous_mileage is None:
                age = age_at_start + (visit_date - DATE_START).days / 365.25
            else:
                delta_years = (visit_date - history[-1][0]).days / 365.25
                age = history[-1][3] + delta_years
                mileage = previous_mileage + annual_miles * delta_years + float(rng.normal(0, 250))
                mileage = max(mileage, previous_mileage)
            mileage = float(np.clip(mileage, 200, 360000))

            base_count = int(rng.poisson(0.30 + 0.05 * age + 0.40 * fragility))
            has_catalyst = int(
                powertrain == "ice"
                and rng.random() < min(0.40, 0.025 + 0.02 * (age > 8) + 0.02 * fragility)
            )
            has_misfire = int(
                powertrain != "ev"
                and rng.random() < min(0.45, 0.035 + 0.004 * age + 0.03 * fragility)
            )
            has_wheel = int(rng.random() < 0.04 + 0.012 * (age > 10))

            families = {name: 0 for name in ("powertrain", "body", "chassis", "network")}
            for _ in range(base_count):
                families[str(rng.choice(["powertrain", "body", "chassis", "network"]))] += 1
            if has_catalyst or has_misfire:
                families["powertrain"] += 1
            if has_wheel:
                families["chassis"] += 1
            dtc_count = int(sum(families.values()))
            if dtc_count == 0:
                severity = 0
                primary = "none"
            else:
                severity = int(min(3, max(1, rng.poisson(0.55 + 0.07 * age))))
                if has_catalyst:
                    severity = max(severity, 2)
                primary = max(
                    ("powertrain", "chassis", "body", "network"),
                    key=lambda name: (families[name], name == "powertrain"),
                )

            if not history:
                prior_count = 0
                prior_cost = 0.0
                days_since = -1.0
                repeat = 0
                is_first = 1
            else:
                cutoff = visit_date - timedelta(days=365)
                recent = [item for item in history if item[0] >= cutoff]
                prior_count = len(recent)
                prior_cost = float(sum(item[1] for item in recent))
                days_since = float((visit_date - history[-1][0]).days)
                repeat = int(
                    any(item[2] == primary and (visit_date - item[0]).days <= 90 for item in history)
                )
                is_first = 0

            warranty = int(age < 3.0 and mileage < 36000)
            log_mean = (
                5.50
                + 0.038 * age
                + 0.12 * max(mileage - 100000.0, 0.0) / 20000.0
                + make_effect[make]
                + powertrain_effect[powertrain]
                + region_effect[region]
                + 0.20 * severity
                + 1.05 * has_catalyst * (1.0 + 0.35 * (age > 8))
                + 0.42 * has_misfire * (1.0 + 0.25 * (mileage > 120000))
                + 0.30 * has_wheel
                + 0.18 * repeat
                + 0.08 * np.log1p(prior_cost) / np.log(50)
                + 0.22 * fragility
            )
            sigma = 0.25 + 0.03 * severity
            cost = float(np.exp(log_mean + sigma * float(rng.normal())))
            catastrophe = int(rng.random() < 0.025)
            if catastrophe:
                cost += float(rng.uniform(1800, 4200))
            cost = round(cost, 2)
            line_items = int(
                np.clip(rng.poisson(1.1 + 0.35 * dtc_count + has_catalyst + 0.5 * has_misfire), 1, 15)
            )

            visit_serial += 1
            rows.append(
                {
                    "visit_id": f"R{visit_serial:07d}",
                    "vehicle_id": f"V{vehicle_serial:05d}",
                    "visit_date": visit_date.isoformat(),
                    "mileage": round(mileage, 1),
                    "vehicle_age_years": round(age, 2),
                    "dtc_count": dtc_count,
                    "dtc_severity_max": severity,
                    "dtc_powertrain": families["powertrain"],
                    "dtc_body": families["body"],
                    "dtc_chassis": families["chassis"],
                    "dtc_network": families["network"],
                    "has_misfire": has_misfire,
                    "has_catalyst": has_catalyst,
                    "has_wheel_speed": has_wheel,
                    "prior_repair_count_12m": prior_count,
                    "prior_repair_cost_12m": round(prior_cost, 2),
                    "days_since_last_repair": days_since,
                    "repeat_component_90d": repeat,
                    "warranty_active": warranty,
                    "is_first_visit": is_first,
                    "make": make,
                    "powertrain": powertrain,
                    "region": region,
                    "primary_dtc_family": primary,
                    "repair_cost_usd": cost,
                    "line_item_count": line_items,
                    "cost_is_catastrophe": catastrophe,
                    "leak_parts_cost": round(cost * float(rng.uniform(0.48, 0.70)), 2),
                    "leak_teardown_quote": round(cost * float(rng.uniform(0.92, 1.08)), 2),
                    "leak_prior_cost_including_current": round(prior_cost + cost, 2),
                }
            )
            history.append((visit_date, cost, primary, age))
            previous_mileage = mileage

    frame = pd.DataFrame(rows)
    frame["leak_next_visit_cost"] = (
        frame.sort_values(["vehicle_id", "visit_date"])
        .groupby("vehicle_id")["repair_cost_usd"]
        .shift(-1)
        .sort_index()
    )
    _assert_generator(frame)
    return frame


def _assert_generator(frame: pd.DataFrame) -> None:
    required = set(feature_contract()["name"])
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"Generator dropped columns: {sorted(missing)}")
    first = frame[frame["is_first_visit"] == 1]
    if not np.allclose(first["prior_repair_cost_12m"], 0.0):
        raise AssertionError("First visits must have zero prior cost")
    if frame.loc[frame["powertrain"] == "ev", "has_catalyst"].sum() != 0:
        raise AssertionError("EV visits must not carry a catalyst code")
    ordered = frame.sort_values(["vehicle_id", "visit_date"])
    mileage_drop = ordered.groupby("vehicle_id")["mileage"].diff().fillna(0)
    if (mileage_drop < -1e-6).any():
        raise AssertionError("Mileage must not fall across a vehicle's visits")
    correlation = frame["leak_parts_cost"].corr(frame["repair_cost_usd"])
    if correlation < 0.9:
        raise AssertionError(f"Parts-cost leak should track the invoice, corr={correlation:.3f}")
    if frame["repair_cost_usd"].min() <= 0:
        raise AssertionError("Repair cost must stay positive")


def qualified_name(table: str) -> str:
    catalog = spark.sql("SELECT current_catalog()").first()[0]
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.repair_shop")
    return f"{catalog}.repair_shop.{table}"


def load_visits() -> pd.DataFrame:
    table = qualified_name("visits_raw")
    frame = spark.table(table).toPandas()
    frame["visit_date"] = frame["visit_date"].astype(str).str.slice(0, 10)
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{table} is missing {missing}. Run notebooks/00_setup_data first."
        )
    return frame


def start_experiment() -> str:
    if mlflow is None:
        raise ImportError("mlflow is not installed")
    user = spark.sql("SELECT current_user()").first()[0]
    experiment = f"/Users/{user}/repair-cost"
    mlflow.set_experiment(experiment)
    return experiment


def log_trained_model(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    run_name: str,
    module: str,
    candidate: str,
    task: str,
    params: dict,
    eligible: bool,
    extra_scores=None,
):
    """Fit on train, score the time-based validation window, log to MLflow.

    `metrics.mae` is always the validation MAE of this run's own target.
    Champion selection in Module 6 keeps only task='expected_cost'.
    `extra_scores(y_true, y_pred)` may add metrics such as pinball loss.
    """
    if mlflow is None:
        raise ImportError("mlflow is not installed")
    model.fit(x_train, y_train)
    prediction = model.predict(x_valid)
    metrics = regression_metrics(y_valid, prediction)
    if extra_scores is not None:
        metrics.update({key: float(value) for key, value in extra_scores(y_valid, prediction).items()})
    example = x_train.head(3)
    signature = mlflow.models.infer_signature(example, model.predict(example))
    clean_params = {}
    for key, value in params.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        clean_params[str(key)] = value
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(clean_params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("module", module)
        mlflow.set_tag("candidate", candidate)
        mlflow.set_tag("task", task)
        mlflow.set_tag("eligible", "true" if eligible else "false")
        mlflow.set_tag("split", "valid_from_2025-10")
        try:
            mlflow.sklearn.log_model(
                model, name="model", signature=signature, input_example=example
            )
        except TypeError:
            mlflow.sklearn.log_model(
                model, "model", signature=signature, input_example=example
            )
    return metrics, prediction


def show(obj) -> None:
    try:
        display(obj)  # noqa: F821 — provided by the Databricks notebook runtime
    except NameError:
        print(obj)
