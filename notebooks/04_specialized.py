# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4 — Specialized models
# MAGIC
# MAGIC The previous modules estimate the conditional mean: the average invoice for a vehicle like this one. A shop also asks different questions.
# MAGIC
# MAGIC | Question | Model | Why it matches |
# MAGIC | --- | --- | --- |
# MAGIC | What is a typical invoice, ignoring a few data-entry disasters? | Huber regression | Down-weights huge residuals instead of letting them rotate the whole line |
# MAGIC | What should we reserve so we are under the invoice about 90% of the time? | Quantile regression at 0.90 | Pinball loss penalizes under-prediction more than over-prediction |
# MAGIC | The invoice is positive and right-skewed. What likelihood matches that? | Gamma or Tweedie | Variance grows with the mean, which is what the Module 1 residual fan showed |
# MAGIC | How many line items will the technician open? | Poisson | The outcome is a count |
# MAGIC | We only have a few thousand rows and the relationship is local. | k-NN | Predicts from similar past visits |
# MAGIC | The relationship is smooth and the row count is small. | SVR | Flexible, and expensive once the table grows |
# MAGIC
# MAGIC k-NN and SVR are here so you can time them and then leave them out of the production shortlist. At shop scale they score one visit by scanning the training set (k-NN) or by evaluating a large set of support vectors (SVR). The boosters from Module 3 answer in a handful of tree lookups.

# COMMAND ----------

# MAGIC %run ./_lib

# COMMAND ----------

import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import (
    GammaRegressor,
    HuberRegressor,
    LinearRegression,
    PoissonRegressor,
    TweedieRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

experiment = start_experiment()
visits = load_visits()
train, valid, test = time_split(visits)
x_train, y_train = features_target(train)
x_valid, y_valid = features_target(valid)
print(f"Experiment: {experiment}")


def median_latency_ms(model, frame, repeats=25):
    row = frame.iloc[[0]]
    model.predict(row)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(row)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Huber versus least squares, on ordinary visits and on catastrophes
# MAGIC
# MAGIC Catastrophe rows are real engine-sized invoices in this generator, not typos. Huber will look better on the quiet middle and can look worse when those jobs are included. Use Huber to detect breakage in the data feed. Use a quantile model when the tail is a job the shop still has to staff.

# COMMAND ----------

huber = Pipeline(
    [
        ("prep", linear_preprocessor()),
        ("model", HuberRegressor(epsilon=1.35, max_iter=500)),
    ]
)
huber_metrics, huber_pred = log_trained_model(
    huber,
    x_train,
    y_train,
    x_valid,
    y_valid,
    run_name="huber",
    module="4",
    candidate="huber",
    task="expected_cost",
    params={"epsilon": 1.35},
    eligible=True,
)

ols = Pipeline([("prep", linear_preprocessor()), ("model", LinearRegression())])
ols.fit(x_train, y_train)
ols_pred = ols.predict(x_valid)

catastrophe = valid["cost_is_catastrophe"].to_numpy().astype(bool)
slice_rows = []
for name, pred in (("ols", ols_pred), ("huber", huber_pred)):
    slice_rows.append({"model": name, "slice": "all_valid", **regression_metrics(y_valid, pred)})
    if (~catastrophe).any():
        slice_rows.append(
            {
                "model": name,
                "slice": "ordinary_only",
                **regression_metrics(y_valid.to_numpy()[~catastrophe], pred[~catastrophe]),
            }
        )
    if catastrophe.any():
        slice_rows.append(
            {
                "model": name,
                "slice": "catastrophe_only",
                **regression_metrics(y_valid.to_numpy()[catastrophe], pred[catastrophe]),
            }
        )
show(pd.DataFrame(slice_rows).round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quantile regression for asymmetric risk
# MAGIC
# MAGIC Pinball loss at level alpha charges `(1 - alpha)` for over-predicting and `alpha` for under-predicting. At 0.90, missing high costs nine times more than missing low. Coverage should land near 90% of validation invoices at or below the P90 prediction. P50 is the typical job. P10 to P90 is a central band you can show a service advisor. P90 alone is the reserve.

# COMMAND ----------

quantile_rows = []
for alpha, candidate in ((0.1, "quantile_p10"), (0.5, "quantile_p50"), (0.9, "quantile_p90")):
    model = Pipeline(
        [
            ("prep", tree_preprocessor()),
            (
                "model",
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=alpha,
                    max_depth=3,
                    n_estimators=160,
                    learning_rate=0.08,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    def extra(y_true, y_pred, level=alpha):
        return {
            f"pinball_{level:.1f}": pinball_loss(y_true, y_pred, level),
            f"coverage_{level:.1f}": quantile_coverage(y_true, y_pred),
        }

    scored, _ = log_trained_model(
        model,
        x_train,
        y_train,
        x_valid,
        y_valid,
        run_name=candidate,
        module="4",
        candidate=candidate,
        task="quantile",
        params={"alpha": alpha, "loss": "quantile", "n_estimators": 160, "max_depth": 3},
        eligible=False,
        extra_scores=extra,
    )
    quantile_rows.append({"candidate": candidate, **scored})

show(pd.DataFrame(quantile_rows).round(3))
print("Coverage near the requested alpha means that quantile is calibrated on the validation window.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Positive dollars, and a count
# MAGIC
# MAGIC Gamma and Tweedie keep predictions positive and let the variance grow with the predicted invoice. Tweedie with `power=1.5` sits between a Poisson count process and a Gamma severity process, which is the insurance version of "how many failures, and how bad." Every row here is already a visit, so you are modeling severity, and Gamma is the matching likelihood. Tweedie is still a fair contender on skewed dollars.
# MAGIC
# MAGIC Poisson is fit to `line_item_count`, a different target. Its MAE is in line items, not dollars, and Module 6 will not let it win the cost championship.

# COMMAND ----------

for name, estimator, params in (
    ("gamma", GammaRegressor(alpha=1.0, max_iter=1000), {"alpha": 1.0, "family": "gamma"}),
    (
        "tweedie",
        TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000),
        {"power": 1.5, "alpha": 1.0, "family": "tweedie"},
    ),
):
    model = Pipeline([("prep", linear_preprocessor()), ("model", estimator)])
    scored, _ = log_trained_model(
        model,
        x_train,
        y_train,
        x_valid,
        y_valid,
        run_name=name,
        module="4",
        candidate=name,
        task="expected_cost",
        params=params,
        eligible=True,
    )
    print(f"{name} validation MAE ${scored['mae']:,.0f}   R^2 {scored['r2']:.3f}")

count_model = Pipeline(
    [("prep", linear_preprocessor()), ("model", PoissonRegressor(alpha=1.0, max_iter=500))]
)
_, count_train = features_target(train, target=SECONDARY_TARGET)
_, count_valid = features_target(valid, target=SECONDARY_TARGET)
count_metrics, count_pred = log_trained_model(
    count_model,
    x_train,
    count_train,
    x_valid,
    count_valid,
    run_name="poisson_line_items",
    module="4",
    candidate="poisson_line_items",
    task="count",
    params={"alpha": 1.0, "target": SECONDARY_TARGET},
    eligible=False,
)
print(
    f"Poisson line-item MAE {count_metrics['mae']:.2f} items "
    f"on a validation mean of {count_valid.mean():.2f}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SVR and k-NN, timed
# MAGIC
# MAGIC SVR is fit on 2,000 training visits. An RBF SVR on the full training window is a quadratic memory cost and will burn the serverless quota before it teaches you anything new. k-NN is fit on the full window so the latency number is the production one: every score walks the history.

# COMMAND ----------

sample_index = x_train.sample(n=min(2000, len(x_train)), random_state=RANDOM_STATE).index
svr = Pipeline(
    [
        ("prep", linear_preprocessor()),
        ("model", SVR(kernel="rbf", C=30.0, epsilon=40.0, gamma="scale", cache_size=500)),
    ]
)
svr_start = time.perf_counter()
svr.fit(x_train.loc[sample_index], y_train.loc[sample_index])
svr_fit_seconds = time.perf_counter() - svr_start
svr_pred = svr.predict(x_valid)
svr_metrics = regression_metrics(y_valid, svr_pred)

knn = Pipeline(
    [
        ("prep", linear_preprocessor()),
        ("model", KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=2)),
    ]
)
knn.fit(x_train, y_train)
knn_metrics = regression_metrics(y_valid, knn.predict(x_valid))

timing = pd.DataFrame(
    [
        {
            "model": "svr_on_2000_rows",
            "valid_mae": svr_metrics["mae"],
            "fit_seconds": svr_fit_seconds,
            "one_row_ms": median_latency_ms(svr, x_valid),
        },
        {
            "model": "knn_k25",
            "valid_mae": knn_metrics["mae"],
            "fit_seconds": np.nan,
            "one_row_ms": median_latency_ms(knn, x_valid),
        },
    ]
)
show(timing.round(2))

for _, row in timing.iterrows():
    with mlflow.start_run(run_name=str(row["model"])):
        mlflow.log_metrics(
            {
                "mae": float(row["valid_mae"]),
                "one_row_ms": float(row["one_row_ms"]),
            }
        )
        mlflow.set_tag("module", "4")
        mlflow.set_tag("candidate", str(row["model"]))
        mlflow.set_tag("task", "latency_comparison")
        mlflow.set_tag("eligible", "false")

print("SVR and k-NN are logged as latency comparisons only. Module 6 will not select them.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which specialized model to keep
# MAGIC
# MAGIC - Use the Module 3 booster or the Tweedie/Gamma model, whichever has the lower validation MAE, as the expected-cost estimate.
# MAGIC - Publish the P90 quantile beside it when the advisor needs a reserve or a not-to-exceed figure. Check coverage before you trust the reserve.
# MAGIC - Keep Huber as a data-quality model. A sudden gap between Huber and the champion is a signal that extreme invoices arrived, which is either a feed failure or a real mix shift. Module 6 writes the mix-shift rule.
# MAGIC - Leave SVR and k-NN in the lab notes. Bring k-NN back only if the feature set stays tiny and the training history stays under a few thousand visits.
# MAGIC
# MAGIC Next: `05_evaluation`.
