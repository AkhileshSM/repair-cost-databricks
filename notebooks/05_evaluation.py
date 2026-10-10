# Databricks notebook source
# MAGIC %md
# MAGIC # Module 5 — Model evaluation
# MAGIC
# MAGIC Every number in this notebook is computed on the **validation** window (2025-10-01 through 2025-12-31). The test window stays sealed until Module 6.
# MAGIC
# MAGIC **MAE, RMSE, and R² answer different questions.**
# MAGIC - **MAE** is the average dollar miss. It is the number to take to a service manager.
# MAGIC - **RMSE** squares the misses before averaging, so one engine-sized error outweighs several small ones. Prefer it when a large miss is disproportionately expensive.
# MAGIC - **R²** is the share of invoice variance you explain relative to predicting a constant mean. On a right-skewed invoice, a useful MAE can sit next to a modest R². Rank candidates by the business number (MAE, or pinball at 0.90 for a reserve), and use R² as context.
# MAGIC - **within_150** is the share of validation visits whose absolute miss is at most $150.
# MAGIC
# MAGIC **Cross-validation** estimates that same MAE when you do not yet trust a single cut. The fold has to mimic deployment. A future visit is not a random other row from the same week, and it is not a second visit of a vehicle whose later invoice is already in the training fold.
# MAGIC
# MAGIC **Leakage** is any input that uses information from after the prediction moment. That includes this visit's invoice, a teardown quote written from that invoice, a history total that already contains today's dollars, and a vehicle average computed with visits that have not happened yet.

# COMMAND ----------

# MAGIC %run ./_lib

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, TweedieRegressor
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

experiment = start_experiment()
visits = load_visits()
train, valid, test = time_split(visits)
x_train, y_train = features_target(train)
x_valid, y_valid = features_target(valid)
print(f"Experiment: {experiment}")
print(f"Validation visits: {len(valid):,}. Test visits left sealed: {len(test):,}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## One validation window, three loss summaries

# COMMAND ----------

candidates = {
    "ridge": Pipeline([("prep", linear_preprocessor()), ("model", Ridge(alpha=10.0))]),
    "tweedie": Pipeline(
        [("prep", linear_preprocessor()), ("model", TweedieRegressor(power=1.5, alpha=1.0, max_iter=1000))]
    ),
    "hist_gbm": Pipeline(
        [
            ("prep", tree_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.06,
                    max_iter=250,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}

metric_rows = []
predictions = {}
for name, model in candidates.items():
    scored, prediction = log_trained_model(
        model,
        x_train,
        y_train,
        x_valid,
        y_valid,
        run_name=f"eval_{name}",
        module="5",
        candidate=name,
        task="expected_cost",
        params={"role": "evaluation_refit"},
        eligible=True,
    )
    metric_rows.append({"model": name, **scored})
    predictions[name] = prediction

comparison = pd.DataFrame(metric_rows).set_index("model")
show(comparison.round(3))
print(
    "Read MAE as average dollars missed, RMSE as the penalty that emphasizes large misses, "
    "and within_150 as the share of quotes inside a $150 band."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross-validation that matches a real next visit
# MAGIC
# MAGIC All three procedures score a Ridge model on the training window only.
# MAGIC
# MAGIC - **KFold** assigns visits at random. Two visits from the same vehicle can fall on opposite sides of the fold.
# MAGIC - **GroupKFold** keeps every visit of a vehicle inside one fold. Use it when the question is performance on a vehicle the model has never seen.
# MAGIC - **TimeSeriesSplit** trains on earlier dates and scores later dates inside the training window. Use it when the question is performance on next quarter, which is the deployment question for this shop.
# MAGIC
# MAGIC The validation MAE from the date cut remains the selection number. These folds are a check that the selection number is not an artifact of one lucky quarter.

# COMMAND ----------

def fold_maes(frame, splitter, groups=None):
    features, target = features_target(frame)
    scores = []
    for train_idx, holdout_idx in splitter.split(features, target, groups):
        model = Pipeline([("prep", linear_preprocessor()), ("model", Ridge(alpha=10.0))])
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        holdout_pred = model.predict(features.iloc[holdout_idx])
        scores.append(regression_metrics(target.iloc[holdout_idx], holdout_pred)["mae"])
    return float(np.mean(scores)), float(np.std(scores))


ordered = train.sort_values("visit_date")
cv_rows = []
for label, mean_mae, std_mae in (
    ("random_kfold", *fold_maes(train, KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))),
    ("group_vehicle", *fold_maes(train, GroupKFold(n_splits=5), groups=train["vehicle_id"].to_numpy())),
    ("time_series", *fold_maes(ordered, TimeSeriesSplit(n_splits=4))),
):
    cv_rows.append({"procedure": label, "mae_mean": mean_mae, "mae_std": std_mae})

date_cut = regression_metrics(y_valid, predictions["ridge"])["mae"]
cv_rows.append({"procedure": "date_cut_validation", "mae_mean": date_cut, "mae_std": 0.0})
show(pd.DataFrame(cv_rows).round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leakage
# MAGIC
# MAGIC Each row adds one forbidden input to the same Ridge model. A sharp drop in MAE means the column is carrying the answer. `leak_parts_cost` is a slice of the invoice. `leak_teardown_quote` is the invoice plus a little noise. `leak_prior_cost_including_current` adds today's invoice to the history total. The vehicle-mean row fit on train+valid copies labels from the window you are scoring.

# COMMAND ----------

def ridge_on_extra(extra_frame_train, extra_frame_valid, extra_columns):
    model = Pipeline(
        [
            ("prep", linear_preprocessor_with(extra_columns)),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    columns = FEATURES + extra_columns
    model.fit(extra_frame_train[columns], extra_frame_train[TARGET])
    prediction = model.predict(extra_frame_valid[columns])
    return regression_metrics(extra_frame_valid[TARGET], prediction)


def with_column(frame, values):
    copied = frame.copy()
    copied["vehicle_mean"] = values
    return copied


fallback = float(train[TARGET].mean())
train_sum = train.groupby("vehicle_id")[TARGET].transform("sum")
train_count = train.groupby("vehicle_id")[TARGET].transform("count")
leave_one_out = (train_sum - train[TARGET]) / (train_count - 1)
leave_one_out = leave_one_out.where(train_count > 1, fallback)
honest_valid_mean = valid["vehicle_id"].map(train.groupby("vehicle_id")[TARGET].mean()).fillna(fallback)
honest_mean_train = with_column(train, leave_one_out.to_numpy())
honest_mean_valid = with_column(valid, honest_valid_mean.to_numpy())

both = pd.concat([train, valid], ignore_index=True)
leaky_means = both.groupby("vehicle_id")[TARGET].mean()
leaky_mean_train = with_column(train, train["vehicle_id"].map(leaky_means).fillna(fallback).to_numpy())
leaky_mean_valid = with_column(valid, valid["vehicle_id"].map(leaky_means).fillna(fallback).to_numpy())

leak_specs = [
    ("honest_contract", train, valid, []),
    ("plus_parts_cost", train, valid, ["leak_parts_cost"]),
    ("plus_teardown_quote", train, valid, ["leak_teardown_quote"]),
    ("plus_history_including_today", train, valid, ["leak_prior_cost_including_current"]),
    ("vehicle_mean_fit_on_train", honest_mean_train, honest_mean_valid, ["vehicle_mean"]),
    ("vehicle_mean_fit_on_train_and_valid", leaky_mean_train, leaky_mean_valid, ["vehicle_mean"]),
]

leak_rows = []
for label, frame_train, frame_valid, extra in leak_specs:
    if extra:
        scored = ridge_on_extra(frame_train, frame_valid, extra)
    else:
        scored = regression_metrics(y_valid, predictions["ridge"])
    leak_rows.append({"inputs": label, **scored})
    with mlflow.start_run(run_name=f"leak_{label}"):
        mlflow.log_metrics(scored)
        mlflow.set_tag("module", "5")
        mlflow.set_tag("candidate", label)
        mlflow.set_tag("task", "leak" if label != "honest_contract" else "audit")
        mlflow.set_tag("eligible", "false")

leak_table = pd.DataFrame(leak_rows)
show(leak_table.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Residuals and where the uncertainty lives
# MAGIC
# MAGIC Absolute error grows with the predicted invoice. A single ±RMSE band around every quote overstates confidence on cheap jobs and understates it on catalyst and catastrophe jobs. The decile table is the practical uncertainty report: quote the decile's MAE, not the global RMSE.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for axis, name in zip(axes, ("ridge", "hist_gbm")):
    pred = predictions[name]
    resid = y_valid.to_numpy() - pred
    axis.scatter(pred, resid, s=8, alpha=0.35)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_title(name)
    axis.set_xlabel("Predicted cost")
    axis.set_ylabel("Residual")
fig.tight_layout()
plt.show()

pred = predictions["hist_gbm"]
bands = pd.DataFrame(
    {
        "predicted": pred,
        "absolute_error": np.abs(y_valid.to_numpy() - pred),
    }
)
bands["decile"] = pd.qcut(bands["predicted"], 10, duplicates="drop")
decile_table = (
    bands.groupby("decile", observed=True)
    .agg(
        visits=("absolute_error", "size"),
        mean_predicted=("predicted", "mean"),
        mae=("absolute_error", "mean"),
        p90_absolute_error=("absolute_error", lambda values: float(np.quantile(values, 0.9))),
    )
    .reset_index()
)
show(decile_table.round(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rules for the rest of the project
# MAGIC
# MAGIC - Select the expected-cost champion by validation MAE among runs tagged `task=expected_cost` and `eligible=true`.
# MAGIC - Select the reserve model by coverage near 0.90 and by pinball loss, from the Module 4 quantile runs. A low MAE does not make a P90 model good.
# MAGIC - Reject any run whose inputs include a `leak_` column or a statistic fit with labels from the scored window.
# MAGIC - Prefer GroupKFold when you care about new vehicles, and the date cut when you care about next quarter. Report the procedure next to the number.
# MAGIC - Publish the decile MAE beside the point prediction so a high predicted invoice carries a wider band.
# MAGIC
# MAGIC Next: `06_production`.
