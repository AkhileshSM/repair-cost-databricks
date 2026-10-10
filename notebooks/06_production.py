# Databricks notebook source
# MAGIC %md
# MAGIC # Module 6 — Production engineering
# MAGIC
# MAGIC This notebook turns the lab runs into a scoring path you can defend.
# MAGIC
# MAGIC 1. Build every candidate as a pipeline so intake columns go in and dollars come out. Modules 1–5 already did that.
# MAGIC 2. Compare them on one validation window. That is the fair benchmark: same features, same dates, same MAE.
# MAGIC 3. Refit the winner on train+validation after selection. Score the test window **once**.
# MAGIC 4. Measure one-row latency and batch throughput on serverless CPU.
# MAGIC 5. Register the model in Unity Catalog when the workspace allows it. Batch-score in this notebook either way.
# MAGIC 6. Compare this week's feature mix to the training mix. Retrain on a real shift, and wait out a single noisy week otherwise.
# MAGIC
# MAGIC Free Edition serves models with tight limits: few active endpoints, no GPU serving, and custom batch inference endpoints are not part of the free tier. The scoring path this module finishes is `mlflow.sklearn.load_model` inside a notebook or a job. An endpoint is an optional click in the Catalog UI after registration, and it may be unavailable for this model type.

# COMMAND ----------

# MAGIC %run ./_lib

# COMMAND ----------

import platform
import resource
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone

experiment = start_experiment()
experiment_id = mlflow.get_experiment_by_name(experiment).experiment_id
visits = load_visits()
train, valid, test = time_split(visits)
x_train, y_train = features_target(train)
x_valid, y_valid = features_target(valid)
x_test, y_test = features_target(test)
print(f"Experiment: {experiment}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fair benchmark
# MAGIC
# MAGIC Eligible runs are expected-cost models trained without leakage columns. The lowest validation MAE wins. When two MAEs are within $10, prefer the earlier module's simpler model only if you care more about reading coefficients than about the extra dollars. The sort below is by MAE, so the first row is the champion under the dollar rule.

# COMMAND ----------

runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])
if runs.empty or "tags.task" not in runs.columns:
    raise RuntimeError("No MLflow runs yet. Run modules 1 through 5, then rerun this notebook.")

pool = runs[(runs["tags.task"] == "expected_cost") & (runs["tags.eligible"] == "true")].copy()
pool = pool.dropna(subset=["metrics.mae"])
if pool.empty:
    raise RuntimeError("No eligible expected-cost runs. Check that modules 1 through 4 finished logging.")

benchmark_columns = [
    column
    for column in (
        "tags.candidate",
        "tags.module",
        "metrics.mae",
        "metrics.rmse",
        "metrics.r2",
        "metrics.within_150",
        "start_time",
    )
    if column in pool.columns
]
benchmark = pool.sort_values(["metrics.mae", "start_time"], ascending=[True, False])
show(benchmark[benchmark_columns].head(15).round(3))

champion_row = benchmark.iloc[0]
print(
    f"Champion: {champion_row.get('tags.candidate', 'unknown')} "
    f"from module {champion_row.get('tags.module', '?')} "
    f"with validation MAE ${champion_row['metrics.mae']:,.0f}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refit, then open the test window once
# MAGIC
# MAGIC The selected run was fit on train only, which kept validation honest. The deployed model is refit on train+validation so the extra quarter is not wasted. Hyperparameters stay frozen. Test MAE is a report, not a tuning signal.

# COMMAND ----------

source_uri = f"runs:/{champion_row.run_id}/model"
loaded = mlflow.sklearn.load_model(source_uri)
try:
    scorer = clone(loaded)
    scorer.fit(pd.concat([x_train, x_valid]), pd.concat([y_train, y_valid]))
    print("Refit the champion on train+validation.")
except Exception as exc:
    scorer = loaded
    print(f"Using the train-only model. Refit was not available: {exc}")

test_prediction = scorer.predict(x_test)
test_metrics = regression_metrics(y_test, test_prediction)
show(pd.DataFrame([{"window": "test", **test_metrics}]).round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Latency and resource use
# MAGIC
# MAGIC One-row latency is what an advisor feels while the repair order is open. Rows per second is what a nightly batch of open orders feels. Both numbers are from this serverless session, so they move when the workspace is busy. Record them on the run rather than comparing them to a laptop.

# COMMAND ----------

row = x_test.iloc[[0]]
scorer.predict(row)
one_row_samples = []
for _ in range(30):
    start = time.perf_counter()
    scorer.predict(row)
    one_row_samples.append((time.perf_counter() - start) * 1000.0)
one_row_ms = float(np.median(one_row_samples))

batch = x_test.iloc[: min(1000, len(x_test))]
scorer.predict(batch)
start = time.perf_counter()
scorer.predict(batch)
elapsed = time.perf_counter() - start
rows_per_second = float(len(batch) / elapsed)

# Linux reports ru_maxrss in kilobytes. macOS reports bytes. The unit is printed with the platform.
max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Platform: {platform.platform()}")
print(f"One-row median latency: {one_row_ms:.2f} ms")
print(f"Batch throughput: {rows_per_second:,.0f} rows/second over {len(batch)} rows")
print(f"Process max RSS (platform units): {max_rss}")

# COMMAND ----------

example = x_train.head(3)
signature = mlflow.models.infer_signature(example, scorer.predict(example))
with mlflow.start_run(run_name="champion_test") as champion_run:
    mlflow.log_metrics(
        {
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_within_150": test_metrics["within_150"],
            "one_row_ms": one_row_ms,
            "rows_per_second": rows_per_second,
        }
    )
    mlflow.log_params(
        {
            "source_run_id": champion_row.run_id,
            "source_candidate": str(champion_row.get("tags.candidate", "")),
            "source_valid_mae": float(champion_row["metrics.mae"]),
        }
    )
    mlflow.set_tag("module", "6")
    mlflow.set_tag("candidate", "champion")
    mlflow.set_tag("task", "champion_eval")
    mlflow.set_tag("eligible", "false")
    try:
        mlflow.sklearn.log_model(scorer, name="model", signature=signature, input_example=example)
    except TypeError:
        mlflow.sklearn.log_model(scorer, "model", signature=signature, input_example=example)
    champion_run_id = champion_run.info.run_id

model_uri = f"runs:/{champion_run_id}/model"
registered_name = qualified_name("repair_cost")
try:
    mlflow.set_registry_uri("databricks-uc")
    registered = mlflow.register_model(model_uri, registered_name)
    print(f"Registered {registered_name} version {registered.version}")
except Exception as exc:
    print("Unity Catalog registration did not complete. Batch scoring still uses the run URI.")
    print(exc)

print(f"Load this model with mlflow.sklearn.load_model('{model_uri}')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference
# MAGIC
# MAGIC The scored table is the test window, which is the stand-in for "visits that arrived after the model was frozen." A nightly job runs the same function on the new day's open repair orders.

# COMMAND ----------

scored = test[["visit_id", "vehicle_id", "visit_date"]].copy()
scored["actual_repair_cost"] = y_test.to_numpy()
scored["predicted_repair_cost"] = test_prediction
scored["absolute_error"] = np.abs(scored["actual_repair_cost"] - scored["predicted_repair_cost"])
scored_table = qualified_name("scored_test")
spark.createDataFrame(scored).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(scored_table)
print(f"Wrote {len(scored):,} scored visits to {scored_table}")
show(spark.table(scored_table).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## One repair order
# MAGIC
# MAGIC This is the payload a service advisor's screen can send. The second row turns on the catalyst code. The predicted dollars should jump. That is a sanity check that the pipeline still sees the scanner flags.

# COMMAND ----------

advisor = pd.DataFrame(
    [
        {
            "mileage": 128400.0,
            "vehicle_age_years": 9.5,
            "dtc_severity_max": 2,
            "dtc_powertrain": 2,
            "dtc_body": 0,
            "dtc_chassis": 0,
            "dtc_network": 0,
            "has_misfire": 1,
            "has_catalyst": 0,
            "has_wheel_speed": 0,
            "prior_repair_count_12m": 1,
            "prior_repair_cost_12m": 640.0,
            "days_since_last_repair": 80.0,
            "repeat_component_90d": 1,
            "warranty_active": 0,
            "is_first_visit": 0,
            "make": "Ford",
            "powertrain": "ice",
            "region": "midwest",
        }
    ]
)
catalyst = advisor.copy()
catalyst["has_catalyst"] = 1
catalyst["dtc_severity_max"] = 3
quotes = pd.DataFrame(
    {
        "scenario": ["misfire", "misfire_and_catalyst"],
        "predicted_repair_cost": [
            float(scorer.predict(advisor)[0]),
            float(scorer.predict(catalyst)[0]),
        ],
    }
)
show(quotes.round(0))

quantile_runs = runs[(runs["tags.candidate"] == "quantile_p90")].copy() if "tags.candidate" in runs.columns else runs.iloc[0:0]
if len(quantile_runs):
    quantile_runs = quantile_runs.dropna(subset=["metrics.mae"]) if "metrics.mae" in quantile_runs.columns else quantile_runs
    latest_quantile = quantile_runs.sort_values("start_time", ascending=False).iloc[0]
    reserve_model = mlflow.sklearn.load_model(f"runs:/{latest_quantile.run_id}/model")
    reserve = pd.DataFrame(
        {
            "scenario": ["misfire", "misfire_and_catalyst"],
            "p90_reserve": [
                float(reserve_model.predict(advisor)[0]),
                float(reserve_model.predict(catalyst)[0]),
            ],
        }
    )
    show(reserve.round(0))
    coverage = latest_quantile.get("metrics.coverage_0.9", float("nan"))
    print(f"P90 validation coverage from the logged run: {coverage}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drift and the retraining rule
# MAGIC
# MAGIC PSI compares the shape of a feature in training with its shape in a new batch.
# MAGIC
# MAGIC - Under 0.10: the mix looks like training.
# MAGIC - 0.10 to 0.25: watch it.
# MAGIC - 0.25 or higher: the intake mix has moved enough to refit.
# MAGIC
# MAGIC A PSI breach retrains immediately. An accuracy breach retrains only after 500 newly labeled invoices and a 14-day cooldown, so one bad week does not spend the daily quota. Accuracy uses invoices that have now closed. PSI does not need labels.

# COMMAND ----------

def monitor_frame(reference, current):
    values = {
        "mileage": population_stability_index(reference["mileage"], current["mileage"]),
        "vehicle_age_years": population_stability_index(
            reference["vehicle_age_years"], current["vehicle_age_years"]
        ),
        "dtc_count": population_stability_index(reference["dtc_count"], current["dtc_count"]),
        "make": categorical_psi(reference["make"], current["make"]),
    }
    return {name: float(value) for name, value in values.items()}


calm_psi = monitor_frame(train, test)
print("PSI, test window versus training (the stationary shop)")
show(pd.Series(calm_psi).round(3))

drifted = test.copy()
drifted["mileage"] = drifted["mileage"] * 1.35
drifted["vehicle_age_years"] = drifted["vehicle_age_years"] + 3
drifted["dtc_count"] = drifted["dtc_count"] + 2
drifted_psi = monitor_frame(train, drifted)
print("PSI, stressed batch (older, higher-mile, more codes)")
show(pd.Series(drifted_psi).round(3))

baseline_mae = float(champion_row["metrics.mae"])
calm_decision = retrain_decision(
    calm_psi,
    recent_mae=test_metrics["mae"],
    baseline_mae=baseline_mae,
    new_labels=int(len(test)),
    days_since_train=120,
)
stress_decision = retrain_decision(
    drifted_psi,
    recent_mae=test_metrics["mae"],
    baseline_mae=baseline_mae,
    new_labels=int(len(test)),
    days_since_train=5,
)
print(f"Stationary test window: retrain={calm_decision[0]} reasons={calm_decision[1]}")
print(f"Stressed batch during the cooldown: retrain={stress_decision[0]} reasons={stress_decision[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to schedule
# MAGIC
# MAGIC The test metrics above are a one-time report. Changing a hyperparameter after you read them spends the holdout. A weekly job should compute PSI and recent MAE on newly closed invoices. Running this whole notebook again re-scores the same synthetic test window, so keep that rerun as a lab replay rather than a new exam.
# MAGIC
# MAGIC Retrain when `retrain_decision` returns true. The new training window is every labeled visit up to the decision date, with a fresh last-quarter validation cut. Register a new version only if the new validation MAE is at least as good as the champion's validation MAE. Keep the previous version queryable so you can score a day twice while you compare.
# MAGIC
# MAGIC Catalog registration, when it succeeds, is `catalog.repair_shop.repair_cost`. Serving that version on a CPU endpoint is optional and is limited on Free Edition. The batch table and the one-row `predict` call are the inference path this course completes.
