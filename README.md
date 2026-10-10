# Repair-cost labs on Databricks Free Edition

Predict the shop invoice for an automotive repair visit from what is known at intake: mileage, vehicle age, diagnostic trouble codes, and repair history. Train with MLflow on serverless compute, then score new visits from the registered pipeline.

The visit table is synthetic and seeded. Coefficients you recover are properties of that generator. They are not estimates from a dealer group.

## Why this curriculum fits

The six modules are one project, not six unrelated demos. Each one answers a question the shop actually has.

| Module | Shop question | What you will see |
| --- | --- | --- |
| 1. Linear regression | What does a straight-line invoice look like, and what is a residual? | Normal equations, coefficients in dollars, squared-error loss on six real rows of the table |
| 2. Regularization | Mileage and age move together. Can we trust either coefficient? | Infinite VIF on duplicated columns, a bootstrap of the age coefficient, Ridge / Lasso / Elastic Net paths |
| 3. Nonlinear models | Cost jumps after a mileage threshold and a catalyst code, instead of adding a fixed amount per mile. | A depth-3 tree you can read, then forest versus boosting |
| 4. Specialized models | A typical invoice, a 90% reserve, a positive skewed dollar amount, and a line-item count are different questions. | Huber, quantiles, Gamma, Tweedie, Poisson, and a timed SVR and k-NN |
| 5. Evaluation | Which number would a service manager use, and which inputs are cheating? | MAE vs RMSE vs R², three cross-validation procedures, a leakage table |
| 6. Production | What do we deploy, how fast does it score, and when do we refit? | One champion, one test score, latency, a batch table, a PSI retraining rule |

Design choices that keep the labs honest:

- The target is the shop invoice (parts plus labor). Warranty changes who pays. In this generator it does not change the invoice.
- History features use only earlier visits. The date split trains on visits before 2025-10-01, validates through 2025-12-31, and leaves 2026 for a single test score.
- `dtc_count` is the sum of the four DTC-family counts, so it is stored and then kept out of the model. Module 2 puts it back to break the design matrix on purpose.
- Columns prefixed `leak_` use today's invoice or the next visit. They exist so Module 5 can show inflated accuracy.
- The champion is whichever eligible model has the lowest validation MAE. A Tweedie or Gamma model can beat a tree here, because the generator is mostly a log-linear invoice plus a mileage threshold. That result is a success, not a broken lab.

## Free Edition limits these labs stay inside

Free Edition is serverless only. There is no classic cluster and no separate "ML runtime" picker. Python notebooks, MLflow, scikit-learn, and a Delta table are enough.

- Training is single-node. The generator writes about 18,000 visits from 8,000 vehicles, which fits in memory. Lower `N_VEHICLES` in notebook 00 if you are close to the daily compute quota.
- Do not call `DataFrame.cache()` or `persist()`. Serverless rejects them.
- XGBoost and LightGBM are part of the current serverless ML libraries. If an import fails, install in its own cell with `%pip install xgboost` or `%pip install lightgbm`, restart Python, and rerun that notebook from the top. CatBoost is optional.
- SVR is capped at 2,000 training rows. An RBF SVR on the full window will spend the quota before it teaches anything the booster has not already shown.
- Model serving on Free Edition limits how many endpoints you can have, and it does not offer GPU serving or custom batch-inference endpoints. Notebook 06 scores with `mlflow.sklearn.load_model`. Registering in Unity Catalog is attempted, and batch scoring still works when registration is declined.
- The experiment is created at `/Users/<your-login>/repair-cost`. Tables land in `<current catalog>.repair_shop`.

## Put the notebooks in the workspace

The notebooks call `%run ./_lib`, so they have to sit in one folder.

1. In the Databricks workspace, open your user folder.
2. Create a folder named `repair-cost`.
3. Import every file in `notebooks/` into that folder (Workspace → Import). Databricks treats a `.py` file that starts with `# Databricks notebook source` as a notebook.
4. Open `00_setup_data`. The compute picker should show serverless. Attach it.
5. Run notebooks in order: `00`, `01`, `02`, `03`, `04`, `05`, `06`. Each later notebook reads the Delta table from notebook 00. Notebook 06 also reads the MLflow runs from the earlier modules.

If a day hits the Free Edition usage limit, the table and the runs remain. Resume the next notebook when the limit resets.

## What each notebook is doing

### 00 — data and the contract

Writes `<catalog>.repair_shop.visits_raw` and `<catalog>.repair_shop.feature_contract`.

Intake features are mileage, age, make, powertrain, region, DTC severity, counts by family (powertrain, body, chassis, network), three high-cost codes (misfire, catalyst, wheel speed), and history from the previous 365 days. `days_since_last_repair` is `-1` when the vehicle has no prior visit, and `is_first_visit` flags that case so a linear model does not treat a new vehicle as a vehicle that was in yesterday.

After it runs, the SQL cells should show BMW and catalyst visits sitting above the rest of the shop, and a median invoice below the mean. That skew is why Module 1's residual plot fans out.

### 01 — least squares

Fits `y = X beta + error` by solving `(X'X) beta = X'y`, then fits the same model as a scikit-learn pipeline and logs it to MLflow. The gap between the two prediction vectors should be numerical noise, well under a cent.

Read `coef_per_unit` as dollars in the original units. Mileage is easier per 10,000 miles than per mile. The catalyst flag's per-unit coefficient is the change from "code off" to "code on", and it will be much larger than the per-standard-deviation coefficient because the code is rare.

Compare the raw warranty average with the adjusted warranty coefficient. The generator never uses warranty in the invoice. Young, low-mile cars are cheaper, so the raw average is lower. The adjusted sign can still come out positive, because age and mileage enter the true invoice as curves and the line cannot absorb those curves.

### 02 — multicollinearity and the penalty

The honest numeric features have moderate VIFs. Age and mileage share information (correlation is high, around 0.87 in the generator) without being copies of each other.

Two constructed frames are singular:

- mileage and mileage divided by 1,000
- `dtc_count` and the sum of the four family counts

VIF goes to infinity and the condition number of `X'X` explodes. That is why `dtc_count` is not in the model matrix.

The bootstrap compares the spread of the age and mileage coefficients under ordinary least squares and under Ridge. The alpha paths are the bias-variance trade-off: small alpha stays near least squares, large alpha shrinks predictions toward a constant and validation MAE rises. The notebook logs the alpha with the best validation MAE for Ridge, Lasso, and Elastic Net.

### 03 — trees, forests, boosting

The depth-3 tree is printed with `export_text`. The first splits should cut on a high-cost code or on severity, because those cuts reduce the variance of the invoice inside a node.

A fully grown tree is fit as a negative example: training MAE collapses and validation MAE gives the memorization back. The random forest averages bootstrap trees and closes that gap. That is bagging.

HistGradientBoosting, XGBoost, and LightGBM then fit small grids. Boosting adds trees that chase the residual of the current sum. The learning rate is the size of each step; depth or `num_leaves` is how much interaction one tree may represent. Permutation importance on the best booster reports how many validation dollars each column is worth. The optional CatBoost cell skips itself when the package is absent.

CPU histogram training is the default (`tree_method="hist"`). A serverless GPU session can set `device="cuda"` on XGBoost after the CPU run is logged, so the two runs stay comparable.

### 04 — other questions, other losses

Huber is compared with least squares on all validation visits, on ordinary visits, and on catastrophe visits (the generator's engine-sized invoices, about 2.5% of rows). Huber is the data-quality model. The tail is still real work, so the reserve model is a quantile, not Huber.

Quantile gradient boosting is logged at 0.10, 0.50, and 0.90. Coverage at 0.90 should land near 90% of validation invoices at or below the prediction. Pinball loss is the number to compare across quantile models. These runs are tagged `task=quantile` so a small pinball number cannot win the dollar championship.

Gamma and Tweedie model positive skewed dollars. Poisson models `line_item_count` and is tagged `task=count` so its MAE, which is in line items, cannot win either.

SVR (2,000 rows) and k-NN (full training window) are timed and logged as latency comparisons only.

### 05 — metrics, folds, leakage

Refits Ridge, Tweedie, and histogram gradient boosting on the same validation window so MAE, RMSE, R², and the share of misses within $150 are comparable.

Cross-validation on the training window:

- random KFold
- GroupKFold by `vehicle_id`, for "a vehicle we have never seen"
- TimeSeriesSplit, for "next dates"
- the date-cut validation MAE from the Ridge row above

The date cut is the selection procedure. The folds tell you whether that number depends on one lucky quarter. This generator is stationary, so the gap between procedures can be modest. The leakage table is where the number moves.

The leakage table adds one forbidden input at a time to the same Ridge model. `leak_parts_cost` is a slice of the invoice you are predicting. Expect validation R² near 0.98 and MAE to fall to roughly $100. That model cannot be deployed, because the parts line is written with the invoice. The vehicle-mean feature fit on train+valid copies labels from the window being scored. The vehicle-mean feature fit on past training visits only is a legal history feature.

The decile table is the uncertainty report: absolute error grows with the predicted invoice, so a high quote needs a wider band than the global RMSE.

### 06 — champion, test, latency, drift

Builds a leaderboard from MLflow runs with `task=expected_cost` and `eligible=true`. Lowest validation MAE wins. The winner is refit on train+validation with frozen hyperparameters, then scored once on the 2026 test window. Test metrics are logged under different names (`test_mae`, not `mae`) and the run is not eligible, so a second pass cannot crown the test score.

The notebook writes `<catalog>.repair_shop.scored_test` and scores two advisor payloads: a misfire, and the same visit with a catalyst code added. The catalyst quote should jump.

It also loads the P90 model when Module 4 logged one, and prints that model's validation coverage.

PSI of the test window against training should sit under 0.10 on this stationary generator. A stressed batch (higher mileage, older vehicles, more codes) should push PSI to about 0.45 and trip the retraining rule even during the 14-day cooldown. An accuracy drop without a PSI breach waits for 500 new labels and the end of the cooldown.

Unity Catalog registration uses the name `<catalog>.repair_shop.repair_cost`. If the workspace rejects it, the printed `runs:/.../model` URI is the model you load.

## Scoring a new visit

After notebook 06:

```python
import mlflow
import pandas as pd

model = mlflow.sklearn.load_model("runs:/<champion_run_id>/model")
order = pd.DataFrame([{
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
}])
float(model.predict(order)[0])
```

The pipeline imputes, encodes, and predicts. Pass the raw intake columns. Do not pass any `leak_` column.

A nightly batch is the same `predict` on a DataFrame of open repair orders, written to a Delta table the way notebook 06 writes `scored_test`.

## Retraining rule

Retrain immediately when PSI on mileage, age, DTC count, or make is at least 0.25.

Retrain for accuracy only when all three are true: at least 500 newly closed invoices, the new MAE is above 1.25 times the champion's validation MAE, and at least 14 days have passed since the last fit.

Register a new version only when the new model's validation MAE is at or below the current champion. Keep the previous version loadable while you compare a day's scores.

## Sanity band from a local smoke test

These figures come from a local run of the same generator with 2,000 vehicles (about 4,600 visits) and seed 42. The Databricks notebooks use 8,000 vehicles, so your validation MAE will move. Treat the band as a wiring check.

| Check | Smoke result |
| --- | --- |
| Median invoice, mean invoice | about $700, about $1,080 |
| Correlation of age and mileage | about 0.87 |
| Condition number of the honest design | about 70 |
| OLS validation MAE, R² | about $450, about 0.40 |
| Tweedie validation MAE | about $390 |
| Histogram gradient boosting validation MAE | about $410 |
| Model that sees `leak_parts_cost` | MAE about $100, R² about 0.98 |
| P90 coverage with a short booster | about 0.87 |
| PSI, test vs train, mileage | about 0.07 |
| PSI, mileage inflated 35% | about 0.45 |

A validation MAE of tens of dollars means a leak column is in the feature list. A validation MAE of several thousand dollars means the target or the date split is wrong. Coverage far from 0.90 means the reserve model needs more trees before anyone uses it as a not-to-exceed number.

This smoke test did not run inside a Databricks workspace, and it did not execute XGBoost, LightGBM, or CatBoost.
