# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1 — Linear regression
# MAGIC
# MAGIC You are fitting a straight line in many dimensions:
# MAGIC
# MAGIC `repair_cost = intercept + mileage * b_mileage + age * b_age + ... + error`
# MAGIC
# MAGIC **Ordinary least squares** chooses the coefficients that minimize the sum of squared residuals.
# MAGIC
# MAGIC 1. Write the visits as `y = X beta + error`. The first column of `X` is all ones, so `beta[0]` is the intercept.
# MAGIC 2. Squared-error loss is `L(beta) = ||y - X beta||^2`.
# MAGIC 3. The minimum satisfies the normal equations `(X'X) beta = X'y`.
# MAGIC 4. A residual is `actual - predicted`. OLS makes those residuals orthogonal to every column of `X`, including the column of ones, so the average residual on the training set is zero.
# MAGIC
# MAGIC The coefficient on a scaled numeric feature is the dollar change for a one-standard-deviation move, holding the other columns fixed. The table below also divides by that standard deviation so you can read dollars per mile, per year, and per catalyst flag.
# MAGIC
# MAGIC Run `00_setup_data` first. This notebook fits on train and scores validation. It does not look at the test window.

# COMMAND ----------

# MAGIC %run ./_lib

# COMMAND ----------

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

experiment = start_experiment()
visits = load_visits()
train, valid, test = time_split(visits)
x_train, y_train = features_target(train)
x_valid, y_valid = features_target(valid)

print(f"Experiment: {experiment}")
print(f"train {len(train):,}   valid {len(valid):,}   test {len(test):,} held out")
print("Normal equations implementation:")
print(inspect.getsource(fit_ols_normal_equation))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Least squares, computed directly
# MAGIC
# MAGIC The preprocessor imputes, standardizes numerics, and one-hot encodes make, powertrain, and region. The dropped baselines are Toyota, ice, and midwest. A make coefficient is the contrast against Toyota after the other columns are in the model.

# COMMAND ----------

preprocessor = linear_preprocessor()
design_train = design_matrix(preprocessor, x_train, fit=True)
beta = fit_ols_normal_equation(design_train, y_train)
design_valid = design_matrix(preprocessor, x_valid, fit=False)
manual_prediction = design_valid @ beta

names = ["intercept"] + list(preprocessor.get_feature_names_out())
beta_table = pd.DataFrame({"term": names, "beta_standardized": beta})
show(beta_table.sort_values("beta_standardized", key=np.abs, ascending=False).head(15).round(1))
print(f"Condition number of X'X: {np.linalg.cond(design_train.T @ design_train):,.0f}")
print("A condition number in the tens or low hundreds is a stable solve. Module 2 pushes it to infinity on purpose.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The same fit through scikit-learn
# MAGIC
# MAGIC `LinearRegression` solves the same problem. Predictions should match the normal-equation result up to numerical noise. The pipeline is what MLflow stores, because scoring then accepts raw intake columns and applies the same preprocessing.

# COMMAND ----------

ols = Pipeline([("prep", linear_preprocessor()), ("model", LinearRegression())])
metrics, prediction = log_trained_model(
    ols,
    x_train,
    y_train,
    x_valid,
    y_valid,
    run_name="ols",
    module="1",
    candidate="ols",
    task="expected_cost",
    params={"estimator": "LinearRegression", "fit_intercept": True},
    eligible=True,
)

gap = float(np.mean(np.abs(manual_prediction - prediction)))
print(f"Mean absolute gap between the normal equation and LinearRegression: ${gap:.6f}")
print("Validation metrics (dollars):")
show(pd.DataFrame([metrics]).round(3))

effects = linear_effects_per_unit(ols)
show(effects.sort_values("coef_per_unit", key=np.abs, ascending=False).round(2))
per_mile = float(effects.set_index("feature").loc["mileage", "coef_per_unit"])
print(f"Mileage, read in a useful step: ${per_mile * 10000:,.0f} per 10,000 miles")
print("Mean training invoice by warranty flag:")
show(train.groupby("warranty_active")[TARGET].mean().rename("mean_repair_cost").round(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Residuals and squared-error loss
# MAGIC
# MAGIC On six validation visits, residual is the miss in dollars and squared error is the term OLS sums up. A $400 miss costs the loss function 16 times as much as a $100 miss. That is why a handful of engine-sized invoices pull a least-squares line around.

# COMMAND ----------

worked = pd.DataFrame(
    {
        "actual": y_valid.to_numpy()[:6],
        "predicted": prediction[:6],
    }
)
worked["residual"] = worked["actual"] - worked["predicted"]
worked["squared_error"] = worked["residual"] ** 2
show(worked.round(1))

residual = y_valid.to_numpy() - prediction
sse = float(np.sum(residual**2))
sst = float(np.sum((y_valid.to_numpy() - y_valid.mean()) ** 2))
print(f"SSE (sum of squared errors) on validation: {sse:,.0f}")
print(f"RMSE is the square root of the mean squared error: ${np.sqrt(np.mean(residual**2)):,.0f}")
print(f"MAE treats a $400 miss as four times a $100 miss, not sixteen: ${np.mean(np.abs(residual)):,.0f}")
print(f"R^2 = 1 - SSE/SST = {1 - sse / sst:.3f}")

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(prediction, residual, s=8, alpha=0.35)
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_xlabel("Predicted cost")
axes[0].set_ylabel("Residual (actual - predicted)")
axes[0].set_title("Validation residuals")
axes[1].hist(residual, bins=40)
axes[1].set_xlabel("Residual")
axes[1].set_title("Residual distribution")
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to read the fit
# MAGIC
# MAGIC - The intercept is the prediction when every scaled numeric feature is at its training mean and make, powertrain, and region sit at their baselines (Toyota, ice, midwest).
# MAGIC - `coef_per_unit` on `has_catalyst` is the dollar add for that code, holding the other modeled columns fixed.
# MAGIC - `coef_per_unit` on `mileage` is dollars per mile. The cell above rescales it to a 10,000-mile step. On a rare flag such as `has_catalyst`, the per-unit number is the dollar change for flipping the flag from 0 to 1, and it is larger than the per-standard-deviation number.
# MAGIC - Age and mileage describe almost the same cars. One fit can hand the shared credit to mileage and leave age small, or flip its sign. Module 2 measures that with a bootstrap. Do not treat the age sign from this single fit as a finding.
# MAGIC - Warranty has no term in the generator. The raw warranty average is lower because those vehicles are young and low-mile. The adjusted coefficient can disagree with that raw average, including a positive sign, because a straight line does not absorb the age and mileage curves. It is not a warranty discount.
# MAGIC - The residual plot fans out as predicted cost grows. Squared error treats that fan as a reason to chase the large invoices. Module 4 fits a quantile and a Tweedie model for that shape.
# MAGIC
# MAGIC Next: `02_regularization`.
