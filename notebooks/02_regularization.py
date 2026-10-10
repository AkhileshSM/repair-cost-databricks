# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2 — Regularization
# MAGIC
# MAGIC Multicollinearity means one feature is nearly a remix of the others. The fit can still predict, while the individual coefficients swing wildly between samples.
# MAGIC
# MAGIC This shop has two versions of that problem.
# MAGIC
# MAGIC - **Structural.** Mileage and age move together (cars get older and accumulate miles). Each coefficient is a partial effect. Credit for "older, higher-mile cars cost more" can land on either column.
# MAGIC - **Algebraic.** `mileage_in_thousands` is mileage divided by 1,000. `dtc_count` is the sum of the four family counts. Those pairs make `X'X` singular. A coefficient is then not a unique number.
# MAGIC
# MAGIC Ridge, Lasso, and Elastic Net add a penalty so the fit prefers smaller coefficients.
# MAGIC
# MAGIC - **Ridge** (`L2`) shrinks coefficients toward zero and keeps every feature. It is the right default when correlated measurements are all real, as mileage and age are.
# MAGIC - **Lasso** (`L1`) can set coefficients to exactly zero. It is useful when you believe many columns are irrelevant.
# MAGIC - **Elastic Net** mixes the two penalties. `l1_ratio=1` is Lasso and `l1_ratio=0` is Ridge.
# MAGIC
# MAGIC The penalty strength is `alpha`. Small alpha stays close to ordinary least squares (low bias, higher variance when columns are redundant). Large alpha pulls predictions toward a constant (higher bias, stable coefficients). The validation MAE curve is the trade-off you can see.

# COMMAND ----------

# MAGIC %run ./_lib

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

experiment = start_experiment()
visits = load_visits()
train, valid, test = time_split(visits)
x_train, y_train = features_target(train)
x_valid, y_valid = features_target(valid)
print(f"Experiment: {experiment}")
print(f"train {len(train):,}   valid {len(valid):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variance inflation on the honest features
# MAGIC
# MAGIC VIF = 1 / (1 - R²), where R² comes from regressing that column on the others. VIF near 1 means the column carries its own information. VIF above 10 means the coefficient is poorly identified.

# COMMAND ----------

import inspect

print(inspect.getsource(vif_table))
honest_vif = vif_table(x_train[NUMERIC_FEATURES])
show(honest_vif.round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perfect collinearity, constructed on purpose
# MAGIC
# MAGIC `dtc_count` is withheld from the model because it is exactly the sum of the family counts. Putting both back in sends VIF to infinity. The same thing happens when the identical odometer is stored in miles and in thousands of miles.

# COMMAND ----------

duplicate_miles = pd.DataFrame(
    {
        "mileage": x_train["mileage"],
        "mileage_in_thousands": x_train["mileage"] / 1000.0,
        "vehicle_age_years": x_train["vehicle_age_years"],
    }
)
show(vif_table(duplicate_miles).round(2))

family_columns = ["dtc_powertrain", "dtc_body", "dtc_chassis", "dtc_network"]
duplicate_codes = pd.DataFrame(
    {
        "dtc_count": train.loc[x_train.index, "dtc_count"].to_numpy(),
        "family_sum": x_train[family_columns].sum(axis=1),
        "dtc_severity_max": x_train["dtc_severity_max"],
    }
)
show(vif_table(duplicate_codes).round(2))

singular = x_train[NUMERIC_FEATURES].copy()
singular["dtc_count"] = train.loc[x_train.index, "dtc_count"].to_numpy()
singular_pipe = Pipeline(
    [("prep", linear_preprocessor_with(["dtc_count"])), ("model", LinearRegression())]
)
singular_pipe.fit(train[FEATURES + ["dtc_count"]], y_train)
print(
    "Condition number with dtc_count added back: "
    f"{design_condition_number(singular_pipe, train[FEATURES + ['dtc_count']]):,.0f}"
)

honest_pipe = Pipeline([("prep", linear_preprocessor()), ("model", LinearRegression())])
honest_pipe.fit(x_train, y_train)
print(f"Condition number on the honest contract: {design_condition_number(honest_pipe, x_train):,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Coefficient variance, OLS versus Ridge
# MAGIC
# MAGIC Twenty bootstrap resamples of the training window. Watch the standard deviation of the age and mileage coefficients (dollars per standard deviation). Ridge should tighten them. Prediction quality barely moves, which is the point: regularization stabilizes the explanation more than it changes the MAE.

# COMMAND ----------

rng = np.random.default_rng(RANDOM_STATE)
row_index = np.arange(len(x_train))
boot_rows = []
for draw in range(20):
    take = rng.choice(row_index, size=len(row_index), replace=True)
    for label, estimator in (("ols", LinearRegression()), ("ridge", Ridge(alpha=10.0))):
        pipe = Pipeline([("prep", linear_preprocessor()), ("model", estimator)])
        pipe.fit(x_train.iloc[take], y_train.iloc[take])
        effects = linear_effects_per_unit(pipe).set_index("feature")
        boot_rows.append(
            {
                "model": label,
                "mileage_per_std": effects.loc["mileage", "coef_per_std"],
                "age_per_std": effects.loc["vehicle_age_years", "coef_per_std"],
            }
        )

boot = pd.DataFrame(boot_rows)
summary = boot.groupby("model").agg(["mean", "std"]).round(1)
show(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuning the penalty on the validation window
# MAGIC
# MAGIC Alpha is chosen by validation MAE. The test window is still untouched.

# COMMAND ----------

alphas = np.logspace(-2, 3, 12)


def penalty_path(factory):
    rows = []
    coefficient_rows = []
    for alpha in alphas:
        pipe = Pipeline([("prep", linear_preprocessor()), ("model", factory(alpha))])
        pipe.fit(x_train, y_train)
        scored = regression_metrics(y_valid, pipe.predict(x_valid))
        rows.append({"alpha": float(alpha), **scored})
        effects = linear_effects_per_unit(pipe).set_index("feature")
        coefficient_rows.append(
            {
                "alpha": float(alpha),
                "mileage": effects.loc["mileage", "coef_per_std"],
                "vehicle_age_years": effects.loc["vehicle_age_years", "coef_per_std"],
                "has_catalyst": effects.loc["has_catalyst", "coef_per_std"],
                "warranty_active": effects.loc["warranty_active", "coef_per_std"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(coefficient_rows)


ridge_path, ridge_coefs = penalty_path(lambda alpha: Ridge(alpha=float(alpha)))
lasso_path, lasso_coefs = penalty_path(lambda alpha: Lasso(alpha=float(alpha), max_iter=8000))
elastic_path, elastic_coefs = penalty_path(
    lambda alpha: ElasticNet(alpha=float(alpha), l1_ratio=0.5, max_iter=8000)
)

print("Ridge validation path")
show(ridge_path.round(3))
print("Lasso validation path")
show(lasso_path.round(3))
print("Elastic Net validation path (l1_ratio=0.5)")
show(elastic_path.round(3))

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].semilogx(ridge_path["alpha"], ridge_path["mae"], marker="o", label="Ridge")
axes[0].semilogx(lasso_path["alpha"], lasso_path["mae"], marker="o", label="Lasso")
axes[0].semilogx(elastic_path["alpha"], elastic_path["mae"], marker="o", label="Elastic Net")
axes[0].set_xlabel("alpha")
axes[0].set_ylabel("Validation MAE")
axes[0].set_title("Bias-variance, seen as validation error")
axes[0].legend()
for column in ("mileage", "vehicle_age_years", "has_catalyst", "warranty_active"):
    axes[1].semilogx(ridge_coefs["alpha"], ridge_coefs[column], marker="o", label=column)
axes[1].set_xlabel("Ridge alpha")
axes[1].set_ylabel("Coefficient per standard deviation")
axes[1].set_title("Ridge coefficient paths")
axes[1].legend(fontsize=8)
fig.tight_layout()
plt.show()

# COMMAND ----------

winners = {
    "ridge": (ridge_path, lambda alpha: Ridge(alpha=float(alpha))),
    "lasso": (lasso_path, lambda alpha: Lasso(alpha=float(alpha), max_iter=8000)),
    "elastic_net": (
        elastic_path,
        lambda alpha: ElasticNet(alpha=float(alpha), l1_ratio=0.5, max_iter=8000),
    ),
}

winner_rows = []
for name, (path, factory) in winners.items():
    best_alpha = float(path.loc[path["mae"].idxmin(), "alpha"])
    model = Pipeline([("prep", linear_preprocessor()), ("model", factory(best_alpha))])
    scored, _ = log_trained_model(
        model,
        x_train,
        y_train,
        x_valid,
        y_valid,
        run_name=f"{name}_alpha_{best_alpha:.4g}",
        module="2",
        candidate=name,
        task="expected_cost",
        params={"estimator": name, "alpha": best_alpha, "l1_ratio": 0.5 if name == "elastic_net" else (1.0 if name == "lasso" else 0.0)},
        eligible=True,
    )
    effects = linear_effects_per_unit(model).set_index("feature")
    nonzero = int(np.sum(np.abs(model.named_steps["model"].coef_) > 1e-6))
    winner_rows.append(
        {
            "model": name,
            "alpha": best_alpha,
            "valid_mae": scored["mae"],
            "nonzero_coefs": nonzero,
            "warranty_per_unit": effects.loc["warranty_active", "coef_per_unit"],
            "catalyst_per_unit": effects.loc["has_catalyst", "coef_per_unit"],
        }
    )

show(pd.DataFrame(winner_rows).round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the paths are saying
# MAGIC
# MAGIC - The left-hand curve is the bias-variance trade-off for this shop. The left side (tiny alpha) is ordinary least squares. The right side shrinks the model toward predicting the average invoice and the validation MAE rises.
# MAGIC - Lasso's nonzero count falls as alpha grows. A coefficient landing on zero means that column did not earn its penalty on this sample. Warranty is not in the invoice formula. Its adjusted coefficient is a leftover from curved age and mileage effects, so a large penalty can shrink it. Check whether validation MAE got worse when it shrank. If it did, the leftover was standing in for nonlinearity that Module 3 will model directly.
# MAGIC - Keep Ridge when you want both mileage and age available to the next scoring service. Keep Lasso when you want a short intake form. Elastic Net is the compromise when groups of correlated codes should enter or leave together.
# MAGIC - Choose the alpha at the bottom of the validation curve, then stop. Retuning on the test window would make the final number a second validation.
# MAGIC
# MAGIC Next: `03_nonlinear`.
