# Databricks notebook source
# MAGIC %md
# MAGIC # Module 3 — Nonlinear models
# MAGIC
# MAGIC A linear model adds a fixed dollar amount per extra mile. Repair cost does not work that way. The generator adds a surcharge only after 100,000 miles, and a catalyst code on an old car costs more than the same code on a new car. A tree represents that with a split: "is mileage above this threshold, and is the catalyst flag on?"
# MAGIC
# MAGIC **How a split is chosen.** A regression tree tries thresholds and keeps the one that most reduces impurity. For squared-error trees, impurity in a node is the variance of the invoices in that node. The split puts high-cost catalyst jobs in one leaf and ordinary visits in another.
# MAGIC
# MAGIC **Bagging versus boosting**
# MAGIC - A single deep tree memorizes the training shop. Its training MAE looks excellent and its validation MAE gives the memorization back.
# MAGIC - **Bagging** (Random Forest) grows many deep trees on bootstrap samples and averages them. Averaging reduces variance. Each tree still sees a random subset of columns, so one noisy code cannot dominate every tree.
# MAGIC - **Boosting** (gradient boosting, XGBoost, LightGBM, CatBoost) grows small trees in sequence. Each new tree fits the remaining residual of the sum so far. A low learning rate and enough trees reduce bias. Too many trees, or a learning rate that is too high, fit noise.
# MAGIC
# MAGIC XGBoost, LightGBM, and CatBoost are the same idea with different split-finding machinery. On this table they should land near each other. The validation MAE picks the one you keep. HistGradientBoosting is the same algorithm from scikit-learn, with no extra install, so the module still runs if a Free Edition network policy blocks pip.

# COMMAND ----------

# MAGIC %run ./_lib

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_text

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
    print("XGBoost is not installed. In a cell above this one run: %pip install xgboost")
    print("Then run dbutils.library.restartPython() and rerun the notebook from the top.")

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
    print("LightGBM is not installed. In a cell above this one run: %pip install lightgbm")
    print("Then restart Python and rerun from the top.")

experiment = start_experiment()
visits = load_visits()
train, valid, test = time_split(visits)
x_train, y_train = features_target(train)
x_valid, y_valid = features_target(valid)
print(f"Experiment: {experiment}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## A shallow tree you can read
# MAGIC
# MAGIC Depth 3 is deliberately small. The text below is the whole model. The first split should be a high-cost code or a high severity, because those cuts reduce invoice variance the most.

# COMMAND ----------

tree = Pipeline(
    [
        ("prep", tree_preprocessor()),
        ("model", DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=RANDOM_STATE)),
    ]
)
tree_metrics, _ = log_trained_model(
    tree,
    x_train,
    y_train,
    x_valid,
    y_valid,
    run_name="decision_tree_depth3",
    module="3",
    candidate="decision_tree",
    task="expected_cost",
    params={"max_depth": 3, "min_samples_leaf": 40},
    eligible=True,
)
feature_names = [str(name) for name in tree.named_steps["prep"].get_feature_names_out()]
print(export_text(tree.named_steps["model"], feature_names=feature_names, decimals=1))
print("Tree validation MAE ${:,.0f}".format(tree_metrics["mae"]))
print("Tree training MAE ${:,.0f}".format(regression_metrics(y_train, tree.predict(x_train))["mae"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forest versus one deep tree
# MAGIC
# MAGIC The deep tree is allowed to memorize. The forest averages 200 trees. Compare training MAE with validation MAE: the gap is the variance that bagging removes.

# COMMAND ----------

deep_tree = Pipeline(
    [
        ("prep", tree_preprocessor()),
        ("model", DecisionTreeRegressor(max_depth=None, min_samples_leaf=2, random_state=RANDOM_STATE)),
    ]
)
deep_tree.fit(x_train, y_train)

forest = Pipeline(
    [
        ("prep", tree_preprocessor()),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=2,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)
forest_metrics, _ = log_trained_model(
    forest,
    x_train,
    y_train,
    x_valid,
    y_valid,
    run_name="random_forest",
    module="3",
    candidate="random_forest",
    task="expected_cost",
    params={"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt"},
    eligible=True,
)

overfit = pd.DataFrame(
    [
        {
            "model": "deep_tree",
            "train_mae": regression_metrics(y_train, deep_tree.predict(x_train))["mae"],
            "valid_mae": regression_metrics(y_valid, deep_tree.predict(x_valid))["mae"],
        },
        {
            "model": "random_forest",
            "train_mae": regression_metrics(y_train, forest.predict(x_train))["mae"],
            "valid_mae": forest_metrics["mae"],
        },
    ]
)
show(overfit.round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boosting
# MAGIC
# MAGIC `learning_rate` is how far each new tree is allowed to move the prediction. `max_depth` / `num_leaves` is how much interaction a single tree may represent. More trees with a smaller learning rate usually beat a few aggressive trees. The grid is small on purpose: Free Edition compute is quota-limited, and eight fits teach the knobs.

# COMMAND ----------

hgb_grid = [
    {"max_depth": 3, "learning_rate": 0.08, "max_iter": 200},
    {"max_depth": 6, "learning_rate": 0.06, "max_iter": 300},
]


def fit_grid(name, grid, builder):
    rows = []
    best = None
    for params in grid:
        model = Pipeline([("prep", tree_preprocessor()), ("model", builder(params))])
        model.fit(x_train, y_train)
        scored = regression_metrics(y_valid, model.predict(x_valid))
        row = {"model": name, **params, "valid_mae": scored["mae"], "valid_rmse": scored["rmse"]}
        rows.append(row)
        if best is None or scored["mae"] < best[0]:
            best = (scored["mae"], params)
    return pd.DataFrame(rows), best[1]


hgb_table, hgb_params = fit_grid(
    "hist_gbm",
    hgb_grid,
    lambda params: HistGradientBoostingRegressor(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    ),
)

grid_tables = [hgb_table]
winners = [("hist_gbm", hgb_params, lambda params: HistGradientBoostingRegressor(
    max_depth=params["max_depth"],
    learning_rate=params["learning_rate"],
    max_iter=params["max_iter"],
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=RANDOM_STATE,
))]

if XGBRegressor is not None:
    xgb_grid = [
        {"max_depth": 3, "learning_rate": 0.08, "n_estimators": 200},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 400},
    ]
    xgb_table, xgb_params = fit_grid(
        "xgboost",
        xgb_grid,
        lambda params: XGBRegressor(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    )
    grid_tables.append(xgb_table)
    winners.append(("xgboost", xgb_params, lambda params: XGBRegressor(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=2,
        random_state=RANDOM_STATE,
    )))

if LGBMRegressor is not None:
    lgb_grid = [
        {"num_leaves": 15, "learning_rate": 0.08, "n_estimators": 250},
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 400},
    ]
    lgb_table, lgb_params = fit_grid(
        "lightgbm",
        lgb_grid,
        lambda params: LGBMRegressor(
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    )
    grid_tables.append(lgb_table)
    winners.append(("lightgbm", lgb_params, lambda params: LGBMRegressor(
        num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=2,
        random_state=RANDOM_STATE,
        verbose=-1,
    )))

show(pd.concat(grid_tables, ignore_index=True).round(2))

# COMMAND ----------

logged = []
champion_model = None
for name, params, builder in winners:
    model = Pipeline([("prep", tree_preprocessor()), ("model", builder(params))])
    scored, _ = log_trained_model(
        model,
        x_train,
        y_train,
        x_valid,
        y_valid,
        run_name=name,
        module="3",
        candidate=name,
        task="expected_cost",
        params=params,
        eligible=True,
    )
    logged.append({"model": name, "valid_mae": scored["mae"], "train_mae": regression_metrics(y_train, model.predict(x_train))["mae"]})
    if champion_model is None or scored["mae"] < champion_model[0]:
        champion_model = (scored["mae"], name, model)

show(pd.DataFrame(logged).round(1))
print(f"Best booster in this module: {champion_model[1]} at validation MAE ${champion_model[0]:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which columns the booster actually uses
# MAGIC
# MAGIC Permutation importance shuffles one validation column at a time and records the rise in MAE. A large rise means the model was relying on that column. This is an explanation of the fitted model, not a causal estimate.

# COMMAND ----------

best_model = champion_model[2]
sample = x_valid.sample(n=min(1500, len(x_valid)), random_state=RANDOM_STATE)
importance = permutation_importance(
    best_model,
    sample,
    y_valid.loc[sample.index],
    n_repeats=4,
    random_state=RANDOM_STATE,
    scoring="neg_mean_absolute_error",
    n_jobs=2,
)
importance_table = pd.DataFrame(
    {
        "feature": sample.columns,
        "mae_increase": importance.importances_mean,
    }
).sort_values("mae_increase", ascending=False)
show(importance_table.round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional CatBoost
# MAGIC
# MAGIC CatBoost uses ordered boosting and its own handling of categoricals. This cell skips cleanly when the package is absent. To try it, run `%pip install catboost` in a preceding cell, restart Python, and rerun the notebook from the top. One configuration is enough to see whether it beats the validation MAE above.

# COMMAND ----------

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
    print("CatBoost is not installed. The rest of the course does not need it.")

if CatBoostRegressor is not None:
    cat_model = Pipeline(
        [
            ("prep", tree_preprocessor()),
            (
                "model",
                CatBoostRegressor(
                    depth=6,
                    learning_rate=0.08,
                    iterations=300,
                    loss_function="RMSE",
                    random_seed=RANDOM_STATE,
                    verbose=0,
                ),
            ),
        ]
    )
    cat_metrics, _ = log_trained_model(
        cat_model,
        x_train,
        y_train,
        x_valid,
        y_valid,
        run_name="catboost",
        module="3",
        candidate="catboost",
        task="expected_cost",
        params={"depth": 6, "learning_rate": 0.08, "iterations": 300},
        eligible=True,
    )
    print(f"CatBoost validation MAE ${cat_metrics['mae']:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to choose among the tree models
# MAGIC
# MAGIC - Keep the shallow tree when you need a rule a service manager can read aloud.
# MAGIC - Keep the forest when you want a strong model and a small train-to-validation gap with almost no tuning.
# MAGIC - Keep a booster when the validation MAE is materially lower. Prefer the one with the shallower trees when two MAEs are within about $10, because the shallower model is easier to debug.
# MAGIC - `tree_method="hist"` trains XGBoost on CPU, which is what a normal Free Edition serverless notebook has. A serverless GPU session can set `device="cuda"` on that estimator. Do that only after the CPU baseline is logged, so the MLflow runs stay comparable.
# MAGIC
# MAGIC Next: `04_specialized`.
