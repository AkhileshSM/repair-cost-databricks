# Databricks notebook source
# MAGIC %md
# MAGIC # Module 0 — Repair visits, feature timing, and the Delta table
# MAGIC
# MAGIC **Question.** At intake, before anyone writes the invoice, what will this visit cost the shop?
# MAGIC
# MAGIC **Target.** `repair_cost_usd` is parts plus labor. It is the shop's cost, not the customer's out-of-pocket total. Warranty changes who pays. It does not, in this generator, change the invoice.
# MAGIC
# MAGIC **Features that exist at intake**
# MAGIC - Identity of the vehicle: mileage, age, make, powertrain, region
# MAGIC - What the scanner says right now: severity, counts by DTC family, and three high-cost codes (misfire, catalyst, wheel speed)
# MAGIC - History strictly before this visit: repairs and dollars in the past 365 days, days since the last repair, and whether the same DTC family returned within 90 days
# MAGIC
# MAGIC **Columns that do not exist at intake.** Anything prefixed `leak_` is computed from this invoice or from the next visit. Module 5 puts them back on purpose so you can see inflated accuracy.
# MAGIC
# MAGIC The table is synthetic, seeded, and fully reproducible. Directions you recover (catalyst and BMW cost more; a straight line struggles to share credit between age and mileage) come from the generator. They are not estimates from a real dealer group.
# MAGIC
# MAGIC Lower `N_VEHICLES` if a serverless session is close to the Free Edition daily quota. 8,000 vehicles is about 18,000 visits and is enough for every module.

# COMMAND ----------

# MAGIC %run ./_lib

# COMMAND ----------

N_VEHICLES = 8000

visits = generate_visits(n_vehicles=N_VEHICLES, seed=RANDOM_STATE)
visits_table = qualified_name("visits_raw")
contract_table = qualified_name("feature_contract")

spark.createDataFrame(visits).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(visits_table)
spark.createDataFrame(feature_contract()).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(contract_table)

print(f"Wrote {len(visits):,} visits to {visits_table}")
print(f"Feature contract: {contract_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the shop looks like
# MAGIC
# MAGIC Mileage and age move together because cars accumulate miles as they age. Catalyst visits and BMW visits sit higher on the invoice. Those two facts are the through-line of the later modules.

# COMMAND ----------

profile = spark.sql(
    f"""
    SELECT
      COUNT(*) AS visits,
      COUNT(DISTINCT vehicle_id) AS vehicles,
      ROUND(AVG(repair_cost_usd), 0) AS avg_cost,
      ROUND(percentile_approx(repair_cost_usd, 0.5), 0) AS median_cost,
      ROUND(percentile_approx(repair_cost_usd, 0.9), 0) AS p90_cost,
      ROUND(AVG(has_catalyst), 3) AS catalyst_rate,
      ROUND(AVG(cost_is_catastrophe), 3) AS catastrophe_rate
    FROM {visits_table}
    """
)
show(profile)

by_make = spark.sql(
    f"""
    SELECT make,
           COUNT(*) AS visits,
           ROUND(AVG(repair_cost_usd), 0) AS avg_cost
    FROM {visits_table}
    GROUP BY make
    ORDER BY avg_cost DESC
    """
)
show(by_make)

by_code = spark.sql(
    f"""
    SELECT code, COUNT(*) AS visits, ROUND(AVG(repair_cost_usd), 0) AS avg_cost
    FROM (
      SELECT repair_cost_usd,
             CASE WHEN has_catalyst = 1 THEN 'catalyst' ELSE 'no catalyst' END AS code
      FROM {visits_table}
    )
    GROUP BY code
    ORDER BY avg_cost DESC
    """
)
show(by_code)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Contract
# MAGIC
# MAGIC Model notebooks train on `role = numeric` and `role = categorical` only.
# MAGIC `diagnostic` columns are real intake fields that are algebraically redundant, and Module 2 uses them to break a design matrix on purpose.
# MAGIC `leak` columns are forbidden at scoring time.

# COMMAND ----------

show(spark.table(contract_table).orderBy("role", "name"))
show(spark.table(visits_table).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## How the later notebooks split time
# MAGIC
# MAGIC | Window | Visits | Used for |
# MAGIC | --- | --- | --- |
# MAGIC | before 2025-10-01 | train | fit every model |
# MAGIC | 2025-10-01 to 2026-01-01 | validation | choose hyperparameters and the champion |
# MAGIC | 2026-01-01 onward | test | one final score in Module 6 |
# MAGIC
# MAGIC A random row split would place a vehicle's March visit in train and its January visit in test. History features would then be built from a future the model cannot know at deployment. The date cut matches the decision: score the next visit that has not happened yet.
# MAGIC
# MAGIC Next: open `01_linear_regression` and run it top to bottom.
