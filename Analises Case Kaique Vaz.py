# Databricks notebook source
# MAGIC %md
# MAGIC # 1.0 Import & Parameters

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Imports

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import *
from pyspark import SparkContext
from pyspark import StorageLevel
from ifood_databricks.toolbelt.data_quality_validator import ifoodDataQualityValidator
from ifood_databricks import datalake, etl
from ifood_databricks.data_products.reader import read
from ifood_databricks.data_products.writer import write
from ifood_databricks.data_products.config import (
    TYPE_BATCH,
    STAGE_SILVER,
    STAGE_GOLD,
    MODE_APPEND,
    MODE_FULL
)

from ifood_databricks.data_products.config import STAGE_SILVER, STAGE_GOLD, MODE_APPEND, TYPE_BATCH, STAGE_TEMP
#Parâmetros para leitura em Silver
read_options_silver = {
  "stage": STAGE_SILVER,
  "type": TYPE_BATCH
}

#Parâmetro para leitura em Gold
read_options_gold = {
  "stage": STAGE_GOLD,
  "type": TYPE_BATCH
}

from ifood_data.commons.order_utils.read import orders_reader
import math
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from itertools import combinations
import requests
import gzip
import tarfile
import io
import pandas as pd
from functools import reduce
import os
from ifood_databricks.gcp import gsheet
import html           
import unicodedata
import re

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.0 Read DF`s

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Read Base AB

# COMMAND ----------

ab_base_0 = spark.read.table("restaurants_sandbox.case_kv")

# 1. Calcular orders_total por customer
consumer_metrics = (
    ab_base_0
    .groupBy("customer_id")
    .agg(countDistinct("order_id").alias("orders_total"))
)

# 2. Percentual da distribuição
w = Window.orderBy(col("orders_total"))

consumer_metrics = consumer_metrics.withColumn(
    "pct_rank", percent_rank().over(w)
)

# 3. Categorizar com novos percentuais
consumer_metrics = consumer_metrics.withColumn(
    "customer_segment",
    when(col("pct_rank") >= 0.90, "High")       # top 10%
    .when(col("pct_rank") >= 0.50, "Medium")    # meio %
    .otherwise("Low")                           # bottom 50%
)

# 4. Juntar na base principal
ab_base = ab_base_0.join(
    consumer_metrics.select("customer_id", "customer_segment"),
    "customer_id",
    "left"
)

from pyspark.sql.functions import (
    col, explode_outer, sum, avg, round, coalesce, lit
)

# 5. explode os items
items_exploded = (
    ab_base
        .select("order_id", explode_outer("items").alias("item"))
)

# 6. explode os garnishItems (às vezes vazio)
garnish_exploded = (
    items_exploded
        .withColumn("garnish", explode_outer("item.garnishItems"))
)

item_value = coalesce(col("item.totalValue.value").cast("double"), lit(0))
item_discount = coalesce(col("item.totalDiscount.value").cast("double"), lit(0))
garnish_value = coalesce(col("garnish.totalValue.value").cast("double"), lit(0))
garnish_discount = coalesce(col("garnish.discount.value").cast("double"), lit(0))

item_final_value_df = (
    garnish_exploded
        .groupBy("order_id", "item")
        .agg(
            sum(garnish_value).alias("garnish_value_sum"),
            sum(garnish_discount).alias("garnish_discount_sum")
        )
        .withColumn("item_value_raw", item_value + col("garnish_value_sum"))
        .withColumn("item_discount_raw", item_discount + col("garnish_discount_sum"))
)


order_item_metrics = (
    item_final_value_df
        .groupBy("order_id")
        .agg(
            round(sum("item_value_raw") / 100, 2).alias("item_total_value_sum"),
            round(avg("item_value_raw") / 100, 2).alias("item_total_value_avg"),
            round(sum("item_discount_raw") / 100, 2).alias("ifood_total_discount")
        )
)

ab_base = ab_base.join(order_item_metrics, "order_id", "left")

display(ab_base)



# COMMAND ----------

# MAGIC %md
# MAGIC #3.0 Exploratory Analysis / AB Test Overview

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.1 AB Group Distribution

# COMMAND ----------

# MAGIC %md
# MAGIC Contagem de usuários por grupo AB

# COMMAND ----------

# Contagem de usuários por grupo AB
ab_base.groupBy("abtest_group") \
       .agg(countDistinct("customer_id").alias("unique_customers"),
            count("order_id").alias("total_orders")) \
       .display()


# COMMAND ----------

# MAGIC %md
# MAGIC ##3.2 Orders & Revenue Overview

# COMMAND ----------

# MAGIC %md
# MAGIC Métricas gerais de pedidos e receita

# COMMAND ----------

ab_base.agg(
    countDistinct("order_id").alias("total_orders"),
    countDistinct("customer_id").alias("total_customers"),
    round(sum("order_total_amount"), 2).alias("total_revenue"),
    round(avg("order_total_amount"), 2).alias("avg_order_amount")
).display()


# COMMAND ----------

# MAGIC %md
# MAGIC ##3.3 Metrics by AB Group

# COMMAND ----------

# MAGIC %md
# MAGIC Métricas segmentadas por grupo (target vs control)

# COMMAND ----------

ab_base.groupBy("abtest_group") \
       .agg(
           countDistinct("customer_id").alias("unique_customers"),
           count("order_id").alias("total_orders"),
           round(sum("order_total_amount"), 2).alias("total_revenue"),
           round(avg("order_total_amount"), 2).alias("avg_order_amount")
       ).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.4 Retention & Engagement Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ###3.4.1 Retention by AB Group

# COMMAND ----------

# MAGIC %md
# MAGIC Número de usuários distintos que fizeram pelo menos um pedido por grupo

# COMMAND ----------

retention_df = ab_base.groupBy("abtest_group") \
    .agg(
        countDistinct("customer_id").alias("unique_customers"),
        count("order_id").alias("total_orders"),
        sum("order_total_amount").alias("total_revenue"),
        avg("order_total_amount").alias("avg_order_amount")
    ).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###3.4.2 Average Orders per Customer

# COMMAND ----------

# MAGIC %md
# MAGIC Frequência média de pedidos por cliente dentro de cada grupo AB

# COMMAND ----------

orders_per_customer_df = ab_base.groupBy("abtest_group", "customer_id") \
    .agg(
        count("order_id").alias("customer_orders"),
        sum("order_total_amount").alias("customer_total_spent")
    ) \
    .groupBy("abtest_group") \
    .agg(
        avg("customer_orders").alias("avg_orders_per_customer"),
        avg("customer_total_spent").alias("avg_revenue_per_customer")
    ).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###3.4.3 – Impacto do teste por grupo AB, dia, semana e mês

# COMMAND ----------

ab_base_enriched_3 = (
    ab_base
    .withColumn("order_date", to_date(col("order_created_at_utc")))
    # semana começando na segunda-feira, formatada como yyyy-MM-dd
    .withColumn( "week", date_format(date_sub(next_day(col("order_date"), "Mon"), 7), "yyyy-MM-dd"))
    # primeiro dia do mês, formatado como yyyy-MM-01
    .withColumn("month", date_format(trunc(col("order_date"), "month"), "yyyy-MM-01"))
    # Concat Cidade e Estado
    .withColumn("city_state", concat_ws(" - ", "merchant_city", "merchant_state"))
)
display(ab_base_enriched_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Função Visões

# COMMAND ----------

def build_metrics(df, period_col, grouping_col):
    metrics = (
        df.groupBy(grouping_col, "abtest_group", period_col)
          .agg(
              countDistinct("customer_id").alias("unique_consumers"),
              round(sum("order_total_amount"), 2).alias("total_revenue"),
              countDistinct("order_id").alias("total_orders"),
              round(avg("order_total_amount"), 2).alias("avg_aov"),

              round(sum("ifood_total_discount"), 2).alias("total_ifood_discount"),
              round(avg("ifood_total_discount"), 2).alias("avg_ifood_discount"),

              round(sum("item_total_value_sum"), 2).alias("total_item_value"),
              round(avg("item_total_value_sum"), 2).alias("avg_item_value_sum"),

              round(avg("item_total_value_avg"), 2).alias("avg_item_value_avg")
          )
          .withColumn(
              "avg_orders_per_consumer",
              round(col("total_orders") / col("unique_consumers"), 2)
          )
          .withColumn(
              "avg_revenue_per_consumer",
              round(col("total_revenue") / col("unique_consumers"), 2)
          )
    )
    return metrics


def build_lift(df, period_col, grouping_col):

    pivoted = (
        df
        .groupBy(period_col, grouping_col)
        .pivot("abtest_group", ["control", "target"])
        .agg(
            first("avg_orders_per_consumer").alias("avg_orders_per_consumer"),
            first("avg_revenue_per_consumer").alias("avg_revenue_per_consumer"),
            first("avg_aov").alias("avg_aov")
        )
    )

def build_lift(df, period_col, grouping_col):

    pivoted = (
        df
        .groupBy(period_col, grouping_col)
        .pivot("abtest_group", ["control", "target"])
        .agg(
            first("avg_orders_per_consumer").alias("avg_orders_per_consumer"),
            first("avg_revenue_per_consumer").alias("avg_revenue_per_consumer"),
            first("avg_aov").alias("avg_aov"),

            first("total_ifood_discount").alias("total_ifood_discount"),
            first("avg_ifood_discount").alias("avg_ifood_discount"),

            first("total_item_value").alias("total_item_value"),
            first("avg_item_value_sum").alias("avg_item_value_sum"),
            first("avg_item_value_avg").alias("avg_item_value_avg")
        )
    )

    lift_df = (
        pivoted
        .withColumn(
            "orders_lift_pct",
            round(
                (col("target_avg_orders_per_consumer") - col("control_avg_orders_per_consumer")) /
                col("control_avg_orders_per_consumer") * 100, 2
            )
        )
        .withColumn(
            "revenue_lift_pct",
            round(
                (col("target_avg_revenue_per_consumer") - col("control_avg_revenue_per_consumer")) /
                col("control_avg_revenue_per_consumer") * 100, 2
            )
        )
        .withColumn(
            "aov_lift_pct",
            round(
                (col("target_avg_aov") - col("control_avg_aov")) /
                col("control_avg_aov") * 100, 2
            )
        )
    )

    return lift_df





# COMMAND ----------

# MAGIC %md
# MAGIC #### Top 3 Cidades

# COMMAND ----------

top10_cities = (
    ab_base_enriched_3.groupBy("city_state")
    .agg(count("*").alias("orders"))
    .orderBy(desc("orders"))
    .limit(3)
)
display(top10_cities)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Diaria

# COMMAND ----------

daily_city = (
    ab_base_enriched_3.join(top10_cities, "city_state", "inner")
)

daily_city_metrics = build_metrics(
    daily_city,
    period_col="order_date",
    grouping_col="city_state"
)

display(daily_city_metrics.filter(col('abtest_group').isNotNull()))

# COMMAND ----------

# DBTITLE 1,Lift
lift_daily_city = build_lift(
    daily_city_metrics.filter(col('abtest_group').isNotNull()),
    period_col="order_date",
    grouping_col="city_state"
)

display(lift_daily_city.orderBy("order_date", "city_state"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Semanal

# COMMAND ----------

weekly_city = (
    ab_base_enriched_3.join(top10_cities, "city_state", "inner")
)

weekly_city_metrics = build_metrics(
    weekly_city,
    period_col="week",
    grouping_col="city_state"
)

display(weekly_city_metrics.orderBy("city_state", "week").filter(col('abtest_group').isNotNull()))


# COMMAND ----------

from ifood_databricks.gcp import gsheet

st = weekly_city_metrics.orderBy("city_state", "week").filter(col('abtest_group').isNotNull())
gsheet.gsheets_data_dump(st, spreadsheet_id= '14ozIt0_YyEsKeK0juEeny4u-babvFQ97pc5y0XDwQ08', range_sheet= 'Cidade Semanal!A1', title=True, clear=True)

# COMMAND ----------

# DBTITLE 1,Lift
lift_weekly_city = build_lift(
    weekly_city_metrics.filter(col('abtest_group').isNotNull()),
    period_col="week",
    grouping_col="city_state"
)

display(lift_weekly_city.orderBy("week", "city_state"))

# COMMAND ----------

# MAGIC %md
# MAGIC #####Mensal

# COMMAND ----------

monthly_city = (
    ab_base_enriched_3.join(top10_cities, "city_state", "inner")
)

monthly_city_metrics = build_metrics(
    monthly_city,
    period_col="month",
    grouping_col="city_state"
)

display(monthly_city_metrics.orderBy("city_state", "month").filter(col('abtest_group').isNotNull()))


# COMMAND ----------

# DBTITLE 1,Lift
lift_monthly_city = build_lift(
    monthly_city_metrics.filter(col('abtest_group').isNotNull()),
    period_col="month",
    grouping_col="city_state"
)

display(lift_monthly_city.orderBy("month", "city_state"))

# COMMAND ----------

# MAGIC %md
# MAGIC ####Classificação User

# COMMAND ----------

# MAGIC %md
# MAGIC #####Diaria

# COMMAND ----------

daily_segment_metrics = build_metrics(
    ab_base_enriched_3,
    period_col="order_date",
    grouping_col="customer_segment"
)

display(daily_segment_metrics.orderBy("customer_segment", "order_date").filter(col('abtest_group').isNotNull()))

# COMMAND ----------

# DBTITLE 1,Lift
lift_daily_segment_metrics = build_lift(
    daily_segment_metrics.filter(col('abtest_group').isNotNull()),
    period_col="order_date",
    grouping_col="customer_segment" 
)

display(lift_daily_segment_metrics.orderBy("order_date", "customer_segment"))

# COMMAND ----------

# MAGIC %md
# MAGIC #####Semanal

# COMMAND ----------

weekly_segment_metrics = build_metrics(
    ab_base_enriched_3,
    period_col="week",
    grouping_col="customer_segment"
)

display(weekly_segment_metrics.orderBy("customer_segment", "week").filter(col('abtest_group').isNotNull()))

# COMMAND ----------

from ifood_databricks.gcp import gsheet

st = weekly_segment_metrics.orderBy("customer_segment", "week").filter(col('abtest_group').isNotNull())
gsheet.gsheets_data_dump(st, spreadsheet_id= '14ozIt0_YyEsKeK0juEeny4u-babvFQ97pc5y0XDwQ08', range_sheet= 'Consumer Class Semanal!A1', title=True, clear=True)

# COMMAND ----------

# DBTITLE 1,Lift
lift_weekly_segment_metrics = build_lift(
    weekly_segment_metrics.filter(col('abtest_group').isNotNull()),
    period_col="week",
    grouping_col="customer_segment"
)

display(lift_weekly_segment_metrics.orderBy("week", "customer_segment"))

# COMMAND ----------

# MAGIC %md
# MAGIC #####Mensal

# COMMAND ----------

monthly_segment_metrics = build_metrics(
    ab_base_enriched_3,
    period_col="month",
    grouping_col="customer_segment"
)

display(monthly_segment_metrics.orderBy("customer_segment", "month").filter(col('abtest_group').isNotNull()))

# COMMAND ----------

# DBTITLE 1,Lift
lift_monthly_segment_metrics = build_lift(
    monthly_segment_metrics.filter(col('abtest_group').isNotNull()),
    period_col="month",
    grouping_col="customer_segment"
)

display(lift_monthly_segment_metrics.orderBy("month", "customer_segment"))

# COMMAND ----------

# MAGIC %md
# MAGIC #4.0 Statistical Relevance
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col
import pandas as pd
from scipy.stats import ttest_ind

segments = ["High", "Medium", "Low"]
results = []

for seg in segments:
    # Filtrar segmento
    df_seg = ab_base.filter(col("customer_segment") == seg).select(
        "customer_id", "abtest_group", "order_id", "order_total_amount"
    ).toPandas()
    
    # Agregar por usuário: frequência de pedidos e AOV
    user_metrics = df_seg.groupby(["customer_id", "abtest_group"]).agg(
        orders_total=("order_id", "count"),
        aov=("order_total_amount", "mean")
    ).reset_index()
    
    # Separar grupos control e target
    control = user_metrics[user_metrics["abtest_group"] == "control"]
    target  = user_metrics[user_metrics["abtest_group"] == "target"]
    
    # T-test para frequência de pedidos
    t_freq, p_freq = ttest_ind(control["orders_total"], target["orders_total"], equal_var=False)
    
    # T-test para AOV
    t_aov, p_aov = ttest_ind(control["aov"], target["aov"], equal_var=False)
    
    results.append({
        "segment": seg,
        "t_stat_orders": t_freq,
        "p_value_orders": p_freq,
        "t_stat_aov": t_aov,
        "p_value_aov": p_aov
    })

significance_df = pd.DataFrame(results)
display(significance_df)


# COMMAND ----------

from pyspark.sql.functions import col
import pandas as pd
from scipy.stats import ttest_ind

segments = ["High", "Medium", "Low"]
results = []

for seg in segments:
    df_seg = ab_base.filter(col("customer_segment") == seg).select(
        "customer_id", "abtest_group", "order_id", "order_total_amount"
    ).toPandas()
    
    # Agregar por usuário
    user_metrics = df_seg.groupby(["customer_id", "abtest_group"]).agg(
        orders_total=("order_id", "count"),
        aov=("order_total_amount", "mean")
    ).reset_index()
    
    control = user_metrics[user_metrics["abtest_group"] == "control"]
    target  = user_metrics[user_metrics["abtest_group"] == "target"]
    
    # T-test
    t_freq, p_freq = ttest_ind(control["orders_total"], target["orders_total"], equal_var=False)
    t_aov, p_aov = ttest_ind(control["aov"], target["aov"], equal_var=False)
    
    results.append({
        "segment": seg,
        "p_value_orders": p_freq,
        "orders_significant": "Yes" if p_freq < 0.05 else "No",
        "p_value_aov": p_aov,
        "aov_significant": "Yes" if p_aov < 0.05 else "No"
    })

significance_df = pd.DataFrame(results)
display(significance_df)
