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

import html           
import unicodedata
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Parameters & Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2.1 Date

# COMMAND ----------

TODAY_STR = date.today()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2.2 URL`s Bases

# COMMAND ----------

URLS = {
    "orders":     "https://data-architect-test-source.s3-sa-east-1.amazonaws.com/order.json.gz",
    "consumer":   "https://data-architect-test-source.s3-sa-east-1.amazonaws.com/consumer.csv.gz",
    "restaurant": "https://data-architect-test-source.s3-sa-east-1.amazonaws.com/restaurant.csv.gz",
    "ab_test":    "https://data-architect-test-source.s3-sa-east-1.amazonaws.com/ab_test_ref.tar.gz"
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2.3 Schema Itens Struct

# COMMAND ----------

item_schema = ArrayType(
    StructType([
        StructField("name", StringType(), True),
        StructField("addition", StructType([
            StructField("value", StringType(), True),
            StructField("currency", StringType(), True)
        ]), True),
        StructField("discount", StructType([
            StructField("value", StringType(), True),
            StructField("currency", StringType(), True)
        ]), True),
        StructField("quantity", DoubleType(), True),
        StructField("sequence", IntegerType(), True),
        StructField("unitPrice", StructType([
            StructField("value", StringType(), True),
            StructField("currency", StringType(), True)
        ]), True),
        StructField("externalId", StringType(), True),
        StructField("totalValue", StructType([
            StructField("value", StringType(), True),
            StructField("currency", StringType(), True)
        ]), True),
        StructField("customerNote", StringType(), True),
        StructField("garnishItems", ArrayType(StructType([
            StructField("name", StringType(), True),
            StructField("addition", StructType([
                StructField("value", StringType(), True),
                StructField("currency", StringType(), True)
            ]), True),
            StructField("discount", StructType([
                StructField("value", StringType(), True),
                StructField("currency", StringType(), True)
            ]), True),
            StructField("quantity", DoubleType(), True),
            StructField("sequence", IntegerType(), True),
            StructField("unitPrice", StructType([
                StructField("value", StringType(), True),
                StructField("currency", StringType(), True)
            ]), True),
            StructField("categoryId", StringType(), True),
            StructField("externalId", StringType(), True),
            StructField("totalValue", StructType([
                StructField("value", StringType(), True),
                StructField("currency", StringType(), True)
            ]), True),
            StructField("categoryName", StringType(), True),
            StructField("integrationId", StringType(), True)
        ])), True),
        StructField("integrationId", StringType(), True),
        StructField("totalAddition", StructType([
            StructField("value", StringType(), True),
            StructField("currency", StringType(), True)
        ]), True),
        StructField("totalDiscount", StructType([
            StructField("value", StringType(), True),
            StructField("currency", StringType(), True)
        ]), True)
    ])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2.4 Function Read

# COMMAND ----------

def load_dataset(url: str):
    print(f"\n📥 Carregando: {url}")

    u = url.lower()

    if u.endswith((".json", ".json.gz", ".csv", ".csv.gz", ".parquet")):
        s3_url = (
            url.replace("https://", "s3://")
               .replace(".s3-sa-east-1.amazonaws.com", ""))

        print(f"🔗 Lendo direto do S3: {s3_url}")
        if s3_url.endswith((".json", ".json.gz")):
            return spark.read.json(s3_url)
        if s3_url.endswith((".csv", ".csv.gz")):
            return spark.read.option("header", True).csv(s3_url)
        if s3_url.endswith(".parquet"):
            return spark.read.parquet(s3_url)

    # 2- TAR.GZ
    if u.endswith((".tar.gz", ".tgz")):
        print("📦 Detectado TAR.GZ — baixando em memória...")
        r = requests.get(url)
        r.raise_for_status()

        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
            target = next(
                (m for m in tar.getmembers()
                 if m.name.endswith(".csv") and not m.name.startswith("._")),
                None)
            if not target:
                raise FileNotFoundError("Nenhum CSV válido dentro do TAR.GZ.")
              
            print(f"➡️ Encontrado CSV interno: {target.name}")
            f = tar.extractfile(target)
            try:
                pdf = pd.read_csv(f)
            except:
                f.seek(0)
                pdf = pd.read_csv(f, encoding="latin-1")

        return spark.createDataFrame(pdf)

    raise Exception(f"❌ Formato não suportado: {url}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2.5 Function Tratament City & Customer Name

# COMMAND ----------

def clean_text(text):
    if text is None:
        return None
    
    # 1. HTML entities
    text = html.unescape(text)
    
    # 2. Strip e múltiplos espaços
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Remove pontuação do começo e fim
    text = re.sub(r'^[^\w]+|[^\w]+$', '', text)
    
    # 4. Maiúscula
    text = text.upper()
    
    # 5. Remove acentos
    text = ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
    
    # 6. Remove caracteres não alfanuméricos no meio (exceto espaço)
    text = re.sub(r'[^A-Z0-9 ]+', '', text)
    
    # 7. Remove múltiplos espaços gerados após remoção
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

clean_text_udf = udf(clean_text, StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2.6 Others Functions

# COMMAND ----------

def clean_text(text):
    if not text:
        return None
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = text.upper()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

clean_text_udf = udf(clean_text, StringType())

# Função para padronizar delivery_address_external_id
def pad_external_id(val):
    if not val:
        return None
    return val.zfill(7)

pad_external_id_udf = udf(pad_external_id, StringType())

# Função para converter items JSON string para array de struct
items_schema = ArrayType(
    StructType([
        StructField("name", StringType(), True),
        StructField("quantity", DoubleType(), True),
        StructField("unitPrice", MapType(StringType(), StringType()), True),
        StructField("totalValue", MapType(StringType(), StringType()), True),
        StructField("externalId", StringType(), True),
        StructField("garnishItems", ArrayType(MapType(StringType(), StringType())), True),
        StructField("customerNote", StringType(), True),
        StructField("integrationId", StringType(), True),
        StructField("addition", MapType(StringType(), StringType()), True),
        StructField("discount", MapType(StringType(), StringType()), True),
        StructField("totalAddition", MapType(StringType(), StringType()), True),
        StructField("totalDiscount", MapType(StringType(), StringType()), True),
        StructField("sequence", DoubleType(), True)
    ])
)

def parse_items(text):
    if not text:
        return []
    try:
        return json.loads(text)
    except:
        return []

parse_items_udf = udf(parse_items, items_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.0 Transforming

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Raw Data Treatment 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.1 Orders

# COMMAND ----------


def clean_orders_df(orders_df):
    orders_clean = (
        orders_df
        .withColumn("rn", row_number().over(window_order))
        .filter(col("rn") == 1)
        .select(
            col("customer_id"),
            clean_text_udf(col("customer_name")).alias("customer_name"),
            clean_text_udf(col("delivery_address_city")).alias("delivery_address_city"),
            pad_external_id_udf(col("delivery_address_external_id")).alias("delivery_address_external_id"),
            col("delivery_address_state"),
            col("merchant_id"),
            to_utc_timestamp(col("order_created_at"), col("merchant_timezone")).alias("order_created_at_utc"),
            col("order_id"),
            round(col("order_total_amount"), 2).alias("order_total_amount"),
            col("origin_platform"),
            col("order_scheduled_date"),
            from_json(col("items"), item_schema).alias("items"),  
            col("merchant_timezone")
        )
    )
    return orders_clean


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.2 Consumer

# COMMAND ----------


def clean_consumer_df(consumer_df):
    consumers_clean = (
        consumer_df
        .select(
            col("customer_id"),
            col("language"),
            col("created_at"),
            col("active"),
            clean_text_udf(col("customer_name")).alias("customer_name"),
            col("customer_phone_area"),
            col("customer_phone_number")
        )
    )
    return consumers_clean


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.3 Restaurant

# COMMAND ----------


def clean_restaurant_df(restaurant_df):
    def cluster_average_ticket(col_ticket):
        return when(col_ticket <= 30, 30)\
            .when(col_ticket <= 40, 40)\
            .when(col_ticket <= 60, 60)\
            .when(col_ticket <= 80, 80)\
            .otherwise(100)

    merchants_clean = (
        restaurant_df
        .select(
            col("id").alias("merchant_id"),
            col("created_at"),
            col("enabled"),
            col("price_range"),
            cluster_average_ticket(col("average_ticket")).alias("average_ticket"),
            col("takeout_time"),
            col("delivery_time"),
            round(col("minimum_order_value"), 2).alias("minimum_order_value"),
            col("merchant_zip_code"),
            clean_text_udf(col("merchant_city")).alias("merchant_city"),
            col("merchant_state"),
            col("merchant_country")
        )
    )
    return merchants_clean

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Joining All

# COMMAND ----------


def def_joining_all(orders, consumers, merchants, abtest):

  orders = orders_clean.alias("o")
  consumers = consumers_clean.alias("c")
  merchants = merchants_clean.alias("m")
  abtest = abtest_df.alias("ab")

  joining_all = (
      orders
      .join(broadcast(merchants), ['merchant_id'], 'left')
      .join(consumers, ['customer_id'], 'left')
      .join(abtest, ['customer_id'], 'left')
      .select(
          # orders
          col("order_id"),
          col("order_created_at_utc"),
          col("order_scheduled_date"),
          col("origin_platform"),
          col("merchant_timezone"),
          coalesce(col("o.customer_name"), col("c.customer_name")).alias("customer_name"),
          coalesce(col("delivery_address_city"), col("merchant_city")).alias("delivery_address_city"),
          col("delivery_address_state"),
          col("delivery_address_external_id"),
          col("order_total_amount"),
          col("items"),
          col("merchant_id"),

          # consumer
          col("customer_id"),
          col("language"),
          col("c.created_at").alias("consumer_created_at"),
          col("active"),
          col("customer_phone_area"),
          col("customer_phone_number"),

          # merchant
          col("m.created_at").alias("merchant_created_at"),
          col("enabled"),
          col("price_range"),
          col("average_ticket"),
          col("takeout_time"),
          col("delivery_time"),
          col("minimum_order_value"),
          col("merchant_city"),
          col("merchant_state"),
          col("merchant_country"),

          # abtest
          col("is_target"). alias("abtest_group")
      )
  )

  joining_all = datalake.dataframe2tempdataset(dataframe=joining_all, namespace="logistics", dataset=f"sales_case_kv_joining_all_{date.today()}", force_s3=True)
  return joining_all


# COMMAND ----------

# MAGIC %md
# MAGIC # 3.0 Validation

# COMMAND ----------

def validate_dataset(df):
  print('--- validating dataset saved as temp...')
  
  string_columns = []
  for s in df.dtypes:
    if s[1] == 'string' or s[1] == 'boolean':
      string_columns.append(s[0])
      
  float_columns = []
  for f in df.dtypes:
    if f[1] == 'double':
      float_columns.append(f[0])
    
  #slack_channel = '#validation_retention_analysis_report'
  validator_columns = ifoodDataQualityValidator(df, displayName="revenue_dataset")

  check_columns = validator_columns\
          .hasUniqueKey('order_id')\
          .isNeverNull('order_id')\
          .hasNumRowsGreaterThan((0))

  validator_columns.run(check_columns)
  
  if not validator_columns.allConstraintsSatisfied():
    raise Exception("Validation failed!")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.0 Save Dataset

# COMMAND ----------

def save_dataset(df):
  print('--- saving dataset on: restaurants_sandbox.case_kv')
  etl.dataframe2sandbox(
      dataframe=(df),
      namespace="restaurants",
      dataset="case_kv",
      delta_mode="batch-s3-full"
  )

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.0  Exc Pipeline

# COMMAND ----------

orders_df       = load_dataset(URLS["orders"])
consumer_df     = load_dataset(URLS["consumer"])
restaurant_df   = load_dataset(URLS["restaurant"])
abtest_df       = load_dataset(URLS["ab_test"])

orders_clean = clean_orders_df(orders_df)
consumers_clean = clean_consumer_df(consumer_df)
merchants_clean = clean_restaurant_df(restaurant_df)

joining_all = def_joining_all(orders, consumers, merchants, abtest)

validate_dataset(joining_all)
save_dataset(joining_all)
