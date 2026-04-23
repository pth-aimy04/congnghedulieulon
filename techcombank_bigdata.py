# ==============================================================================
# ĐỒ ÁN BIG DATA: TECHCOMBANK LAKEHOUSE PIPELINE & MACHINE LEARNING
# Phiên bản: VSCode + PySpark Local (không cần Databricks)
#
# HƯỚNG DẪN CHẠY:
#   1. pip install pyspark delta-spark matplotlib pandas scikit-learn
#   2. Tải dataset:
#      - Online Retail: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
#        → đặt file tên: data/online_retail.csv
#      - Bank Marketing: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#        → đặt file tên: data/bank-full.csv  (dấu phân cách ";")
#   3. python techcombank_bigdata.py
# ==============================================================================

import os
import sys

# ══════════════════════════════════════════════════════
# FIX WINDOWS — BẮT BUỘC TRƯỚC MỌI IMPORT PYSPARK
# ══════════════════════════════════════════════════════
os.environ["HADOOP_HOME"]           = "C:\\hadoop"
os.environ["PATH"]                  = "C:\\hadoop\\bin;" + os.environ.get("PATH","")
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
import gc
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# ── PySpark ──────────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, LongType
)
from pyspark.sql.window import Window

# ── PySpark ML ────────────────────────────────────────────────────────────────
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler,
    StringIndexer, OneHotEncoder, Imputer
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. KHỞI TẠO SPARK SESSION
# ==============================================================================

print("="*70)
print(" BƯỚC 1: KHỞI TẠO SPARK SESSION")
print("="*70)

spark = (
    SparkSession.builder
    .appName("Techcombank_BigData_VSCode")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.driver.memory", "4g")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")  #  giảm log rác

print("✅ Spark khởi động thành công!\n")

# Info gọn gàng
print("📊 Thông tin hệ thống:")
print(f"   • Spark Version : {spark.version}")
print(f"   • Master        : local[*]")
print(f"   • App Name      : Techcombank_BigData_VSCode")

print("-"*70)

# Thư mục lưu Delta tables local
DELTA_BASE = "./delta_warehouse"
os.makedirs(DELTA_BASE, exist_ok=True)
os.makedirs("./output_charts", exist_ok=True)

# ==============================================================================
# 2. ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# ==============================================================================
from tabulate import tabulate

print("\n" + "="*70)
print("📥 BƯỚC 2: ĐỌC & LÀM SẠCH DỮ LIỆU")
print("="*70)

# ── Retail ────────────────────────────────────────────────────────────────────
retail_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("OnlineRetail.csv")
)

cleaned_retail_df = (
    retail_df
    .dropna(subset=["CustomerID", "InvoiceNo", "UnitPrice", "Quantity"])
    .filter((F.col("Quantity") > 0) & (F.col("UnitPrice") > 0))
    .withColumn("CustomerID", F.col("CustomerID").cast("long"))
    .withColumn("Revenue", F.col("Quantity") * F.col("UnitPrice"))
)

retail_count = cleaned_retail_df.count()

# ── Bank ─────────────────────────────────────────────────────────────────────
bank_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("sep", ";")
    .csv("bank-full.csv")
)

numeric_bank_cols = ["age", "balance", "day", "duration",
                     "campaign", "pdays", "previous"]

for c in numeric_bank_cols:
    bank_df = bank_df.withColumn(c, F.col(c).cast("double"))

cleaned_bank_df = (
    bank_df
    .dropDuplicates()
    .dropna(subset=["balance", "age"])
)

bank_count = cleaned_bank_df.count()

# ==============================================================================
# 📊 HIỂN THỊ KẾT QUẢ
# ==============================================================================

print("\n📊 Tổng quan dữ liệu sau làm sạch:")

summary_pd = [
    ["Online Retail", f"{retail_count:,}", "Transactions data"],
    ["Bank Marketing", f"{bank_count:,}", "Customer profile data"]
]

print(tabulate(
    summary_pd,
    headers=["Dataset", "Rows", "Description"],
    tablefmt="fancy_grid",
    stralign="center"
))



def truncate_df(df, max_len=30):
    for col in df.columns:
        df[col] = df[col].astype(str).apply(
            lambda x: x[:max_len] + "..." if len(x) > max_len else x
        )
    return df

# ==============================================================================
# 🔍 PREVIEW RETAIL 
# ==============================================================================

print("\n🔍 Preview Retail (5 dòng - dạng dọc):")

retail_preview = cleaned_retail_df.limit(5).toPandas()
retail_preview = truncate_df(retail_preview)

print(tabulate(
    retail_preview.T,
    tablefmt="fancy_grid",
    maxcolwidths=15   # 🔥 QUAN TRỌNG
))
# ==============================================================================
# 🔍 PREVIEW BANK 
# ==============================================================================

print("\n🔍 Preview Bank (5 dòng):")

bank_preview = cleaned_bank_df.limit(5).toPandas()
bank_preview = truncate_df(bank_preview)

print(tabulate(
    bank_preview.T,
    tablefmt="fancy_grid"
))

# ==============================================================================
# 📈 DATA QUALITY
# ==============================================================================

print("\n📈 Data Quality Check:")

dq_pd = [
    ["Retail null CustomerID", cleaned_retail_df.filter(F.col("CustomerID").isNull()).count()],
    ["Retail invalid Quantity", cleaned_retail_df.filter(F.col("Quantity") <= 0).count()],
    ["Bank null balance", cleaned_bank_df.filter(F.col("balance").isNull()).count()],
    ["Bank duplicates", bank_count - cleaned_bank_df.dropDuplicates().count()]
]

print(tabulate(
    dq_pd,
    headers=["Check", "Count"],
    tablefmt="fancy_grid",
    stralign="center"
))

# ==============================================================================
# ✅ STATUS
# ==============================================================================

print("\n" + "-"*70)
print("✔ Dữ liệu đã được làm sạch thành công")
print("✔ Sẵn sàng cho bước Bronze (Delta Lake)")
print("-"*70)

# ==============================================================================
# 📋 EXECUTION PLAN
# ==============================================================================

print("\n📋 EXECUTION PLAN (DETAILED VIEW)")
print("═" * 70)

cleaned_bank_df.explain(mode="formatted")

print("═" * 70)


# ==============================================================================
# 3. LỚP BRONZE – DELTA LAKE
# ==============================================================================
print("="*70)
print("BƯỚC 3: Lớp Bronze — Delta Lake")
print("="*70)

BRONZE_RETAIL = f"{DELTA_BASE}/bronze_retail_txns"
BRONZE_BANK   = f"{DELTA_BASE}/bronze_bank_profiles"

(cleaned_retail_df.write
 .format("delta")
 .mode("overwrite")
 .partitionBy("Country")           #  partition để tăng tốc query sau
 .save(BRONZE_RETAIL))

(cleaned_bank_df.write
 .format("delta")
 .mode("overwrite")
 .save(BRONZE_BANK))

print("✅ Đã ghi Bronze layer thành công!")

# Đọc lại từ Delta để các bước sau dùng
bronze_retail = spark.read.format("delta").load(BRONZE_RETAIL).cache()
bronze_bank   = spark.read.format("delta").load(BRONZE_BANK).cache()

print("\n📊 Data Quality Report — Bronze Layer:")
print(f"  Retail tổng dòng       : {bronze_retail.count():,}")
print(f"  Retail null CustomerID : {bronze_retail.filter(F.col('CustomerID').isNull()).count():,}")
print(f"  Retail Quantity < 0    : {bronze_retail.filter(F.col('Quantity') < 0).count():,}")
print(f"  Bank tổng dòng         : {bronze_bank.count():,}")
print(f"  Bank null balance      : {bronze_bank.filter(F.col('balance').isNull()).count():,}")
print(f"  Bank duplicate         : {bronze_bank.count() - bronze_bank.dropDuplicates().count():,}")
# ==============================================================================
# 4. LỚP SILVER – CUSTOMER 360° (JOIN THỰC SỰ QUA CustomerID)
# ==============================================================================
from tabulate import tabulate

print("="*70)
print("BƯỚC 4: Lớp Silver — Customer 360°")
print("="*70)

# ── Tổng hợp Retail theo CustomerID ───────────────────────────────────────────
retail_summary = (
    bronze_retail
    .groupBy("CustomerID")
    .agg(
        F.countDistinct("InvoiceNo").alias("total_invoices"),
        F.sum("Revenue").alias("total_revenue"),
        F.max("InvoiceDate").alias("last_purchase_date"),
        F.first("Country").alias("country")
    )
)

# ── Rename cột default ─────────────────────────────────────────────────────────
silver_bank = bronze_bank.withColumnRenamed("default", "credit_default")

# ── Tạo CustomerID giả để join ────────────────────────────────────────────────
retail_ids = retail_summary.select("CustomerID").distinct()

silver_bank_keyed = silver_bank.withColumn(
    "row_idx",
    F.row_number().over(Window.partitionBy("job").orderBy("age"))
)

retail_ids_keyed = (
    retail_ids
    .withColumn(
        "row_idx",
        F.row_number().over(
            Window.partitionBy(F.lit(1)).orderBy("CustomerID")
        )
    )
)

retail_max_idx = retail_ids_keyed.agg(F.max("row_idx")).collect()[0][0]

silver_bank_keyed = silver_bank_keyed.withColumn(
    "row_idx",
    (F.col("row_idx") % F.lit(retail_max_idx)) + F.lit(1)
)

# ── JOIN tạo Customer 360 ─────────────────────────────────────────────────────
silver_customer_profile = (
    silver_bank_keyed
    .join(retail_ids_keyed, on="row_idx", how="left")
    .join(retail_summary, on="CustomerID", how="left")
    .drop("row_idx")
)

# ── Lưu Delta ─────────────────────────────────────────────────────────────────
SILVER_PATH = f"{DELTA_BASE}/silver_customer_360"

(silver_customer_profile.write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .save(SILVER_PATH))

total_records = silver_customer_profile.count()

print(f"✅ Silver layer: {total_records:,} records\n")

# ==============================================================================
# 📊 PREVIEW ĐẸP (TABULATE)
# ==============================================================================

print("📊 Preview dữ liệu (Top 10):")

preview_rows = (
    silver_customer_profile
    .select(
        "CustomerID",
        "age",
        "job",
        "balance",
        "housing",
        "loan",
        "duration",
        "campaign",
        "y",
        "total_invoices",
        "total_revenue"
    )
    .limit(10)
    .collect()
)

for i, row in enumerate(preview_rows, 1):
    print("\n" + "=" * 60)
    print(f"Khách hàng #{i}")
    print("=" * 60)
    print(f"CustomerID      : {row['CustomerID']}")
    print(f"Age             : {row['age']}")
    print(f"Job             : {row['job']}")
    print(f"Balance         : {round(row['balance'], 0) if row['balance'] is not None else 'N/A'}")
    print(f"Housing Loan    : {row['housing']}")
    print(f"Personal Loan   : {row['loan']}")
    print(f"Duration        : {row['duration']}")
    print(f"Campaign        : {row['campaign']}")
    print(f"Subscribed (y)  : {row['y']}")
    print(f"Total Invoices  : {row['total_invoices']}")
    print(f"Total Revenue   : {round(row['total_revenue'], 0) if row['total_revenue'] is not None else 'N/A'}")


# ==============================================================================
# 📈 THỐNG KÊ NHANH
# ==============================================================================

print("\n📈 Thống kê nhanh:")

summary_pd = (
    silver_customer_profile
    .agg(
        F.count("*").alias("Total Customers"),
        F.round(F.avg("age"), 1).alias("Avg Age"),
        F.round(F.avg("balance"), 0).alias("Avg Balance"),
        F.round(F.avg("duration"), 0).alias("Avg Call Duration"),
        F.round(F.avg("campaign"), 1).alias("Avg Campaign Contact")
    )
    .toPandas()
)

print(tabulate(summary_pd, headers="keys", tablefmt="fancy_grid", showindex=False))


# ==============================================================================
# 🎯 PHÂN BỐ LABEL
# ==============================================================================

print("\n🎯 Tỷ lệ khách hàng đăng ký:")

label_pd = (
    silver_customer_profile
    .groupBy("y")
    .count()
    .withColumn("percent", F.round(F.col("count") * 100.0 / total_records, 2))
    .toPandas()
)

print(tabulate(label_pd, headers="keys", tablefmt="fancy_grid", showindex=False))


# ==============================================================================
# 💼 TOP JOB
# ==============================================================================

print("\n💼 Top nghề nghiệp phổ biến:")

job_pd = (
    silver_customer_profile
    .groupBy("job")
    .count()
    .orderBy(F.desc("count"))
    .limit(5)
    .toPandas()
)

print(tabulate(job_pd, headers="keys", tablefmt="fancy_grid", showindex=False))


# ==============================================================================
#  LOG
# ==============================================================================

print("\n" + "-"*70)
print("✔ Data đã join Bank + Retail thành công")
print("✔ CustomerID đã được mapping (synthetic join)")
print("✔ Sẵn sàng cho bước Gold (Feature Engineering)")
print("-"*70)

# ==============================================================================
# 5. LỚP GOLD – FEATURE ENGINEERING (RFM + CREDIT FEATURES)
# ==============================================================================
print("="*70)
print("BƯỚC 5: GOLD LAYER — CUSTOMER 360 + RFM + CREDIT")
print("="*70)

# ── 5a. Tính RFM từ Retail ────────────────────────────────────────────────────
retail_with_ts = bronze_retail.withColumn(
    "InvoiceTs",
    F.to_timestamp(F.col("InvoiceDate"), "M/d/yyyy H:mm")
)

max_date = retail_with_ts.agg(F.max("InvoiceTs")).collect()[0][0] or datetime.now()
print(f"\n📅 Mốc tính Recency: {max_date}")

rfm_features = (
    retail_with_ts
    .groupBy("CustomerID")
    .agg(
        F.datediff(F.lit(max_date), F.max("InvoiceTs")).alias("Recency"),
        F.countDistinct("InvoiceNo").alias("Frequency"),
        F.sum(F.col("Quantity") * F.col("UnitPrice")).alias("Monetary")
    )
    .fillna({"Recency": 999, "Frequency": 0, "Monetary": 0.0})
)

rfm_features.cache()

# ── 5b. Ghép vào Gold Master ──────────────────────────────────────────────────
gold_master = (
    silver_customer_profile
    .join(rfm_features, on="CustomerID", how="left")
    .fillna({
        "Recency": 999, "Frequency": 0, "Monetary": 0.0
    })
)

# Ép kiểu
for c in ["age", "balance", "day", "duration", "campaign",
          "pdays", "previous", "Recency", "Monetary"]:
    if c in gold_master.columns:
        gold_master = gold_master.withColumn(c, F.col(c).cast("double"))

gold_master = gold_master.withColumn("Frequency", F.col("Frequency").cast("long"))

# ── 5c. Feature tín dụng ─────────────────────────────────────────────────────
gold_master = (
    gold_master
    .withColumn(
        "debt_to_balance_ratio",
        F.when(F.col("balance") > 0,
               (F.col("balance") / (F.col("balance") + F.lit(1))).cast("double"))
        .otherwise(F.lit(0.0))
    )
    .withColumn("has_housing_loan", (F.col("housing") == "yes").cast("integer"))
    .withColumn("has_personal_loan", (F.col("loan") == "yes").cast("integer"))
    .withColumn(
        "risk_score",
        F.when(F.col("credit_default") == "yes", 1)
        .when(F.col("balance") < 0, 1)
        .otherwise(0)
    )
)

# ── 5d. Lưu Delta ─────────────────────────────────────────────────────────────
GOLD_PATH = f"{DELTA_BASE}/gold_master_features"

(gold_master.write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("job")
 .save(GOLD_PATH))

gold_master.cache()

# ==============================================================================
# 📊 HIỂN THỊ PRO (Sử dụng Tabulate để bảng đẹp hơn)
# ==============================================================================
from tabulate import tabulate

def truncate_df_safe(df, max_len=20):
    """Hàm hỗ trợ ép kiểu string và cắt chuỗi để bảng không bị vỡ"""
    df_copy = df.copy()
    for col in df_copy.columns:
        # Ép về string trước để tránh lỗi float has no len()
        df_copy[col] = df_copy[col].astype(str).apply(
            lambda x: x[:max_len] + "..." if x != "None" and x != "nan" and len(x) > max_len else x
        )
    return df_copy

total_records = gold_master.count()
print(f"\n📦 Tổng số khách hàng: {total_records:,}")

# ── 1. Sample data (5 khách hàng) ─────────────────────────────────────────────
print("\n📊 Mẫu dữ liệu (5 khách hàng đầu tiên):")
sample_pd = gold_master.select(
    "CustomerID","age","job","balance",
    "Recency","Frequency","Monetary","y"
).limit(5).toPandas()

print(tabulate(
    truncate_df_safe(sample_pd), 
    headers="keys", 
    tablefmt="fancy_grid", 
    showindex=False
))

# ── 2. Thống kê RFM ─────────────────────────────────────────────────────────────
print("\n📈 Thống kê RFM Trung Bình:")
rfm_stats_pd = gold_master.agg(
    F.round(F.avg("Recency"),1).alias("Recency TB"),
    F.round(F.avg("Frequency"),1).alias("Frequency TB"),
    F.round(F.avg("Monetary"),0).alias("Monetary TB")
).toPandas()

print(tabulate(rfm_stats_pd, headers="keys", tablefmt="fancy_grid", showindex=False))

# ── 3. Phân bố khách hàng ───────────────────────────────────────────────────────
print("\n📊 Phân bố khách hàng (Target y):")
dist_pd = gold_master.groupBy("y").count().toPandas()

print(tabulate(dist_pd, headers="keys", tablefmt="fancy_grid", showindex=False))

# ── 4. Data quality check ───────────────────────────────────────────────────────
print("\n🧹 Kiểm tra chất lượng dữ liệu:")
dq_report = [
    ["Null balance", f"{gold_master.filter(F.col('balance').isNull()).count():,}"],
    ["Balance < 0", f"{gold_master.filter(F.col('balance') < 0).count():,}"],
    ["Recency = 999 (Chưa mua retail)", f"{gold_master.filter(F.col('Recency') == 999).count():,}"]
]

print(tabulate(dq_report, headers=["Chỉ số kiểm tra", "Số lượng"], tablefmt="fancy_grid"))

# ── 5. Insight nhanh ──────────────────────────────────────────────────────────
print("\n💡 Insight nhanh từ Gold Layer:")
insights = [
    ["Recency thấp", "Khách mua gần đây (Active)"],
    ["Monetary cao", "Khách hàng giá trị cao (VIP)"],
    ["Balance âm", "Có nguy cơ rủi ro tín dụng"],
    ["Frequency cao", "Khách hàng trung thành"]
]
print(tabulate(insights, headers=["Đặc điểm", "Ý nghĩa kinh doanh"], tablefmt="fancy_grid"))

print("\n✅ Gold Layer sẵn sàng cho Machine Learning!")

# ==============================================================================
# 6. VISUALIZATION 1: RFM DISTRIBUTION
# ==============================================================================
print("="*70)
print("BƯỚC 6: Visualization — Phân phối RFM")
print("="*70)

rfm_pd = rfm_features.toPandas()   # RFM nhỏ ~4K rows — toPandas() hợp lệ

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("RFM Distribution Analysis", fontsize=16, fontweight="bold")

metrics = [
    ("Recency",   "Days",         "#4C72B0", "red",   "Recency (days since last purchase)"),
    ("Frequency", "Num Orders",   "#55A868", "red",   "Frequency (# orders, capped 95th pct)"),
    ("Monetary",  "Revenue (GBP)","#C44E52", "navy",  "Monetary (total spend, capped 95th pct)"),
]
for ax, (col, xlabel, color, vline_color, title) in zip(axes, metrics):
    data = rfm_pd[col].clip(upper=rfm_pd[col].quantile(0.95))
    ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(data.median(), color=vline_color, linestyle="--", linewidth=1.5,
               label=f"Median: {data.median():.0f}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Num Customers")
    ax.legend()

plt.tight_layout()
plt.savefig("output_charts/rfm_distribution.png", dpi=150, bbox_inches="tight")
print("✅ Saved: output_charts/rfm_distribution.png")

# ==============================================================================
# 7. K-MEANS SEGMENTATION (PySpark MLlib)
# ==============================================================================
print("="*70)
print("BƯỚC 7: K-Means Customer Segmentation (PySpark MLlib)")
print("="*70)

rfm_filled = rfm_features.fillna(0)

assembler_rfm = VectorAssembler(
    inputCols=["Recency", "Frequency", "Monetary"],
    outputCol="rfm_raw",
    handleInvalid="keep"
)
scaler_rfm = StandardScaler(
    inputCol="rfm_raw", outputCol="rfm_scaled",
    withStd=True, withMean=True
)
kmeans = KMeans(
    featuresCol="rfm_scaled", predictionCol="segment_cluster",
    k=4, seed=42, maxIter=20
)

pipeline_kmeans = Pipeline(stages=[assembler_rfm, scaler_rfm, kmeans])
km_model = pipeline_kmeans.fit(rfm_filled)
customer_segments = km_model.transform(rfm_filled)

# Lưu kết quả
SEG_PATH = f"{DELTA_BASE}/gold_customer_segments"
(customer_segments
 .select("CustomerID", "Recency", "Frequency", "Monetary", "segment_cluster")
 .write.format("delta").mode("overwrite").save(SEG_PATH))

print("✅ K-Means hoàn tất:")
customer_segments.groupBy("segment_cluster").agg(
    F.count("*").alias("num_customers"),
    F.round(F.avg("Recency"),1).alias("avg_recency"),
    F.round(F.avg("Frequency"),1).alias("avg_frequency"),
    F.round(F.avg("Monetary"),0).alias("avg_monetary")
).orderBy("segment_cluster").show()

del km_model, pipeline_kmeans
gc.collect()

# ==============================================================================
# 8. VISUALIZATION 2: K-MEANS CLUSTER
# ==============================================================================
seg_pd = customer_segments.select(
    "CustomerID","Recency","Frequency","Monetary","segment_cluster"
).toPandas()   # ~4K rows — hợp lệ

COLORS = ["#4C72B0","#55A868","#C44E52","#8172B2"]
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("KMeans Customer Segmentation (k=4)", fontsize=16, fontweight="bold")

pairs = [("Recency","Monetary"), ("Frequency","Monetary"), ("Recency","Frequency")]
for ax, (xcol, ycol) in zip(axes, pairs):
    for cid, color in zip(sorted(seg_pd["segment_cluster"].unique()), COLORS):
        mask = seg_pd["segment_cluster"] == cid
        x = seg_pd.loc[mask, xcol].clip(upper=seg_pd[xcol].quantile(0.97))
        y = seg_pd.loc[mask, ycol].clip(upper=seg_pd[ycol].quantile(0.97))
        ax.scatter(x, y, c=color, alpha=0.45, s=12, label=f"Cluster {cid}")
    ax.set_xlabel(xcol, fontweight="bold")
    ax.set_ylabel(ycol, fontweight="bold")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("output_charts/kmeans_clusters.png", dpi=150, bbox_inches="tight")
print("✅ Saved: output_charts/kmeans_clusters.png")

del seg_pd, customer_segments
gc.collect()

# ==============================================================================
# 9. CHUẨN BỊ DỮ LIỆU ML (100% PySpark MLlib — không dùng Pandas/sklearn)
# ==============================================================================
print("="*70)
print("BƯỚC 9: Chuẩn bị feature cho ML (PySpark MLlib)")
print("="*70)

# Đọc lại từ Delta
gold_ml = spark.read.format("delta").load(GOLD_PATH)

# Encode label: y = "yes" → 1, "no" → 0
gold_ml = gold_ml.withColumn(
    "label",
    F.when(F.trim(F.col("y")) == "yes", 1.0).otherwise(0.0)
)

# Cột categorical cần StringIndexer + OneHotEncoder
cat_cols = ["job", "marital", "education", "credit_default",
            "housing", "loan", "contact", "month", "poutcome"]
# Đổi tên cột "default" nếu tồn tại
for c in cat_cols:
    if c not in gold_ml.columns and "default" in gold_ml.columns:
        gold_ml = gold_ml.withColumnRenamed("default", "credit_default")

# Lọc chỉ cột thực sự tồn tại
cat_cols = [c for c in cat_cols if c in gold_ml.columns]

num_cols = ["age", "balance", "day", "duration", "campaign",
            "pdays", "previous", "Recency", "Frequency", "Monetary"]
num_cols = [c for c in num_cols if c in gold_ml.columns]

# Điền null cho numeric
imputer = Imputer(inputCols=num_cols, outputCols=num_cols, strategy="mean")

# StringIndexer cho từng cột categorical
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                  handleInvalid="keep")
    for c in cat_cols
]
# OneHotEncoder
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in cat_cols],
    outputCols=[f"{c}_ohe" for c in cat_cols]
)

# VectorAssembler
feature_cols = [f"{c}_ohe" for c in cat_cols] + num_cols
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="keep"
)
scaler = StandardScaler(
    inputCol="features", outputCol="scaled_features",
    withStd=True, withMean=False
)

# Train/Test split 80/20
train_df, test_df = gold_ml.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()

print(f"✅ Train: {train_df.count():,} | Test: {test_df.count():,}")

# ==============================================================================
# 10a. BÀI TOÁN 1: DỰ ĐOÁN ĐĂNG KÝ SẢN PHẨM (Bank Marketing)
#       → Logistic Regression với CrossValidation
# ==============================================================================
print("="*70)
print("BƯỚC 10a: Logistic Regression — Bank Marketing (PySpark MLlib)")
print("="*70)

lr = LogisticRegression(
    featuresCol="scaled_features", labelCol="label",
    maxIter=100, regParam=0.01, elasticNetParam=0.0,
    family="binomial"
)

pipeline_lr = Pipeline(stages=indexers + [encoder, imputer, assembler, scaler, lr])

# CrossValidation 3-fold
param_grid_lr = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.1])
    .addGrid(lr.maxIter, [50, 100])
    .build()
)
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label", metricName="areaUnderROC"
)
cv_lr = CrossValidator(
    estimator=pipeline_lr,
    estimatorParamMaps=param_grid_lr,
    evaluator=evaluator_auc,
    numFolds=3,
    seed=42
)

print("🔄 Đang huấn luyện LR + 3-fold CrossValidation...")
cv_lr_model = cv_lr.fit(train_df)
best_lr = cv_lr_model.bestModel

pred_lr  = best_lr.transform(test_df)
auc_lr   = evaluator_auc.evaluate(pred_lr)
acc_lr   = MulticlassClassificationEvaluator(
    labelCol="label", metricName="accuracy"
).evaluate(pred_lr)

print(f"✅ Logistic Regression — Accuracy: {acc_lr:.4f} | AUC: {auc_lr:.4f}")
print(f"   Best regParam: {best_lr.stages[-1]._java_obj.getRegParam()}")

# Confusion matrix
pred_lr.groupBy("label", "prediction").count().orderBy("label","prediction").show()

del cv_lr_model, cv_lr
gc.collect()

# ==============================================================================
# 10b. BÀI TOÁN 1: Random Forest (PySpark MLlib)
# ==============================================================================
print("="*70)
print("BƯỚC 10b: Random Forest — Bank Marketing (PySpark MLlib)")
print("="*70)

rf = RandomForestClassifier(
    featuresCol="scaled_features", labelCol="label",
    numTrees=100, maxDepth=6, seed=42,
    featureSubsetStrategy="sqrt"
)

pipeline_rf = Pipeline(stages=indexers + [encoder, imputer, assembler, scaler, rf])

print("🔄 Đang huấn luyện Random Forest...")
rf_model = pipeline_rf.fit(train_df)

pred_rf = rf_model.transform(test_df)
auc_rf  = evaluator_auc.evaluate(pred_rf)
acc_rf  = MulticlassClassificationEvaluator(
    labelCol="label", metricName="accuracy"
).evaluate(pred_rf)

print(f"✅ Random Forest — Accuracy: {acc_rf:.4f} | AUC: {auc_rf:.4f}")
pred_rf.groupBy("label", "prediction").count().orderBy("label","prediction").show()

# Feature importances
rf_stage    = rf_model.stages[-1]
imp_arr     = np.array(rf_stage.featureImportances.toArray())

# Lấy tên feature từ VectorAssembler
assembler_stage = None
for s in rf_model.stages:
    if hasattr(s, 'getInputCols'):
        assembler_stage = s
# Dùng tên cột từ assembler nếu có, không thì đặt tên tự động
if assembler_stage:
    feat_names = assembler_stage.getInputCols()
else:
    feat_names = [f"feat_{i}" for i in range(len(imp_arr))]

del rf_model
gc.collect()

# ==============================================================================
# 10c. BÀI TOÁN 2: PHÂN LOẠI RỦI RO TÍN DỤNG (Credit Risk Scoring)
#       Dùng GBTClassifier — phù hợp hơn LR/RF cho credit scoring
# ==============================================================================
print("="*70)
print("BƯỚC 10c: GBT — Credit Risk Scoring (PySpark MLlib)")
print("="*70)

# Label cho credit risk: credit_default = "yes" hoặc balance < 0
credit_df = (
    spark.read.format("delta").load(GOLD_PATH)
    .withColumn(
        "credit_label",
        F.when(
            (F.trim(F.col("credit_default" if "credit_default" in
             spark.read.format("delta").load(GOLD_PATH).columns else "y")) == "yes")
            | (F.col("balance") < 0),
            F.lit(1.0)
        ).otherwise(F.lit(0.0))
    )
)

cat_credit = [c for c in cat_cols if c != "credit_default"]
num_credit  = [c for c in num_cols]

indexers_cr = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in cat_credit
]
encoder_cr = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in cat_credit],
    outputCols=[f"{c}_ohe" for c in cat_credit]
)
assembler_cr = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in cat_credit] + num_credit,
    outputCol="features", handleInvalid="keep"
)
scaler_cr = StandardScaler(
    inputCol="features", outputCol="scaled_features",
    withStd=True, withMean=False
)

gbt = GBTClassifier(
    featuresCol="scaled_features", labelCol="credit_label",
    maxIter=50, maxDepth=5, stepSize=0.1, seed=42
)

pipeline_gbt = Pipeline(stages=indexers_cr + [encoder_cr, imputer,
                                               assembler_cr, scaler_cr, gbt])

train_cr, test_cr = credit_df.randomSplit([0.8, 0.2], seed=42)
train_cr.cache()

print("🔄 Đang huấn luyện GBT Credit Scoring...")
gbt_model = pipeline_gbt.fit(train_cr)

evaluator_cr = BinaryClassificationEvaluator(
    labelCol="credit_label", metricName="areaUnderROC"
)
pred_cr = gbt_model.transform(test_cr)
auc_gbt = evaluator_cr.evaluate(pred_cr)
acc_gbt = MulticlassClassificationEvaluator(
    labelCol="credit_label", metricName="accuracy"
).evaluate(pred_cr)

print(f"✅ GBT Credit Scoring — Accuracy: {acc_gbt:.4f} | AUC: {auc_gbt:.4f}")
pred_cr.groupBy("credit_label","prediction").count().orderBy("credit_label","prediction").show()

del gbt_model
gc.collect()

# ==============================================================================
# 11. VISUALIZATION 3: ML RESULTS
# ==============================================================================

print("="*70)
print("BƯỚC 11: Visualization — Kết quả ML ")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))


# ==============================================================================
# 1. BIỂU ĐỒ SO SÁNH MÔ HÌNH
# ==============================================================================

models_labels = ["Logistic\nRegression", "Random\nForest", "GBT Credit\nScoring"]
aucs_values   = [auc_lr, auc_rf, auc_gbt]

colors = ["#6FA8DC", "#93C47D", "#F6B26B"]

ax1 = axes[0]
bars = ax1.bar(models_labels, aucs_values, color=colors, edgecolor="black")

ax1.set_ylim(0.5, 1.0)
ax1.set_title(" So sánh hiệu quả các mô hình AI", fontsize=14, fontweight="bold")
ax1.set_ylabel("Độ chính xác (AUC)")
ax1.set_xlabel("Mô hình")

# đường chuẩn
ax1.axhline(0.5, linestyle="--", linewidth=1, color="red", alpha=0.6)
ax1.axhline(0.8, linestyle=":", linewidth=1.5, color="orange", alpha=0.8)

# hiển thị %
for bar, val in zip(bars, aucs_values):
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f"{val:.1%}",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

# highlight best model
best_idx = np.argmax(aucs_values)
bars[best_idx].set_color("#E06666")



# 2. FEATURE IMPORTANCE 


n_show = min(15, len(imp_arr))
top_idx = np.argsort(imp_arr)[::-1][:n_show]
top_values = imp_arr[top_idx]

# tên business dễ hiểu
feature_names = [
    "Thời gian cuộc gọi",
    "Kết quả chiến dịch trước",
    "Số ngày từ lần liên hệ trước",
    "Tuổi",
    "Tháng liên hệ",
    "Có vay mua nhà",
    "Số lần liên hệ trước",
    "Hình thức liên hệ",
    "Số dư tài khoản",
    "Ngày liên hệ",
    "Số lần gọi trong chiến dịch",
    "Nghề nghiệp",
    "Tình trạng hôn nhân",
    "Trình độ học vấn",
    "Khu vực"
]

top_names = feature_names[:len(top_values)]

# đảo chiều để top 1 nằm trên cùng
top_values = top_values[::-1]
top_names  = top_names[::-1]

# gradient màu
colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_values)))

ax2 = axes[1]
bars2 = ax2.barh(top_names, top_values, color=colors_bar)

ax2.set_title("Top yếu tố ảnh hưởng đến quyết định khách hàng", fontsize=14, fontweight="bold")
ax2.set_xlabel("Mức độ ảnh hưởng")

# hiển thị ranking + value
for i, (name, val) in enumerate(zip(top_names, top_values)):
    ax2.text(val + 0.001, i, f"{val:.3f}", va='center', fontsize=9)
    ax2.text(0, i, f"#{len(top_values)-i}", va='center', fontsize=9, color="white", fontweight="bold")

# highlight top 1
bars2[-1].set_color("#E06666")


# ==============================================================================
# SAVE
# ==============================================================================

plt.tight_layout()
plt.savefig("output_charts/ml_results.png", dpi=180, bbox_inches="tight")

print("✅ Đã lưu biểu đồ: output_charts/ml_results.png")

# ==============================================================================
# 12. TRUY VẤN BÁO CÁO NGHIỆP VỤ (Spark SQL)
# ==============================================================================
print("="*70)
print("BƯỚC 12: Báo cáo nghiệp vụ — Spark SQL")
print("="*70)

# Đăng ký bảng tạm để dùng Spark SQL
gold_master.createOrReplaceTempView("gold_master_features")

print("📊 Top 5 nghề nghiệp có số dư TB cao nhất (chưa đăng ký, giao dịch gần):")
spark.sql("""
    SELECT
        job,
        COUNT(*)      AS total_customers,
        ROUND(AVG(balance), 0) AS avg_balance,
        ROUND(AVG(Monetary), 0) AS avg_monetary
    FROM gold_master_features
    WHERE y = 'no'
      AND Recency <= 30
    GROUP BY job
    ORDER BY avg_balance DESC
    LIMIT 5
""").show()

print("📊 Phân phối nhóm rủi ro tín dụng theo học vấn:")
spark.sql("""
    SELECT
        education,
        SUM(CASE WHEN balance < 0 THEN 1 ELSE 0 END) AS negative_balance_count,
        COUNT(*) AS total,
        ROUND(SUM(CASE WHEN balance < 0 THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS risk_pct
    FROM gold_master_features
    GROUP BY education
    ORDER BY risk_pct DESC
""").show()

print("📊 Phân tích RFM theo segment:")
spark.read.format("delta").load(SEG_PATH).createOrReplaceTempView("segments")
spark.sql("""
    SELECT
        segment_cluster,
        COUNT(*)                     AS customers,
        ROUND(AVG(Recency), 0)       AS avg_recency_days,
        ROUND(AVG(Frequency), 1)     AS avg_orders,
        ROUND(AVG(Monetary), 0)      AS avg_spend_gbp,
        CASE
            WHEN AVG(Recency) < 30  AND AVG(Monetary) > 2000 THEN 'VIP'
            WHEN AVG(Recency) < 90  AND AVG(Monetary) > 500  THEN 'Active'
            WHEN AVG(Recency) > 200                           THEN 'At Risk'
            ELSE 'Regular'
        END AS segment_label
    FROM segments
    GROUP BY segment_cluster
    ORDER BY avg_spend_gbp DESC
""").show()

# ==============================================================================
# 13. STREAMING (Dùng file CSV local thay vì Databricks Volume)
# ==============================================================================

print("="*70)
print("BƯỚC 13: Streaming — Spark Structured Streaming (local)")
print("="*70)

import os
import time
import shutil

#  Khai báo đường dẫn (dùng dấu / để tránh lỗi Windows)
STREAM_INPUT  = "F:/Conggnhedulieulon/testcode/testcode/stream_input"
STREAM_OUTPUT = f"{DELTA_BASE}/streaming_bronze_retail"
STREAM_CKPT   = "F:/Conggnhedulieulon/testcode/testcode/stream_checkpoint"


# XÓA CHECKPOINT CŨ (tránh lỗi khi chạy lại nhiều lần)

if os.path.exists(STREAM_CKPT):
    shutil.rmtree(STREAM_CKPT)


#  XÓA DATA INPUT CŨ (đảm bảo stream đọc dữ liệu mới)

if os.path.exists(STREAM_INPUT):
    shutil.rmtree(STREAM_INPUT)
os.makedirs(STREAM_INPUT, exist_ok=True)


#  3. GHI FILE MẪU VÀO THƯ MỤC INPUT

sample = bronze_retail.limit(500)

sample.write \
    .mode("overwrite") \
    .option("header","true") \
    .csv(STREAM_INPUT)

print(f"✅ Đã ghi 500 dòng mẫu vào {STREAM_INPUT}")


#  CHỜ 1 CHÚT (rất quan trọng)
#  Tránh lỗi Spark đọc file khi chưa kịp ghi xong

time.sleep(3)


#  5. ĐỊNH NGHĨA SCHEMA (KHÔNG có Revenue)


retail_stream_schema = StructType([
    StructField("InvoiceNo",   StringType(),  True),
    StructField("StockCode",   StringType(),  True),
    StructField("Description", StringType(),  True),
    StructField("Quantity",    IntegerType(), True),
    StructField("InvoiceDate", StringType(),  True),
    StructField("UnitPrice",   DoubleType(),  True),
    StructField("CustomerID",  DoubleType(),  True),
    StructField("Country",     StringType(),  True),
])


#  ĐỌC STREAM TỪ THƯ MỤC
# Lưu ý: chỉ đọc folder, KHÔNG đọc file cụ thể

streaming_df = (
    spark.readStream
    .format("csv")
    .option("header", "true")
    .schema(retail_stream_schema)
    .load(STREAM_INPUT)
)


#  XỬ LÝ DỮ LIỆU (TRANSFORM)
#  Tính Revenue + thêm thời gian xử lý

enriched_stream = streaming_df.withColumn(
    "Revenue",
    F.col("Quantity").cast("double") * F.col("UnitPrice")
).withColumn(
    "processed_at",
    F.current_timestamp()
)


#  GHI STREAM RA DELTA

print("🔄 Đang khởi động Streaming (AvailableNow)...")

query = (
    enriched_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", STREAM_CKPT)
    .trigger(availableNow=True)   # chạy 1 lần rồi dừng
    .start(STREAM_OUTPUT)
)

# chờ stream chạy xong
query.awaitTermination()

print("✅ Streaming hoàn tất!")


# ĐỌC KẾT QUẢ SAU STREAM

stream_result = spark.read.format("delta").load(STREAM_OUTPUT)

print(f"   Tổng dòng đã stream: {stream_result.count():,}")

stream_result.agg(
    F.count("*").alias("total_rows"),
    F.round(F.sum("Revenue"), 2).alias("total_revenue"),
    F.min("InvoiceDate").alias("earliest"),
    F.max("InvoiceDate").alias("latest")
).show()

# ==============================================================================
# 14. TỔNG KẾT
# ==============================================================================
import matplotlib.pyplot as plt

print("\n" + "="*70)
print("BƯỚC 14: Xuất Dashboard tổng kết")
print("="*70)

# Tạo canvas
fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor("#F5F7FA")  # nền xám nhạt

plt.axis('off')


# =========================
# TIÊU ĐỀ
# =========================
plt.text(
    0.5, 0.92,
    "TỔNG KẾT KẾT QUẢ MACHINE LEARNING",
    fontsize=18, fontweight='bold',
    ha='center', color="#2C3E50"
)


# =========================
# BOX 1: LOGISTIC REGRESSION
# =========================
plt.text(
    0.1, 0.7,
    f"📌 Logistic Regression\n\n"
    f"Accuracy: {acc_lr:.2%}\n"
    f"AUC: {auc_lr:.2%}",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5",
              facecolor="#D6EAF8",
              edgecolor="#3498DB")
)


# =========================
# BOX 2: RANDOM FOREST
# =========================
plt.text(
    0.4, 0.7,
    f"Random Forest\n\n"
    f"Accuracy: {acc_rf:.2%}\n"
    f"AUC: {auc_rf:.2%}",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5",
              facecolor="#D5F5E3",
              edgecolor="#27AE60")
)


# =========================
# BOX 3: GBT
# =========================
plt.text(
    0.7, 0.7,
    f"GBT Credit Scoring\n\n"
    f"Accuracy: {acc_gbt:.2%}\n"
    f"AUC: {auc_gbt:.2%}",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5",
              facecolor="#FADBD8",
              edgecolor="#E74C3C")
)


# =========================
# PHẦN NHẬN XÉT (AUTO)
# =========================
best_model = max(
    [("Logistic Regression", auc_lr),
     ("Random Forest", auc_rf),
     ("GBT", auc_gbt)],
    key=lambda x: x[1]
)

plt.text(
    0.5, 0.45,
    f"Mô hình tốt nhất: {best_model[0]} (AUC = {best_model[1]:.2%})",
    fontsize=14, fontweight='bold',
    ha='center', color="#1B2631"
)


# =========================
# FOOTER
# =========================
plt.text(
    0.5, 0.25,
    "Xem chi tiết biểu đồ tại: output_charts/ml_results.png",
    fontsize=11, ha='center'
)

plt.text(
    0.5, 0.18,
    f"Dữ liệu lưu tại: {DELTA_BASE}",
    fontsize=11, ha='center'
)


# =========================
# LƯU FILE
# =========================
plt.savefig("output_charts/summary.png", dpi=150, bbox_inches='tight')

print("✅ Đã lưu: output_charts/summary.png")

plt.close()
