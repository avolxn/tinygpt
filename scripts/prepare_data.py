"""Prepare deterministic Parquet training shards with PySpark."""

from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

parser = argparse.ArgumentParser(description="Prepare a deduplicated text dataset with PySpark")
parser.add_argument("--input", required=True, help="Input path or glob understood by Spark")
parser.add_argument("--input-format", choices=["text", "json", "parquet"], default="text")
parser.add_argument("--text-field", default="text", help="Text column for JSON or Parquet input")
parser.add_argument("--output", required=True, help="Output dataset directory")
parser.add_argument("--validation-fraction", type=float, default=0.01)
parser.add_argument("--min-chars", type=int, default=32)
parser.add_argument("--partitions", type=int, default=0, help="Output partitions per split; 0 keeps Spark defaults")
parser.add_argument("--master", default="", help="Optional Spark master, for example local[*] or spark://host:7077")
parser.add_argument("--write-mode", choices=["error", "overwrite"], default="error")
args = parser.parse_args()

if not 0.0 < args.validation_fraction < 1.0:
    raise ValueError("--validation-fraction must be between 0 and 1")
if args.min_chars < 1:
    raise ValueError("--min-chars must be positive")
if args.partitions < 0:
    raise ValueError("--partitions must be non-negative")

builder = SparkSession.builder.appName("tinygpt-prepare-data")
if args.master:
    builder = builder.master(args.master)
spark = builder.getOrCreate()


def load_documents() -> DataFrame:
    """Load the selected input format into a single string column named text."""
    if args.input_format == "text":
        return spark.read.text(args.input).select(F.col("value").alias("text"))
    reader = spark.read.format(args.input_format).load(args.input)
    return reader.select(F.col(args.text_field).cast("string").alias("text"))


def set_partitions(frame: DataFrame) -> DataFrame:
    return frame.repartition(args.partitions) if args.partitions > 0 else frame


documents = (
    load_documents()
    .select(F.trim(F.regexp_replace(F.col("text"), r"\r\n?", "\n")).alias("text"))
    .filter(F.col("text").isNotNull())
    .filter(F.length("text") >= args.min_chars)
    .dropDuplicates(["text"])
    .cache()
)

bucket_count = 1_000_000
validation_cutoff = round(args.validation_fraction * bucket_count)
bucketed = documents.withColumn("_split_bucket", F.pmod(F.xxhash64("text"), F.lit(bucket_count)))
train = bucketed.filter(F.col("_split_bucket") >= validation_cutoff).drop("_split_bucket")
validation = bucketed.filter(F.col("_split_bucket") < validation_cutoff).drop("_split_bucket")

train_count = train.count()
validation_count = validation.count()
spark_write_mode = "errorifexists" if args.write_mode == "error" else "overwrite"
set_partitions(train).write.mode(spark_write_mode).parquet(f"{args.output}/train")
set_partitions(validation).write.mode(spark_write_mode).parquet(f"{args.output}/validation")

manifest = spark.createDataFrame(
    [
        {
            "input": args.input,
            "input_format": args.input_format,
            "text_field": "text",
            "min_chars": args.min_chars,
            "validation_fraction": args.validation_fraction,
            "split_hash": "pmod(xxhash64(text), 1000000)",
            "train_documents": train_count,
            "validation_documents": validation_count,
        }
    ]
)
manifest.coalesce(1).write.mode(spark_write_mode).json(f"{args.output}/_manifest")

print(f"Prepared {train_count:,} train and {validation_count:,} validation documents at {args.output}")
spark.stop()
