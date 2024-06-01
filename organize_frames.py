import cv2
import os
import sys
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BinaryType
from pyspark.sql.functions import col, lit

csv_path = sys.argv[1]

spark = SparkSession.builder.appName("Organize Frames and Annotations").getOrCreate()

annotations_df = spark.read.csv(csv_path, header=True, inferSchema=True)

annotations_df = annotations_df.withColumnRenamed("timestamp_ms", "timestamp")
annotations_df = annotations_df.withColumnRenamed("youtube_id", "video_id")

print('----------------------Annotations----------------------')
annotations_df.printSchema()
annotations_df.show(5)

frames_df = spark.read.parquet('processed_frames')

print('----------------------Frames original----------------------')
frames_df.printSchema()
frames_df.show(5)

frames_df = frames_df.withColumn("video_id", col("video_name"))

frames_df = frames_df.withColumn("timestamp", col("frame_id") * lit(1000))

print('----------------------Frames processed----------------------')
frames_df.show(5)

joined_df = frames_df.join(annotations_df, on=["video_id", "timestamp"])

print('----------------------Final----------------------')
joined_df.printSchema()
joined_df.show(5)

joined_df.write.mode('overwrite').parquet('organized_frames_annotations')
