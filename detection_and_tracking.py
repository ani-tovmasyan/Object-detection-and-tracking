import os
import cv2
import torch
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import pandas_udf, col, explode
from pyspark.sql.types import StructType, StructField, FloatType, StringType, ArrayType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from ultralytics import YOLO

# Define the schema for detections
detection_schema = ArrayType(StructType([
    StructField("xmin", FloatType(), True),
    StructField("ymin", FloatType(), True),
    StructField("xmax", FloatType(), True),
    StructField("ymax", FloatType(), True),
    StructField("confidence", FloatType(), True),
    StructField("class", StringType(), True),
    StructField("name", StringType(), True)
]))

def detect_objects(frame_bytes):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)
    boxes = results[0].boxes
    detection_list = []
    for box in boxes:
        bbox = box.xyxy[0].cpu().numpy()  # Extract bbox as numpy array
        detection_list.append({
            "xmin": float(bbox[0]),
            "ymin": float(bbox[1]),
            "xmax": float(bbox[2]),
            "ymax": float(bbox[3]),
            "confidence": float(box.conf.cpu().numpy()),
            "class": str(int(box.cls.cpu().numpy())),
            "name": results[0].names[int(box.cls)]
        })
        # Draw bounding box on the image
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(img, results[0].names[int(box.cls)], (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return img, detection_list

@pandas_udf(detection_schema)
def detect_objects_udf(frame_bytes_series):
    results_list = []
    for frame_bytes in frame_bytes_series:
        img, detections = detect_objects(frame_bytes)
        results_list.append(detections)
        # Display the image with detections
        cv2.imshow("Detections", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return pd.Series(results_list)

if __name__ == '__main__':
    spark = SparkSession.builder.appName("Object Detection and Tracking").getOrCreate()
    
    model = YOLO('yolov8s.pt')
    
    frames_df = spark.read.parquet('processed_frames')
    detections_df = frames_df.withColumn("detections", detect_objects_udf(col("frame_data")))
    
    exploded_df = detections_df.select("video_name", "frame_id", explode("detections").alias("detection"))
    
    exploded_df = exploded_df.select(
        col("video_name"),
        col("frame_id"),
        col("detection.xmin").alias("xmin"),
        col("detection.ymin").alias("ymin"),
        col("detection.xmax").alias("xmax"),
        col("detection.ymax").alias("ymax"),
        col("detection.confidence").alias("confidence"),
        col("detection.class").alias("class"),
        col("detection.name").alias("name")
    )
    
    assembler = VectorAssembler(
        inputCols=["xmin", "ymin", "xmax", "ymax"],
        outputCol="features"
    )
    feature_df = assembler.transform(exploded_df)
    
    kmeans = KMeans(k=10, seed=1)
    model = kmeans.fit(feature_df.select("features"))
    
    predictions = model.transform(feature_df)
    
    predictions.write.mode('overwrite').parquet('tracking_results')
    
    # Release the OpenCV window
    cv2.destroyAllWindows()

