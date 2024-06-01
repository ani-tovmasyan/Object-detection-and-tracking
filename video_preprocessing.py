import cv2
import os
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BinaryType


def extract_frames(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_rate == 0:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            frame_list.append((os.path.basename(video_path), frame_id, frame_bytes))
        frame_id += 1
    cap.release()
    return frame_list

def process_videos(video_dir, frame_rate=1):
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    frames_data = []
    for video_file in video_files:
        frames_data.extend(extract_frames(video_file, frame_rate))
    return frames_data

if __name__=='__main__':
    spark = SparkSession.builder.appName("Video Preprocessing").getOrCreate()
    video_directory = 'videos'
    
    frames_data = process_videos(video_directory, frame_rate=30)  # Extract one frame per second
    
    schema = StructType([
        StructField("video_name", StringType(), True),
        StructField("frame_id", IntegerType(), True),
        StructField("frame_data", BinaryType(), True)
    ])
    
    frames_df = spark.createDataFrame([Row(video_name=fd[0], frame_id=fd[1], frame_data=fd[2]) for fd in frames_data], schema)
    
    frames_df.printSchema()
    
    frames_df.show()
    
    frames_df.write.mode('overwrite').parquet('processed_frames')

