import os
import sys
import yt_dlp as youtube_dl
import pandas as pd

annotations_path = sys.argv[1] 
annotations_df = pd.read_csv(annotations_path)

# Create a directory to store the downloaded videos
download_dir = 'videos'
os.makedirs(download_dir, exist_ok=True)

# Extract unique video IDs from the annotations
video_ids = annotations_df.iloc[:, 0].unique()

# Define a function to download a single video using yt-dlp
def download_video(video_id):
    url = f'https://www.youtube.com/watch?v={video_id}'
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(download_dir, f'{video_id}.mp4')
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Download a few videos for testing
for video_id in video_ids[:5]:  # Adjust the range to download more videos
    download_video(video_id)
