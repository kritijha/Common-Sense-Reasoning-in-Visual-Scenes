import pytube
from pytube import YouTube

# Get the video URL from train_video_ids.txt file

video_ids = open('train_video_ids.txt', 'r').read().split('\n')

BASE_URL = 'https://www.youtube.com/watch?v='

video_urls = [BASE_URL+video_id for video_id in video_ids]

# Print the video URLs

for video_url in video_urls[:10]:
    # Download the video and place in another directory videos
    print("Video url ",video_url)
    yt = YouTube(video_url)
    print(yt.title)
    #download the video
    yt.streams.first().download(output_path='../videos',filename=yt.title)
    print(video_url)