from moviepy.editor import VideoFileClip

video_path=r"feature_extract\Out\skip_frame_5\abuse.mp4"
# Load the video file
clip = VideoFileClip(video_path)
# Extract the segment from 3 to 9 seconds
clip = clip.subclip(69, 93)
# Convert to gif
clip.write_gif(video_path.replace(".mp4",".gif"))
