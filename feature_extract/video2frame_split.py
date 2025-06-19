import os
import cv2

def save_video_frames_and_remove(video_path):
    # Check if the video path exists
    if not os.path.exists(video_path):
        print(f"Video path '{video_path}' does not exist.")
        return
    
    # Get the video name without the extension and create a folder with the same name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(os.path.dirname(video_path), video_name)
    output_folder = output_folder.replace('Data', 'UCF_Crime_Frames')

    # Create a new directory for storing frames
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    # Initialize frame count
    frame_count = 0

    while True:
        # Read each frame from the video
        ret, frame = cap.read()

        # Break the loop if no frame is returned (end of video)
        if not ret:
            break

        # Save the frame as an image in the new folder
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:08d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Frames saved in folder '{output_folder}'.")

#class_folders=os.listdir("feature_extract/Data")
#for class_folder in class_folders:
#    video_names=os.listdir("feature_extract/Data/"+class_folder)
#    for video_name in video_names:
#        video_path="feature_extract/Data/"+class_folder+"/"+video_name
#        save_video_frames_and_remove(video_path)

# Run commented code for every video.
save_video_frames_and_remove(r"feature_extract\Data\roadaccident\roadaccident.mp4")