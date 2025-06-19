from model import WSAD
import torch
import numpy as np
import os 
import cv2

device='cpu'

net = WSAD(input_size = 1024, flag = "Test", a_nums = 60, n_nums = 60)
model_file=r"models\ucf_trans_2022.pkl"
net.load_state_dict(torch.load(model_file, map_location = device))
net.eval()

ra=np.load(r"feature_extract\UCF_ten\roadaccident_i3d.npy")

with torch.no_grad():
    pred=net(torch.from_numpy(ra[None]).to(device))["frame"]


def draw_line_on_video(video_path, values, output_path):
    """
    Draws a horizontal line on each frame of a video according to the provided values and saves the modified video.

    Parameters:
        video_path (str): Path to the input video file.
        values (list of float): A list of values between 0 and 1, where each value determines the vertical position of the line for each frame.
        output_path (str): Path to the output video file. Defaults to 'output_video.avi'.
    """

    # Open the input video
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure the number of values matches the number of frames
    if len(values) < total_frames:
        print("Error: The number of values does not match the number of frames in the video.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use XVID codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the vertical position to draw the line
        line_y = int((1 - values[frame_idx]) * frame_height)

        # Draw a horizontal line on the frame
        frame = cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 2)

        # Add text near the line to display the value
        text_position = (10, line_y - 10 if line_y - 10 > 10 else line_y + 20)  # Adjust text position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)  # Green color (same as line)
        thickness = 2
        
        # Put the value as text on the frame
        frame = cv2.putText(frame, f'Value: {values[frame_idx]:.2f}', text_position, font, font_scale, color, thickness)

        # Write the modified frame to the output video
        out.write(frame)

        frame_idx += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {output_path}")

def get_video_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()

    return fps

# Example usage
video_path = r'feature_extract\Data\roadaccident\roadaccident.mp4'  # Replace with your video path
fpre_ = np.repeat(pred.detach().cpu().numpy(), 16)  # Replace with your list of values between 0 and 1, length must match the number of frames
draw_line_on_video(video_path, fpre_, r"feature_extract\Out\roadaccident.mp4")


topk_values, topk_indices=torch.topk(pred.detach().cpu(), k=5, dim=1)
see_seconds_of_anomaly=(topk_indices*16)/get_video_fps(video_path)
print(see_seconds_of_anomaly,topk_values)