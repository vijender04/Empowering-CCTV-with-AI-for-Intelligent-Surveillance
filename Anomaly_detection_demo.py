from feature_extract.i3dpt import I3D
import numpy as np
from PIL import Image
import torch
import cv2
from model import WSAD
import os

device="cpu"
segment_length=16
skip_frame=1

i3d = I3D(400, modality='rgb', dropout_prob=0, name='inception')
i3d.eval()
i3d.load_state_dict(torch.load("feature_extract/model_rgb.pth"))
i3d.to(device)

net = WSAD(input_size = 1024, flag = "Test", a_nums = 60, n_nums = 60)
model_file="models/ucf_trans_2022.pkl"
net.load_state_dict(torch.load(model_file,map_location=torch.device(device)))
net.to(device)
net.eval()


def draw_line_on_video(video_path, values, output_path, skip_frame=skip_frame):
    """
    Draws a horizontal line on each frame of a video according to the provided values and saves the modified video.
    Skips frames according to the skip_frame parameter and uses cv2.CAP_PROP_POS_FRAMES to set frame positions.

    Parameters:
        video_path (str): Path to the input video file.
        values (list of float): A list of values between 0 and 1, where each value determines the vertical position of the line for each frame.
        output_path (str): Path to the output video file.
        skip_frame (int): Number of frames to skip while drawing and saving.
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

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(output_path, fourcc, fps // skip_frame, (frame_width, frame_height))

    frame_idx = 0
    saved_frame_idx = 0  # To keep track of the index in the `values` list
    while True:
        # Set the frame position based on skip_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        # Get the vertical position to draw the line
        line_y = int((1 - values[saved_frame_idx]) * frame_height)

        # Draw a horizontal line on the frame
        frame = cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 2)

        # Add text near the line to display the value
        text_position = (10, line_y - 10 if line_y - 10 > 10 else line_y + 20)  # Adjust text position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)  # Green color (same as line)
        thickness = 2

        # Put the value as text on the frame
        frame = cv2.putText(frame, f'Value: {values[saved_frame_idx]:.2f}', text_position, font, font_scale, color, thickness)

        # Write the modified frame to the output video
        out.write(frame)

        saved_frame_idx += 1
        frame_idx += skip_frame  # Skip frames according to the skip_frame value

        # Break if we've processed all values
        if saved_frame_idx >= len(values):
            break

    # Release everything if the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {output_path}")

    return fps // skip_frame

def run_model(data):
    with torch.no_grad():
        b_data=data.float().to(device)
        b_features=i3d(b_data,feature_layer=5)
    b_features = b_features[0][:,:,0,0,0] #(10,1024)

    return b_features

def oversample_data(data): # (16, 224, 224, 2)  # 10 crop 

    data_flip = np.array(data[:,:,::-1,:])

    data_1 = np.array(data[:, :224, :224, :])
    data_2 = np.array(data[:, :224, -224:, :])
    data_3 = np.array(data[:, 16:240, 58:282, :])   
    data_4 = np.array(data[:, -224:, :224, :])
    data_5 = np.array(data[:, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, -224:, -224:, :])

    return [data_1, data_2, data_3, data_4, data_5,
        data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]

def prepare_frames_run_model(video_path, skip_frame=skip_frame):
    video_capture = cv2.VideoCapture(video_path)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames=total_frames+segment_length-(total_frames%segment_length)

    # Make sure when skip_frame is more than 1, net outputs on 5 segments so topk can work.
    assert (total_frames//skip_frame)//segment_length>=5, f"Please set skip_frame to {total_frames//(5*segment_length)}."

    frames = []
    frame_number = 0
    segment_features=[]
    while True:
        # Set the position of the next frame to read
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = video_capture.read()
        if not ret:
            if len(frames)<=segment_length and len(frames)>0:
                frames=frames+[frames[-1]]*(segment_length-len(frames))
                data=np.array(frames)
                data=np.array(oversample_data(data)).transpose([0,4,1,2,3])
                data=torch.from_numpy(data)

                segment_features.append(run_model(data))
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (340, 256))
        frame = np.array(frame).astype(float)
        frame = (frame * 2 / 255) - 1

        assert(frame.max() <= 1.0)
        assert(frame.min() >= -1.0)
        frames.append(frame)

        if len(frames)==segment_length:
            data=np.array(frames)
            data=np.array(oversample_data(data)).transpose([0,4,1,2,3])
            data=torch.from_numpy(data)

            segment_features.append(run_model(data))

            frames=[]

        # Jump to the next frame position based on skip_frame
        frame_number += skip_frame

    video_capture.release()
    
    return torch.stack(segment_features, dim=1)

data_dir="feature_extract/Data/"
video_dirs=os.listdir(data_dir)

for video_dir in video_dirs:
    video_names=os.listdir(data_dir+video_dir)
    
    for video_name in video_names:
        video_path=data_dir+video_dir+"/"+video_name

        segment_features=segment_features=prepare_frames_run_model(video_path)[None]
        print("segment_features shape:",segment_features.shape)

        with torch.no_grad():
            pred=net(segment_features)["frame"]

        per_frame_pred = np.repeat(pred.detach().cpu().numpy(), segment_length)

        os.makedirs("feature_extract/Out/skip_frame_"+str(skip_frame),exist_ok=True)

        fps=draw_line_on_video(video_path, per_frame_pred, os.path.join("feature_extract/Out/skip_frame_"+str(skip_frame),os.path.basename(video_path)))

        topk_values, topk_indices=torch.topk(pred.detach().cpu(), k=5, dim=1)
        see_seconds_of_anomaly=(topk_indices*segment_length)/fps
        print(see_seconds_of_anomaly,topk_values)
