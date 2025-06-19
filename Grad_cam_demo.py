from feature_extract.i3dpt import I3D
import numpy as np
from PIL import Image
import torch
import cv2
from model import WSAD
import os
import matplotlib.pyplot as plt

device="cpu"
segment_length=16
cam_temporal_mean=True
oversampling=True
skip_frame=1

# Initialize models
anomaly_model = WSAD(input_size=1024, flag="Test", a_nums=60, n_nums=60)
anomaly_model.eval()
model_file="models/ucf_trans_2022.pkl"
anomaly_model.load_state_dict(torch.load(model_file, map_location = device))
anomaly_model.to(device)
i3d_model = I3D(400, modality='rgb', dropout_prob=0, name='inception')
i3d_model.load_state_dict(torch.load("feature_extract/model_rgb.pth", map_location = device))
i3d_model.eval()  # Set to evaluation mode
i3d_model.to(device)

def overlay_cam_on_video(video_path, cam, scores, output_path, frame_size=(224, 224), cam_opacity=0.3):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the video's width, height, and FPS (Frames Per Second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Video writer to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame_number >= len(cam):  # Exit if video ends or CAM frames are exhausted
            print(output_path)
            break

        # Resize the frame to the target size
        frame_resized = cv2.resize(frame, frame_size)

        # Resize the corresponding CAM
        # Normalize CAM to range [0, 255] for visualization
        cam_rerange = cv2.normalize(cam[frame_number], None, 0, 255, cv2.NORM_MINMAX)#[0,255]
        cam_resized = cv2.resize(cam_rerange, frame_size)

        cam_resized = np.uint8(cam_resized)

        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

        # Overlay CAM on the frame with some opacity
        overlay = cv2.addWeighted(heatmap, cam_opacity, frame_resized, 1 - cam_opacity, 0)

        # Put the score text on the frame
        score_text = f"Score: {scores[frame_number]:.2f}"
        cv2.putText(overlay, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the frame to the output video
        out.write(overlay)

        # Move to the next frame and CAM
        frame_number += 1

    # Release the video and writer objects
    cap.release()
    out.release()

def apply_gradcam_for_temporal_frames(segment_features, i3d_model, anomaly_model, target_layer_name='mixed_5c'):
    """
    Applies Grad-CAM to I3D feature extraction to get heatmaps for each temporal frame.

    :param segment_features: Input features for a specific segment of shape (1, 3, 16, 224, 224)
    :param i3d_model: Pre-trained I3D model for feature extraction
    :param anomaly_model: Model that predicts anomaly scores
    :param target_layer_name: Name of the target layer in I3D for Grad-CAM
    :return: List of heatmaps for each temporal frame
    """
    # Enable gradient tracking
    segment_features.requires_grad = True

    # Extract features using I3D model
    def hook_function(module, input, output):
        nonlocal feature_map
        feature_map = output
        output.register_hook(save_grad)

    # Hook to save gradients
    gradients = []
    def save_grad(grad):
        gradients.append(grad)

    # Register the hook to the target layer
    layer = i3d_model.get_submodule(target_layer_name)
    handle = layer.register_forward_hook(hook_function)

    # Extract features from I3D model
    features = i3d_model(segment_features, feature_layer=5)[0][:,:,0,0,0]
    if oversampling:
        features = features.view(1,features.size()[0], 1, -1)  # Reshape to shape (1, 10, 1, 1024)
    else:
        features = features.view(1, 1, -1)  # Reshape to shape (1, 1, 1024)

    # Forward pass through anomaly model to get scores
    scores = anomaly_model(features)['frame']  # Output anomaly score

    # Backward pass to compute gradients
    anomaly_model.zero_grad()
    i3d_model.zero_grad()  # Clear gradients for I3D model
    scores.backward()

    # Compute Grad-CAM for each temporal frame
    if oversampling:
        gradients = gradients[0].cpu().data.numpy().mean(axis=0)  # Retrieve saved gradients
        feature_map = feature_map.cpu().data.numpy().mean(axis=0)  # Retrieve saved feature map
    else:
        gradients = gradients[0].cpu().data.numpy().squeeze()  # Retrieve saved gradients
        feature_map = feature_map.cpu().data.numpy().squeeze()  # Retrieve saved feature map

    # Number of temporal frames
    num_frames = feature_map.shape[1]

    heatmaps = []
    for t in range(num_frames):
        # Get gradients and feature map for the t-th temporal frame
        grad_t = gradients[:,t,:,:]  # (1024, 7, 7)
        fmap_t = feature_map[:,t,:,:]  # (1024, 7, 7)

        # Compute Grad-CAM
        weights = np.mean(grad_t, axis=(1, 2))  # Mean across height and width
        cam = np.zeros(fmap_t.shape[1:], dtype=np.float32)

        # Weight the feature map by the gradients
        for i, w in enumerate(weights):
            cam += w * fmap_t[i, :, :]

        # Apply ReLU to the result
        cam = np.maximum(cam, 0)

        # Normalize and scale the heatmap
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-8)#np.max(cam)#

        heatmaps.append(cam)

    # Clean up hooks
    handle.remove()

    return np.array(heatmaps),scores.cpu().data.numpy()

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

def segment_gradcam(video_path, skip_frame=skip_frame):
    video_capture = cv2.VideoCapture(video_path)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames=total_frames+segment_length-(total_frames%segment_length)
    print("total_frames:",total_frames)

    heatmaps=[]
    scores=[]
    frames = []
    frame_number = 0
    while True:
        # Set the position of the next frame to read
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = video_capture.read()
        if not ret:
            if len(frames)<=segment_length and len(frames)>0:
                frames=frames+[frames[-1]]*(segment_length-len(frames))
                data=np.array(frames)
                if oversampling:
                    data=np.array(oversample_data(data)).transpose([0,4,1,2,3])
                else:
                    data=data[None].transpose([0,4,1,2,3])#[:, 16:240, 58:282, :]
                data=torch.from_numpy(data).to(device)

                heatmap,score=apply_gradcam_for_temporal_frames(data, i3d_model, anomaly_model)
                heatmaps.append(heatmap)
                scores.append(score)

            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if oversampling:
            frame = cv2.resize(frame, (340, 256))
        else:
            frame = cv2.resize(frame, (224, 224))
        frame = np.array(frame).astype(np.float32)
        frame = (frame * 2 / 255) - 1

        assert(frame.max() <= 1.0)
        assert(frame.min() >= -1.0)
        frames.append(frame)

        if len(frames)==segment_length:
            data=np.array(frames)
            if oversampling:
                data=np.array(oversample_data(data)).transpose([0,4,1,2,3])
            else:
                data=data[None].transpose([0,4,1,2,3])#[:, 16:240, 58:282, :]
            data=torch.from_numpy(data).to(device)

            heatmap,score=apply_gradcam_for_temporal_frames(data, i3d_model, anomaly_model)
            heatmaps.append(heatmap)
            scores.append(score)

            frames=[]

        # Jump to the next frame position based on skip_frame
        frame_number += skip_frame

    video_capture.release()
    
    return np.array(heatmaps),np.array(scores).squeeze()

video_path=r"feature_extract\Trim_data\explosion\explosion.mp4"

cam_list,scores=segment_gradcam(video_path)

if cam_temporal_mean:
    cam_list=np.mean(cam_list,keepdims=True,axis=1)
cam_per_fame=np.repeat(cam_list,segment_length//cam_list.shape[1],axis=0)
cam_per_fame=cam_per_fame.reshape(cam_per_fame.shape[0]*cam_per_fame.shape[1], 7, 7)
print("cam_per_fame.shape",cam_per_fame.shape)

per_frame_pred = np.repeat(scores, segment_length)
print("per_frame_pred.shape",per_frame_pred.shape)

os.makedirs("feature_extract/Out/gradcam_skip_frame_"+str(skip_frame),exist_ok=True)

output_path=os.path.join("feature_extract/Out/gradcam_skip_frame_"+str(skip_frame),os.path.basename(video_path))
overlay_cam_on_video(video_path, cam_per_fame, per_frame_pred, output_path)