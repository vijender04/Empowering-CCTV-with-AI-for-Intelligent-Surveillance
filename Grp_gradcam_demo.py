from feature_extract.i3dpt import I3D
import numpy as np
from PIL import Image
import torch
import cv2
from model import WSAD
import os

device="cuda"
segment_length=16
skip_frame=1
cam_temporal_mean=True
oversampling=False

i3d = I3D(400, modality='rgb', dropout_prob=0, name='inception')
i3d.eval()
i3d.load_state_dict(torch.load("/content/drive/MyDrive/Colab Notebooks/urdmu/feature_extract/model_rgb.pth"))

net = WSAD(input_size = 1024, flag = "Test", a_nums = 60, n_nums = 60)
net.eval()
model_file="/content/drive/MyDrive/Colab Notebooks/urdmu/models/ucf_trans_2022.pkl"
net.load_state_dict(torch.load(model_file,map_location=torch.device(device)))

class Model(torch.nn.Module):
    def __init__(self, feature_model,score_model):
        super(Model, self).__init__()
        self.feature_model=feature_model
        self.score_model=score_model

    def forward(self,inputs):
        b=inputs[0].shape[0]
        features=torch.zeros(b,len(inputs),1024).to(device)

        for i,x in enumerate(inputs):
            feature=self.feature_model(x,feature_layer=5)
            features[:,i,:]=feature[0][:,:,0,0,0]

        if oversampling:
            features=features.unsqueeze(0) #(1,10,34,1024)

        out=self.score_model(features)
        return out["frame"]

def overlay_cam_on_video(video_path, cam, scores, output_path, frame_size=(224, 224), cam_opacity=0.4):
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
            break

        # Resize the frame to the target size
        frame_resized=cv2.resize(frame, frame_size)#frame[16:240, 58:282, :]

        # Resize the corresponding CAM
        cam_resized = cv2.resize(cam[frame_number], frame_size)

        # Normalize CAM to range [0, 255] for visualization
        cam_resized = ((cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min()) * 255).astype(np.uint8)

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

def oversample_data(data): # (16, 224, 224, 2)  # 10 crop

    data_flip = np.array(data[:,:,::-1,:])

    data_1 = np.array(data[:, :224, :224, :])
    data_2 = np.array(data[:, :224, -224:, :])
    data_3 = np.array(data[:, 16:240, 58:282, :])
    data_4 = np.array(data[:, -224:, :224, :])
    data_5 = np.array(data[:, -224:, -224:, :])

    #data_f_1 = np.array(data_flip[:, :224, :224, :])
    #data_f_2 = np.array(data_flip[:, :224, -224:, :])
    #data_f_3 = np.array(data_flip[:, 16:240, 58:282, :])
    #data_f_4 = np.array(data_flip[:, -224:, :224, :])
    #data_f_5 = np.array(data_flip[:, -224:, -224:, :])

    return [data_1, data_2, data_3, data_4, data_5]#,data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]

def segment_video(video_path, skip_frame=skip_frame):
    video_capture = cv2.VideoCapture(video_path)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames=total_frames+segment_length-(total_frames%segment_length)

    # Make sure when skip_frame is more than 1, net outputs on 5 segments so topk can work.
    assert (total_frames//skip_frame)//segment_length>=5, f"Please set skip_frame to {total_frames//(5*segment_length)}."

    frames = []
    frame_number = 0
    segments=[]
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

                data.requires_grad = True
                segments.append(data)
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

            data.requires_grad = True
            segments.append(data)

            frames=[]

        # Jump to the next frame position based on skip_frame
        frame_number += skip_frame

    video_capture.release()

    return segments

model=Model(i3d,net)
model.to(device)

target_layer=model.feature_model.mixed_5c

def forward_hook(module,input,output):
    activation.append(output)

def backward_hook(module,grad_in,grad_out):
    grad.append(grad_out[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

activation=[]
grad=[]

video_path="/content/drive/MyDrive/Colab Notebooks/urdmu/feature_extract/Trim_data/roadaccident/roadaccident.mp4"
segments=segment_video(video_path)
print("len(segments):",len(segments),"segments[0].shape:",segments[0].shape)

out=model(segments)

# List to store CAMs for each of the 34 outputs
cam_list = []

# Loop over each output (out has shape (1, 34))
for idx in range(out.shape[1]):

    score = out[0, idx]  # Select the score for the current output

    # Zero the gradients before backward pass
    model.zero_grad()
    grad.clear()

    # Perform backward pass to calculate gradients for the selected output
    score.backward(retain_graph=True)

    # Check if gradients are captured
    if len(grad) == 0:
        raise RuntimeError("Backward hook did not capture the gradients. Check if the hook is correctly registered.")

    # Extract gradients and activations
    if oversampling or grad[0].shape[0]>1:#when oversampling is used
        grads = grad[0].mean(dim=0).cpu().data.numpy()  # Shape: (channels, temporal_dim, height, width)
        fmap = activation[0].mean(dim=0).cpu().data.numpy()  # Shape: (channels, temporal_dim, height, width)
    else:
        grads = grad[0].cpu().data.numpy().squeeze()  # Shape: (channels, temporal_dim, height, width)
        fmap = activation[0].cpu().data.numpy().squeeze()  # Shape: (channels, temporal_dim, height, width)

    channel_dim = grads.shape[0]
    temporal_dim = grads.shape[1]

    # Reshape grads for easier mean calculation
    tmp = grads.reshape([channel_dim, temporal_dim, -1])

    # Compute weights as the mean of gradients across the spatial dimensions
    weights = np.mean(tmp, axis=2)  # Shape: (channels, temporal_dim)

    # Initialize CAM as zeros for the temporal and spatial dimensions
    cam = np.zeros(grads.shape[1:])  # Shape: (temporal_dim, height, width)

    # Multiply the weights with the corresponding feature maps
    for i in range(temporal_dim):
        for j in range(channel_dim):
            cam[i] += weights[j, i] * fmap[j, i, :]  # Multiply weights and feature maps

    # ReLU-like operation: only keep positive values
    cam = np.maximum(cam, 0)

    # Normalize CAM to the range [0, 255] for visualization
    cam = cam / (cam.max(axis=(1, 2), keepdims=True)+1e-10) * 255

    # Append the CAM to the list
    cam_list.append(cam)

    print("cam.shape",cam.shape)

# Now cam_list contains the CAMs for all n segments outputs
print(f"Generated {len(cam_list)} CAMs.")

cam_list=np.array(cam_list)
if cam_temporal_mean:
    cam_list=np.mean(cam_list,keepdims=True,axis=1)
cam_per_fame=np.repeat(cam_list,segment_length//cam_list.shape[1],axis=0)
cam_per_fame=cam_per_fame.reshape(cam_per_fame.shape[0]*cam_per_fame.shape[1], 7, 7)
print("cam_per_fame.shape",cam_per_fame.shape)

per_frame_pred = np.repeat(out.detach().cpu().numpy(), segment_length)
print("per_frame_pred.shape",per_frame_pred.shape)

os.makedirs("/content/drive/MyDrive/Colab Notebooks/urdmu/feature_extract/Out/gradcam_skip_frame_"+str(skip_frame),exist_ok=True)

output_path=os.path.join("/content/drive/MyDrive/Colab Notebooks/urdmu/feature_extract/Out/gradcam_skip_frame_"+str(skip_frame),os.path.basename(video_path))
overlay_cam_on_video(video_path, cam_per_fame, per_frame_pred, output_path)