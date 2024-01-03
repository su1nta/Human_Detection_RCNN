import numpy as np
import cv2
import torch

# Open the video file
cap = cv2.VideoCapture('Crowd.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Read the video frame by frame
frames = []
frames = [cv2.cap(cap, cv2.CAP_NEXT)]

print(frames)

# Resize the frame
frame_res = []
print("Resizing frames...")
for frame in frames:
    frame_res.append(cv2.resize(frame, (width, height)))
# frames = cv2.resize(frame, (width, height))

# Convert the frame to RGB
print("Converting frames to RGB...")
frame_rgb = []
for frame in frame_res:
    frame_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Normalize the pixel values
# frame = frame.astype(np.float32) / 255.0
print("Normalizing frames...")
frame_norm = []
for frame in frame_rgb:
    frame_norm.append(frame.astype(np.float32) / 255.0)

# Convert the frame to a PyTorch tensor
print("Converting frames to tensor...")
frame_processed = torch.tensor(frame_norm)

print(frame_processed)