import cv2
from torchvision import transforms as T

resize = T.Resize((800, 800))
tensor_transform = T.ToTensor()

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = tensor_transform(resize(frame))
    return frame
