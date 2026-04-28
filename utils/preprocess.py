
import cv2

def preprocess(frame):
    img = cv2.resize(frame, (320, 320))
    img = img / 255.0
    return img
