from tensorflow import keras
from detect import FrameMap
import numpy as np
import os


model = keras.models.load_model("model/VGGNet15")
root = os.path.join(os.getcwd(), 'dataset', 'valid')
selected = 16
X = []
for filename in os.listdir(root):
    frame_map = FrameMap(os.path.join(root, filename))
    frames = frame_map.video2frames()
    coordinates, ignored = frame_map.frames2coordinates(frames, min_detection_confidence=0.8, min_tracking_confidence=0.7)
    features = np.array([[[dot.x, dot.y, dot.z] if dot else [0] * 3 for dot in dots] for dots in coordinates], dtype='float32')
    if features.shape[0] < selected:
        print(f"deleted {filename} whose frames: {features.shape}.")
        continue
    print(f"from {filename} imported frames {features.shape}.")
    X.append(features[np.linspace(0, features.shape[0]-1, selected).astype(int)])

X = np.array(X, dtype='float32')
print(f"generated X: {X.shape}.")
print(np.argmax(model.predict(X), axis=1))
