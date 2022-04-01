from tensorflow import keras
from detect import FrameMap
from sklearn.cluster import KMeans
import numpy as np
import os


class Solution(object):
    def __init__(self, model: str = 'VGGNet15', interval: int = 1, min_detection=0.8, min_tracking=0.4):
        self.coords = None
        self.landmarks = None
        self.frames = None
        self.model = model
        self.interval = interval
        self.clusters = 16
        self.kmeans = KMeans(n_clusters=self.clusters)
        self.min_detection = min_detection
        self.min_tracking = min_tracking

    def process(self, video: str):
        fm = FrameMap(
                min_detection_confidence=self.min_detection,
                min_tracking_confidence=self.min_tracking
        )
        self.frames = fm.video2frames(video, self.interval)
        self.landmarks = np.array([[*fm.frame2landmarks(f)] for f in self.frames])
        coords, dropped = [], []
        for index, landmark in enumerate(self.landmarks):
            if landmark[0] is not None and landmark[1] is not None:
                coords.append(fm.landmarks2value(*landmark))
                continue
            dropped.append(index)
        self.coords = fm.remove_frames(np.array(coords), np.array(dropped)) if dropped else np.array(coords)
        self.frames = fm.remove_frames(self.frames, np.array(dropped)) if dropped else self.frames

    def select_key_frames(self):
        if not self.coords:
            return
        self.kmeans.fit(self.coords)
        return self.kmeans.predict(self.coords)

    def predict(self):
        if self.coords.shape[0] < self.clusters:
            return
        X = [self.coords[np.linspace(0, self.coords.shape[0] - 1, self.clusters).astype(int)], ]
        X = np.array(X, dtype='float32')
        return self.get_model(self.model).predict(X)

    @staticmethod
    def get_model(model: str):
        path = os.path.join(os.getcwd(), 'model', model)
        return keras.models.load_model(path)


if __name__ == '__main__':
    s = Solution(model='VGGNet15', interval=1)
    s.process("dataset/valid/003_001_003.mp4")
    print(s.coords.shape)
    ret = s.predict()
    print(ret)
    print(np.argmax(ret, axis=1)+1)
    print(s.select_key_frames())

