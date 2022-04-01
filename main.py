from tensorflow import keras
from detect import FrameMap
import numpy as np
import os


class Solution(object):
    def __init__(self, model: str, interval: int = 1):
        self.coordinates = None
        self.landmarks = None
        self.frames = None
        self.model = model
        self.interval = interval
        self.selected = 16

    def process(self, video: str):
        frame_map = FrameMap()
        self.frames = frame_map.video2frames(video, self.interval)
        print('frames 0', self.frames.shape)
        self.landmarks = np.array([[*frame_map.frame2landmarks(f)] for f in self.frames])
        print('landmarks', self.landmarks.shape)
        coordinates, ignored = [], []
        for index, landmark in enumerate(self.landmarks):
            if landmark[0] is not None and landmark[1] is not None:
                coordinates.append(frame_map.landmarks2value(*landmark))
                continue
            ignored.append(index)
        print('ignored', ignored)
        self.coordinates = np.array(coordinates)
        print('coordinates', self.coordinates.shape)
        self.frames = frame_map.remove_frames(self.frames, np.array(ignored)) if ignored else self.frames
        print('frames 1', self.frames.shape)
        frame_map.close()

    def predict(self):
        if self.coordinates.shape[0] < self.selected:
            return
        X = [self.coordinates[np.linspace(0, self.coordinates.shape[0]-1, self.selected).astype(int)], ]
        X = np.array(X, dtype='float32')
        return self.get_model(self.model).predict(X)

    @staticmethod
    def get_model(model: str):
        path = os.path.join(os.getcwd(), 'model', model)
        return keras.models.load_model(path)


if __name__ == '__main__':
    s = Solution(model='VGGNet15', interval=1)
    s.process("dataset/valid/003_001_003.mp4")
    print(s.coordinates.shape)
    ret = s.predict()
    print(ret)
    print(np.argmax(ret, axis=1)+1)

