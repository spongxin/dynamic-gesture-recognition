from resource.detect import FrameMap
import numpy as np
import keras


class Solution(object):
    def __init__(self, model: str, clusters: int, interval: int = 1, min_detection=0.8, min_tracking=0.4):
        self.fm = None
        self.index = None
        self.dropped = None
        self.coords = None
        self.landmarks = None
        self.frames = None
        self.model = model
        self.interval = interval
        self.clusters = clusters
        # self.kmeans = KMeans(n_clusters=self.clusters)
        self.min_detection = min_detection
        self.min_tracking = min_tracking

    def process(self, video: str):
        fm = FrameMap(
                min_detection_confidence=self.min_detection,
                min_tracking_confidence=self.min_tracking
        )
        print('传入视频：', video)
        self.frames = fm.video2frames(video, self.interval)
        print('导入视频：', video)
        self.landmarks = np.array([[*fm.frame2landmarks(f)] for f in self.frames])
        coords, dropped = [], []
        for index, landmark in enumerate(self.landmarks):
            if landmark[0] is not None and landmark[1] is not None:
                coords.append(fm.landmarks2value(*landmark))
                continue
            dropped.append(index)
        self.dropped = np.array(dropped, dtype='uint')
        self.coords = np.array(coords)
        self.index = fm.remove_frames(np.arange(self.frames.shape[0]), self.dropped)
        self.fm = fm
        fm.close()

    def select_key_frames(self):
        return
        # if not self.coords:
        #     return
        # self.kmeans.fit(self.coords)
        # return self.kmeans.predict(self.coords)

    def predict(self):
        if self.coords.shape[0] < self.clusters:
            return
        X = [self.coords[np.linspace(0, self.coords.shape[0] - 1, self.clusters).astype(int)], ]
        X = np.array(X, dtype='float32')
        return self.get_model(self.model).predict(X)

    @staticmethod
    def get_model(model: str):
        return keras.models.load_model(model)
