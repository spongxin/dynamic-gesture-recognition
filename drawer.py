from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


class Drawer:
    def __init__(self, coordinates: np.ndarray):
        """
        Drawer for data <T, 75, 3>
        :param coordinates: landmark-groups of a video.
        """
        self.coordinates = coordinates

    def draw_vector(self, dots: list):
        """视频骨骼点位移矢量图
        :param dots: 绘画骨骼点索引
        :return: none
        """
        plt.figure(figsize=(50, 50))
        for dot in dots:
            plt.subplot(15, 5, dot + 1)
            x, y = self.coordinates[:, dot, 0], self.coordinates[:, dot, 1]
            if np.sum(x) and np.sum(y):
                plt.plot(x, y)
        plt.show()

    def draw_body(self, frame: int = 0):
        """画身体骨骼3d图
        :param frame: 视频下标帧序号
        :return: none
        """
        fig = plt.figure()
        ax = Axes3D(fig)
        self._draw3d(ax, [8, 6, 5, 4, 0, 1, 2, 3, 7], frame)
        self._draw3d(ax, [23, 11, 12, 24], frame)
        self._draw3d(ax, [12, 14, 16], frame)
        self._draw3d(ax, [11, 13, 15], frame)
        plt.show()

    def draw_hand(self, mode: int = 0, frame: int = 0):
        """画手部骨骼3d图
        :param frame: 视频下标帧序号
        :param mode: 左手0/右手1
        :return:
        """
        fig = plt.figure()
        ax = Axes3D(fig)
        base = 21 * mode
        # 大拇指
        tmp = np.array([33, 34, 35, 36, 37]) + base
        self._draw3d(ax, tmp, frame)
        # 食指
        tmp = np.array([33, 38, 39, 40, 41]) + base
        self._draw3d(ax, tmp, frame)
        # 手关节
        tmp = np.array([38, 42, 46, 50]) + base
        self._draw3d(ax, tmp, frame)
        # 中指
        tmp = np.array([42, 43, 44, 45]) + base
        self._draw3d(ax, tmp, frame)
        # 无名指
        tmp = np.array([46, 47, 48, 49]) + base
        self._draw3d(ax, tmp, frame)
        # 小拇指
        tmp = np.array([33, 50, 51, 52, 53]) + base
        self._draw3d(ax, tmp, frame)
        plt.show()

    @staticmethod
    def normalization(X):
        _range = np.max(X) - np.min(X)
        return (X - np.min(X)) / _range

    def _draw3d(self, ax, index, frame: int = 0):
        x, y, z = self.coordinates[frame, index, 0], 1-self.coordinates[frame, index, 1], self.coordinates[frame, index, 2]
        ax.plot3D(x, z, y, color='red', alpha=0.5, linestyle='--', linewidth=2)


if __name__ == '__main__':
    data = np.load('dataset/NPZ/001_009_004.npz')['y']
    print(data.shape)
    Drawer(data).draw_hand(frame=5)
