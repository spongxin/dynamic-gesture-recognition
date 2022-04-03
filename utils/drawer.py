from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


class Drawer:
    def __init__(self, coordinates: np.ndarray):
        """
        Drawer for data <T, 75, 3>
        :param coordinates: landmark-groups of a video.
        """
        self.ani = None
        self.coordinates = coordinates
        self.dic = {13: 'left_elbow', 14: 'right_elbow', 15: 'left_wrist', 16: 'right_wrist',
               33: 'thumb_cmc', 34: 'thumb_mcp', 35: 'thumb_ip', 36: 'thumb_tip', 37: 'index_finger_mcp',
               38: 'index_finger_pip', 39: 'index_finger_dip', 40: 'index_finger_tip', 41: 'index_finger_tip',
               42: 'middle_finger_mcp', 43: 'middle_finger_pip', 44: 'middle_finger_dip', 45: 'middle_finger_tip',
               46: 'right_finger_mcp', 47: 'right_finger_pip', 48: 'right_finger_dip', 49: 'right_finger_tip',
               50: 'pinky_mcp', 51: 'pinky_pip', 52: 'pinky_dip', 53: 'pinky_tip'}
    def draw_vector(self, dots: list):
        """视频骨骼点位移矢量图,一共25个点的矢量图
        :param dots: 绘画骨骼点索引
        :return: none
        """
        fig=plt.figure(figsize=(30, 30))
        fig.canvas.set_window_title('关键点位移矢量图')
        m=1
        for dot in dots:
            plt.subplot(5, 5, m)
            m+=1
            plt.subplots_adjust(wspace=0.5,hspace=1)
            plt.title(self.dic.get(dot))
            ax1 = plt.gca()
            ax1.set_xlim([0,1])  # 重点是对获取到的axes对象进行操作
            ax1.set_ylim([0,1])  # 重点是对获取到的axes对象进行操作
            x, y = self.coordinates[:, dot, 0], self.coordinates[:, dot, 1]
            if np.sum(x) and np.sum(y):
                plt.plot(x, y)
        plt.show()

    def draw_body(self, frame: int = 0, flag: int = 0, ax=None):
        """画身体骨骼3d图
        :param ax:
        :param frame: 视频下标帧序号
        :param flag: flag!=0时为画动图，不需要show()
        :return: none
        """
        if flag == 0:
            fig = plt.figure()
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
        self._draw3d(ax, [8, 6, 5, 4, 0, 1, 2, 3, 7], frame)
        self._draw3d(ax, [23, 11, 12, 24], frame)
        self._draw3d(ax, [12, 14, 16], frame)
        line = self._draw3d(ax, [11, 13, 15], frame)
        if flag == 0:
            plt.show()
        return line

    def draw_hand(self, mode: int = 0, frame: int = 0, flag: int = 0, ax=None):
        """画手部骨骼3d图
        :param ax:
        :param flag:
        :param frame: 视频下标帧序号
        :param mode: 左手0/右手1
        :return:
        """
        if flag == 0:
            fig = plt.figure()
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
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
        line = self._draw3d(ax, tmp, frame)
        if flag == 0:
            plt.show()
        return line

    def draw_ani(self, intervals: int = 200, mod: int = 0):
        """画身体骨骼3d图
            :param intervals: 播放每帧的切换速度
            :param mod: 0/左手,1/右手,其他/身体,默认为左手0
            :return: none
        """
        fig = plt.figure()
        name=["惯用手-3D检测","非惯用手-3D检测","POSE-3D检测"]
        fig.canvas.set_window_title('{}'.format(name[mod]))
        ax = Axes3D(fig,auto_add_to_figure=False)
        fig.add_axes(ax)

        def ani_func(i):
            ax.cla()
            if mod == 0 or mod == 1:
                line = self.draw_hand(mod, i % self.coordinates.shape[0], 1, ax)
            else:
                line = self.draw_body(i % self.coordinates.shape[0], 1, ax)
            return line

        def init():
            if mod == 0 or mod == 1:
                line = self.draw_hand(0, 0, 1, ax)
            else:
                line = self.draw_body(0, 1, ax)
            return line

        self.ani = animation.FuncAnimation(fig, func=ani_func, frames=100, init_func=init, interval=intervals, blit=False)
        plt.show()

    @staticmethod
    def normalization(X):
        _range = np.max(X) - np.min(X)
        return (X - np.min(X)) / _range

    def _draw3d(self, ax, index, frame: int = 0):
        """
        :param index: 位点
        :param frame:视频索引下标
        :return: 只有画动图里面用到
        """
        x, y, z = self.coordinates[frame, index, 0], 1 - self.coordinates[frame, index, 1], self.coordinates[
            frame, index, 2]
        line = ax.plot3D(x, z, y, color='red', alpha=0.5, linestyle='--', linewidth=2)
        return line


if __name__ == '__main__':
    data = np.load('../dataset/NPZ/031_009_004.npz')['y']
    print(data.shape)
    tmp=[13,14,15,16]
    for x in range(33,54):
        tmp.append(x)
    Drawer(data).draw_vector(tmp)

