import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

class draw_para:
    def __init__(self):#传参shape(75,3)
        self.X_train,self. y_train=0,0
    def set_para(self,X_train,y_train):#传入不同shape的X_train
        self.X_train, self.y_train = X_train, y_train
    def draw3d(self,ax,index):#被调函数，勿直接调用,shape(75,3)
        x, y, z = [], [], []
        for i in index:
            a, b, c = self.X_train[i][0], self.X_train[i][1], self.X_train[i][2]
            if a==0 and b==0:
                continue
            x.append(a)
            y.append(1 - b)
            z.append(c)
            print(a, b, c)
        ax.plot3D(x, z, y, color='red', alpha=0.5, linestyle='--', linewidth=2)
    def draw_body(self):#video是视频下标，pic是帧的下标,画身体的3d图,shape(75,3)
        fig = plt.figure()
        ax = Axes3D(fig)
        self.draw3d(ax,[8,6,5,4,0,1,2,3,7])
        self.draw3d(ax,[23,11,12,24])
        self.draw3d(ax,[12,14,16])
        self.draw3d(ax,[11,13,15])
        plt.show()
    def draw_left_hand(self):#video是视频下标，pic是帧的下标,画左手的3d图,shape(75,3)
        fig = plt.figure()
        ax = Axes3D(fig)
        #大拇指
        tmp = [33,34,35,36,37]
        self.draw3d(ax, tmp)
        #食指
        tmp = [33, 38, 39, 40, 41]
        self.draw3d(ax, tmp)
        #手关节
        tmp = [38, 42, 46, 50]
        self.draw3d(ax, tmp)
        # 中指
        tmp = [42,43,44,45]
        self.draw3d(ax, tmp)
        # 无名指
        tmp = [46, 47, 48, 49]
        self.draw3d(ax, tmp)
        #小拇指
        tmp = [33,50,51,52,53]
        self.draw3d(ax, tmp)
        plt.show()
    def draw_right_hand(self):#video是视频下标，pic是帧的下标,画右手的3d图,shape(75,3)
        fig = plt.figure()
        ax = Axes3D(fig)
        # 大拇指
        tmp = [54, 55, 56, 57, 58]
        self.draw3d(ax, tmp)
        # 食指
        tmp = [54, 59, 60, 61, 62]
        self.draw3d(ax, tmp)
        # 手关节
        tmp = [59, 63, 67, 71]
        self.draw3d(ax, tmp)
        # 中指
        tmp = [63, 64, 65, 66]
        self.draw3d(ax, tmp)
        # 无名指
        tmp = [67, 68, 69, 70]
        self.draw3d(ax, tmp)
        # 小拇指
        tmp = [54, 71, 72, 73, 74]
        self.draw3d(ax, tmp)
        plt.show()
    def dataset_feature(self,x):  # x代表位点,画出每个位点在测试属猪中的出现次数柱状图,shape(2611,16,75,3)
        cnt, index = [], []
        for i in range(self.X_train.shape[0]):
            tmp = 0
            index.append(i)
            for k in range(self.X_train.shape[1]):
                # for m in range(X_train.shape[2])
                if self.X_train[i][k][x][0] != 0 and self.X_train[i][k][0][1] != 0:
                    tmp += 1
            cnt.append(tmp)
        plt.figure(figsize=(10, 5), dpi=100)
        plt.bar(index, cnt)
        plt.show()
    def draw_vector(self):  # key值-[0,74]代表不同的位点，画出某一种视频矢量图
        x, y = [], []
        plt.figure(figsize=(15, 15))
        for m in range(75):
            plt.subplot(15,5, m + 1)
            for j in range(self.X_train.shape[0]):
                x.append(self.X_train[j][m][0])
                y.append(self.X_train[j][m][1])
            plt.plot(x, y)
            x, y = [], []
        plt.show()

def get_data():#测试用
        labels = ["Opaque", "Red", "Green", "Yellow", "Bright", "Light-blue", "Colors", "Red", "Women", "Enemy", "Son",
                  "Man", "Away", "Drawer", "Born", "Learn",
                  "Call", "Skimmer", "Bitter", "Sweet milk", "Milk", "Water", "Food", "Argentina", "Uruguay", "Country",
                  "Last name", "Where", "Mock", "Birthday", "Breakfast", "Photo",
                  "Hungry", "Map", "Coin", "Music", "Ship", "None", "Name", "Patience", "Perfume", "Deaf", "Trap",
                  "Rice",
                  "Barbecue", "Candy", "Chewing-gum", "Spaghetti",
                  "Yogurt", "Accept", "Thanks", "Shut down", "Appear", "To land", "Catch", "Help", "Dance", "Bathe",
                  "Buy",
                  "Copy", "Run", "Realize", "Give", "Find"]

        data_dir = os.path.join(os.getcwd(), 'dataset', 'npz')
        datasets = os.listdir(data_dir)
        selected_frames = 16

        X, y = [], []

        for filename in datasets:
            content = np.load(os.path.join(data_dir, filename))
            length = content['x'].shape[0]
            if length < selected_frames:
                continue
            selected = content['y'][np.linspace(0, length - 1, selected_frames).astype(int)]
            # x = [line.flatten() for line in selected]
            X.append(np.array(selected, dtype='float32'))
            y.append(int(filename.split('_')[0]) - 1)

        X = np.array(X, dtype='float32')
        y = np.array(y, dtype='uint8')
        y = to_categorical(y).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        return X_train, y_train
if __name__ == '__main__':
    my_draw=draw_para()
    X_train,y_train=get_data()#测试用
    my_draw.set_para(X_train[10][5],y_train[10][5])#传入不同shape的数据
    my_draw.draw_body()
    my_draw.draw_left_hand()
    my_draw.draw_right_hand()