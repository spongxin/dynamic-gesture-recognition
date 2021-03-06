import os
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from resource import Solution
from utils.drawer import Drawer
import cv2


labels = ["Opaque", "Red", "Green", "Yellow", "Bright", "Light-blue", "Colors", "Red", "Women", "Enemy", "Son", "Man", "Away", "Drawer", "Born", "Learn",
          "Call", "Skimmer", "Bitter", "Sweet milk", "Milk", "Water", "Food", "Argentina", "Uruguay", "Country", "Last name", "Where", "Mock", "Birthday", "Breakfast", "Photo",
          "Hungry", "Map", "Coin", "Music", "Ship", "None", "Name", "Patience", "Perfume", "Deaf", "Trap", "Rice", "Barbecue", "Candy", "Chewing-gum", "Spaghetti",
          "Yogurt", "Accept", "Thanks", "Shut down", "Appear", "To land", "Catch", "Help", "Dance", "Bathe", "Buy", "Copy", "Run", "Realize", "Give", "Find"]


class Ui_MainWindow(object):
    def __init__(self):
        self.task3 = None
        self.task2 = None
        self.task = None
        self.drawer = None
        self.solution = None
        self.video = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1071, 679)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(70, 500, 381, 61))
        self.pushButton_5.setStyleSheet("height: 25px;\n"
                                        "font: 13pt \"黑体\";")
        self.pushButton_5.setObjectName("pushButton_5")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(550, 80, 451, 301))
        self.tableView.setObjectName("tableView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(550, 30, 161, 41))
        self.label.setStyleSheet("font: 12pt \"华光黑体_CNKI\";")
        self.label.setObjectName("label")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(550, 420, 451, 61))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setStyleSheet("height: 40px;\n"
                                        "font: 13pt \"黑体\";")
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_4.setStyleSheet("height: 40px;\n"
                                        "font: 13pt \"黑体\";")
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout.addWidget(self.pushButton_4)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(70, 30, 161, 41))
        self.label_5.setStyleSheet("font: 12pt \"华光黑体_CNKI\";")
        self.label_5.setObjectName("label_5")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(550, 500, 451, 61))
        self.pushButton_2.setStyleSheet("height: 40px;\n"
                                        "font: 13pt \"黑体\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(70, 80, 381, 291))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setStyleSheet("height: 10px;\n"
                                   "font: 12pt \"华光大黑二_CNKI\";")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit.setStyleSheet("height: 25px;")
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setStyleSheet("height: 10px;\n"
                                   "font: 12pt \"华光大黑二_CNKI\";")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_2.setStyleSheet("height: 25px;")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_3.addWidget(self.lineEdit_2)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_4.setStyleSheet("height: 10px;\n"
                                   "font: 12pt \"华光大黑二_CNKI\";")
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_3.setStyleSheet("height: 25px;")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_4.addWidget(self.lineEdit_3)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_6.setStyleSheet("height: 10px;\n"
                                   "font: 12pt \"华光大黑二_CNKI\";")
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_5.addWidget(self.label_6)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_4.setStyleSheet("height: 25px;")
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_5.addWidget(self.lineEdit_4)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayoutWidget_7 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(70, 420, 381, 71))
        self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        self.pushButton.setStyleSheet("height: 40px;\n"
                                      "font: 13pt \"黑体\";")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_7.addWidget(self.pushButton)
        self.pushButton_6 = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        self.pushButton_6.setStyleSheet("height: 40px;\n"
                                        "font: 13pt \"黑体\";")
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout_7.addWidget(self.pushButton_6)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1071, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_5.setText(_translate("MainWindow", "开始处理"))
        self.label.setText(_translate("MainWindow", "手势预测结果："))
        self.pushButton_3.setText(_translate("MainWindow", "POSE-3D检测"))
        self.pushButton_4.setText(_translate("MainWindow", "惯用手-3D检测"))
        self.label_5.setText(_translate("MainWindow", "系统输入："))
        self.pushButton_2.setText(_translate("MainWindow", "复合视频播放"))
        self.label_2.setText(_translate("MainWindow", "检测置信度："))
        self.label_3.setText(_translate("MainWindow", "跟踪置信度："))
        self.label_4.setText(_translate("MainWindow", "关键帧数量："))
        self.label_6.setText(_translate("MainWindow", "输入总帧数："))
        self.pushButton.setText(_translate("MainWindow", "选择视频文件"))
        self.pushButton_6.setText(_translate("MainWindow", "位移矢量图"))
        self.lineEdit.setText('0.5')
        self.lineEdit_2.setText('0.5')
        self.lineEdit_3.setText('12')
        self.lineEdit_3.setDisabled(True)
        self.lineEdit_4.setDisabled(True)

        self.pushButton.clicked.connect(self.select_video)
        self.pushButton_2.clicked.connect(self.display_video)
        self.pushButton_3.clicked.connect(self.display_pose)
        self.pushButton_4.clicked.connect(self.display_hand)
        self.pushButton_5.clicked.connect(self.start_process)
        self.pushButton_6.clicked.connect(self.display_vector)
        self.table_model = QtGui.QStandardItemModel(7, 2)
        self.table_model.setHorizontalHeaderLabels(['Type', 'Probability'])
        self.tableView.setModel(self.table_model)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.setColumnWidth(0, 200)
        for line in [self.lineEdit, self.lineEdit_2, self.lineEdit_3, self.lineEdit_4]:
            line.setAlignment(QtCore.Qt.AlignCenter)

    def select_video(self):
        video = QtWidgets.QFileDialog.getOpenFileName(None, "选择mp4文件", "./", "Mp4 Files (*.mp4)")
        if video and video[0]:
            self.video = video[0]
            self.message('已选择文件：'+self.video)
        
    def message(self, msg):
        self.statusbar.showMessage(msg)

    def start_process(self):
        try:
            if not self.video:
                return self.message('请选择视频文件！')
            elif not self.lineEdit.text() or not self.lineEdit_2.text():
                return self.message('请输入检测阈值！')
            self.message('正在处理：' + self.video)
            min_detection = float(self.lineEdit.text())
            min_tracking = float(self.lineEdit_2.text())
            for button in [self.pushButton, self.pushButton_2, self.pushButton_3, self.pushButton_4, self.pushButton_5, self.pushButton_6]:
                button.setEnabled(False)
            self.solution = SolutionThread(video=self.video, clusters=12, model=os.path.join(os.getcwd(), 'model', 'VGGNet12'), min_detection=min_detection, min_tracking=min_tracking)
            self.solution.sign.connect(self.listen_ret)
            self.solution.start()
        except Exception as err:
            self.message(err.__str__())

    def listen_ret(self, ret):
        self.message('处理完毕：' + self.video)
        if ret is not None and ret.size != 0:
            self.drawer = Drawer(coordinates=self.solution.solution.coords)
            ret = ret[0]
            pro = np.argsort(ret)[::-1]
            for row in range(7):
                item1 = QtGui.QStandardItem(labels[pro[row]])
                item2 = QtGui.QStandardItem(str(ret[pro[row]]))
                self.table_model.setItem(row, 0, item1)
                self.table_model.setItem(row, 1, item2)
            self.tableView.setModel(self.table_model)
        else:
            self.message('未取得预测结果！')
        self.lineEdit.setText(str(self.solution.solution.min_detection))
        self.lineEdit_2.setText(str(self.solution.solution.min_tracking))
        self.lineEdit_3.setText(str(self.solution.solution.clusters))
        self.lineEdit_4.setText(str(self.solution.solution.frames.shape[0]))
        # except Exception as err:
        #     self.message(err.__str__())
        # finally:
        for button in [self.pushButton, self.pushButton_2, self.pushButton_3, self.pushButton_4, self.pushButton_5, self.pushButton_6]:
            button.setEnabled(True)

    def display_pose(self):
        if self.drawer is None:
            return self.message("无POSE-3D数据！")
        self.drawer.draw_ani(200, 2)
    
    def display_hand(self):
        if self.drawer is None:
            return self.message("无HAND-3D数据！")
        self.drawer.draw_ani(200, 0)

    def display_vector(self):
        if self.drawer is None:
            return self.message("无矢量图数据！")
        for button in [self.pushButton, self.pushButton_2, self.pushButton_3, self.pushButton_4, self.pushButton_5, self.pushButton_6]:
            button.setEnabled(False)
        self.drawer.draw_vector(np.array([i for i in range(13, 17)] + [i for i in range(33, 54)]))
        for button in [self.pushButton, self.pushButton_2, self.pushButton_3, self.pushButton_4, self.pushButton_5, self.pushButton_6]:
            button.setEnabled(True)

    def display_video(self):
        if self.solution is None or self.solution.solution is None:
            return self.message("无视频数据！")
        for index in range(len(self.solution.solution.frames)):
            img = self.solution.solution.fm.draw_landmarks(self.solution.solution.frames[index], *self.solution.solution.landmarks[index])
            cv2.imshow('Video', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            time.sleep(0.05)
            if cv2.waitKey(1) == ord('Q'):
                break


class SolutionThread(QtCore.QThread):
    sign = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, video: str, clusters=12, model=None, min_detection=0.5, min_tracking=0.5):
        super(SolutionThread, self).__init__()
        if model is None:
            model = os.path.join(os.getcwd(), 'model', 'VGGNet12')
        print(f"model: {model}, video: {video}")
        self.solution = Solution(clusters=clusters, model=model, min_detection=min_detection, min_tracking=min_tracking)
        self.video = video

    def run(self) -> None:
        self.solution.process(self.video)
        ret = self.solution.predict()
        self.sign.emit(ret)
        # except Exception as err:
        #     self.sign.emit(np.array([]))
        #     with open('errors.txt', 'w') as f:
        #         f.write(err.__str__() + os.path.join(os.getcwd(), 'model', 'VGGNet12') + self.video)
