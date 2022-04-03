from ui.window import Ui_MainWindow
from resource import Solution
from PyQt5 import QtWidgets
import numpy as np
import sys
import os


labels = ["Opaque", "Red", "Green", "Yellow", "Bright", "Light-blue", "Colors", "Red", "Women", "Enemy", "Son", "Man", "Away", "Drawer", "Born", "Learn",
          "Call", "Skimmer", "Bitter", "Sweet milk", "Milk", "Water", "Food", "Argentina", "Uruguay", "Country", "Last name", "Where", "Mock", "Birthday", "Breakfast", "Photo",
          "Hungry", "Map", "Coin", "Music", "Ship", "None", "Name", "Patience", "Perfume", "Deaf", "Trap", "Rice", "Barbecue", "Candy", "Chewing-gum", "Spaghetti",
          "Yogurt", "Accept", "Thanks", "Shut down", "Appear", "To land", "Catch", "Help", "Dance", "Bathe", "Buy", "Copy", "Run", "Realize", "Give", "Find"]

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())
    # video = "dataset/valid/003_001_001.mp4"
    # s = Solution(model=os.path.join(os.getcwd(), 'model', 'VGGNet12'), clusters=12, min_detection=0.8)
    # s.process(video)
    # print(s.coords.shape)

