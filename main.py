import numpy as np
from resource import Solution
import os


labels = ["Opaque", "Red", "Green", "Yellow", "Bright", "Light-blue", "Colors", "Red", "Women", "Enemy", "Son", "Man", "Away", "Drawer", "Born", "Learn",
          "Call", "Skimmer", "Bitter", "Sweet milk", "Milk", "Water", "Food", "Argentina", "Uruguay", "Country", "Last name", "Where", "Mock", "Birthday", "Breakfast", "Photo",
          "Hungry", "Map", "Coin", "Music", "Ship", "None", "Name", "Patience", "Perfume", "Deaf", "Trap", "Rice", "Barbecue", "Candy", "Chewing-gum", "Spaghetti",
          "Yogurt", "Accept", "Thanks", "Shut down", "Appear", "To land", "Catch", "Help", "Dance", "Bathe", "Buy", "Copy", "Run", "Realize", "Give", "Find"]

if __name__ == '__main__':
    video = "dataset/LSA64/005_001_003.mp4"
    s = Solution(model=os.path.join(os.getcwd(), 'model', 'VGGNet12'), clusters=12)
    s.process(video)
    print(s.coords.shape)
    ret = s.predict()
    if ret is not None and ret.size:
        ret = ret[0]
        for k in np.argsort(ret)[::-1][:5]:
            print(f"Type: {k+1}, Probability: {ret[k]}")
        print(f"The result of {video} is {np.argmax(ret)+1}, {labels[np.argmax(ret)]}")
