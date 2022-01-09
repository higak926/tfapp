from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

flowers = [
    "1cymbidium", "2freesia", "3tulips", "4sakura",
    "5valley", "6rose", "7lily", "8sunflower", "9dahlia",
    "10gerbera", "11cyclamen", "12poinsettia"
]

num_classes = len(flowers)
image_size = 50

# 画像の読み込み

X = []
Y = []
for index, flower in enumerate(flowers):
    photos_dir = "./" + flower
    files = glob.glob(photos_dir + "/*jpg")
    for i, file in enumerate(files):
        print(i)
        if i >= 10: break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./flower.npy", xy)
