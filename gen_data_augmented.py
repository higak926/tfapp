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
num_testdata = 5

# 画像の読み込み

X_train = []
X_test = []
Y_train = []
Y_test = []

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
        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20, 25, 5):
                # 回転
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # 反転
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)


        # X.append(data)
        # Y.append(index)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)
print("値をみてみよう")
# print(X_train)
# print(X_test)
print(y_train)
print(y_test)

# X = np.array(X)
# Y = np.array(Y)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./flower_aug.npy", xy)
