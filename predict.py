import sys

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn import model_selection


flowers = [
    "1cymbidium", "2freesia", "3tulips", "4sakura",
    "5valley", "6rose", "7lily", "8sunflower", "9dahlia",
    "10gerbera", "11cyclamen", "12poinsettia"
]
num_classes = len(flowers)
image_size = 50

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(50, 50, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12))
    model.add(Activation('softmax'))

    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    # モデルのロード
    model = load_model("./flower_cnn_aug.h5")

    return model

def main():
    image = Image.open(sys.argv[1])
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image)/255

    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    print("Xみてみよう")
    print(X)
    print("model.predictみてみよう")
    print(model.predict([X]))
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    print('{0} ({1})'.format(flowers[predicted], percentage))

if __name__ == '__main__':
    main()