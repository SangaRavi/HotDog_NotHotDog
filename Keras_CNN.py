import cv2
import  numpy as np
import os
from random import  shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt


train_data = './Train'
test_data = './Test'

def one_hot_label(img):
    label = img.split('.')[0]
    if label == 'HotDog':
        ohl = np.array([1,0])
    elif label == 'Not_HotDog':
        ohl = np.array([0,1])
    return ohl


def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(64,64))
        train_images.append([np.array(img),one_hot_label(i)])
    shuffle(train_images)
    return train_images

def data_test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data,i)
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        test_images.append([np.array(img), one_hot_label(i)])
    shuffle(test_images)
    return  test_images


training_images = train_data_with_label()
tr_img_data = np.array([i[0] for i in
                        training_images]).reshape(-1, 64, 64, 3)
tr_lbl_data = np.array([i[1] for i in training_images])

test_images = data_test_data_with_label()

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


model = Sequential()

model.add(InputLayer(input_shape=[64,64,3]))
model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters= 100, kernel_size=5, strides=1, padding= 'same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='softmax'))
optimizer = Adam(lr=1e-3) 


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=tr_img_data, y=tr_lbl_data, epochs=50, batch_size=50)
model.summary()

fig = plt.figure(figsize=(14, 14))

for cnt, data in enumerate(test_images[:]):
    y= fig.add_subplot(6, 5, cnt+1)
    img = data[0]
    data = img.reshape(1, 64, 64, 3)
    model_out = model.predict([data])

    if np.argmax(model_out) == 1:
        str_label = 'Not HotDog'
    else:
        str_label = 'HotDog'

    y.imshow(img)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()