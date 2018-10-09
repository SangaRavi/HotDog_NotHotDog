import os
from random import  shuffle
from tqdm import tqdm
import cv2
import numpy as np

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
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)
    return train_images

def data_test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        test_images.append([np.array(img), one_hot_label(i)])
    shuffle(test_images)
    return test_images

