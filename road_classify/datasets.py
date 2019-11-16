import cv2
import numpy as np
import math
import os
import glob
import random

## load images from folder
def loadImagesData(path):
    images = []

    for file in glob.glob(os.path.join(path, '*.jpg')):
        for i in range(1, 3):
            image = cv2.imread(file)
            image = image.astype(np.float32)
            images.append(image)
    
    random.shuffle(images)   
    return images

## load images from the folder
def loadData(data_root_folder, class_names):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []

    classNum = 0
    for name in class_names:
        images = loadImagesData(os.path.join(data_root_folder, name))

        (train, test) = splitData(images)
        
        train_images = train_images + train
        test_images = test_images + test
    
        train_labels = train_labels + ([[classNum]] * len(train)) ## repeat the name
        test_labels = test_labels + ([[classNum]] * len(test)) ## repeat the name

        print("Loaded %s images for class '%s'" % (len(images), name))
        classNum += 1

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return (train_images, train_labels), (test_images, test_labels)

## split the data 80/20 for test and train
def splitData(images):
    split = math.floor(len(images) * 0.8)
    train = images[:split]
    test = images[split:]
    
    return (train, test)
