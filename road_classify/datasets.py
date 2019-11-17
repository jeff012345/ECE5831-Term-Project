import cv2
import numpy as np
import math
import os
import glob
import random
import copy

## load images from folder
def load_images_data(path, extension):
    images = []

    for file in glob.glob(os.path.join(path, extension)):
        image = cv2.imread(file)
        image = image.astype(np.float32)
        images.append(image)    
     
    return images

## load images from the folder
def loadData(data_root_folder, class_names, extension = '*.*'):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []

    classNum = 0
    for name in class_names:
        images = load_images_data(os.path.join(data_root_folder, name), extension)

        (train, test) = split_data(images)
        
        ## add each training image 3 times in random order
        for i in range(0, 1):
            train_copy = copy.deepcopy(train)
            random.shuffle(train_copy)  
            train_images = train_images + train_copy

            train_labels = train_labels + ([[classNum]] * len(train)) ## repeat the name
        
        test_images = test_images + test        
        test_labels = test_labels + ([[classNum]] * len(test)) ## repeat the name

        print("Loaded %s images for class '%s'" % (len(images), name))
        classNum += 1    

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return (train_images, train_labels), (test_images, test_labels)

def load_data_no_split(data_root_folder, class_names, extension = '*.*'):
    images = []
    
    class_num = 0
    for name in class_names:
        class_images = load_images_data(os.path.join(data_root_folder, name), extension)

        for img in class_images:
            images.append([img, class_num])        

        print("Loaded %s images for class '%s'" % (len(class_images), name))
        class_num += 1

    
    random.shuffle(images)

    a = []
    b = []
    for element in images:
        a.append(element[0])
        b.append(element[1])

    return (np.array(a), np.array(b))

## split the data 80/20 for test and train
def split_data(images, percent = 0.8):
    split = math.floor(len(images) * percent)
    train = images[:split]
    test = images[split:]
    
    return (train, test)
