import cv2
import numpy as np
import math
import os
import glob
import random
import copy

## load images from folder
def load_images_data(path, extension, max):
    images = []

    for file in glob.glob(os.path.join(path, extension)):
        image = cv2.imread(file)
        image = image.astype(np.float32)
        images.append(image)    
    
    random.shuffle(images)
    if max is not None:
        return images[0:max]
    return images

## counts the number of images for each class and returns the minimum
def find_min_num_class_images(data_root_folder, class_names, extension = '*.*'):
    min = 9999999999

    for name in class_names:
        class_folder = os.path.join(data_root_folder, name)
        count = len(glob.glob(os.path.join(class_folder, extension)))

        if count < min:
            min = count

    return min

## load images from the folder
def loadData(data_root_folder, class_names, extension = '*.*', even = False):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []

    if even is True:
        min_class_images = find_min_num_class_images(data_root_folder, class_names, extension = '*.*')
        print("Minimum number of images from the classes = ", min_class_images)
    else:
        min_class_images = None

    classNum = 0
    for name in class_names:
        images = load_images_data(os.path.join(data_root_folder, name), extension, min_class_images)
        
        (train, test) = split_data(images)
        
        ## add each training      
        train_images = train_images + train
        train_labels = train_labels + ([[classNum]] * len(train)) ## repeat the name
        
        test_images = test_images + test        
        test_labels = test_labels + ([[classNum]] * len(test)) ## repeat the name

        print("Loaded %s train images and %s test images for class '%s'" % (len(train), len(test) , name))
        classNum += 1    

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return (train_images, train_labels), (test_images, test_labels)

def load_data_no_split(data_root_folder, class_names, extension = '*.*', even = False):
    images = []
    
    if even is True:
        min_class_images = find_min_num_class_images(data_root_folder, class_names, extension = '*.*')
        print("Minimum number of images from the classes = ", min_class_images)
    else:
        min_class_images = None

    class_num = 0
    for name in class_names:
        class_images = load_images_data(os.path.join(data_root_folder, name), extension, min_class_images)

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
