import os
import shutil
import ntpath
from model import create_model
import numpy as np
import tensorflow as tf
import cv2

checkpoint_dir = None #'F:/ece5831/ECE5831-Term-Project/road_classify/saved_models/100'
model_filepath = 'F:/ece5831/ECE5831-Term-Project/road_classify/saved_models/new/my_model.h5'
predict_data_root_dir = 'F:/ece5831/windows/Screens_256Pix'
predections_output = "F:/ece5831/windows/Screens_256Pix/predictions"

## function for loading the images
def loadImagesData(path):
    images = []
    filepaths = []
        
    cnt = 0
    for root, dirs, files in os.walk(path):
        if cnt >= 100:
            break

        for file in files:
            if file.endswith(".jpg"):
                file = os.path.join(root, file)
                filepaths.append(file)
                image = cv2.imread(file)
                image = image.astype(np.float32)
                images.append(image)
                cnt += 1

    ## normalize the input because that's how the model was trained
    images = np.array(images) / 255
    return (images, filepaths)

def className(pred_class):
    return "class_" + str(pred_class)

def loadModel():
    if checkpoint_dir is None:
        return tf.keras.models.load_model(model_filepath)

    model = create_model()

    weights = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(weights)

    return model

## make output folder
try:
    os.mkdir(predections_output)
except OSError:
    if len(os.listdir(predections_output)) != 0:
        print("Error: Output folder is not empty.")
        quit()    

## load the model
model = loadModel()

## load the images to predict
(predict_images, filepaths) = loadImagesData(predict_data_root_dir)

## predict
predictions = model.predict(predict_images)
print("predictions shape = ", predictions.shape)

## create folders in output folder for each class
output_folders = {}
for pred_class in range(0, predictions.shape[1]):
    name = className(pred_class)
    output_folders[name] = os.path.join(predections_output, name)
    os.mkdir(output_folders[name])

## copy the source images into their prediction folder
i = 0
for p in predictions:
    pred_class = p.argmax()
    name = className(pred_class)
    filename = ntpath.basename(filepaths[i])
    print(filepaths[i], os.path.join(output_folders[name], filename))
    shutil.copyfile(filepaths[i], os.path.join(output_folders[name], filename))
    i += 1