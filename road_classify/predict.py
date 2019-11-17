import os
import shutil
import ntpath
from model import create_model
import numpy as np
import tensorflow as tf
import cv2
from util import create_dir_or_is_empty

## config
checkpoint_dir = None #'F:/ece5831/ECE5831-Term-Project/road_classify/saved_models/100'
model_filepath = 'F:/ece5831/ECE5831-Term-Project/road_classify/saved_models/new/my_model.h5'
predict_data_root_dir = 'F:/ece5831/windows/Screens_256Pix/data'
predections_output = "F:/ece5831/windows/Screens_256Pix/predictions"
## END config

## Gbobals Rock!
output_folders = {}

## function for loading the images
def loadImagesData(files):
    images = []
    filepaths = []        

    for file in files:        
        filepaths.append(file)
        image = cv2.imread(file)
        image = image.astype(np.float32)
        images.append(image)

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

## create folders in output folder for each class
def make_class_folders(predictions):
    if len(output_folders):
        return 

    for pred_class in range(0, predictions.shape[1]):
        name = className(pred_class)
        output_folders[name] = os.path.join(predections_output, name)
        os.mkdir(output_folders[name])

def copy_predicted_images(predictions, filepaths):
    make_class_folders(predictions)

    ## copy the source images into their prediction folder
    i = 0
    for p in predictions:
        pred_class = p.argmax()
        name = className(pred_class)
        filename = ntpath.basename(filepaths[i])

        shutil.copyfile(filepaths[i], os.path.join(output_folders[name], filename))
        i += 1

def predict(model, files):
    ## load the images to predict
    (predict_images, filepaths) = loadImagesData(files)

    ## predict
    predictions = model.predict(predict_images)

    ## copy the predections into folders sorted by class
    copy_predicted_images(predictions, filepaths)

def gather_image_filepaths():
    predict_images = []

    for root, dirs, files in os.walk(predict_data_root_dir):
        for file in files:
            if file.endswith(".jpg"):
                predict_images.append(os.path.join(root, file))

    return predict_images

def run():
    ## load the model
    model = loadModel()

    ## predict in batches so we don't store all the images in RAM
    predict_images = gather_image_filepaths()

    num_of_images = len(predict_images)

    batch_size = 1000
    count = 0
    while len(predict_images) > batch_size:        
        predict(model, predict_images[:batch_size])
        predict_images = predict_images[batch_size:]
        count += batch_size
        print("Completed %s of %s" % (count, num_of_images))

    ## remainder
    predict(model, predict_images)
    print("Completed")

## make output folder
create_dir_or_is_empty(predections_output)

# run the predictions
run()