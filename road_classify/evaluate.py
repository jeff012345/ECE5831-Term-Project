from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import datasets
from model import create_model

##
## setup values

##

## where are the images
#data_root_folder = 'F:/ece5831/windows/Screens_256Pix/separated/2_class'
data_root_folder = 'F:/ece5831/windows/Ricky/groupings/separated_2class'

## within the folder, what extensions are we looking for
data_images_extension = '*.*'

## h5 file of model
model_filepath = "F:/ece5831/windows/models/jeff_200epochs/trained_model.h5"

##
## END setup values
##

## find the class names
class_names = [name for name in os.listdir(data_root_folder) if os.path.isdir(os.path.join(data_root_folder,name))]
classes = list(range(0, len(class_names)))
classesDict = {}
i = 0
for c in class_names:
    classesDict[c] = i
    i+=1
print("classes = ", class_names)
print("classes = ", classes)
print("classesDict = ", classesDict)

## load the data
(train_images, train_labels) = datasets.load_data_no_split(data_root_folder, class_names, data_images_extension, even=True)

## normalize the RGB values
train_images = train_images / 255.0

## create the model
print("load model");
model = tf.keras.models.load_model(model_filepath)

results = model.evaluate(train_images, train_labels, verbose=2)
print('test loss, test acc:', results)

predictions = model.predict(train_images)

## use the max value from array for each prediction
predictions = np.argmax(predictions, axis = 1)

## convert the label to the class
#train_classes = list(map(lambda l: classesDict[l], train_labels))

confusion_matrix = tf.math.confusion_matrix(
        train_labels,
        predictions
    ).numpy()

print("confusion_matrix")
print(confusion_matrix)

con_mat_norm = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
 
con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)


figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot = True, cmap = plt.cm.Blues, square = True)
#plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()