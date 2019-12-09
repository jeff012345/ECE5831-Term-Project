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
data_root_folder = 'F:/ece5831/ECE5831-Term-Project/road_classify/training/corrected_images/test_images'

## within the folder, what extensions are we looking for
data_images_extension = '*.*'

## h5 file of model
model_filepath = "F:/ece5831/ECE5831-Term-Project/road_classify/training/corrected_images/trained_model.h5"

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

## make the ROC curve 
## https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
n_classes = 2

y_test = []
for label in train_labels:
    if label == 0:
        y_test.append([1,0])
    else:
        y_test.append([0,1])
y_test = np.asarray(y_test)

y_score = model.predict(train_images)

from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
