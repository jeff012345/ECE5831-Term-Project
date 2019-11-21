from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
import cv2

from datasets import loadData
from model import create_model
import util

##
## setup values
##

## where are the images
#data_root_folder = 'F:/ece5831/windows/Screens_256Pix/separated'
data_root_folder = 'F:/ece5831/windows/Ricky/groupings/full_urban_rural'

## within the folder, what extensions are we looking for
data_images_extension = '*.*'

##
## END setup values
##

## find the class names
class_names = [name for name in os.listdir(data_root_folder) if os.path.isdir(os.path.join(data_root_folder,name))]
print("classes = ", class_names)

## load the data
(train_images, train_labels), (test_images, test_labels) = loadData(data_root_folder, class_names, data_images_extension, even = True)

## create test images output folders
for class_name in class_names:
    util.create_dir_or_is_empty(os.path.join("training", "test_" + class_name))

## save the test images
cnt = 0
for img in test_images:
    class_name = class_names[test_labels[cnt][0]]
    img_file_path = os.path.join(os.path.join("training", "test_" + class_name), str(cnt) + ".bmp")
    cv2.imwrite(img_file_path, img)
    cnt += 1

## normalize the RGB values
train_images, test_images = train_images / 255.0, test_images / 255.0

## preview the images
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i][0]])
#plt.show()

## create the model
model = create_model(len(class_names))

#weights = tf.train.latest_checkpoint('F:/ece5831/ECE5831-Term-Project/road_classify/training')
#model.load_weights(weights)

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(train_images, train_labels, epochs=50, 
                    validation_data=(test_images, test_labels),
                    callbacks=[cp_callback])

model.save('training/trained_model.h5') 

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
