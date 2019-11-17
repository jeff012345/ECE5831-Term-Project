from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
import math

from datasets import load_data_no_split, split_data
from model import create_model

##
## setup values
##

## where are the images
data_root_folder = 'F:/ece5831/windows/Screens_256Pix/separated'

## within the folder, what extensions are we looking for
data_images_extension = '*.*'

##
## END setup values
##

## find the class names
class_names = [name for name in os.listdir(data_root_folder) if os.path.isdir(os.path.join(data_root_folder,name))]
print("classes = ", class_names)

## load the data
(images, labels) = load_data_no_split(data_root_folder, class_names, data_images_extension)

print(images.shape, labels.shape)

## normalize the RGB values
images = images / 255.0

## save off a portion for validation
split = math.floor(len(images) * 0.8)

train_images = images[:split]
train_labels = labels[:split]

test_images = images[split:]
test_labels = labels[split:]

images = None
labels = None

## create the model
model = create_model()

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

## run fit with k-folds
n_split = 10
history = None

for train_index, test_index in KFold(n_split).split(train_images):
    x_train, x_test = train_images[train_index], train_images[test_index]
    y_train, y_test = train_labels[train_index], train_labels[test_index]

    history = model.fit(x_train, y_train, epochs=5, 
                        validation_data=(x_test, y_test),
                        callbacks=[cp_callback])

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    

model.save('training/trained_model.h5') 

print("final:")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

