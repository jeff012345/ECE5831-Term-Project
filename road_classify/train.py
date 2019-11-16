from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib.pyplot as plt
import os

from datasets import loadData
from model import create_model

## setup values
data_root_folder = 'F:/ece5831/ECE5831-Term-Project/road_classify/data_paved_unpaved'
## END setup values

## find the class names
class_names = [name for name in os.listdir(data_root_folder) if os.path.isdir(os.path.join(data_root_folder,name))]
print("classes = ", class_names)

## load the data
(train_images, train_labels), (test_images, test_labels) = loadData(data_root_folder, class_names)

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
model = create_model()

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels),
                    callbacks=[cp_callback])

model.save('my_model.h5') 

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
