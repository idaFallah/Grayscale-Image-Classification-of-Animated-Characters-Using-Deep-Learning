# import libraries
import cv2
import os
import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

tf.__version__

from google.colab import drive
drive.mount('/content/drive')

path = '/content/drive/MyDrive/Computer Vision/Datasets/homer_bart_1.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

directory = '/content/homer_bart_1'
files = [os.path.join(directory, f) for f in sorted(os.listdir(directory))]
print(files)

width, height = 128, 128

images = []
classes = []

image.shape

128 * 128 * 3, 128 * 128  # nodes in input layer of NN

for image_path in files:
  #print(image_path)

  try:   # to debug
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]
  except:
    continue
  image = cv2.resize(image, (width, height))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  cv2_imshow(image)  # we cannot show an image after we've converted it into a vector

  image = image.ravel()   # to turn the matrix of pixels into vector of pixels
  print(image.shape)

  images.append(image)

  image_name = os.path.basename(os.path.normpath(image_path))

  if image_name.startswith('b'):
    class_name = 0
  else:
    class_name = 1

  classes.append(class_name)
  print(class_name)

# when working with grayscale images -> all the values of RGB are the same

images

classes

type(images), type(classes)

x = np.asarray(images)
y = np.asarray(classes)

type(x), type(y)

x.shape

y.shape

x[0].reshape(width, height).shape  # reshaping the vector of image to a matrix to be able to show it

cv2_imshow(x[0].reshape(width, height))

sns.countplot(y)

np.unique(y, return_counts=True)

x[0].max(), x[0].min()

# normalizing the data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x[0].max(), x[0].min()

# splitting the train/ test set

from sklearn.model_selection import train_test_split

x_tarin, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

x_tarin.shape, y_train.shape

x_test.shape, y_test.shape

# building & training the NN via tenserflow

128 * 128

(16384+2)/2   # estimated num of units in hidden layer

network1 = tf.keras.models.Sequential()
network1.add(tf.keras.layers.Dense(input_shape=(16384,), units=8193, activation='relu'))
network1.add(tf.keras.layers.Dense(units=8193, activation='relu'))
network1.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))   # since it's binary classification, it's better to use sigmoid

network1.summary()

# training the NN

network1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

history = network1.fit(x_tarin, y_train, epochs=50)

# evaluating the NN

history.history.keys()

plt.plot(history.history['loss']);  # not all 50 epochs are needed

plt.plot(history.history['accuracy']);

x_test, x_test.shape

predictions = network1.predict(x_test)
predictions

# 0/ false = bart(little boy)
# 1/ true = homer (man)

predictions = (predictions > 0.5)  # setting a threshold of 0.5 to group the data into true/false

predictions

y_test   # expected output

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
cm

sns.heatmap(cm, annot=True);

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# saving & loading the model for when i wanna use it again

model_json = network1.to_json()
with open('network1.json', 'w') as json_file:
  json_file.write(model_json)

from keras.models import save_model
network1_saved = save_model(network1, 'weights1.hdf5')

with open('network1.json') as json_file:
  json_saved_model = json_file.read()
json_saved_model

network1_loaded = tf.keras.models.model_from_json(json_saved_model)
network1_loaded.load_weights('/content/weights1.hdf5')
network1_loaded.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

network1_loaded.summary()

# classifying one single image

x[0].shape

test_image = x_test[34]
cv2_imshow(test_image.reshape(width, height))

# since it's showing a black pic after normalizing, we need to convert it to its original format

test_image = scaler.inverse_transform(test_image.reshape(1, -1))

test_image

cv2_imshow(test_image.reshape(width, height))

network1_loaded.predict(test_image)[0]

if network1_loaded.predict(test_image)[0][0] < 0.5 :
  print('Bart')
else:
  print('Homer')

















