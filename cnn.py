import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models 
import cv2
import os
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt
datapath = r"C:\Users\hp\OneDrive\Desktop\Handwritten letters.v4i.folder\train"
def image_processing(datapath , target_size = (32, 32)):
  data=[]
  labels=[]
  for folder in os.listdir(datapath):
    folder_path = os.path.join(datapath,folder)
    for file in os.listdir(folder_path):
      image = os.path.join(folder_path, file)
      img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
      if img is not None:
       img = cv2.resize(img, target_size)
       img = cv2.GaussianBlur(img, (3, 3), 0)
       img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
       img = img / 255.0
       data.append(img)
       labels.append(folder)
  return np.array(data), np.array(labels)
data, labels = image_processing(datapath)
labels_encoding = LabelEncoder()
encoded_labels = labels_encoding.fit_transform(labels)
num_classes = len(np.unique(labels))
print(f"Number of classes: {num_classes}")
indices = np.arange(len(data))
np.random.shuffle(indices)
data = data[indices]
encoded_labels = encoded_labels[indices]
train_end = int(0.6 * len(indices))
valid_end = int(0.8 * len(indices))
train, valid, test = (
  indices[:train_end] , 
  indices[train_end : valid_end] , 
  indices[valid_end:]
)

(x_train , y_train), (x_test, y_test) , (x_valid, y_valid) = (
  (data[train], encoded_labels[train]) 
  , (data[test], encoded_labels[test]) 
  , (data[valid], encoded_labels[valid]) )

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dropout(0.8),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.002)),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')

])

model.compile(optimizer= 'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'])

history = model.fit(x_train, y_train,
  epochs= 30,
  batch_size=64,
  validation_split=0.3,
  verbose=2)


print(history)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"Test accuracy = {test_acc:.3f}")
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='valid')
plt.legend()
plt.title('Accuracy')
plt.show()