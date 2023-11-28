import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten

from model import Model_Classification_Numbers

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mnist = tf.keras.datasets.mnist

(training_data, training_labels), (test_data, test_labels) = mnist.load_data()




# is3train = np.array([1 if val==3 else 0 for val in training_labels])
# is3test = np.array([1 if val==3 else 0 for val in test_labels])

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=tf.keras.metrics.SparseCategoricalAccuracy()
)

model.fit(training_data,
          training_labels, 
          epochs=50,
          batch_size=10)

model.evaluate(test_data, test_labels)

predictions = model.predict(test_data)
predictions = [np.argmax(val) for val in predictions]

image_index=30

plt.title('True: {} \nPredict: {}'.format(test_labels[image_index], predictions[image_index]))
plt.imshow(test_data[image_index], cmap='Greys')

plt.show()

accuracy = accuracy_score(test_labels, predictions)
print(accuracy)

for i in range(10):
    if test_labels[i] != predictions[i]:
        print(test_labels[i], 'test_labels[i]')
        print(predictions[i], 'predictions[i]')