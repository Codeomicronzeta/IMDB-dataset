# -*- coding: utf-8 -*-


from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

train_data.shape
train_labels.shape

max([max(sequence) for sequence in train_data])

word_index = imdb.get_word_index()                                                      #word_index is dictionary mapping words to integer index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])        # reverse_word_index reverses mapping , maps from integer to words
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

"""Vectorizing the data: One Hot encoding the integer sequences"""
def vectorize_sequences(sequences, dimension = 10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results
  
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

"""Setting up the model"""
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

"""Compiling the model"""
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])

model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size = 512, validation_data = (x_val, y_val))
results = model.evaluate(x_train, y_train)

model.fit(x_test, y_test, epochs = 4, batch_size = 512)
results = model.evaluate(x_test, y_test)









