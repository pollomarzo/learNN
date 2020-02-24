#testing graph (not working. will check mistake next model) and memory (brain). went well!

import tensorflow as tf
import matplotlib.pyplot as plt

(X_train,y_train), (X_test,y_test) = tf.keras.datasets.mnist.load_data()

y_test = tf.keras.utils.to_categorical(y_test)
y_train = tf.keras.utils.to_categorical(y_train)

X_train = tf.reshape(X_train,(60000,28,28,1))
X_test = tf.reshape(X_test,(X_test.shape[0],28,28,1))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Convolution2D(32,(3,3), activation = 'relu',input_shape=(28,28,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['accuracy'])
model.summary()
history = model.fit(X_train,y_train,
          batch_size = 32, epochs = 1)

score = model.evaluate(X_test, y_test, verbose=1)


plt.figure()
plt.plot(history.history['accuracy'])
plt.show()