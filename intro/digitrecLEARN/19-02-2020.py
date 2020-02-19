"""
Using Theano backend.
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 26, 26)        320       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 24, 24)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 12, 12)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 12, 12)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               589952    
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 600,810
Trainable params: 600,810
Non-trainable params: 0
_________________________________________________________________
Epoch 1/1
25200/25200 [==============================] - 57s 2ms/step - loss: 0.2264 - accuracy: 0.9302
16800/16800 [==============================] - 12s 741us/step
test loss: 0.0844411751561399
Kinda useless.. 
 test accuracy: 0.9744642972946167
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from matplotlib import pyplot as plt
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

data = np.genfromtxt('./train.csv', delimiter=',')

X = np.delete(data,0,1)
X = np.delete(X,0,0)

X = X.reshape(X.shape[0],1,28,28)
X = X.astype('float32')
X /= 255

y = data.T[0]
y = np.delete(y,0)
y = y.T
y = np_utils.to_categorical(y,10)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4)

model = Sequential()
model.add(Convolution2D(32, (3,3), activation="relu", input_shape=(1,28,28)))
# 32 3x3 filters
model.add(Convolution2D(32, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2))) # reduces dimension "by 2"
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train,
          batch_size=32, epochs=1, 
          verbose = 1)
score = model.evaluate(X_test, y_test, verbose=1)

print('test loss:', score[0] )
print('Kinda useless.. \n test accuracy:', score[1])