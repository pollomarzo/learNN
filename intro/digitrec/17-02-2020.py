"""
- Keras
- Using Theano backend.
- Flatten() layer.

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 1, 28, 512)        14848     
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 28, 512)        0         
_________________________________________________________________
dense_2 (Dense)              (None, 1, 28, 512)        262656    
_________________________________________________________________
dropout_2 (Dropout)          (None, 1, 28, 512)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 14336)             0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                143370    
=================================================================
Total params: 420,874
Trainable params: 420,874
Non-trainable params: 0
_________________________________________________________________
Epoch 1/1
28140/28140 [==============================] - 68s 2ms/step - loss: 0.2612 - accuracy: 0.9204
13860/13860 [==============================] - 22s 2ms/step
test loss: 0.143684159071436
Kinda useless.. 
test accuracy: 0.9575757384300232
 """


import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import csv
from matplotlib import pyplot as plt
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

data = np.genfromtxt('./train.csv', delimiter=',')


X = np.delete(data,0,1)
X = np.delete(X,0,0)
# plt.imshow(np.reshape(X[1],[28,28]))
# plt.show()

X = X.reshape(X.shape[0],1,28,28)
X = X.astype('float32')
X /= 255

y = data.T[0]
y = np.delete(y,0)
y = y.T
y = np_utils.to_categorical(y,10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(1,28,28)))
model.add(Dropout(0.2)) # randomly drop units (along with their connections) from the 
                        # neural network during training. This prevents units from 
                        # co-adapting too much.
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax')) 
"""
takes as input a vector of K real numbers, and normalizes it into a probability 
distribution consisting of K probabilities proportional to the exponentials 
of the input numbers. That is, prior to applying softmax, some vector components
could be negative, or greater than one; and might not sum to 1; but after applying
softmax, component will be in the interval ( 0 , 1 ) , and the components will 
add up to 1, so that they can be interpreted as probabilities. Furthermore, the
larger input components will correspond to larger probabilities.
"""

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', #then check with RMSprop()
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=32, epochs=1, 
          verbose = 1)
score = model.evaluate(X_test, y_test, verbose=1)

print('test loss:', score[0] )
print('Kinda useless.. \n test accuracy:', score[1])