"""

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 64)          1280000   
_________________________________________________________________
lstm_1 (LSTM)                (None, 128)               98816     
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 129       
=================================================================
Total params: 1,378,945
Trainable params: 1,378,945
Non-trainable params: 0

Epoch 3/3
25000/25000 [==============================] - 133s 5ms/step - loss: 0.3007 - accuracy: 0.8792 - val_loss: 0.4058 - val_accuracy: 0.8289
25000/25000 [==============================] - 28s 1ms/step
Test accuracy: 0.8288800120353699

OVERFIT
"""

import numpy as np
import keras as K
from sklearn.model_selection import train_test_split

from keras.datasets import imdb #let's start with this one

max_features = 20000
maxlen = 80
batch_size = 32
embed_vec_len = 64

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
# Each word of a review is converted into a unique integer ID where 4 is used for the most 
# frequent word in the training data ("the"), 5 is used for the second most common word 
# ("and") and so on. A value of 0 is reserved for padding. A value of 1 is used to indicate
# the beginning of a sequence/sentence. Words that aren't among the most common 20,000 
# words are assigned a value of 2 and are called out-of-vocabulary (OOV) words. A value 
# of 3 is reserved for custom usage.

X_train = K.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = K.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

model = K.Sequential()
model.add(K.layers.Embedding(input_dim=max_features,output_dim=embed_vec_len))
# Although it is possible to feed integer-encoded sentences directly to an LSTM network, 
# better results are obtained by converting each integer ID into a vector of real values. 
# For example, the word "the" has index value 4 but will be converted to a vector like 
# (0.1234, 0.5678, . . 0.3572). This is called a word embedding. The idea is to construct 
# vectors so that similar words, such as "man" and "male," have vectors that are numeri-
# cally close. The length of the vector must be determined by trial and error. This
# uses size 32 but for most problems a vector size of 100 to 500 is more common. There 
# are three main ways to create word embeddings for an LSTM network. One approach is to
# use an external tool such as Word2Vec to create the embeddings. A second approach is
# to use a set of pre-built embeddings such as GloVe ("global vectors for word repre-
# sentation"), which is constructed using the text of Wikipedia. The demo program uses
# the third approach, which is to create embeddings on the fly. These embeddings will
# be specific to the vocabulary of the problem scenario. 
# https://visualstudiomagazine.com/articles/2018/11/01/sentiment-analysis-using-keras.aspx

model.add(K.layers.LSTM(units=128,dropout=0.4, recurrent_dropout=0.4))
model.add(K.layers.Dropout(0.2))
model.add(K.layers.Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train,y_train,batch_size=batch_size,epochs=3,
          verbose=1,validation_data=(X_test,y_test))
score = model.evaluate(X_test,y_test,batch_size=batch_size)
print('Test accuracy:', score[1])

model.save("./models/sentanalysis.h5")