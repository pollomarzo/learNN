"""
Implements a BLSTM convolutional neural network.

Based on modelwrapper to avoid unnecessary confusion, only overrides build method
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 1000, 100)         12554200  
_________________________________________________________________
lstm_1 (LSTM)                (None, 1000, 100)         80400     
_________________________________________________________________
dropout (Dropout)            (None, 1000, 100)         0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 1000, 32)          16032     
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 500, 32)           0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 500, 64)           6208      
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 250, 64)           0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 250, 64)           256       
_________________________________________________________________
flatten (Flatten)            (None, 16000)             0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               4096256   
_________________________________________________________________
dense_4 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 129       
=================================================================
Total params: 16,786,377
Trainable params: 4,232,049
Non-trainable params: 12,554,328

Train on 16640 samples, validate on 2080 samples
Epoch 1/3
16640/16640 [==============================] - 365s 22ms/sample - loss: 0.7863 - accuracy: 0.7533 - val_loss: 0.4665 - val_accuracy: 0.7639
Epoch 2/3
16640/16640 [==============================] - 349s 21ms/sample - loss: 0.3016 - accuracy: 0.8742 - val_loss: 0.6964 - val_accuracy: 0.7183
Epoch 3/3
16640/16640 [==============================] - 350s 21ms/sample - loss: 0.2327 - accuracy: 0.9037 - val_loss: 0.3881 - val_accuracy: 0.8317
"""
from tensorflow import keras as K
import embed_utils
from modelwrapper import ModelWrapper, TRAINABLE_GLOVE


class BlstmCnnUtility(ModelWrapper):
    """
    just shutting up pylint
    """
    ##############################################
    # Constructor
    # def __init__(self, data, labels, sequence_length, GLOVE=False, glove_dir=None):
    #    super().__init__(data, labels, sequence_length, GLOVE, glove_dir)

    # BUILD MODEL, COMPILE
    def build_model(self, embedding_size,
                    filter_sizes, num_filters, num_cells=100):
        print("building model...")

        # embed->blstm->cnn->pool->out
        self.model = K.Sequential()
        # EMBEDDING

        if self.GLOVE:
            print("\tcreating embedding matrix with glove values...")
            print('\t', end='')

            # maybe should try random vs zeros. you never know [scratch that]
            embedding_matrix = embed_utils.pretrained_embedding_matrix(
                embedding_size, self.tokenizer.word_index, self.embeddings_index)
            print("\tembedding matrix created!")
        else:
            # create embedding_matrix with word2vec
            pass
        self.model.add(K.layers.Embedding(len(self.tokenizer.word_index) + 1, embedding_size,
                                          input_length=self.sequence_length, weights=[
                                              embedding_matrix],
                                          trainable=False))

        # self.model.add(K.layers.Bidirectional(K.layers.LSTM(
        #   units=num_cells, dropout=0.4, recurrent_dropout=0.4,return_sequences=True)))

        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). to be checked again... convolution wants 3-dim i think
        # noot sure i understand this fully, but just know it's right
        self.model.add(K.layers.LSTM(units=num_cells, input_shape=(None, embedding_size),
                                     dropout=0.4, return_sequences=True, batch_size=1))
        self.model.add(K.layers.Dropout(0.2))

        print("\t adding convolution layers...")
        for i, size in enumerate(filter_sizes):
            self.model.add(K.layers.Conv1D(
                filters=num_filters[i], kernel_size=size,
                padding='same', activation='relu'))
            self.model.add(K.layers.MaxPool1D(pool_size=2))
        print("\t ..added convolution!")

        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dense(256))
        self.model.add(K.layers.Dense(128))
        #self.model.add(K.layers.Dense(128, activation='relu'))
        self.model.add(K.layers.Dense(1, activation='sigmoid'))

        # maybe try with last layer Dense(1, activation='sigmoid')

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy'])
        print("..model built!")

        self.model.summary()
