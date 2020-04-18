"""
Implements a BLSTM convolutional neural network.

Based on modelwrapper to avoid unnecessary confusion, only overrides build method
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 1000, 100)         12554200  
_________________________________________________________________
conv1d (Conv1D)              (None, 1000, 32)          16032     
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 500, 32)           0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 500, 64)           6208      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 250, 64)           0         
_________________________________________________________________
lstm (LSTM)                  (None, 100)               66000     
_________________________________________________________________
batch_normalization (BatchNo (None, 100)               400       
_________________________________________________________________
dense (Dense)                (None, 256)               25856     
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
=================================================================
Total params: 12,701,721
Trainable params: 147,321
Non-trainable params: 12,554,400
_________________________________________________________________
Train on 16640 samples, validate on 2080 samples
Epoch 1/3
16640/16640 [==============================] - 92s 6ms/sample - loss: 0.3624 - accuracy: 0.8348 - val_loss: 0.4225 - val_accuracy: 0.8293
Epoch 2/3
16640/16640 [==============================] - 93s 6ms/sample - loss: 0.2238 - accuracy: 0.9089 - val_loss: 0.1896 - val_accuracy: 0.9274
Epoch 3/3
16640/16640 [==============================] - 95s 6ms/sample - loss: 0.1785 - accuracy: 0.9257 - val_loss: 0.2534 - val_accuracy: 0.9091
"""
from tensorflow import keras as K
import embed_utils
from modelwrapper import ModelWrapper, TRAINABLE_GLOVE


class CnnBlstmUtility(ModelWrapper):
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
        if self.GLOVE:
            attr = "GLOVE"
        else:
            attr = "word2vec"
        self.name = f"{attr}_{len(num_filters)}xConv_LSTM{num_cells}cell_fakenews"
        print("building model...")

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
                                          trainable=TRAINABLE_GLOVE))

        # self.model.add(K.layers.Bidirectional(K.layers.LSTM(
        #   units=num_cells, dropout=0.4, recurrent_dropout=0.4,return_sequences=True)))

        print("\t adding convolution layers...")
        for i, size in enumerate(filter_sizes):
            self.model.add(K.layers.Conv1D(
                filters=num_filters[i], kernel_size=size,
                padding='same', activation='relu'))
            self.model.add(K.layers.MaxPool1D(pool_size=2))
        print("\t ..added convolution!")

        self.model.add(K.layers.LSTM(
            units=num_cells, dropout=0.2))
        # self.model.add(K.layers.Dropout(0.2))

        self.model.add(K.layers.BatchNormalization())
        # self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dense(256))
        self.model.add(K.layers.Dense(128))
        # self.model.add(K.layers.Dense(128, activation='relu'))
        self.model.add(K.layers.Dense(1, activation='sigmoid'))

        # K.optimizers.Adam(lr=0.0001) to experiment with convergence speeds
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        print("..model built!")

        self.model.summary()
