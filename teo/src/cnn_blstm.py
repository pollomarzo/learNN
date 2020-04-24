"""
Implements a BLSTM convolutional neural network.

Based on modelwrapper to avoid unnecessary confusion, only overrides build method

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
bidirectional (Bidirectional (None, 200)               132000    
_________________________________________________________________
batch_normalization (BatchNo (None, 200)               800       
_________________________________________________________________
dense (Dense)                (None, 256)               51456     
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
=================================================================
"""
from tensorflow import keras as K
from modelwrapper import ModelWrapper


class CnnBlstmUtility(ModelWrapper):
    """
    just shutting up pylint
    """
    ##############################################
    # Constructor
    # def __init__(self, data, labels, sequence_length, GLOVE=False, EMBED_FILE=None):
    #    super().__init__(data, labels, sequence_length, GLOVE, EMBED_FILE)

    # BUILD MODEL, COMPILE
    def build_model(self, embedding_size,
                    filter_sizes, num_filters, num_cells=100):
        # just giving a name to the model. have to find a better spot
        self.name = f"{self.embed_type}_{len(num_filters)}xConv_LSTM{num_cells}cell_fakenews"

        print("building model...")
        self.model = K.Sequential()

        self.model.add(self.embedding.build_embedding())

        print("\t adding convolution layers...")
        for i, size in enumerate(filter_sizes):
            self.model.add(K.layers.Conv1D(
                filters=num_filters[i], kernel_size=size,
                padding='same', activation='relu'))
            self.model.add(K.layers.MaxPool1D(pool_size=2))
        print("\t ..added convolution!")

        self.model.add(K.layers.Bidirectional(K.layers.LSTM(
            units=num_cells, dropout=0.2)))
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
