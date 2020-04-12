import os
import numpy as np
from tensorflow import keras as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import nltk


TRAINABLE_GLOVE = False
MAX_NB_WORDS = 200000


class BlstmCnnUtility():
    ##############################################
    # Constructor
    def __init__(self, data, labels, sequence_length, GLOVE=False, glove_dir=None):
        nltk.download('stopwords')

        print("initializing variables and tokenizing...")
        self.validation_accuracy = 0
        self.sequence_length = sequence_length
        self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = (
            None, None, None, None, None, None)
        self.model = None

        self.labels = to_categorical(np.asarray(labels), num_classes=2)

        self.data = data
        self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        self.tokenizer.fit_on_texts(self.data)

        self.data = pad_sequences(self.tokenizer.texts_to_sequences(
            self.data), maxlen=sequence_length)
        self.split_data()
        print("..done!")

        if GLOVE:
            print("loading glove...")
            self.GLOVE = GLOVE
            self.glove_dir = glove_dir
            # embeddings_index is a dict, word to glove embedding vector
            self.embeddings_index = self.loadglove(glove_dir)
            print("..done!")

    # BUILD MODEL, COMPILE
    def build_model(self, embedding_size,
                    filter_sizes, num_filters, num_cells=100):
        print("building model...")

        # embed->blstm->cnn->pool->out
        self.model = K.Sequential()
        # EMBEDDING
        # consider subs in
        # https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
        # or https://github.com/pmsosa/CS291K/blob/master/batchgen.py

        if self.GLOVE:
            print("\tcreating embedding matrix with glove values...")
            print('\t', end='')

            # maybe should try random vs zeros. you never know [scratch that]
            embedding_matrix = self.pretrained_embedding_matrix(embedding_size)
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

        # By setting return_sequences to True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). to be checked again... convolution wants 3-dim i think
        # noot sure i understand this fully, but just know it's right
        self.model.add(K.layers.LSTM(units=num_cells, input_shape=(None, embedding_size),
                                     dropout=0.4, return_sequences=True, batch_size=1))
        # self.model.add(K.layers.Dropout(0.2))

        print("\t adding convolution layers...")
        for i, size in enumerate(filter_sizes):
            self.model.add(K.layers.Conv1D(
                filters=num_filters[i], kernel_size=size,
                padding='same', activation='relu'))
            self.model.add(K.layers.MaxPool1D(pool_size=2))
        print("\t ..added convolution!")

        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dropout(0.2))
        self.model.add(K.layers.Dense(128, activation='relu'))
        self.model.add(K.layers.Dense(2, activation='softmax'))

        # maybe try with last layer Dense(1, activation='sigmoid')

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy'])
        print("..model built!")

        self.model.summary()

    # TRAIN MODEL
    def train(self, epochs=1, batch_size=64):
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size,
                       epochs=epochs, validation_data=(self.x_test, self.y_test))

    # SAVE TO FILE
    def save_model(self, save_dir='./models'):
        self.model.save(save_dir)

    # RECOMPUTE VALIDATION
    # (validation data is NEVER used, so can be used to compare different models)
    def get_validation_score(self):
        loss, self.validation_accuracy = self.model.evaluate(
            self.x_val, self.y_val)

    # PREDICT ON NEW DATA
    # honestly, it's still not done but im tired
    def predict(self, sentences, labels=None):
        to_predict = []
        # leaving this for matteo: this is what reinventing the wheel looks like...
        # thankfully it was a small wheel. this time.
        """
        for line in sentences:
            encoded_line = []
            clean = utils.clean_text(line).split()
            for word in clean:
                index = self.tokenizer.word_index.get(word)
                if index is not None:
                    encoded_line.append(index)
                else:
                    encoded_line.append(0)
            to_predict.append(encoded_line)
        """
        to_predict = pad_sequences(self.tokenizer.texts_to_sequences(
            sentences), maxlen=self.sequence_length)
        preds = self.model.predict(to_predict)
        predictions = np.argmax(preds, axis=1)

        if labels is not None:
            # return predictions
            labels = [int(label) for label in labels]
            correct = [
                # look im kinda getting tired with python list comprehension so
                # if you can figure out how to do, element-wise,
                # correct[i] = (predictions[i] == labels[i]) good for you.
                # then, uncomment accuracy line
            ]
            accuracy = 0
            #accuracy = correct.count(True)/len(correct)
            return accuracy, predictions

        return predictions

    ######################################################################################################
    ########### UTILITY FUNCTIONS! NOT FUN BUT NECESSARY. other are included in utils.py #################
    def loadglove(self, glove_dir):
        embeddings_index = {}
        try:
            glove_file = open(glove_dir, encoding="utf8")
        except OSError:
            print(
                f"{glove_dir} does not lead to GloVe file. Please check and run again")
            raise OSError
        for line in glove_file:
            values = line.strip().split()
            # print(values[1:])
            token = values[0]
            # associate the Glove embedding vector to that token (word)
            embeddings_index[token] = np.asarray(values[1:], dtype=np.float64)
        glove_file.close()
        return embeddings_index

    def pretrained_embedding_matrix(self, embedding_size):

        # adding 1 to account for masking of index 0
        vocab_size = len(self.tokenizer.word_index) + 1

        embedding_matrix = np.zeros((vocab_size, embedding_size))
        # np.random.random((vocab_size + 1, embedding_size))  # not confident on this +1
        for word, i in self.tokenizer.word_index.items():
            if i % 10000 == 0:
                print('.', end='', flush=True)
            # create embedding: word index to Glove word embedding
            embedding_matrix[i, :] = self.embeddings_index.get(word)
            """embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector"""
        print()
        return embedding_matrix

    def split_data(self):
        # for int, like range but returns ndarray
        indices = np.arange(len(self.labels))
        np.random.shuffle(indices)
        x = self.data[indices]
        y = self.labels[indices]

        # keep validation data
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x, y, test_size=0.10, random_state=42)
        # 10% goes straight to validation. of the remaining, 70% train - 30% test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_train, self.y_train, test_size=0.30, random_state=42)
