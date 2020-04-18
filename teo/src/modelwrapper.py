"""
Base class for implementing similar networks

I'll use this as a boilerplate to simplify rest of the code
"""
from abc import ABC, abstractmethod
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import nltk
import embed_utils

TRAINABLE_GLOVE = False
# number of words to keep (ordered by most frequent)
MAX_NB_WORDS = 20000


class ModelWrapper(ABC):
    """
    Abstract base class for networks

    Build method to be implemented on each subclass (hence network type)
    """

    def __init__(self, data, sequence_length, GLOVE=False, glove_dir=None):
        """
        Constructor, saves data as class attribute

        Expects training data to be the first element, and labels as the second
        """
        nltk.download('stopwords')

        print("initializing variables and tokenizing...")
        self.validation_accuracy = 0
        self.sequence_length = sequence_length
        self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = (
            None, None, None, None, None, None)
        self.model = None
        self.data = data[0]
        # next line is necessary if clean_and_save was not run in main
        # self.data = [utils.clean_text(entry) for entry in texts]
        self.labels = np.asarray(data[1])
        # self.labels = to_categorical(np.asarray(data[1]), num_classes=2)
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
            self.embeddings_index = embed_utils.loadglove(glove_dir)
            print("..done!")

    # BUILD MODEL, COMPILE
    @abstractmethod
    def build_model(self, embedding_size,
                    filter_sizes, num_filters, num_cells=100):
        """
        suclasses have to implement to specify arch
        """
        pass

    def train(self, epochs=1, batch_size=64):
        """
        Wraps fit method to avoid data input

        as the data is already saved in constructor, no need to input again. 
        uses test for validation
        """
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size,
                       epochs=epochs, validation_data=(self.x_test, self.y_test))

    def save_model(self, save_dir='./models'):
        """
        Wraps save model to avoid exposing model.

        yeah, it's public anyway, but this way i have to type less
        """
        self.model.save(save_dir)

    # RECOMPUTE VALIDATION
    # (validation data is NEVER used, so can be used to compare different models)
    def get_validation_score(self):
        """
        Recomputes validation every time it is called, using validation data.

        validation data is NEVER used in model
        """

        _, self.validation_accuracy = self.model.evaluate(
            self.x_val, self.y_val)

    # PREDICT ON NEW DATA
    # honestly, it's still not done but im tired
    def predict(self, sentences, labels=None):
        """
        Predicts for new data given in input.

        still have to clean, tokenize, and fix
        """
        to_predict = []

        to_predict = pad_sequences(self.tokenizer.texts_to_sequences(
            sentences), maxlen=self.sequence_length)
        preds = self.model.predict(to_predict)
        predictions = np.argmax(preds, axis=1)

        if labels is not None:
            # return predictions
            labels = [int(label) for label in labels]
            correct = [predictions[i] == labels[i] for i in range(len(predictions))
                       # look im kinda getting tired with python list comprehension so
                       # if you can figure out how to do, element-wise,
                       # correct[i] = (predictions[i] == labels[i]) good for you.
                       # then, uncomment accuracy line
                       ]
            accuracy = 0
            accuracy = correct.count(True)/len(correct)
            return accuracy, predictions

        return predictions

    ################################################################################################
    ########### UTILITY FUNCTIONS! others are included in utils.py #################

    def split_data(self):
        """
        Splits into train-test-validate, 80-10-10.
        """
        # for int, like range but returns ndarray
        indices = np.arange(len(self.labels))
        np.random.shuffle(indices)
        x = self.data[indices]
        y = self.labels[indices]

        # keep validation data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.20)
        # 10% goes straight to validation.
        self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(
            self.x_test, self.y_val, test_size=0.50, random_state=42)
