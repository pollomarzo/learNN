"""
Base class for implementing similar networks

I'll use this as a boilerplate to simplify rest of the code
"""
from abc import ABC, abstractmethod
import os
import re
import numpy as np
import nltk
import pickle
from tensorflow import keras as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import embed_utils
from matplotlib import pyplot as plt

TRAINABLE_GLOVE = False
# number of words to keep (ordered by most frequent)
MAX_NB_WORDS = 20000
HISTORY_DIR = 'training_history/'


class ModelWrapper(ABC):
    """
    Abstract base class for networks

    Build method to be implemented on each subclass(hence network type)
    """

    def __init__(self, data, sequence_length, GLOVE=False, glove_dir=None, just_load=False):
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
        self.name = None
        # dictionary, NOT History class
        self.history = None
        self.data = data[0]
        # next line is necessary if clean_and_save was not run in main
        # self.data = [utils.clean_text(entry) for entry in texts]
        self.labels = np.asarray(data[1])
        # self.labels = to_categorical(np.asarray(data[1]), num_classes=2)
        if not just_load:
            self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            self.tokenizer.fit_on_texts(self.data)

            self.data = pad_sequences(
                self.tokenizer.texts_to_sequences(self.data), maxlen=sequence_length)
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
        self.history = self.model.fit(self.x_train, self.y_train, batch_size=batch_size,
                                      epochs=epochs, validation_data=(self.x_test, self.y_test)).history

    def save_model(self, model_dir='../models/', results_dir='../results/'):
        """
        Wraps save model to avoid exposing model and pickles history.

        yeah, it's public anyway, but this way i have to type less
        """
        if self.model != None:
            print(f"Saving model with name {self.name}.h5")
            self.model.save(os.path.join(model_dir, f"{self.name}.h5"))
            with open(os.path.join(results_dir, HISTORY_DIR, f"{self.name}HISTORY.pickled"), 'wb') as fp:
                pickle.dump(self.history, fp)
        else:
            print("Hmmm... you'll need to train something first!")

    def load_model(self, model_dir='../models/', results_dir='../results/'):
        """
        Wraps load model to avoid exposing and loading history if possible.
        """
        possiblemodels = [f for f in os.listdir(model_dir)
                          if os.path.isfile(os.path.join(model_dir, f))]
        if possiblemodels == []:
            print("No models found!")
            return None
        else:
            # i don't want the extension, so split on '.' at most once and take first piece
            self.name = possiblemodels[
                self.have_user_pick(possiblemodels)].split('.', 1)[0]
        print("Loading model...")
        with open(os.path.join(model_dir, f"{self.name}.h5")) as f:
            self.model = K.models.load_model(
                os.path.join(model_dir, f"{self.name}.h5"))
            self.load_history(results_dir)
            print("Model loaded!")

    def load_history(self, results_dir='../results/'):
        print("\tLoading history...")
        with open(os.path.join(results_dir, HISTORY_DIR, f"{self.name}HISTORY.pickled"), 'rb') as f:
            self.history = pickle.load(f)
            print("\tHistory loaded!")

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
            self.x_test, self.y_test, test_size=0.50)

    def draw_graph(self):
        """
        Uses pyplot to plot loss and accuracy on both training and testing set.

        """
        if self.history != None:
            # plot accuracy during training
            # judge overfit/early stop
            plt.plot(self.history['accuracy'])
            plt.plot(self.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
            # plot loss during training
            plt.plot(self.history['loss'])
            plt.plot(self.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
        else:
            print("You can't draw yet... maybe load or train a model first!")

    def have_user_pick(self, options):
        """
        Presents possible models found in dedicated directory
        """
        print("Please choose:")
        for idx, element in enumerate(options):
            print(f"{idx}\t {element}")
        i = input("Enter number: ")
        try:
            if 0 <= int(i) <= len(options):
                return int(i)
        except:
            print("Wrong input, try again.")
        return self.have_user_pick(options)
