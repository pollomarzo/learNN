import os
import numpy as np
from sklearn.model_selection import train_test_split
from ex_QuantLeNet import QuantLeNet

# number of words to keep (ordered by most frequent)
MAX_NB_WORDS = 20000
HISTORY_DIR = 'training_history/'


class ModelWrapper():

    def __init__(self, data, embedding_size,
                 filter_sizes, num_filters):
        """
        Constructor, saves data as class attribute
        Expects training data to be the CLEAN and first element, and labels as the second
        """
        self.build_model(embedding_size,
                         filter_sizes, num_filters)

    def buid_embedding(self):
        """
        wraps the embedding layer function for cleanliness
        """
        pass

    def build_model(self, embedding_size,
                    filter_sizes, num_filters):
        self.model = QuantLeNet()

    def train(self, epochs=1, batch_size=64):
        ...


"""
    def save_model(self, current_dir, DIRECTORIES):

        results_dir = DIRECTORIES.RESULTS_DIR
        model_dir = os.path.join(current_dir, DIRECTORIES.MODEL_DIR)
        history_dir = os.path.join(
            current_dir, DIRECTORIES.TRAINING_HISTORY_DIR)
        test_result_dir = os.path.join(
            current_dir, DIRECTORIES.TEST_RESULT_DIR)
        if self.model != None:
            print(f"Saving model with name {self.name}.h5")
            self.model.save(os.path.join(model_dir, f"{self.name}.h5"))
            with open(os.path.join(
                    results_dir, history_dir, f"{self.name}HISTORY.pickled"), 'wb') as fp:
                pickle.dump(self.history, fp)
            with open(os.path.join(
                    results_dir, test_result_dir, f"{self.name}test_results.txt"), 'w') as f:
                print('This model\'s training ended with train, test accuracy = ',
                      self.history['accuracy'][-1],
                      self.history['val_accuracy'][-1], file=f)
        else:
            print("Hmmm... you'll need to train something first!") """
""" 
    def load_model(self, model_dir='../models/', results_dir='../results/'):
        possiblemodels = [f for f in os.listdir(model_dir)
                          if os.path.isfile(os.path.join(model_dir, f))]
        if possiblemodels == []:
            print("No models found!")
        else:
            # i don't want the extension, so split on '.' at most once and take first piece
            self.name = possiblemodels[
                self.have_user_pick(possiblemodels)].split('.', 1)[0]
        print("Loading model...")
        with open(os.path.join(model_dir, f"{self.name}.h5")) as f:
            self.model = K.models.load_model(
                os.path.join(model_dir, f"{self.name}.h5"))
            self.load_history(results_dir)
            print("Model loaded!") """
""" 
    def load_history(self, results_dir='../results/'):
        print("\tLoading history...")
        with open(os.path.join(results_dir, HISTORY_DIR, f"{self.name}HISTORY.pickled"), 'rb') as f:
            self.history = pickle.load(f)
            print("\tHistory loaded!")
 """

""" 
    def get_validation_score(self):
        _, self.validation_accuracy = self.model.evaluate(
            self.x_val, self.y_val)
"""
""" 
    def predict(self, sentences, labels=None):
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
 """
""" 
       def split_data(self):
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
"""
""" 
    def draw_graph(self):
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
"""
""" 
    def have_user_pick(self, options):
        print("Please choose:")
        for idx, element in enumerate(options):
            print(f"{idx}\t {element}")
        i = input("Enter number: ")
        try:
            if 0 <= int(i) <= len(options):
                return int(i)
        except:
            print("Wrong input, try again.")
        return self.have_user_pick(options) """
