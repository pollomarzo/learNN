import os
import pandas as pd
from blstm_cnn import BlstmCnnUtility
from tensorflow import keras as K


# both relative to the SCRIPT, to avoid workdir hassle
DATA_DIR = '../data/train.csv'
GLOVE_DIR = '../glove/glove.6B.100d.txt'
MODEL_DIR = '../models/teofakenews.h5'

SEQ_LEN = 20
VOC_SIZE = 200000
EMBED_DIM = 100
FILTER_SIZES = [5, 3]
NUM_FILTERS = [32, 64]

full_data_dir = os.path.join(os.path.dirname(__file__), DATA_DIR)
full_glove_dir = os.path.join(os.path.dirname(__file__), GLOVE_DIR)

data_train = pd.read_csv(full_data_dir)

# *merge* title and text. consider adding author/website
# possible need for cleanup, included function in utils.py
x_train = [str(data_train.title[i]) + str(data_train.text[i])
           for i in range(len(data_train.title))]
y_train = data_train.label

# initialize network
handler = BlstmCnnUtility(x_train, y_train, SEQ_LEN, GLOVE=True,
                          glove_dir=full_glove_dir)
if False:
    # build model with specified convolution layers
    handler.build_model(EMBED_DIM, FILTER_SIZES, NUM_FILTERS)
    handler.train()

    handler.save_model(os.path.join(os.path.dirname(__file__), MODEL_DIR))
else:
    handler.model = K.models.load_model(
        os.path.join(os.path.dirname(__file__), MODEL_DIR))
    pp = ['i like this sentence is good', 'why does this not work frick shoot']
    labels = [True, False]
    handler.predict(pp, labels)
