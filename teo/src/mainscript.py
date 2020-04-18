import os
import pandas as pd
from cnn_blstm import BlstmCnnUtility
from tensorflow import keras as K
import numpy as np
import utils

DONT_TRAIN_JUST_LOAD = False

# both relative to the SCRIPT, to avoid workdir hassle
DATA_DIR = '../data/train.csv'
GLOVE_DIR = '../glove/glove.6B.100d.txt'
MODEL_DIR = '../models/teofakenews.h5'
CLEAN_DATA_DIR = '../clean_data/clean_train.csv'

SEQ_LEN = 1000
VOC_SIZE = 200000
EMBED_DIM = 100
FILTER_SIZES = [5, 3]
NUM_FILTERS = [32, 64]
USE_GPU = False  # requires cuda properly set up or running in a tensorflow-gpu
# enabled docker container
if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

CURRENT = os.path.dirname(__file__)

FULL_DATA_DIR = os.path.join(CURRENT, DATA_DIR)
FULL_GLOVE_DIR = os.path.join(CURRENT, GLOVE_DIR)

# create clean file usign utils.clean_and_save if not present
try:
    f = open(os.path.join(CURRENT, CLEAN_DATA_DIR))
except FileNotFoundError:
    data_train = pd.read_csv(FULL_DATA_DIR)

    # *merge* title and text. consider adding author/website
    x_train = [str(data_train.title[i]) + "" + str(data_train.text[i])
               for i in range(len(data_train.title))]
    y_train = data_train.label
    data_train = [x_train, y_train]
    utils.clean_and_save(data_train)


data_train = pd.read_csv(os.path.join(CURRENT, CLEAN_DATA_DIR))
x_train = [str(elem) for elem in data_train.data]  # necessary
y_train = data_train.labels

# initialize network
handler = BlstmCnnUtility([x_train, y_train], SEQ_LEN, GLOVE=True,
                          glove_dir=FULL_GLOVE_DIR)
if not DONT_TRAIN_JUST_LOAD:
    # build model with specified convolution layers
    handler.build_model(EMBED_DIM, FILTER_SIZES, NUM_FILTERS)
    handler.train(1)

    handler.save_model(os.path.join(CURRENT, MODEL_DIR))
else:
    handler.model = K.models.load_model(
        os.path.join(CURRENT, MODEL_DIR))
    pp = ['i like this sentence is good', 'why does this not work frick shoot']
    labels = [1, 0]
    handler.predict(pp, labels)
