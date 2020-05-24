import os
from collections import namedtuple
import pandas as pd
from cnn_blstm import CnnBlstmUtility
from embeddings import GloveEmbedding, Word2vecEmbedding
import utils

DONT_TRAIN_JUST_LOAD = False

CURRENT_DIR = os.path.dirname(__file__)
# all relative to the SCRIPT, to avoid workdir hassle
dirstruct = namedtuple(
    'dirstruct', 'DATA_DIR EMBED_FILE MODEL_DIR CLEAN_DATA_DIR CLEAN_DATA_FILE RESULTS_DIR TRAINING_HISTORY_DIR TEST_RESULT_DIR')
DIRECTORIES = dirstruct(
    DATA_DIR='../data/train.csv',
    EMBED_FILE='../word2vec/word2vec.100d.txt',
    MODEL_DIR='../models/',
    CLEAN_DATA_DIR='../clean_data/',
    CLEAN_DATA_FILE='../clean_data/clean_train.csv',
    RESULTS_DIR='../results/',
    TRAINING_HISTORY_DIR='../results/training_history/',
    TEST_RESULT_DIR='../results/test_results/'
)

SEQ_LEN = 1000
VOC_SIZE = 2000
EMBED_DIM = 100
FILTER_SIZES = [5, 3]
NUM_FILTERS = [32, 64]
USE_GPU = False  # requires cuda properly set up or running in a tensorflow-gpu
# enabled docker container
if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


FULL_DATA_DIR = os.path.join(CURRENT_DIR, DIRECTORIES.DATA_DIR)
FULL_EMBED_FILE = os.path.join(CURRENT_DIR, DIRECTORIES.EMBED_FILE)

# create clean file usign utils.clean_and_save if not present
utils.prepare_workspace(CURRENT_DIR, DIRECTORIES)
try:
    f = open(os.path.join(CURRENT_DIR, DIRECTORIES.CLEAN_DATA_FILE))
except FileNotFoundError:
    data_train = pd.read_csv(FULL_DATA_DIR)

    # *merge* title and text. consider adding author/website
    x_train = [str(data_train.title[i]) + "" + str(data_train.text[i])
               for i in range(len(data_train.title))]
    y_train = data_train.label
    data_train = [x_train, y_train]
    utils.clean_and_save(data_train, CURRENT_DIR, DIRECTORIES.CLEAN_DATA_FILE)


data_train = pd.read_csv(os.path.join(
    CURRENT_DIR, DIRECTORIES.CLEAN_DATA_FILE))
x_train = [str(elem) for elem in data_train.data]  # necessary
y_train = data_train.labels

# initialize network
cbhandler = CnnBlstmUtility([x_train, y_train], SEQ_LEN,
                            GloveEmbedding, FULL_EMBED_FILE, just_load=False)

if not DONT_TRAIN_JUST_LOAD:
    # build model with specified convolution layers
    cbhandler.build_model(EMBED_DIM, FILTER_SIZES, NUM_FILTERS)
    cbhandler.train(1)

    cbhandler.save_model(CURRENT_DIR, DIRECTORIES)
else:
    cbhandler.load_model(os.path.join(
        CURRENT_DIR, DIRECTORIES.MODEL_DIR), os.path.join(CURRENT_DIR, DIRECTORIES.RESULTS_DIR))
    cbhandler.draw_graph()
    # pp = ['i like this sentence is good', 'why does this not work frick shoot']
    # labels = [1, 0]
    # cbhandler.predict(pp, labels)
