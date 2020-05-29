import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe
from sklearn.model_selection import train_test_split
import utils
import pandas as pd
import numpy as np

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar


from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from TextCNN import TextCNN
from FakeNewsDataset import FakeNewsDataset
from ex_QuantLeNet import QuantLeNet
########################################################################################################################################
"""
Let's walk through what the function of the trainer does:

    Sets model in train mode.
    Sets the gradients of the optimizer to zero.
    Generate x and y from batch.
    Performs a forward pass to calculate y_pred using model and x.
    Calculates loss using y_pred and y.
    Performs a backward pass using loss to calculate gradients for the model parameters.
    model parameters are optimized using gradients and optimizer.
    Returns scalar loss.
"""


def process_function(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch.text, batch.label
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


"""
With torch.no_grad(), no gradients are calculated for any succeding steps.
Ignite suggests attaching metrics to evaluators and not trainers because during the training the model parameters are constantly changing and it is best to evaluate model on a stationary model. This information is important as there is a difference in the functions for training and evaluating. Training returns a single scalar loss. Evaluating returns y_pred and y as that output is used to calculate metrics per batch for the entire dataset.

All metrics in Ignite require y_pred and y as outputs of the function attached to the Engine.
"""


def eval_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch.text, batch.label
        y_pred = model(x)
        return y_pred, y
########################################################################################################################################


CLEAN_DATA_FILE = '../clean_data/small.csv'

device = torch.device('cpu')
"""
STUFF = None
num_filters = [5, 3]
filter_sizes = [32, 64]
data_train = pd.read_csv(CLEAN_DATA_FILE)
data = [str(elem) for elem in data_train.data]
labels = data_train.labels
x_train, x_test, x_val, y_train, y_test, y_val = utils.split_data(data, labels)
"""


SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

TEXT = data.Field(lower=True, batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

text_data = FakeNewsDataset(CLEAN_DATA_FILE, TEXT, LABEL)


train_data, test_data = text_data.split()
# train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='../tmp/imdb/')
train_data, valid_data = train_data.split(
    split_ratio=0.8, random_state=random.seed(SEED))

TEXT.build_vocab(train_data, vectors=GloVe(
    name='6B', dim=100, cache='../tmp/glove/'))

LABEL.build_vocab(train_data)
# BucketIterator pads every element of a batch to the length of the longest element of the batch.
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                           batch_size=32,
                                                                           device=device)
vocab_size, embedding_dim = TEXT.vocab.vectors.shape

model = QuantLeNet(vocab_size=vocab_size,
                   embedding_dim=embedding_dim,
                   kernel_sizes=[3, 4, 5],
                   num_filters=100,
                   num_classes=1,
                   d_prob=0.5,
                   mode='static',
                   TEXT=TEXT)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
criterion = nn.BCELoss()

trainer = Engine(process_function)
train_evaluator = Engine(eval_function)
validation_evaluator = Engine(eval_function)

# To attach a metric to engine, the following format is used:
#   Metric(output_transform=output_transform, ...).attach(engine, 'metric_name')
#
RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

# For Accuracy, Ignite requires y_pred and y to be comprised of 0's and 1's only.
# Since our model outputs from a sigmoid layer, values are between 0 and 1.
# We'll need to write a function that transforms engine.state.output which is
# comprised of y_pred and y.


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


Accuracy(output_transform=thresholded_output_transform).attach(
    train_evaluator, 'accuracy')
Loss(criterion).attach(train_evaluator, 'bce')

Accuracy(output_transform=thresholded_output_transform).attach(
    validation_evaluator, 'accuracy')
Loss(criterion).attach(validation_evaluator, 'bce')

pbar = ProgressBar(persist=True, bar_format="")
pbar.attach(trainer, ['loss'])


def score_function(engine):
    val_loss = engine.state.metrics['bce']
    return -val_loss


handler = EarlyStopping(
    patience=5, score_function=score_function, trainer=trainer)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)

# to attach custom functions at certain events, two possible syntaxes
# exist. i chose the decorator one because it's cooler.


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_iterator)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_bce))


# Lastly, we want to checkpoint this model. It's important to do so,
# as training processes can be time consuming and if for some reason
# something goes wrong during training, a model checkpoint can be helpful
# to restart training from the point of failure.
# Below we'll use Ignite's ModelCheckpoint handler to checkpoint
# models at the end of each epoch.
checkpointer = ModelCheckpoint(
    '../tmp/models', 'textcnn', n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED,
                          checkpointer, {'textcnn': model})
trainer.run(train_iterator, max_epochs=20)
