"""fully taken from ignite example"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
"""

Here is the replication of the model, here are the operations of the model:

    Embedding: Embeds a batch of text of shape (N, L) to (N, L, D), where N is batch size, L is maximum length of the batch, D is the embedding dimension.

    Convolutions: Runs parallel convolutions across the embedded words with kernel sizes of 3, 4, 5 to mimic trigrams, four-grams, five-grams. This results in outputs of (N, L - k + 1, D) per convolution, where k is the kernel_size.

    Activation: ReLu activation is applied to each convolution operation.

    Pooling: Runs parallel maxpooling operations on the activated convolutions with window sizes of L - k + 1, resulting in 1 value per channel i.e. a shape of (N, 1, D) per pooling.

    Concat: The pooling outputs are concatenated and squeezed to result in a shape of (N, 3D). This is a single embedding for a sentence.

    Dropout: Dropout is applied to the embedded sentence.

    Fully Connected: The dropout output is passed through a fully connected layer of shape (3D, 1) to give a single output for each example in the batch. sigmoid is applied to the output of this layer.

    load_embeddings: This is a method defined for TextCNN to load embeddings based on user input. There are 3 modes - rand which results in randomly initialized weights, static which results in frozen pretrained weights, nonstatic which results in trainable pretrained weights.

Let's note that this model works for variable text lengths! The idea to embed the words of a sentence, use convolutions, maxpooling and concantenation to embed the sentence as a single vector! This single vector is passed through a fully connected layer with sigmoid to output a single value. This value can be interpreted as the probability a sentence is positive (closer to 1) or negative (closer to 0).

The minimum length of text expected by the model is the size of the smallest kernel size of the model.
"""


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters, num_classes, d_prob, mode, TEXT):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.d_prob = d_prob
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.load_embeddings(TEXT)
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                             out_channels=num_filters,
                                             kernel_size=k, stride=1) for k in kernel_sizes])
        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = self.embedding(x).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.fc(self.dropout(x))
        return torch.sigmoid(x).squeeze()

    def load_embeddings(self, TEXT):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(TEXT.vocab.vectors)
            if 'non' not in self.mode:
                self.embedding.weight.data.requires_grad = False
                print('Loaded pretrained embeddings, weights are not trainable.')
            else:
                self.embedding.weight.data.requires_grad = True
                print('Loaded pretrained embeddings, weights are trainable.')
        elif self.mode == 'rand':
            print('Randomly initialized embeddings are used.')
        else:
            raise ValueError(
                'Unexpected value of mode. Please choose from static, nonstatic, rand.')
