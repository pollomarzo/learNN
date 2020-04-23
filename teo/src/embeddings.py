"""
Contains embedding-related functions and classes
"""

import numpy as np
from tensorflow import keras as K

TRAINABLE_GLOVE = False
EMBED_DIM = 100


class GloveEmbedding:
    """
    Implements a GloVe embedding, from a GloVe file and a word_index (for use with tokenizer).

    Call build_embedding to obtain the embedding layer
    """

    def __init__(self, glove_file, sequence_length, word_index):
        print("loading glove...")
        self.name = 'GloVe'
        self.embedding_layer = None
        self.sequence_length = sequence_length
        self.word_index = word_index
        # embeddings_index is a dict, word to glove embedding vector
        self.embeddings_index = self.loadglove(glove_file)
        print("..done!")

    def build_embedding(self):
        """
        Constructs the embedding layer, from tokenizer index to GloVe embedding.
        """
        print("\tcreating embedding matrix with glove values...")
        print('\t', end='')

        embedding_weights = pretrained_embedding_matrix(
            EMBED_DIM, self.word_index, self.embeddings_index)
        print("\tembedding matrix created!")
        print("\tcreating embedding layer...")

        self.embedding_layer = K.layers.Embedding(
            len(self.word_index) + 1, EMBED_DIM,
            input_length=self.sequence_length, weights=[
                embedding_weights],
            trainable=TRAINABLE_GLOVE)

        print("\tembedding layer set!")
        return self.embedding_layer

    def loadglove(self, glove_file):
        """
        Loads the GloVe file, setting embeddings in embeddings_index, returned to caller.
        """
        embeddings_index = {}
        try:
            glove_file = open(glove_file, encoding="utf8")
        except OSError:
            print(
                f"{glove_file} does not lead to GloVe file. Please check and run again")
            raise OSError
        for line in glove_file:
            values = line.split()
            token = values[0]
            # associate the Glove embedding vector to that token (word)
            embeddings_index[token] = np.asarray(values[1:], dtype=np.float32)
        glove_file.close()
        return embeddings_index


def pretrained_embedding_matrix(embedding_size, word_index, embeddings_index):
    # adding 1 to account for masking of index 0
    vocab_size = len(word_index) + 1

    embedding_matrix = np.zeros((vocab_size, embedding_size))
    # np.random.random((vocab_size + 1, embedding_size))  # not confident on this +1
    for word, i in word_index.items():
        if i % 10000 == 0:
            print('.', end='', flush=True)
        # create embedding: word index to Glove word embedding
        embedding_vect = embeddings_index.get(word)
        if embedding_vect is not None:
            embedding_matrix[i] = embeddings_index.get(word)

    print()
    return embedding_matrix
