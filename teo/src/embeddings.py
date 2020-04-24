"""
Contains embedding-related functions and classes
"""
import os
import numpy as np
from tensorflow import keras as K
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
#LabeledSentence = gensim.models.doc2vec.LabeledSentence

TRAINABLE_EMBEDS = False
EMBED_DIM = 100

# f"../word2vec/word2vec.{EMBED_DIM}d.txt"


class Word2vecEmbedding:
    """
    Implements a Word2Vec embedding. If a file is given, loads from file. 
    Otherwise, trains Word2Vec embedding and saves it to a file.

    Call build_embedding to train and obtain embedding layer
    """

    def __init__(self, sequence_length, tokenizer,
                 save_file, data):
        self.name = 'Word2Vec'
        self.embedding_layer = None
        self.sequence_length = sequence_length
        self.word_index = tokenizer.word_index
        self.data = tokenizer.sequences_to_texts(data)
        self.save_file = save_file
        self.save_file_present = os.path.isfile(save_file)
        if not self.save_file_present:
            #self.model = Word2Vec(size=EMBED_DIM, min_count=10)
            #self.model.build_vocab([text for text in tqdm(self.data)])
            self.model = Word2Vec(
                sentences=self.data, size=EMBED_DIM, window=5, workers=4, min_count=5)
            print("Built model", flush=True)
            #[x.tolist() for x in self.data]

    def build_embedding(self):
        if not self.save_file_present:
            print("\tBuilding Word2Vec embedding. Training an entire model!", flush=True)
            # comprehension just to show progress bar
            # self.model.train(["".join(str(sentence)) for sentence in tqdm(
            #    self.data)], total_examples=self.model.corpus_count)
            self.model.train(self.data, epochs=3,
                             total_examples=self.model.corpus_count)
            print("\t...done")
            print("\tSaving to file...", flush=True)
            self.model.wv.save_word2vec_format(self.save_file, binary=False)
            print("\t...done.")
        embeddings_index = loadembed(self.save_file)
        embedding_weights = pretrained_embedding_matrix(
            EMBED_DIM, self.word_index, embeddings_index)
        print("\tcreating embedding layer...")

        self.embedding_layer = K.layers.Embedding(
            len(self.word_index) + 1, EMBED_DIM,
            input_length=self.sequence_length, weights=[
                embedding_weights],
            trainable=TRAINABLE_EMBEDS)

        print("\tembedding layer set!")
        return self.embedding_layer


class GloveEmbedding:
    """
    Implements a GloVe embedding, from a GloVe file and a word_index (for use with tokenizer).

    Call build_embedding to obtain the embedding layer
    """

    def __init__(self, sequence_length, tokenizer, save_file, data):
        print("\tloading glove...")
        self.name = 'GloVe'
        self.embedding_layer = None
        self.sequence_length = sequence_length
        self.word_index = tokenizer.word_index
        # embeddings_index is a dict, word to glove embedding vector
        self.embeddings_index = loadembed(save_file)
        print("\t..done!")

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
            trainable=TRAINABLE_EMBEDS)

        print("\tembedding layer set!")
        return self.embedding_layer


def loadembed(save_file):
    """
    Loads the pretrained embed file, setting embeddings in embeddings_index, returned to caller.
    """
    embeddings_index = {}
    try:
        save_file = open(save_file, encoding="utf8")
    except OSError:
        print(
            f"{save_file} does not lead to GloVe file. Please check and run again")
        raise OSError
    for line in save_file:
        values = line.split()
        token = values[0]
        # associate the Glove embedding vector to that token (word)
        embeddings_index[token] = np.asarray(values[1:], dtype=np.float32)
    save_file.close()
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
