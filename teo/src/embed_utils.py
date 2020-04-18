import numpy as np


def loadglove(glove_dir):
    embeddings_index = {}
    try:
        glove_file = open(glove_dir, encoding="utf8")
    except OSError:
        print(
            f"{glove_dir} does not lead to GloVe file. Please check and run again")
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
