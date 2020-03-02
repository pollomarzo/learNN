import tensorflow as tf  # noqa
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import numpy as np
import matplotlib.pyplot as plt


# class to manage encoding/decoding/prediction from probability vector
class CharacterTable(object):
    def __init__(self, chars):
        # initialize table, given which characters can appear
        self.chars = sorted(set(chars))
        # "bidirectional" maps
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        # one-hot encoding (in matrix (2D array from now on))
        # num_rows to maintain dimensions
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """ can use same func for decode and finding predictions
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot represen-
                tations; or a vector of character indices (used with
                `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        # now x is vector of indices
        return ''.join(self.indices_char[x] for x in x)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# model-dataset parameters:
TRAIN_SIZE = 50000
DIGITS = 3
REVERSE = True
# maxlen is "int+int"
MAXLEN = DIGITS + 1 + DIGITS

# numbers, plus and space (padding)
chars = '0123456789+ '
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()
print('Generating data...')

while len(questions) < TRAIN_SIZE:
    def f(): return int(''.join(np.random.choice(list('0123456789'))
                                for i in range(np.random.randint(1, DIGITS + 1))))  # noqa
    a, b = f(), f()
    # we need to skip repeated and commuted. so, we sort
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # pad until MAXLEN
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # answers of size DIGITS + 1 (500+500)
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('total questions:', len(questions))

print('Vectorizing...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS+1)

# shuffle to spread larger digits from bottom
indices = np.arange(len(y))  # for int, like range but returns ndarray
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# keep validation data
split_at = len(x) - len(x) // 10  # // "floor" division
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

print('Building model..')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# As the decoder RNN's input, repeatedly provide with the last output of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(DIGITS+1))

for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    # noot sure i understand this
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
model.add(layers.TimeDistributed(
    layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                    epochs=30, validation_data=(x_val, y_val))

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

"""
for iteration in range(1, 20):
    print()
    print('Iteration:', iteration)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE,
              epochs=5, validation_data=(x_val, y_val))

    # Select 10 samples from the validation set at random so we can visualize
    # errors. (straight up copied)
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)
"""
