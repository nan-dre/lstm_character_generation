# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import random
import io
import json
import re

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# %%
# path = keras.utils.get_file(
#     "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
# )
path = './manele.json'


# %%
text = []
with io.open(path, encoding="utf-8") as f:
    file = json.load(f)
    for manea in file:
        for lyric in manea['lyrics']:
            text.append(lyric)
text = ''.join(text)


# %%
text = text.lower()
to_replace = list('!"$&()*+/:;<=>@[]^_~{}')
to_replace.append("refren")
to_replace.append("ref")
to_replace.append("x2")
to_replace.append("2x")
to_replace.append("florin salam")
to_replace.append("salam")
to_replace.append("bis")

for word in to_replace:
    text = text.replace(word, '')
    pass
text = re.sub(r'\n\s*\n', '\n\n', text)
text = re.sub(r'\.{4,}', '...', text)
# print(text)
# print("Corpus length:", len(text))

chars = sorted(list(set(text)))
# print(chars)
# print("Total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 100
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
# print("Number of sequences:", len(sentences))
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# %%
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# %%
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# %%
# load the network weights
filename = "weights-10-0.9841.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# %%
generated_length = 300
start_index = random.randint(0, len(text) - maxlen - 1)
for diversity in [0.2]:
    # print("...Diversity:", diversity)

    generated = ""
    original_sentence = text[start_index : start_index + maxlen]
    sentence = original_sentence
    # print('...Generating with seed:\n "' + sentence + '"')

    for i in range(generated_length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        sentence = sentence[1:] + next_char
        generated += next_char

    print(original_sentence + generated)


