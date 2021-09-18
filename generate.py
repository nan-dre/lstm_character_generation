import random
import io
import json
import re
import numpy as np
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import config

import tensorflow as tf

def generate():
    gpus = config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        config.experimental.set_memory_growth(gpu, True)

    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    #path = './manele.json'
    path = tf.keras.utils.get_file(
        "manele.json", origin="https://raw.githubusercontent.com/nan-dre/lstm_character_generation/main/manele.json"
    )
    model_path = tf.keras.utils.get_file( 
        "model-latest.h5", origin="https://raw.githubusercontent.com/nan-dre/lstm_character_generation/main/model-latest.h5"
    )
    model = load_model(model_path)


    text = []
    with io.open(path, encoding="utf-8") as f:
        file = json.load(f)
        for manea in file:
            for lyric in manea['lyrics']:
                text.append(lyric)
    text = ''.join(text)

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

    generated_length = 300
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.5]:
        generated = ""
        original_sentence = text[start_index : start_index + maxlen]
        sentence = original_sentence

        for i in range(generated_length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char

        return original_sentence + generated

if __name__ == '__main__':
    text = generate()
    print(text)