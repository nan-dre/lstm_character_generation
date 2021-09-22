from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding, GRU
import numpy as np
import sys
import io
import os
import re
import json
import tensorflow as tf

# Parameters: change to experiment different configurations
SEQUENCE_LEN = 10
MIN_WORD_FREQUENCY = 50
STEP = 1
BATCH_SIZE = 32


def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


# Data generator for fit and evaluate
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices[w]
            y[i] = word_indices[next_word_list[index % len(sentence_list)]]
            index = index + 1
        yield x, y


def get_model(dropout=0.2):
    print('Build model: ')
    model = Sequential()
    model.add(Embedding(input_dim=len(words), output_dim=1024))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model



# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(sentences+sentences_test))
    seed = (sentences+sentences_test)[seed_index]

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sentence = seed
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))

        for i in range(50):
            x_pred = np.zeros((1, SEQUENCE_LEN))
            for t, word in enumerate(sentence):
                x_pred[0, t] = word_indices[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" "+next_word)
        examples_file.write('\n')
    examples_file.write('='*80 + '\n')
    examples_file.flush()


if __name__ == "__main__":
    # Argument check
    examples = sys.argv[1]

    if not os.path.isdir('./checkpoints/'):
        os.makedirs('./checkpoints/')
    # Loading the training data
    path = '../data/manele-merged.json'
    text = []
    with io.open(path, encoding="utf-8") as f:
        file = json.load(f)
        for manea in file:
            for lyric in manea['lyrics']:
                text.append(lyric)
    text = ''.join(text)

    # Cleaning the text
    text = text.lower()
    to_replace = list('!"$&()*+/:;<=>@[]^_~{}#%\\|–…\ufeff\xa0§«»')
    to_replace.append("'")
    to_replace.append("refren")
    to_replace.append("ref")
    to_replace.append("florin salam")
    to_replace.append("salam")
    to_replace.append("bis")
    to_replace.append("augustin")
    to_replace.append("nicolae guta")
    to_replace.append("nicoleta guta")
    to_replace.append("guta")
    to_replace.append("costel biju")
    to_replace.append("liviu pustiu")
    to_replace.append("dani mocanu")
    to_replace.append("vali vijelie")
    to_replace.append("solo")
    to_replace.append("x2")
    to_replace.append("2x")

    for word in to_replace:
        text = text.replace(word, '')

    text = re.sub('â|ă|а', 'a', text)
    text = re.sub('í|î|ï|і|ἰ', 'i', text)
    text = re.sub('ş|ș|ѕ', 's', text)
    text = re.sub('ţ', 't', text)
    text = re.sub('ν', 'v', text)
    text = re.sub('в', 'b', text)
    text = re.sub('е', 'e', text)
    text = re.sub('к', 'k', text)
    text = re.sub('м', 'm', text)
    text = re.sub('н', 'h', text)
    text = re.sub('о', 'o', text)
    text = re.sub('р', 'p', text)
    text = re.sub('с', 'c', text)
    text = re.sub('т', 't', text)
    text = re.sub('у', 'y', text)
    text = re.sub('х', 'x', text)
    text = re.sub('ј', 'j', text)
    text = re.sub('k', 'ca', text)


    text = re.sub(r'\d\.', '', text)
    text = re.sub(r'st?rofa \d*', '', text)
    text = re.sub(r'-{2,}', '', text)
    text = re.sub(r'sh', 's', text)
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.replace('\n', ' \n ')
    print("Corpus length:", len(text))

    chars = sorted(list(set(text)))
    print(chars)
    print("Total chars:", len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))


    print('Corpus length in characters:', len(text))

    text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
    print('Corpus length in words:', len(text_in_words))

    # Calculate word frequency
    word_freq = {}
    for word in text_in_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)

    words = set(text_in_words)
    print('Unique words before ignoring:', len(words))
    print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
    words = sorted(set(words) - ignored_words)
    print('Unique words after ignoring:', len(words))

    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))

    # cut the text in semi-redundant sequences of SEQUENCE_LEN words
    sentences = []
    next_words = []
    ignored = 0
    for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
        # Only add the sequences where no word is in ignored_words
        if len(set(text_in_words[i: i+SEQUENCE_LEN+1]).intersection(ignored_words)) == 0:
            sentences.append(text_in_words[i: i + SEQUENCE_LEN])
            next_words.append(text_in_words[i + SEQUENCE_LEN])
        else:
            ignored = ignored + 1
    print('Ignored sequences:', ignored)
    print('Remaining sequences:', len(sentences))

    # x, y, x_test, y_test
    (sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(
        sentences, next_words
    )

    model = get_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    file_path = "../checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
                    "loss{loss:.4f}-val_loss{val_loss:.4f}" % \
                    (len(words), SEQUENCE_LEN, MIN_WORD_FREQUENCY)

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    callbacks_list = [print_callback]

    examples_file = open(examples, "w")
    model.fit(generator(sentences, next_words, BATCH_SIZE),
                        steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                        epochs=15,
                        callbacks=callbacks_list,
                        validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                        validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)
    model.save("saved_model.h5", save_format="h5")