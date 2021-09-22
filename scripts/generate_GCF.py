# generate function used in google cloud functions
import argparse
import numpy as np
import re
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def validate_seed(vocabulary, seed):
    """Validate that all the words in the seed are part of the vocabulary"""
    seed_words = seed.split(" ")
    valid = True
    problem_words = []
    for w in seed_words:
        if w not in vocabulary:
            problem_words.append(w)
            valid = False
    return valid, problem_words


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, indices_word, word_indices, seed,
                  sequence_length, diversity, quantity):
    """
    Similar to lstm_train::on_epoch_end
    Used to generate text using a trained model

    :param model: the trained Keras model (with model.load)
    :param indices_word: a dictionary pointing to the words
    :param seed: a string to be used as seed (already validated and padded)
    :param sequence_length: how many words are given to the model to generate
    :param diversity: is the "temperature" of the sample function (usually between 0.1 and 2)
    :param quantity: quantity of words to generate
    :return: Nothing, for now only writes the text to console
    """
    sentence = seed.split(" ")
    text = ""

    for i in range(quantity):
        x_pred = np.zeros((1, sequence_length))
        for t, word in enumerate(sentence):
            x_pred[0, t] = word_indices[word]

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]

        sentence = sentence[1:]
        sentence.append(next_word)

        text = text + " " + next_word
    return text


def main(request=None):

    try:
        seed = request.get_json()['seed']
    except:
        seed = "orice s-ar intampla doar tu esti viata mea"
    vocabulary_file = "vocabulary.txt"
    model_file = "model"
    sequence_length = 10
    diversity = 0.5
    quantity = 60

    model = load_model(model_file)
    vocabulary = open(vocabulary_file, "r").readlines()

    vocabulary = [re.sub(r'(\S+)\s+', r'\1', w) for w in vocabulary]
    vocabulary = sorted(set(vocabulary))

    word_indices = dict((c, i) for i, c in enumerate(vocabulary))
    indices_word = dict((i, c) for i, c in enumerate(vocabulary))

    valid, problem_words = validate_seed(vocabulary, seed)
    if valid:
        text = seed
        # repeat the seed in case is not long enough, and take only the last elements
        seed = " ".join((((seed+" ")*sequence_length)+seed).split(" ")[-sequence_length:])
        text += generate_text(
            model, indices_word, word_indices, seed, sequence_length, diversity, quantity
        )
    else:
        text = ""
        text += 'ERROR: Please fix the seed string\n'
        for w in problem_words:
            text += w
            text += " ✗ NOT in vocabulary\n"
    
    return text

print(main())