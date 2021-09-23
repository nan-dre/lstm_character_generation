# generate function used in google cloud functions
import argparse
import numpy as np
import re
from keras.models import load_model
import os

def download_model_file():
    from google.cloud import storage
    BUCKET_NAME        = "lstm_word_model"
    PROJECT_ID         = "hallowed-pager-326510"
    GCS_MODEL_FILE     = "saved_model.h5"
    GCS_VOCABULARY_FILE= "vocabulary.txt"

    # Initialise a client
    client   = storage.Client(PROJECT_ID)
    
    # Create a bucket object for our bucket
    bucket   = client.get_bucket(BUCKET_NAME)
    
    # Create a blob object from the filepath
    model_blob     = bucket.blob(GCS_MODEL_FILE)
    vocabulary_blob     = bucket.blob(GCS_VOCABULARY_FILE)
    
    folder = '/tmp/'
    if not os.path.exists(folder):
      os.makedirs(folder)
    # Download the file to a destination
    model_blob.download_to_filename(folder + "saved_model.h5")
    vocabulary_blob.download_to_filename(folder + "vocabulary.txt")

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
    sentence = seed.split(" ")
    text = ""

    # Generating minimum lenght
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
    
    # Generating until reaching newline
    while next_word != '\n':
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
        json = request.get_json()
        seed = str(json['seed'])
        sequence_length = int(json['sequence_length'])
        diversity = float(json['diversity'])
        quantity = int(json['quantity'])
    except Exception as e: 
        print(e)
        seed = 'iubire'
        sequence_length = 8
        diversity = 0.5
        quantity = 100
    vocabulary_file = "/tmp/vocabulary.txt"
    model_file = "/tmp/saved_model.h5"

    download_model_file()
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