"""
The file contains some useful functions :
- data preprocessing
- creating training samples
"""


import numpy as np
from nltk.corpus import stopwords
import string
from collections import Counter
import random


# Convert document into a list of word

def convert_text(text):
    """
    Create list of words from a text document
    :param text: raw text file
    :return: text_list (list)
    """

    text = text.lower()
    text = text.split()
    text_list = [word.strip(string.punctuation) for word in text.split()]

    stop_words = set(stopwords.words('english'))

    for index, word in enumerate(text_list):
        if word in stop_words:
            del text_list[index]

    return text_list


# Pre-processing

def building_dataset(text, n_words):
    """
    Function to build the data set
    :param text: list of words
    :param n_words: number of desired words for vocabulary
    :return: data (text where each word is replaced by its index), occurrence, dictionary
    """

    occurrence = Counter(text).most_common(n_words - 1)  # list of (word, occurrence)
    occurrence.extent(['ukn', 0])                        # add couple (unknown, 0)

    dictionary = dict()                                  # dictionary of couples (word, index)
    dictionary['ukn'] = -1

    inv_dictionary = dict()                              # dictionary of couples (index, word)

    for index, key in enumerate(occurrence, 1):
        dictionary[key] = index
        inv_dictionary[index] = key

    for word in text:
        if word not in dictionary:                      # if a word does not belong to the vocabulary
            occurrence[n_words][1] += 1                 # then add 1 occurrence to 'ukn'

    data = list()                                       # creates data where each word is replaced by its index
    for word in text:
        if word in dictionary:
            data.append(dictionary[word])
        else:
            data.append(dictionary['ukn'])

    return data, occurrence, dictionary, inv_dictionary


# Generate data for training Word2Vec

def data_train_bis(data, window_size=2, nb_draws=2):
    """
    Creates training samples, composed of one input word associated with a vector of context words
    :param data: list
    :param window_size: size of the window around input word
    :param nb_draws: number of context words to be drawn to construct the context word
    :return: x [n_samples, n_draws], y [n_samples, n_draws]
    """

    x = np.ndarray(shape=(len(data), nb_draws), dtype=np.int32)   # matrix of input words
    y = np.ndarray(shape=(len(data), nb_draws), dtype=np.int32)   # matrix of context words

    for i in range(len(data)):

        # Create input word
        input_word = np.ndarray(shape=nb_draws, dtype=np.int32)

        if i < window_size:
            # Create context words
            window = data[:window_size]
            context_words = random.sample(window, nb_draws)

        elif i > len(data)-window_size:
            # Create context words
            window = data[len(data) - window_size:]
            context_words = random.sample(window, nb_draws)

        else:
            window = data[i - window_size: i + window_size + 1]
            del window[window_size]                               # remove central word
            context_words = random.sample(window, nb_draws)

        x[i] = input_word
        y[i] = context_words

    return x, y


def data_train(data, window_size=2, nb_draws=2):
    """
    Create training sample in the form of a list [[input, output],...]
    :param data: list of words
    :param window_size: size of the window around input word
    :param nb_draws: number of context words to be drawn
    :return: samples (list)
    """

    samples = []

    for index, input_word in enumerate(data):

        if index < window_size:
            window = data[:window_size]

        elif index > len(data)-window_size:
            window = data[len(data) - window_size:]

        else:
            window = data[index - window_size: index + window_size + 1]
            del window[window_size]

        context_words = random.sample(window, nb_draws)

        for context_word in context_words:
            samples.append([input_word, context_word])

    return samples


# function to convert numbers to one hot vectors

def one_hot_vector(index, vocab_size):
    """
    Convert integer-word to one-hot word
    :param index: integer representation of word
    :param vocab_size: size of vocabulary
    :return: temp : one-hot representation of word
    """
    temp = np.zeros(vocab_size)
    temp[index] = 1

    return temp


# create batch data

def make_batch(x_train, y_train, batch_size):
    """
    Create batches for learning
    :param x_train: array of one-hot representation of training inputs [len(data), vocab_size]
    :param y_train: array of one-hot representation of training context words [len(data), vocab_size]
    :param batch_size: size of batches
    :return:
    """

    batch_indexes = np.random.choice(len(data) - 1, batch_size)
    x_batch = x_train[batch_indexes, :]
    y_batch = y_train[batch_indexes, :]

    return x_batch, y_batch


def data_train_3(data, window_size=2, nb_draws=2):
    """
    Creates training samples, composed of one input word associated with a vector of context words
    :param data: list
    :param window_size: size of the window around input word
    :param nb_draws: number of context words to be drawn to construct the context word
    :return: x [n_samples, n_draws], y [n_samples, n_draws]
    """

    x = np.ndarray(shape=(len(data), nb_draws), dtype=np.int32)   # matrix of context words
    y = np.ndarray(shape=(len(data), nb_draws), dtype=np.int32)   # matrix of target words

    for i in range(len(data)):

        # Create target word
        target_word = np.ndarray(shape=nb_draws, dtype=np.int32)

        if i < window_size:
            # Create context words
            window = data[:window_size]
            context_words = random.sample(window, nb_draws)

        elif i > len(data)-window_size:
            # Create context words
            window = data[len(data) - window_size:]
            context_words = random.sample(window, nb_draws)

        else:
            window = data[i - window_size: i + window_size + 1]
            del window[window_size]                               # remove central word
            context_words = random.sample(window, nb_draws)

        x[i] = context_words
        y[i] = target_word

    return x, y
