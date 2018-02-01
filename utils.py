"""
Processing data :

1) From a document, create a list of words via convert_text
2) From the list of words, create correspondence word <-> integer and replace the list of words by a list of integers
    via building_dataset
3) Create x and y, where one row contains context words (y) and target word (x) via data_train
4) From x and y, randomly select one row i to construct a batch
5) Convert each word of x and y in a one-hot word vector
"""

import numpy as np
from nltk.corpus import stopwords
import string
from collections import Counter
import random


# Complete pre-processing function

def process_text(text, vocab_size, window_size):
    """
    Pre-process data before in a convenient way
    :param text: raw text document to be process
    :param vocab_size: size of the vocabulary we want to define
    :param batch_size: number of batches, which is also the number of context words to be sampled for each target word
    :return: X [len(text_list), batch_size] context words, y [len(text_list), batch_size] target words
    """

    text_list = convert_text(text)  # converts document into a list of words
    data, word_to_index, index_to_word, _ = building_dataset(text=text_list, vocab_size=vocab_size)
    X, y = data_train(data, window_size=window_size-1, nb_draws=window_size-1)

    return X, y, word_to_index, index_to_word


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

def building_dataset(text, vocab_size):
    """
    Function to build the data set
    :param text: list of words
    :param vocab_size: number of desired words for vocabulary
    :return: data (text where each word is replaced by its index), occurrence, dictionary
    """

    occurrence = Counter(text).most_common(vocab_size - 1)  # list of (word, occurrence)
    occurrence.append(['ukn', 0])                           # add couple (unknown, 0)

    word_to_index = dict()                                  # dictionary of couples (word, index)
    word_to_index['ukn'] = -1

    index_to_word = dict()                                  # dictionary of couples (index, word)
    index_to_word[-1] = 'ukn'

    for index, word in enumerate(occurrence, 1):
        key = word[0]
        word_to_index[key] = index
        index_to_word[index] = key

    for word in text:
        if word not in word_to_index:                       # if a word does not belong to the vocabulary
            occurrence[vocab_size][1] += 1                  # then add 1 occurrence to 'ukn'

    data = list()                                           # creates data where each word is replaced by its index
    for word in text:
        if word in word_to_index:
            data.append(word_to_index[word])
        else:
            data.append(word_to_index['ukn'])

    return data, word_to_index, index_to_word, occurrence


# Data train

def data_train(data, window_size=2, nb_draws=2):
    """
    Creates training samples, composed of one input word associated with a vector of context words
    :param data: list
    :param window_size: size of the window around input word
    :param nb_draws: number of context words to be drawn to construct the context word
    :return: x [n_samples, n_draws], y [n_samples, 1]
    """

    x = np.ndarray(shape=(len(data), nb_draws), dtype=np.int32)   # matrix of context words
    y = np.ndarray(shape=(len(data), nb_draws), dtype=np.int32)   # matrix of target words

    for i in range(len(data)):

        # Create target word
        target_word = np.ndarray(shape=(nb_draws,), dtype=np.int32)

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
        y = y[:, 0]

    return x, y


# create batch data

def make_batch(x_train, y_train, K=5):
    """
    Create batches for learning
    :param x_train: array of training inputs [len(data), vocab_size]
    :param y_train: array of training context words [len(data), vocab_size]
    :param K: number of negative sampling
    :return: x_batch [size = batch_size], y_batch [size = 1+K]
    """

    batch_index = np.random.choice(len(x_train[0]) - 1)
    x_batch = x_train[batch_index, :]
    y_batch = y_train[batch_index]

    # select K negative samples
    negative_samples = np.random.choice([y for y in y_train if (y not in x_batch) & (y != y_train[0])], K)
    y_batch = np.append(y_batch, negative_samples)  # add negative samples to y_batch

    return x_batch, y_batch


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

