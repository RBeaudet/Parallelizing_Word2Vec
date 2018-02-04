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
from numpy import random
from threading import Thread


# Complete pre-processing function

def process_text(text, vocab_size, window_size):
    """
    Pre-process data before in a convenient way
    :param text: raw text document to be process
    :param vocab_size: size of the vocabulary we want to define
    :param window_size: number of words to get when construction the window, in EACH SIDES of the target word
    :return: X [len(text_list), batch_size] context words, y [len(text_list), batch_size] target words
    """

    text_list = convert_text(text)  # converts document into a list of words
    data, word_to_index, index_to_word, _ = building_dataset(text=text_list, vocab_size=vocab_size)
    X, y = data_train(data, window_size=window_size, nb_draws=2*window_size)

    return X, y, word_to_index, index_to_word


# Convert document into a list of word

def convert_text(text):
    """
    Create list of words from a text document
    :param text: raw text file
    :return: text_list (list)
    """

    with open(text, 'r') as file:
        text = file.read().translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

    text_list = text.lower().split()

    # deleting stopwords
    stop_words = set(stopwords.words('english'))
    text_list = [word for word in text_list if word not in stop_words]

    return text_list


# Pre-processing

def building_dataset(text, vocab_size):
    """
    Function to build the data set
    :param text: list of words
    :param vocab_size: number of desired words for vocabulary
    :return: data (text where each word is replaced by its index), occurrence, dictionary
    """

    occurrence = [['ukn', 0]] + Counter(text).most_common(vocab_size)  # list of (word, occurrence)

    # dictionary of couples (word, index)
    word_to_index = dict()
    # dictionary of couples (index, word)
    index_to_word = dict()

    for index, word in enumerate(occurrence):
        key = word[0]
        word_to_index[key] = index
        index_to_word[index] = key

    for word in text:
        # if a word does not belong to the vocabulary
        if word not in word_to_index:
            # then add 1 occurrence to 'ukn'
            occurrence[0][1] += 1

    # creates data where each word is replaced by its index
    data = list()
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
    y = np.ndarray(shape=(len(data),), dtype=np.int32)   # matrix of target words

    for i in range(len(data)):

        # Create target word
        target_word = data[i]

        if i < window_size:
            # Create context words
            window = data[:(i + 1 + window_size)]
            del window[i]
            context_words = random.choice(window, nb_draws, replace=True)

        elif i >= len(data)-window_size:
            # Create context words
            window = data[(i - window_size):]
            del window[window_size]
            context_words = random.choice(window, nb_draws, replace=True)

        else:
            window = data[i - window_size: (i + window_size + 1)]
            # remove central word
            del window[window_size]
            context_words = random.choice(window, nb_draws, replace=False)

        x[i, :] = context_words
        y[i] = target_word

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

    batch_index = random.choice(x_train.shape[0] - 1)
    x_batch = x_train[batch_index, :]
    y_true = y_train[batch_index]

    # select K negative samples
    y_batch = np.zeros(shape=(len(x_batch), K+1), dtype=np.int32)
    for _ in range(len(x_batch)):
        negative_samples = np.random.choice([y for y in y_train if (y not in x_batch) & (y != y_train[0])], K)
        y_batch_line = np.append(y_true, negative_samples)
        y_batch[_, :] = y_batch_line

    return x_batch, y_batch


def make_batch_SGEMM(x_train, y_train, K=5):
    """
    Create batches for learning with SGEMM
    :param x_train: array of training inputs [len(data), vocab_size]
    :param y_train: array of training context words [len(data), vocab_size]
    :param K: number of negative sampling
    :return: x_batch [size = batch_size], y_batch [size = 1+K]
    """

    batch_index = random.choice(x_train.shape[0] - 1)
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

# training

class stoppable_thread(Thread):
    '''
    A slight modification of the Thread class so threads can be easily stoppable
    '''

    def __init__(self, target, args):
        Thread.__init__(self, group=None, target=target, args=args)
        Thread.daemon = True


    def run(self):
        try:
            if self._target:
                self.keepRunning = True
                while self.keepRunning:
                    self._target(*self._args, **self._kwargs)

        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs


if __name__ == '__main__':
    import parameters

    params = getattr(parameters, 'debug_text')
    text_list = convert_text(params['file'])
    #print(text_list)
    data, word_to_index, index_to_word, occurence = building_dataset(text=text_list, vocab_size=params['vocab_size'])
    #print(data)
    #print(word_to_index)
    #print(index_to_word)
    #print(occurence)
    X, y = data_train(data, window_size=params['window_size'], nb_draws=2*params['window_size'])
    #print(X)
    #print(y)
    X_batch, y_batch = make_batch(X, y, K=params['n_negative'])
    print(X_batch)
    print(y_batch)