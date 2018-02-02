import numpy as np
from scipy import spatial
from utils import process_text
from training_functions import Hogwild


class Word2Vec(object):

    def __init__(self, text, window_size, learning_rate, vocab_size, embedding_size, n_negative):

        self.text = text
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_negative = n_negative


    def fit(self, n_iter, num_proc=2):
        '''
        Trains the Word2vec
        :param n_iter: (int)
        :param num_proc: (int) number of parallel threads
        '''

        self.n_iter = n_iter
        self.X, self.y, self.word_to_index, self.index_to_word = process_text(text=self.text,
                                                                              vocab_size=self.vocab_size,
                                                                              window_size=self.window_size)

        # Input layer weights
        self.M_in = np.random.uniform(low=-1.0, high=1.0, size=(self.vocab_size, self.embedding_size)).astype(np.float32)

        # Output layer weights
        self.M_out = np.random.uniform(low=-1.0, high=1.0, size=(self.vocab_size, self.embedding_size)).astype(np.float32)

        self.M_in, self.M_out = Hogwild(self.X, self.y, self.n_iter, self.M_in, self.M_out,
                                        self.embedding_size, self.learning_rate, num_proc=num_proc)

    def closest_to(self, words, n_neighbors):
        """
        Returns closest words to a list of words
        :param words: list of words
        :param n_neighbors: number of closest points to consider
        :return: neighbors [n_words, n_neighbors] : each row represents one word and is composed
        of the closest words to this word
        """

        sub_dictionary = {key: self.word_to_index[key] for key in words}  # dictionary containing words of interest
        words_int = list(sub_dictionary.values())

        # Compute similarity between words
        similarity = np.zeros(shape=(len(words_int), self.vocab_size))
        for i in words_int:
            for j in range(self.embedding_size):
                similarity[i, j] = 1 - spatial.distance.cosine(self.M_in[i, :], self.M_in[j, :])  # cosine distance

        similarity.sort(axis=1)

        # Select closest neighbors
        neighbors = similarity[:, similarity.shape[1]-n_neighbors:]
        for i in range(neighbors.shape[0]):
            for j in range(neighbors.shape[1]):
                neighbors[i, j] = self.index_to_word(neighbors[i, j])  # Translate index to associated word

        return neighbors









