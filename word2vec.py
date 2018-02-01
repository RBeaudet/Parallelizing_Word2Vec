import numpy as np
import utils

class Word2Vec(object):

    def __init__(self, n_epoch, batch_size, learning_rate, vocab_size, embedding_size, n_batch, n_negative):

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_batch = n_batch
        self.n_negative = n_negative

    def fit(self, X_train, y_train):

        self.M_in = np.random.uniform(low=-1.0, high=1.0, size=(self.vocab_size, self.embedding_size))
        self.b_in = np.zeros(self.embedding_size)

        self.M_out = np.random.uniform(low=-1.0, high=1.0, size=(self.embedding_size, self.vocab_size))
        self.b_out = np.zeros(self.vocab_size)

    def eval(self):


