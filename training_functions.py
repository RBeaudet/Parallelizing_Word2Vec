'''
Implementations of several parallel training functions
'''


import queue
from numba import jit
import numpy as np

from utils import make_batch, stoppable_thread


def Hogwild(X, y, n_iter, M_in, M_out, embedding_size, learning_rate, K, num_proc=2):
    '''
    Implementation of the Hogwild algorithm for Word2Vec
    :param X_train: (np array) one line is a window
    :param y_train: (np array) output words
    :param n_iter: (int) number of windows to be processed
    :param M_in: (np array, np.float32) [vocab_size, embedding_size]
    :param M_out: (np array, np.float32) [embedding_size, vocab_size]
    :param embedding_size: (int)
    :param learning_rate: (float32)
    :param num_proc: (int) number of parallel threads
    :return: trained M_in and M_out
    '''

    # Creating queue to feed the threads
    q = queue.Queue()

    # Initialisation of threads
    workers = []
    for _ in range(num_proc):
        worker = stoppable_thread(target=_One_Hogwild_pass, args=(q, M_in, M_out, embedding_size, learning_rate,))
        worker.start()
        workers.append(worker)

    # Training loop
    # Note that workers never wait for each other, even at the end of batches, this is allowed by the sparsity of
    # The target function in X, y
    for iter in range(n_iter):
        X_batch, y_batch = make_batch(X, y, K)
        for _ in range(len(X_batch)):
            q.put([X_batch[_], y_batch[_, :]])

    q.join()

    # Shutting down threads
    for worker in workers:
        worker.keepRunning = False

    return M_in, M_out


def _One_Hogwild_pass(q, *args):
    '''
    :param q: queue
    :param *args: arguments for the Hogwild pass
    '''

    context_word, target_words = q.get()
    _One_Hogwild_pass_jitted(context_word, target_words, *args)
    q.task_done()


@jit(nopython=True, nogil=True)
def _One_Hogwild_pass_jitted(context_word, target_words, M_in, M_out, embedding_size, learning_rate):
    '''
    Performs on forward feeding and update of the Hogwild algorithm
    :param q: queue where to fetch data
    :param M_in: (np array, np.float32) [vocab_size, embedding_size]
    :param M_out: (np array, np.float32) [embedding_size, vocab_size]
    :param embedding_size: (int)
    :param learning_rate: (float32)
    '''

    M_in_update = np.zeros(shape=(embedding_size,), dtype=np.float32)

    for k in range(len(target_words)):
        target_word = target_words[k]
        if k == 0:
            label = 1

        else:
            label = 0

        # Forward feed
        out = 0
        for _ in range(embedding_size):
            out += M_in[context_word, _] * M_out[target_word, _]

        error = label - 1 / (1 + np.exp(-out))

        # Calculating updates
        for _ in range(embedding_size):
            M_in_update[_] += error * M_out[target_word, _]

        for _ in range(embedding_size):
            M_out[target_word, _] += learning_rate * error * M_out[target_word, _]

    # Updating input matrix
    for _ in range(embedding_size):
        M_out[context_word, _] += learning_rate * M_in_update[_]