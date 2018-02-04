'''
Implementations of several parallel training functions
'''


from multiprocessing import JoinableQueue
from numba import jit
import numpy as np
import time
from multiprocessing import Process

from utils import make_batch, stoppable_thread


def Hogwild(X, y, n_iter, M_in, M_out, embedding_size, learning_rate, window_size, K, num_proc=2):
    '''
    Implementation of the Hogwild algorithm for Word2Vec
    :param X: (np array) one line is a window
    :param y: (np array) output words
    :param n_iter: (int) number of windows to be processed
    :param M_in: (np array, np.float32) [vocab_size, embedding_size]
    :param M_out: (np array, np.float32) [embedding_size, vocab_size]
    :param embedding_size: (int)
    :param learning_rate: (float32)
    :param num_proc: (int) number of parallel threads
    :return: trained M_in and M_out
    '''

    # For convenience n-iter as to be a multiple of 2500
    n_iter = int(n_iter/2500)*2500

    # Creating queue to feed the threads
    q = JoinableQueue()

    # Initialisation of threads
    workers = []
    loss = [0]*int(n_iter/2500)
    process_time = [0]*int(n_iter/2500)
    for _ in range(num_proc):
        worker = Process(target=_One_Hogwild_pass, args=(q, loss, process_time, window_size, K,
                                                                  M_in, M_out, embedding_size, learning_rate,))
        worker.start()
        workers.append(worker)

    # Training loop
    # Note that workers never wait for each other, even at the end of batches, this is allowed by the sparsity of
    # The target function in X, y
    for iter in range(n_iter+1):
        X_batch, y_batch = make_batch(X, y, K)
        for _ in range(len(X_batch)):
            # The queue is fed with the batch, and instructions for the recording of the error
            q.put([X_batch[_], y_batch[_, :], _ == len(X_batch)-1, iter])

    q.join()

    return M_in, M_out, loss, process_time


def _One_Hogwild_pass(q, loss, process_time, window_size, *args):
    '''
    :param q: queue
    :param *args: arguments for the Hogwild pass
    '''

    while True:
        context_word, target_words, get_info, batch_number = q.get()

        _One_Hogwild_pass_jitted(context_word, target_words, loss, batch_number, *args)

        # Every batch we average the error
        if (get_info) & (batch_number % 2500 == 0):
            print(batch_number, time.time())
            _get_info_jitted(loss, process_time, time.time(), batch_number, window_size)

        q.task_done()


@jit(nopython=True, nogil=True)
def _One_Hogwild_pass_jitted(context_word, target_words, loss, batch_number,
                             K, M_in, M_out, embedding_size, learning_rate):
    '''
    Performs on forward feeding and update of the Hogwild algorithm
    :param context_word: (int)
    :param target_words: (int)
    :param K: (int) number of negative samplings
    :param M_in: (np array, np.float32) [vocab_size, embedding_size]
    :param M_out: (np array, np.float32) [embedding_size, vocab_size]
    :param embedding_size: (int)
    :param learning_rate: (float32)
    '''

    M_in_update = np.zeros(shape=(embedding_size,), dtype=np.float32)
    out_error = 0

    for k in range(K):
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
        out_error += error

        # Calculating updates
        for _ in range(embedding_size):
            M_in_update[_] += error * M_out[target_word, _]

        for _ in range(embedding_size):
            M_out[target_word, _] += learning_rate * error * M_out[target_word, _]

    # Updating input matrix
    for _ in range(embedding_size):
        M_out[context_word, _] += learning_rate * M_in_update[_]

    loss[int(batch_number / 2500)] += out_error / K


@jit(nopython=True, nogil=True)
def _get_info_jitted(loss, process_time, t, batch_number, window_size):

    loss[int(batch_number / 2500)] = loss[int(batch_number / 2500)] / (window_size * 2500)
    process_time[int(batch_number / 2500)] = t