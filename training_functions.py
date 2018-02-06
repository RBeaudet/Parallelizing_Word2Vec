'''
Implementations of several parallel training functions
'''


from multiprocessing import JoinableQueue, Queue
from numba import jit
import numpy as np
import time
from multiprocessing import Process, Manager, Pool
import multiprocessing as mp

from utils import make_batch, stoppable_thread, sharedArray, negative_sampling


def Hogwild(X, y, n_iter, vocab_size, embedding_size, learning_rate, window_size, K, occurence, num_proc=2):
    '''
    Implementation of the Hogwild algorithm for Word2Vec
    :param X: (np array) one line is a window
    :param y: (np array) output words
    :param n_iter: (int) number of windows to be processed
    :param M_in: (np array, np.float32) [vocab_size, embedding_size]
    :param M_out: (np array, np.float32) [embedding_size, vocab_size]
    :param vocab_size: vocabulary size
    :param embedding_size: (int)
    :param learning_rate: (float32)
    :param num_proc: (int) number of parallel threads
    :return: trained M_in and M_out
    '''

    # On ne s'interresse qu'aux occurences
    frequence = np.array(occurence)[:, 1].astype(np.int)

    M_in = sharedArray(base_array=np.random.uniform(low=-1.0, high=1.0, size=(vocab_size+1, embedding_size)), lock=False)
    M_out = sharedArray(base_array=np.random.uniform(low=-1.0, high=1.0, size=(vocab_size+1, embedding_size)), lock=False)

    # For convenience n-iter as to be a multiple of 2500
    n_iter = int(n_iter/2500)*2500

    # Initialisation of threads
    workers = []
    input_q = JoinableQueue()
    output_q = JoinableQueue()

    loss = np.array([0]*(int(n_iter/2500)+1))
    loss = sharedArray(base_array=np.expand_dims(loss, axis=1), lock=False)

    process_time = np.array([0]*(int(n_iter/2500)+1))
    process_time = sharedArray(base_array=np.expand_dims(process_time, axis=1), lock=False)

    for _ in range(num_proc):
        worker = Process(target=_One_Hogwild_pass, args=(input_q, output_q, frequence, K, M_in.shape,
                                         embedding_size, learning_rate,))
        worker.start()
        workers.append(worker)

    # Training loop
    # Note that workers never wait for each other, even at the end of batches, this is allowed by the sparsity of
    # The target function in X, y
    print('making_batches')
    X_batchs = []
    y_batchs = []
    for iter in range(n_iter+1):
        X_batch, y_batch = make_batch(X, y)
        X_batchs.append(X_batch)
        y_batchs.append(y_batch)

    print('Training starts')
    for iter in range(n_iter+1):
        X_batch = X_batchs[iter]
        y_batch = y_batchs[iter]
        for _ in range(len(X_batch)):
            # The queue is fed with the batch, and instructions for the recording of the error
            input_q.put([X_batch[_], X_batch, y_batch, M_in, M_out, False])

        input_q.join()

        _update_coefs(output_q, M_in, M_out, iter, window_size, loss, process_time, iter % 2500 == 0)

    input_q.join()
    for worker in workers:
        input_q.put([None, None, None, None, None, True])
    for worker in workers:
        worker.join()

    return M_in, M_out, loss, process_time


def _One_Hogwild_pass(q_in, q_out, frequence, K, Shape, *args):
    '''
    :param q: queue
    :param *args: arguments for the Hogwild pass
    '''

    M_in_update = np.zeros(shape=Shape, dtype=np.float32)
    M_out_update = np.zeros(shape=Shape, dtype=np.float32)
    STOP = False

    while True:
        if q_in.empty()==False:
            context_word, context_words, target_word, M_in, M_out, STOP = q_in.get()

            # stop condition
            if STOP:
                break
            # Sampling negative samples
            target_words = [target_word] + negative_sampling(frequence, context_words, K)

            # Executing hogwild pass
            M_in_update, M_out_update, loss = _One_Hogwild_pass_jitted(context_word, target_words,
                                                                       K, M_in, M_out, M_in_update, M_out_update, *args)

            q_in.task_done()
            q_out.put([M_in_update, M_out_update, loss, context_word, target_words])

        if STOP:
            break


@jit(nopython=True, nogil=True)
def _One_Hogwild_pass_jitted(context_word, target_words, K, M_in, M_out,
                             M_in_update, M_out_update, embedding_size, learning_rate):
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

    out_error = 0
    loss = 0

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
            M_in_update[target_word, _] += learning_rate * error * M_out[target_word, _]

        for _ in range(embedding_size):
            M_out_update[target_word, _] += learning_rate * error * M_out[target_word, _]

    loss += out_error / K

    return M_in_update, M_out_update, loss


def _update_coefs(q, M_in, M_out, batch_number, window_size, loss, process_time, get_info):

    loss_update = 0

    while q.empty()==False:
        M_in_update, M_out_update, loss_update_temp, to_update_in, to_update_out = q.get_nowait()
        loss_update += loss_update_temp
        _update_jitted(M_in, M_out, M_in_update, M_out_update, to_update_in, to_update_out)
        q.task_done()

    if get_info:
        print(batch_number, time.time())
        _get_info_jitted(loss, loss_update, process_time, time.time(), batch_number, window_size)


@jit(nopython=True, nogil=True)
def _update_jitted(M_in, M_out, M_in_update, M_out_update, to_update_in, to_update_out):

    for _ in range(M_in.shape[1]):
        M_in[to_update_in, _] += M_in_update[to_update_in, _]

        for i in to_update_out:
            M_out[i, _] += M_out_update[i, _]


@jit(nopython=True, nogil=True)
def _get_info_jitted(loss, loss_update, process_time, t, batch_number, window_size):
    """
    :param loss:
    :param process_time:
    :param t:
    :param batch_number:
    :param window_size:
    :return:
    """
    loss[int(batch_number / 2500)] = loss_update / (window_size * 2500)
    process_time[int(batch_number / 2500)] = t
