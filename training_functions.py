'''
Implementations of several parallel training functions
'''


from threading import Thread
import queue
from numba import jit

from utils import make_batch


class _stopable_thread(Thread):
    def __init__(self, target, args):
        Thread.__init__(self, target, args)

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


def Hogwild(X_train, y_train, n_iter, M_in, b_in, M_out, b_out, embedding_size, K, num_proc=2):

    q = queue.Queue()

    for _ in range(num_proc):
        worker = _stopable_thread(target=_One_Hogwild_pass, args=(q, M_in, b_in, M_out, b_out, embedding_size))
        worker.start()

    for iter in range(n_iter):
        X, y = make_batch(X_train, y_train, K)
        for _ in X.shape()[0]:
            q.put([X[_], y])

    return M_in, b_in, M_out, b_out


@jit(nogil=True)
def _One_Hogwild_pass(q, M_in, b_in, M_out, b_out, embedding_size):

    context_word, target_words = q.get()
    for target_word in target_words:

        out = 0
        for _ in range(embedding_size):
            out += M_in[target_word, _] * M_out[]




