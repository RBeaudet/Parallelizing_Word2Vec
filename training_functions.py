'''
Implementations of several parallel training functions
'''


from threading import Thread
import queue
import cython
from numba import jit

from utils import make_batch

def Hogwild(X_train, y_train, n_iter, M_in, b_in, M_out, b_out, num_proc=2):

    q = queue.Queue()

    for _ in range(num_proc):
        worker = Thread(target=_One_Hogwild_pass, args=(q, M_in, b_in, M_out, b_out))
        worker.start()

    for iter in range(n_iter):
        X, y = make_batch(X_train, y_train)
        for _ in X.shape()[0]:
            q.put([X[_], y])

        q.join()

    return M_in, b_in, M_out, b_out


@jit
def _One_Hogwild_pass(q, M_in, b_in, M_out, b_out):
    with nogil:


