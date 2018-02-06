'''Experiment'''

import matplotlib.pyplot as plt
import numpy as np

from word2vec import Word2Vec
import parameters

def main(text):

    params = getattr(parameters, text)

    w2v = Word2Vec(params['file'], window_size=params['window_size'],
                   learning_rate=params['learning_rate'],
                   vocab_size=params['vocab_size'],
                   embedding_size=params['embedding_size'],
                   n_negative=params['n_negative'])

    w2v.fit(n_iter=params['n_iter'], num_proc=params['num_proc'])

    print(w2v.process_time[-1]-w2v.process_time[0])

if __name__ == '__main__':
    main('debug_text')


