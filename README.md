# Parallelizing Word2Vec

This repository is a school project for the course "Eléments logiciels pour le traitement des données massives" of our third year at ENSAE ParisTech. 

The goal is to implement a version of the article the article <b>Parallelizing Word2Vec in Shared and Distributed Memory<\b>, by Shihao Ji, Nadathur Satish, Sheng Li, Pradeep Dubey (2016, Parallel Computing Lab, Intel Labs, USA).

## Abstract

<i>From the original paper</i>

Word2Vec is a widely used algorithm for extracting low-dimensional vector representations of words. It generated considerable excitement in the machine learning and natural language processing (NLP) communities recently due to its exceptional performance in many NLP applications such as named entity recognition, sentiment analysis, machine translation and question answering. 

State-of-the-art algorithms including those by Mikolov et al. have been parallelized for multi-core CPU architectures but are based on vector-vector operations that are memory-bandwidth intensive and do not efficiently use computational resources. In this paper, we improve reuse of various data structures in the algorithm through the use of minibatching, hence allowing us to express the problem using matrix multiply operations. 

We also explore different techniques to distribute word2vec computation across nodes in a compute cluster, and demonstrate good strong scalability up to 32 nodes. In combination, these techniques allow us to scale up the computation near linearly across cores and nodes, and process hundreds of millions of words per second, which is the fastest word2vec implementation to the best of our knowledge.
