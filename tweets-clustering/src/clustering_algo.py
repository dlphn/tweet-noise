# -*- coding: utf-8 -*-
"""
@author: bmazoyer

edits by dshi, hbaud, vlefranc
"""


from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, vstack, issparse
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import logging

class ClusteringAlgo():

    def __init__(self, threshold=0.65, window_size=300000, batch_size=2500, distance="cosine"):
        self.M = None
        self.t = threshold
        self.w = window_size
        self.batch_size = batch_size
        self.index = None
        self.zeros_vectors = None
        self.thread_id = 0
        self.distance = distance

    def add_vectors(self, vectors):
        self.M = vectors
        if issparse(vectors):
            self.zeros_vectors = vectors.getnnz(1) == 0
        else:
            self.zeros_vectors = ~vectors.any(axis=1)

    def iter_on_matrix(self, ):
        matrix = self.M[~self.zeros_vectors]
        for idx in range(0, matrix.shape[0], self.batch_size):
            vectors = matrix[idx:min(idx + self.batch_size, matrix.shape[0])]
            yield idx, vectors

    def reindex(self, method, data):
        if method == "brute":
            self.index = NearestNeighbors(n_neighbors=1, algorithm='brute', metric=self.distance).fit(data)

    def brute_NN(self, tweets):

        distance, neighbor_exact = self.index.kneighbors(tweets)
        return distance.transpose()[0], neighbor_exact.transpose()[0]

    def incremental_clustering(self, method="brute"):
        if issparse(self.M):
            T = csr_matrix((self.w, self.M.shape[1]))
        else:
            T = np.zeros((self.w, self.M.shape[1]), dtype=self.M.dtype)
        threads = np.zeros(self.w, dtype="int")
        total_threads = []

        for idx, tweets in self.iter_on_matrix():
            i = idx % self.w
            j = i + tweets.shape[0]
            if idx == 0:
                threads[:j] = np.arange(self.thread_id, self.thread_id + j)
                self.thread_id = self.thread_id + j
                t0 = time.time()
            else:
                self.reindex(method=method, data=T)
                if method == "brute":
                    distances, neighbors = self.brute_NN(tweets)

                # points that have a close neighbor in the window get the label of that neighbor
                under_t = np.array(distances) < self.t
                threads[i:j][under_t] = threads[neighbors[under_t]]

                # assign new labels to points that do not have close enough neighbors
                distant_neighbors = neighbors[~under_t]
                new_labels = np.arange(self.thread_id, self.thread_id + len(distant_neighbors))
                threads[i:j][~under_t] = new_labels
                if new_labels.size != 0:
                    self.thread_id = max(new_labels) + 1

            if issparse(self.M):
                T = vstack([T[:i], tweets, T[j:]])  # much faster than T[i:j] = tweets
            else:
                T[i:j] = tweets

            total_threads.extend(threads[i:j])
            clock_time = time.time() - t0
            logging.info("{} tweets treated in {} seconds".format(idx + tweets.shape[0], clock_time))

        total_threads = np.array(total_threads)
        total_threads_with_zeros_vectors = np.zeros(self.M.shape[0], dtype="int")
        total_threads_with_zeros_vectors[self.zeros_vectors] = -1
        total_threads_with_zeros_vectors[~self.zeros_vectors] = total_threads
        return total_threads_with_zeros_vectors.tolist()


