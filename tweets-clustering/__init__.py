# -*- coding: utf-8 -*-
"""
@author: bmazoyer

edits by dshi, hbaud, vlefranc
"""

import logging
from src.clustering_algo import ClusteringAlgo
from src.compute_tfidf import TfIdf, W2V, ResNetLayer
from src.eval import evaluate, cluster_event_match, average_distances
from src.load_data import load_data
from src.visualize_clusters import Visu
from docs.config import LOG_LEVEL, DATA_PATH
import pandas as pd
from sklearn.cluster import KMeans


logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=getattr(logging, LOG_LEVEL))



if __name__ == "__main__":

    """t = 0.4
    w = 100000
    batch_size = 100"""
    t = 0.7  # < 0.8
    w = 1000
    batch_size = 100
    distance = "cosine"
    # day = "2018-07-30"
    day = "2019-03-01"

    logging.info("loading data")
    data = load_data("tweets_{}.csv".format(day))
    logging.info("loaded {} tweets".format(len(data)))

    clustering = ClusteringAlgo(threshold=t, window_size=w, batch_size=batch_size, distance=distance)
    transformer = TfIdf()
    count_vectors = transformer.load_history(DATA_PATH + day).add_new_samples(data)
    # print(count_vectors)
    vectors = transformer.compute_vectors(count_vectors, min_df=10)
    clustering.add_vectors(vectors)
    labels = clustering.incremental_clustering(method="brute")

    logging.info("Nb unique clusters: {}".format(len(set(labels))))

    data["pred"] = pd.Series(labels, dtype=data.label.dtype)

    data.to_csv(DATA_PATH + "clustering_{0}_{1}_{2}_{3}.csv".format(day, t, w, batch_size), index=False)

    visualization = Visu(data, labels)
    visualization.plot("{0}_{1}_{2}_{3}".format(day, t, w, batch_size))
    visualization.write_html("{0}_{1}_{2}_{3}".format(day, t, w, batch_size))
    visualization.open_html("{0}_{1}_{2}_{3}".format(day, t, w, batch_size))
