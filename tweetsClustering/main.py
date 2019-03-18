# -*- coding: utf-8 -*-
"""
@author: bmazoyer

edits by dshi, hbaud, vlefranc
"""

import logging
from src.clustering_algo import ClusteringAlgo
from src.compute_tfidf import TfIdf, W2V, ResNetLayer
from src.eval import evaluate, cluster_event_match, average_distances, evaluate_classification
from src.load_data import load_data
from src.visualize_clusters import Visu
from docs.config import LOG_LEVEL, DATA_PATH
import pandas as pd
from sklearn.cluster import KMeans
from spreadsheet.spreadsheet import SpreadSheet


logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=getattr(logging, LOG_LEVEL))


if __name__ == "__main__":

    t = 0.7
    w = 100000
    batch_size = 100
    distance = "cosine"
    # day = "2018-07-30"
    # day = "2019-03-01"
    day = "2019-03-06"
    # day = "base_fr"
    # embedding_day = "base_fr"
    embedding_day = "base_fr_2"
    # embedding_day = "2018-07-30"

    logging.info("loading data")
    data = load_data("./data/tweets_{}.csv".format(day))
    logging.info("loaded {} tweets".format(len(data)))

    """clustering = ClusteringAlgo(threshold=t, window_size=w, batch_size=batch_size, distance=distance)
    transformer = TfIdf()
    count_vectors = transformer.load_history(DATA_PATH + embedding_day).add_new_samples(data)
    # print(count_vectors)
    vectors = transformer.compute_vectors(count_vectors, min_df=10)
    clustering.add_vectors(vectors)
    labels = clustering.incremental_clustering(method="brute")

    unique_clusters = set(labels)
    logging.info("Nb unique clusters: {}".format(len(unique_clusters)))

    data["pred"] = pd.Series(labels, dtype=data.label.dtype)

    data.to_csv(DATA_PATH + "clustering_{0}_{1}_{2}_{3}_{4}.csv".format(day, t, w, batch_size, embedding_day), index=False)

    visualization = Visu(data, labels)
    visualization.plot("{0}_{1}_{2}_{3}_{4}".format(day, t, w, batch_size, embedding_day))
    visualization.plot("{0}_{1}_{2}_{3}_{4}".format(day, t, w, batch_size, embedding_day), "category")
    visualization.write_html("{0}_{1}_{2}_{3}_{4}".format(day, t, w, batch_size, embedding_day))
    visualization.open_html("{0}_{1}_{2}_{3}_{4}".format(day, t, w, batch_size, embedding_day))

    # cluster_event_match(data, labels)
    # evaluate(data, labels)
    stats = evaluate_classification(data)"""

    """ Save stats in Google SpreadSheet
    setup = [
        embedding_day,
        t,
        w,
        batch_size,
        distance,
        len(unique_clusters),  # nb of unique clusters
        labels.count(-1),  # cluster -1
        stats['spam_only'],
        stats['actualité_only'],
        stats['unit']
    ]

    spreadsheet_api = SpreadSheet("Tests clustering")
    print(setup)
    # spreadsheet_api.write(setup) """

    """ Save embedding """
    # transformer.save(day)
