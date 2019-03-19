# -*- coding: utf-8 -*-
"""
@author: bmazoyer

edits by dshi, hbaud, vlefranc
"""

import logging
import numpy as np
import time

from src.clustering_algo import ClusteringAlgo
from src.compute_tfidf import TfIdf, W2V, ResNetLayer
from src.eval import evaluate_classification
from src.load_data import load_data
from docs.config import LOG_LEVEL, DATA_PATH
import pandas as pd
from spreadsheet.spreadsheet import SpreadSheet

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=getattr(logging, LOG_LEVEL))

embedding_days = ['2018-07-30', 'base_fr', 'base_fr_2']
thresholds = np.arange(0.2, 0.9, 0.1)
window_sizes = [1000, 10000, 100000]
batch_sizes = [10, 100]

if __name__ == "__main__":

    start = time.time()

    spreadsheet_api = SpreadSheet("Tests clustering")
    row_id = spreadsheet_api.get_next_row_id()

    distance = "cosine"

    # day = "2018-07-30"
    # day = "2019-03-01"
    day = "2019-03-06"
    # day = "base_fr"

    logging.info("loading data")
    data = load_data("tweets_{}.csv".format(day))
    logging.info("loaded {} tweets".format(len(data)))

    for embedding_day in embedding_days:
        for t in thresholds:
            for w in window_sizes:
                for batch_size in batch_sizes:

                    clustering = ClusteringAlgo(threshold=t, window_size=w, batch_size=batch_size, distance=distance)
                    transformer = TfIdf()
                    count_vectors = transformer.load_history(DATA_PATH + embedding_day).add_new_samples(data)
                    vectors = transformer.compute_vectors(count_vectors, min_df=10)
                    clustering.add_vectors(vectors)
                    labels = clustering.incremental_clustering(method="brute")

                    unique_clusters = set(labels)
                    logging.info("Nb unique clusters: {}".format(len(unique_clusters)))

                    data["pred"] = pd.Series(labels, dtype=data.label.dtype)

                    stats = evaluate_classification(data)

                    """ Save stats in Google SpreadSheet """
                    setup = [
                        row_id,
                        embedding_day,
                        t,
                        w,
                        batch_size,
                        distance,
                        len(unique_clusters),  # nb of unique clusters
                        labels.count(-1),  # cluster -1
                        stats['spam_only'],
                        stats['spam_only_tweets'],
                        stats['actualité_only'],
                        stats['actualité_only_tweets'],
                        stats['unit']
                    ]

                    spreadsheet_api.write(setup)

                    row_id += 1

    end = time.time()
    logging.info("Tested different params in {} seconds".format(end - start))
