# -*- coding: utf-8 -*-
"""
@author: bmazoyer

edits by dshi, hbaud, vlefranc
"""


from sklearn import metrics
import logging
import pandas as pd
import numpy as np
import time
from docs.config import DATA_PATH


def cluster_event_match(data, pred):
    data["pred"] = pd.Series(pred, dtype=data.label.dtype)
    logging.info("{} labels, {} preds".format(len(data.label.unique()), len(data.pred.unique())))
    t0 = time.time()

    match = data.groupby(["label", "pred"], sort=False).size().reset_index(name="a")
    b, c = [], []
    for idx, row in match.iterrows():
        b_ = ((data["label"] != row["label"]) & (data["pred"] == row["pred"]))
        b.append(b_.sum())
        c_ = ((data["label"] == row["label"]) & (data["pred"] != row["pred"]))
        c.append(c_.sum())
    logging.info("match clusters with events took: {} seconds".format(time.time() - t0))
    match["b"] = pd.Series(b)
    match["c"] = pd.Series(c)
    # recall = nb true positive / (nb true positive + nb false negative)
    match["r"] = match["a"] / (match["a"] + match["c"])
    # precision = nb true positive / (nb true positive + nb false positive)
    match["p"] = match["a"] / (match["a"] + match["b"])
    match["f1"] = 2 * match["r"] * match["p"] / (match["r"] + match["p"])
    match = match.sort_values("f1", ascending=False)
    macro_average_f1 = match.drop_duplicates("label").f1.mean()
    return match, macro_average_f1


def average_distances(vectors, data, metric):
    labels = data.label.unique()
    print(len(labels))
    inter_dist = []
    intra_dist = []
    avg_dist = np.zeros((labels.size, labels.size))
    for i, ilabel in enumerate(labels):
        t0 = time.time()
        for j, jlabel in enumerate(labels):
            if i <= j:
                mean_pairwise_distance = metrics.pairwise_distances(
                    vectors[(data.label == ilabel).values],
                    vectors[(data.label == jlabel).values],
                    metric=metric
                ).mean()
                avg_dist[i, j] = mean_pairwise_distance
                if i < j:
                    inter_dist.append(mean_pairwise_distance)
                elif i == j:
                    max_pairwise_distance = metrics.pairwise_distances(
                        vectors[(data.label == ilabel).values],
                        vectors[(data.label == jlabel).values],
                        metric=metric
                    ).max()
                    intra_dist.append(mean_pairwise_distance)
                    logging.info("mean inter_distance event {}: {}".format(i, mean_pairwise_distance))
                    logging.info("max inter_distance event {}: {}".format(i, max_pairwise_distance))
    logging.info("mean inter distance: {}".format(np.array(inter_dist).mean()))
    logging.info("mean intra distance: {}".format(np.array(intra_dist).mean()))


def evaluate(data, pred):
    pred = pd.Series(pred)
    logging.info("nb of detected clusters: {}".format(pred.unique().size))
    data = data.assign(pred=pred.values)
    data = data[data.label.notnull()]
    logging.info("nb of annotated tweets: {}".format(data.shape[0]))
    logging.info("nb of events in annotated tweets: {}".format(data.label.unique().size))
    logging.info("nb of detected clusters in annotated tweets: {}".format(data.pred.unique().size))
    logging.info("V-measure: {}".format(metrics.v_measure_score(data["label"], data["pred"])))
