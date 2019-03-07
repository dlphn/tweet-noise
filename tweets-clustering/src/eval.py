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


def compute_proportions(data, row):
    count_spam, count_actualite, count_other = 0, 0, 0
    for _, tweet in data[data["pred"] == row["pred"]].iterrows():
        if tweet["label"] == 'spam':
            count_spam += 1
        elif tweet["label"] == 'actualité':
            count_actualite += 1
        else:
            count_other += 1
    return count_spam, count_actualite, count_other


def evaluate_classification(data):
    df = data.groupby(["label", "pred"]).size().unstack().fillna(0)
    clusters = df.reindex(df.sum().sort_values(ascending=False).index, axis=1)
    clusters = clusters.T[:]
    # print(clusters.head())

    nb_unit_clusters = 0
    nb_only_spam = 0
    nb_only_actualité = 0
    other = 0
    for index, cluster in clusters.iterrows():
        if cluster.name == -1:
            pass
        elif cluster['spam'] + cluster['actualité'] == 1:
            nb_unit_clusters += 1
        elif cluster['spam'] == cluster['spam'] + cluster['actualité']:
            nb_only_spam += 1
        elif cluster['actualité'] == cluster['spam'] + cluster['actualité']:
            nb_only_actualité += 1
        else:
            other += 1
    logging.info("nb of spam-only clusters: {}".format(nb_only_spam))
    logging.info("nb of actualité-only clusters: {}".format(nb_only_actualité))
    logging.info("nb of unit clusters: {}".format(nb_unit_clusters))
    logging.info("nb of other clusters: {}".format(other))
