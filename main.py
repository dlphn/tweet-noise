
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from tweetsClustering.src.load_data import load_data
from tweetsClustering.src.clustering_algo import ClusteringAlgo
from tweetsClustering.src.compute_tfidf import TfIdf

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":

    day = "2019-03-06"
    logging.info("loading data")
    data = load_data("./data/tweets_{}.csv".format(day))
    logging.info("loaded {} tweets".format(len(data)))

    # Split data in train/test
    train, test = train_test_split(data, test_size=0.3)
    train = train.copy()
    test = test.copy()

    # Build clusters for train
    t = 0.7
    w = 100000
    batch_size = 100
    distance = "cosine"
    embedding_day = "base_fr_2"
    DATA_PATH = "tweetsClustering/data/"

    clustering = ClusteringAlgo(threshold=t, window_size=w, batch_size=batch_size, distance=distance)
    transformer = TfIdf()
    count_vectors = transformer.load_history(DATA_PATH + embedding_day).add_new_samples(train)
    vectors = transformer.compute_vectors(count_vectors, min_df=10)
    clustering.add_vectors(vectors)
    labels = clustering.incremental_clustering(method="brute")

    unique_clusters = set(labels)
    logging.info("Nb unique clusters: {}".format(len(unique_clusters)))

    train["pred"] = pd.Series(labels, dtype=train.label.dtype)

    # For each tweet in test :
    #   1. Predict cluster
    #   2. Analyze corresponding cluster
    #   3. According to analysis :
    #       - classify
    #       - go to next step
    #   4. Predict tweet class with random forest
    #   (5. Update clusters)
    #   6. Validation
    # for row in test.iterrows():
    row = test.iloc[0]
    print(row)
    test_row = train.copy().append(row)
    count_vectors_test = transformer.add_new_samples(test_row)
    vectors_test = transformer.compute_vectors(count_vectors_test, min_df=10)
    clustering.add_vectors(vectors_test)
    labels = clustering.incremental_clustering(method="brute")

    test_row["pred"] = pd.Series(labels, dtype=test_row.label.dtype)

    print(train.head())
    print(test_row.head())
    print(test_row[test_row["id"] == row["id"]])

    # train["pred"] = pd.Series(labels, dtype=train.label.dtype)
    # print(train.head())

