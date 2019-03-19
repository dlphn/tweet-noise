
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from tweetsClustering.src.load_data import load_data, load_all_data
from tweetsClustering.src.clustering_algo import ClusteringAlgo
from tweetsClustering.src.compute_tfidf import TfIdf

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":

    # day = "2019-03-06"
    day = "2019-03-18"
    # day = "test"
    logging.info("loading data")
    data = load_all_data("./data/tweets_{}.csv".format(day))
    logging.info("loaded {} tweets".format(len(data)))

    # Split data in train/test
    """train, test = train_test_split(data, test_size=0.3)
    train = train.copy()
    test = test.copy()"""

    # Build clusters for train
    t = 0.7
    w = 100000
    batch_size = 100
    distance = "cosine"
    embedding_day = "base_fr_2"
    DATA_PATH = "tweetsClustering/data/"

    clustering = ClusteringAlgo(threshold=t, window_size=w, batch_size=batch_size, distance=distance)
    transformer = TfIdf()
    # count_vectors = transformer.load_history(DATA_PATH + embedding_day).add_new_samples(train)
    count_vectors = transformer.load_history(DATA_PATH + embedding_day).add_new_samples(data)
    vectors = transformer.compute_vectors(count_vectors, min_df=10)
    clustering.add_vectors(vectors)
    labels = clustering.incremental_clustering(method="brute")

    unique_clusters = set(labels)
    logging.info("Nb unique clusters: {}".format(len(unique_clusters)))

    data["pred"] = pd.Series(labels, dtype=data.label.dtype)
    print(data.head())

    # Split data in train/test
    train, test = train_test_split(data, test_size=0.3)
    train = train.copy()
    test = test.copy()
    train['dataset'] = 'train'
    test['dataset'] = 'test'

    df_new = pd.concat([train, test]).sort_index()
    print(df_new.head())

    # TODO: send train to Random Forest classifier

    # TODO: send test one by one
    for row in test.iterrows():
        # TODO: send to analysis
        pass

    # For each tweet in test :
    #   1. Predict cluster
    #   2. Analyze corresponding cluster
    #   3. According to analysis :
    #       - classify
    #       - go to next step
    #   4. Predict tweet class with random forest
    #   (5. Update clusters)
    #   6. Validation

