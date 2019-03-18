
import logging
from tweetsClustering.src.load_data import load_data

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":

    day = "2019-03-06"
    logging.info("loading data")
    data = load_data("./data/tweets_{}.csv".format(day))
    logging.info("loaded {} tweets".format(len(data)))

    # Split data in train/test
    print(data.head())

    # Build clusters for train

    # For each tweet in test :
    #   1. Predict cluster
    #   2. Analyze corresponding cluster
    #   3. According to analysis :
    #       - classify
    #       - go to next step
    #   4. Predict tweet class with random forest
    #   (5. Update clusters)
    #   6. Validation

