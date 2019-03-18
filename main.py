
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    print('hello')

    # Split data in train/test

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

