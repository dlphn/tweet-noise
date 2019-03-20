
import json
import logging
from config import MONGODB, DATA_DIR
from pymongo import MongoClient

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


class LocalTweetsLoader:
    """
    Import a local json file of tweets to be added in the Mongo database
    """

    def __init__(self, file_name):
        self.data_file_name = DATA_DIR + file_name
        self.count = 0
        self.data = []
        client = MongoClient("mongodb+srv://"
                             + MONGODB["USER"] + ":"
                             + MONGODB["PASSWORD"] + "@"
                             + MONGODB["HOST"] + "/"
                             + MONGODB["DATABASE"] + "?retryWrites=true")
        self.db = client[MONGODB["DATABASE"]]

    def load(self, write=False):
        with open(self.data_file_name) as data_file:
            for line in data_file:
                tweet = json.loads(line, encoding='utf-8')
                if write:
                    self.db.imported.insert_one(tweet)
                    self.count += 1
                    # if self.count % 100 == 0:
                    logging.info("{} elements imported".format(self.count))
                else:
                    print(tweet['id'], tweet['text'])
                    self.data.append(tweet)


if __name__ == "__main__":
    loader = LocalTweetsLoader('labelled_tweets.json')
    loader.load()
