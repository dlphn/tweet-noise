# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 09:51 2018

@author: dshi, hbaud, vlefranc
"""

import logging
# from datetime import datetime, timezone
from config import FILEDIR, FILEBREAK, MONGODB
from pymongo import MongoClient
import time
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


class JSONBuilder:
    """
    Retrieve data from the MongoDB database and write tweets in spam and actualité json files.

    """
    def __init__(self, category):

        self.do_continue = True
        self.count = 0
        self.line_count = 0
        self.number = 0
        self.category = category
        # self.current_file = FILEDIR + "tweets_" + category + "_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".json"
        self.current_file = FILEDIR + "tweets_" + category + ".json"
        # connect to MongoDB
        client = MongoClient("mongodb+srv://" + MONGODB["USER"] + ":" + MONGODB["PASSWORD"] + "@" + MONGODB["HOST"] + "/" + MONGODB["DATABASE"] + "?retryWrites=true")
        self.db = client[MONGODB["DATABASE"]]

    def retrieve(self):
        start = time.time()
        logging.info("Retrieving " + self.category + " data...")
        tweets = self.db.tweets.find({"spam": self.category == "spam"})
        logging.info("Building features file...")
        for obj in tweets:
            self.count += 1
            self.write(obj)
            if self.count % 100 == 0:
                logging.info("{} elements retrieved".format(self.count))
        with open(self.current_file, "a+", encoding='utf-8') as f:
            f.write(']')
        end = time.time()
        logging.info("Total of {0} elements retrieved in {1} seconds".format(self.count, end - start))

    def write(self, data):
        tweet = {"text": data['text'], "spam": data['spam']}
        json_string = json.dumps(tweet)
        datastore = json.loads(json_string)
        with open(self.current_file, "a+", encoding='utf-8') as f:
            if self.line_count == 0:
                f.write('[')
            else:
                f.write(',')
            json.dump(datastore, f)
        self.line_count += 1

        if self.line_count > FILEBREAK:
            with open(self.current_file, "a+", encoding='utf-8') as f:
                f.write(']')
            logging.info("Closing file {}".format(self.current_file))
            self.number += 1
            self.current_file = FILEDIR + "tweets_" + self.category + "_" + self.number + ".json"
            self.line_count = 0


if __name__ == "__main__":
    mongo = JSONBuilder("spam")
    mongo.retrieve()
    mongo2 = JSONBuilder("info")
    mongo2.retrieve()
