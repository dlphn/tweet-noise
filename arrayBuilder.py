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


class ArrayBuilder:
    """
    Retrieve data from the MongoDB database (text and label only).

    """
    def __init__(self):

        self.do_continue = True
        self.count = 0
        # connect to MongoDB
        client = MongoClient("mongodb+srv://" + MONGODB["USER"] + ":" + MONGODB["PASSWORD"] + "@" + MONGODB["HOST"] + "/" + MONGODB["DATABASE"] + "?retryWrites=true")
        self.db = client[MONGODB["DATABASE"]]
        self.data = []

    def retrieve(self):
        start = time.time()
        logging.info("Retrieving data...")
        tweets = self.db.tweets.find()
        logging.info("Building text array...")
        for obj in tweets:
            self.count += 1
            self.data.append(obj['text'])
        end = time.time()
        logging.info("Total of {0} elements retrieved in {1} seconds".format(self.count, end - start))
        return self.data

    def retrieve_text_and_labels(self):
        start = time.time()
        logging.info("Retrieving data...")
        tweets = self.db.tweets.find()
        logging.info("Building tweets array...")
        texts, labels = [], []
        for obj in tweets:
            self.count += 1
            texts.append(obj['text'])
            #if 'spam' in obj:
            #    labels.append('spam' if obj['spam'] else 'actualité')
            #else:
            #    labels.append('?')
            if 'type' in obj :
                if obj['type'] == "actualité":
                    labels.append('type actu' )
                if obj['type'] == "reaction" :
                    labels.append('type reaction')
                if obj['type'] == "conversation" :
                    labels.append('type conversation')
                if obj['type'] == "publicité" :
                    labels.append('type pub')
                if obj['type'] == "bot":
                    labels.append('type bot')
                if  obj['type'] == "other spam":
                    labels.append('type autre')
            else:
                labels.append('type ?')
        end = time.time()
        logging.info("Total of {0} elements retrieved in {1} seconds".format(self.count, end - start))
        return texts, labels


if __name__ == "__main__":
    mongo = ArrayBuilder()
    data = mongo.retrieve_text_and_labels()
    print(data[0])
    print(data[1])

