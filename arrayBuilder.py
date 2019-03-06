# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 09:51 2018

@author: dshi, hbaud, vlefranc
"""

import logging
from datetime import datetime, timezone
from config import FILEDIR, FILEBREAK, MONGODB
from pymongo import MongoClient
import time
import json
import os
import csv

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
        self.line_count = 0
        self.file_count = 1
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.current_file = FILEDIR + "tweets_all_" + self.date + ".csv"

    def retrieve(self):
        start = time.time()
        logging.info("Retrieving data...")
        tweets = self.db.tweets.find()
        logging.info("Building text array...")
        for obj in tweets:
            self.count += 1
            self.data.append(obj["extended_tweet"]["full_text"] if obj["truncated"] else obj["text"])
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
            texts.append(obj["extended_tweet"]["full_text"] if obj["truncated"] else obj["text"])
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
                if obj['type'] == "other spam":
                    labels.append('type autre')
            else:
                labels.append('type ?')
        end = time.time()
        logging.info("Total of {0} elements retrieved in {1} seconds".format(self.count, end - start))
        return texts, labels

    def write(self):
        """
        Retrieve MongoDB tweets and save id, label, text in csv file
        """
        start = time.time()
        logging.info("Retrieving data...")
        tweets = self.db.tweets.find()
        logging.info("Building tweets csv...")
        try:
            os.remove(self.current_file)
        except OSError:
            pass
        writer = csv.writer(open(self.current_file, 'w'))
        writer.writerow(['id', 'label', 'category', 'text'])

        for obj in tweets:
            line = [
                obj["id_str"],
                'spam' if obj["spam"] else 'actualité',
                obj["type"],
                obj["extended_tweet"]["full_text"] if obj["truncated"] else obj["text"]
            ]
            writer.writerow(line)
            self.line_count += 1
        end = time.time()
        logging.info("Total of {0} elements retrieved in {1} seconds".format(self.line_count, end - start))


if __name__ == "__main__":
    mongo = ArrayBuilder()
    # data = mongo.retrieve_text_and_labels()
    # print(data[0])
    # print(data[1])
    mongo.write()

