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
        self.current_file = FILEDIR + "tweets_" + self.date + ".csv"

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

    def write(self):
        start = time.time()
        logging.info("Retrieving data...")
        tweets = self.db.tweets.find()
        logging.info("Building tweets csv...")
        # TODO : delete file if already exists
        with open(self.current_file, "a+", encoding='utf-8') as f:
            f.write('id,label,text\n')
            for obj in tweets:
                f.write(
                    obj["id_str"] +
                    "," + ('spam' if obj["spam"] else 'actualité') +
                    "," + obj["text"].replace("\n", " ") +
                    "," + obj["type"] +
                    "\n")
                self.line_count += 1
        end = time.time()
        logging.info("Total of {0} elements retrieved in {1} seconds".format(self.line_count, end - start))


if __name__ == "__main__":
    mongo = ArrayBuilder()
    # data = mongo.retrieve_text_and_labels()
    # print(data[0])
    # print(data[1])
    mongo.write()

