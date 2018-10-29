# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:35 2018

@author: dshi, hbaud, vlefranc
"""

import logging
from config import MONGODB
from pymongo import MongoClient

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


class DataLabelling:
    """
    Retrieve data from the MongoDB database.

    """
    def __init__(self):

        self.do_continue = True
        self.count = 0
        # connect to MongoDB
        client = MongoClient("mongodb+srv://" + MONGODB["USER"] + ":" + MONGODB["PASSWORD"] + "@" + MONGODB["HOST"] + "/" + MONGODB["DATABASE"] + "?retryWrites=true")
        self.db = client[MONGODB["DATABASE"]]

    def retrieve(self):

        for obj in self.db.tweets.find():
            self.count += 1
            self.label(obj)
        logging.info("Total of {} elements retrieved".format(self.count))

    def label(self, data):
        # display tweet and allow input from user true/false
        # save in mongodb


if __name__ == "__main__":
    data = DataLabelling()
    data.label()
