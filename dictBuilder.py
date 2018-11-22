# -*- coding: utf-8 -*-
import json
import logging
import time

from config import FILEDIR

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


class DictBuilder:
    """
    Build spam and whitelist keywords lists
    """

    def __init__(self, category):
        self.category = category
        self.current_file = FILEDIR + "tweets_" + category + ".json"
        self.tweets = []
        self.count = 0

        # initialize our vocabulary and its size
        self.vocabulary = {}
        self.vocabularySize = 0

        # number of documents we have learned from
        self.totalDocuments = 0

        # document frequency table for each of our categories
        # => for each category, how often were documents mapped to it
        self.docCount = {}

        # for each category, how many words total were mapped to it
        self.wordCount = {}

        # word frequency table for each category
        # => for each category, how frequent was a given word mapped to it
        self.wordFrequencyCount = {}

        # hashmap of our category names
        self.categories = {}

    def retrieve(self):
        logging.info("Retrieving " + self.category + " data...")
        with open(self.current_file) as json_data:
            self.tweets = json.load(json_data)

    def initialize_category(self):
        if self.category not in self.categories.keys():
            self.docCount[self.category] = 0
            self.wordCount[self.category] = 0
            self.wordFrequencyCount[self.category] = {}
            self.categories[self.category] = True

    def build(self):
        start = time.time()
        self.retrieve()
        logging.info("Building " + self.category + " frequency tables...")
        self.initialize_category()
        print(self.categories)
        end = time.time()


if __name__ == "__main__":
    data = DictBuilder("info")
    data.build()
