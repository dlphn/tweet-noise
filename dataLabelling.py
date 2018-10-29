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
    Retrieve data from the MongoDB database and let the user label the tweets.

    """
    def __init__(self):

        self.do_continue = True
        self.count = 0
        # connect to MongoDB
        client = MongoClient("mongodb+srv://" + MONGODB["USER"] + ":" + MONGODB["PASSWORD"] + "@" + MONGODB["HOST"] + "/" + MONGODB["DATABASE"] + "?retryWrites=true")
        self.db = client[MONGODB["DATABASE"]]

    def retrieve(self):
        """
        Retrieve tweets from mongo database and save user's label for the tweets.
        """
        print("=============================================\n")
        print("Tweets labelling - possible inputs: y/yes, n/no, skip/next/pass, stop/end, help.\nIs the tweet displayed considered as spam?\n")
        print("=============================================\n")
        for obj in self.db.tweets.find():
            try:
                obj["spam"]
            except KeyError:
                classification = self.label(obj)
                if classification == "stop" or classification == "end" or classification == "x":
                    break
                elif isinstance(classification, bool):
                    self.db.tweets.update_one({"_id": obj.get("_id")}, {"$set": {"spam": classification}})
                    self.count += 1
        logging.info("Total of {} elements labelled".format(self.count))

    @staticmethod
    def label(data):
        """
        Display the tweet and ask if it is considered as spam or not and return the answer.
        :param data: the tweet to label
        :return: the user's answer (True, False, or an action to skip or stop)
        """
        # display tweet and allow input from user true/false
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        other_actions = ["stop", "end", "x", "skip", "next", "pass"]
        while True:
            print("https://twitter.com/test/status/" + data["id_str"] + " : " + data["text"])
            choice = input("Spam? [y/n]\n").lower()
            if choice in valid:
                return valid[choice]
            elif choice in other_actions:
                return choice
            elif choice == "help":
                print("Possible inputs: y/yes, n/no, skip/next/pass, stop/end, help")
            else:
                print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


if __name__ == "__main__":
    dataLabel = DataLabelling()
    dataLabel.retrieve()
