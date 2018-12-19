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
        print("Tweets labelling - possible inputs: bot/b, conv/c, pub/p, actu/a, reaction/r, other, skip/next/pass, "
              "stop/end, help.\nWhat is the type of the tweet displayed?\n")
        print("=============================================\n")
        for obj in self.db.tweets.find():
            try:
                obj["type"]
            except KeyError:
                classification = self.label(obj)
                if classification == "stop" or classification == "end" or classification == "x":
                    break
                elif classification in ["bot","conversation","publicité", "other spam", "actualité","reaction"]:
                    spam_value = True
                    if classification in ["actualité","reaction"] :
                        spam_value = False
                    self.db.tweets.update_one({"_id": obj.get("_id")}, {"$set": {"type": classification,"spam":spam_value}})
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
        valid = {"bot" : "bot", "b" : "bot", "conv" : "conversation", "c" : "conversation", "pub": "publicité",
                 "p": "publicité", "other" : "other spam", "actu" : "actualité","a" : "actualité", "reaction" : "reaction", "r" : "reaction"}
        other_actions = ["stop", "end", "x", "skip", "next", "pass"]
        while True:
            print("https://twitter.com/test/status/" + data["id_str"] + " : " + data["text"])
            choice = input("Type? [bot/conv/pub/actu/reaction or other]\n").lower()
            if choice in valid:
                return valid[choice]
            elif choice in other_actions:
                return choice
            elif choice == "help":
                print("Possible inputs: bot/b, conv/c, pub/p, actu/a, reaction/r, other, skip/next/pass, stop/end, help")
            else:
                print("Please respond with 'bot'/'b', 'conv'/'c', 'pub'/'p', 'actu'/'a', 'reaction'/'r' or 'other' .\n")


if __name__ == "__main__":
    dataLabel = DataLabelling()
    dataLabel.retrieve()
