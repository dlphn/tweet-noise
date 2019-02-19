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
        print("Tweets labelling - possible inputs: actu/a/1, reaction/r/2, conv/c/5, pub/p/6, "
              "bot/b/7, other/o/8, skip/next/pass, stop/end, help.\nWhat is the type of the tweet displayed?\n")
        print("=============================================\n")
        for obj in self.db.tweets.find():
            try:
                obj["type"]
            except KeyError:
                classification = self.label(obj)
                if classification == "stop" or classification == "end" or classification == "x":
                    break
                elif classification in ["actualité","reaction","conversation","publicité","bot","other spam"]:
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
        valid =  {"actu" : "actualité","a" : "actualité","1" : "actualité", "reaction" : "reaction", "r" : "reaction",
                  "2": "reaction","conv": "conversation", "c": "conversation", "5": "conversation",
                  "pub": "publicité","p": "publicité", "6": "publicité", "bot" : "bot", "b": "bot", "7": "bot",
                  "other" : "other spam","o" : "other spam","8" : "other spam"}

        other_actions = ["stop", "end", "x", "skip", "next", "pass"]
        while True:
            print("https://twitter.com/test/status/" + data["id_str"] + " : " + data["text"])
            choice = input("Type? [bot/conv/pub/actu/reaction or other]\n").lower()
            if choice in valid:
                return valid[choice]
            elif choice in other_actions:
                return choice
            elif choice == "help":
                print("Possible inputs: actu/a/1, reaction/r/2, conv/c/5, pub/p/6, bot/b/7,"
                      " other/o/8, skip/next/pass, stop/end, help")
            else:
                print("Please respond with 'actu'/'a'/'1', 'reaction'/'r'/'2',"
                      "'conv'/'c'/'5','pub'/'p'/'6','bot'/'b'/'7' or 'other'/'o'/'8' .\n")

    def correct(self):
        """
        correct tweet previously labelled.
        """
        for obj in self.db.tweets.find():
            if obj["type"] == "actualité par personnalité" :
                self.db.tweets.update_one({"_id": obj.get("_id")},
                                          {"$set": {"type": "actualité"}})
                self.count += 1
            if obj["type"] == "spam par personnalité" :
                self.db.tweets.update_one({"_id": obj.get("_id")},
                                          {"$set": {"type": "conversation"}})
                self.count += 1
        logging.info("Total of {} elements with label changed".format(self.count))



if __name__ == "__main__":
    dataLabel = DataLabelling()
    dataLabel.retrieve()
