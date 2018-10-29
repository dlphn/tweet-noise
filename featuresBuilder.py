# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:35 2018

@author: dshi, hbaud, vlefranc
"""

import logging
from datetime import datetime, timezone
from config import FILEDIR, FILEBREAK, MONGODB
from pymongo import MongoClient

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


class FeaturesBuilder:
    """
    Retrieve data from the MongoDB database.

    """
    def __init__(self):

        self.do_continue = True
        self.count = 0
        self.line_count = 0
        self.current_file = FILEDIR + "tweets_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".txt"
        # connect to MongoDB
        client = MongoClient("mongodb+srv://" + MONGODB["USER"] + ":" + MONGODB["PASSWORD"] + "@" + MONGODB["HOST"] + "/" + MONGODB["DATABASE"] + "?retryWrites=true")
        self.db = client[MONGODB["DATABASE"]]

    def retrieve(self):

        for obj in self.db.tweets.find():
            self.count += 1
            self.write(obj)
        logging.info("Total of {} elements retrieved".format(self.count))

    def write(self, data):
        with open(self.current_file, "a+", encoding='utf-8') as f:
            if self.line_count == 0:
                f.write("\"nb_follower\" \"nb_following\" \"verified\" \"reputation\" \"age\" \"nb_tweets\" \n")
            user = data["user"]
            created_at = datetime.strptime(user["created_at"], '%a %b %d %H:%M:%S %z %Y')
            now = datetime.now(timezone.utc)
            age = (now - created_at).days
            reputation = user["followers_count"]/(user["followers_count"] + user["friends_count"])
            f.write(
                str(user["followers_count"]) + " " +
                str(user["friends_count"]) + " " +
                ("\"true\"" if user["verified"] else "\"false\"") + " " +
                ("%.2f" % round(reputation, 2)) + " " +
                str(age) + " " +
                str(user["statuses_count"]) +
                "\n")
        self.line_count += 1

        if self.line_count > FILEBREAK:
            logging.info("Closing file {}".format(self.current_file))
            self.current_file = FILEDIR + "tweets_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".txt"
            self.line_count = 0


if __name__ == "__main__":
    mongo = FeaturesBuilder()
    mongo.retrieve()
