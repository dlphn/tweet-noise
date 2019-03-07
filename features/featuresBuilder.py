# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:35 2018

@author: dshi, hbaud, vlefranc
"""

import logging
from datetime import datetime, timezone
from config import FILEDIR, FILEBREAK, MONGODB
from features import Keywords
from pymongo import MongoClient
import enchant
import unidecode
import time
import re
import fr_core_news_md
nlp = fr_core_news_md.load()

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


class FeaturesBuilder:
    """
    Retrieve data from the MongoDB database.

    """
    def __init__(self):

        self.do_continue = True
        self.count = 0
        self.line_count = 0
        self.file_count = 1
        self.date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        self.current_file = FILEDIR + "tweets_" + self.date + ".csv"
        # connect to MongoDB
        client = MongoClient("mongodb+srv://" + MONGODB["USER"] + ":" + MONGODB["PASSWORD"] + "@" + MONGODB["HOST"] + "/" + MONGODB["DATABASE"] + "?retryWrites=true")
        self.db = client[MONGODB["DATABASE"]]

    def retrieve(self):
        start = time.time()
        logging.info("Retrieving data...")
        tweets = self.db.tweets.find({"spam": {"$exists": True}})
        logging.info("Building features file...")
        for obj in tweets:
            self.count += 1
            self.write(obj)
            if self.count % 100 == 0:
                logging.info("{} elements retrieved".format(self.count))
        end = time.time()
        logging.info("Total of {0} elements retrieved in {1} seconds".format(self.count, end - start))

    def write(self, data):
        with open(self.current_file, "a+", encoding='utf-8') as f:
            if self.line_count == 0:
                f.write('"id","nb_follower","nb_following","verified","reputation","age","nb_tweets","posted_at",'
                        '"text","length","proportion_spamwords","proportion_whitewords","orthographe","nb_hashtag",'
                        '"nb_urls","guillemets","nb_emoji","named_id","retweet_count","favorite_count",'
                        '"type","spam"\n')
            f.write(
                data["id_str"] +
                self.user_features(data) +
                self.information_content(data) +
                "," + data["type"] +
                "," + ('"true"' if data["spam"] else '"false"') +
                "\n")
        self.line_count += 1

        if self.line_count > FILEBREAK:
            logging.info("Closing file {}".format(self.current_file))
            self.file_count += 1
            self.current_file = FILEDIR + "tweets_" + self.date + "_" + self.file_count + ".csv"
            self.line_count = 0

    @staticmethod
    def user_features(data):
        user = data["user"]
        created_at = datetime.strptime(user["created_at"], '%a %b %d %H:%M:%S %z %Y')
        now = datetime.now(timezone.utc)
        age = (now - created_at).days
        if user["followers_count"] + user["friends_count"] > 0:
            reputation = user["followers_count"]/(user["followers_count"] + user["friends_count"])
        else:
            reputation = 0
        ts = int(data["timestamp_ms"])/1000
        posted_at = datetime.utcfromtimestamp(ts).strftime('%H:%M:%S')
        result = "," + str(user["followers_count"])
        result += "," + str(user["friends_count"])
        result += "," + ("1" if user["verified"] else "0")
        result += "," + ("%.2f" % round(reputation, 2))
        result += "," + str(age)
        result += "," + str(user["statuses_count"])
        result += "," + "\"" + posted_at + "\""
        return result

    @staticmethod
    def information_content(data):
        message = data['text'].lower()
        doc = nlp(message)
        # On récupère une liste de tous les mots qui composent les tweet et on les compare au dictionnaire pour voir s'ils sont bien orthographies/existent
        liste = [str(token) for token in doc]
        spell_dict = enchant.Dict('fr_FR')
        mot_bien_orth = 0
        for mot in liste:
            if spell_dict.check(mot):
                mot_bien_orth += 1
        ratio_orth = mot_bien_orth / len(liste)
        #On compte le nombre de spamwords
        spamword_count = 0
        for i in spamwords:
            if i in message_min_sansaccent:
                spamword_count += 1
        ratio_spamword = spamword_count/len(liste_mot)
        #On compte le nombre de whitewords
        whiteword_count =0
        for i in whitewords:
            if i in message_min_sansaccent:
                whiteword_count += 1

        # On compte le nombre d'emoji dans le tweet
        emojiList = emojilist
        emoji = 0
        for j in emojiList:
            if j in message:
                emoji += 1

        result = ",\"" + str(data['text']) + "\""
        result += "," + str(len(data['text']))
        result += "," + str(ratio_spamword)
        result += "," + str(whiteword_count)
        result += "," + ("%.2f" % round(ratio_orth, 2))
        result += "," + str(len(data['entities']['urls']))
        # On compte le nb de hashtag
        result += "," + str(message.count('#'))
        result += "," + str(emoji)
        # On récupère le nombre d'entites nommees
        result += "," + str(len(doc.ents))
        result += "," + str(data['retweet_count'])
        result += "," + str(data['favorite_count'])
        return result



if __name__ == "__main__":
    features = FeaturesBuilder()
    features.retrieve()
