# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:35 2018

@author: dshi, hbaud, vlefranc
"""

import logging
from datetime import datetime, timezone
from config import FILEDIR, FILEBREAK, MONGODB
from spamKeywords import keywords_blacklist
from whitelistKeywords import keywords_whitelist
from Emojilist import emojilist
from pymongo import MongoClient
import enchant
import unidecode
import time
import re
import spacy
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
        self.current_file = FILEDIR + "tweets_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".csv"
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
                f.write("\"id\",\"nb_follower\",\"nb_following\",\"verified\",\"reputation\",\"age\",\"nb_tweets\",\"posted_at\","
                        "\"proportion_spamwords\",\"proportion_whitewords\",\"orthographe\",\"nb_hashtag\","
                        "\"guillemets\",\"nb_emoji\",\"named_id\",\"spam\"\n")
            f.write(
                data["id_str"] +
                self.user_features(data) +
                self.information_contenu(data) +
                "," + ("\"true\"" if data["spam"] else "\"false\"") +
                "\n")
        self.line_count += 1

        if self.line_count > FILEBREAK:
            logging.info("Closing file {}".format(self.current_file))
            self.current_file = FILEDIR + "tweets_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".csv"
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
    def information_contenu(data):
        message = data['text']
        message_min = message.lower()
        message_min_sansaccent = unidecode.unidecode(message_min)
        liste_mot = re.sub("[,.#]",'', message_min).split()
        emojiList = emojilist
        emoji = 0
        spamwords = keywords_blacklist
        whitewords = keywords_whitelist
        spamword_count = 0
        whiteword_count = 0
        spell_dict = enchant.Dict('fr_FR')
        mot_bien_orth = 0

        for i in spamwords:
            if i in message_min_sansaccent:
                spamword_count += 1
        ratio_spamword = spamword_count/len(liste_mot)
        for i in whitewords:
            if i in message_min_sansaccent:
                whiteword_count += 1
        for mot in liste_mot:
            if spell_dict.check(mot):
                mot_bien_orth += 1
        ratio_orth = mot_bien_orth/len(liste_mot)
        nb_hashtag = message.count('#')
        guillements = message.count('\"')
        for j in emojiList:
            if j in message:
                emoji += 1

        #On transforme le message en format compatible avec nlp
        doc = nlp(message_min_sansaccent)

        result = "," + ("%.2f" % round(ratio_spamword, 2))
        result += "," + ("%.2f" % round(whiteword_count, 2))
        result += "," + ("%.2f" % round(ratio_orth, 2))
        result += "," + str(nb_hashtag)
        result += "," + str(guillemets)
        result += "," + str(emoji)
        result += "," + str(len(doc.ents))
        return result





if __name__ == "__main__":
    features = FeaturesBuilder()
    features.retrieve()
