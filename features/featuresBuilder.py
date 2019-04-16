# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:35 2018

@author: dshi, hbaud, vlefranc
"""

import logging
from datetime import datetime, timezone
from features.Keywords import keywords_blacklist, keywords_whitelist_freq, emojilist
from pymongo import MongoClient
import enchant
import time
import fr_core_news_md
import sys
from config import FILEDIR, FILEBREAK, MONGODB

sys.path.append('..')

nlp = fr_core_news_md.load()

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


def user_features(data, as_array=False):
    """
    Build user features for a specific tweet

    :param data: tweet
    :param as_array: (bool) return features as array or string
    :return: user features
    """
    user = data["user"]
    created_at = datetime.strptime(user["created_at"], '%a %b %d %H:%M:%S %z %Y')
    now = datetime.now(timezone.utc)
    age = (now - created_at).days
    if user["followers_count"] + user["friends_count"] > 0:
        reputation = user["followers_count"] / (user["followers_count"] + user["friends_count"])
    else:
        reputation = 0
    ts = int(data["timestamp_ms"]) / 1000
    posted_at = datetime.utcfromtimestamp(ts).strftime('%H:%M:%S')
    if as_array:
        result = [
            user["followers_count"],
            user["friends_count"],
            1 if user["verified"] else 0,
            round(reputation, 2),
            age,
            user["statuses_count"],
            posted_at,
        ]
    else:
        result = "," + str(user["followers_count"])
        result += "," + str(user["friends_count"])
        result += "," + ("1" if user["verified"] else "0")
        result += "," + ("%.2f" % round(reputation, 2))
        result += "," + str(age)
        result += "," + str(user["statuses_count"])
        result += "," + "\"" + posted_at + "\""
    return result


def information_content(data, as_array=False):
    """
    Build content features for a specific tweet

    :param data: tweet
    :param as_array: (bool) return features as array or string
    :return: content features
    """
    message = data['text'].lower()
    doc = nlp(message)
    # On récupère une liste de tous les mots qui composent les tweets et on les compare au dictionnaire pour voir
    # s'ils sont bien orthographiés/existent
    tweet_words = [str(token) for token in doc]
    spell_dict = enchant.Dict('fr_FR')
    mot_bien_orth = 0
    for word in tweet_words:
        if spell_dict.check(word):
            mot_bien_orth += 1
    ratio_orth = mot_bien_orth / len(tweet_words)

    # On compte le nombre de spamwords
    spamword_count = 0
    for word in tweet_words:
        if word in keywords_blacklist:
            spamword_count += 1
    ratio_spamword = spamword_count / len(tweet_words)

    # On compte le nombre de whitewords
    whiteword_count = 0
    for word in tweet_words:
        if word in keywords_whitelist_freq:
            whiteword_count += 1

    # On compte le nombre d'emoji dans le tweet
    nb_emoji = 0
    for emoji in emojilist:
        if emoji in message:
            nb_emoji += 1

    if as_array:
        result = [
            len(data['text']),
            ratio_spamword,
            whiteword_count,
            round(ratio_orth, 2),
            len(data['entities']['urls']),
            message.count('#'),  # nb of hashtags
            nb_emoji,
            len(doc.ents)  # nb of named entities
        ]
    else:
        # result = ",\"" + str(data['text']) + "\""
        result = "," + str(len(data['text']))
        result += "," + str(ratio_spamword)
        result += "," + str(whiteword_count)
        result += "," + ("%.2f" % round(ratio_orth, 2))
        result += "," + str(len(data['entities']['urls']))
        result += "," + str(message.count('#'))  # nb of hashtags
        result += "," + str(nb_emoji)
        result += "," + str(len(doc.ents))  # nb of named entities
    return result


class FeaturesBuilder:
    """
    Retrieve data from the MongoDB database and build csv file with tweets features.

    """
    def __init__(self):

        self.do_continue = True
        self.count = 0
        self.line_count = 0
        self.file_count = 1
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.current_file = FILEDIR + "tweets_features_" + self.date + ".csv"
        # connect to MongoDB
        client = MongoClient("mongodb+srv://" + MONGODB["USER"] + ":" + MONGODB["PASSWORD"] + "@" + MONGODB["HOST"] + "/" + MONGODB["DATABASE"] + "?retryWrites=true")
        self.db = client[MONGODB["DATABASE"]]

    def retrieve(self):
        start = time.time()
        logging.info("Retrieving data...")
        tweets = self.db.tweets.find({"spam": {"$exists": True}})  # only fetch labelled tweets
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
                        '"length","proportion_spamwords","proportion_whitewords","orthographe","nb_hashtag",'
                        '"nb_urls","nb_emoji","named_id",'
                        '"type","spam"\n')
            f.write(
                data["id_str"] +
                user_features(data) +
                information_content(data) +
                "," + (data["type"] if type in data else "") +
                "," + ('"true"' if data["spam"] else '"false"') +
                "\n")
        self.line_count += 1

        if self.line_count > FILEBREAK:
            logging.info("Closing file {}".format(self.current_file))
            self.file_count += 1
            self.current_file = FILEDIR + "tweets_features_" + self.date + "_" + str(self.file_count) + ".csv"
            self.line_count = 0


if __name__ == "__main__":
    features = FeaturesBuilder()
    features.retrieve()
