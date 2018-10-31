# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:35 2018

@author: dshi, hbaud, vlefranc
"""

import logging
from datetime import datetime, timezone
from config import FILEDIR, FILEBREAK, MONGODB
from pymongo import MongoClient
import enchant
import unidecode

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
        logging.info("Retrieving data...")
        tweets = self.db.tweets.find({"spam": {"$exists": True}})
        logging.info("Building features file...")
        for obj in tweets:
            self.count += 1
            self.write(obj)
            if self.count % 100 == 0:
                logging.info("{} elements retrieved".format(self.count))
        logging.info("Total of {} elements retrieved".format(self.count))

    def write(self, data):
        with open(self.current_file, "a+", encoding='utf-8') as f:
            if self.line_count == 0:
                f.write("\"nb_follower\",\"nb_following\",\"verified\",\"reputation\",\"age\",\"nb_tweets\",\"proportion_spamwords\",\"orthographe\",\"nb_emoji\",\"RT\",\"spam\"\n")
            f.write(
                self.user_features(data) +
                self.information_contenu(data) +
                "," + ("\"true\"" if data["spam"] else "\"false\"") +
                "\n")
        self.line_count += 1

        if self.line_count > FILEBREAK:
            logging.info("Closing file {}".format(self.current_file))
            self.current_file = FILEDIR + "tweets_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".txt"
            self.line_count = 0

    @staticmethod
    def user_features(data):
        user = data["user"]
        created_at = datetime.strptime(user["created_at"], '%a %b %d %H:%M:%S %z %Y')
        now = datetime.now(timezone.utc)
        age = (now - created_at).days
        reputation = user["followers_count"]/(user["followers_count"] + user["friends_count"])
        result = str(user["followers_count"])
        result += "," + str(user["friends_count"])
        result += "," + ("\"true\"" if user["verified"] else "\"false\"")
        result += "," + ("%.2f" % round(reputation, 2))
        result += "," + str(age)
        result += "," + str(user["statuses_count"])
        return result

    @staticmethod
    def information_contenu(data):
        message = data['text']
        message_min = message.lower()
        message_min_sansaccent = unidecode.unidecode(message_min)
        liste_mot = message.split()
        emojiList = [":)", ":(", ":P", ":-*", "XD"]
        emoji = 0
        spamwords = ["sexe", "hot", "sexy", "chaud", "viagra", "cure", "sexuel", "hormone", "perdre du poids", "regime",
             "rides", "agrandissez votre penis", "performance", "celibataires","casino", "blackjack", "poker", "jetons", "roulette",
             "opportunite", "sans frais", "cb", "carte bancaire", "paypal", "interet", "taux d'interet",
             "carte de credit", "bonne affaire", "pas de frais", "facture", "cartes acceptees", "cheque",
             "meilleur prix", "prix les plus bas", "promotion speciale", "pour seulement", "gratuit", "100% gratuit",
             "installation gratuite", "acces gratuit", "echantillon gratuit", "essai gratuit", "cadeau",
             "réduction", "free", "duree limite", "annulation à tout moment", "inscrivez-vous gratuitement aujourd’hui",
             "nouveaux clients uniquement", "obtenez-le maintenant", "agissez des maintenant",
             "commandez aujourd’hui", "quantites limitees", "temps limite", "cliquez ici", "cliquez",
             "profitez aujourd'hui", "postulez", "devenez membre", "appelez", "annulez à tout moment", "certifie",
             "doublez", "gagnez", "offre exclusive", "aucun cout", "pas de frais", "aucun engagement",
             "sans engagement", "commandez maintenant","aucun risque", "bon plan", "felicitations", "incroyable", "spam", "escroquerie", "unique"]
        spamword_count = 0
        rt = False
        spell_dict = enchant.Dict('fr_FR')
        mot_bien_orth = 0

        for i in spamwords:
            if i in message_min_sansaccent:
                spamword_count += 1
        ratio_spamword = spamword_count/len(liste_mot)
        for mot in liste_mot:
            if spell_dict.check(mot):
                mot_bien_orth += 1
        ratio_orth = mot_bien_orth/len(liste_mot)
        for j in emojiList:
            if j in message:
                emoji += 1
        if 'RT @' in message:
            rt = True

        result = "," + ("%.2f" % round(ratio_spamword, 2))
        result += "," + ("%.2f" % round(ratio_orth, 2))
        result += "," + str(emoji)
        result += "," + ("\"true\"" if rt else "\"false\"")
        return result


if __name__ == "__main__":
    mongo = FeaturesBuilder()
    mongo.retrieve()
