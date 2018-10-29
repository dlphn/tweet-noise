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

        for obj in self.db.tweets.find():
            self.count += 1
            self.write(obj)
        logging.info("Total of {} elements retrieved".format(self.count))

    def write(self, data):
        with open(self.current_file, "a+", encoding='utf-8') as f:
            if self.line_count == 0:
                f.write("\"nb_follower\" \"nb_following\" \"verified\" \"reputation\" \"age\" \"nb_tweets\"proportion_spamwords\"orthographe\"nb_emoji\"RT\" \n")
            #recuperation des données informations utilisateurs
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
                information_contenu(data)+
                "\n")
        self.line_count += 1

        if self.line_count > FILEBREAK:
            logging.info("Closing file {}".format(self.current_file))
            self.current_file = FILEDIR + "tweets_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".txt"
            self.line_count = 0

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
        RT = False
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
            RT = True

        return (" "+ratio_spamword+" "+ratio_orth+" "+emoji+" "+RT)


if __name__ == "__main__":
    mongo = FeaturesBuilder()
    mongo.retrieve()
