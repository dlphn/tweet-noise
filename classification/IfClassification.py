import logging
from datetime import datetime, timezone
from config import FILEDIR, FILEBREAK
import re


class IfClassification:

    def __init__(self):

        self.current_file = '/Users/delphineshi/Downloads/temp/tweets_2018-10-31T10:29:59.502392.txt'
        self.nf_line_count =0
        self.new_file = FILEDIR + "tweets_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".txt"


    def open_write(self):
        with open(self.current_file, "r", encoding='utf-8') as f:
            data = f.readlines()
            #print (data)
            for i in range (1,len(data)) :
                ligne = data [i]
                data_split = ligne.split(',')
                print (data_split)
                self.id = re.sub("'",'', data_split[0])
                self.nb_follower = int(re.sub("'",'', data_split[1]))
                self.nb_following = int(re.sub("'",'', data_split[2]))
                self.verified = re.sub("'",'', data_split[3])
                self.reputation = float(re.sub("'",'', data_split[4]))
                self.age = re.sub("'",'', data_split[5])
                self.nb_tweets = int(re.sub("'",'', data_split[6]))
                self.time = re.sub("'",'', data_split[7])
                self.proportion_spamwords= float(re.sub("'",'', data_split[8]))
                self.orthographe= float(re.sub("'",'', data_split[9]))
                self.nb_emoji= int(re.sub("'",'', data_split[10]))
                self.RT = re.sub("[\"']",'', data_split[11])
                self.spam = re.sub("[\"'\\n]", '', data_split[12])
                with open(self.new_file, "a+", encoding='utf-8') as nf:
                    nf.write(ligne + self.classification() + "\n")
                self.nf_line_count += 1

                if self.nf_line_count > FILEBREAK:
                    logging.info("Closing file {}".format(self.current_file))
                    self.current_file = FILEDIR + "tweets_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".txt"
                    self.nf_line_count = 0

    def classification(self):
        potentialspam = "true"
        verdict =0
        if self.nb_emoji == 0 :
            if self.proportion_spamwords < 0.1 :
                if self.orthographe > 0.5 :
                    if self.RT == "true":
                        potentialspam = "false"
        if potentialspam == self.spam:
            verdict = 1
        print(verdict)
        return potentialspam + ","+ str(verdict)


fichier = IfClassification()
fichier.open_write()
