# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:35 2018

@author: dshi, hbaud, vlefranc
"""

import logging
from datetime import datetime, timezone
import time
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import sys
sys.path.append('..')
from config import FILEDIR, FILEBREAK, MONGODB
pd.options.display.max_columns = 15


logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


class FetchTweet:
    """
    Retrieve data from the MongoDB database.

    """
    def __init__(self):

        self.do_continue = True
        self.count = 0
        self.df = pd.DataFrame(columns=['id','retweeted','favorite','spam'])
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
            self.add_to_df(obj)
        end = time.time()
        logging.info("Total of {0} elements retrieved in {1} seconds".format(self.count, end - start))
        return self.df

    def add_to_df(self, data):
        self.df.loc[self.count] = [data['id'],data['retweet_count'],data['favorite_count'],data['spam']]

def show(df):
    i=1
    print(df.dtypes)
    print(df.columns)
    for column_index, column in enumerate(df.columns):
        if column == 'spam':
            continue
        if column == 'id':
            continue
        plt.subplot(1, 2, i)
        sns.violinplot(x='spam', y=column, data=df[df[column]<100])
        i +=1
    plt.show()



if __name__ == "__main__":
    tweets = FetchTweet()
    df = tweets.retrieve()
    df['retweeted'] = pd.to_numeric(df['retweeted'])
    df['favorite'] = pd.to_numeric(df['favorite'])
    print(df.groupby('spam').describe())
    show(df)