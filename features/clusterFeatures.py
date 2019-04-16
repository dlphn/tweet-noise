# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:35 2019

@author: dshi, hbaud, vlefranc
"""

import pandas as pd
from pymongo import MongoClient
import sys
from Medias import MEDIAS, listemedias

sys.path.append('..')
from config import FILEDIR, FILEBREAK, MONGODB

pd.set_option('max_columns', 10)


class Classification:

    def __init__(self, filename):
        self.filepath = FILEDIR + filename + '.csv'
        # connect to MongoDB
        client = MongoClient(
            "mongodb+srv://" + MONGODB["USER"] + ":" + MONGODB["PASSWORD"] + "@" + MONGODB["HOST"] + "/" + MONGODB[
                "DATABASE"] + "?retryWrites=true")
        self.db = client[MONGODB["DATABASE"]]
        self.count = 0

    def create_dataframe(self):
        df = pd.read_csv(self.filepath)
        df.label = df.label.apply(self.categorize_label)
        df = df[df['pred'] != -1]
        df2 = df.groupby('pred').agg({'id': lambda x: list(x),'label' : sum})
        #print(df2)
        df_cluster = pd.DataFrame(columns=['nb_of_cluster','nb_tweets', 'nb_tweets_actu','hashtag','url','media','contains_media','list_media'])
        for index, row in df2.iterrows():
            # print(row)
            if len(row['id']) > 2:
                self.count +=1
                self.get_features(row['id'], index)
                df_cluster.loc[self.count] = [index, len(row['id']), row['label'],self.medias, self.hashtags, self.urls, self.actu_source, self.sources]
        df_cluster.to_csv(FILEDIR+'cluster_data.csv')

    # print(self.get_features(row[1]),index)

    def get_features(self, cluster, cluster_number):
        self.medias = dict()
        self.hashtags = dict()
        self.urls = dict()
        self.sources = []
        for tweet_id in cluster:
            tweets = self.db.tweets.find({"id_str": tweet_id})
            # print(tweets)
            for tweet in tweets:
                # print(tweet)
                self.get_medias(tweet, self.medias)
                self.get_hashtags(tweet, self.hashtags)
                self.get_urls(tweet, self.urls)
                self.get_mediasource(tweet, self.sources)
        # On cree une variable booleene qui nous dit si il y a au moins un media dans le cluster
        if len(self.sources) > 0:
            self.actu_source = True
        else:
            self.actu_source = False

    def categorize_label(self,x):
        if x == 'spam' :
            return 0
        else :
            return 1


    def get_medias(self, tweet, medias):
        if 'extended_tweet' in tweet.keys():
            if 'media' in tweet['extended_tweet']['entities'].keys():
                for i in range(len(tweet['extended_tweet']['entities']['media'])):
                    if tweet['extended_tweet']['entities']['media'][i]['media_url'] in medias.keys():
                        medias[tweet['extended_tweet']['entities']['media'][i]['media_url']] += 1
                    else:
                        medias[tweet['extended_tweet']['entities']['media'][i]['media_url']] = 1
        elif 'entities' in tweet.keys():
            if 'media' in tweet['entities'].keys():
                for i in range(len(tweet['entities']['media'])):
                    if tweet['entities']['media'][i]['media_url'] in medias.keys():
                        medias[tweet['entities']['media'][i]['media_url']] += 1
                    else:
                        medias[tweet['entities']['media'][i]['media_url']] = 1

    def get_hashtags(self, tweet, hashtags):
        if 'extended_tweet' in tweet.keys():
            for i in range(len(tweet['extended_tweet']['entities']['hashtags'])):
                if tweet['extended_tweet']['entities']['hashtags'][i]['text'] in hashtags.keys():
                    hashtags[tweet['extended_tweet']['entities']['hashtags'][i]['text']] += 1
                else:
                    hashtags[tweet['extended_tweet']['entities']['hashtags'][i]['text']] = 1
        elif 'entities' in tweet.keys():
            for i in range(len(tweet['entities']['hashtags'])):
                if tweet['entities']['hashtags'][i]['text'] in hashtags.keys():
                    hashtags[tweet['entities']['hashtags'][i]['text']] += 1
                else:
                    hashtags[tweet['entities']['hashtags'][i]['text']] = 1

    def get_urls(self, tweet, urls):
        if 'extended_tweet' in tweet.keys():
            for i in range(len(tweet['extended_tweet']['entities']['urls'])):
                if tweet['extended_tweet']['entities']['urls'][i]['url'] in urls.keys():
                    urls[tweet['extended_tweet']['entities']['urls'][i]['url']] += 1
                else:
                    urls[tweet['extended_tweet']['entities']['urls'][i]['url']] = 1
        elif 'entities' in tweet.keys():
            for i in range(len(tweet['entities']['urls'])):
                if tweet['entities']['urls'][i]['url'] in urls.keys():
                    urls[tweet['entities']['urls'][i]['url']] += 1
                else:
                    urls[tweet['entities']['urls'][i]['url']] = 1

    def get_mediasource(self, tweet, sources):
        # print(tweet['user']['screen_name'])
        if tweet['user']['screen_name'] in listemedias:
            sources.append(tweet['user']['screen_name'])


if __name__ == "__main__":
    classif = Classification('clustering_2019-03-06_0.7_100000_100_base_fr_2')
    classif.create_dataframe()
