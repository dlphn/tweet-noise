import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timeit

from config import FILEDIR

pd.set_option('display.width', None)

current_file = FILEDIR + "tweets_2019-01-23T14:26:33.598574.csv"
df = pd.read_csv(current_file, encoding="utf-8")

columns = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'posted_at', 'proportion_spamwords',
       'orthographe', 'nb_emoji', 'named_id', 'retweet_count', 'favorite_count', 'spam']
df_tweets = df[columns]

# df.info()
print(df_tweets.head())


# sns.distplot(df.reputation.dropna())


plt.figure(figsize=(10, 10))
for column_index, column in enumerate(df_tweets.columns):
    if column == 'spam' or column == 'posted_at':
        continue
    g = sns.FacetGrid(df_tweets, row='spam')
    g.map(sns.distplot, column)
plt.show()
