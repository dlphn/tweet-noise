import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timeit

from config import FILEDIR

pd.set_option('display.width', None)

current_file = FILEDIR + "tweets_2018-10-31T14:25:32.735253.csv"
df = pd.read_csv(current_file, encoding="utf-8")

columns = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'time', 'proportion_spamwords',
       'orthographe', 'nb_emoji', 'RT', 'spam']
df_tweets = df[columns]

# df.info()
print(df_tweets.head())


# sns.distplot(df.reputation.dropna())

g = sns.FacetGrid(df_tweets, row='spam')
g.map(sns.distplot, "age")


plt.show()