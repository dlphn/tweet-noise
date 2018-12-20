# import numpy as np
import pandas as pd
import nltk
import re
import os
# import codecs
from sklearn import feature_extraction
# import mpld3

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import arrayBuilder
from featuresBuilder.stopKeywords import keywords_stoplist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS



# nltk.download('stopwords')

# liste des tweets
titles = ['The Godfather', 'The Shawshank Redemption', "Schindler's List", 'Raging Bull', 'Casablanca', "One Flew Over the Cuckoo's Nest", 'Gone with the Wind', 'Citizen Kane', 'The Wizard of Oz', 'Titanic']
synopses = ["On the day of his only daughter's wedding, Vito Corleone hears requests in his role as the Godfather, the Don of a New York crime family. Vito's youngest son, Michael, in a Marine Corps uniform, introduces his girlfriend, Kay Adams, to his family at the sprawling reception.",
            "In 1947, banker Andy Dufresne is convicted of murdering his wife and her lover and sentenced to two consecutive life sentences at the fictional Shawshank State Penitentiary in the state of Maine. Andy befriends contraband smuggler Ellis 'Red' Redding, an inmate serving a life sentence. Red procures a rock hammer and later a large poster of Rita Hayworth for Andy. Working in the prison laundry, Andy is regularly assaulted by the 'bull queer' gang 'the Sisters' and their leader, Bogs.  In 1947, banker Andy Dufresne is convicted of murdering his wife and her lover and sentenced to two consecutive life sentences at the fictional Shawshank State Penitentiary in the state of Maine.",
            "In a brief scene in 1964, an aging, overweight Italian American, Jake LaMotta (Robert De Niro), practices a comedy routine. The rest of the film then occurs in flashback. In 1941, LaMotta is in a major boxing match against Jimmy Reeves, where he received his first loss. Jake's brother Joey LaMotta (Joe Pesci) discusses a potential shot for the middleweight title with one of his Mafia connections, Salvy Batts (Frank Vincent)."]

mongo = arrayBuilder.ArrayBuilder()
tweets = mongo.retrieve()


stopwords = nltk.corpus.stopwords.words('french')
stemmer = SnowballStemmer("french")


def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


totalvocab_stemmed = []
totalvocab_tokenized = []

for i in tweets:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
print('There are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

print(vocab_frame.head())

# vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(
    # max_df=0.8, max_features=200000,
    # min_df=0.2,
    stop_words=stopwords,
    # use_idf=True, tokenizer=tokenize_and_stem,
    ngram_range=(1, 3)
)

tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)
print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()
# print(terms)

#cosine similarity
dist = 1 - cosine_similarity(tfidf_matrix)
# print(dist)

#K-means clustering
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# joblib.dump(km, 'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

data = {'tweets': tweets, 'clusters': clusters}

frame = pd.DataFrame(data, index=[clusters], columns=['tweets','cluster'])
frame['cluster'].value_counts()  # nb of tweets per cluster

MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)

xs, ys = pos[:, 0], pos[:, 1]


cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
cluster_names = {0: 'Family, home, war',
                 1: 'Police, killed, murders',
                 2: 'Father, New York, brothers',
                 3: 'Dance, singing, love',
                 4: 'Killed, soldiers, captain'}

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=tweets))
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(17,9))
ax.margins(0.05)

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['tweet'], size=8)

plt.show()  # show the plot
