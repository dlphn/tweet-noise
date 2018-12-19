# import numpy as np
import pandas as pd
import nltk
import re
# import os
# import codecsfrom sklearn import feature_extraction
# import mpld3

from nltk.stem.snowball import SnowballStemmer
import arrayBuilder

# nltk.download('stopwords')

# liste des tweets
titles = ['The Godfather', 'The Shawshank Redemption', "Schindler's List", 'Raging Bull', 'Casablanca', "One Flew Over the Cuckoo's Nest", 'Gone with the Wind', 'Citizen Kane', 'The Wizard of Oz', 'Titanic']
synopses = ["On the day of his only daughter's wedding, Vito Corleone hears requests in his role as the Godfather, the Don of a New York crime family. Vito's youngest son, Michael, in a Marine Corps uniform, introduces his girlfriend, Kay Adams, to his family at the sprawling reception.",
            "In 1947, banker Andy Dufresne is convicted of murdering his wife and her lover and sentenced to two consecutive life sentences at the fictional Shawshank State Penitentiary in the state of Maine. Andy befriends contraband smuggler Ellis 'Red' Redding, an inmate serving a life sentence. Red procures a rock hammer and later a large poster of Rita Hayworth for Andy. Working in the prison laundry, Andy is regularly assaulted by the 'bull queer' gang 'the Sisters' and their leader, Bogs.  In 1947, banker Andy Dufresne is convicted of murdering his wife and her lover and sentenced to two consecutive life sentences at the fictional Shawshank State Penitentiary in the state of Maine.",
            "In a brief scene in 1964, an aging, overweight Italian American, Jake LaMotta (Robert De Niro), practices a comedy routine. The rest of the film then occurs in flashback. In 1941, LaMotta is in a major boxing match against Jimmy Reeves, where he received his first loss. Jake's brother Joey LaMotta (Joe Pesci) discusses a potential shot for the middleweight title with one of his Mafia connections, Salvy Batts (Frank Vincent)."]

mongo = arrayBuilder.ArrayBuilder()
tweets = mongo.retrieve()


stopwords = nltk.corpus.stopwords.words('english')
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

