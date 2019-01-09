import re
import collections

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import arrayBuilder
from featuresBuilder import stopKeywords


class TextClustering:

    def __init__(self):
        self.vectorizer = None
        self.km_model = None

    @staticmethod
    def process_text(text):
        """ Tokenize text removing punctuation """
        # stop = set(stopwords.words('french'))
        stop = stopKeywords.keywords_stoplist
        text = re.sub(r'(?:\@|https?\://)\S+', '', text)  # remove links and @username
        text = re.sub(r'[^\w\s]', ' ', text)  # remove non-alphanumeric characters
        text = ' '.join([token for token in word_tokenize(text.lower()) if token not in stop])  # remove stopwords
        return text

    @staticmethod
    def tokenize(text, stem=True):
        """ Tokenize text and stem words removing punctuation """
        text = re.sub(r'(?:\@|https?\://)\S+', '', text)  # remove links and @username
        text = re.sub(r'[^\w\s]', ' ', text)  # remove non-alphanumeric characters
        tokens = word_tokenize(text)

        if stem:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens]

        return tokens

    def cluster_texts(self, texts, k=3):
        """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
        self.vectorizer = TfidfVectorizer(
            # tokenizer=self.tokenize,
            # stop_words=stopwords.words('english'),
            # stop_words=stopwords.words('french'),
            # max_df=0.8,
            # min_df=0.1,
            # lowercase=True
        )
        texts = [self.process_text(text) for text in texts]
        print(texts)

        tfidf_model = self.vectorizer.fit_transform(texts)
        self.km_model = KMeans(n_clusters=k, init='k-means++', n_init=10)
        self.km_model.fit(tfidf_model)

        clustering = collections.defaultdict(list)

        print("Top terms per cluster:")
        order_centroids = self.km_model.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names()
        for i in range(k):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
            print

        for idx, label in enumerate(self.km_model.labels_):
            clustering[label].append(idx)

        return clustering

    def predict(self, text):
        y = self.vectorizer.transform([text])
        prediction = self.km_model.predict(y)
        print(prediction)


def get_tweets():
    data = arrayBuilder.ArrayBuilder()
    return data.retrieve_text_and_labels()


if __name__ == "__main__":
    # Cat/Google test
    # articles = ["This little kitty came to play when I was eating at a restaurant.",
    #             "Merley has the best squooshy kitten belly.",
    #             "Google Translate app is incredible.",
    #             "If you open 100 tab in google you get a smiley face.",
    #             "Best cat photo I've ever taken.",
    #             "Climbing ninja cat.",
    #             "Impressed with google map feedback.",
    #             "Key promoter extension for Google Chrome."]
    # model = TextClustering()
    # clusters = model.cluster_texts(articles, 2)
    # print(dict(clusters))
    # model.predict("chrome browser to open.")
    # model.predict("My cat is hungry.")

    # French test
    articles = ["J'aime les frites https://t.co/JkpvrfmC6F oui",
                "Je suis d'accord, j'aime pas la pluie... #@tropbeaulavie !",
                "Les frites je les adore",
                "La taxe d'habitation ne sera pas augmentée annonce le gouvernement",
                "Cool pas d'augmentation de la taxe d'habitation !!!",
                "Il fait trop moche aujourd'hui", ]
    # model = TextClustering()
    # clusters = model.cluster_texts(articles, 3)
    # print(dict(clusters))
    # model.predict("Des manifestants dans la rue pour exprimer leur colère face à l'augmentation de la taxe d'habitation.")
    # model.predict("Lol je sors sans parapluie trankil et là il pleut.")

    # Tweets
    articles, labels = get_tweets()
    print(articles)
    model = TextClustering()
    clusters = model.cluster_texts(articles, 20)
    result = dict(clusters)
    print
    print(result)
    print
    print('Cluster', 'Count', 'Labels')
    for key in result.keys():
        print(key, len(result[key]), collections.Counter([labels[tweet_id] for tweet_id in result[key]]))

