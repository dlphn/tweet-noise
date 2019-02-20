import re
import collections
import operator
import pickle

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import arrayBuilder
# from features import stopKeywords


class TextClustering:
    """
    Clusterize tweets for classification
    """

    def __init__(self):
        self.vectorizer = None
        self.km_model = None

    # @staticmethod
    # def process_text(text):
    #     """ Tokenize text removing punctuation """
    #     # stop = set(stopwords.words('french'))
    #     stop = stopKeywords.keywords_stoplist
    #     text = re.sub(r'(?:\@|https?\://)\S+', '', text)  # remove links and @username
    #     text = re.sub(r'[^\w\s]', ' ', text)  # remove non-alphanumeric characters
    #     text = ' '.join([token for token in word_tokenize(text.lower()) if token not in stop])  # remove stopwords
    #     return text

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
            tokenizer=self.tokenize,
            max_features=1500,
            # stop_words=stopwords.words('english'),
            stop_words=stopwords.words('french'),
            max_df=0.7,
            min_df=5,
            lowercase=True
        )
        # texts = [self.process_text(text) for text in texts]
        # print(texts)

        tfidf_model = self.vectorizer.fit_transform(texts)
        self.km_model = KMeans(n_clusters=k, init='k-means++', n_init=10)
        self.km_model.fit(tfidf_model)

        with open('clustering_text_classifier', 'wb') as picklefile:
            pickle.dump(self.km_model, picklefile)

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

        # with open('clustering_text_classifier', 'rb') as training_model:
        #     loaded_model = pickle.load(training_model)
        #     prediction2 = loaded_model.predict(y)
        #     print(prediction2)


def get_tweets():
    data = arrayBuilder.ArrayBuilder()
    return data.retrieve_text_and_labels()


if __name__ == "__main__":

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
    # print(articles)
    model = TextClustering()
    clusters = model.cluster_texts(articles, 20)
    result = dict(clusters)
    # print()
    # print(result)
    print()
    major_labels = {}
    print('Cluster', 'Count', 'Label')
    for key in result.keys():
        counter = collections.Counter([labels[tweet_id] for tweet_id in result[key]])
        major_labels[key] = max(counter.items(), key=operator.itemgetter(1))[0]
        print(key, len(result[key]), counter, max(counter.items(), key=operator.itemgetter(1))[0])
    print()
    print("Unlabelled tweets")
    for tweet_id in range(len(articles)):
        if labels[tweet_id] == '?':
            for key in result.keys():
                if tweet_id in result[key]:
                    print(key, major_labels[key], articles[tweet_id])

    # print()
    # print("Predict unlabelled tweets")
    # for tweet_id in range(len(articles)):
    #     if labels[tweet_id] == '?':
    #         model.predict(articles[tweet_id])
