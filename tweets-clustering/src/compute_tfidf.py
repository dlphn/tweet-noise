# -*- coding: utf-8 -*-
"""
@author: bmazoyer

edits by dshi, hbaud, vlefranc
"""


import pickle
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import normalize
from scipy import sparse
import numpy as np
import re
from docs.config import STOP_WORDS, BINARY_COUNT
import os.path
# from keras.models import Model
# from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.preprocessing import image


class Vectorizer:

    def __init__(self):
        self.df = np.array([])
        self.features_names = []
        self.n_samples = 0
        self.name = "tfidf"

    def load_history(self, path):
        for attr in ["df", "features_names", "n_samples"]:
            with open(path + "_" + attr, "rb") as f:
                setattr(self, attr, pickle.load(f))

        # print(self.df)
        # print(self.features_names)
        # print(self.n_samples)
        return self

    def save(self, path):
        for attr in ["df", "features_names", "n_samples"]:
            with open(path + "_" + attr, "wb") as f:
                pickle.dump(getattr(self, attr), f)
        logging.info("Saved data to {}".format(path))


class TfIdf(Vectorizer):

    def get_new_features(self, data):
        features_set = set(self.features_names)
        fit_model = CountVectorizer(stop_words=STOP_WORDS)
        # see https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af for custom analyzr/tokenizr
        fit_model.fit(data["text"].tolist())
        for term in fit_model.get_feature_names():
            if term not in features_set:
                self.features_names.append(term)

    def build_count_vectors(self, data):
        # sort words following features_name order, absent words will be counted as 0
        count_model = CountVectorizer(binary=BINARY_COUNT, vocabulary=self.features_names)
        return count_model.transform(data["text"].tolist())

    def compute_df(self, count_vectors):
        # add zeros to the end of the stored df vector
        zeros = np.zeros(count_vectors.shape[1] - len(self.df), dtype=self.df.dtype)
        df = np.append(self.df, zeros)
        # compute new df array
        df = df + np.bincount(count_vectors.indices)
        return df

    def add_new_samples(self, data):
        self.get_new_features(data)
        count_vectors = self.build_count_vectors(data)
        self.df = self.compute_df(count_vectors)
        return count_vectors

    def compute_vectors(self, count_vectors, min_df):

        if min_df > 0:
            mask = self.df > min_df
            df = self.df[mask]
            count_vectors = count_vectors[:,mask]
        else:
            df = self.df
        self.n_samples += count_vectors.shape[0]

        # compute smoothed idf
        idf = np.log((self.n_samples + 1) / (df + 1)) + 1
        transformer = TfidfTransformer()
        transformer._idf_diag = sparse.diags(idf, offsets=0, shape=(len(df), len(df)), format="csr", dtype=df.dtype)
        X = transformer.transform(count_vectors)

        # equivalent to:
        # X = normalize(X * transformer._idf_diag, norm='l2', copy=False)
        return X


class W2V(Vectorizer):

    def __init__(self, n_features, min_count=0):
        super().__init__()
        self.n_features = n_features
        self.model = Word2Vec(min_count=min_count, size=n_features)
        self.vocab = {}
        self.name = "w2v"
        self.wv = self.model.wv

    def load_history(self, path, vectors=True):
        super().load_history(path)
        if vectors:
            self.wv = KeyedVectors.load(path + ".kv", mmap='r')
            self.vocab = self.wv.vocab
        else:
            if os.path.isfile(path + "_" + "w2v"):
                self.model = Word2Vec.load(path + "_" + "w2v")
                self.vocab = self.model.wv.vocab


    def save(self, path, vectors=True):
        if vectors:
            self.model.wv.save(path + ".kv")
        else:
            self.model.save(path + "_" + "w2v")

    def preprocess(self, data):
        logging.info("preprocess")
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        # same preprocessing as default sklearn CountVectorizer
        text = data.text.str.lower().str.findall(token_pattern)
        return text

    def add_new_samples(self, text, min_df=0):
        logging.info("add new samples")
        freq_dict = {k: v for k, v in zip(self.features_names, self.df)}
        if self.vocab == {}:
            self.model.build_vocab_from_freq(freq_dict, update=False)
        elif len(self.features_names) != len(self.vocab):
            self.model.build_vocab_from_freq(freq_dict, update=True)
        self.model.train(text, total_examples=len(text), epochs=self.model.epochs)
        self.vocab = self.model.wv.vocab

    def compute_vectors(self, text):
        logging.info("compute vectors")
        vectors = np.zeros((len(text), self.n_features))
        for idx, sentence in text.iteritems():
            if sentence != []:
                try:
                    vectors[idx] = np.array(
                        [self.wv[w] if w in self.vocab else np.zeros(self.n_features) for w in sentence]
                    ).mean(axis=0)
                except KeyError:
                    logging.error(sentence)
                    raise
        del self.model
        return vectors


class ResNetLayer:

    def __init__(self, ):
        base_model = ResNet50(weights='imagenet', include_top=True)
        self.featurizer = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        self.name = "ResNet"

    def compute_vectors(self, images, image_path, vectors_path):
        if os.path.isfile(vectors_path):
            X = np.load(vectors_path)
        else:
            unique_lsh = images.image.unique().tolist()
            X = []
            for id in unique_lsh:
                img = image.load_img(image_path + "/" + str(id), target_size=(224, 224))
                X.append(image.img_to_array(img))
            X = preprocess_input(np.array(X))
            X = self.featurizer.predict(X)
            mapping = dict(zip(unique_lsh, X))
            X = np.array([mapping[key] for i, key in images.image.iteritems()])
            np.save(vectors_path, X)

        return X
