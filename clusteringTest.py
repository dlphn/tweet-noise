import sys

import nltk
import re
import spacy

from math import log, sqrt
from itertools import combinations

from features import arrayBuilder

nlp = spacy.load('fr_core_news_md')


def cosine_distance(a, b):
    cos = 0.0
    a_tfidf = a["tfidf"]
    for token, tfidf in b["tfidf"].items():
        if token in a_tfidf:
            cos += tfidf * a_tfidf[token]
    return cos


def normalize(features):
    norm = 1.0 / sqrt(sum(i**2 for i in features.values()))
    for k, v in features.items():
        features[k] = v * norm
    return features


def add_tfidf_to(documents):
    tokens = {}
    for id, doc in enumerate(documents):
        # print(id, doc)
        tf = {}
        doc["tfidf"] = {}
        doc_tokens = doc.get("tokens", [])
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        num_tokens = len(doc_tokens)
        if num_tokens > 0:
            for token, freq in tf.items():
                tokens.setdefault(token, []).append((id, float(freq) / num_tokens))

    print(tokens)
    doc_count = float(len(documents))
    for token, docs in tokens.items():
        # print(token, docs)
        idf = log(doc_count / len(docs))
        for id, tf in docs:
            tfidf = tf * idf
            if tfidf > 0:
                documents[id]["tfidf"][token] = tfidf

    for doc in documents:
        doc["tfidf"] = normalize(doc["tfidf"])
    print(documents)


def choose_cluster(node, cluster_lookup, edges):
    new = cluster_lookup[node]
    if node in edges:
        seen, num_seen = {}, {}
        for target, weight in edges.get(node, []):
            seen[cluster_lookup[target]] = seen.get(
                cluster_lookup[target], 0.0) + weight
        for k, v in seen.items():
            num_seen.setdefault(v, []).append(k)
        new = num_seen[max(num_seen)][0]
    return new


def majorclust(graph):
    cluster_lookup = dict((node, i) for i, node in enumerate(graph.nodes))

    count = 0
    movements = set()
    finished = False
    while not finished:
        finished = True
        for node in graph.nodes:
            new = choose_cluster(node, cluster_lookup, graph.edges)
            move = (node, cluster_lookup[node], new)
            if new != cluster_lookup[node] and move not in movements:
                movements.add(move)
                cluster_lookup[node] = new
                finished = False

    clusters = {}
    for k, v in cluster_lookup.items():
        clusters.setdefault(v, []).append(k)

    return clusters.values()


def get_distance_graph(documents):
    class Graph(object):
        def __init__(self):
            self.edges = {}

        def add_edge(self, n1, n2, w):
            self.edges.setdefault(n1, []).append((n2, w))
            self.edges.setdefault(n2, []).append((n1, w))

    graph = Graph()
    doc_ids = range(len(documents))
    graph.nodes = set(doc_ids)
    for a, b in combinations(doc_ids, 2):
        graph.add_edge(a, b, cosine_distance(documents[a], documents[b]))
    return graph


def tokenize_only(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def get_documents():
    texts = [
        "J'aime les frites",
        "Je suis d'accord, j'aime pas la pluie...",
        "Les frites je les adore",
        "La taxe d'habitation ne sera pas augmentée annonce le gouvernement",
        "Cool pas d'augmentation de la taxe d'habitation !!!",
        "Il fait trop moche aujourd'hui",
    ]
    result = []
    for i, text in enumerate(texts):
        doc = nlp(text)
        tokens = [token.text for token in doc]
        result.append({"text": text, "tokens": tokens})
    return result


def get_tweets():
    mongo = arrayBuilder.ArrayBuilder()
    tweets = mongo.retrieve()
    result = []
    for i, text in enumerate(tweets):
        doc = nlp(text)
        tokens = [token.text for token in doc]
        result.append({"text": text, "tokens": tokens})
    return result


def main(args):
    # documents = get_documents()
    documents = get_tweets()
    add_tfidf_to(documents)
    dist_graph = get_distance_graph(documents)

    clusters = majorclust(dist_graph)
    for cluster in clusters:
        print("=========")
        for doc_id in cluster:
            print(documents[doc_id]["text"])

    print(len(clusters))


if __name__ == '__main__':
    main(sys.argv)
