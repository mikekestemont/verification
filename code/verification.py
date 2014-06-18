#!usr/bin/env python

from operator import itemgetter
import re, os
import random

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def min_max(vectorA, vectorB):
    distance = 0.0
    mins, maxs = 0.0, 0.0
    for i, j in zip(vectorA, vectorB):
        mins+=min(i, j)
        maxs+=max(i, j)
    distance+=(mins/maxs)
    return distance


class Text():
    def __init__(self, author, title, words=None):
        self.author = author
        self.title = title
        # extract n-grams according to Koppel et al.:
        self.features = []
        if words:
            for word in words:
                if len(word) <= 4:
                    self.features.append(word)
                else:
                    word = "%"+word+"*" # extra: add character to mark beginning and ending of word
                    si, ei = 0, 4
                    while ei <= len(word):
                        w = word[si:ei]
                        self.features.append(w)
                        si+=1
                        ei+=1
        return

    def to_dict(self, features):
        D = {}
        for feature in features:
            D[feature] = self.features.count(feature)/float(len(self.features))
        return D


def verification():
    corpusD = "../data/english/"
    cutoff = 5000
    nr_features = 100 # most frequent features
    proportion_random_features = 0.50
    nr_random_features = int(proportion_random_features*nr_features)
    sigma_threshold = 0.40
    m_impostors = 2
    k_iterations = 100
    # make each dict with for each author a list of Text-objects:
    corpus = {}
    for fName in os.listdir(corpusD):
        if fName.endswith(".txt"):
            author, title = fName.replace(".txt", "").split("_")
            if author not in corpus:
                corpus[author] = []
            with open(corpusD+fName, 'r') as inF:
                words = []
                for word in inF.read().lower().split()[:cutoff]:
                    w = "".join([char for char in word if char.isalpha()])
                    words.append(w)
                text = Text(author, title, words)
                corpus[author].append(text)
    # make a global, cumulative frequency list of the features:
    cumul_vocab = {}
    for author, texts in corpus.items():
        for text in texts:
            for feature in text.features:
                try:
                    cumul_vocab[feature]+=1
                except KeyError:
                    cumul_vocab[feature]=1
    # sort the vocubulary to extract the most frequent features:
    cumul_vocab = cumul_vocab.items()
    cumul_vocab.sort(key=itemgetter(1), reverse=True)
    features = [item[0] for item in cumul_vocab[:nr_features]]
    vectors, authors, titles = [], [], []
    for author, texts in corpus.items():
        for text in texts:
            D = text.to_dict(features) # get a dict of relative frequencies
            vectors.append(D)
            authors.append(author)
            titles.append(text.title)
    # convert frequencies to tf-idf scores:
    vec = DictVectorizer()
    corpus = vec.fit_transform(vectors)
    transformer = TfidfTransformer()
    corpus = transformer.fit_transform(corpus)
    # zip the tf-idf vectors with the authors and titles:
    textsD = {}
    for vector, author, title in zip(corpus.toarray(), authors, titles):
        if title not in textsD:
            t = Text(author, title)
            t.tfidf = vector
            textsD[title] = t
    texts = textsD.values()
    authors = set(authors)
    # combine each text with each other text:
    for textA in texts:
        print "======================================================="
        for textB in texts:
            if textB.title != textA.title:
                # start new verification:
                same_author = False
                sigmas = []
                # get m potential impostors by different authors via min-max:
                print "\t\tFetching impostors..."
                distances = []
                for impostor in texts:
                    # make sure that we select different texts, by different authors than this pair
                    if (impostor.title != textA.title)\
                            and (impostor.author != textB.author)\
                            and (impostor.author != textA.author):
                        mima = min_max(textA.tfidf, impostor.tfidf)
                       distances.append((impostor.tfidf, impostor.author, mima))
                distances.sort(key=itemgetter(2), reverse=True)
                # apply cutoff
                impostors = distances[:m_impostors]
                # create a arrays:
                X, y = [], []
                # add the target (i.e. textB from the pair)
                X.append(textB.tfidf)
                y.append("target")
                # add the impostors
                X.extend([impostor_sample for impostor_sample,_,_ in impostors])
                y.extend([impostor_author for _,impostor_author,_ in impostors])
                X = np.array(X)
                closest = []
                for k in range(k_iterations):
                    feature_indices = np.arange(X.shape[1])
                    random_feature_indices = random.sample(feature_indices, nr_random_features)
                    truncated_X = X[:,random_feature_indices]
                    candidates = zip(truncated_X, y)
                    distances = []
                    for candidate in candidates:
                        distances.append((candidate[1], min_max(candidate[0], textA.tfidf)))
                    distances.sort(key=itemgetter(1), reverse=True)
                    closest_candidate = distances[0][0]
                    closest.append(closest_candidate)
                sigma = closest.count("target")/float(len(closest))
                sigmas.append(sigma)
                mean_sigma = np.mean(sigmas)
                print "\t\tsigma for text: "+str(mean_sigma)
                if mean_sigma >= sigma_threshold:
                    same_author = True
                print "\t\t"+textA.title +" (by "+textA.author+") same author as "+textB.title+" (by "+textB.author+") = "+str(same_author)
    return

if __name__ == "__main__":
    verification()