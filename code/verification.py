import glob
import os
import random

from collections import defaultdict, namedtuple
from functools import partial
from itertools import combinations

import numpy as np 
from numba import jit

from sklearn import base
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

Dataset = namedtuple('Dataset', ['texts', 'titles', 'authors'])

@jit
def min_max(a, b):
    mins = 0.0
    maxs = 0.0
    for i in xrange(a.shape[0]):
        mins += min(a[i], b[i])
        maxs += max(a[i], b[i])
    return mins / maxs

def prepare_corpus(dirname, cutoff=5000):
    authors, titles, texts = [], [], []
    for filename in glob.glob(dirname + "/*.txt"):
        print filename
        author, title = filename.replace(".txt", "").split('_')
        authors.append(author)
        titles.append(title)
        with open(filename) as infile:
            words = [''.join(char for char in word if char.isalpha())
                     for word in infile.read().lower().split()[:cutoff]]
            texts.append(words)
    return Dataset(texts, titles, authors)

def analyzer(words, n=4):
    for word in words:
        if len(word) <= n:
            yield word
        else:
            word = "%" + word + "*"
            for i in range(len(word)-n-1):
                yield word[i:i+n]

class Verification(base.BaseEstimator):
    def __init__(self, n_features=100, random_prop=0.5,
                 sigma=0.4, n_char=4, imposters=2, iterations=100):
        self.n_features = n_features
        self.rand_features = int(random_prop * n_features)
        self.sigma = sigma
        self.n_char = 4
        self.imposters = imposters
        self.iterations = iterations

    def fit(self, dataset):
        self.vectorizer = CountVectorizer(analyzer=partial(analyzer, n=self.n_char))
        texts, titles, authors = dataset
        X = self.vectorizer.fit_transform(texts)
        features = np.asarray(X.sum(0).argsort())[0][-self.n_features:]
        X = X[:,features]
        self.X = TfidfTransformer().fit_transform(X).toarray() # no longer sparse
        return self

    def predict(self, dataset):
        texts, titles, authors = dataset
        for i, j in combinations(range(len(authors)), 2):
            vec_i, title_i, author_i = self.X[i], titles[i], authors[i]
            vec_j, title_j, author_j = self.X[j], titles[j], authors[j]
            distances = []
            for k in range(len(authors)):
                text, title, author = texts[k], titles[k], authors[k]
                if k != i and author not in (author_i, author_j):
                    distances.append((k, author, min_max(self.X[i], self.X[k])))
            distances.sort(key=lambda i: i[-1], reverse=True)
            if distances:
                indexes, imposters, _ = zip(*distances[:self.imposters])
                X = self.X[list(indexes)]
                closest = []
                sigmas = np.zeros(self.iterations)
                for k in range(self.iterations):
                    indices = np.random.randint(0, X.shape[1], size=self.rand_features)
                    truncated_X = X[:,indices]
                    distances = []
                    for idx, candidate in enumerate(imposters):
                        distances.append((candidate, min_max(truncated_X[idx], self.X[i])))
                    distances.append(('target', min_max(self.X[j,indices], self.X[i])))
                    closest.append(max(distances, key=lambda i: i[1])[0])
                    sigma = closest.count("target") / float(len(closest))
                    sigmas[k] = sigma
                mean_sigma = sigmas.mean()
                print "sigma for text: ", mean_sigma
                same_author = True if mean_sigma >= self.sigma else False
                print "%s (by %s) same author as %s (by %s) = %s" % (
                    title_i, author_i, title_j, author_j, same_author)

    verify = predict

if __name__ == '__main__':
    verification = Verification(imposters=10, n_features=1000)
    dataset = prepare_corpus('../data/english')
    verification.fit(dataset)
    verification.verify(dataset)