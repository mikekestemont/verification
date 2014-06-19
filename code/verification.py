import glob
import logging
import os
import random
import unidecode
import sys

from collections import defaultdict, namedtuple
from functools import partial
from itertools import combinations

import numpy as np 
from numba import jit

import pandas as pd
import seaborn as sb

from sklearn import base
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

Dataset = namedtuple('Dataset', ['texts', 'titles', 'authors'])

def dummy_author():
    i = 0
    while True:
        yield str(i)
        i += 1

DUMMY_AUTHORS = dummy_author()

@jit('float64(float64[:],float64[:])')
def min_max(a, b):
    mins = 0.0
    maxs = 0.0
    for i in range(a.shape[0]):
        mins += min(a[i], b[i])
        maxs += max(a[i], b[i])
    return mins / maxs

def prepare_corpus(dirname, cutoff=5000):
    authors, titles, texts = [], [], []
    for filename in glob.glob(dirname + "/*.txt"):
        if '_' in filename:
            author, title = filename.split('/')[-1].replace(".txt", "").split('_')
        else:
            author, title = DUMMY_AUTHORS.next(), unidecode.unidecode(filename.split('/')[-1])
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
        logging.info("Fitting model.")
        self.vectorizer = CountVectorizer(analyzer=partial(analyzer, n=self.n_char))
        texts, titles, authors = dataset
        X = self.vectorizer.fit_transform(texts)
        features = np.asarray(X.sum(0).argsort())[0][-self.n_features:]
        X = X[:,features]
        self.X = TfidfTransformer().fit_transform(X).toarray() # no longer sparse
        return self

    def predict(self, dataset):
        texts, titles, authors = dataset
        n_samples, _ = self.X.shape
        scores = np.zeros((n_samples, n_samples))
        for i, j in combinations(range(n_samples), 2):
            logging.info("Predicting scores for %s - %s" % (authors[i], authors[j]))
            vec_i, title_i, author_i = self.X[i], titles[i], authors[i]
            vec_j, title_j, author_j = self.X[j], titles[j], authors[j]
            similarities = []
            for k in range(n_samples):
                author = authors[k]
                if author not in (author_i, author_j):
                    similarities.append((k, author, min_max(vec_i, self.X[k])))
            similarities.sort(key=lambda s: s[-1], reverse=True)
            indexes, imposters, _ = zip(*similarities[:self.imposters])
            X = self.X[list(indexes)]
            targets = 0.0
            sigmas = np.zeros(self.iterations)
            for k in range(self.iterations):
                indices = np.random.randint(0, X.shape[1], size=self.rand_features)
                truncated_X = X[:,indices]
                most_similar = None
                vec_i_trunc = vec_i[indices]
                for idx in range(len(imposters)):
                    score = min_max(truncated_X[idx], vec_i_trunc)
                    if most_similar is None or score > most_similar:
                        most_similar = score
                if min_max(vec_j[indices], vec_i_trunc) > most_similar:
                    targets += 1
                sigma = targets / (k+1)
                sigmas[k] = sigma
            scores[i, j] = sigmas.mean()
            scores[j, i] = scores[i, j]
            logging.info("Sigma for %s - %s = %.3f" % (authors[i], authors[j], scores[i, j]))
        return scores

    verify = predict

def precision_recall_curve(scores, dataset):
    _, _, authors = dataset
    precisions, recalls = [], []
    for sigma in np.arange(0.1, 1.1, 0.01):
        preds, true, = [], []
        for i, j in combinations(range(len(authors)), 2):
            preds.append(1 if scores[i,j] >= sigma else 0)
            true.append(1 if authors[i] == authors[j] else 0)
        precisions.append(precision_score(preds, true))
        recalls.append(recall_score(preds, true))
    sb.plt.plot(precisions, recalls)


if __name__ == '__main__':
    verification = Verification(imposters=10, n_features=1000)
    print verification
    dataset = prepare_corpus(sys.argv[1])
    verification.fit(dataset)
    scores = verification.verify(dataset)
    # precision_recall_curve(scores, dataset)