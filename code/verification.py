import glob
import logging
import os
import random
#import unidecode
import sys

from collections import defaultdict, namedtuple
from functools import partial
from itertools import combinations, islice

import numpy as np 
from numba import jit

import pandas as pd
import seaborn as sb

from sklearn import base
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score

from gensim.utils import tokenize

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
    return mins/maxs

def prepare_corpus(dirname, text_cutoff=100000):
    authors, titles, texts = [], [], []
    for filename in glob.glob(dirname + "/*.txt"):
        if '_' in filename:
            author, title = filename.split('/')[-1].replace(".txt", "").split('_')
        else:
            author, title = DUMMY_AUTHORS.next(), filename.split('/')[-1]
        authors.append(author)
        titles.append(title)
        print(title)
        with open(filename) as infile:
            texts.append(list(islice(tokenize(infile.read(), lowercase=True, deacc=True), 0, text_cutoff)))
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
                 sigma=0.4, n_char=4, n_imposters=2, iterations=100, analyzer=analyzer):
        self.n_features = n_features
        self.rand_features = int(random_prop * n_features)
        self.sigma = sigma
        self.n_char = n_char
        self.n_imposters = n_imposters
        self.iterations = iterations
        self.analyzer = partial(analyzer, n=self.n_char)

    def fit(self, background_dataset, devel_dataset):
        """
        Feature selection + tf-idf calculation on dataset
        """
        self.background_dataset = background_dataset
        self.devel_dataset = devel_dataset
        logging.info("Fitting model.")
        self.vectorizer = CountVectorizer(analyzer=self.analyzer)
        # unpack:
        background_texts, devel_texts = self.background_dataset.texts, self.devel_dataset.texts
        # temporarily join both sets:
        all_texts = background_texts+devel_texts
        X = self.vectorizer.fit_transform(all_texts)
        # select top-frequency features:
        self.feature_indices = np.asarray(X.sum(0).argsort())[0][-self.n_features:] 
        X = X[:,self.feature_indices]
        # now re-vectorize, but use tf-idfs of ngrams:
        X = TfidfTransformer().fit_transform(X).toarray() # no longer sparse!
        # divide the sets again:
        self.X_background = X[:len(background_texts)]
        self.X_devel = X[len(background_texts):]
        return self

    def predict(self):
        _, devel_titles, devel_authors = self.devel_dataset
        _, background_titles, background_authors = self.background_dataset
        n_background_samples = self.X_background.shape[0]
        n_devel_samples = self.X_devel.shape[0]
        scores = np.zeros((n_devel_samples, n_devel_samples))
        for i, j in combinations(range(n_devel_samples), 2): # get the combination of each two texts
            logging.info("Predicting scores for %s - %s" % (devel_authors[i], devel_authors[j]))
            vec_i, title_i, author_i = self.X_devel[i], devel_titles[i], devel_authors[i]
            vec_j, title_j, author_j = self.X_devel[j], devel_titles[j], devel_authors[j]
            # get impostors from the background corpus:
            background_similarities = []
            for k in range(n_background_samples):
                background_author = background_authors[k]
                # make sure the background corpus isn't polluted:
                assert background_author not in (author_i, author_j)
                background_similarities.append((k, background_author, min_max(vec_i, self.X_background[k])))
            background_similarities.sort(key=lambda s:s[-1], reverse=True)
            indexes, imposters, _ = zip(*background_similarities[:self.n_imposters])
            tmp_X = self.X_background[list(indexes)]
            targets = 0.0
            sigmas = np.zeros(self.iterations)
            for k in range(self.iterations):
                rand_feat_indices = np.random.randint(0, tmp_X.shape[1], size=self.rand_features)
                truncated_X = tmp_X[:,rand_feat_indices]
                most_similar = None
                vec_i_trunc, vec_j_trunk = vec_i[rand_feat_indices], vec_j[rand_feat_indices]
                for idx in range(len(imposters)):
                    score = min_max(truncated_X[idx], vec_i_trunc)
                    if most_similar is None or score > most_similar:
                        most_similar = score
                if min_max(vec_i_trunc, vec_j_trunk) > most_similar:
                    targets+=1.0
                sigmas[k] = targets/(k+1.0)
            scores[i,j] = sigmas.mean()
            scores[j,i] = scores[i,j]
            logging.info("Sigma for %s (%s) - %s (%s) = %.3f" % (
                devel_titles[i], devel_authors[i], devel_titles[j], devel_authors[j], scores[i, j]))
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
        try:
            precisions.append(precision_score(preds, true))
        except:
            precisions.append(0.0)
        try:
            recalls.append(recall_score(preds, true))
        except:
            recalls.append(0.0)
    sb.plt.plot(precisions, recalls)
    sb.plt.show()


if __name__ == '__main__':
    n_char = 4
    n_imposters = 10
    n_features = 100000
    random_prop = 0.5
    sigma = 0.4
    iterations = 100
    text_cutoff = 100000
    verification = Verification(n_imposters = n_imposters,
                                n_features = n_features,
                                n_char = n_char,
                                random_prop = random_prop,
                                sigma = sigma,
                                iterations = 100)
    print(verification)
    background_dataset = prepare_corpus(dirname=sys.argv[1], text_cutoff=text_cutoff)
    devel_dataset = prepare_corpus(dirname=sys.argv[2], text_cutoff=text_cutoff)
    verification.fit(background_dataset=background_dataset, devel_dataset=devel_dataset)
    scores = verification.verify()
    #precision_recall_curve(scores, background_dataset)