import glob
import logging
import os
import random
import sys
import re
from collections import defaultdict, namedtuple
from functools import partial
from itertools import combinations, islice

import numpy as np
from sklearn import base
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score
from gensim.utils import tokenize

# for remote plotting:
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

Dataset = namedtuple('Dataset', ['texts', 'titles', 'authors'])

def dummy_author():
    i = 0
    while True:
        yield str(i)
        i+=1

def identity(x):
    return x

DUMMY_AUTHORS = dummy_author()

try:
    # attempt to load the compiled cython-version of the min_max distance function:
    from minmax import min_max
except:
    # if that fails, fall back to the numba-version (which is still pretty fast)
    from numba import jit
    @jit('float64(float64[:],float64[:])')
    def min_max(a, b):
        mins = 0.0
        maxs = 0.0
        for i in range(a.shape[0]):
            mins += min(a[i], b[i])
            maxs += max(a[i], b[i])
        return mins/maxs

def prepare_corpus(dirname, text_cutoff):
    underscore = re.compile(r'\_')
    authors, titles, texts = [], [], []
    for filename in glob.glob(dirname + "/*.txt"):
        if '_' in filename:
            author, title = underscore.split(os.path.split(filename)[-1].replace(".txt", ""), maxsplit=1)
        else:
            author, title = DUMMY_AUTHORS.next(), os.sep.split(filename)[-1]
        authors.append(author)
        titles.append(title)
        print(title)
        with open(filename) as infile:
            texts.append(list(islice(tokenize(infile.read(), lowercase=True, deacc=True), 0, text_cutoff)))
    return Dataset(texts, titles, authors)

def analyzer(words, n):
    for word in words:
        if len(word) <= n:
            yield word
        else:
            word = "%" + word + "*"
            for i in range(len(word)-n-1):
                yield word[i:i+n]

class Verification(base.BaseEstimator):
    def __init__(self, n_features, random_prop,
                 sigma, n_actual_imposters, iterations,
                 feature_type, feature_ngram_range, m_potential_imposters):
        self.n_features = n_features
        self.rand_features = int(random_prop * n_features)
        self.sigma = sigma
        self.n_actual_imposters = n_actual_imposters
        self.m_potential_imposters = m_potential_imposters
        self.iterations = iterations
        self.feature_type = feature_type
        self.feature_ngram_range = feature_ngram_range

    def fit(self, background_dataset, devel_dataset):
        """
        Feature selection + tf-idf calculation on dataset
        """
        logging.info("Fitting model.")
        self.background_dataset = background_dataset
        self.devel_dataset = devel_dataset
        if self.feature_type == "char":
            self.analyzer = partial(analyzer, n=self.feature_ngram_range)
            self.vectorizer = CountVectorizer(analyzer=self.analyzer)
        elif self.feature_type == "word":
            self.vectorizer = CountVectorizer(analyzer=identity, ngram_range=self.feature_ngram_range)
        # unpack:
        background_texts, devel_texts = self.background_dataset.texts, self.devel_dataset.texts
        # temporarily join both sets:
        all_texts = background_texts + devel_texts
        X = self.vectorizer.fit_transform(all_texts)
        # select top-frequency features:
        self.most_frequent_feature_indices = np.asarray(X.sum(0).argsort())[0][-self.n_features:] 
        # now re-vectorize, but use tf-idfs of ngrams:
        X = TfidfTransformer().fit_transform(X).toarray() # no longer sparse!
        # Q: only select most frequent features after tf-idf transformation?
        X = X[:,self.most_frequent_feature_indices]
        # divide the sets again:
        self.X_background = X[:len(background_texts)]
        self.X_devel = X[len(background_texts):]
        return self

    def predict(self):
        _, devel_titles, devel_authors = self.devel_dataset
        _, background_titles, background_authors = self.background_dataset
        n_background_samples = self.X_background.shape[0]
        n_devel_samples = self.X_devel.shape[0]
        # pre-calculate distances!!!!!!!!!!!!!!
        min_max_distances = {}
        scores = np.zeros((n_devel_samples, n_devel_samples))
        for i in range(n_devel_samples):
            for j in range(n_devel_samples):
                if i == j:
                    continue
                vec_i, title_i, author_i = self.X_devel[i], devel_titles[i], devel_authors[i]
                vec_j, title_j, author_j = self.X_devel[j], devel_titles[j], devel_authors[j]
                if title_i.split("_")[0] == title_j.split("_")[0]:
                    continue
                logging.info("Predicting scores for %s - %s" % (devel_authors[i], devel_authors[j]))
                # get impostors from the background corpus:
                background_similarities = []
                for k in range(n_background_samples):
                    background_author = background_authors[k]
                    # make sure the background corpus isn't polluted (this step is supervised...):
                    if background_author in (author_i, author_j):
                        continue
                    background_similarities.append((k, background_author, min_max(vec_i, self.X_background[k])))
                background_similarities.sort(key=lambda s:s[-1], reverse=True)
                # select m potential imposters
                m_indexes, m_imposters, _ = zip(*background_similarities[:self.m_potential_imposters])
                m_X = self.X_background[list(m_indexes)]
                # start the verification sampling:
                targets = 0.0
                sigmas = np.zeros(self.iterations)
                for k in range(self.iterations):
                    # randomly select n_actual_impostors from m_potential_imposters:
                    rand_imposter_indices = np.random.randint(0, m_X.shape[0], size=self.n_actual_imposters)
                    truncated_X = m_X[rand_imposter_indices,:]
                    # select random features:
                    rand_feat_indices = np.random.randint(0, truncated_X.shape[1], size=self.rand_features)
                    truncated_X = truncated_X[:,rand_feat_indices]
                    most_similar = None
                    vec_i_trunc, vec_j_trunk = vec_i[rand_feat_indices], vec_j[rand_feat_indices]
                    for idx in range(n_actual_imposters):
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
    plt.figure()
    plt.plot(precisions, recalls)
    plt.savefig("prec_rec.pdf")


if __name__ == '__main__':
    m_potential_imposters = 50
    n_actual_imposters = 30
    n_features = 50000
    random_prop = 0.5
    sigma = 0.8
    iterations = 100
    text_cutoff = 5000
    feature_type = "word" # "char"
    feature_ngram_range = (1,2) # 4
    verification = Verification(n_actual_imposters = n_actual_imposters,
                                m_potential_imposters = m_potential_imposters,
                                iterations = iterations,
                                n_features = n_features,
                                random_prop = random_prop,
                                sigma = sigma,
                                feature_type = feature_type,
                                feature_ngram_range = feature_ngram_range)
    print(verification)
    background_dataset = prepare_corpus(dirname=sys.argv[1], text_cutoff=text_cutoff)
    devel_dataset = prepare_corpus(dirname=sys.argv[2], text_cutoff=text_cutoff)
    verification.fit(background_dataset=background_dataset, devel_dataset=devel_dataset)
    scores = verification.verify()
    precision_recall_curve(scores, devel_dataset)