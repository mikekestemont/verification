import logging
from functools import partial
from itertools import combinations
from operator import itemgetter

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sb

import numpy as np
import scipy.sparse as sp
import pandas as pd
from scipy.cluster.hierarchy import set_link_color_palette
from matplotlib.colors import rgb2hex

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import dist_metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, Normalizer
from gensim.utils import tokenize
from sparse_plm import SparsePLM

from preprocessing import analyzer, identity
from distances import minmax, divergence, cityblock, cosine, euclidean

import theanets

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class DeltaWeightScaler(BaseEstimator):

    def fit(self, X, y=None):
        self.weights = StandardScaler(with_mean=False).fit(X).std_
        return self

    def transform(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)
        for i in range(X.shape[0]):
            start, end = X.indptr[i], X.indptr[i+1]
            X.data[start:end] /= self.weights[X.indices[start:end]]
        # normalize to unit norm:
        return Normalizer(norm="l2", copy=False).fit_transform(X)

pipelines = {
    'tf': Pipeline([('tf', TfidfVectorizer(analyzer=identity, use_idf=False))]),
    'std': Pipeline([('tf', TfidfVectorizer(analyzer=identity, use_idf=False)),
                     ('scaler', DeltaWeightScaler())]),
    'idf': Pipeline([('tf', CountVectorizer(analyzer=identity)),
                     ('tfidf', TfidfTransformer())]),
    'plm': Pipeline([('tf', CountVectorizer(analyzer=identity)),
                     ('plm', SparsePLM())])
}

distance_metrics = {
    "minmax": minmax,
    "cityblock": cityblock,
    "euclidean": euclidean,
    "cosine": cosine,
    "divergence": divergence,
}

class Verification(object):

    def __init__(self, n_features=1000, random_prop=0.5, sample_features=False,
                 sample_authors=False, metric='cosine', text_cutoff=None,
                 sample_iterations=10, n_potential_imposters=30,
                 n_actual_imposters=10, n_train_pairs=None, n_test_pairs=None, random_state=None,
                 vector_space_model='std', weight=0.1, em_iterations=10,
                 ngram_range=(1, 1), norm='l2', top_rank=1, eps=0.01,
                 balanced_pairs=False, deep=False):

        self.n_features = n_features
        self.random_prop = int(random_prop * n_features)
        self.sample_features = sample_features
        self.sample_authors = sample_authors
        self.metric = distance_metrics[metric]
        self.text_cutoff = text_cutoff
        self.sample_iterations = sample_iterations
        self.n_potential_imposters = n_potential_imposters
        if n_actual_imposters and n_actual_imposters < n_potential_imposters:
            self.n_actual_imposters = n_actual_imposters
        else:
            self.n_actual_imposters = n_potential_imposters
        self.n_train_pairs = n_train_pairs
        self.n_test_pairs = n_test_pairs
        self.vector_space_model = vector_space_model
        self.weight = weight
        self.em_iterations = em_iterations
        self.norm = norm
        self.rnd = np.random.RandomState(random_state)
        if balanced_pairs:
            self._setup_pairs = self._setup_balanced_pairs
        self.top_rank = top_rank
        self.deep = deep

        self.parameters = {'tf__max_features': n_features,
                           'tf__ngram_range': ngram_range,
                           'tf__min_df': 2}
        if self.vector_space_model == 'idf':
            self.parameters['tfidf__norm'] = norm
        elif self.vector_space_model == 'plm':
            self.parameters['plm__eps'] = eps
            self.parameters['plm__weight'] = weight
            self.parameters['plm__iterations'] = em_iterations
            self.parameters['plm__norm'] = norm

    def fit(self, train_dataset, test_dataset):
        self.train_data, self.train_titles, self.train_authors = train_dataset
        self.test_data, self.test_titles, self.test_authors = test_dataset
        transformer = pipelines[self.vector_space_model]
        transformer.set_params(**self.parameters)
        transformer.fit(self.train_data + self.test_data)
        self.X_train = transformer.transform(self.train_data)
        logging.info("Training corpus: n_samples=%s / n_features=%s" % (
            self.X_train.shape))
        self.X_test = transformer.transform(self.test_data)
        logging.info("Test corpus: n_samples=%s / n_features=%s" % (
            self.X_test.shape))
        return self


    def _setup_pairs(self, phase='train'):
        pairs = []
        if phase == "train":
            titles, authors = self.train_titles, self.train_authors
            n_pairs = self.n_train_pairs
        elif phase == "test":
            titles, authors = self.test_titles, self.test_authors
            n_pairs = self.n_test_pairs
        pairs = []
        for i in range(len(titles)):
            for j in range(len(titles)):
                if i != j:
                    title_i, title_j = titles[i], titles[j]
                    if "_" in title_i and "_" in title_j:
                        if title_i.split("_")[0] == title_j.split('_')[0]:
                            continue
                    if authors[i] == authors[j]:
                        pairs.append((i, j))
                    else:
                        pairs.append((i, j))
        self.rnd.shuffle(pairs)
        if n_pairs == None:
            return pairs
        return pairs[:n_pairs]

    def _setup_balanced_pairs(self, phase='train'):
        if phase == 'train':
            titles, authors = self.train_titles, self.train_authors
            n_pairs = self.n_train_pairs
        elif phase == 'test':
            titles, authors = self.test_titles, self.test_authors
            n_pairs = self.n_test_pairs
        same_author_pairs, diff_author_pairs = [], []
        for i in range(len(titles)):
            for j in range(len(titles)):
                if i != j:
                    title_i, title_j = titles[i], titles[j]
                    if "_" in title_i and "_" in title_j:
                        if title_i.split("_")[0] == title_j.split('_')[0]:
                            continue
                    if authors[i] == authors[j]:
                        same_author_pairs.append((i, j))
                    else:
                        diff_author_pairs.append((i, j))
        self.rnd.shuffle(same_author_pairs)
        self.rnd.shuffle(diff_author_pairs)
        same_author_pairs = same_author_pairs[:int(n_pairs / 2.0)]
        diff_author_pairs = diff_author_pairs[:len(same_author_pairs)]
        pairs = same_author_pairs + diff_author_pairs
        self.rnd.shuffle(pairs) # needed for proportional evaluation
        return pairs

    def compute_distances(self, pairs=[], phase='train'):
        distances, labels = [], []
        X = self.X_train if phase == 'train' else self.X_test
        authors = self.train_authors if phase == 'train' else self.test_authors
        for k, (i, j) in enumerate(pairs):
            logging.info("Verifying pair %s / %s" % (k+1, len(pairs)))
            dist = self.metric(X[i], X[j])
            assert not (np.isnan(dist) or np.isinf(dist))
            distances.append(dist)
            labels.append(
                "same_author" if authors[i] == authors[j] else
                "diff_author")
        return distances, labels

    def compute_sigmas(self, pairs=[], phase='train'):
        if phase == "test":
            test_X, test_authors, test_titles = self.X_test, self.test_authors, self.test_titles
            background_X, background_authors, background_titles = self.X_train, self.train_authors, self.train_titles
        elif phase == "train":
            test_X, test_authors, test_titles = self.X_train, self.train_authors, self.train_titles
            background_X, background_authors, background_titles = self.X_train, self.train_authors, self.train_titles
        sigmas, labels = [], []
        for k, (i, j) in enumerate(pairs):
            logging.info("Verifying pair %s / %s" % (k+1, len(pairs)))
            author_i, author_j = test_authors[i], test_authors[j]
            background_sims = []
            # first, select n_potential_imposters
            for k in range(background_X.shape[0]):
                if "_" in background_titles[k]:
                    xt = background_titles[k].split('_')[0]
                    if "_" in test_titles[i]:
                        t1 = test_titles[i].split("_")[0]
                        if t1 == xt:
                            continue
                    if "_" in test_titles[j]:
                        t2 = test_titles[j].split("_")[0]
                        if t2 == xt:
                            continue
                background_sims.append((k, background_authors[k],
                                       self.metric(test_X[i], background_X[k])))
            background_sims.sort(key=lambda sim: sim[-1])            
            indexes, imposters, _ = zip(*background_sims[:self.n_potential_imposters])
            X_imposters = background_X[list(indexes), :]
            # start the iteration for the pairs:
            targets = 0.0
            for iteration in range(self.sample_iterations):
                # randomly select imposters:
                if self.sample_authors:
                    rnd_imposters = self.rnd.randint(
                        X_imposters.shape[0], size=self.n_actual_imposters)
                    X_truncated = X_imposters[rnd_imposters, :]
                else:
                    X_truncated = X_imposters
                # randomly select features:
                if self.sample_features:
                    rnd_features = self.rnd.randint(
                        X_truncated.shape[1], size=self.random_prop)
                else:
                    rnd_features = range(X_truncated.shape[1])
                vec_i, vec_j = test_X[i], test_X[j]
                # compute distance to target doc:
                all_candidates = [self.metric(vec_i, vec_j, rnd_features)]
                # compute distance to imposters:
                all_candidates += [self.metric(vec_i, X_truncated[k], rnd_features)
                                   for k in range(self.n_actual_imposters)]
                all_candidates = np.array(all_candidates)
                # find rank of target doc in ranking:
                rank_target = np.where(all_candidates.argsort() == 0)[0][0] + 1
                if self.top_rank == 1:
                    # standard rank checking:
                    targets += 1.0 if rank_target == 1 else 0.0
                else:
                    # or a variation on mean reciprocal rank:
                    if rank_target <= self.top_rank:
                        targets += 1.0 / rank_target
            # append the correct label:
            if author_i == author_j:
                labels.append("same_author")
                #print("same_author")
            else:
                labels.append("diff_author")
                #print("diff_author")
            # append the sigma as a distance measure (1 - sigma)
            sigma = 1 - targets / self.sample_iterations
            #print str(sigma)+"!!!"
            sigmas.append(sigma)
        return sigmas, labels

    def _verification_without_sampling(self, train_pairs, test_pairs):
        # compute distances plain distances between pairs without sampling:
        train_dists, train_labels = self.compute_distances(train_pairs, phase="train")
        test_dists, test_labels = self.compute_distances(test_pairs, phase="test")
        # scale the train and test distances together:
        distances = train_dists + test_dists
        min_dist, max_dist = min(distances), max(distances)
        scale = lambda d: (d - min_dist) / (max_dist - min_dist)
        train_scores = zip(train_labels, map(scale, train_dists))
        test_scores = zip(test_labels, map(scale, test_dists))
        return train_scores, test_scores

    def _verification_with_sampling(self, train_pairs, test_pairs):
        train_sigmas, train_labels = self.compute_sigmas(train_pairs, phase="train")
        test_sigmas, test_labels = self.compute_sigmas(test_pairs, phase="test")
        train_scores = zip(train_labels, train_sigmas)
        test_scores = zip(test_labels, test_sigmas)
        return train_scores, test_scores

    def _verification_with_network(self, train_pairs, test_pairs):
        print "test"
        train_scores, test_scores = [], []
        pretrain_matrix, train_labels = [], []
        X, authors = self.X_train.toarray(), self.train_authors
        print train_pairs
        for (i, j) in train_pairs:
            vector = np.concatenate((X[i],X[j]))
            pretrain_matrix.append(vector)
            if authors[i] == authors[j]:
                train_labels.append("same_author")
            else:
                train_labels.append("diff_author")
        print type(pretrain_matrix)
        pretrain_matrix = np.asarray(pretrain_matrix)
        e = theanets.Experiment(theanets.Autoencoder, layers=(len(pretrain_matrix[0]), 100, len(pretrain_matrix[0])))
        e.train()
        return train_scores, test_scores


    def get_distance_table(self, dists, pairs, phase):
        if phase == 'train':
            titles, authors = self.train_titles, self.train_authors
        elif phase == 'test':
            titles, authors = self.test_titles, self.test_authors
        textlabels = [a+"_"+t for a, t in zip(authors, titles)]
        df = pd.DataFrame(columns=(["id"]+textlabels))
        # prepopulate with zeros:
        for i, tl in enumerate(textlabels):
            df.loc[i] = [tl]+list(np.zeros(len(textlabels)))
        # populate with the scores (that we have, cf. limited nr of pairs):
        for d, (i, j) in zip(dists, pairs):
            df.ix[i,j+1] = d
        # set index:
        df = df.set_index("id")
        # save:
        df.to_csv("../plots/"+phase+"_dists.txt", sep="\t", encoding="utf-8")
        return df

    def verify(self):
        "Start the verification procedure."
        # set the pairs:
        train_pairs = self._setup_pairs(phase="train")
        test_pairs = self._setup_pairs(phase="test")
        if self.sample_authors or self.sample_features:
            train_scores, test_scores = self._verification_with_sampling(train_pairs, test_pairs)
        elif self.deep == True:
            train_scores, test_scores = self._verification_with_network(train_pairs, test_pairs)
        else:
            train_scores, test_scores = self._verification_without_sampling(train_pairs, test_pairs)
        train_dists, test_dists = [score for label, score in train_scores], [score for label, score in test_scores]
        # dump the distance tables (in so far as they are filled):
        self.df_train = self.get_distance_table(train_dists, train_pairs, "train")
        self.df_test = self.get_distance_table(test_dists, test_pairs, "test")
        return (train_scores, test_scores)
