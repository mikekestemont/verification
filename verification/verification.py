import logging
from functools import partial
from itertools import combinations
from operator import itemgetter

import numpy as np
import scipy.sparse as sp
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import dist_metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from gensim.utils import tokenize
from sparse_plm import SparsePLM

from preprocessing import analyzer, identity
from distances import minmax, divergence, cityblock, cosine, euclidean
from plotting import prec_recall_curve, plot_test_densities, plot_test_results


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
        return X


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
                 n_actual_imposters=10, n_test_pairs=None, random_state=None,
                 vector_space_model='std', weight=0.1, em_iterations=10,
                 ngram_range=(1, 1), norm='l2', top_rank=1, eps=0.01,
                 balanced_test_pairs=False):

        self.n_features = n_features
        self.random_prop = int(random_prop * n_features)
        self.sample_features = sample_features
        self.sample_authors = sample_authors
        self.metric = distance_metrics[metric]
        self.text_cutoff = text_cutoff
        self.sample_iterations = sample_iterations
        self.n_potential_imposters = n_potential_imposters
        self.n_actual_imposters = n_actual_imposters
        self.n_test_pairs = n_test_pairs
        self.vector_space_model = vector_space_model
        self.weight = weight
        self.em_iterations = em_iterations
        self.norm = norm
        self.rnd = np.random.RandomState(random_state)
        if balanced_test_pairs:
            self._setup_test_pairs = self._setup_balanced_test_pairs
        self.top_rank = top_rank
        # TODO: als we met plm werken is max_features eigen wat gek
        #       omdat plm max)features gaat opzoeken...
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
            self.X_train.shape))
        return self

    def _setup_test_pairs(self, phase='train'):
        test_pairs = []
        titles = self.train_titles if phase == 'train' else self.test_titles
        for i in range(len(titles)):
            for j in range(i):
                if i != j:
                    title_i, title_j = titles[i], titles[j]
                    if title_i.split("_")[0] != title_j.split('_')[0]:
                        test_pairs.append((i, j))
        self.rnd.shuffle(test_pairs)
        if self.n_test_pairs == None:
            return test_pairs
        return test_pairs[:self.n_test_pairs]

    def _setup_balanced_test_pairs(self, phase='train'):
        if phase == 'train':
            titles, authors = self.train_titles, self.train_authors
        else:
            titles, authors = self.test_titles, self.test_authors
        same_author_pairs, diff_author_pairs = [], []
        for i in range(len(titles)):
            for j in range(i):
                if i != j:
                    title_i, title_j = titles[i], titles[j]
                    if title_i.split("_")[0] != title_j.split('_')[0]:
                        if authors[i] == authors[j]:
                            same_author_pairs.append((i, j))
                        else:
                            diff_author_pairs.append((i, j))
        self.rnd.shuffle(same_author_pairs)
        self.rnd.shuffle(diff_author_pairs)
        same_author_pairs = same_author_pairs[:int(self.n_test_pairs / 2.0)]
        diff_author_pairs = diff_author_pairs[:int(self.n_test_pairs / 2.0)]
        test_pairs = same_author_pairs + diff_author_pairs
        self.rnd.shuffle(test_pairs) # needed for proportional evaluation
        return test_pairs

    def compute_distances(self, phase='train'):
        distances, labels = [], []
        test_pairs = self._setup_test_pairs(phase)
        X = self.X_train if phase == 'train' else self.X_test
        authors = self.train_authors if phase == 'train' else self.test_authors
        for k, (i, j) in enumerate(test_pairs):
            logging.info("Verifying pair %s / %s" % (k+1, len(test_pairs)))
            dist = self.metric(X[i], X[j])
            assert not (np.isnan(dist) or np.isinf(dist))
            distances.append(dist)
            labels.append(
                "same_author" if authors[i] == authors[j] else
                "diff_author")
        return distances, labels, test_pairs

    def _verification(self, train_dists, train_labels, test_dists, test_labels):
        distances = train_dists + test_dists
        min_dist, max_dist = min(distances), max(distances)
        scale = lambda d: (d - min_dist) / (max_dist - min_dist)
        train_scores = zip(train_labels, map(scale, train_dists))
        test_scores = zip(test_labels, map(scale, test_dists))
        return train_scores, test_scores

    def _verification_with_sampling(self):
        test_pairs = self._setup_test_pairs()
        for k, (i, j) in enumerate(test_pairs):
            logging.info("Verifying pair %s / %s" % (k+1, len(test_pairs)))
            author_i, author_j = self.dev_authors[i], self.dev_authors[j]
            train_sims = []
            for k in range(self.X_train.shape[0]):
                if self.train_authors[k] not in (author_i, author_j):
                    train_sims.append((k, self.train_authors[k],
                                       self.metric(self.X_dev[i], self.X_train[k])))
            train_sims.sort(key=lambda sim: sim[-1])
            indexes, imposters, _ = zip(*train_sims[:self.n_potential_imposters])
            X_imposters = self.X_train[list(indexes), :]
            targets = 0.0
            #sigmas = np.zeros(self.sample_iterations)
            for iteration in range(self.sample_iterations):
                rnd_imposters = self.rnd.randint(
                    X_imposters.shape[0], size=self.n_actual_imposters)
                X_truncated = X_imposters[rnd_imposters, :]
                rnd_features = self.rnd.randint(
                    X_truncated.shape[1], size=self.random_prop)
                vec_i, vec_j = self.X_dev[i], self.X_dev[j]
                # most_similar = min(self.metric(vec_i, X_truncated[k], rnd_features)
                #                    for k in range(self.n_actual_imposters))
                # target_dist = self.metric(vec_i, vec_j, rnd_features)
                # if target_dist <= most_similar:
                #     targets += 1
                all_candidates = [self.metric(vec_i, vec_j, rnd_features)]
                all_candidates += [self.metric(vec_i, X_truncated[k], rnd_features)
                                   for k in range(self.n_actual_imposters)]
                all_candidates = np.array(all_candidates)
                # find rank of target in ranking
                rank_target = np.where(all_candidates.argsort() == 0)[0][0] + 1
                if self.top_rank == 1:
                    targets += 1.0 if rank_target == 1 else 0.0
                else:
                    if rank_target <= self.top_rank:
                        targets += 1.0 / rank_target
                #sigmas[iteration] = targets / (iteration + 1.0)
            yield ("same_author" if author_i == author_j else "diff_author",
                   #1 - sigmas.mean())
                   1 - targets / self.sample_iterations)

    def get_distance_table(self, dists, pairs, phase):
        if phase == 'train':
            titles, authors = self.train_titles, self.train_authors
        else:
            titles, authors = self.test_titles, self.test_authors
        textlabels = [a+"_"+t for a, t in zip(authors, titles)]
        df = pd.DataFrame(columns=(["id"]+textlabels))
        # prepopulate with zeros:
        for i, tl in enumerate(textlabels):
            r = [tl]+list(np.zeros(len(textlabels)))
            df.loc[i] = [tl]+list(np.zeros(len(textlabels)))
        # populate with the scores that we have:
        for d, index in zip(dists, pairs):
            df[index] = d
        # save:
        with open("../plots/"+phase+"_dists.txt", "w+") as F:
            F.write(df.to_string())
        return df

    def verify(self):
        "Start the verification procedure."
        if self.sample_authors or self.sample_features:
            raise ValueError("Must be reimplemented...")
            return self._verification_with_sampling()
        else:
            # get original distances:
            train_dists, train_labels, train_pairs = self.compute_distances(phase="train")
            test_dists, test_labels, test_pairs = self.compute_distances(phase="test")
            # normalize:
            train_scores, test_scores = self._verification(train_dists, train_labels, test_dists, test_labels)
            train_dists, test_dists = [score for label, score in train_scores], [score for label, score in train_scores]
            # dump the distance tables (in so far as they are filled):s
            train_df = self.get_distance_table(train_dists, train_pairs, "train")
            test_df = self.get_distance_table(test_dists, test_pairs, "test")
            return train_scores, test_scores
