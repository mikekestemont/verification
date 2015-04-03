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
from sklearn.preprocessing import StandardScaler, Normalizer
from gensim.utils import tokenize
from sparse_plm import SparsePLM

from preprocessing import ngram_analyzer, identity
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
        # normalize to unit norm:
        return X

def get_pipelines(feature_type="", ngrams=3):
    if feature_type == "words":
        pipelines = {
            'tf': Pipeline([('tf', TfidfVectorizer(analyzer=identity, use_idf=False))]),
            'std': Pipeline([('tf', TfidfVectorizer(analyzer=identity, use_idf=False)),
                             ('scaler', DeltaWeightScaler())]),
            'tfidf': Pipeline([('tf', CountVectorizer(analyzer=identity)),
                             ('tfidf', TfidfTransformer())]),
            'plm': Pipeline([('tf', CountVectorizer(analyzer=identity)),
                             ('plm', SparsePLM())]),
            'bin': Pipeline([('tf', CountVectorizer(analyzer=identity, binary=True, dtype=np.float64))])
        }
    elif feature_type == "chars":
        pipelines = {
            'tf': Pipeline([('tf', TfidfVectorizer(analyzer=partial(ngram_analyzer,
                                                   n=ngrams), use_idf=False))]),
            'std': Pipeline([('tf', TfidfVectorizer(analyzer=partial(ngram_analyzer,
                                                   n=ngrams), use_idf=False)),
                             ('scaler', DeltaWeightScaler())]),
            'tfidf': Pipeline([('tf', CountVectorizer(analyzer=partial(ngram_analyzer,
                                                   n=ngrams))),
                             ('tfidf', TfidfTransformer())]),
            'plm': Pipeline([('tf', CountVectorizer(analyzer=partial(ngram_analyzer,
                                                   n=ngrams))),
                             ('plm', SparsePLM())]),
            'bin': Pipeline([('tf', CountVectorizer(analyzer=partial(ngram_analyzer,
                                                   n=ngrams), binary=True, dtype=np.float64))])
        }
    else:
        raise NotImplementedError("The feature type requested is not available...")
    return pipelines

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
                 n_actual_imposters=10, n_dev_pairs=None, n_test_pairs=None, random_state=None,
                 vector_space_model='std', weight=0.1, em_iterations=10,
                 ngram_range=(1, 1), norm='l2', top_rank=1, eps=0.01,
                 balanced_pairs=False, feature_type="words", control_pairs=True):

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
        self.n_dev_pairs = n_dev_pairs
        self.n_test_pairs = n_test_pairs
        self.control_pairs = control_pairs
        self.vector_space_model = vector_space_model
        self.weight = weight
        self.em_iterations = em_iterations
        self.norm = norm
        self.rnd = np.random.RandomState(random_state)
        if balanced_pairs:
            self._setup_pairs = self._setup_balanced_pairs
        self.top_rank = top_rank
        self.feature_type = feature_type
        self.parameters = {'tf__max_features': n_features,
                           'tf__min_df': 2}
        if self.vector_space_model == 'idf':
            self.parameters['tfidf__norm'] = norm
        elif self.vector_space_model == 'plm':
            self.parameters['plm__eps'] = eps
            self.parameters['plm__weight'] = weight
            self.parameters['plm__iterations'] = em_iterations
            self.parameters['plm__norm'] = norm

    def vectorize(self, dev_dataset=None, test_dataset=None):
        # initialize:
        self.dev_data, self.dev_titles, self.dev_authors = None, None, None
        self.test_data, self.test_titles, self.test_authors = None, None, None
        self.X_dev, self.X_test = None, None

        # set pipeline:
        pipelines = get_pipelines(self.feature_type)
        transformer = pipelines[self.vector_space_model]
        transformer.set_params(**self.parameters)

        # fit:
        if test_dataset != None:
            self.dev_data, self.dev_titles, self.dev_authors = dev_dataset
            self.test_data, self.test_titles, self.test_authors = test_dataset
            transformer.fit(self.dev_data + self.test_data)
        else:
            self.dev_data, self.dev_titles, self.dev_authors = dev_dataset
            transformer.fit(self.dev_data)

        # transform:
        self.X_dev = Normalizer(norm="l2", copy=False).fit_transform(transformer.transform(self.dev_data))
        logging.info("Development corpus: n_samples=%s / n_features=%s" % (self.X_dev.shape))
        if test_dataset:
            self.X_test = Normalizer(norm="l2", copy=False).fit_transform(transformer.transform(self.test_data))
            logging.info("Test corpus: n_samples=%s / n_features=%s" % (self.X_test.shape))

    def _setup_pairs(self, phase='dev'):
        if phase == "dev":
            titles, authors = self.dev_titles, self.dev_authors
            n_pairs = self.n_dev_pairs
        elif phase == "test":
            titles, authors = self.test_titles, self.test_authors
            n_pairs = self.n_test_pairs

        pairs = []
        for i in range(len(titles)):
            for j in range(len(titles)):
                if i != j:
                    title_i, title_j = titles[i], titles[j]
                    if self.control_pairs:
                        if "_" in title_i and "_" in title_j:
                            if title_i.split("_")[0] == title_j.split('_')[0]:
                                continue
                    pairs.append((i, j))

        self.rnd.shuffle(pairs)
        if n_pairs == None:
            return pairs
        return pairs[:n_pairs]

    def _setup_balanced_pairs(self, phase='dev'):
        if phase == 'dev':
            titles, authors = self.dev_titles, self.dev_authors
            n_pairs = self.n_dev_pairs
        elif phase == 'test':
            titles, authors = self.test_titles, self.test_authors
            n_pairs = self.n_test_pairs

        same_author_pairs, diff_author_pairs = [], []
        for i in range(len(titles)):
            for j in range(len(titles)):
                if i != j:
                    if self.control_pairs:
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

    def compute_distances(self, pairs=[], phase='dev'):
        distances, labels = [], []

        if phase == "dev":
            X, authors = self.X_dev, self.dev_authors
        elif phase == "test":
            X, authors = self.X_test, self.test_authors

        for k, (i, j) in enumerate(pairs):
            logging.info("Verifying pair %s / %s" % (k+1, len(pairs)))
            dist = self.metric(X[i], X[j])
            assert not (np.isnan(dist) or np.isinf(dist))
            distances.append(dist)
            labels.append("same_author" if authors[i] == authors[j] else "diff_author")

        return distances, labels

    def compute_sigmas(self, pairs=[], phase='dev', filter_imposters=False):
        # get the right right data to test on and sample imposters from:
        if phase == "test":
            test_X, test_authors, test_titles = self.X_test, self.test_authors, self.test_titles
            background_X, background_authors, background_titles = self.X_dev, self.dev_authors, self.dev_titles
        elif phase == "dev":
            test_X, test_authors, test_titles = self.X_dev, self.dev_authors, self.dev_titles
            if self.X_test != None:
                background_X, background_authors, background_titles = self.X_test, self.test_authors, self.test_titles
            else:
                background_X, background_authors, background_titles = test_X, test_authors, test_titles

        # start the imposter-based verification:
        sigmas, labels = [], []
        for k, (i, j) in enumerate(pairs):
            logging.info("Verifying pair %s / %s" % (k+1, len(pairs)))
            author_i, author_j = test_authors[i], test_authors[j]

            title_i, title_j = test_titles[i], test_titles[j]
            logging.debug("pair: "+author_i +" / "+title_i+
                " vs "+author_j+" / "+title_j)
            # first, select n_potential_imposters
            background_sims = []

            for b in range(background_X.shape[0]):
                if filter_imposters:
                    if background_authors[b] in (author_i, author_j):
                        continue
                if "_" in background_titles[b]:
                    xt = background_titles[b].split('_')[0]
                    if "_" in test_titles[i]:
                        t1 = test_titles[i].split("_")[0]
                        if t1 == xt:
                            continue
                    if "_" in test_titles[j]:
                        t2 = test_titles[j].split("_")[0]
                        if t2 == xt:
                            continue
                background_sims.append((b, background_authors[b],
                                       self.metric(test_X[i], background_X[b])))

            background_sims.sort(key=lambda sim: sim[-1])
            indexes, imposters, _ = zip(*background_sims[:self.n_potential_imposters])
            X_imposters = background_X[list(indexes), :]

            logging.debug(str([background_titles[f] for f in indexes]))

            # start the iteration for the pair:
            targets = 0.0
            for iteration in range(self.sample_iterations):

                # randomly select actual imposters, if required:
                if self.sample_authors:
                    rnd_imposters = self.rnd.randint(X_imposters.shape[0],
                                                     size=self.n_actual_imposters)
                    X_truncated = X_imposters[rnd_imposters, :]
                else:
                    X_truncated = X_imposters

                # randomly select features, if required:
                if self.sample_features:
                    rnd_features = self.rnd.randint(X_truncated.shape[1],
                                                     size=self.random_prop)
                else:
                    rnd_features = range(X_truncated.shape[1])

                # get vector of source and target document:
                vec_i, vec_j = test_X[i], test_X[j]

                # compute distance to target doc:
                all_candidates = [self.metric(vec_i, vec_j, rnd_features)]

                # compute distance to imposters:
                all_candidates += [self.metric(vec_i, X_truncated[s], rnd_features)
                                           for s in range(self.n_actual_imposters)]
                all_candidates = np.array(all_candidates)

                # find rank of target doc in ranking:
                rank_target = np.where(all_candidates.argsort() == 0)[0][0] + 1

                # scoring:
                if self.top_rank == 1:
                    # standard rank checking:
                    targets += 1.0 if rank_target == 1 else 0.0
                else:
                    logging.debug("rank target: "+str(rank_target))
                    # or mean reciprocal rank:
                    if rank_target <= (self.top_rank+1):
                        targets += (1.0 / rank_target)

            # append the correct label:
            if author_i == author_j:
                labels.append("same_author")
            else:
                labels.append("diff_author")

            # append the sigma as a distance measure (1 - sigma)
            sigma = 1 - (targets / self.sample_iterations)
            #logging.debug("sigma: "+str(sigma))
            sigmas.append(sigma)

        return sigmas, labels

    def _verification_without_sampling(self, dev_pairs=[], test_pairs=None):
        if test_pairs == None:
            # compute distances plain distances between pairs without sampling:
            dev_dists, dev_labels = self.compute_distances(pairs=dev_pairs, phase="dev")
            # scale the dev and test distances together:
            distances = dev_dists
            min_dist, max_dist = min(distances), max(distances)
            scale = lambda d: (d - min_dist) / (max_dist - min_dist)
            dev_scores = zip(dev_labels, map(scale, dev_dists))
            return dev_scores
        else:
            dev_dists, dev_labels = self.compute_distances(dev_pairs, phase="dev")
            test_dists, test_labels = self.compute_distances(test_pairs, phase="test")
            # scale the dev and test distances together:
            distances = dev_dists + test_dists
            min_dist, max_dist = min(distances), max(distances)
            scale = lambda d: (d - min_dist) / (max_dist - min_dist)
            dev_scores = zip(dev_labels, map(scale, dev_dists))
            test_scores = zip(test_labels, map(scale, test_dists))
            return dev_scores, test_scores

    def _verification_with_sampling(self, dev_pairs=[], test_pairs=None, filter_imposters=False):
        if test_pairs == None:
            dev_sigmas, dev_labels = self.compute_sigmas(pairs=dev_pairs, phase="dev",
                                                         filter_imposters=filter_imposters)
            dev_scores = zip(dev_labels, dev_sigmas)
            return dev_scores
        else:
            dev_sigmas, dev_labels = self.compute_sigmas(pairs=dev_pairs, phase="dev",
                                                         filter_imposters=filter_imposters)
            test_sigmas, test_labels = self.compute_sigmas(pairs=test_pairs, phase="test",
                                                           filter_imposters=filter_imposters)
            dev_scores = zip(dev_labels, dev_sigmas)
            test_scores = zip(test_labels, test_sigmas)
            return dev_scores, test_scores

    def get_distance_table(self, dists, pairs, phase):
        if phase == 'dev':
            titles, authors = self.dev_titles, self.dev_authors
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
        # save:
        df = df.set_index("id")
        df.to_csv("../plots/"+phase+"_dists.txt", sep="\t", encoding="utf-8")
        return df

    def fit(self, filter_imposters = False):
        "Start the verification procedure."
        self.dev_pairs = self._setup_pairs(phase="dev")
        
        if self.sample_authors or self.sample_features:
            self.dev_scores = self._verification_with_sampling(dev_pairs=self.dev_pairs,\
                                                       filter_imposters=filter_imposters)
        else:
            self.dev_scores = self._verification_without_sampling(self.dev_pairs)

        self.dev_dists = [sc for _, sc in self.dev_scores]
        return self.dev_scores


    def predict(self, filter_imposters = False):
        "Start the verification procedure."
        self.dev_pairs = self._setup_pairs(phase = "dev")
        self.test_pairs = self._setup_pairs(phase = "test")

        if self.sample_authors or self.sample_features:
            self.dev_scores, self.test_scores = self._verification_with_sampling(
                                                    dev_pairs = self.dev_pairs,
                                                    test_pairs = self.test_pairs,
                                                    filter_imposters = filter_imposters)
        else:
            self.dev_scores, self.test_scores = self._verification_without_sampling(
                                                    dev_pairs = self.dev_pairs,
                                                    test_pairs = self.test_pairs)

        self.dev_dists = [sc for _, sc in self.dev_scores]
        self.test_dists = [sc for _, sc in self.test_scores]
        return (self.dev_scores, self.test_scores)


