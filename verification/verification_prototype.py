import glob
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os
import random
import sys
import math
import re
import configparser
from collections import defaultdict, namedtuple
from functools import partial
from itertools import combinations, islice
from operator import itemgetter

import numpy as np
from sklearn import base
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from gensim.utils import tokenize
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.cluster.hierarchy import linkage, dendrogram

# for (remote) plotting:
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from plm import ParsimoniousLM

Dataset = namedtuple('Dataset', ['texts', 'titles', 'authors'])


def dummy_author():
    i = 0
    while True:
        yield str(i)
        i += 1

def identity(x):
    return x

DUMMY_AUTHORS = dummy_author()

try:
    # attempt to load the compiled cython-version of the minmax distance
    # function:
    from distances import minmax
    from distances import divergence
except:
    # if that fails, fall back to the numba-version (which is still pretty
    # fast)
    from numba import jit
    logging.info("USING NUMBA")

    @jit('float64(float64[:],float64[:])')
    def minmax(a, b):
        # minmax (ruzicka) by Koppel & Winter, but distance
        mins = 0.0
        maxs = 0.0
        for i in range(a.shape[0]):
            mins += min(a[i], b[i])
            maxs += max(a[i], b[i])
        return 1-mins/maxs

    @jit('float64(float64[:],float64[:])')
    def divergence(a, b):
        # divergence by keselj, stamatatos, etc.
        dist = 0.0
        for i in range(len(a)):
            term_a = 2 * (a[i] - b[i])
            term_b = a[i] + b[i]
            if term_b:
                update = math.pow(term_a / term_b)
                if not math.isnan(update):
                    dist += update
        return dist

DISTANCE_METRICS = {"divergence": divergence,
                    "minmax": minmax,
                    "manhattan": cityblock,
                    "cosine": cosine,
                    "euclidean": euclidean}

def prepare_corpus(dirname, text_cutoff=10000000):
    underscore = re.compile(r'\_')
    authors, titles, texts = [], [], []
    for filename in sorted(glob.glob(dirname + "/*")):
        logging.info("Reading file: %s" % filename)
        if '_' in filename:
            author, title = underscore.split(
                os.path.split(filename)[-1].replace(".txt", ""), maxsplit=1)
        else:
            author, title = next(DUMMY_AUTHORS), os.path.basename(filename).replace(".txt", "")
        authors.append(author)
        titles.append(title)
        logging.info("Reading: %s" % title)
        with open(filename) as infile:
            texts.append(
                list(islice(tokenize(infile.read(), lowercase=True, deacc=True), 0, text_cutoff)))
    return Dataset(texts, titles, authors)


def analyzer(words, n):
    for word in words:
        if len(word) <= n:
            yield word
        else:
            word = "%" + word + "*"
            for i in range(len(word) - n - 1):
                yield word[i:i + n]


class Verification(base.BaseEstimator):

    def __init__(self, n_features, random_prop, sample, metric, text_cutoff,
                 n_actual_impostors, iterations, nr_test_pairs, vector_space_model,
                 feature_type, feature_ngram_range, m_potential_impostors,
                 nr_same_author_test_pairs, nr_diff_author_test_pairs, random_seed,
                 plm_lambda, plm_iterations, sample_authors):
        self.sample = sample
        if metric not in DISTANCE_METRICS:
            raise ValueError("Metric `%s` is not supported." % metric)
        self.metric = DISTANCE_METRICS[metric]
        if vector_space_model not in ("idf", "tf", "std", "plm"):
            raise ValueError("Vector space model `%s` is not supported." % vector_space_model)
        self.vector_space_model = vector_space_model
        self.plm_lambda = plm_lambda
        self.plm_iterations = plm_iterations
        self.rnd = np.random.RandomState(random_seed)
        self.n_features = n_features
        self.sample_authors = sample_authors
        self.rand_features = int(random_prop * n_features)
        self.n_actual_impostors = n_actual_impostors
        self.m_potential_impostors = m_potential_impostors
        self.iterations = iterations
        if feature_type not in ("word", "char"):
            raise ValueError("Feature type `%s` is not supported." % feature_type)
        self.feature_type = feature_type
        self.feature_ngram_range = feature_ngram_range
        self.nr_same_author_test_pairs = nr_same_author_test_pairs
        self.nr_diff_author_test_pairs = nr_diff_author_test_pairs
        self.nr_test_pairs = nr_test_pairs
        self.text_cutoff = text_cutoff
        self.make_plots = False

    def fit(self, background_dataset, devel_dataset):
        """
        Build vector space model
        """
        logging.info("Fitting model.")
        self.background_dataset = background_dataset
        self.devel_dataset = devel_dataset
        # unpack:
        background_texts, devel_texts = self.background_dataset.texts, self.devel_dataset.texts
        # fit:
        if self.vector_space_model in ("tf", "std"):
            if self.feature_type == "char":
                self.analyzer = partial(analyzer, n=self.feature_ngram_range)
                self.vectorizer = TfidfVectorizer(
                    analyzer=self.analyzer, max_features=self.n_features, use_idf=False)
            elif self.feature_type == "word":
                self.vectorizer = TfidfVectorizer(
                    analyzer=identity, ngram_range=self.feature_ngram_range, max_features=self.n_features, use_idf=False)
            # fit vectorizer (with truncated vocabulary) on background corpus:
            self.X_background = self.vectorizer.fit_transform(
                background_texts).toarray()
            _, n_features = self.X_background.shape
            if n_features < self.n_features:
                self.n_features = n_features
            # apply vectorizer to devel corpus (get matrix of unnormalized
            # relative term frequencies)
            self.X_devel = self.vectorizer.transform(devel_texts).toarray()
            if self.vector_space_model == "std":
                # extract std-weights from background texts:
                delta_weights = StandardScaler().fit(self.X_background).std_
                self.X_background = np.divide(self.X_background, delta_weights)
                self.X_devel = np.divide(self.X_devel, delta_weights)
        elif self.vector_space_model == "idf":
            if self.feature_type == "char":
                self.analyzer = partial(analyzer, n=self.feature_ngram_range)
                self.vectorizer = CountVectorizer(analyzer=self.analyzer)
            elif self.feature_type == "word":
                self.vectorizer = CountVectorizer(
                    analyzer=identity, ngram_range=self.feature_ngram_range)
            # temporarily join both sets to determine feature universe:
            all_texts = background_texts + devel_texts
            X = self.vectorizer.fit_transform(all_texts)
            _, n_features = X.shape
            if n_features < self.n_features:
                self.n_features = n_features
            # select top-frequency features:
            self.most_frequent_feature_indices = np.asarray(
                X.sum(0).argsort())[0][-self.n_features:]
            # now re-vectorize, but use tf-idfs of ngrams:
            X = TfidfTransformer().fit_transform(
                X).toarray()  # no longer sparse!
            # Q: only select most frequent features after tf-idf
            # transformation?
            X = X[:, self.most_frequent_feature_indices]
            # divide the sets again:
            self.X_background = X[:len(background_texts)]
            self.X_devel = X[len(background_texts):]
        elif self.vector_space_model == "plm":
            all_texts = background_texts + devel_texts
            self.plm = ParsimoniousLM(all_texts, self.plm_lambda)
            n_features = self.plm.pc.shape[0]
            if n_features < self.n_features:
                self.n_features = n_features
            self.most_frequent_feature_indices = np.asarray(
                self.plm.vectorizer.transform(all_texts).sum(0).argsort())[0][-self.n_features:]
            self.plm.fit(all_texts, iterations=self.plm_iterations)
            _, models = zip(*self.plm.fitted_)
            self.plm.pc = self.plm.pc[self.most_frequent_feature_indices]
            self.X = np.exp(np.array(models)) # convert to probabilities
            self.X = self.X[:, self.most_frequent_feature_indices]
            self.X_background = self.X[:len(background_texts)]
            self.X_devel = self.X[len(background_texts):]
        return self

    def plot_weight_properties(self):
        # why is this a method of the verification class? It doesn't use any
        # of its function, only variables which can be passed to a seperate
        # function. Perhaps some more general plotting library. ALSO: make sure
        # functions are no longer than 20 lines or else they become rather
        # unreadable
        logging.info("Calculating weight properties.")
        # get delta weights:
        self.background_dataset = background_dataset
        self.devel_dataset = devel_dataset
        # unpack:
        all_texts = self.background_dataset.texts + self.devel_dataset.texts
        tmp_analyzer, tmp_vectorizer = None, None
        if self.feature_type == "char":
            tmp_analyzer = partial(analyzer, n=self.feature_ngram_range)
            tmp_vectorizer = TfidfVectorizer(
                analyzer=self.analyzer, max_features=self.n_features, use_idf=False)
        elif self.feature_type == "word":
            tmp_vectorizer = TfidfVectorizer(
                analyzer=identity, ngram_range=self.feature_ngram_range, max_features=self.n_features, use_idf=False)
        plain_freq_X = tmp_vectorizer.fit_transform(all_texts).toarray()
        frequency_mass = plain_freq_X.sum(axis=0)
        scaler = StandardScaler().fit(plain_freq_X)
        tmp_delta_weights = scaler.std_
        tmp_idf_weights = TfidfTransformer().fit(plain_freq_X).idf_
        properties = list(
            zip(frequency_mass, tmp_delta_weights, tmp_idf_weights))
        properties.sort(key=itemgetter(0), reverse=True)
        properties = list(zip(*properties))
        # add int for ranking:
        properties.append(list(range(1, len(properties[0]) + 1)))
        if not os.path.isdir("plots"):
            os.mkdir("plots")
        # set seaborn params:
        if self.make_plots:
            rc = {'axes.labelsize': 3, 'font.size': 3, 'legend.fontsize': 3.0, 'axes.titlesize': 3, "font.family": "sans-serif",
                  'xlabel.major.size': 0.3, 'xlabel.minor.size': 0.3, 'ylabel.major.size': 0.3, 'ylabel.minor.size': 0.3,
                  'font.family': 'Arial', 'font.sans-serif': ['Bitstream Vera Sans'], }
            sns.set_style("darkgrid", rc=rc)
            sns.plt.xlabel('Rank', fontsize=7)
            sns.plt.ylabel('Frequency mass', fontsize=7)
            sns.plt.plot(properties[-1], properties[0])
            sns.plt.savefig("plots/freq_mass.pdf")
            sns.plt.clf()
            sns.plt.xlabel('Rank', fontsize=7)
            sns.plt.ylabel('Standard deviation', fontsize=7)
            sns.plt.plot(properties[-1], properties[1])
            sns.plt.savefig("plots/delta_std.pdf")
            sns.plt.clf()
            sns.plt.xlabel('Rank', fontsize=7)
            sns.plt.ylabel('If-Idf', fontsize=7)
            sns.plt.plot(properties[-1], properties[2])
            sns.plt.savefig("plots/tfidf.pdf")
            sns.plt.clf()

    def predict(self):
        logging.info("Verification started.")
        _, devel_titles, devel_authors = self.devel_dataset
        _, background_titles, background_authors = self.background_dataset
        n_background_samples = self.X_background.shape[0]
        n_devel_samples = self.X_devel.shape[0]
        # determine the pairs from devel/test which we would like to verify:
        # FK: THIS STEP IS SUPERVISED. DO WE REALLY NEED IT?
        same_author_pairs, diff_author_pairs = [], []
        for i in range(n_devel_samples):
            for j in range(n_devel_samples):
                #print(j)
                # don't pair identical samples:
                if i != j:
                    vec_i, title_i, author_i = self.X_devel[i], devel_titles[i], devel_authors[i]
                    vec_j, title_j, author_j = self.X_devel[j], devel_titles[j], devel_authors[j]
                    # don't pair samples from the same text:
                    if title_i.split("_")[0] != title_j.split("_")[0]:
                        if author_i == author_j:
                            same_author_pairs.append((i, j))
                        else:
                            diff_author_pairs.append((i, j))
        # if nr_test_pairs is specified, randomly select n same_author_pairs
        # and n diff_author_pairs:
        self.test_pairs = []
        if self.nr_test_pairs:
            # randomly select n pairs from all pairs
            self.test_pairs = same_author_pairs + diff_author_pairs
            self.rnd.shuffle(self.test_pairs)
            self.test_pairs = self.test_pairs[:self.nr_test_pairs]
        elif self.nr_same_author_test_pairs and self.nr_diff_author_test_pairs:
            # randomly select n different author pairs and m same author pairs
            self.rnd.shuffle(same_author_pairs)
            self.rnd.shuffle(diff_author_pairs)
            same_author_pairs = same_author_pairs[:self.nr_same_author_test_pairs]
            diff_author_pairs = diff_author_pairs[:self.nr_diff_author_test_pairs]
            self.test_pairs = same_author_pairs + diff_author_pairs
        else:
            self.test_pairs = same_author_pairs + diff_author_pairs
        # initialize score list:
        self.scores = []
        if not self.sample:
            distances, labels = [], []
            # verify each pair:
            for i, j in (self.test_pairs):
                vec_i, title_i, author_i = self.X_devel[i], devel_titles[i], devel_authors[i]
                vec_j, title_j, author_j = self.X_devel[j], devel_titles[j], devel_authors[j]
                distances.append(self.metric(vec_i, vec_j))
                labels.append("same_author" if author_i == author_j else "diff_author")
            r_distances = np.zeros((n_devel_samples, n_devel_samples))
            for dist, label in zip(distances, labels):
                self.scores.append((label, (dist - min(distances)) / (max(distances) - min(distances))))
            for index, item in enumerate(self.test_pairs):
                i, j = item
                _, title_i, author_i = self.X_devel[i], devel_titles[i], devel_authors[i]
                _, title_j, author_j = self.X_devel[j], devel_titles[j], devel_authors[j]
                logging.info("Distance for %s (%s) - %s (%s) = %.3f" %
                             (title_i, author_i, title_j, author_j, self.scores[index][1]))
                r_distances[i,j] = self.scores[index][1]
                r_distances[j,i] = r_distances[i,j]
            for method in ('single', 'complete', 'average', 'ward'):
                plt.figure()
                tree = dendrogram(linkage(r_distances, method=method), labels=devel_titles, orientation="left", leaf_font_size=3)
                plt.savefig("%s-dist.pdf" % method)
        else:
            r_distances = np.zeros((n_devel_samples, n_devel_samples))
            # verify each pair:
            for i, j in (self.test_pairs):
                vec_i, title_i, author_i = self.X_devel[i], devel_titles[i], devel_authors[i]
                vec_j, title_j, author_j = self.X_devel[j], devel_titles[j], devel_authors[j]
                logging.info("Predicting scores for %s - %s" % (author_i, author_j))
                # get impostors from the background corpus:
                background_similarities = []
                for k in range(n_background_samples):
                    background_author = background_authors[k]
                    # make sure the background corpus isn't polluted (this step
                    # is supervised...): #FK: or is this completely unrealistic????
                    if background_author not in (author_i, author_j):
                        background_similarities.append(
                            (k, background_author, self.metric(vec_i, self.X_background[k])))
                background_similarities.sort(key=lambda s: s[-1])

                logging.debug("Background authors: %s" % ', '.join(
                    a for _, a, _ in background_similarities[:self.m_potential_impostors]))

                logging.debug("Test pair: %s, %s" % (author_i, author_j))
                # select m potential impostors # FK THIS IS NOT WHAT YOU ARE DOING... BECAUSE
                # `background_similarities` MAY CONTAIN DUPLICATE AUTHORS!
                m_indexes, m_impostors, _ = zip(*background_similarities[:self.m_potential_impostors])
                m_X = self.X_background[list(m_indexes)]
                # start the verification sampling:
                targets = 0.0
                sigmas = np.zeros(self.iterations)
                ##################### put this inside loop??? #################
                # randomly select n_actual_impostors from
                # m_potential_impostors:
                rand_impostor_indices = self.rnd.randint(
                    0, m_X.shape[0], size=self.n_actual_impostors)
                truncated_X = m_X[rand_impostor_indices, :]
                logging.debug("truncated shape=%s:%s" % truncated_X.shape)
                ###############################################################
                for k in range(self.iterations):
                    if self.sample_authors:
                        rand_imposter_indices = self.rnd.randint(
                            m_X.shape[0], size=self.n_actual_impostors)
                        truncated_X = m_X[rand_imposter_indices, :]
                    # logging.debug("truncated shape=%s:%s" % truncated_X.shape)
                    # select random features:
                    rand_feat_indices = self.rnd.randint(
                        truncated_X.shape[1], size=self.rand_features)
                    truncated_X_rand = truncated_X[:, rand_feat_indices]
    #                logging.debug("random truncated shape=%s:%s" % truncated_X_rand.shape)
                    vec_i_trunc, vec_j_trunk = vec_i[rand_feat_indices], vec_j[rand_feat_indices]
                    most_similar = min(self.metric(vec_i_trunc, truncated_X_rand[idx])
                                       for idx in range(self.n_actual_impostors))
                    target_distance = self.metric(vec_i_trunc, vec_j_trunk)
                    if (author_j == author_i):
                        logging.debug("NN: %s, Target: %s" % (most_similar, target_distance))
                    if target_distance < most_similar:
                        targets += 1.0
                    sigmas[k] = targets / (k + 1.0)
                self.scores.append(("same_author" if author_i == author_j else "diff_author", sigmas.mean()))
                logging.info("Sigma for %s (%s) - %s (%s) = %.3f" % (
                    devel_titles[i], author_i, devel_titles[j], author_j, sigmas.mean()))
                r_distances[i,j] = sigmas.mean()
                r_distances[j,i] = r_distances[i,j]
            tree = dendrogram(linkage(r_distances, method='single'), labels=devel_titles, orientation="left", leaf_font_size=3)
            plt.savefig("dist.pdf")
        return self.scores

    verify = predict

    def plot_results(self):
        # This really doesn't belong to the verification class. It only needs the RESULTS
        # of the verification class. Refactor to the already suggested plotting module.
        # split pairs into dev and test set:
        self.dev_scores = self.scores[:int(len(self.scores)/2)]
        self.test_scores = self.scores[int(len(self.scores)/2):]
        # determine threshold that maximizes F1 on dev scores:
        dev_f1_scores = []
        for threshold in np.arange(0.001, 1.001, 0.001):
            preds, true, = [], []
            for category, score in self.dev_scores:
                if self.sample:
                    preds.append(1 if score >= threshold else 0)
                else:
                    preds.append(1 if score <= threshold else 0)
                true.append(1 if category == "same_author" else 0)
            f1 = f1_score(preds, true)
            dev_f1_scores.append((f1, threshold))
        best_f1_dev = max(dev_f1_scores, key=itemgetter(0))
        # first, plot precision-recall curves (for non-zero combinations of
        # precision and recall)
        precisions, recalls, f1_scores = [], [], []
        f1_test = None
        for threshold in np.arange(0.001, 1.001, 0.001):
            preds, true, = [], []
            for category, score in self.test_scores:
                if self.sample:
                    preds.append(1 if score >= threshold else 0)
                else:
                    preds.append(1 if score <= threshold else 0)
                true.append(1 if category == "same_author" else 0)
            f1 = f1_score(preds, true)
            f1_scores.append((f1, threshold))
            precision = precision_score(preds, true)
            recall = recall_score(preds, true)
            if precision and recall:
                precisions.append((precision, threshold))
                recalls.append((recall, threshold))
            if threshold == best_f1_dev[1]:
                print("Best F1 on dev set = {0[0]}, @ threshold = {0[1]}".format(best_f1_dev))
                print("Test results @ threshold = {1[1]}".format(f1, best_f1_dev))
                print("\tF1: {0}".format(f1))
                print("\tPrecision: {0}".format(precision))
                print("\tRecall: {0}".format(recall))
        # plot precision recall-curve
        # set param:
        if self.make_plots:
            rc = {'axes.labelsize': 3, 'font.size': 3, 'legend.fontsize': 3.0,
                  'axes.titlesize': 3, "font.family": "sans-serif",
                  'xlabel.major.size': 0.3, 'xlabel.minor.size': 0.3, 'ylabel.major.size': 0.3,
                  'ylabel.minor.size': 0.3, 'font.family': 'Arial', 'font.sans-serif': ['Bitstream Vera Sans']}
            sns.set_style("darkgrid", rc=rc)
            sns.set_style("darkgrid", rc=rc)
            sns.plt.xlabel('recall', fontsize=7)
            sns.plt.ylabel('precision', fontsize=7)
            sns.plt.xlim(0, 1)
            sns.plt.ylim(0, 1)
            sns.plt.plot(
                [prec for prec, _ in precisions], [rec for rec, _ in recalls])

        with open(self.metric.__name__ + ".txt", "wt") as F:
            for prec, rec in zip(precisions, recalls):
                F.write(str(prec[0]) + "\t" + str(rec[0]) + "\n")

        if self.make_plots:
            sns.plt.savefig("prec_rec.pdf")
            sns.plt.clf()
            # now plot kernel density estimate, using a gaussian kernel:
            sns.set_style("darkgrid", rc=rc)
            fig, ax_left = sns.plt.subplots()
            same_author_densities = np.asarray(
                [score for cat, score in self.scores if cat == "same_author"])
            diff_author_densities = np.asarray(
                [score for cat, score in self.scores if cat == "diff_author"])
            c1, c2, c3, c4, c5, c6 = sns.color_palette("Set1")[:6]
            sns.plt.xlim(0, 1)
            sns.kdeplot(diff_author_densities, shade=True,
                        label="different author pairs", legend=False, c=c1)
            sns.kdeplot(same_author_densities, shade=True,
                        label="same author pairs", legend=False, c=c2)
            sns.plt.legend(loc=0)
            sns.plt.savefig("densities.pdf")
            sns.plt.clf()
            fig, ax_left = sns.plt.subplots()
            sns.set_style("darkgrid", rc=rc)
            sns.plt.plot(
                [s for _, s in f1_scores], [f for f, _ in f1_scores], label="f1 score", c=c1)
            sns.plt.plot([s for _, s in precisions], [
                p for p, _ in precisions], label="precision", c=c2)
            sns.plt.plot(
                [s for _, s in recalls], [r for r, _ in recalls], label="recall", c=c3)
            sns.plt.ylim(0, 1.005)
            # optimal best_f1 = max(f1_scores, key=itemgetter(0))
            # plot dev F1:
            max_y = sns.plt.axis()[3]
            sns.plt.axvline(x=best_f1_dev[1], linewidth=1, c=c4)
            sns.plt.text(
                best_f1_dev[1], max_y, "f1: " + str(round(best_f1_dev[0], 2)), rotation=0, fontsize=5)
            sns.plt.legend(loc=0)
            sns.plt.title(self.metric.__name__.capitalize())
            sns.plt.xlabel('threshold', fontsize=7)
            sns.plt.xlim(0, 1)
            sns.plt.savefig("curves.pdf")
            sns.plt.clf()
        logging.warn("f1: %s" % f1)
        return best_f1_dev, f1, precision, recall

if __name__ == '__main__':
    # parse config file passed via cmd line:
    config_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read("config.txt")
    # set options
    background_dataset_dir = config.get('datasets', 'background_dataset_dir')
    devel_dataset_dir = config.get('datasets', 'devel_dataset_dir')
    sample = config.getboolean('impostors', 'sample')
    sample_authors = config.getboolean('impostors', 'sample_authors')
    m_potential_impostors = config.getint('impostors', 'm_potential_impostors')
    n_actual_impostors = config.getint('impostors', 'n_actual_impostors')
    random_prop = config.getfloat('impostors', 'random_prop')
    iterations = config.getint('impostors', 'iterations')
    metric = config.get("features", "metric")
    vector_space_model = config.get("features", "vector_space_model")
    feature_type = config.get("features", "feature_type")
    feature_ngram_min = config.getint("features", "feature_ngram_min")
    feature_ngram_max = config.getint("features", "feature_ngram_max")
    feature_ngram_range = (feature_ngram_min, feature_ngram_max)
    if feature_type == "char":
        feature_ngram_range = feature_ngram_max
    n_features = config.getint("features", "n_features")
    text_cutoff = config.getint("features", "text_cutoff")
    vector_space_model = config.get("features", "vector_space_model")
    random_seed = config.getint("evaluation", "random_seed")
    nr_same_author_test_pairs = config.getint("evaluation", "nr_same_author_test_pairs")
    nr_diff_author_test_pairs = config.getint("evaluation", "nr_diff_author_test_pairs")
    nr_test_pairs = config.getint("evaluation", "nr_test_pairs")
    plm_lambda = config.getfloat('plm', 'plm_lambda')
    plm_iterations = config.getint('plm', 'plm_iterations')
    # start the verification
    verification = Verification(sample=sample,
                                sample_authors=sample_authors,
                                vector_space_model=vector_space_model,
                                metric=metric,
                                n_actual_impostors=n_actual_impostors,
                                m_potential_impostors=m_potential_impostors,
                                iterations=iterations,
                                text_cutoff=text_cutoff,
                                n_features=n_features,
                                random_prop=random_prop,
                                feature_type=feature_type,
                                feature_ngram_range=feature_ngram_range,
                                nr_same_author_test_pairs=nr_same_author_test_pairs,
                                nr_diff_author_test_pairs=nr_diff_author_test_pairs,
                                nr_test_pairs=nr_test_pairs,
                                random_seed=random_seed,
                                plm_lambda=plm_lambda,
                                plm_iterations=plm_iterations)
    background_dataset = prepare_corpus(
        dirname=background_dataset_dir, text_cutoff=text_cutoff)
    devel_dataset = prepare_corpus(
        dirname=devel_dataset_dir, text_cutoff=text_cutoff)
    verification.fit(
        background_dataset=background_dataset, devel_dataset=devel_dataset)
    #verification.plot_weight_properties()
    verification.verify()
    print(verification)
    verification.plot_results()