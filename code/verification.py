import glob
import logging
import os
import random
import sys
import math
import re
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
from scipy.spatial.distance import cosine, euclidean

# for (remote) plotting:
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

# set seeds:
rnd = random.Random(1066)
np.random.seed(1302)

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
    from distances import min_max
    from distances import burrows_delta
    from distances import normalized_distance
    from distances import manhattan
except:
    # if that fails, fall back to the numba-version (which is still pretty fast)
    from numba import jit

    # minmax by Koppel & Winter
    @jit('float64(float64[:],float64[:])')
    def min_max(a, b):
        mins = 0.0
        maxs = 0.0
        for i in range(a.shape[0]):
            mins += min(a[i], b[i])
            maxs += max(a[i], b[i])
        return mins/maxs

    # Burrows's delta:
    @jit('float64(float64[:],float64[:],float64[:])')
    def burrows_delta(a, b, weights):
        delta = 0.0
        for i in range(a.shape[0]):
            delta+= abs(a[i]-b[i])/weights[i]
        return delta

    @jit('float64(float64[:],float64[:],int8)')
    def normalized_distance(a, b, cutoff):
        numerator = 0.0
        for i in range(len(a)):
            term_a = 2*(a[i]-b[i])
            term_b = a[i]+b[i]
            if term_b:
                update = math.pow(term_a/term_b)
                if not math.isnan(update):
                    numerator+=update
        denominator=4.0*cutoff
        return numerator/denominator

    # Manhattan city block distance:
    @jit('float64(float64[:],float64[:])')
    def manhattan(a, b):
        distance = 0.0
        for i in range(a.shape[0]):
            distance+= abs(a[i]-b[i])
        return distance

def prepare_corpus(dirname, text_cutoff):
    underscore = re.compile(r'\_')
    authors, titles, texts = [], [], []
    for filename in sorted(glob.glob(dirname + "/*.txt")):
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
    def __init__(self, n_features, random_prop, sample, metric, text_cutoff,
                 n_actual_imposters, iterations, nr_test_pairs, use_idf,
                 feature_type, feature_ngram_range, m_potential_imposters,
                 nr_same_author_test_pairs, nr_diff_author_test_pairs):
        self.sample = sample
        assert metric in ("stamatatos", "burrows", "cv_delta", "minmax", "manhattan", "cosine", "euclidean")
        self.metric = metric
        self.use_idf = use_idf
        self.n_features = n_features
        self.rand_features = int(random_prop * n_features)
        self.n_actual_imposters = n_actual_imposters
        self.m_potential_imposters = m_potential_imposters
        self.iterations = iterations
        assert feature_type in ("word", "char")
        self.feature_type = feature_type
        self.feature_ngram_range = feature_ngram_range
        self.nr_same_author_test_pairs = nr_same_author_test_pairs
        self.nr_diff_author_test_pairs = nr_diff_author_test_pairs
        self.nr_test_pairs = nr_test_pairs
        self.text_cutoff = text_cutoff

    def fit(self, background_dataset, devel_dataset):
        """
        Feature selection + tf-idf calculation on dataset
        """
        logging.info("Fitting model.")
        self.background_dataset = background_dataset
        self.devel_dataset = devel_dataset
        # unpack:
        background_texts, devel_texts = self.background_dataset.texts, self.devel_dataset.texts
        # fit:
        if not self.use_idf:
            if self.feature_type == "char":
                self.analyzer = partial(analyzer, n=self.feature_ngram_range)
                self.vectorizer = TfidfVectorizer(analyzer=self.analyzer, max_features=self.n_features, use_idf=False)
            elif self.feature_type == "word":
                self.vectorizer = TfidfVectorizer(analyzer=identity, ngram_range=self.feature_ngram_range, max_features=self.n_features, use_idf=False)
            # fit vectorizer (with truncated vocabulary) on background corpus:
            self.X_background = self.vectorizer.fit_transform(background_texts).toarray()
            # apply vectorizer to devel corpus (get matrix of unnormalized relative term frequencies)
            self.X_devel = self.vectorizer.transform(devel_texts).toarray()
            if self.metric == "burrows":
                # extract std-weights from background texts:
                self.delta_weights = StandardScaler().fit(self.X_background).std_
            elif self.metric == "cv_delta":
                # extract std-weights from background texts:
                self.delta_weights = StandardScaler().fit(self.X_background).std_
        else:
            if self.feature_type == "char":
                self.analyzer = partial(analyzer, n=self.feature_ngram_range)
                self.vectorizer = CountVectorizer(analyzer=self.analyzer)
            elif self.feature_type == "word":
                self.vectorizer = CountVectorizer(analyzer=identity, ngram_range=self.feature_ngram_range)
            # temporarily join both sets to determine feature universe:
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
            if self.metric == "burrows":
                # extract std-weights from background texts:
                self.delta_weights = StandardScaler().fit(self.X_background).std_
            elif self.metric == "cv_delta":
                scaler = StandardScaler().fit(plain_freq_X)
                self.delta_weights = scaler.std_/scaler.mean_
        
        return self

    def plot_weight_properties(self):
        logging.info("Calculating weight properties.")
        # get delta weights:
        self.background_dataset = background_dataset
        self.devel_dataset = devel_dataset
        # unpack:
        all_texts = self.background_dataset.texts+self.devel_dataset.texts
        tmp_analyzer, tmp_vectorizer = None, None
        if self.feature_type == "char":
            tmp_analyzer = partial(analyzer, n=self.feature_ngram_range)
            tmp_vectorizer = TfidfVectorizer(analyzer=self.analyzer, max_features=self.n_features, use_idf=False)
        elif self.feature_type == "word":
            tmp_vectorizer = TfidfVectorizer(analyzer=identity, ngram_range=self.feature_ngram_range, max_features=self.n_features, use_idf=False)
        plain_freq_X = tmp_vectorizer.fit_transform(all_texts).toarray()
        frequency_mass = plain_freq_X.sum(axis=0)
        scaler = StandardScaler().fit(plain_freq_X)
        tmp_delta_weights = scaler.std_
        tmp_cv_weights = scaler.std_/scaler.mean_
        tmp_idf_weights = TfidfTransformer().fit(plain_freq_X).idf_
        properties = list(zip(frequency_mass, tmp_delta_weights, tmp_cv_weights, tmp_idf_weights))
        properties.sort(key=itemgetter(0), reverse=True)
        properties = list(zip(*properties))
        # add int for ranking:
        properties.append(list(range(1,len(properties[0])+1)))
        if not os.path.isdir("plots"):
            os.mkdir("plots")
        # set seaborn params:
        rc={'axes.labelsize': 3, 'font.size': 3, 'legend.fontsize': 3.0, 'axes.titlesize': 3, "font.family": "sans-serif",
        'xlabel.major.size': 0.3, 'xlabel.minor.size': 0.3, 'ylabel.major.size': 0.3, 'ylabel.minor.size': 0.3,
        'font.family': 'Arial', 'font.sans-serif': ['Bitstream Vera Sans'],}
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
        sns.plt.ylabel('Coefficient of variation', fontsize=7)
        sns.plt.plot(properties[-1], properties[2])
        sns.plt.savefig("plots/cv.pdf")
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
        same_author_pairs, diff_author_pairs = [], []
        for i in range(n_devel_samples):
            for j in range(n_devel_samples):
                # don't pair identical samples:
                if i == j:
                    continue
                vec_i, title_i, author_i = self.X_devel[i], devel_titles[i], devel_authors[i]
                vec_j, title_j, author_j = self.X_devel[j], devel_titles[j], devel_authors[j]
                # don't pair samples from the same text:
                if title_i.split("_")[0] == title_j.split("_")[0]:
                    continue
                if author_i == author_j:
                    same_author_pairs.append((i, j))
                else:
                    diff_author_pairs.append((i, j))
        # if nr_test_pairs is specified, randomly select n same_author_pairs and n diff_author_pairs:
        self.test_pairs = []
        if self.nr_test_pairs:
            # randomly select n pairs from all pairs 
            self.test_pairs = same_author_pairs+diff_author_pairs
            rnd.shuffle(self.test_pairs)
            self.test_pairs = rnd.sample(self.test_pairs, self.nr_test_pairs)
        elif self.nr_same_author_test_pairs and self.nr_diff_author_test_pairs:
            # randomly select n different author pairs and m same author pairs
            rnd.shuffle(same_author_pairs)
            rnd.shuffle(diff_author_pairs)
            same_author_pairs = rnd.sample(same_author_pairs, self.nr_same_author_test_pairs)
            diff_author_pairs = rnd.sample(diff_author_pairs, self.nr_diff_author_test_pairs)
            self.test_pairs = same_author_pairs+diff_author_pairs
        else:
            self.test_pairs = same_author_pairs+diff_author_pairs
        # initialize score list:
        self.scores = []
        if not self.sample:
            distances, labels = [], []
            # verify each pair:
            for i, j in (self.test_pairs):
                vec_i, title_i, author_i = self.X_devel[i], devel_titles[i], devel_authors[i]
                vec_j, title_j, author_j = self.X_devel[j], devel_titles[j], devel_authors[j]
                if self.metric in ("burrows", "cv_delta"):
                    dist = burrows_delta(vec_i, vec_j, self.delta_weights)
                elif self.metric == "stamatatos":
                    dist = normalized_distance(vec_i, vec_j, self.text_cutoff)
                elif self.metric == "minmax":
                    dist = min_max(vec_i, vec_j)
                elif self.metric == "manhattan":
                    dist = manhattan(vec_i, vec_j)
                elif self.metric == "cosine":
                    dist = cosine(vec_i, vec_j)
                elif self.metric == "euclidean":
                    dist = euclidean(vec_i, vec_j)
                distances.append(dist)
                if author_i == author_j:
                    labels.append("same_author")
                else:
                    labels.append("diff_author")
            for dist, label in zip(distances, labels):
                score = (dist-min(distances))/(max(distances)-min(distances))
                self.scores.append((label, score))
            for index, item in enumerate(self.test_pairs):
                i, j = item
                vec_i, title_i, author_i = self.X_devel[i], devel_titles[i], devel_authors[i]
                vec_j, title_j, author_j = self.X_devel[j], devel_titles[j], devel_authors[j]
                logging.info("Distance for %s (%s) - %s (%s) = %.3f" % (title_i, author_i, title_j, author_j, self.scores[index][1]))
        else:
            # verify each pair:
            for i, j in (self.test_pairs):
                vec_i, title_i, author_i = self.X_devel[i], devel_titles[i], devel_authors[i]
                vec_j, title_j, author_j = self.X_devel[j], devel_titles[j], devel_authors[j]
                logging.info("Predicting scores for %s - %s" % (devel_authors[i], devel_authors[j]))
                # get impostors from the background corpus:
                background_similarities = []
                for k in range(n_background_samples):
                    background_author = background_authors[k]
                    # make sure the background corpus isn't polluted (this step is supervised...):
                    if background_author in (author_i, author_j):
                        continue
                    if self.metric in ("burrows", "cv_delta"):
                        background_similarities.append((k, background_author, burrows_delta(vec_i, self.X_background[k], self.delta_weights)))
                    elif self.metric == "stamatatos":
                        background_similarities.append((k, background_author, normalized_distance(vec_i, self.X_background[k], self.text_cutoff)))
                    elif self.metric == "manhattan":
                        background_similarities.append((k, background_author, manhattan(vec_i, self.X_background[k])))
                    elif self.metric == "cosine":
                        background_similarities.append((k, background_author, cosine(vec_i, self.X_background[k])))
                    elif self.metric == "euclidean":
                        background_similarities.append((k, background_author, euclidean(vec_i, self.X_background[k])))
                    elif self.metric == "minmax":
                        background_similarities.append((k, background_author, min_max(vec_i, self.X_background[k])))
                if self.metric == "minmax":
                    background_similarities.sort(key=lambda s:s[-1], reverse=True)
                else:
                    background_similarities.sort(key=lambda s:s[-1], reverse=False)
                # select m potential imposters
                m_indexes, m_imposters, _ = zip(*background_similarities[:self.m_potential_imposters])
                m_X = self.X_background[list(m_indexes)]
                # start the verification sampling:
                targets = 0.0
                sigmas = np.zeros(self.iterations)
                # randomly select n_actual_impostors from m_potential_imposters:
                rand_imposter_indices = np.random.randint(0, m_X.shape[0], size=self.n_actual_imposters)
                truncated_X = m_X[rand_imposter_indices,:]
                for k in range(self.iterations):
                    # select random features:
                    rand_feat_indices = np.random.randint(0, truncated_X.shape[1], size=self.rand_features)
                    truncated_X = truncated_X[:,rand_feat_indices]
                    most_similar = None
                    vec_i_trunc, vec_j_trunk = vec_i[rand_feat_indices], vec_j[rand_feat_indices]
                    for idx in range(n_actual_imposters):
                        if self.metric in ("burrows", "cv_delta"):
                            score = burrows_delta(truncated_X[idx], vec_i_trunc, self.delta_weights)
                            if most_similar is None or score < most_similar:
                                most_similar = score
                        elif self.metric == "stamatatos":
                            score = normalized_distance(truncated_X[idx], vec_i_trunc, self.text_cutoff)
                            if most_similar is None or score < most_similar:
                                most_similar = score
                        elif self.metric == "minmax":
                            score = min_max(truncated_X[idx], vec_i_trunc)
                            if most_similar is None or score > most_similar:
                                most_similar = score
                        elif self.metric == "manhattan":
                            score = manhattan(truncated_X[idx], vec_i_trunc)
                            if most_similar is None or score < most_similar:
                                most_similar = score
                        elif self.metric == "cosine":
                            score = cosine(truncated_X[idx], vec_i_trunc)
                            if most_similar is None or score < most_similar:
                                most_similar = score
                        elif self.metric == "euclidean":
                            score = euclidean(truncated_X[idx], vec_i_trunc)
                            if most_similar is None or score < most_similar:
                                most_similar = score
                    if self.metric == "minmax":
                        if min_max(vec_i_trunc, vec_j_trunk) > most_similar:
                            targets+=1.0
                    elif self.metric == "stamatatos":
                        if normalized_distance(vec_i_trunc, vec_j_trunk, self.text_cutoff) < most_similar:
                            targets+=1.0
                    elif self.metric in ("burrows", "cv_delta"):
                        if burrows_delta(vec_i_trunc, vec_j_trunk, self.delta_weights) < most_similar:
                            targets+=1.0
                    elif self.metric == "manhattan":
                        if manhattan(vec_i_trunc, vec_j_trunk) < most_similar:
                            targets+=1.0
                    elif self.metric == "cosine":
                        if cosine(vec_i_trunc, vec_j_trunk) < most_similar:
                            targets+=1.0
                    elif self.metric == "euclidean":
                        if euclidean(vec_i_trunc, vec_j_trunk) < most_similar:
                            targets+=1.0
                    sigmas[k] = targets/(k+1.0)
                if devel_authors[i] == devel_authors[j]:
                    self.scores.append(("same_author", sigmas.mean()))
                else:
                    self.scores.append(("diff_author", sigmas.mean()))
                logging.info("Sigma for %s (%s) - %s (%s) = %.3f" % (devel_titles[i], devel_authors[i], devel_titles[j], devel_authors[j], sigmas.mean()))
            return
    verify = predict

    def plot_results(self):
        # set param:
        rc={'axes.labelsize': 3, 'font.size': 3, 'legend.fontsize': 3.0, 'axes.titlesize': 3, "font.family": "sans-serif",
        'xlabel.major.size': 0.3, 'xlabel.minor.size': 0.3, 'ylabel.major.size': 0.3, 'ylabel.minor.size': 0.3,
        'font.family': 'Arial', 'font.sans-serif': ['Bitstream Vera Sans'],}
        sns.set_style("darkgrid", rc=rc)
        # first, plot precision-recall curves (for non-zero combinations of precision and recall)
        precisions, recalls, f1_scores = [], [], []
        for threshold in np.arange(0.001, 1.001, 0.001):
            preds, true, = [], []
            for category, score in self.scores:
                if self.metric in ("stamatatos", "burrows", "manhattan", "cosine", "euclidean"):
                    preds.append(1 if score <= threshold else 0)
                else:
                    preds.append(1 if score >= threshold else 0)
                true.append(1 if category == "same_author" else 0)
            try:
                f1 = f1_score(preds, true)
                if f1:
                    f1_scores.append((f1, threshold))
            except:
                pass
            try:
                precision = precision_score(preds, true)
                recall = recall_score(preds, true)
                if precision and recall:
                    precisions.append((precision, threshold))
                    recalls.append((recall, threshold))
            except:
                continue
        # plot precision recall-curve
        sns.set_style("darkgrid", rc=rc)
        sns.plt.xlabel('recall', fontsize=7)
        sns.plt.ylabel('precision', fontsize=7)
        sns.plt.xlim(0, 1)
        sns.plt.ylim(0, 1)
        sns.plt.plot([prec for prec,_ in precisions], [rec for rec,_ in recalls])
        with open(self.metric+".txt", "wt") as F:
        #with open("unigrams.txt", "wt") as F:
            for prec,rec in zip(precisions, recalls):
                F.write(str(prec[0])+"\t"+str(rec[0])+"\n")
        sns.plt.savefig("prec_rec.pdf")
        sns.plt.clf()
        # now plot kernel density estimate, using a gaussian kernel:
        sns.set_style("darkgrid", rc=rc)
        fig, ax_left = sns.plt.subplots()
        same_author_densities = np.asarray([score for cat,score in self.scores if cat == "same_author"])
        diff_author_densities = np.asarray([score for cat,score in self.scores if cat == "diff_author"])
        c1, c2, c3, c4, c5, c6 = sns.color_palette("Set1")[:6]
        sns.plt.xlim(0, 1)
        sns.kdeplot(diff_author_densities, shade=True, label="different author pairs", legend=False, c=c1)
        sns.kdeplot(same_author_densities, shade=True, label="same author pairs", legend=False, c=c2)
        sns.plt.legend(loc=0)
        sns.plt.savefig("densities.pdf")
        sns.plt.clf()
        fig, ax_left = sns.plt.subplots()
        sns.set_style("darkgrid", rc=rc)
        sns.plt.plot([s for _,s in f1_scores], [f for f,_ in f1_scores], label="f1 score", c=c1)
        sns.plt.plot([s for _,s in precisions], [p for p,_ in precisions], label="precision", c=c2)
        sns.plt.plot([s for _,s in recalls], [r for r,_ in recalls], label="recall", c=c3)
        sns.plt.ylim(0, 1)
        # plot best precision:
        best_f1 = max(f1_scores, key=itemgetter(0))
        max_y = sns.plt.axis()[3]
        sns.plt.axvline(x=best_f1[1], linewidth=1, c=c4)
        sns.plt.text(best_f1[1], max_y, "f1: "+str(round(best_f1[0], 2)), rotation=0, fontsize=5)
        print("f1: "+str(round(best_f1[0], 2))+" @thresholds="+str(best_f1[1]))
        sns.plt.legend(loc=0)
        if self.metric == "burrows":
            sns.plt.title("Burrows's Delta (with threshold)")
        if self.metric == "cv_delta":
            sns.plt.title("Coeff. Var. Delta (with threshold)")
        elif self.metric == "stamatatos":
            sns.plt.title("Stamatatos's Normalized Distance (with threshold)")
        elif self.metric == "manhattan":
            sns.plt.title("Manhattan City Block Distance (with threshold)")
        elif self.metric == "cosine":
            sns.plt.title("Cosine Distance (with threshold)")
        elif self.metric == "euclidean":
            sns.plt.title("Euclidean Distance (with threshold)")
        else:
            sns.plt.title('Min-max distance (with threshold)')
        sns.plt.xlabel('threshold', fontsize=7)
        sns.plt.xlim(0, 1)
        sns.plt.savefig("curves.pdf")
        sns.plt.clf()
        return

if __name__ == '__main__':
    sample = False # whether or not to sample from author and features
    metric = "minmax" # ("burrows", "cv_delta", stamatatos" "minmax", "manhattan", "cosine", "euclidean") # (dis)similarity metric to use
    use_idf = True
    m_potential_imposters = 30
    n_actual_imposters = 5
    nr_same_author_test_pairs = 250 # or None, if specified we sample n same_author_pairs and n diff_author_pairs
    nr_diff_author_test_pairs = 1000
    nr_test_pairs = None # nr of randomly selected pairs (both same and diff), or None: all texts will be paired exhaustively
    n_features = 1000
    random_prop = 0.5
    iterations = 100
    text_cutoff = 50000
    feature_type = "word"
    feature_ngram_range = (1,1) #(1,1) # word: 4
    verification = Verification(sample =                    sample,
                                use_idf =                   use_idf,
                                metric =                    metric,
                                n_actual_imposters =        n_actual_imposters,
                                m_potential_imposters =     m_potential_imposters,
                                iterations =                iterations,
                                text_cutoff =               text_cutoff,
                                n_features =                n_features,
                                random_prop =               random_prop,
                                feature_type =              feature_type,
                                feature_ngram_range =       feature_ngram_range,
                                nr_same_author_test_pairs = nr_same_author_test_pairs,
                                nr_diff_author_test_pairs = nr_diff_author_test_pairs,
                                nr_test_pairs =             nr_test_pairs)
    background_dataset = prepare_corpus(dirname=sys.argv[1], text_cutoff=text_cutoff)
    devel_dataset = prepare_corpus(dirname=sys.argv[2], text_cutoff=text_cutoff)
    verification.fit(background_dataset=background_dataset, devel_dataset=devel_dataset)
    verification.plot_weight_properties()
    verification.verify()
    print(verification)
    verification.plot_results()