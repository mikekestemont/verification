#! /usr/bin/env python

import logging
import numpy as np
from heapq import nlargest
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer

old_settings = np.seterr(all='ignore')

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def logsum(x):
    """Computes the sum of x assuming x is in the log domain.

    Returns log(sum(exp(x))) while minimizing the possibility of
    over/underflow.

    Examples
    ========

    >>> import numpy as np
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsum(a)
    9.4586297444267107
    """
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = x.max(axis=0)
    out = np.log(np.sum(np.exp(x - vmax), axis=0))
    out += vmax
    return out

class ParsimoniousLM(BaseEstimator):

    def __init__(self, weight=0.1, iterations=50, eps=1e-5, min_df=1,
                 max_df=1.0, analyzer='word', max_features=None):
        self.weight = np.log(weight)
        self.iterations = iterations
        self.eps = eps
        self.vectorizer = CountVectorizer(
            min_df=min_df, max_df=max_df, analyzer=analyzer,
            max_features=max_features)

    def topK(self, k, document, iterations=50, eps=1e-5):
        ptf = self.lm(document, iterations, eps)
        return nlargest(k, zip(self.vectorizer.get_feature_names(), ptf), lambda tp: tp[1])

    def lm(self, document, iterations, eps):
        tf = self.vectorizer.transform([document]).toarray()[0]
        ptf = np.log(tf > 0) - np.log((tf > 0).sum())
        ptf = self.EM(tf, ptf, iterations, eps)
        return ptf

    def EM(self, tf, ptf, iterations, eps):
        tf = np.log(tf)
        for i in range(1, iterations + 1):
            ptf += self.weight

            E = tf + ptf - np.logaddexp(self.pc, ptf)
            M = E - logsum(E) # np.logaddexp.reduce(E)

            diff = M - ptf
            ptf = M
            if (diff < eps).all():
                break
        return ptf

    def fit(self, X, y=None):
        cf = np.array(self.vectorizer.fit_transform(X).sum(axis=0))[0]
        self.pc = (np.log(cf) - np.log(np.sum(cf))) + np.log(1 - self.weight)
        return self

    def transform(self, X):
        return np.exp(np.array([self.lm(x) for x in X]))

    def cross_entropy(self, qlm, rlm):
        return -np.sum(np.exp(qlm) * np.logaddexp(self.pc, rlm + self.weight))

    def predict_proba(self, query):
        if not hasattr(self, 'fitted_'):
            raise ValueError("No Language Model fitted.")
        for i in range(len(self.fitted_)):
            score = self.cross_entropy(query, self.fitted_[i][1])
            yield self.fitted_[i][0], score


def demo():
    documents = ['er loopt een man op straat', 'de man is vies',
                 'de man heeft een gek hoofd', 'de hele straat kijkt naar de man']
    request = 'de straat is vies'
    # initialize a parsimonious language model
    plm = ParsimoniousLM(documents, 0.1)
    # compute a LM for each document in the document collection
    plm.fit(documents)
    # compute a LM model for the test or request document
    qlm = plm.lm(request, 50, 1e-5)
    # compute the cross-entropy between the LM of the test document and all training document LMs
    # sort by increasing entropy
    print([(documents[i], score) for i, score in sorted(plm.predict_proba(qlm), key=lambda i: i[1])])

if __name__ == '__main__':
    demo()
