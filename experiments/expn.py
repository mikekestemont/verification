""" Experiment 1: Baseline experiment. """

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import seaborn as sb

import pandas as pd

from verification.verification import Verification
from verification.evaluation import evaluate, evaluate_with_threshold
from verification.preprocessing import prepare_corpus, Dataset
from sklearn.cross_validation import train_test_split
import numpy as np

def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


data_path = "../data/"
train = data_path+"caesar_background"
test = train
# we prepare the corpus:
logging.info("preparing corpus")
data = prepare_corpus(train)
train_texts, test_texts, train_titles, test_titles, train_authors, test_authors = train_test_split(
    data.texts, data.titles, data.authors, test_size=0.5, random_state=1000)
X_train = Dataset(train_texts, train_titles, train_authors)
X_test = Dataset(test_texts, test_titles, test_authors)
# we determine the size of the vocabulary
V = len(set(sum(X_train.texts, []) + sum(X_test.texts, [])))
# we define the intervals which which to increase the top-n features (MFW)
print("baseline!")
verifier = Verification(random_state=1000,
                        metric="minmax",
                        sample_authors=False,
                        sample_features=False,
                        n_features=5000,
                        n_test_pairs=500,
                        n_train_pairs=500,
                        em_iterations=100,
                        vector_space_model="std",
                        weight=0.2,
                        eps=0.01,
                        norm="l2",
                        balanced_pairs=True)
logging.info("Starting verification [train / test]")
verifier.fit(X_train, X_test)
results, test_results = verifier.verify()
logging.info("Computing results")
train_f, train_p, train_r, train_t = evaluate(results)
best_t = train_t[np.nanargmax(train_f)]
print "best train F1: "+str(np.nanmax(train_f))
test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
print "best test F1: "+str(test_f)
vsms = ('std', 'plm', 'tf', 'idf')
dms = ('minmax', 'euclidean', 'cityblock')
for vsm in vsms:
      print vsm
      for dm in dms:
            print dm
            print("====================\nimpostors!")
            verifier = Verification(random_state=1000,
                                    metric=dm,
                                    sample_authors=True,
                                    sample_features=True,
                                    n_features=20000,
                                    n_test_pairs=500,
                                    n_train_pairs=500,
                                    em_iterations=100,
                                    random_prop=0.5,
                                    sample_iterations = 100,
                                    vector_space_model=vsm,
                                    n_potential_imposters=60,
                                    n_actual_imposters=10,
                                    weight=0.2,
                                    top_rank=10,
                                    eps=0.01,
                                    norm="l2",
                                    balanced_pairs=True)
            logging.info("Starting verification [train / test]")
            verifier.fit(X_train, X_test)
            results, test_results = verifier.verify()
            logging.info("Computing results")
            train_f, train_p, train_r, train_t = evaluate(results)
            best_t = train_t[np.nanargmax(train_f)]
            print "best train F1: "+str(np.nanmax(train_f))
            test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
            print "best test F1: "+str(test_f)

