import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sb
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from verification.verification import Verification
from verification.evaluation import evaluate, evaluate_with_threshold
from verification.preprocessing import prepare_corpus, split_corpus

random_state = 1000
data_path = "../data/"
corpus = "du_essays"
n_pairs = 1000
n_features = 5000

logging.info("preparing corpus")
X_dev, X_test = split_corpus(prepare_corpus(data_path+corpus), controlled="authors", random_state=random_state)

verifier = Verification(random_state=random_state,
                        metric="cityblock",
                        feature_type="words",
                        sample_authors=False,
                        sample_features=False,
                        n_features=n_features,
                        n_test_pairs=n_pairs,
                        n_dev_pairs=n_pairs,
                        em_iterations=100,
                        vector_space_model="tf",
                        weight=0.2,
                        eps=0.01,
                        norm="l2",
                        balanced_pairs=True)
logging.info("Starting verification [dev / test]")
verifier.vectorize(X_dev, X_test)
dev_results, test_results = verifier.predict()
logging.info("Computing results")

# first prec rec curve of test results:
test_Fs, test_Ps, test_Rs, test_Ts = evaluate(test_results)

# get max for dev:
dev_Fs, dev_Ps, dev_Rs, dev_Ts = evaluate(test_results)
best_t = dev_Ts[np.nanargmax(dev_Fs)]
baseline_test_f, test_p, test_r = evaluate_with_threshold(test_results, t=best_t)
print "Baseline F1: "+str(baseline_test_f)

dev_pairs = verifier._setup_pairs(phase = "dev")
test_pairs = verifier._setup_pairs(phase = "test")

X_dev, y_dev = [], []
data_dev, authors_dev, titles_dev = verifier.X_dev.toarray(), verifier.dev_authors, verifier.dev_titles
for (i, j) in dev_pairs:
    X_dev.append(data_dev[i])
    X_dev.append(data_dev[j])
    if authors_dev[i] == authors_dev[j]:
        y_dev.append(0.0)
        y_dev.append(0.0)
    else:
        y_dev.append(1.0)
        y_dev.append(1.0)
X_dev, y_dev = np.array(X_dev, dtype='float32'), np.array(y_dev, dtype='float32')

X_test, y_test = [], []
data_test, authors_test, titles_test = verifier.X_test.toarray(), verifier.test_authors, verifier.test_titles
for (i, j) in test_pairs:
      X_test.append(data_test[i])
      X_test.append(data_test[j])
      if authors_test[i] == authors_test[j]:
            y_test.append(0.0)
            y_test.append(0.0)
      else:
            y_test.append(1.0)
            y_test.append(1.0)
X_test, y_test = np.array(X_test, dtype='float32'), np.array(y_test, dtype='float32')

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Scaler
from keras.optimizers import SGD

input_dim = X_dev.shape[1]

from keras.utils import np_utils

print "=============="
print X_dev.shape
print X_test.shape
print y_dev.shape
print y_test.shape
print "=============="


""
model = Sequential()
model.add(Scaler(input_dim, init='uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, l2=0.5)
model.compile(loss='siamese_euclidean', sgd)

model.fit(X_test, y_test, nb_epoch=10000, batch_size=2, validation_split=0.0, show_accuracy=True)
print ":::::::::::"
classes = model.predict_classes(X_test, batch_size=2)
print classes
acc = np_utils.accuracy(classes, y_test)

""" #werkt!
model = Sequential()
model.add(Dense(input_dim, 1000, init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1000, 60, init='uniform'))
model.add(Activation('linear'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='siamese_euclidean', optimizer=sgd)

model.fit(X_dev, y_dev, nb_epoch=10000, batch_size=2, validation_split=0.0, show_accuracy=True)
print ":::::::::::"
classes = model.predict_classes(X_test, batch_size=10)
print classes
acc = np_utils.accuracy(classes, y_test)
print('Test accuracy:', acc)
"""