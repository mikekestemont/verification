import argparse
import sys
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from verification import Verification, prepare_corpus

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--imposters", dest="imposters", type=int, default=10)
parser.add_argument("-f", "--features", dest="n_features", type=int, default=1000)
parser.add_argument("-d", "--directory", dest="directory", default='../data/english')

args = parser.parse_args()

verification = Verification(imposters=args.imposters, n_features=args.n_features)
print verification
dataset = prepare_corpus(args.directory)
verification.fit(dataset)
scores = verification.verify(dataset)

dm = 1 - scores
np.fill_diagonal(dm, 0.0)
_, titles, authors = dataset
labels = [str(a) + '-' + t[:30] + '...' for a, t in zip(authors, titles)]
pd.DataFrame(dm, columns=labels).to_csv("../dm.csv", index=False)

Z = linkage(squareform(dm), method='average')

dendrogram(Z, labels=labels, orientation='right')