import sys
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from verification import Verification, prepare_corpus

verification = Verification(imposters=20, n_features=10000)
print verification
dataset = prepare_corpus(sys.argv[1])
verification.fit(dataset)
scores = verification.verify(dataset)

dm = 1 - scores
np.fill_diagonal(dm, 0.0)
_, titles, authors = dataset
pd.DataFrame(dm, columns=authors).to_csv("../dm.csv", index=False)

Z = linkage(squareform(dm), method='average')

dendrogram(Z, labels=[str(a) + '-' + t[:30] + '...' for a, t in zip(authors, titles)], orientation='right')