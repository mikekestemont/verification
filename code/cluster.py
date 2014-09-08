
import numpy as np

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


from verification import Verification, prepare_corpus

verification = Verification(imposters=25, n_features=50000)
print verification
dataset = prepare_corpus('../data/english')
verification.fit(dataset)
scores = verification.verify(dataset)

dm = 1 - scores
np.fill_diagonal(dm, 0.0)

_, titles, authors = dataset

Z = linkage(squareform(dm), method='average')

import matplotlib.pyplot as plt
dendrogram(Z, labels=[t[:40] + '...' for t in authors], orientation='right')
plt.show()