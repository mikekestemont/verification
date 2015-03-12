""" Experiment 0: Generate metadata. """

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import pandas as pd

from verification.verification import Verification
from verification.preprocessing import prepare_corpus, split_corpus

data_path = "../data/"
corpora = ["du_essays", "gr_articles", "caesar_dev", "sp_articles"]
random_state = 1000
df = pd.DataFrame(columns=["name", "total words", "unique words", "authors", "docs", "SADPs", "DADPs"])

for corpus in corpora:
    print("=== "+corpus+" ===")
    # prepare data:
    X_dev, X_test = split_corpus(prepare_corpus(data_path+corpus), controlled=True)
    verifier = Verification(random_state=random_state,
                            n_test_pairs=None,
                            n_dev_pairs=None,
                            balanced_pairs=False)
    verifier.fit(X_dev, X_test)
    # first dev data:
    nr_docs = len(X_dev.texts)
    total_nr_words = sum((len(doc) for doc in X_dev.texts))
    unique_words = len(set(sum(X_dev.texts, [])))
    distinct_authors = len(set(X_dev.authors))
    dev_pairs = verifier._setup_pairs(phase="dev")
    SADPs, DADPs = 0, 0
    for (i, j) in dev_pairs:
        if X_dev.authors[i] == X_dev.authors[j]:
            SADPs+=1
        else:
            DADPs+=1
    row = [corpus+" (dev)", total_nr_words, unique_words, distinct_authors, nr_docs, SADPs, DADPs]
    df.loc[len(df.index)+1] = row
    # now test:
    nr_docs = len(X_test.texts)
    total_nr_words = sum((len(doc) for doc in X_test.texts))
    unique_words = len(set(sum(X_test.texts, [])))
    distinct_authors = len(set(X_test.authors))
    test_pairs = verifier._setup_pairs(phase="test")
    SADPs, DADPs = 0, 0
    for (i, j) in test_pairs:
        if X_test.authors[i] == X_test.authors[j]:
            SADPs+=1
        else:
            DADPs+=1
    row = [corpus+" (test)", total_nr_words, unique_words, distinct_authors, nr_docs, SADPs, DADPs]
    df.loc[len(df.index)+1] = row

# set indices:
df = df.set_index("name")
print(str(df.to_latex()))
