""" Generation of metadata on modern corpora. Used for Table 1."""

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

from verification.verification import Verification
from verification.preprocessing import prepare_corpus, split_corpus

data_path = "../data/"
corpora = ["caesar_dev"]
random_state = 1000
df = pd.DataFrame(columns=["name", "total words", "unique words", "authors", "docs", "SADPs", "DADPs"])

for corpus in corpora:
    print("=== "+corpus+" ===")
    # prepare data:
    X_dev, X_test = split_corpus(prepare_corpus(data_path+corpus), controlled="authors", random_state=random_state)
    # first *all* pairs
    verifier = Verification(random_state=random_state,
                            n_test_pairs=None,
                            n_dev_pairs=None,
                            balanced_pairs=False)
    verifier.vectorize(X_dev, X_test)
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
    # now restricted pairs (i.e. balanced + cutoff):
    verifier = Verification(random_state=random_state,
                            n_test_pairs=500,
                            n_dev_pairs=500,
                            balanced_pairs=True)
    verifier.vectorize(X_dev, X_test)
    # dev pairs:
    dev_pairs = verifier._setup_pairs(phase="dev")
    with open("../outputs/"+corpus+"_dev_pairs.tsv", "wt") as F:
        F.write("author_i\ttitle_i\tauthor_i\ttitle_j\tclass\n")
        SADPs, DADPs = 0, 0
        for (i, j) in dev_pairs:
            F.write(X_dev.authors[i]+"\t"+X_dev.titles[i]+"\t")
            F.write(X_dev.authors[j]+"\t"+X_dev.titles[j]+"\t")
            if X_dev.authors[i] == X_dev.authors[j]:
                F.write("SADP\n")
            else:
                F.write("DADP\n")
    # test pairs:
    test_pairs = verifier._setup_pairs(phase="test")
    with open("../outputs/"+corpus+"_test_pairs.tsv", "wt") as F:
        F.write("author_i\ttitle_i\tauthor_i\ttitle_j\tclass\n")
        SADPs, DADPs = 0, 0
        for (i, j) in test_pairs:
            F.write(X_test.authors[i]+"\t"+X_test.titles[i]+"\t")
            F.write(X_test.authors[j]+"\t"+X_test.titles[j]+"\t")
            if X_test.authors[i] == X_test.authors[j]:
                F.write("SADP\n")
            else:
                F.write("DADP\n")
# set indices:
df = df.set_index("name")
# print to console in Latex-format:
print(str(df.to_latex()))
