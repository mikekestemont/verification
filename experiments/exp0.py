""" Experiment 1: Baseline experiment. """

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

import pandas as pd

from verification.verification import Verification
from verification.preprocessing import prepare_corpus, Dataset
from sklearn.cross_validation import train_test_split

data_path = "../data/"
corpora = ["du_essays", "gr_articles", "caesar_background", "sp_articles"]

df = pd.DataFrame(columns=["name", "total words", "unique words", "authors", "docs", "SADPs", "DADPs"])

for corpus in corpora:
    print("=== "+corpus+" ===")
    # prepare data:
    data = prepare_corpus(data_path+corpus)
    # split:
    train_texts, test_texts, train_titles, test_titles, train_authors, test_authors = train_test_split(
        data.texts, data.titles, data.authors, test_size=0.5, random_state=1000)
    train = Dataset(train_texts, train_titles, train_authors)
    test = Dataset(test_texts, test_titles, test_authors)
    verifier = Verification(random_state=1000,
                            n_test_pairs=None,
                            n_train_pairs=None,
                            balanced_pairs=False)
    verifier.fit(train, test)
    # first train data:
    nr_docs = len(train.texts)
    total_nr_words = sum((len(doc) for doc in train.texts))
    unique_words = len(set(sum(train.texts, [])))
    distinct_authors = len(set(train.authors))
    train_pairs = verifier._setup_pairs(phase="train")
    SADPs, DADPs = 0, 0
    for (i, j) in train_pairs:
        if train_authors[i] == train_authors[j]:
            SADPs+=1
        else:
            DADPs+=1
    row = [corpus+" (train)", total_nr_words, unique_words, distinct_authors, nr_docs, SADPs, DADPs]
    df.loc[len(df.index)+1] = row
    # now test:
    nr_docs = len(test.texts)
    total_nr_words = sum((len(doc) for doc in test.texts))
    unique_words = len(set(sum(test.texts, [])))
    distinct_authors = len(set(test.authors))
    test_pairs = verifier._setup_pairs(phase="test")
    SADPs, DADPs = 0, 0
    for (i, j) in test_pairs:
        if test_authors[i] == test_authors[j]:
            SADPs+=1
        else:
            DADPs+=1
    row = [corpus+" (test)", total_nr_words, unique_words, distinct_authors, nr_docs, SADPs, DADPs]
    df.loc[len(df.index)+1] = row

# set indices:
df = df.set_index("name")
print(str(df.to_latex()))
