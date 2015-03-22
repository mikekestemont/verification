import glob
import os
import re

from collections import namedtuple
from itertools import islice
from gensim.utils import tokenize
from sklearn.cross_validation import train_test_split
import numpy as np

Dataset = namedtuple('Dataset', ['texts', 'titles', 'authors'])

def dummy_author():
    i = 0
    while True:
        yield str(i)
        i += 1

def identity(x):
    return x

DUMMY_AUTHORS = dummy_author()


def ngram_analyzer(words, n=1):
    for word in words:
        if len(word) <= n:
            yield word
        else:
            for i in range(len(word) - n - 1):
                yield word[i:i + n]

def prepare_corpus(dirname, text_cutoff=1000000):
    underscore = re.compile(r'\_')
    authors, titles, texts = [], [], []
    for filename in sorted(glob.glob(dirname + "/*")):
        if '_' in filename:
            author, title = underscore.split(
                os.path.split(filename)[-1].replace(".txt", ""), maxsplit=1)
        else:
            author, title = next(DUMMY_AUTHORS), os.path.basename(filename).replace(".txt", "")
        authors.append(author)
        titles.append(title)
        with open(filename) as infile:
            texts.append(
                list(islice(tokenize(infile.read(), lowercase=True, deacc=True), 0, text_cutoff)))
    return Dataset(texts, titles, authors)

def split_corpus(data, controlled="authors", random_state=10):
    # controlled: has to be None (random division at sample level),
    # "author" (random division at author level) or "title" (random division of text-title pairs)
    dev_texts, dev_authors, dev_titles = [], [], []
    test_texts, test_authors, test_titles = [], [], []
    if controlled == "authors":
        # easy author split:
        authors = list(set(data.authors))
        np.random.RandomState(random_state).shuffle(authors)
        uni_dev_authors, uni_test_authors = authors[:int(len(authors)/2.0)],\
                                            authors[int(len(authors)/2.0):]
        for i, author in enumerate(data.authors):
            if author in uni_dev_authors:
                dev_texts.append(data.texts[i])
                dev_authors.append(data.authors[i])
                dev_titles.append(data.titles[i])
            elif author in uni_test_authors:
                test_texts.append(data.texts[i])
                test_authors.append(data.authors[i])
                test_titles.append(data.titles[i])
    elif controlled == "titles":
        titles_authors = [(author, title.split("_")[0])\
                            for author, title in zip(data.authors, data.titles)]
        titles_authors = list(set(titles_authors))
        np.random.RandomState(random_state).shuffle(titles_authors)
        uni_dev_titles, uni_test_titles = set(titles_authors[:int(len(titles_authors)/2.0)]),\
                                          set(titles_authors[int(len(titles_authors)/2.0):])
        for i, author, title in zip(range(len(data.authors)), data.authors, data.titles):
            title = title.split("_")[0]
            if (author, title) in uni_dev_titles:
                dev_texts.append(data.texts[i])
                dev_authors.append(data.authors[i])
                dev_titles.append(data.titles[i])
            elif (author, title) in uni_test_titles:
                test_texts.append(data.texts[i])
                test_authors.append(data.authors[i])
                test_titles.append(data.titles[i])
    elif controlled == None:
        # difficult author split:
        dev_texts, test_texts, dev_titles, test_titles, dev_authors, test_authors = train_test_split(
            data.texts, data.titles, data.authors, test_size=0.5, random_state=random_state)
    # turn into a data set:
    X_dev = Dataset(dev_texts, dev_titles, dev_authors)
    X_test = Dataset(test_texts, test_titles, test_authors)
    return (X_dev, X_test)
