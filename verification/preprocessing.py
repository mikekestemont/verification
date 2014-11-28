import glob
import os
import re

from collections import namedtuple
from itertools import islice
from gensim.utils import tokenize

Dataset = namedtuple('Dataset', ['texts', 'titles', 'authors'])

def dummy_author():
    i = 0
    while True:
        yield str(i)
        i += 1

def identity(x):
    return x

DUMMY_AUTHORS = dummy_author()


def analyzer(words, n=1):
    for word in words:
        if len(word) <= n:
            yield word
        else:
            word = "%" + word + "*"
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
