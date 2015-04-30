"""Baseline experiment on modern corpora. Used to generate Figs. 1.x and Tables 2.x """

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.ERROR)

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sb
import pandas as pd
import numpy as np

from verification.verification import Verification
from verification.evaluation import evaluate
from verification.preprocessing import prepare_corpus, split_corpus

random_state = 1000
data_path = "../data/"
corpus = "caesar_dev"
n_dev_pairs = 500
n_features = 10000

logging.info("preparing corpus")
X_dev, X_test = split_corpus(prepare_corpus(data_path+corpus), controlled="authors", random_state=random_state)

print corpus
vsms = ('std', 'tf', 'tfidf')
dms = ('minmax', 'euclidean', 'cityblock')
ftypes = ('chars', 'words')

for ftype in ftypes:
    # first baseline:
    df = pd.DataFrame(columns=["vector space model"]+list(dms))

    for vsm_cnt, vsm in enumerate(vsms):
        print("\t+ "+vsm)
        fscore_row = [vsm]
        for dm_cnt, dm in enumerate(dms):
            print("\t\t* "+dm)
            verifier = Verification(random_state=random_state,
                                    metric=dm,
                                    feature_type=ftype,
                                    sample_authors=False,
                                    sample_features=False,
                                    n_features=n_features,
                                    n_test_pairs=0,
                                    ngram_range=4,
                                    n_dev_pairs=n_dev_pairs,
                                    vector_space_model=vsm,
                                    balanced_pairs=True)
            logging.info("Starting verification [dev / test]")
            verifier.vectorize(X_dev, X_test)
            dev_results = verifier.fit()

            logging.info("Computing results")
            dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
            max_f = np.nanmax(dev_f)
            print "\t\t\t + F1: "+str(max_f)
            fscore_row.append(format(max_f*100, '.1f'))
            
        df.loc[vsm_cnt] = fscore_row

    # take care of the tables:
    df = df.set_index("vector space model")
    # row and col names:
    df.columns.name = "distance metric"
    df.index.name = "distance metric"
    df.to_csv("../outputs/"+corpus+"_baseline_"+ftype+".csv")
    print("=== baseline F1-scores: "+ftype+" ===")
    print str(df.to_latex())


### now imposters:
for ftype in ftypes:
    df = pd.DataFrame(columns=["vector space model"]+list(dms))

    for vsm_cnt, vsm in enumerate(vsms):
        print("\t+ "+vsm)
        fscore_row = [vsm]
        for dm_cnt, dm in enumerate(dms):
            print("\t\t* "+dm)
            verifier = Verification(random_state=random_state,
                                    metric=dm,
                                    feature_type=ftype,
                                    sample_authors=True,
                                    sample_iterations=100,
                                    sample_features=True,
                                    ngram_range=4,
                                    n_features=n_features,
                                    random_prop=0.5,
                                    n_test_pairs=0,
                                    n_dev_pairs=n_dev_pairs,
                                    vector_space_model=vsm,
                                    n_potential_imposters=100,
                                    n_actual_imposters=10,
                                    top_rank=10,
                                    balanced_pairs=True)
            logging.info("Starting verification [dev / test]")
            verifier.vectorize(X_dev, X_test)
            dev_results = verifier.fit()

            logging.info("Computing results")
            dev_f, dev_p, dev_r, dev_t = evaluate(dev_results)
            max_f = np.nanmax(dev_f)
            print "\t\t\t+ F1: "+str(max_f)
            fscore_row.append(format(max_f*100, '.1f'))

        df.loc[vsm_cnt] = fscore_row

    # take care of the tables:
    df = df.set_index("vector space model")
    # row and col names:
    df.columns.name = "distance metric"
    df.index.name = "distance metric"
    df.to_csv("../outputs/"+corpus+"_imposters_"+ftype+".csv")
    print("=== imposter F1-scores:"+ftype+" ===")
    print str(df.to_latex())
