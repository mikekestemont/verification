import ConfigParser
from verification import *
from multiprocessing import Pool
import numpy as np

config_path = sys.argv[1]
config = ConfigParser.ConfigParser()
config.read(config_path)

def experiment(params):
    (sample, sample_authors, vector_space_model, metric, n_actual_impostors,
     m_potential_impostors, iterations, text_cutoff, n_features, random_prop,
     feature_type, feature_ngram_range, nr_same_author_test_pairs,
     nr_diff_author_test_pairs, random_seed, plm_lambda, plm_iterations) = params
    verification = Verification(sample=sample,
                                sample_authors=sample_authors,
                                vector_space_model=vector_space_model,
                                metric=metric,
                                n_actual_impostors=n_actual_impostors,
                                m_potential_impostors=m_potential_impostors,
                                iterations=iterations,
                                text_cutoff=text_cutoff,
                                n_features=n_features,
                                random_prop=random_prop,
                                feature_type=feature_type,
                                feature_ngram_range=feature_ngram_range,
                                nr_same_author_test_pairs=nr_same_author_test_pairs,
                                nr_diff_author_test_pairs=nr_diff_author_test_pairs,
                                nr_test_pairs=nr_test_pairs,
                                random_seed=random_seed,
                                plm_lambda=plm_lambda,
                                plm_iterations=plm_iterations)

    verification.fit(
        background_dataset=background_dataset, devel_dataset=devel_dataset)
    verification.verify()
    return verification.plot_results() + params


background_dataset_dir = config.get('datasets', 'background_dataset_dir')
devel_dataset_dir = config.get('datasets', 'devel_dataset_dir')
for background, devel in zip(background_dataset_dir.split(','), devel_dataset_dir.split(',')):
    # read the corpus
    background_dataset = prepare_corpus(
        dirname=background, text_cutoff=10000000)
    devel_dataset = prepare_corpus(
        dirname=devel, text_cutoff=10000000)
    sample = map(int, config.get('impostors', 'sample').split(','))
    sample_authors = map(int, config.get('impostors', 'sample_authors').split(','))
    m_potential_impostors = range(
        *map(int, config.get('impostors', 'm_potential_impostors').split(':')))
    n_actual_impostors = range(
        *map(int, config.get('impostors', 'n_actual_impostors').split(':')))
    random_prop = config.getfloat('impostors', 'random_prop')
    iterations = config.getint('impostors', 'iterations')
    metrics = config.get("features", "metric").split(',')
    vector_space_models = config.get("features", "vector_space_model").split(',')
    feature_type = config.get("features", "feature_type")
    feature_ngram_min = config.getint("features", "feature_ngram_min")
    feature_ngram_max = config.getint("features", "feature_ngram_max")
    feature_ngram_range = (feature_ngram_min, feature_ngram_max)
    if feature_type == "char":
        feature_ngram_range = feature_ngram_max
    n_features = range(*map(int, config.get("features", "n_features").split(':')))
    text_cutoff = config.getint("features", "text_cutoff")
    random_seed = config.getint("evaluation", "random_seed")
    nr_same_author_test_pairs = config.getint("evaluation", "nr_same_author_test_pairs")
    nr_diff_author_test_pairs = config.getint("evaluation", "nr_diff_author_test_pairs")
    nr_test_pairs = config.getint("evaluation", "nr_test_pairs")
    plm_lambda = np.arange(*map(float, config.get('plm', 'plm_lambda').split(':')))
    plm_iterations = config.getint('plm', 'plm_iterations')

    params = []
    for sampling in sample:
        if sampling:
            for sampling_authors in sample_authors:
                for p_impostors in m_potential_impostors:
                    for n_impostors in n_actual_impostors:
                        if n_impostors > p_impostors:
                            continue
                        for metric in metrics:
                            for model in vector_space_models:
                                for n_feature in n_features:
                                    if model == "plm":
                                        for Lambda in plm_lambda:
                                            params.append((sampling, sampling_authors, model, metric,
                                                           n_impostors, p_impostors, iterations,
                                                           text_cutoff, n_feature, random_prop,
                                                           feature_type, feature_ngram_range,
                                                           nr_same_author_test_pairs,
                                                           nr_diff_author_test_pairs,
                                                           random_seed, Lambda, plm_iterations))
                                    else:
                                        params.append((sampling, sampling_authors, model, metric,
                                                       n_impostors, p_impostors, iterations,
                                                       text_cutoff, n_feature, random_prop,
                                                       feature_type, feature_ngram_range,
                                                       nr_same_author_test_pairs,
                                                       nr_diff_author_test_pairs,
                                                       random_seed, 0.1, plm_iterations))
        else:
            for metric in metrics:
                for model in vector_space_models:
                    for n_feature in n_features:
                        if model == "plm":
                            for Lambda in plm_lambda:
                                params.append((sampling, False, model, metric,
                                               0, 0, 0, text_cutoff, n_feature, random_prop,
                                               feature_type, feature_ngram_range,
                                               nr_same_author_test_pairs, nr_diff_author_test_pairs,
                                               nr_test_pairs, random_seed, Lambda, plm_iterations))
                        else:
                            params.append((sampling, False, model, metric,
                                           0, 0, 0, text_cutoff, n_feature, random_prop,
                                           feature_type, feature_ngram_range,
                                           nr_same_author_test_pairs, nr_diff_author_test_pairs,
                                           nr_test_pairs, random_seed, 0.0, 0))

    pool = Pool(15)
    results = pool.map(experiment, params, chunksize=len(params) / 15)
    pool.close()
    pool.join()
    with open("results.txt", "a") as outfile:
        for result in results:
            outfile.write('%s\t%s\n' % (background, '\t'.join(map(str, result))))
