import numpy as np

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import recall_score, precision_score, f1_score

def evaluate(results, beta=2):
    y_true = np.array([0 if l == "diff_author" else 1 for l, _ in results])
    scores = np.array([score for _, score in results])
    scores = 1 - scores # highest similarity highest in rank
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    precisions, recalls = precisions[:-1], recalls[:-1] # cut off 1
    f_scores = beta * (precisions * recalls) / (precisions + recalls)
    return f_scores, precisions, recalls, thresholds

def evaluate_with_threshold(results, t, beta=2):
    y_true = np.array([0 if l == "diff_author" else 1 for l, _ in results])
    scores = np.array([score for _, score in results])
    scores = 1 - scores # highest similarity 
    preds = scores >= t 
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f_score = beta * (precision * recall) / (precision + recall)
    return f_score, precision, recall

def average_precision_score(results):
    y_true = np.array([0 if l == "diff_author" else 1 for l, _ in results])
    scores = np.array([score for _, score in results])
    scores = 1 - scores # highest similarity highest in rank
    return ap(y_true, scores)

def rank_predict(results, method="proportional", N=None):
    """Use the distance (or similarity) rankings to make a binary prediction.
    There are 3 threshold methods:
       1) proportional: set N proportional to N' in the training data.
       2) calibrated: set N to the exact number of positives in the test data.
       3) break-even-point: return the optimized F-score.
    Note that the latter two make use of information in the test data not available
    in real world applications. Theoretically however, they can show us valuable
    insights about for instance the upper bound performance.
    """
    y_true = np.array([0 if l == "diff_author" else 1 for l, _ in results])
    scores = np.array([score for _, score in results])
    scores = 1 - scores # highest similarity highest in rank
    y_pred = np.zeros(y_true.shape)
    if method == "proportional":
        if N is None:
            raise ValueError("No estimate given of N positive.")
        y_pred[scores.argsort()[::-1][:N]] = 1
    elif method == "calibrated":
        y_pred[scores.argsort()[::-1][:y_true.sum()]] = 1
    elif method == "break-even-point":
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        f_scores = 2 * precisions * recalls / (precisions + recalls)
        index = np.nanargmax(f_scores)
        return f_scores[index], precisions[index], recalls[index]
    else:
        raise ValueError("Unsupported method `%s`" % method)
    return f1_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)
