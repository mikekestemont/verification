import numpy as np

from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.metrics import average_precision_score, auc_score, precision_score
from sklearn.metrics import recall_score

def one_error(results):
    y_true = np.array([0 if l == "diff_author" else 1 for l, _ in results])
    scores = np.array([score for _, score in results])
    return y_true[scores.argmax()]

def evaluate(results, beta=2):
    y_true = np.array([0 if l == "diff_author" else 1 for l, _ in results])
    scores = np.array([score for _, score in results])
    scores = 1 - scores # highest similarity highest in rank
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f_scores = beta * (precisions * recalls) / (precisions + recalls)
    return f_scores, precisions, recalls, thresholds

def evaluate_with_threshold(results, t, beta=2):
    y_true = np.array([0 if l == "diff_author" else 1 for l, _ in results])
    preds = np.array([1 if score <= t else 0 for _, score in results])
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f_score = beta * (precision * recall) / (precision + recall)
    return f_score, precision, recall
