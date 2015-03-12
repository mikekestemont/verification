import numpy as np
import seaborn as sb
from scipy.cluster.hierarchy import set_link_color_palette


rc = {'axes.labelsize': 3, 'font.size': 3, 'legend.fontsize': 3.0,
      'axes.titlesize': 3, "font.family": "sans-serif",
      'xlabel.major.size': 0.3, 'xlabel.minor.size': 0.3,
      'ylabel.major.size': 0.3, 'ylabel.minor.size': 0.3,
      'font.family': 'Arial', 'font.sans-serif': ['Bitstream Vera Sans']}

sb.set_style("darkgrid", rc=rc)

def prec_recall_curve(scores, filename="prec_rec.pdf", fontsize=7):
    fig = sb.plt.figure()
    sb.plt.xlabel("recall", fontsize=fontsize)
    sb.plt.ylabel("precision", fontsize=fontsize)
    sb.plt.xlim(0, 1); sb.plt.ylim(0, 1)
    _, _, precisions, recalls = zip(*scores)
    sb.plt.plot(recalls, precisions)
    sb.plt.savefig(filename)

def plot_test_densities(results=[], dev_t=None, filename="test_densities.pdf",
                        fontsize=7, title=""):
    fig = sb.plt.figure()
    same_author_densities = np.asarray(
        [sc for c, sc in results if c == "same_author"])
    diff_author_densities = np.asarray(
        [sc for c, sc in results if c == "diff_author"])
    c1, c2, c3 = sb.color_palette("Set1")[:3]
    sb.plt.xlim(0, 1)
    ax = sb.kdeplot(diff_author_densities, shade=True,
               label="Different author pairs", legend=False, c=c1)
    if title:
        sb.plt.title(title)
    sb.kdeplot(same_author_densities, shade=True,
               label="Same author pairs", legend=False, c=c2)
    if dev_t:
        sb.plt.axvline(x=dev_t, linewidth=1, c=c3)
    sb.plt.legend(loc=0)
    sb.plt.savefig(filename)
    sb.plt.clf()

def plot_test_results(f1_scores, thresholds, precisions, recalls,
                      dev_t, filename="test_curve.pdf", fontsize=7):
    fig = sb.plt.figure()
    c1, c2, c3, c4 = sb.color_palette("Set1")[:4]
    sb.plt.plot(thresholds, f1_scores, label="F1 score", c=c1)
    sb.plt.plot(thresholds, precisions, label="Precision", c=c2)
    sb.plt.plot(thresholds, recalls, label="Recall", c=c3)
    sb.plt.xlim(0, 1)
    sb.plt.ylim(0, 1.005)
    sb.plt.axvline(x=dev_t, linewidth=1, c=c4)
    sb.plt.legend(loc=0)
    sb.plt.xlabel('Threshold', fontsize=fontsize)
    sb.plt.ylabel('Score', fontsize=fontsize)
    sb.plt.savefig(filename)
    sb.plt.clf()

def draw_tree(tree_df, label):
    sb.set_palette('Set1', 10, 0.65)
    sb.set_style('white')
    sb.dendrogram(tree_df, rotate=True)
    sb.plt.savefig("../plots/"+label+"_tree.pdf")
    sb.plt.clf()
