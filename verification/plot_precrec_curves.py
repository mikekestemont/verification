# for (remote) plotting:
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

# set param:
rc={'axes.labelsize': 3, 'font.size': 3, 'legend.fontsize': 3.0, 'axes.titlesize': 3, "font.family": "sans-serif",
'xlabel.major.size': 0.3, 'xlabel.minor.size': 0.3, 'ylabel.major.size': 0.3, 'ylabel.minor.size': 0.3,
'font.family': 'Arial', 'font.sans-serif': ['Bitstream Vera Sans'],}
sns.set_style("darkgrid", rc=rc)
sns.plt.xlim(0, 1)
sns.plt.ylim(0, 1)
sns.plt.xlabel('recall', fontsize=7)
sns.plt.ylabel('precision', fontsize=7)

for filename in ("burrows", "stamatatos", "minmax"):
#for filename in ("char", "word"):
    prec_rec = [line.split() for line in open(filename+".txt", "rt")]
    prec = tuple(float(prec) for prec,_ in prec_rec)
    rec = tuple(float(rec) for _,rec in prec_rec)
    sns.plt.plot(prec, rec, label=filename)
sns.plt.legend()
sns.plt.savefig("multiple_prec_rec.pdf")