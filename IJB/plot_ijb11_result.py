import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap

parser = argparse.ArgumentParser(description='show IJB test results.')
parser.add_argument('--root-path', required=True, type=str,
                    help='Data root path.')
parser.add_argument('--target', default='IJBC', type=str,
                    help='target, set to IJBC or IJBB')
parser.add_argument('--results-path', type=str, help='Results root path.')
parser.add_argument('--save-path', type=str, help='path to save result.')

args = parser.parse_args()

if __name__ == '__main__':
    results_path = Path(args.results_path)
    methods = []
    scores = []

    for file in results_path.glob('**/*.npy'):
        methods.append(file.stem)
        scores.append(np.load(file))

    lable_path = os.path.join(args.root_path, args.target, 'meta', 'label.npy')
    label = np.load(lable_path)

    methods = np.array(methods)
    scores = dict(zip(methods, scores))
    colours = dict(zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
    # x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
    fig = plt.figure()
    for method in methods:
        fpr, tpr, _ = roc_curve(label, scores[method])
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        plt.plot(fpr, tpr, color=colours[method], lw=1,
                 label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc * 100)))
        tpr_fpr_row = ["%s-%s" % (method, args.target)]
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            # tpr_fpr_row.append('%.4f' % tpr[min_index])
            tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
        tpr_fpr_table.add_row(tpr_fpr_row)
    plt.xlim([10 ** -6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels)
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on IJB')
    plt.legend(loc="lower right")
    # plt.show()
    fig.savefig(os.path.join(args.save_path, 'results.pdf'))
    print(tpr_fpr_table)
