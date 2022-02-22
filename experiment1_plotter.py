#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laurens Sluijterman
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import klepto
import matplotlib as mpl
matplotlib.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 17
plt.rcParams['axes.linewidth'] = 0.2


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', required=True,
                    help="name of directory of dataset")
args = parser.parse_args()

data_directory = args.dir
_SHELVE_LOCATION = "./Results/" + data_directory + "/raw_results/PICIdata"
_PLOT_LOCATION = "./Results/" + data_directory + "/plots/"


def create_violin(data, labels, alpha, ylabel):
    """Create a violin plot containing either the PICF or CICF.

    Parameters:
        data: A list containing the coverage probabilities for each method
            during each of the simulations
        labels: A list with the names of the methods
        alpha: The significance level of the intervals
        ylabel: Either 'PICF' or "CICF.

    Returns:
        None
        """
    fig = plt.figure(dpi=350)
    ax = plt.subplot(111)
    data = [np.mean(x, axis=0)[i] for x in data] # average over simulations
    pos = [j for j in range(len(labels))]
    plt.violinplot(data, pos)
    plt.axhline(1 - alpha, color='k', linestyle='dashed', linewidth=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    plt.ylim((0, 1))
    plt.ylabel(ylabel)
    plt.tight_layout()
    alpha_string = repr(alphas[i]).replace('.', '-')
    fig.savefig(_PLOT_LOCATION + data_directory + f'violin_{ylabel}_alpha{alpha_string}.png')
    plt.ioff()
    plt.close(fig)


def create_bias_plot(bias_method_1, bias_method_2, intervals_correct_method_1, intervals_correct_method_2, alpha, ylabel,
                     fontsize=17):
    """Plot the bias against the PICF or CICF."""
    fig = plt.figure()
    y_BDE = np.mean(intervals_correct_method_1, axis=0)[i]  # Average over simulations to get CICF/PICF
    y_DE = np.mean(intervals_correct_method_2, axis=0)[i]
    plt.plot(bias_method_1, y_BDE, 'o', color='c', alpha=0.5,
             label='BDE')
    plt.plot(bias_method_2, y_DE, 'o', color='r', alpha=0.5,
             label='DE')
    plt.axhline(1 - alpha, color='k', linestyle='dashed', linewidth=1)
    plt.ylim((0, 1))
    plt.xlabel("bias", fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    alpha_string = repr(alphas[i]).replace('.', '-')
    fig.savefig(_PLOT_LOCATION + data_directory + f'biasv{ylabel}alpha{alpha_string}.png')
    plt.ioff()
    plt.close(fig)


def create_error_hist(metrics, fontsize=17):
    """Plot the errors in the function values"""
    errors_f_hati_f = metrics['error_fhati_f']
    errors_f_hathat_i_f_hati = metrics['error_fhathati_fhati']
    for i in range(np.shape(errors_f_hati_f)[1]):
        bins = np.histogram(np.hstack((errors_f_hati_f[:, i],
                                       errors_f_hathat_i_f_hati[:, i])), bins=15)[1]
        fig = plt.figure()
        plt.hist(errors_f_hati_f[:, i], bins=bins,
                 label="$\hat{f}_{i} - f$", alpha=0.5)
        plt.hist(errors_f_hathat_i_f_hati[:, i], bins=bins,
                 label="$\hat{\hat{f}_{i}} - \hat{f}_{i}$", alpha=0.5)
        plt.xlabel("error", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(_PLOT_LOCATION + 'errorplots/' + data_directory + f'X_{i}.png')
        plt.close(fig)

metrics = klepto.archives.dir_archive(_SHELVE_LOCATION, serialized=True)
metrics.load()

#
BDE_metrics = metrics['BDE_metrics']
DE_metrics = metrics['DE_metrics']
Dropout_metrics = metrics['Dropout_metrics']
QD_metrics = metrics['QD_metrics']
NB_metrics = metrics['NB_metrics']
alphas = metrics['alphas']

create_bias_plot(bias, bias, BDE_metrics.CI_correct, DE_metrics.CI_correct, alphas[i], "CICF")

CI_labels = ['BDE', 'DE', 'NB', 'CD']
PI_labels = ['BDE', 'DE', 'QDE', 'CD']
CI_data = [BDE_metrics.CI_correct, DE_metrics.CI_correct, NB_metrics.CI_correct, Dropout_metrics.CI_correct]
PI_data = [BDE_metrics.PI_correct, DE_metrics.PI_correct, QD_metrics.PI_correct, Dropout_metrics.PI_correct]
for i in range(len(alphas)):
    create_violin(CI_data, CI_labels, alphas[i], 'CICF')
    create_violin(PI_data, PI_labels, alphas[i], 'PICF')
    create_bias_plot(DE_metrics.bias, DE_metrics.bias, BDE_metrics.PI_correct, DE_metrics.PI_correct, alphas[i], "PICF")
    create_bias_plot(DE_metrics.bias, DE_metrics.bias, BDE_metrics.CI_correct, DE_metrics.CI_correct, alphas[i], "CICF")


