from glob import glob
import os
import re

import numpy as np

# make plotting work over ssh, must be imported before pyplot
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lib.tools.classification import calc_stable_core, load_link_prediction_results

algorithms = {"hope": "HOPE", "line": "LINE", "node2vec": "node2vec", "sdne": "SDNE", "graphsage": "GraphSAGE"}
classifiers = {"adaboost": "AdaBoost", "decision_tree": "Decision Tree", "neural_network": "Neural Network", "random_forest": "Random Forest"}
#datasets = {"[bB]log": "BlogCatalog", "cora": "Cora", "protein": "Protein", "wikipedia": "Wikipedia"}
datasets = {"[bB]log": "BlogCatalog"}
axis_color = "#CCCCCC"
color_dict = {
    "graphsage": ["#00549F", "#8EBAE5"],
    "hope": ["#612158", "#A8859E"],
    "line": ["#E30066", "#F19EB1"],
    "node2vec": ["#57AB27", "#B8D698"],
    "sdne": ["#F6A800", "#FDD48F"]
}


#def calc_stable_core(predictions):
#    predictions = np.array(predictions)
#    return np.count_nonzero(np.all(predictions == predictions[0,:], axis=0))


ind_datasets = np.arange(4)
width = 0.12

def plot(self_data, data, self_data_std=None, data_std=None):
    #fig, axs = plt.subplots(1, 4 , sharey=True, constrained_layout=True)
    fig = plot.figure(figsize=(1,4), sharey=True, constrained_layout=True)
    gs1 = gridspec.GridSpec(1, 4)
    gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

    for i, (classifier, title) in enumerate(classifiers.items()):
        #ax = axs[i]
        for j, (algorithm, label) in enumerate(algorithms.items()):
            # Plot the self_overlap bar first, as it should be the background
            # bar and almost never lower than the overlap bar itself.
            ax = fig.add_subplot(gs1[i])
            ax.bar(
                ind_datasets + (j - len(algorithms) / 2) * width,
                self_data[classifier][algorithm],
                width,
                align="edge",
                color=color_dict[algorithm][1],
                yerr=self_data_std[classifier][algorithm] if self_data_std is not None else None)
            ax.bar(
                ind_datasets + (j - len(algorithms) / 2) * width,
                data[classifier][algorithm],
                width,
                align="edge",
                color=color_dict[algorithm][0],
                label=label if i == 0 else None, # label only for this bar so labels are not duplicated in legend
                yerr=data_std[classifier][algorithm] if data_std is not None else None)

        ax.tick_params(axis="both", which="major", labelsize=7, color=axis_color)
        ax.set_xticks(ind_datasets)
        ax.set_xticklabels(datasets.values())
        ax.set_title(title, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(axis_color)
        ax.spines["left"].set_color(axis_color)

    #fig.set_size_inches(5, 2.5)
    fig.legend(loc="upper center", ncol=len(algorithms), prop={"size": 7}, bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, fancybox=False)
    #fig.tight_layout()

    return fig, axs

def plot_only_on_dataset(self_data, data, self_data_std=None, data_std=None):
    fig, axs = plt.subplots(1, 1, sharey=True, constrained_layout=True)
    ax = axs
    for i, (classifier, title) in enumerate(classifiers.items()):
        for j, (algorithm, label) in enumerate(algorithms.items()):
            # Plot the self_overlap bar first, as it should be the background
            # bar and almost never lower than the overlap bar itself.
            ax.bar(
                i + (j - len(algorithms) / 2) * width,
                self_data[classifier][algorithm],
                width,
                align="edge",
                color=color_dict[algorithm][1],
                yerr=self_data_std[classifier][algorithm] if self_data_std is not None else None)
            ax.bar(
                i + (j - len(algorithms) / 2) * width,
                data[classifier][algorithm],
                width,
                align="edge",
                color=color_dict[algorithm][0],
                label=label if i == 0 else None,  # label only for this bar so labels are not duplicated in legend
                yerr=data_std[classifier][algorithm] if data_std is not None else None)

        ax.tick_params(axis="both", which="major", labelsize=7, color=axis_color)
        ax.set_xticks(ind_datasets)
        ax.set_xticklabels(classifiers.values())
        ax.set_title("BlogCatalog", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(axis_color)
        ax.spines["left"].set_color(axis_color)

    fig.set_size_inches(3.5, 2.5)
    fig.legend(loc="upper center", ncol=len(algorithms), prop={"size": 7}, bbox_to_anchor=(0.5, 0),
               bbox_transform=fig.transFigure, fancybox=False)

    return fig, axs


basedir = os.path.dirname(os.path.abspath(__file__)) + "/results/link_prediction"

stable_core = {}
self_stable_core = {}

for i, (classifier, title) in enumerate(classifiers.items()):
    stable_core[classifier] = {}
    self_stable_core[classifier] = {}

    for j, (algorithm, label) in enumerate(algorithms.items()):
        stable_core[classifier][algorithm] = np.zeros(len(datasets))
        self_stable_core[classifier][algorithm] = np.zeros(len(datasets))

        for k, dataset in enumerate(datasets):
            predictions = []
            for filename in glob(f"{basedir}/{classifier}/{algorithm}_*{dataset}*.txt"):
                _, _, _, scores = load_link_prediction_results(filename, load_predictions=True)
                if len(predictions) < 10:
                    predictions.append(scores["test_predictions"])
            if len(predictions) != 10:
                print(f"Missing {10 - len(predictions)} results for", algorithm, classifier, dataset)
                continue

            test_size = len(predictions[0][0])

            stable_core[classifier][algorithm][k] = calc_stable_core([result[0] for result in predictions]) / test_size
            self_stable_core[classifier][algorithm][k] = calc_stable_core(predictions[0]) / test_size

# stable core
fig, axs = plot_only_on_dataset(self_stable_core, stable_core)
axs.set_ylabel("Relative Stable Core", fontsize=7)
fig.savefig(f"downstream_classification/results/plots/link_stable_core.pdf", bbox_inches="tight")
