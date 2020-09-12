from glob import glob
import os
import re

import numpy as np

# make plotting work over ssh, must be imported before pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

algorithms = {"hope": "HOPE", "line": "LINE", "node2vec": "node2vec", "sdne": "SDNE", "graphsage": "GraphSAGE"}
classifiers = {"adaboost": "AdaBoost", "decision_tree": "Decision Tree", "neural_network": "Neural Network", "random_forest": "Random Forest"}
datasets = {"[bB]log": "BlogCatalog", "cora": "Cora", "protein": "Protein", "wikipedia": "Wikipedia"}

axis_color = "#CCCCCC"
color_dict = {
    "graphsage": ["#00549F", "#8EBAE5"],
    "hope": ["#612158", "#A8859E"],
    #"hope": ["#7A6FAC", "#BCB5D7"],
    "line": ["#E30066", "#F19EB1"],
    "node2vec": ["#57AB27", "#B8D698"],
    "sdne": ["#F6A800", "#FDD48F"]
}

ind_datasets = np.arange(len(datasets))
width = 0.18


def plot(self_data, data, self_data_std=None, data_std=None):
    fig, axs = plt.subplots(1, 4, sharey=True, constrained_layout=True)

    for i, (classifier, title) in enumerate(classifiers.items()):
        ax = axs[i]

        for j, (algorithm, label) in enumerate(algorithms.items()):
            # Plot the self_overlap bar first, as it should be the background
            # bar and almost never lower than the overlap bar itself.
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

    fig.set_size_inches(9, 2.5)
    fig.legend(loc="upper center", ncol=len(algorithms), prop={"size": 7}, bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure, fancybox=False)

    return fig, axs


basedir = os.path.dirname(os.path.abspath(__file__)) + "/results"

overlap = {}
overlap_std = {}
stable_core = {}
self_overlap = {}
self_overlap_std = {}
self_stable_core = {}
self_stable_core_std = {}

for t in ["total", "one_wrong"]:
    overlap[t] = {}
    overlap_std[t] = {}
    self_overlap[t] = {}
    self_overlap_std[t] = {}

    for i, (classifier, title) in enumerate(classifiers.items()):
        overlap[t][classifier] = {}
        overlap_std[t][classifier] = {}
        self_overlap[t][classifier] = {}
        self_overlap_std[t][classifier] = {}

        if t == "total":
            stable_core[classifier] = {}
            self_stable_core[classifier] = {}
            self_stable_core_std[classifier] = {}

        for j, (algorithm, label) in enumerate(algorithms.items()):
            overlap[t][classifier][algorithm] = np.zeros(len(datasets))
            overlap_std[t][classifier][algorithm] = np.zeros(len(datasets))
            self_overlap[t][classifier][algorithm] = np.zeros(len(datasets))
            self_overlap_std[t][classifier][algorithm] = np.zeros(len(datasets))

            if t == "total":
                stable_core[classifier][algorithm] = np.zeros(len(datasets))
                self_stable_core[classifier][algorithm] = np.zeros(len(datasets))
                self_stable_core_std[classifier][algorithm] = np.zeros(len(datasets))

            for k, dataset in enumerate(datasets):
                filename = glob(f"{basedir}/compare_embedding_errors/{t}*/{algorithm}_{classifier}_*{dataset}*.txt")
                if filename:
                    with open(filename[0]) as result_file:
                        first_line = result_file.readline()
                        test_size = int(re.search("Size of test set: (\d+)", first_line).group(1))

                        if t == "total":
                            stable_core[classifier][algorithm][k] = int(re.search("Stable Core: (\d+)", first_line).group(1)) / test_size

                        array = np.loadtxt(result_file, dtype=int, delimiter=",")
                        array = 1.0 - (array[np.triu_indices_from(array, 1)] / test_size)

                        overlap[t][classifier][algorithm][k] = np.mean(array)
                        overlap_std[t][classifier][algorithm][k] = np.std(array)
                else:
                    print("Missing overlap results for", t, algorithm, classifier, dataset)

                all_datapoints = np.empty(0)
                all_cores = []
                count = 0
                for filename in glob(f"{basedir}/compare_embedding_errors/{t}*/self_{classifier}*{algorithm}*{dataset}*.txt"):
                    count += 1
                    with open(filename) as result_file:
                        first_line = result_file.readline()

                        test_size = int(re.search("Size of test set: (\d+)", first_line).group(1))
                        array = np.loadtxt(result_file, dtype=int, delimiter=",")

                        if t == "total":
                            all_cores.append(int(re.search("Stable Core: (\d+)", first_line).group(1)) / test_size)

                    array = array[np.triu_indices_from(array, 1)] / test_size
                    all_datapoints = np.append(all_datapoints, 1.0 - array)

                if count != 5:
                    print("Missing", 5 - count, "self-overlap results for", t, algorithm, classifier, dataset)

                self_overlap[t][classifier][algorithm][k] = np.mean(all_datapoints)
                self_overlap_std[t][classifier][algorithm][k] = np.std(all_datapoints)

                if t == "total":
                    self_stable_core[classifier][algorithm][k] = np.mean(all_cores)
                    self_stable_core_std[classifier][algorithm][k] = np.std(all_cores)

for t in ["total", "one_wrong"]:
    # overlap with std
    fig, axs = plot(self_overlap[t], overlap[t], self_overlap_std[t], overlap_std[t])
    axs[0].set_ylabel("Average Relative Overlap", fontsize=7)
    fig.savefig(f"downstream_classification/results/plots/overlap_{t}.pdf", bbox_inches="tight")

    # overlap without std
    fig, axs = plot(self_overlap[t], overlap[t])
    axs[0].set_ylabel("Average Relative Overlap", fontsize=7)
    fig.savefig(f"downstream_classification/results/plots/overlap_{t}_without_std.pdf", bbox_inches="tight")

# stable core with std
fig, axs = plot(self_stable_core, stable_core, self_stable_core_std)
axs[0].set_ylabel("Relative Stable Core", fontsize=7)
fig.savefig(f"downstream_classification/results/plots/stable_core.pdf", bbox_inches="tight")

# stable core
fig, axs = plot(self_stable_core, stable_core)
axs[0].set_ylabel("Relative Stable Core", fontsize=7)
fig.savefig(f"downstream_classification/results/plots/stable_core_without_std.pdf", bbox_inches="tight")
