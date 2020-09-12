from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

from lib.tools.classification import load_classification_results, load_link_prediction_results
from scale_to_latex import get_columnwidth, get_textwidth, get_figsize
import patch_lvplot  # removes outlier plotting from letter value plots

# %%

color_dict = {"LINE": "#E30066", "HOPE": "#612158", "SDNE": "#F6A800", "GraphSAGE": "#00549F", "node2vec": "#57AB27"}
classification_methods = {"adaboost": "AdaBoost", "decision_tree": "Decision Tree", "neural_network": "Neural Network",
                          "random_forest": "Random Forest"}
datasets = {"BlogCatalog": "BlogCatalog" }#, "cora": "Cora", "protein": "Protein", "wikipedia": "Wikipedia"}
embedding_methods = {"hope": "HOPE", "line": "LINE", "node2vec": "node2vec", "sdne": "SDNE", "graphsage": "GraphSAGE"}

# %%

data_accuracies = []
data_f1_micro = []
data_f1_macro = []
data_auc = []

def main():

    for classification_method in classification_methods.keys():
        for embedding_method in embedding_methods:
            for graph, graph_name in datasets.items():
                files = glob(
                    f"downstream_classification/results/link_prediction/{classification_method}/{embedding_method}_{graph}_reduced_*_*{'one_hidden' if embedding_method == 'neural_network' else ''}.txt")

                missing = ""

                if len(files) != 10:
                    missing = f"- Missing {10 - len(files)} results"
                print("Loading", classification_method, embedding_method, graph_name, missing)

                for i, file in enumerate(files):
                    if i >= 10:
                        break
                    parts = os.path.basename(file).split("_")
                    embedding_method = parts[0]

                    if classification_method == "neural_network":
                        number = int(parts[-3])
                    else:
                        number = int(parts[-1].split(".")[0])

                    res = load_link_prediction_results(file)

                    for val in res[3]["test_score_accuracy"]:
                        data_accuracies.append(
                            [graph_name, embedding_methods[embedding_method], classification_methods[classification_method],
                             number, val])
                    for val in res[3]["test_score_f1_micro"]:
                        data_f1_micro.append(
                            [graph_name, embedding_methods[embedding_method], classification_methods[classification_method],
                             number, val])
                    for val in res[3]["test_score_f1_macro"]:
                        data_f1_macro.append(
                            [graph_name, embedding_methods[embedding_method], classification_methods[classification_method],
                             number, val])
                    for val in res[3]["test_score_auc"]:
                        data_auc.append(
                            [graph_name, embedding_methods[embedding_method],
                             classification_methods[classification_method],
                             number, val])

    data_accuracies_frames = pd.DataFrame(data=data_accuracies,
                                          columns=["Graph Name", "Embedding Method", "Classification Method", "run",
                                                   "Accuracy"])
    data_f1_micro_frames = pd.DataFrame(data=data_f1_micro,
                                        columns=["Graph Name", "Embedding Method", "Classification Method", "run",
                                                 "F1 Micro"])
    data_f1_macro_frames = pd.DataFrame(data=data_f1_macro,
                                        columns=["Graph Name", "Embedding Method", "Classification Method", "run",
                                                 "F1 Macro"])
    data_auc_frames = pd.DataFrame(data=data_auc,
                                        columns=["Graph Name", "Embedding Method", "Classification Method", "run",
                                                 "AUC"])

    # %%

    columnwidth = get_columnwidth()
    textwidth = get_textwidth()
    light_gray = ".8"
    dark_gray = ".15"
    sns.set(context="notebook", style="whitegrid", font_scale=1, rc={
        "xtick.color": dark_gray, "ytick.color": dark_gray,
        "xtick.bottom": True,
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7,
        #"legend.fontsize": 5.7,
        "axes.linewidth": 1,
        #"xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        #"xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "lines.linewidth": 0.7,
        #"xtick.major.size": 3, "ytick.major.size": 3,
        #"xtick.minor.size": 2, "ytick.minor.size": 2,
        "grid.linewidth": 0.5,
        "patch.linewidth": 0.1,
    })

    #width, height, aspect = get_figsize(textwidth, wf=1 / 4)
    #print(width, height, aspect)


    # %%

    def plot(data, name, y_label):
        fig, axs = plt.subplots(1, len(datasets), sharey=True, constrained_layout=True) #figsize=(2.5 * len(datasets), 1 * 1.7514))

        """for i, classification_method in enumerate(classification_methods.values()):
            g = sns.boxenplot(data=data.loc[data["Classification Method"] == classification_method], x="Graph Name",
                              y=y_label, hue="Embedding Method", palette=color_dict, ax=axs[i])
            g.set_xlabel("")
            if i != 0:
                g.set_ylabel("")
            g.set_title(classification_method)
            g.legend().remove()"""

        for i, graph_name in enumerate(datasets.values()):
            g = sns.boxenplot(data=data.loc[data["Graph Name"] == graph_name], x="Classification Method",
                              y=y_label, hue="Embedding Method", palette=color_dict, ax=axs)
            g.set_xlabel("")
            if i != 0:
                g.set_ylabel("")
            g.set_title(graph_name)
            g.legend().remove()
        sns.despine(fig)
        axs.set(ylim=(0.5,1))
        angle = 40
        #for ax in axs:
        #    ax.tick_params(axis="both", which="major",
        #                        color=light_gray)  # TODO: This is not possible to set in seaborn's style above?
        axs.tick_params(axis="both", which="major", color=light_gray)
        rectangles, labels = g.get_legend_handles_labels()
        for rectangle in rectangles:
            rectangle.set_edgecolor(None)
        leg = fig.legend(rectangles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0),
                         bbox_transform=fig.transFigure, fancybox=False, prop={"size": 7})
        fig.set_size_inches(3.5, 2.5)
        fig.savefig(f"downstream_classification/results/plots/{name}.pdf", bbox_inches="tight")


    plot(data_accuracies_frames, "accuracies-boxplot", "Accuracy")
    plot(data_f1_micro_frames, "f1-micro-boxplot", "F1 Micro")
    plot(data_f1_macro_frames, "f1-macro-boxplot", "F1 Macro")
    plot(data_auc_frames, "auc-boxplot", "AUC")

if __name__ == "__main__":
    main()