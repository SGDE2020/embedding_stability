from glob import glob
import os

import numpy as np
import pandas as pd

from lib.tools.classification import load_classification_results

algorithms = ["GraphSAGE", "HOPE", "LINE", "node2vec", "SDNE"]
classifiers = ["adaboost", "decision_tree", "neural_network", "random_forest"]
datasets = ["[bB]log[cC]atalog", "cora", "protein", "wikipedia"]

for dataset in datasets:
    data = {}
    for algorithm in algorithms:
        data[algorithm] = {}

        for classifier in classifiers:
            if classifier == "neural_network":
                filenames = glob(os.path.dirname(os.path.abspath(__file__)) + "/results/{0}/{1}*{2}_*_one_hidden.txt".format(classifier, algorithm.lower(), dataset))
            else:
                filenames = glob(os.path.dirname(os.path.abspath(__file__)) + "/results/{0}/{1}*{2}_*.txt".format(classifier, algorithm.lower(), dataset))

            results = np.empty(0)

            if len(filenames) < 30:
                print("Missing", 30 - len(filenames), "of", dataset, algorithm, classifier)
                data[algorithm][classifier] = len(filenames) - 30
            else:
                for filename in filenames:
                    _, _, _, scores = load_classification_results(filename)
                    results = np.append(results, scores["test_score"])

                data[algorithm][classifier] = np.std(results)

    data = pd.DataFrame(data)
    print("\n\n", dataset)
    print(data.to_latex())
