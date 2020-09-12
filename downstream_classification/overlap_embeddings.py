import argparse
from functools import partial
from itertools import combinations
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lib.tools import parse_graph
from lib.tools.classification import calc_stable_core, prepare_classification
from lib.tools.embedding import read_embedding
from downstream_classification.classify import adaboost, decision_tree, neural_network, random_forest


def differences(predictions, node_labels_test, stable_core_threshold = 0.9):
    """
    Calculates pairwise for different runs the total number of nodes

      (i)  which have a different prediction
      (ii) for which one prediction was correct and the other not.

    Parameters
    ----------
    predictions : list of array
        each entry in the list should be an array of predictions for all nodes
        in the test set
    node_labels_test : array
        true labels for the nodes in the test set
    stable_core_threshold : float between 0 and 1
        Percentage of how many runs have to agree on a particular label for the
        node to be considered stable. E.g. a node is considered part of a
        stable core if it is classified the same in 3 out of 30 runs.

    Returns
    -------
    differences_total : array of shape [len(predictions), len(predictions)]
    differences_one_wrong : array of shape [len(predictions), len(predictions)]
    """
    n = len(predictions)
    differences_total = np.zeros((n, n), dtype=int)
    differences_one_wrong = np.zeros((n, n), dtype=int)

    for (i, pred), (j, pred_prime) in combinations(enumerate(predictions), 2):
        differences_total[i, j] = sum(not np.array_equal(x, y) for x, y in zip(pred, pred_prime))
        differences_one_wrong[i, j] = sum((not np.array_equal(t, x) and np.array_equal(t, y)) or (np.array_equal(t, x) and not np.array_equal(t, y)) for t, x, y in zip(node_labels_test, pred, pred_prime))

    stable_core = calc_stable_core(predictions, stable_core_threshold)
    return differences_total, differences_one_wrong, stable_core


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", type=str, help="Path to the dataset which should be considered.")
    arg_parser.add_argument("embeddings", nargs="+", type=argparse.FileType("r"), help="Path to the embeddings which should be compared conserning their performance in downstream classification.")
    arg_parser.add_argument("classifier", choices=["adaboost", "decision_tree", "neural_network", "random_forest"])
    args = arg_parser.parse_args()

    graph_name, graph = parse_graph(args.dataset)
    embeddings = list(map(read_embedding, args.embeddings))
    node_labels, _ = prepare_classification(graph, embeddings[0]) # embedding only used for validation, don't care which one

    del graph # free memory

    splits = train_test_split(node_labels, *embeddings)
    node_labels_train = splits[0]
    node_labels_test = splits[1]
    embeddings_train = [splits[i] for i in range(2, len(splits), 2)]
    embeddings_test = [splits[i] for i in range(3, len(splits), 2)]

    # We train one classifier for each embedding using the same set of training
    # nodes. Predictions will contain for each embedding an array of
    # predictions for each node not in the training set. We later compare how
    # stable the predictions of these nodes are across different embeddings.
    predictions = []
    for i, (train, test) in enumerate(zip(tqdm(embeddings_train), embeddings_test)):
        if args.classifier == "adaboost":
            fn = adaboost
        elif args.classifier == "decision_tree":
            fn = decision_tree
        elif args.classifier == "neural_network":
            fn = neural_network
        elif args.classifier == "random_forest":
            fn = random_forest

        _, prediction = fn(train, test, node_labels_train, node_labels_test)
        predictions.append(prediction)

    differences_total, differences_one_wrong, stable_core = differences(predictions, node_labels_test)

    algorithm = args.embeddings[0].name.split("/")[-1].split("_")[0]
    filename = os.path.dirname(os.path.abspath(__file__)) + "/results/compare_embedding_errors/{}/" + str(algorithm) + "_" + args.classifier + "_" + graph_name
    np.savetxt(filename.format("total_diff") + ".txt", differences_total, fmt="%5d", delimiter=",", header=f"Number how often the classifiers yielded different predictions. Size of test set: {len(node_labels_test)}, Stable Core: {stable_core}")
    np.savetxt(filename.format("one_wrong") + ".txt", differences_one_wrong, fmt="%5d", delimiter=",", header="Number how often only one classifier was right. Size of test set: " + str(len(node_labels_test)))
    np.save(filename.format("predictions"), predictions)
