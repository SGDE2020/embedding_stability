import argparse
from functools import partial
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lib.tools import parse_graph
from lib.tools.classification import prepare_classification
from lib.tools.embedding import read_embedding
from downstream_classification.classify import adaboost, decision_tree, neural_network, random_forest
from downstream_classification.overlap_embeddings import differences


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", type=str, help="Path to the dataset which should be considered.")
    arg_parser.add_argument("embeddings", nargs="+", type=argparse.FileType("r"), help="Path to the embedding(s) which should be compared conserning their performance in downstream classification.")
    arg_parser.add_argument("-classifiers", nargs="+", default=["adaboost", "decision_tree", "neural_network", "random_forest"], choices=["adaboost", "decision_tree", "neural_network", "random_forest"])
    arg_parser.add_argument("-n", default=10, type=int, help="How often the classifier should be trained. The performances of these different runs are then compared.")
    args = arg_parser.parse_args()

    _, graph = parse_graph(args.dataset)
    node_labels, _ = prepare_classification(graph)
    del graph # free memory

    for embedding_file in args.embeddings:
        embedding_name = embedding_file.name.split("/")[-1].split(".")[0]
        print("Start embedding", embedding_name)

        embedding = read_embedding(embedding_file)
        node_labels_train, node_labels_test, embedding_train, embedding_test = train_test_split(node_labels, embedding)

        for clf in args.classifiers:
            print("Start classifier", clf)

            # We train n classifiers on the same embedding and always use the same
            # set of training and test nodes. We then compare how stable the
            # predictions of the test nodes where across multiple runs of the same
            # classifier. This gives us a metric of a classifier's stability, which
            # helps us to estimate how much the unstability of the embedding
            # algorithms influence the overall outcome and which instability is
            # attributed to the classifiers themself.
            predictions = []
            for i in tqdm(range(args.n)):
                if clf == "adaboost":
                    fn = adaboost
                elif clf == "decision_tree":
                    fn = decision_tree
                elif clf == "neural_network":
                    fn = neural_network
                elif clf == "random_forest":
                    fn = random_forest

                _, prediction = fn(embedding_train, embedding_test, node_labels_train, node_labels_test)
                predictions.append(prediction)

            differences_total, differences_one_wrong, stable_core = differences(predictions, node_labels_test)

            filename = os.path.dirname(os.path.abspath(__file__)) + "/results/compare_embedding_errors/{}/self_" + clf + "_" + embedding_name
            np.savetxt(filename.format("total_diff") + ".txt", differences_total, fmt="%5d", delimiter=",", header=f"Number how often the classifiers yielded different predictions. Size of test set: {len(node_labels_test)}, Stable Core: {stable_core}")
            np.savetxt(filename.format("one_wrong") + ".txt", differences_one_wrong, fmt="%5d", delimiter=",", header="Number how often only one classifier was right. Size of test set: " + str(len(node_labels_test)))
            np.save(filename.format("predictions"), predictions)
