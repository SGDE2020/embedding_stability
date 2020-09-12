import os
import pickle
from argparse import ArgumentParser
from functools import partial
from itertools import product
from multiprocessing import Pool

import numpy as np
from lib.tools.classification import save_classification_results
from lib.tools.embedding import read_embedding
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import shuffle

from downstream_classification.classify import (adaboost, decision_tree,
                                                neural_network, random_forest)


def run(task, n_repeats):
    embedding, classifier = task
    print(f"Start of {classifier} classifier")

    if classifier == "adaboost":
        fn = adaboost
    elif classifier == "decision_tree":
        fn = decision_tree
    elif classifier == "neural_network":
        fn = neural_network
    elif classifier == "random_forest":
        fn = random_forest

    scores = {
        "train_score_accuracy": list(),
        "test_score_accuracy": list(),
        "train_score_f1_micro": list(),
        "test_score_f1_micro": list(),
        "train_score_f1_macro": list(),
        "test_score_f1_macro": list(),
        "train_score_auc": list(),
        "test_score_auc": list(),
        "test_predictions": list()
    }

    for i in range(n_repeats):
        train_data = np.concatenate([
            embedding["true_embeddings_train"],
            embedding["false_embeddings_train"]
        ])
        train_label = np.concatenate([
            np.ones(len(embedding["true_embeddings_train"]), dtype=int),
            np.zeros(len(embedding["false_embeddings_train"]), dtype=int)
        ])
        train_data, train_label = shuffle(train_data, train_label)

        test_data = np.concatenate([
            embedding["true_embeddings"],
            embedding["false_embeddings"]
        ])
        test_label = np.concatenate([
            np.ones(len(embedding["true_embeddings"]), dtype=int),
            np.zeros(len(embedding["false_embeddings"]), dtype=int)
        ])
        test_indices = np.arange(len(test_label))
        test_data, test_label, test_indices = shuffle(test_data, test_label, test_indices)

        train_pred, test_pred = fn(train_data, test_data, train_label, test_label)

        scores["train_score_accuracy"].append(accuracy_score(train_label, train_pred))
        scores["test_score_accuracy"].append(accuracy_score(test_label, test_pred))
        scores["train_score_f1_micro"].append(f1_score(train_label, train_pred, average="micro"))
        scores["test_score_f1_micro"].append(f1_score(test_label, test_pred, average="micro"))
        scores["train_score_f1_macro"].append(f1_score(train_label, train_pred, average="macro"))
        scores["test_score_f1_macro"].append(f1_score(test_label, test_pred, average="macro"))
        scores["train_score_auc"].append(roc_auc_score(train_label, train_pred))
        scores["test_score_auc"].append(roc_auc_score(test_label, test_pred))

        # Unshuffle the test data before saving them. Otherwise we cannot
        # compare them, as we wouldn't have any idea which entry belongs to
        # which (nonexisting) edge.
        unshuffled_pred = np.empty((len(test_pred)), dtype=int)
        for i, j in enumerate(test_indices):
            unshuffled_pred[j] = test_pred[i]
        scores["test_predictions"].append(list(unshuffled_pred))

    filename = os.path.dirname(os.path.abspath(__file__)) + "/results/link_prediction/" + classifier + "/" + embedding["embedding_name"]
    if args.cluster_job_id is not None:
        filename += "_" + str(args.cluster_job_id)

    if classifier == "neural_network":
        filename += "_one_hidden"

    save_classification_results(filename + ".txt", scores, 2, 1, n_repeats)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("dataset", type=str)
    arg_parser.add_argument("embeddings", nargs="+")
    arg_parser.add_argument("-classifiers", nargs="+", choices=["adaboost", "decision_tree", "neural_network", "random_forest"], default=["adaboost", "decision_tree", "neural_network", "random_forest"])
    arg_parser.add_argument("-repeats", default=10, type=int)
    arg_parser.add_argument("-processes", type=int, help="Number of processes to use in parallel, defaults to number of cpus.")
    arg_parser.add_argument("--cluster-job-id", type=int, help="Integer number e.g. your job id if running on cluster to avoid lost updates, only with single prediction per call possible")

    args = arg_parser.parse_args()
    if args.cluster_job_id:
        assert len(args.embeddings) == 1, "Number of Embeddings must be one when using --cluster-job-id"
        assert len(args.classifiers) == 1, "Number of Classifiers must be on when using --cluster-job-id"
        assert args.processes == 1, "It does not make sense to spread over multiple cores, when using --cluster-job-id"

    with open(args.dataset.replace('.edgelist', '_edges.pickle'), 'rb') as f:
        true_edges = pickle.load(f)
    with open(args.dataset.replace('.edgelist', '_nonedges.pickle'), 'rb') as f:
        false_edges = pickle.load(f)
    with open(args.dataset.replace('.edgelist', '_edges_train.pickle'), 'rb') as f:
        true_edges_train = pickle.load(f)
    with open(args.dataset.replace('.edgelist', '_nonedges_train.pickle'), 'rb') as f:
        false_edges_train = pickle.load(f)

    # Read node embeddings and transform them into edge embeddings for the
    # selected positive and negative edges
    tasks = {}
    for filename in args.embeddings:
        with open(filename, "r") as file:
            node_embeddings = read_embedding(file)

        true_embeddings = np.empty((len(true_edges), node_embeddings.shape[1]))
        false_embeddings = np.empty((len(false_edges), node_embeddings.shape[1]))
        true_embeddings_train = np.empty((len(true_edges_train), node_embeddings.shape[1]))
        false_embeddings_train = np.empty((len(false_edges_train), node_embeddings.shape[1]))

        for i, (u, v) in enumerate(true_edges):
            true_embeddings[i] = np.multiply(node_embeddings[u], node_embeddings[v])

        for i, (u, v) in enumerate(false_edges):
            false_embeddings[i] = np.multiply(node_embeddings[u], node_embeddings[v])

        for i, (u, v) in enumerate(true_edges_train):
            true_embeddings_train[i] = np.multiply(node_embeddings[u], node_embeddings[v])

        for i, (u, v) in enumerate(false_edges_train):
            false_embeddings_train[i] = np.multiply(node_embeddings[u], node_embeddings[v])

        tasks[filename] = {
            "embedding_name": os.path.basename(filename).split(".")[0],
            "true_embeddings": true_embeddings,
            "false_embeddings": false_embeddings,
            "true_embeddings_train": true_embeddings_train,
            "false_embeddings_train": false_embeddings_train
        }

    print("Preprocessing finished, starting to run classifiers...")
    with Pool(args.processes) as pool:
        fn = partial(run, n_repeats=args.repeats)
        list(pool.imap_unordered(fn, product(tasks.values(), args.classifiers)))
