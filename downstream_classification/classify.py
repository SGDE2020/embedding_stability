import argparse
import os
from functools import partial
from itertools import product
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedKFold

from lib.tools import parse_graph
from lib.tools.classification import (ClassificationType,
                                      determine_classification_task,
                                      prepare_classification,
                                      save_classification_results)
from lib.tools.embedding import read_embedding


def get_predictions(node_labels_train, node_labels_test, train_proba, test_proba):
    def true_prop(proba):
        for output in filter(lambda x: x.shape[1] > 1, proba):
            yield output[:, 1]

    classification_type = determine_classification_task(node_labels_train)
    if classification_type == ClassificationType.MULTILABEL:
        if isinstance(train_proba, list):
            train_proba = list(true_prop(train_proba))
            train_proba = np.array(train_proba).T
            test_proba = list(true_prop(test_proba))
            test_proba = np.array(test_proba).T

        target_classes = node_labels_train.shape[-1]
        train_pred = np.zeros((node_labels_train.shape[0], target_classes))
        test_pred = np.zeros((node_labels_test.shape[0], target_classes))

        for i, labels in enumerate(node_labels_train):
            train_pred[i][np.argsort(-train_proba[i])[0:np.count_nonzero(labels)]] = 1
        for i, labels in enumerate(node_labels_test):
            test_pred[i][np.argsort(-test_proba[i])[0:np.count_nonzero(labels)]] = 1
    elif classification_type == ClassificationType.MULTICLASS:
        train_pred = np.argmax(train_proba, axis=1)
        test_pred = np.argmax(test_proba, axis=1)

    return train_pred, test_pred

def adaboost(node_embeddings_train, node_embeddings_test, node_labels_train, node_labels_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.multiclass import OneVsRestClassifier

    if determine_classification_task(node_labels_train) == ClassificationType.MULTILABEL:
        # Unlike decision trees and random forests, the adaboost implementation
        # from scikit-learn does not support multilabel classification.
        # As documented in https://scikit-learn.org/stable/modules/multiclass.html,
        # OneVsRestClassifier can be used in this case.
        clf = OneVsRestClassifier(AdaBoostClassifier())
    else:
        clf = AdaBoostClassifier()

    clf.fit(node_embeddings_train, node_labels_train)

    train_proba = clf.predict_proba(node_embeddings_train)
    test_proba = clf.predict_proba(node_embeddings_test)
    return get_predictions(node_labels_train, node_labels_test, train_proba, test_proba)


def decision_tree(node_embeddings_train, node_embeddings_test, node_labels_train, node_labels_test):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier()
    clf.fit(node_embeddings_train, node_labels_train)

    train_proba = clf.predict_proba(node_embeddings_train)
    test_proba = clf.predict_proba(node_embeddings_test)
    return get_predictions(node_labels_train, node_labels_test, train_proba, test_proba)


def get_model(name, embedding_dimension, target_classes, classification_type=ClassificationType.MULTICLASS):
    """
    Constructs the model according to the given parameters:

    - Input of size `embedding_dimension`
    - Hidden layers according to the model name, which can be "small",
      "one_hidden" or "deep". For the small model, no hidden layers are used,
      for the medium one we use one hidden layer and for the deep three hidden
      layers. Hidden layers always have 100 neurons and use ReLU activation.
    - Output of size `target_classes`. Activation will be `softmax` or
      `sigmoid`, depending on the classification type.
    """
    from keras.layers import Dense, Input
    from keras.models import Model

    if classification_type == ClassificationType.MULTICLASS:
        final_activation = "softmax"
        loss = "sparse_categorical_crossentropy"
    elif classification_type == ClassificationType.MULTILABEL:
        final_activation = "sigmoid"
        loss = "binary_crossentropy"

    inp = Input(shape=(embedding_dimension,))

    if name == "small":
        x = Dense(target_classes, activation=final_activation)(inp)
    elif name == "one_hidden":
        x = Dense(100, activation="relu")(inp)
        x = Dense(target_classes, activation=final_activation)(x)
    elif name == "deep":
        x = Dense(100, activation="relu")(inp)
        x = Dense(100, activation="relu")(x)
        x = Dense(100, activation="relu")(x)
        x = Dense(target_classes, activation=final_activation)(x)
    else:
        raise ValueError("Unknown model name.")

    model = Model(inputs=inp, outputs=x)
    model.compile("adam", loss, metrics=["accuracy"])
    return model


def neural_network(node_embeddings_train, node_embeddings_test, node_labels_train, node_labels_test):
    from keras.callbacks import EarlyStopping

    classification_type = determine_classification_task(node_labels_train)
    if classification_type == ClassificationType.MULTILABEL:
        target_classes = node_labels_train.shape[-1]
    elif classification_type == ClassificationType.MULTICLASS:
        target_classes = np.max(node_labels_train) + 1

    model = get_model("one_hidden", node_embeddings_train.shape[1], target_classes, classification_type)
    callbacks = [EarlyStopping(min_delta=0.005, patience=10)]
    validation_data = (node_embeddings_test, node_labels_test)
    model.fit(node_embeddings_train, node_labels_train, epochs=500, callbacks=callbacks, validation_data=validation_data, verbose=True)

    train_proba = model.predict(node_embeddings_train)
    test_proba = model.predict(node_embeddings_test)

    return get_predictions(node_labels_train, node_labels_test, train_proba, test_proba)


def random_forest(node_embeddings_train, node_embeddings_test, node_labels_train, node_labels_test):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10) # suppress warning that this will be increased in the future, we use old standard parameter
    clf.fit(node_embeddings_train, node_labels_train)

    train_proba = clf.predict_proba(node_embeddings_train)
    test_proba = clf.predict_proba(node_embeddings_test)
    return get_predictions(node_labels_train, node_labels_test, train_proba, test_proba)


def run(task, n_splits, n_repeats):
    """
    Trains the specified classifier on the node embedding. This method is
    called in parallel on multiple `tasks`.
    The results of the training is saved to disk in the `results` subfolder.

    Parameters
    ----------
    task : tuple
        The first entry contains the name of the classifier, the second a fict
        with the keys `node_embeddings`, `embedding_name`, `node_labels` and
        `disting_node_labels`.
    n_splits : int
    n_repeats : int
    """
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
        "test_predictions": list()
    }

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats).split(node_embeddings)
    for i, (train_index, test_index) in enumerate(rkf):
        train_pred, test_pred = fn(embedding["node_embeddings"][train_index], embedding["node_embeddings"][test_index], embedding["node_labels"][train_index], embedding["node_labels"][test_index])

        scores["train_score_accuracy"].append(accuracy_score(embedding["node_labels"][train_index], train_pred))
        scores["test_score_accuracy"].append(accuracy_score(embedding["node_labels"][test_index], test_pred))
        scores["train_score_f1_micro"].append(f1_score(embedding["node_labels"][train_index], train_pred, average="micro"))
        scores["test_score_f1_micro"].append(f1_score(embedding["node_labels"][test_index], test_pred, average="micro"))
        scores["train_score_f1_macro"].append(f1_score(embedding["node_labels"][train_index], train_pred, average="macro"))
        scores["test_score_f1_macro"].append(f1_score(embedding["node_labels"][test_index], test_pred, average="macro"))
        scores["test_predictions"].append({index: pred for index, pred in zip(test_index, test_pred)})

    filename = os.path.dirname(os.path.abspath(__file__)) + "/results/" + classifier + "/" + embedding["embedding_name"]
    if classifier == "neural_network":
        filename += "_one_hidden"

    save_classification_results(filename + ".txt", scores, embedding["distinct_node_labels"], n_splits, n_repeats)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Runs all classifiers on all embeddings for the given graph in parallel.")
    arg_parser.add_argument("dataset", type=str)
    arg_parser.add_argument("embeddings", nargs="+")
    arg_parser.add_argument("-classifiers", nargs="+", choices=["adaboost", "decision_tree", "neural_network", "random_forest"], default=["adaboost", "decision_tree", "neural_network", "random_forest"])
    arg_parser.add_argument("-splits", default=10, type=int)
    arg_parser.add_argument("-repeats", default=10, type=int)
    arg_parser.add_argument("-processes", type=int, help="Number of processes to use in parallel, defaults to number of cpus.")

    args = arg_parser.parse_args()

    graph_name, graph = parse_graph(args.dataset)
    node_labels, distinct_node_labels = prepare_classification(graph)
    del graph # free memory

    files = {}
    for filename in args.embeddings:
        with open(filename, "r") as file:
            node_embeddings = read_embedding(file)

        files[filename] = {
            "embedding_name": os.path.basename(filename).split(".")[0],
            "node_embeddings": node_embeddings,
            "node_labels": node_labels,
            "distinct_node_labels": distinct_node_labels
        }

    print("Preprocessing finished, starting to run classifiers...")
    with Pool(args.processes) as pool:
        f = partial(run, n_splits=args.splits, n_repeats=args.repeats)
        list(pool.imap_unordered(f, product(files.values(), args.classifiers)))
